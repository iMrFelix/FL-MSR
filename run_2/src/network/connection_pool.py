"""Connection pool: manages TCP connections to neighbors with traffic class support.

In the federated learning emulator each node maintains persistent TCP
connections to every neighbor.  In monolithic mode (Stage 1) a single
connection per neighbor suffices, but in per-layer mode (Stage 2+) each
layer update is assigned a traffic class and must be sent over a socket
whose IP_TOS byte matches the DSCP value for that class.

The pool therefore opens *num_classes* sockets per neighbor, each with a
different TOS value derived from the experiment's ``dscp_mapping`` config.
Callers pass a ``traffic_class`` index when sending; the pool picks the
matching socket (or falls back to class 0 if the specific class is missing).

This design keeps all DSCP/socket details out of the training engine and
algorithm code — they just say "send this on class X" and the pool
translates that to the correct socket.

The pool also supports connecting to the monitor container as a special
neighbor (always on class 0, since metrics don't need QoS differentiation).
"""

from __future__ import annotations

import asyncio
import logging
import math
import socket
import sys
import time
from typing import Any

from src.network.framing import encode_message
from src.proto_gen import federation_pb2

logger = logging.getLogger(__name__)

# SO_SNDBUF sizing horizon (seconds of line-rate data the kernel may buffer).
# Gate ruling G1 (writeup/01-candidate-selection.md): autotuned send buffers
# swallowed entire slow-class byte shares, so `send()` returned at kernel-
# accept time and measured nothing.  Pinning the buffer to ~250 ms of the
# class's shaped bandwidth makes sock_sendall back-pressure reflect actual
# wire drain AND bounds the zombie bytes a cancelled tail can leave behind.
_SNDBUF_HORIZON_S = 0.25

# Linux internally doubles every SO_SNDBUF request (`man 7 socket`); other
# kernels (macOS/BSD, where the tests run) grant the request verbatim.  Knowing
# which contract applies is what makes a silent net.core.wmem_max clamp
# detectable at all: on Linux a granted value below 2x the request means the
# request was capped BEFORE doubling (audit NT-08).
_KERNEL_DOUBLES_SNDBUF = sys.platform.startswith("linux")


class ConnectionPool:
    """TCP connection pool with per-traffic-class socket management.

    Each (neighbor_id, traffic_class) pair maps to one TCP socket.  All
    sockets are non-blocking and use TCP_NODELAY.  The IP_TOS byte on each
    socket is set to ``dscp_value << 2`` so that the kernel marks outgoing
    packets with the correct DSCP codepoint.

    The pool is the only component that touches raw sockets directly.  The
    transport server (which handles incoming connections) is separate —
    there is no symmetry requirement between send and receive paths.
    """

    def __init__(self, source_node: str, dscp_mapping: dict[int, int] | None = None):
        """Create a connection pool.

        Args:
            source_node: This node's identifier (used only for logging).
            dscp_mapping: Maps traffic-class index (0, 1, ...) to DSCP value
                (e.g. {0: 46, 1: 10}).  Class 0 should be the highest-priority
                class.  When *None* or empty, a single class with DSCP 0
                (best-effort) is used.
        """
        self._source_node = source_node
        self._dscp_mapping = dscp_mapping or {0: 0}
        # (neighbor_id, traffic_class) -> non-blocking TCP socket.
        self._connections: dict[tuple[str, int], socket.socket] = {}
        # (neighbor_id, traffic_class) -> what _pin_sndbuf asked for and what
        # the kernel granted.  Exported into the run telemetry (audit NT-08 /
        # BYTE-08): the buffer-accept instrument's bias is proportional to the
        # buffer, so an asymmetric grant across classes is an arm-dependent
        # bias that has to be visible in the data, not only in a log line.
        self._sndbuf_pins: dict[tuple[str, int], dict[str, Any]] = {}

    async def connect(
        self,
        neighbor_id: str,
        host: str,
        port: int,
        num_classes: int = 1,
        max_retries: int = 30,
        retry_delay: float = 2.0,
        class_bandwidths_mbps: dict[int, float] | None = None,
    ) -> None:
        """Open *num_classes* TCP connections to a single neighbor.

        For each traffic class 0..num_classes-1 a separate socket is created
        with the corresponding DSCP marking.  Each socket retries independently
        up to *max_retries* times with *retry_delay* seconds between attempts.

        In monolithic mode the caller passes num_classes=1 and only class 0 is
        opened.  In per-layer mode the caller passes the experiment's
        traffic_classes.num_classes so that every class has its own socket.

        Args:
            class_bandwidths_mbps: Optional traffic-class index -> shaped
                bandwidth in Mbps for the edge towards this neighbor (from
                the node config's ``outgoing_edges[*].classes``).  When a
                finite positive bandwidth is given for a class, the socket's
                ``SO_SNDBUF`` is pinned to ~250 ms of line-rate data so that
                send back-pressure tracks actual wire drain (gate ruling
                G1).  Unshaped classes (missing, ``None`` upstream encoded
                as ``inf``) keep the kernel's autotuned buffer.
        """
        loop = asyncio.get_running_loop()
        bandwidths = class_bandwidths_mbps or {}
        for class_idx in range(num_classes):
            dscp = self._dscp_mapping.get(class_idx, 0)
            tos_byte = dscp << 2  # IP_TOS encodes DSCP in the top 6 bits

            for attempt in range(max_retries):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setblocking(False)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    if tos_byte > 0:
                        sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, tos_byte)
                    self._pin_sndbuf(
                        sock, neighbor_id, class_idx,
                        bandwidths.get(class_idx),
                    )
                    await loop.sock_connect(sock, (host, port))
                    self._connections[(neighbor_id, class_idx)] = sock
                    logger.info(
                        f"[{self._source_node}] Connected to {neighbor_id} "
                        f"class {class_idx} (DSCP {dscp}) at {host}:{port}"
                    )
                    break
                except (ConnectionRefusedError, OSError) as e:
                    if attempt < max_retries - 1:
                        logger.debug(
                            f"[{self._source_node}] Connection to {neighbor_id} "
                            f"class {class_idx} failed (attempt {attempt + 1}): {e}"
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        raise ConnectionError(
                            f"Failed to connect to {neighbor_id} class {class_idx} "
                            f"after {max_retries} attempts"
                        ) from e

    def _pin_sndbuf(
        self,
        sock: socket.socket,
        neighbor_id: str,
        class_idx: int,
        bandwidth_mbps: float | None,
    ) -> None:
        """Pin ``SO_SNDBUF`` to ~250 ms of the class's shaped bandwidth.

        Requested size = ``B_c[bit/s] / 8 * 0.25``.  Linux doubles the
        requested value internally (`man 7 socket`) to account for
        bookkeeping overhead, so the effective value read back via
        ``getsockopt`` is recorded alongside the request — analysis must use
        the effective value when reasoning about buffered-but-unsent bytes.

        Unshaped classes (``None`` / non-finite / non-positive bandwidth)
        keep kernel autotuning, which is harmless because the measurement
        artifact only exists where tc shapes the drain rate below the
        autotuned buffer's fill rate.  Their granted buffer is still recorded
        (``pinned: false``) so no socket is invisible to analysis.

        Clamp detection (audit NT-08): ``net.core.wmem_max`` caps the request
        BEFORE the kernel doubles it, so a class whose request exceeds the
        sysctl silently ends up with a shorter buffering horizon than its
        peers — in w3 the 10 Mbps monolithic socket got 0.34 s of line rate
        while every per-layer class got the intended 0.50 s.  Reading the
        effective value back without comparing it to the expectation is what
        made that invisible, so a short grant now warns and is flagged in the
        exported record.
        """
        key = (neighbor_id, class_idx)
        shaped = (
            bandwidth_mbps is not None
            and math.isfinite(bandwidth_mbps)
            and bandwidth_mbps > 0
        )
        requested = (
            int(bandwidth_mbps * 1e6 / 8 * _SNDBUF_HORIZON_S) if shaped else 0
        )
        try:
            if requested > 0:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, requested)
            effective = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        except OSError as e:
            logger.warning(
                f"[{self._source_node}] Could not pin SO_SNDBUF for "
                f"{neighbor_id} class {class_idx}: {e}"
            )
            return
        if requested <= 0:
            self._sndbuf_pins[key] = {
                "neighbor": neighbor_id,
                "traffic_class": class_idx,
                "bandwidth_mbps": (
                    float(bandwidth_mbps) if bandwidth_mbps is not None else None
                ),
                "pinned": False,
                "requested_bytes": None,
                "expected_bytes": None,
                "effective_bytes": int(effective),
                "granted_line_s": None,
                "clamped": False,
            }
            return
        expected = 2 * requested if _KERNEL_DOUBLES_SNDBUF else requested
        clamped = effective < expected
        # Seconds of this class's line rate the kernel will absorb — the
        # comparable quantity across classes, and the size of the discount
        # the sender-side enqueue clock silently grants this arm.
        granted_line_s = effective * 8.0 / (bandwidth_mbps * 1e6)
        self._sndbuf_pins[key] = {
            "neighbor": neighbor_id,
            "traffic_class": class_idx,
            "bandwidth_mbps": float(bandwidth_mbps),
            "pinned": True,
            "requested_bytes": requested,
            "expected_bytes": expected,
            "effective_bytes": int(effective),
            "granted_line_s": granted_line_s,
            "clamped": clamped,
        }
        message = (
            f"[{self._source_node}] SO_SNDBUF for {neighbor_id} "
            f"class {class_idx}: requested {requested} B "
            f"({bandwidth_mbps:g} Mbps x {_SNDBUF_HORIZON_S:g} s), "
            f"expected {expected} B, effective {effective} B "
            f"= {granted_line_s:.3f} s of line rate"
        )
        if clamped:
            logger.warning(
                f"{message} — CLAMPED: the kernel granted less than expected, "
                f"so this class buffers a shorter horizon than its peers and "
                f"its sender-side enqueue times carry a different bias "
                f"(raise net.core.wmem_max to at least {requested} B)"
            )
        else:
            logger.info(message)

    def sndbuf_telemetry(self) -> list[dict[str, Any]]:
        """Per-(neighbor, class) SO_SNDBUF request vs grant, for the report.

        Sorted for stable diffs.  Emitted into the ``_sender`` telemetry
        block every round so the buffer-discount asymmetry behind NT-08 is
        recoverable from the data alone, without re-reading run logs.
        """
        return [self._sndbuf_pins[key] for key in sorted(self._sndbuf_pins)]

    async def send(
        self,
        neighbor_id: str,
        envelope: federation_pb2.Envelope,
        traffic_class: int = 0,
    ) -> int:
        """Send a protobuf envelope to a neighbor on the given traffic class.

        The envelope is length-prefixed (4-byte big-endian header) before being
        written to the socket.  If no socket exists for the requested traffic
        class, class 0 is used as a fallback (this allows monolithic-mode
        callers to work transparently even when the pool was opened with
        multiple classes).

        Before encoding, the envelope is stamped with
        ``t_send_start_sender_s = time.monotonic()`` — the last instant under
        this process's control before the payload is handed to the socket
        (audit NT-01/NT-05).  Since every container shares the host kernel's
        CLOCK_MONOTONIC, the receiver subtracts this from its own arrival
        stamp to obtain a true one-way wire time.  The stamp necessarily
        precedes serialization (it has to be *inside* the bytes), so a one-way
        time includes this envelope's own encode — sub-millisecond against
        payloads of tens to hundreds of kB, and it biases the measurement
        upward, never down.

        Returns the number of bytes written (header + payload).
        """
        loop = asyncio.get_running_loop()
        key = (neighbor_id, traffic_class)
        sock = self._connections.get(key)
        if sock is None:
            # Fallback: try class 0.  This is hit when:
            #   1. Per-layer mode and the socket for the requested class is
            #      missing (e.g. connection failed for that class).
            #   2. The monitor connection which is always class 0.
            # In case 1 the message gets high-priority treatment instead of
            # the intended class — log a warning so the user can diagnose.
            if traffic_class != 0:
                logger.warning(
                    f"[{self._source_node}] No socket for {neighbor_id} "
                    f"class {traffic_class}, falling back to class 0"
                )
            sock = self._connections.get((neighbor_id, 0))
        if sock is None:
            raise ConnectionError(
                f"[{self._source_node}] No connection to {neighbor_id} "
                f"(class {traffic_class})"
            )
        envelope.t_send_start_sender_s = time.monotonic()
        data = encode_message(envelope)
        await loop.sock_sendall(sock, data)
        return len(data)

    async def close_all(self) -> None:
        """Close every socket in the pool.

        Performs a half-close (SHUT_WR / FIN) before sock.close() so the
        peer receives a clean end-of-stream rather than a TCP RST.  This
        matters at the end of training when the aggregator's final
        broadcast (and the workers' final MetricsReports to the monitor)
        may still be in flight: an abrupt close would discard those.
        Combined with the brief grace-period sleep in ``node.py`` before
        this call, in-flight data has time to drain.
        """
        import socket as _socket
        for (neighbor_id, class_idx), sock in self._connections.items():
            try:
                sock.shutdown(_socket.SHUT_WR)
            except OSError:
                pass  # peer may have already closed
            try:
                sock.close()
            except OSError:
                pass
            logger.debug(
                f"[{self._source_node}] Closed connection to "
                f"{neighbor_id} class {class_idx}"
            )
        self._connections.clear()

    def is_connected(self, neighbor_id: str) -> bool:
        """Check whether at least one connection to *neighbor_id* exists."""
        return any(nid == neighbor_id for nid, _ in self._connections)

    def get_neighbors(self) -> set[str]:
        """Return the set of neighbor IDs that have at least one connection."""
        return {nid for nid, _ in self._connections}
