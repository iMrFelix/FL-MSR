"""TCP transport server for incoming federated learning messages.

Provides a non-blocking asyncio-based TCP server that accepts incoming
connections from neighbor nodes, reads length-prefixed protobuf messages
using the FrameReader state machine, and dispatches complete Envelope
messages to a registered handler callback.

Outgoing connections are handled by the ConnectionPool (see
``connection_pool.py``), which supports per-traffic-class DSCP marking.
The server side does not need traffic class awareness — all incoming
messages are received on whatever port the server listens on regardless
of the sender's DSCP markings.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from typing import Callable, Awaitable

from src.network.framing import FrameReader
from src.proto_gen import federation_pb2

logger = logging.getLogger(__name__)

# Called with (envelope, first_byte_monotonic_s, last_byte_monotonic_s) for
# every inbound envelope, synchronously, just before the message handler runs.
ArrivalObserver = Callable[[federation_pb2.Envelope, float, float], None]


class TransportServer:
    """Non-blocking TCP server using asyncio event loop + raw sockets.

    Accepts incoming connections, reads length-prefixed protobuf messages,
    and dispatches each complete Envelope to the registered handler.  The
    handler is typically set by the TrainingEngine to route messages to the
    correct algorithm callback (model updates) or metric collector.

    The server must be started (``await server.start()``) before any
    neighbor connections are established, because peers may start sending
    as soon as their outbound connect succeeds.  The handler must also
    be registered before connecting (``server.set_handler(...)``), otherwise
    early messages will be silently dropped.
    """

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._server_sock: socket.socket | None = None
        self._running = False
        self._handler: Callable[[federation_pb2.Envelope], Awaitable[None]] | None = None
        self._arrival_observer: ArrivalObserver | None = None
        self._accept_task: asyncio.Task | None = None
        self._connection_tasks: set[asyncio.Task] = set()

    def set_handler(self, handler: Callable[[federation_pb2.Envelope], Awaitable[None]]) -> None:
        """Set the message handler callback."""
        self._handler = handler

    def set_arrival_observer(self, observer: ArrivalObserver | None) -> None:
        """Register a transport-level arrival-stamp callback (audit NT-05).

        The observer is invoked synchronously with ``(envelope,
        first_byte_s, last_byte_s)`` immediately before the handler, so a
        consumer sees the true receive-side instants rather than a
        ``time.monotonic()`` taken after parsing and dispatch.  It exists
        because receiver-side instrumentation used to hang off the importance
        manifest, which made it structurally unavailable to the two flows the
        system is compared against: the monolithic ModelUpdate and the
        aggregator's downlink broadcast, neither of which carries a manifest.

        Kept separate from ``set_handler`` so consumers that do not care
        (the monitor's collector) need no signature change.  Observers must
        not block or await; exceptions are logged and swallowed, because a
        telemetry defect must never take down the receive path.
        """
        self._arrival_observer = observer

    async def start(self) -> None:
        """Start listening for connections."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.setblocking(False)
        self._server_sock.bind((self._host, self._port))
        self._server_sock.listen(128)
        self._running = True
        self._accept_task = asyncio.create_task(self._accept_loop())
        logger.info(f"Transport server listening on {self._host}:{self._port}")

    async def stop(self) -> None:
        """Stop the server and close all connections."""
        self._running = False
        if self._accept_task:
            self._accept_task.cancel()
            try:
                await self._accept_task
            except asyncio.CancelledError:
                pass
        for task in list(self._connection_tasks):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._connection_tasks.clear()
        if self._server_sock:
            self._server_sock.close()
            self._server_sock = None
        logger.info("Transport server stopped")

    async def _accept_loop(self) -> None:
        """Accept incoming connections in a loop.

        Each accepted connection spawns a background task that reads
        framed protobuf messages until the connection closes or the
        server stops.
        """
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                client_sock, addr = await loop.sock_accept(self._server_sock)
                client_sock.setblocking(False)
                client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                task = asyncio.create_task(self._handle_connection(client_sock, addr))
                self._connection_tasks.add(task)
                task.add_done_callback(self._connection_tasks.discard)
                logger.debug(f"Accepted connection from {addr}")
            except OSError as e:
                if self._running:
                    logger.error(f"Accept error: {e}")
                    await asyncio.sleep(0.1)

    async def _handle_connection(self, sock: socket.socket, addr: tuple) -> None:
        """Read frames from a single connection and dispatch to the handler.

        Uses the FrameReader state machine to handle partial TCP reads —
        ``sock_recv`` may return fewer bytes than a complete message, so the
        reader accumulates data across calls and yields complete Envelope
        messages as they become available.

        Arrival stamping (audit NT-05): every read is stamped once, and the
        stamp of the read that carried a frame's *first* byte is remembered
        until that frame completes.  ``first_byte`` is therefore exact for
        payloads spanning several reads (the large ones — a monolithic model
        or a fat conv layer) and collapses onto ``last_byte`` for frames that
        arrive whole inside one read, which is the honest resolution limit of
        a userspace receiver.
        """
        loop = asyncio.get_running_loop()
        reader = FrameReader()
        frame_start: float | None = None
        try:
            while self._running:
                data = await loop.sock_recv(sock, 65536)
                arrival = time.monotonic()
                if not data:
                    break  # Connection closed
                if frame_start is None:
                    frame_start = arrival
                envelopes = reader.feed(data)
                # Leftover bytes belong to the next frame and arrived in this
                # read; nothing left means the next frame starts in a future
                # read, whose own stamp will open it.
                next_start = arrival if reader.has_partial_frame() else None
                for index, envelope in enumerate(envelopes):
                    # Only the first envelope of the batch can have started
                    # before this read; any further ones were fully contained
                    # in it.
                    first_byte = frame_start if index == 0 else arrival
                    self._notify_arrival(envelope, first_byte, arrival)
                    if self._handler:
                        await self._handler(envelope)
                if envelopes:
                    frame_start = next_start
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            logger.debug(f"Connection from {addr} closed: {e}")
        finally:
            sock.close()

    def _notify_arrival(
        self,
        envelope: federation_pb2.Envelope,
        first_byte_s: float,
        last_byte_s: float,
    ) -> None:
        """Hand arrival stamps to the observer, never failing the receive."""
        if self._arrival_observer is None:
            return
        try:
            self._arrival_observer(envelope, first_byte_s, last_byte_s)
        except Exception as exc:  # noqa: BLE001 - telemetry must not drop data
            logger.warning(f"Arrival observer raised, ignoring: {exc}")
