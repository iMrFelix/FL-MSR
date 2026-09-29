"""Traffic Control (tc) setup for container egress.

Generates and applies Linux tc rules based on per-link class definitions
from the node's outgoing_edges config.  Each node calls this once at
startup, before connections are established and the training loop begins.

The tc hierarchy uses HTB (Hierarchical Token Bucket) as the queuing
discipline, with netem (Network Emulator) qdiscs attached to leaf classes
for latency and packet-loss simulation.  The full tree for a node with
two outgoing edges and two traffic classes looks like:

    Root qdisc 1: htb (default class 1)
    ├── Default class 1:1 (rate 10gbit)
    │       Catches unmatched traffic (monitor metrics, ARP, etc.)
    │
    ├── Edge 0 parent class 1:100
    │   ├── Leaf 1:101  htb rate 1000mbit ceil 1000mbit
    │   │   └── netem 1010: delay 5ms
    │   └── Leaf 1:102  htb rate 500mbit ceil 500mbit
    │       └── netem 1020: delay 10ms loss 1%
    │
    └── Edge 1 parent class 1:200
        ├── Leaf 1:201  htb rate 1000mbit ceil 1000mbit
        │   └── netem 2010: delay 5ms
        └── Leaf 1:202  htb rate 500mbit ceil 500mbit
            └── netem 2020: delay 10ms loss 1%

    Filters (all at root, compound u32 match):
        match ip dst 10.0.0.3/32 AND ip tos 0xb8 0xfc → flowid 1:101
        match ip dst 10.0.0.3/32 AND ip tos 0x28 0xfc → flowid 1:102
        match ip dst 10.0.0.5/32 AND ip tos 0xb8 0xfc → flowid 1:201
        match ip dst 10.0.0.5/32 AND ip tos 0x28 0xfc → flowid 1:202

Classid numbering scheme:
    parent_minor  = (edge_idx + 1) * 100
    leaf_minor    = (edge_idx + 1) * 100 + class_idx + 1
    netem_handle  = leaf_minor * 10

This supports up to 99 traffic classes per edge and 655 edges per node
without overflowing the 16-bit tc minor number space.

Filters are placed at the root qdisc level with compound ``match ip dst``
+ ``match ip tos`` conditions, routing packets directly to leaf classes
in a single lookup.  The ``0xfc`` mask on TOS extracts the 6-bit DSCP
field (the TOS byte encodes DSCP in its top 6 bits and ECN in the bottom
2; the mask zeroes the ECN bits).
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

# Rate applied to parent classes and to leaf classes whose bandwidth_mbps
# is None (unlimited).  10 Gbit/s is high enough that HTB won't impose
# any practical rate limit.
_UNLIMITED_RATE = "10gbit"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_tc_rules(
    interface: str,
    outgoing_edges: list[dict],
    dscp_mapping: dict[int, int],
) -> None:
    """Apply HTB + netem tc rules to the egress interface.

    This function is called once at container startup from ``node.py``.
    It translates the per-link, per-class QoS parameters in the node's
    outgoing_edges config into a set of ``tc`` commands that enforce
    bandwidth limits (HTB) and add latency / packet loss (netem).

    Args:
        interface: Network interface name (typically ``"eth0"`` inside
            Docker containers on a bridge network).
        outgoing_edges: List of edge dicts, each containing:

            - ``"dst_ip"``  — Destination IP address (e.g. ``"10.0.0.3"``).
            - ``"classes"`` — Dict mapping traffic-class index (``int``) to
              a dict with ``"bandwidth_mbps"`` (``float | None``),
              ``"latency_ms"`` (``float``), ``"drop_rate"`` (``float``).

        dscp_mapping: Global mapping of traffic-class index → DSCP value
            (e.g. ``{0: 46, 1: 10}``).  Used to compute the TOS byte that
            each filter matches on (``tos_byte = dscp << 2``).

    Raises:
        RuntimeError: If any critical tc command fails.  The caller in
            ``node.py`` catches this and logs a warning, allowing training
            to continue without traffic shaping.
    """
    if not outgoing_edges:
        logger.info(f"TC setup: no outgoing edges, skipping on {interface}")
        return

    logger.info(
        f"TC setup: configuring {len(outgoing_edges)} edges on {interface}"
    )

    # 1. Clean slate — remove any existing root qdisc.  On first boot there
    #    is no qdisc to remove, so this command is allowed to fail.
    _run_tc(f"qdisc del dev {interface} root", check=False)

    # 2. Root HTB qdisc.  The ``default 1`` clause sends any packet that
    #    doesn't match a filter to class 1:1 (the default class below).
    _run_tc(f"qdisc add dev {interface} root handle 1: htb default 1")

    # 3. Default class — generous rate for unmatched traffic (monitor
    #    metrics, ARP replies, anything not destined for a neighbor).
    _run_tc(
        f"class add dev {interface} parent 1: classid 1:1 "
        f"htb rate {_UNLIMITED_RATE}"
    )

    # 4. Per-edge: parent class, leaf classes, netem qdiscs, filters.
    for edge_idx, edge in enumerate(outgoing_edges):
        _setup_edge(interface, edge_idx, edge, dscp_mapping)

    # 5. Log the resulting tc state so the user can verify the rules
    #    applied correctly (appears once at startup, at INFO level).
    _log_tc_state(interface)


# ---------------------------------------------------------------------------
# Per-edge setup
# ---------------------------------------------------------------------------

def _setup_edge(
    interface: str,
    edge_idx: int,
    edge: dict,
    dscp_mapping: dict[int, int],
) -> None:
    """Set up HTB classes, netem qdiscs, and filters for one outgoing edge.

    Creates a parent HTB class directly under the root, then for each
    traffic class defined on this edge creates:

    1. A **leaf HTB class** (child of the parent) with the configured
       bandwidth as both ``rate`` and ``ceil``.  Setting ceil == rate means
       the class cannot borrow bandwidth from its parent — each traffic
       class gets exactly the bandwidth specified in the config.

    2. A **netem qdisc** (child of the leaf) that introduces the configured
       one-way latency and/or packet-loss probability.  Skipped entirely
       when both values are zero, leaving the default pfifo_fast qdisc in
       place to avoid unnecessary per-packet processing overhead.

    3. A **u32 filter** at the root qdisc that matches packets by
       destination IP *and* TOS byte and routes them directly to the leaf
       class.

    Args:
        interface: Network interface name.
        edge_idx: Zero-based index of this edge in the outgoing_edges list.
            Determines the classid numbering block for this edge.
        edge: Edge dict with ``"dst_ip"`` and ``"classes"`` keys.
        dscp_mapping: Global traffic-class index → DSCP value mapping.
    """
    dst_ip = edge["dst_ip"]
    classes = edge["classes"]
    parent_minor = (edge_idx + 1) * 100

    # Parent class for this destination.  Rate is set to the unlimited cap
    # since per-class bandwidth enforcement happens at the leaf level.
    _run_tc(
        f"class add dev {interface} parent 1: classid 1:{parent_minor} "
        f"htb rate {_UNLIMITED_RATE}"
    )

    logger.info(
        f"TC edge {edge_idx}: dst={dst_ip}, "
        f"{len(classes)} traffic class(es), parent=1:{parent_minor}"
    )

    # Leaf classes, netem qdiscs, and filters — one set per traffic class.
    for class_idx_raw, params in sorted(classes.items()):
        class_idx = int(class_idx_raw)  # YAML may parse keys as strings
        leaf_minor = parent_minor + class_idx + 1
        _setup_leaf(interface, parent_minor, leaf_minor, params)
        _setup_filter(interface, dst_ip, leaf_minor, class_idx, dscp_mapping)


# ---------------------------------------------------------------------------
# Leaf class + netem
# ---------------------------------------------------------------------------

def _setup_leaf(
    interface: str,
    parent_minor: int,
    leaf_minor: int,
    params: dict,
) -> None:
    """Create an HTB leaf class and optionally attach a netem qdisc.

    The leaf class enforces a bandwidth ceiling via HTB.  When the edge
    config specifies non-zero latency or drop rate, a netem qdisc is
    attached below the leaf to emulate those conditions.  If both are zero,
    netem is omitted and the leaf uses its default pfifo_fast qdisc.

    Args:
        interface: Network interface name.
        parent_minor: Minor number of the parent HTB class.
        leaf_minor: Minor number for this leaf class.
        params: Dict with ``"bandwidth_mbps"``, ``"latency_ms"``,
            ``"drop_rate"``.
    """
    bw = params.get("bandwidth_mbps")
    latency = params.get("latency_ms", 0)
    drop = params.get("drop_rate", 0)

    # Bandwidth: None (or missing) means unlimited.
    rate_str = f"{bw}mbit" if bw is not None else _UNLIMITED_RATE

    # HTB leaf class.  ceil == rate → no borrowing from the parent.
    _run_tc(
        f"class add dev {interface} parent 1:{parent_minor} "
        f"classid 1:{leaf_minor} htb rate {rate_str} ceil {rate_str}"
    )

    # Netem qdisc — only if there is actual impairment to apply.
    netem_args: list[str] = []
    if latency > 0:
        netem_args.append(f"delay {latency}ms")
    if drop > 0:
        # drop_rate is a probability in [0, 1]; netem expects a percentage.
        loss_pct = drop * 100
        netem_args.append(f"loss {loss_pct}%")

    if netem_args:
        # Handle numbering: leaf_minor * 10 guarantees uniqueness across
        # all edges and classes (e.g. leaf 101 → handle 1010:).
        netem_handle = leaf_minor * 10
        _run_tc(
            f"qdisc add dev {interface} parent 1:{leaf_minor} "
            f"handle {netem_handle}: netem {' '.join(netem_args)}"
        )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _setup_filter(
    interface: str,
    dst_ip: str,
    leaf_minor: int,
    class_idx: int,
    dscp_mapping: dict[int, int],
) -> None:
    """Add a u32 filter at the root routing packets to a leaf class.

    The filter uses two compound match conditions evaluated as logical AND:

    - ``match ip dst {dst_ip}/32`` — exact destination IP.
    - ``match ip tos {tos_hex} 0xfc`` — 6-bit DSCP field in the TOS byte.

    The TOS byte layout is ``[DSCP (6 bits)][ECN (2 bits)]``, so the DSCP
    value is left-shifted by 2 to get the TOS byte, and the mask ``0xfc``
    (binary ``11111100``) zeroes the ECN bits during comparison.

    Args:
        interface: Network interface name.
        dst_ip: Destination IP address to match.
        leaf_minor: Minor classid of the target leaf class.
        class_idx: Traffic class index (used to look up DSCP value).
        dscp_mapping: Global traffic-class index → DSCP value mapping.
    """
    dscp = dscp_mapping.get(class_idx, 0)
    tos_byte = dscp << 2
    tos_hex = f"0x{tos_byte:02x}"

    _run_tc(
        f"filter add dev {interface} parent 1: protocol ip u32 "
        f"match ip dst {dst_ip}/32 "
        f"match ip tos {tos_hex} 0xfc "
        f"flowid 1:{leaf_minor}"
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _log_tc_state(interface: str) -> None:
    """Log the current tc configuration for debugging and verification.

    Runs ``tc -s class show`` (to see the HTB hierarchy with statistics)
    and ``tc filter show`` (to see the u32 filter rules) and logs both
    at INFO level.  Called once after all rules are applied.
    """
    for cmd in [
        f"tc -s class show dev {interface}",
        f"tc filter show dev {interface}",
    ]:
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            logger.info(f"TC state ({cmd}):\n{result.stdout}")
        elif result.returncode != 0:
            logger.warning(f"TC state query failed ({cmd}): {result.stderr}")


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------

def _run_tc(cmd: str, check: bool = True) -> None:
    """Execute a tc command via subprocess.

    The *cmd* argument is the tc subcommand without the leading ``tc``
    prefix (e.g. ``"qdisc add dev eth0 ..."``).  The function prepends
    ``tc`` and splits on whitespace to build the argv list.

    Args:
        cmd: The tc subcommand string.
        check: If ``True`` (default), raise :class:`RuntimeError` on a
            non-zero exit code.  Set to ``False`` for cleanup commands
            that may fail harmlessly (e.g. deleting a qdisc that doesn't
            exist yet on first boot).
    """
    full_cmd = f"tc {cmd}"
    logger.debug(f"TC: {full_cmd}")
    result = subprocess.run(full_cmd.split(), capture_output=True, text=True)
    if result.returncode != 0:
        if check:
            raise RuntimeError(
                f"TC command failed (rc={result.returncode}): "
                f"{full_cmd}\nstderr: {result.stderr}"
            )
        else:
            logger.debug(
                f"TC command returned non-zero (ignored): "
                f"{full_cmd}\n{result.stderr.strip()}"
            )
