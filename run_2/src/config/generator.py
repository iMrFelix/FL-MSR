"""Generate docker-compose and per-node configs from a validated ExperimentConfig.

This module is the bridge between the experiment-level YAML config (which
describes the whole federation) and the per-node YAML configs (which tell
each container exactly what it needs to know).  It also produces the
``docker-compose.yml`` dict that wires up all containers on a shared Docker
bridge network.

Key responsibilities:
- Assign deterministic IPs to every node and the monitor.
- Build per-node neighbor lists from the topology edges.
- Propagate training hyper-parameters, traffic class info, and DSCP
  mappings into each node's config so the ConnectionPool can open the
  right number of sockets with the correct TOS markings.
- Generate the docker-compose service definitions including volume mounts,
  capabilities (NET_ADMIN for tc), and network config.
"""

from __future__ import annotations

import os

from src.config.schema import ExperimentConfig, TopologyEdge

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-run subnet offset for CONCURRENT stacks (writeup/11 §2, P0 code work):
# FL_SUBNET_OCTET=k gives this run the disjoint subnet 10.k.0.0/24 and
# suffixed container names, so multiple docker-compose stacks can coexist.
# Default 0 preserves the historical 10.0.0.0/24 + bare-name behaviour.
_OCTET = int(os.environ.get("FL_SUBNET_OCTET", "0"))
_SUBNET = f"10.{_OCTET}.0.0/24"
_GATEWAY = f"10.{_OCTET}.0.1"
_MONITOR_IP = f"10.{_OCTET}.0.254"
_NODE_PORT = 5000
_MONITOR_PORT = 5100
_NODE_IMAGE = "fl-node:latest"
_MONITOR_IMAGE = "fl-monitor:latest"
_NETWORK_NAME = "fl-net"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_node_ip(node_id: int) -> str:
    """Return the deterministic IP for a node: ``10.0.0.(node_id + 2)``.

    Node 0 gets .2, node 1 gets .3, etc.  The .1 address is reserved for
    the Docker gateway and .254 for the monitor.
    """
    return f"10.{_OCTET}.0.{node_id + 2}"


def _service_name(node_id: int) -> str:
    """Canonical docker-compose service name for a node (e.g. ``node-0``)."""
    return f"node-{node_id}"


def _role_for_node(config: ExperimentConfig, node_id: int) -> str:
    """Resolve the role of a node, defaulting to ``'worker'``.

    The roles mapping in the config is optional; most experiments use
    pure decentralized topologies where every node is a worker.
    """
    return config.federation.roles.get(node_id, "worker")


def _edge_to_dict(edge: TopologyEdge) -> dict:
    """Serialize a TopologyEdge to a plain dict (suitable for YAML/JSON).

    Used when embedding edge information in per-node configs so that the
    node process can inspect its own outgoing/incoming link parameters.
    Includes resolved IP addresses (``src_ip``, ``dst_ip``) so that the
    tc/netem setup code (Stage 3) can create filter rules targeting
    specific destination IPs without needing to look up IPs from node IDs.
    """
    return {
        "src": edge.src,
        "dst": edge.dst,
        "src_ip": get_node_ip(edge.src),
        "dst_ip": get_node_ip(edge.dst),
        "classes": {
            cls_id: {
                "bandwidth_mbps": params.bandwidth_mbps,
                "latency_ms": params.latency_ms,
                "drop_rate": params.drop_rate,
            }
            for cls_id, params in edge.classes.items()
        },
    }


def _neighbors_for_node(
    config: ExperimentConfig,
    node_id: int,
) -> list[dict]:
    """Build the neighbor list for *node_id* from the topology edges.

    A neighbor is any node that shares an edge (in either direction) with
    *node_id*.  The returned list contains the Docker service name, the
    deterministic IP, and the listen port for each neighbor.
    """
    neighbor_ids: set[int] = set()
    for edge in config.federation.topology.edges:
        if edge.src == node_id:
            neighbor_ids.add(edge.dst)
        elif edge.dst == node_id:
            neighbor_ids.add(edge.src)

    return [
        {
            "neighbor_id": _service_name(nid),
            "ip": get_node_ip(nid),
            "port": _NODE_PORT,
        }
        for nid in sorted(neighbor_ids)
    ]


# ---------------------------------------------------------------------------
# Docker-compose generation
# ---------------------------------------------------------------------------

def generate_docker_compose(
    config: ExperimentConfig,
    data_dir: str,
) -> dict:
    """Generate a ``docker-compose.yml`` content dict.

    Creates:
    - One service per node (``node-0``, ``node-1``, ...).
    - One ``monitor`` service.
    - A single shared bridge network ``fl-net`` with subnet ``10.0.0.0/24``.
    - Node *i* gets IP ``10.0.0.(i+2)``.
    - Monitor gets IP ``10.0.0.254``.
    - Each node mounts the shared data directory (read-only) and its own
      per-node YAML config.
    - All nodes receive the ``NET_ADMIN`` capability so ``tc``/``netem``
      rules can be applied from within the container (Stage 3).

    Parameters
    ----------
    config:
        A validated :class:`ExperimentConfig`.
    data_dir:
        Host path to the root data directory containing the pre-partitioned
        ``.npz`` files.  Mounted as ``/data:ro`` inside every node container.

    Returns
    -------
    dict
        A dict that can be serialized directly to ``docker-compose.yml``.
    """
    services: dict[str, dict] = {}

    for node_id in range(config.federation.num_nodes):
        name = _service_name(node_id)
        ip = get_node_ip(node_id)
        role = _role_for_node(config, node_id)

        services[name] = {
            "image": _NODE_IMAGE,
            "container_name": name if _OCTET == 0 else f"{name}-o{_OCTET}",
            "hostname": name,
            "cap_add": ["NET_ADMIN"],
            "environment": {
                "NODE_ID": str(node_id),
                "NODE_NAME": name,
                "NODE_IP": ip,
                "NODE_PORT": str(_NODE_PORT),
                "ROLE": role,
                "MONITOR_IP": _MONITOR_IP,
                "MONITOR_PORT": str(_MONITOR_PORT),
            },
            "volumes": [
                f"{data_dir}:/data:ro",
                f"./configs/{name}.yaml:/config/node.yaml:ro",
            ],
            "networks": {
                _NETWORK_NAME: {
                    "ipv4_address": ip,
                },
            },
        }


    # Monitor service
    monitor_ports: list[str] = []
    if config.monitoring.tensorboard:
        tb_port = config.monitoring.tensorboard_port
        monitor_ports.append(f"{tb_port}:{tb_port}")

    monitor_service: dict = {
        "image": _MONITOR_IMAGE,
        "container_name": "monitor" if _OCTET == 0 else f"monitor-o{_OCTET}",
        "hostname": "monitor",
        "environment": {
            "MONITOR_IP": _MONITOR_IP,
            "MONITOR_PORT": str(_MONITOR_PORT),
            "TENSORBOARD_ENABLED": str(config.monitoring.tensorboard).lower(),
            "TENSORBOARD_PORT": str(config.monitoring.tensorboard_port),
            "REPORT_OUTPUT": config.monitoring.report_output,
        },
        "volumes": [
            f"{config.monitoring.report_output}:/results",
            "./configs/monitor.yaml:/config/monitor.yaml:ro",
        ],
        "networks": {
            _NETWORK_NAME: {
                "ipv4_address": _MONITOR_IP,
            },
        },
    }
    if monitor_ports:
        monitor_service["ports"] = monitor_ports

    services["monitor"] = monitor_service

    compose: dict = {
        "services": services,
        "networks": {
            _NETWORK_NAME: {
                "driver": "bridge",
                "ipam": {
                    "config": [
                        {
                            "subnet": _SUBNET,
                            "gateway": _GATEWAY,
                        },
                    ],
                },
            },
        },
    }

    return compose


# ---------------------------------------------------------------------------
# Per-node config generation
# ---------------------------------------------------------------------------

def generate_node_configs(config: ExperimentConfig) -> dict[str, dict]:
    """Generate per-node configuration dicts.

    Each node config contains everything the node process needs at runtime:
    its identity, neighbor list, relevant edges, training hyper-parameters,
    traffic class / DSCP information, and monitor coordinates.

    The traffic class info (``num_traffic_classes`` and ``dscp_mapping``) is
    placed at the top level of the node config (not inside ``training``)
    because it's a transport-layer concern: node.py reads it to configure
    the ConnectionPool, and the engine reads ``num_traffic_classes`` from the
    training sub-dict to know how many classes to pass to the traffic mapping
    function.

    Parameters
    ----------
    config:
        A validated :class:`ExperimentConfig`.

    Returns
    -------
    dict[str, dict]
        Mapping from service name (e.g. ``"node-0"``) to its config dict.
    """
    training = config.training
    tc = config.traffic_classes
    edges = config.federation.topology.edges

    node_configs: dict[str, dict] = {}

    for node_id in range(config.federation.num_nodes):
        name = _service_name(node_id)

        outgoing = [_edge_to_dict(e) for e in edges if e.src == node_id]
        incoming = [_edge_to_dict(e) for e in edges if e.dst == node_id]

        node_cfg: dict = {
            "node_id": name,
            "ip": get_node_ip(node_id),
            "port": _NODE_PORT,
            "role": _role_for_node(config, node_id),
            "seed": training.dataset.partition.seed,
            # Neighbors
            "neighbors": _neighbors_for_node(config, node_id),
            "outgoing_edges": outgoing,
            "incoming_edges": incoming,
            # Traffic classes — used by node.py to configure the
            # ConnectionPool (how many sockets per neighbor, which TOS byte
            # on each).  Also passed into the training sub-dict so the
            # engine knows how many classes are available for the importance
            # -> traffic class mapping.
            "num_traffic_classes": tc.num_classes,
            "dscp_mapping": tc.dscp_mapping,
            # Training hyper-parameters — consumed by TrainingEngine
            "training": {
                "algorithm": training.algorithm,
                "model": training.model,
                "dataset": training.dataset.name,
                "partition_strategy": training.dataset.partition.strategy,
                "epochs_per_round": training.epochs_per_round,
                "total_rounds": training.total_rounds,
                "batch_size": training.batch_size,
                "learning_rate": training.learning_rate,
                "optimizer": training.optimizer,
                "momentum": training.momentum,
                # Competent-baseline knobs (writeup/16 §5): selective
                # passthrough — without these lines container nodes
                # silently run the legacy weak baseline.
                "clipnorm": training.clipnorm,
                "lr_schedule": training.lr_schedule,
                "lr_decay_factor": training.lr_decay_factor,
                "lr_decay_every": training.lr_decay_every,
                "lr_eta_min": training.lr_eta_min,
                "server_momentum": training.server_momentum,
                "update_mode": training.update_mode,
                "importance_metric": training.importance_metric,
                # ε-deadline tolerance for per-layer round completion;
                # ignored in monolithic mode.  See extension 02.
                "epsilon_deadline": training.epsilon_deadline,
                # How to handle layers arriving after ε-trigger fires.  See
                # extension 03 and gate ruling G3.
                "late_layer_policy": training.late_layer_policy,
                # Overnight-evaluation knobs (gate doc §7; see extension 04
                # for which component consumes which key).  The passthrough
                # here is selective by design, so every new TrainingConfig
                # field MUST be listed explicitly or nodes silently fall
                # back to engine-side defaults.
                "importance_metric_v2": training.importance_metric_v2,
                "assignment_strategy": training.assignment_strategy,
                "aging_mode": training.aging_mode,
                "aging_lambda": training.aging_lambda,
                "aging_tau_max": training.aging_tau_max,
                # Audit fixes TRIG-1/ML-01 and TRIG-5/ML-07: both default to
                # the CORRECTED semantics, so a missing passthrough here
                # would silently run the pre-fix mechanism.
                "aging_age_basis": training.aging_age_basis,
                "epsilon_budget_metric": training.epsilon_budget_metric,
                "cyclic_k": training.cyclic_k,
                "epsilon_warmup_rounds": training.epsilon_warmup_rounds,
                "watchdog_factor": training.watchdog_factor,
                # Skip-feedback v2 (plan T1): without this passthrough
                # container nodes silently run skip_feedback='off'
                # (interface-doc rule 1.7).
                "skip_feedback": training.skip_feedback,
                "skip_fedluar_count": training.skip_fedluar_count,
                # Async-specific parameters (ignored by sync algorithms).
                "staleness_threshold": training.staleness_threshold,
                "eval_every": training.eval_every,
                # K0: evaluate on the shared global held-out test set instead
                # of the skewed per-node local val split.  Selective
                # passthrough — must be listed explicitly or nodes silently
                # fall back to local-val (global_eval=False).
                "global_eval": training.global_eval,
                # Seed is needed by GossipSGD for reproducible neighbor
                # selection.  The top-level seed is also used by node.py
                # for model init, but algorithms only see training_config.
                "seed": training.dataset.partition.seed,
                # Duplicated here because the engine reads it from the
                # training dict to decide how many traffic classes to use
                # when mapping importance scores.
                "num_traffic_classes": tc.num_classes,
                # Role is needed by FedAvg to distinguish aggregator from
                # worker behavior.  Other algorithms ignore this field.
                "role": _role_for_node(config, node_id),
            },
            # Monitoring
            "monitor_ip": _MONITOR_IP,
            "monitor_port": _MONITOR_PORT,
        }

        node_configs[name] = node_cfg

    return node_configs
