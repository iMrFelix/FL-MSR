#!/usr/bin/env python3
"""Generate the 10-node pilot configs (2026-08-06, pre-meeting).

Design (pre-registered; see conversation with Felix):
  - Question: does the accuracy cost of layer-shedding shrink with client
    count (1/N dilution hypothesis)? Accuracy-only pilot — timing claims are
    out of scope at 10 nodes until ingress shaping exists (egress-only tc).
  - 4 arms x 5 seeds (41-45), 20 rounds, serial execution on one box:
      mono        monolithic baseline (1 traffic class @ 10 Mbps, as wave-3)
      eps0        per-layer, eps=0, coverage_eft — equivalence + data-integrity
                  canary: must match mono exactly on clean partitions
      drop_eps03  per-layer, eps=0.3, coverage_eft, drop — THE anchor arm,
                  transported unchanged from wave-3
      cyclic_k7   per-layer, eps=0, blind rotation k=7 of 14 layers — blind
                  control; k kept at the wave-3-calibrated 7/14 layer fraction
                  for comparability (realized bytes reported, not re-calibrated)
  - Topology scales the wave-3 pattern 3x: 9 workers + aggregator (node 0),
    per-layer arms use the same 3 link classes (6/3/1 Mbps, 5 ms) per worker,
    mono uses the single 10 Mbps class. Everything in `training:` matches
    wave-3 verbatim except the partition, which Dirichlet-splits over 10 nodes.
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "experiments" / "n10pilot"
NUM_NODES = 10
SEEDS = [41, 42, 43, 44, 45]

CLASSES_MONO = {0: {"bandwidth_mbps": 10, "latency_ms": 5, "drop_rate": 0.0}}
CLASSES_3 = {
    0: {"bandwidth_mbps": 6, "latency_ms": 5, "drop_rate": 0.0},
    1: {"bandwidth_mbps": 3, "latency_ms": 5, "drop_rate": 0.0},
    2: {"bandwidth_mbps": 1, "latency_ms": 5, "drop_rate": 0.0},
}


def topology(classes: dict) -> dict:
    edges = []
    for w in range(1, NUM_NODES):
        for src, dst in ((w, 0), (0, w)):
            edges.append({"src": src, "dst": dst, "classes": classes})
    return {"edges": edges}


def base_training(seed: int) -> dict:
    return {
        "dataset": {
            "name": "cifar10",
            "partition": {"strategy": "dirichlet", "alpha": 0.1, "seed": seed},
        },
        "model": "deep_cnn",
        "algorithm": "fedavg",
        "epochs_per_round": 1,
        "total_rounds": 20,
        "batch_size": 64,
        "learning_rate": 0.1,
        "optimizer": "sgd",
        "momentum": 0.0,
        "global_eval": True,
    }


def arm_config(arm: str, seed: int) -> dict:
    if arm == "mono":
        classes, num_classes, dscp = CLASSES_MONO, 1, {0: 0}
    else:
        classes, num_classes, dscp = CLASSES_3, 3, {0: 46, 1: 10, 2: 0}
    cfg = {
        "federation": {
            "num_nodes": NUM_NODES,
            "roles": {0: "aggregator"},
            "topology": topology(classes),
        },
        "traffic_classes": {"num_classes": num_classes, "dscp_mapping": dscp},
        "training": base_training(seed),
        "monitoring": {"tensorboard": False, "report_output": "./results/"},
    }
    t = cfg["training"]
    if arm == "mono":
        t["update_mode"] = "monolithic"
    else:
        t.update({
            "update_mode": "per_layer",
            "epsilon_warmup_rounds": 0,
            "importance_metric_v2": "delta_sq_norm",
            "late_layer_policy": "drop",
            "aging_mode": "none",
            "watchdog_factor": 3.0,
        })
        if arm == "eps0":
            t.update({"epsilon_deadline": 0.0, "assignment_strategy": "coverage_eft"})
        elif arm == "drop_eps03":
            t.update({"epsilon_deadline": 0.3, "assignment_strategy": "coverage_eft"})
        elif arm == "cyclic_k7":
            t.update({"epsilon_deadline": 0.0, "assignment_strategy": "cyclic",
                      "cyclic_k": 7})
        else:
            raise ValueError(arm)
    return cfg


def main() -> None:
    n = 0
    for arm in ("mono", "eps0", "drop_eps03", "cyclic_k7"):
        for seed in SEEDS:
            path = OUT / arm / f"seed{seed}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                yaml.dump(arm_config(arm, seed), f, default_flow_style=False,
                          sort_keys=False)
            n += 1
    print(f"wrote {n} configs under {OUT}")


if __name__ == "__main__":
    main()
