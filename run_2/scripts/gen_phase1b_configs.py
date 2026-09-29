"""Generate the phase-1b first-phase experiment configs (writeup/11-experiment-plan.md).

Backbone arms (no code additions required): EXP-4 monolithic reference,
EXP-6 epsilon=0 per-layer integrity, EXP-1 coverage grid eps in {0.2,0.3,0.4},
EXP-5 C=1 single-pipe. CIFAR-10 + deep_cnn + Dirichlet(0.1), FedAvg,
6 seeds {41..46}, lr 0.1, momentum 0.0, delta_sq_norm trigger metric,
coverage_eft assignment, drop slippage, no aging, warmup 0.

Substrate: 3 workers + 1 aggregator. Per-layer arms use 3 classes at 6/3/1 Mbps
(sum 10). monolithic + C=1 arms use one 10 Mbps class. 5 ms latency, 0 loss.

Usage: python scripts/gen_phase1b_configs.py [--rounds N] [--out DIR]
"""
from __future__ import annotations
import argparse
import os
import yaml

SEEDS = [41, 42, 43, 44, 45, 46]
WORKERS = [1, 2, 3]
AGG = 0


def edges_3class():
    cls = {
        0: {"bandwidth_mbps": 6, "latency_ms": 5, "drop_rate": 0.0},
        1: {"bandwidth_mbps": 3, "latency_ms": 5, "drop_rate": 0.0},
        2: {"bandwidth_mbps": 1, "latency_ms": 5, "drop_rate": 0.0},
    }
    e = []
    for w in WORKERS:
        e.append({"src": w, "dst": AGG, "classes": {k: dict(v) for k, v in cls.items()}})
        e.append({"src": AGG, "dst": w, "classes": {k: dict(v) for k, v in cls.items()}})
    return e


def edges_1class(mbps=10):
    cls = {0: {"bandwidth_mbps": mbps, "latency_ms": 5, "drop_rate": 0.0}}
    e = []
    for w in WORKERS:
        e.append({"src": w, "dst": AGG, "classes": {0: dict(cls[0])}})
        e.append({"src": AGG, "dst": w, "classes": {0: dict(cls[0])}})
    return e


def base_training(seed, rounds):
    return {
        "dataset": {"name": "cifar10", "partition": {"strategy": "dirichlet", "alpha": 0.1, "seed": seed}},
        "model": "deep_cnn",
        "algorithm": "fedavg",
        "epochs_per_round": 1,
        "total_rounds": rounds,
        "batch_size": 64,
        "learning_rate": 0.1,
        "optimizer": "sgd",
        "momentum": 0.0,
    }


def perlayer_knobs(eps):
    return {
        "update_mode": "per_layer",
        "epsilon_deadline": eps,
        "epsilon_warmup_rounds": 0,
        "importance_metric_v2": "delta_sq_norm",
        "assignment_strategy": "coverage_eft",
        "late_layer_policy": "drop",
        "aging_mode": "none",
        "watchdog_factor": 3.0,
    }


def make_config(arm_kind, seed, rounds, eps=0.0, classes=3):
    tr = base_training(seed, rounds)
    if arm_kind == "monolithic":
        tr["update_mode"] = "monolithic"
        tc = {"num_classes": 1, "dscp_mapping": {0: 0}}
        topo = edges_1class(10)
    else:  # per_layer
        tr.update(perlayer_knobs(eps))
        if classes == 3:
            tc = {"num_classes": 3, "dscp_mapping": {0: 46, 1: 10, 2: 0}}
            topo = edges_3class()
        else:  # C=1
            tc = {"num_classes": 1, "dscp_mapping": {0: 0}}
            topo = edges_1class(10)
    return {
        "federation": {"num_nodes": 4, "roles": {0: "aggregator"}, "topology": {"edges": topo}},
        "traffic_classes": tc,
        "training": tr,
        "monitoring": {"tensorboard": False, "report_output": "./results/"},
    }


def write(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--out", default="configs/experiments/phase1b")
    args = ap.parse_args()
    n = 0
    specs = []  # (exp, arm, seed, relpath)
    for seed in SEEDS:
        # EXP-4 monolithic reference
        p = f"{args.out}/exp4_monolithic/mono/seed{seed}.yaml"
        write(make_config("monolithic", seed, args.rounds), p); specs.append(("exp4", "mono", seed, p)); n += 1
        # EXP-6 epsilon=0 per-layer integrity (3-class)
        p = f"{args.out}/exp6_eps0/coverage_eft_eps0/seed{seed}.yaml"
        write(make_config("per_layer", seed, args.rounds, eps=0.0, classes=3), p); specs.append(("exp6", "eps0", seed, p)); n += 1
        # EXP-1 coverage grid
        for eps in (0.2, 0.3, 0.4):
            tag = str(eps).replace(".", "")
            p = f"{args.out}/exp1_grid/coverage_eft_eps{tag}/seed{seed}.yaml"
            write(make_config("per_layer", seed, args.rounds, eps=eps, classes=3), p); specs.append(("exp1", f"eps{tag}", seed, p)); n += 1
        # EXP-5 C=1 single-pipe eps=0.3
        p = f"{args.out}/exp5_c1/coverage_eft_c1_eps03/seed{seed}.yaml"
        write(make_config("per_layer", seed, args.rounds, eps=0.3, classes=1), p); specs.append(("exp5", "c1_eps03", seed, p)); n += 1
    # write a specs manifest (run order: exp4 -> exp6 -> exp1 -> exp5) for the runner
    order = {"exp4": 0, "exp6": 1, "exp1": 2, "exp5": 3}
    specs.sort(key=lambda s: (order[s[0]], s[1], s[2]))
    with open(f"{args.out}/specs.txt", "w") as f:
        for exp, arm, seed, path in specs:
            f.write(f"{exp}\t{arm}\t{seed}\t{path}\n")
    print(f"wrote {n} configs to {args.out} (rounds={args.rounds}); specs.txt has {len(specs)} runs")


if __name__ == "__main__":
    main()
