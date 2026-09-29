#!/usr/bin/env python3
"""Generate the h150 late-stage campaign (2026-08-05, Felix's convergence ask).

Motivation: no prior run ever reached late-stage training — the 50r runs were
arrested on a constant-lr noise plateau ~25-30pp below this net's ceiling
(writeup/16; the central fl_matched run reaches ~73% by epoch 14 with the same
optimizer, so the plateau is a federation+recipe artifact).  Under the
competent baseline with cosine annealing, 150 rounds x 1 local epoch over the
full 50k (workers_only) ~= 50 central-epoch-equivalents — the region where the
central competent recipe converges.  This makes the endpoint a converged-for-
budget read and lets us report time/bytes-to-target-accuracy curves.

Arms: mono / drop_eps03 / recycle_eps03, competent-baseline bundle
(clipnorm=1.0, FedAvgM 0.9, cosine over 150r, workers_only), seeds 41-43.

Emits: configs/experiments/h150/<arm>/seed<seed>.yaml + specs.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config.schema import load_config  # noqa: E402

PL_BASE = ROOT / "configs/experiments/phase1b/exp1_grid/coverage_eft_eps03/seed41.yaml"
MONO_BASE = ROOT / "configs/experiments/phase1b/exp4_monolithic/mono/seed41.yaml"
OUT = ROOT / "configs/experiments/h150"
SEEDS = [41, 42, 43]
ROUNDS = 150

BASELINE = {
    "clipnorm": 1.0,
    "server_momentum": 0.9,
    "lr_schedule": "cosine",
}

ARMS = {
    "mono": (MONO_BASE, {}),
    "drop_eps03": (PL_BASE, {"late_layer_policy": "drop"}),
    "recycle_eps03": (PL_BASE, {"late_layer_policy": "recycle_last_delta"}),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = []
    for arm, (base, patch) in ARMS.items():
        arm_dir = OUT / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            cfg = yaml.safe_load(base.read_text())
            tr = cfg["training"]
            tr["dataset"]["partition"]["seed"] = seed
            tr["dataset"]["partition"]["workers_only"] = True
            tr["total_rounds"] = ROUNDS
            tr["momentum"] = 0.0
            tr["global_eval"] = True
            tr.update(BASELINE)
            tr.update(patch)
            path = arm_dir / f"seed{seed}.yaml"
            path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            try:
                v = load_config(str(path))
                assert v.training.total_rounds == ROUNDS
                assert v.training.lr_schedule == "cosine"
                assert v.training.dataset.partition.workers_only is True
            except Exception as e:  # noqa: BLE001
                print(f"INVALID {path}: {e}")
                return 1
            specs.append((0, arm, seed, str(path.relative_to(ROOT))))

    # Interleave arms so each 3-batch holds 3 DIFFERENT seeds of ONE arm
    # (same-seed different-arm in one batch is fine since the launcher
    # project-name fix, but same-arm batches keep pull/analyze simple and
    # pair mono earliest).
    specs.sort(key=lambda r: (r[1], r[2]))
    with (OUT / "specs.txt").open("w") as f:
        for tier, arm, seed, cfgpath in specs:
            f.write(f"concurrent\t{tier}\t{arm}\t{seed}\t{cfgpath}\n")
    print(f"OK: wrote + validated {len(specs)} h150 configs "
          f"(~{len(specs)/3 * 100:.0f} min @3-way, 150r/run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
