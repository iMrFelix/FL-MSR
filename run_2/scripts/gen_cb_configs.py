#!/usr/bin/env python3
"""Generate the competent-baseline (CB) campaign (2026-08-05).

The forensics verdict (writeup/16): the referee-grade threat is the weak
baseline — no clipping, no schedule, no momentum, 29% of the data stranded
on the aggregator.  This campaign re-runs the core triptych contrast under
the fixed baseline to answer: does the mechanism story (recycle > drop,
bounded ~few-pp cost vs mono) survive a fair baseline?

Fixed-baseline bundle (every arm):
  clipnorm=1.0, server_momentum=0.9 (FedAvgM), lr_schedule=cosine,
  dataset.partition.workers_only=true (workers jointly train on 100%).

Arms: mono / drop_eps03 / recycle_eps03, at
  - 20 rounds x seeds 41-46 (n=6, pairs with the wave-3 design), and
  - 50 rounds x seeds 41-43 (horizon + collapse-rate-under-clipping).

Emits: configs/experiments/cb/<arm>_r<rounds>/seed<seed>.yaml + specs.txt
(specs tier 0 = 20r arms first: they carry the headline).
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
OUT = ROOT / "configs/experiments/cb"

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

HORIZONS = [  # (rounds, seeds, tier)
    (20, [41, 42, 43, 44, 45, 46], 0),
    (50, [41, 42, 43], 1),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = []
    for rounds, seeds, tier in HORIZONS:
        for arm, (base, patch) in ARMS.items():
            arm_dir = OUT / f"{arm}_r{rounds}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                cfg = yaml.safe_load(base.read_text())
                tr = cfg["training"]
                tr["dataset"]["partition"]["seed"] = seed
                tr["dataset"]["partition"]["workers_only"] = True
                tr["total_rounds"] = rounds
                tr["momentum"] = 0.0
                tr["global_eval"] = True
                tr.update(BASELINE)
                tr.update(patch)
                path = arm_dir / f"seed{seed}.yaml"
                path.write_text(
                    yaml.dump(cfg, default_flow_style=False, sort_keys=False)
                )
                try:
                    v = load_config(str(path))
                    assert v.training.clipnorm == 1.0
                    assert v.training.server_momentum == 0.9
                    assert v.training.lr_schedule == "cosine"
                    assert v.training.dataset.partition.workers_only is True
                    assert v.training.total_rounds == rounds
                except Exception as e:  # noqa: BLE001
                    print(f"INVALID {path}: {e}")
                    return 1
                specs.append(
                    (tier, f"{arm}_r{rounds}", seed,
                     str(path.relative_to(ROOT)))
                )

    specs.sort(key=lambda r: (r[0], r[1], r[2]))
    with (OUT / "specs.txt").open("w") as f:
        for tier, arm, seed, cfgpath in specs:
            f.write(f"concurrent\t{tier}\t{arm}\t{seed}\t{cfgpath}\n")
    n20 = sum(1 for s in specs if s[1].endswith("_r20"))
    n50 = len(specs) - n20
    print(f"OK: wrote + validated {len(specs)} configs "
          f"({n20} x 20r, {n50} x 50r)")
    print(f"  rough estimate: ~{(n20/3)*11 + (n50/3)*30:.0f} min @3-way")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
