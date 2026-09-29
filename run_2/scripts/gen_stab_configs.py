#!/usr/bin/env python3
"""Generate the focused stability sweep (Felix's idea; handover §5.4).

Question: does shedding shift the stability boundary of unclipped constant-lr
FedAvg on tiny non-IID shards?  The forensics could only say 1/3 mono vs 1/8
shedding collapsed (Fisher p=0.49, no power).  This sweep measures collapse
RATE per cell: lr x arm x 6 seeds, DELIBERATELY under the legacy weak
baseline (no clipnorm, constant lr, stranded-shard partitioning) because the
instability itself is the measurand.  50 rounds (both observed collapses hit
by r45).

Cells: lr in {0.1, 0.15, 0.05} x arm in {mono, drop_eps03} x seeds 41-46.
Priority (tier) order: lr=0.1 first (replication + power at the operating
point), then lr=0.15 (pushes the boundary), then lr=0.05 (stable side).
36 runs x 50r ~= 6h @3-way; the runner is restartable, partial data usable.

Emits: configs/experiments/stab/<arm>_lr<tag>/seed<seed>.yaml + specs.txt
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
OUT = ROOT / "configs/experiments/stab"
SEEDS = [41, 42, 43, 44, 45, 46]

LRS = [(0.1, "010", 0), (0.15, "015", 1), (0.05, "005", 2)]
ARMS = {
    "mono": (MONO_BASE, {}),
    "drop_eps03": (PL_BASE, {"late_layer_policy": "drop"}),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = []
    for lr, tag, tier in LRS:
        for arm, (base, patch) in ARMS.items():
            arm_dir = OUT / f"{arm}_lr{tag}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                cfg = yaml.safe_load(base.read_text())
                tr = cfg["training"]
                tr["dataset"]["partition"]["seed"] = seed
                tr["total_rounds"] = 50
                tr["momentum"] = 0.0
                tr["global_eval"] = True
                tr["learning_rate"] = lr
                # legacy weak baseline ON PURPOSE: no clipnorm, constant
                # lr, default (stranded-shard) partitioning — the
                # instability is the measurand.
                tr.update(patch)
                path = arm_dir / f"seed{seed}.yaml"
                path.write_text(
                    yaml.dump(cfg, default_flow_style=False, sort_keys=False)
                )
                try:
                    v = load_config(str(path))
                    assert v.training.learning_rate == lr
                    assert v.training.clipnorm is None
                    assert v.training.lr_schedule == "constant"
                    assert v.training.dataset.partition.workers_only is False
                except Exception as e:  # noqa: BLE001
                    print(f"INVALID {path}: {e}")
                    return 1
                specs.append((tier, f"{arm}_lr{tag}", seed,
                              str(path.relative_to(ROOT))))

    specs.sort(key=lambda r: (r[0], r[1], r[2]))
    with (OUT / "specs.txt").open("w") as f:
        for tier, arm, seed, cfgpath in specs:
            f.write(f"concurrent\t{tier}\t{arm}\t{seed}\t{cfgpath}\n")
    print(f"OK: wrote + validated {len(specs)} stab configs "
          f"(~{len(specs)/3*0.5:.1f}h @3-way, 50r)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
