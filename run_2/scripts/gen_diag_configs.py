#!/usr/bin/env python3
"""Generate the collapse-diagnostic configs (writeup/16 §5 diagnostic 1).

The two collapsed 50-round runs — mono/seed41 (NaN at r39) and
recycle_aging_eps03/seed42 (7.7e18 at r44 -> NaN) — re-run UNCHANGED except
for exactly one knob per arm:

- diag_clip:   + clipnorm=1.0            (mechanism-sufficiency test: the
               forensics say exploding gradients under unclipped lr=0.1;
               if the collapse vanishes at fixed seed, confirmed)
- diag_lr005:  learning_rate 0.1 -> 0.05 (step-size sufficiency variant)

Everything else (seed, partition, arms, 50 rounds, global val) is byte-for-
byte the horizon50 configuration, so the comparison is a paired single-knob
contrast against the recorded collapse trajectories.

Emits: configs/experiments/diag/<arm>/seed<seed>.yaml + specs.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config.schema import load_config  # noqa: E402

H50 = ROOT / "configs/experiments/horizon50"
OUT = ROOT / "configs/experiments/diag"

# (arm, base config, seed, training patch)
RUNS = [
    ("clip_mono",   H50 / "mono/seed41.yaml",               41, {"clipnorm": 1.0}),
    ("clip_aging",  H50 / "recycle_aging_eps03/seed42.yaml", 42, {"clipnorm": 1.0}),
    ("lr005_mono",  H50 / "mono/seed41.yaml",               41, {"learning_rate": 0.05}),
    ("lr005_aging", H50 / "recycle_aging_eps03/seed42.yaml", 42, {"learning_rate": 0.05}),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = []
    for arm, base, seed, patch in RUNS:
        cfg = yaml.safe_load(base.read_text())
        assert cfg["training"]["dataset"]["partition"]["seed"] == seed
        assert cfg["training"]["total_rounds"] == 50
        cfg["training"].update(patch)
        arm_dir = OUT / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        path = arm_dir / f"seed{seed}.yaml"
        path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        try:
            v = load_config(str(path))
            assert v.training.total_rounds == 50
        except Exception as e:  # noqa: BLE001
            print(f"INVALID {path}: {e}")
            return 1
        specs.append((arm, seed, str(path.relative_to(ROOT))))

    with (OUT / "specs.txt").open("w") as f:
        for arm, seed, cfgpath in specs:
            f.write(f"concurrent\t0\t{arm}\t{seed}\t{cfgpath}\n")
    print(f"OK: wrote + validated {len(specs)} diagnostic configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
