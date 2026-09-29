#!/usr/bin/env python3
"""Generate cp2fix — corrected cyclic control arms (audit findings TRIG-4/BYTE-07).

The original CP2 cyclic arms left epsilon_deadline=0.3 active, so the arm shed
TWICE (rotation omission + eps-trigger drop on the transmitted subset): the
aggregated fresh bytes were ~30% below the byte-matched design point and the
kappa/coverage semantics were incomparable with the treatment arm.

Fix: identical arms with epsilon_deadline=0.0 — the trigger then requires all
k announced layers (pure rotation, single shedding mechanism), and cyclic_k7
becomes the true byte-matched blind-selection control (374,702 B/round
rotated AND aggregated ~= 1.05x the drop_eps03 head).

Emits: configs/experiments/cp2fix/<arm>/seed<seed>.yaml + specs.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config.schema import load_config  # noqa: E402

PL_BASE = ROOT / "configs/experiments/phase1b/exp1_grid/coverage_eft_eps03/seed41.yaml"
OUT = ROOT / "configs/experiments/cp2fix"

ARMS = {
    "cyclic_k7_eps0": (7, [41, 42, 43, 44, 45, 46]),
    "cyclic_k3_eps0": (3, [41, 42, 43]),
    "cyclic_k2_eps0": (2, [41, 42, 43]),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = []
    for arm, (k, seeds) in ARMS.items():
        arm_dir = OUT / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            cfg = yaml.safe_load(PL_BASE.read_text())
            tr = cfg["training"]
            tr["dataset"]["partition"]["seed"] = seed
            tr["total_rounds"] = 20
            tr["momentum"] = 0.0
            tr["global_eval"] = True
            tr["assignment_strategy"] = "cyclic"
            tr["cyclic_k"] = k
            tr["epsilon_deadline"] = 0.0
            tr["late_layer_policy"] = "drop"
            path = arm_dir / f"seed{seed}.yaml"
            path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            try:
                v = load_config(str(path))
                assert v.training.epsilon_deadline == 0.0
                assert v.training.cyclic_k == k
            except Exception as e:  # noqa: BLE001
                print(f"INVALID {path}: {e}")
                return 1
            specs.append((0, arm, seed, str(path.relative_to(ROOT))))

    specs.sort(key=lambda r: (r[1], r[2]))
    with (OUT / "specs.txt").open("w") as f:
        for tier, arm, seed, cfgpath in specs:
            f.write(f"concurrent\t{tier}\t{arm}\t{seed}\t{cfgpath}\n")
    print(f"OK: wrote + validated {len(specs)} cp2fix configs (~45 min @3-way)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
