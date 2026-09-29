#!/usr/bin/env python3
"""Generate the 50-round horizon campaign (2026-07-23 pre-meeting sprint, wave 2).

Tests the #1 open question raised by the wave-3 result (writeup/14 §New data):
do the 20-round accuracy gaps (drop/recycle/recycle+aging all ~-4pp vs mono)
wash out at convergence, or are they a real standing cost?

Same 4 core arms as the wave-3 triptych + mono anchor, IDENTICAL seeds 41-46,
global_eval (K0), momentum 0, epsilon 0.3 — the ONLY change vs wave-3 is
total_rounds 20 -> 50, so the 20r vs 50r contrast is clean (same seeds, same
val standard). All arms concurrent (accuracy-only; wire timing was already
certified serially in wave-3). Per-round val_accuracy in report.json gives the
full gap-vs-horizon trajectory, not just the endpoint.
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
OUT = ROOT / "configs/experiments/horizon50"
SEEDS = [41, 42, 43, 44, 45, 46]
ROUNDS = 50
AGING = {"aging_mode": "additive_capped", "aging_lambda": 0.5, "aging_tau_max": 3}

# (base template, training-patch) — same knobs as the wave-3 triptych.
ARMS: dict[str, tuple] = {
    "mono":                (MONO_BASE, {}),
    "drop_eps03":          (PL_BASE, {"late_layer_policy": "drop"}),
    "recycle_eps03":       (PL_BASE, {"late_layer_policy": "recycle_last_delta"}),
    "recycle_aging_eps03": (PL_BASE, {"late_layer_policy": "recycle_last_delta", **AGING}),
}


def build(base: Path, seed: int, patch: dict) -> dict:
    cfg = yaml.safe_load(base.read_text())
    tr = cfg["training"]
    tr["dataset"]["partition"]["seed"] = seed
    tr["total_rounds"] = ROUNDS
    tr["momentum"] = 0.0
    tr["global_eval"] = True          # K0 — certify on the shared held-out set
    tr.update(patch)
    return cfg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs: list[tuple] = []
    for arm, (base, patch) in ARMS.items():
        (OUT / arm).mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            cfg = build(base, seed, patch)
            path = OUT / arm / f"seed{seed}.yaml"
            path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            try:
                v = load_config(str(path))
                assert v.training.total_rounds == ROUNDS
                assert v.training.global_eval is True
            except Exception as e:  # noqa: BLE001
                print(f"INVALID {path}: {e}")
                return 1
            specs.append((arm, seed, str(path.relative_to(ROOT))))

    specs.sort(key=lambda r: (r[0], r[1]))
    (OUT / "specs.txt").write_text(
        "".join(f"concurrent\t1\t{a}\t{s}\t{c}\n" for a, s, c in specs)
    )
    print(f"OK: wrote + validated {len(specs)} configs "
          f"({len(ARMS)} arms x {len(SEEDS)} seeds x {ROUNDS}r)")
    print(f"  specs -> {(OUT / 'specs.txt').relative_to(ROOT)}")
    print(f"  rough estimate @6-way: ~{(len(specs) / 6) * 33 / 60:.1f}h "
          f"(50r 6-way batch ~33min, {len(specs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
