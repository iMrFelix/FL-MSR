#!/usr/bin/env python3
"""Generate the timing re-measurement campaign (audit NT-01/02/03/05, 2026-08-05).

Every wall-clock claim was retracted because sender-side timers measured
kernel-buffer acceptance, not wire time, and because the "wire floor" was a
fluid bound no whole-layer schedule can reach.  The fixed stack (f2461d7)
measures TRUE one-way wire times from the shared host CLOCK_MONOTONIC, and
gives monolithic + downlink flows receiver-side completion stamps for the
first time — so mono and per-layer are finally comparable in ONE clock domain.

Design:
- SERIAL only (FL_SUBNET_OCTET 0, one stack at a time).  The audit found the
  testbed compute-bound; 3-way co-tenancy would inject compute contention
  into exactly the quantity under measurement.
- mono anchor + the eps grid {0.0, 0.2, 0.3, 0.4} at 20 rounds, seeds 41-43.
  eps=0 is the granularity control (whole-layer optimum makespan); mono is
  the wire baseline that previously had no receiver clock at all.
- Legacy weak baseline ON PURPOSE (no clipnorm/momentum/schedule): these runs
  measure the NETWORK path, and keeping the optimizer identical to p1/w3
  makes the timing directly comparable with the historical campaigns.

Outputs per run (new instrumentation): layer_wire_times_oneway_s,
manifest_wire_time_oneway_s, t_cover_oneway_s, _receiver_flows completion
stamps, _clock_domains legend + oneway_negative_count (must be 0 — a nonzero
count disproves the shared-clock premise and invalidates the campaign).

Emits: configs/experiments/timing/<arm>/seed<seed>.yaml + specs.txt
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
OUT = ROOT / "configs/experiments/timing"
SEEDS = [41, 42, 43]

ARMS = {
    "mono": (MONO_BASE, {}),
    "eps0": (PL_BASE, {"epsilon_deadline": 0.0}),
    "eps02": (PL_BASE, {"epsilon_deadline": 0.2}),
    "eps03": (PL_BASE, {"epsilon_deadline": 0.3}),
    "eps04": (PL_BASE, {"epsilon_deadline": 0.4}),
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
            tr["total_rounds"] = 20
            tr["momentum"] = 0.0
            tr["global_eval"] = True
            tr["late_layer_policy"] = "drop"
            tr.update(patch)
            path = arm_dir / f"seed{seed}.yaml"
            path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            try:
                v = load_config(str(path))
                assert v.training.clipnorm is None
                assert v.training.lr_schedule == "constant"
            except Exception as e:  # noqa: BLE001
                print(f"INVALID {path}: {e}")
                return 1
            specs.append((arm, seed, str(path.relative_to(ROOT))))

    with (OUT / "specs.txt").open("w") as f:
        for arm, seed, cfgpath in specs:
            f.write(f"serial\t0\t{arm}\t{seed}\t{cfgpath}\n")
    print(f"OK: wrote + validated {len(specs)} timing configs "
          f"(SERIAL, ~{len(specs) * 8} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
