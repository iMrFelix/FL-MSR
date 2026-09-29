#!/usr/bin/env python3
"""Generate the wave-3 config matrix (2026-07-23 pre-meeting sprint).

Every arm runs with ``global_eval: true`` (K0) so accuracy is certified on the
shared 10k CIFAR-10 held-out set instead of the skewed per-node local val
(kills the ~4pp cross-seed noise; writeup/12 §3).  All arms are 20-round
CIFAR-10 / deep_cnn / Dirichlet(0.1), momentum 0, seeds 41-46, built off the
committed phase-1b templates so the only deltas are the knobs under test.

Two run modes (writeup/12 §5): ``serial`` arms are timing-KPI arms and get the
whole box (no co-tenancy); ``concurrent`` arms are accuracy-only and run 3-way
(FL_SUBNET_OCTET 9/10/11).  specs.txt is emitted in priority-tier order so a
short night still lands the headline (aging n=6 + triptych) first.

Emits:  configs/experiments/wave3/<arm>/seed<seed>.yaml
        configs/experiments/wave3/specs.txt   (mode<TAB>tier<TAB>arm<TAB>seed<TAB>cfgpath)

Each emitted file is validated through load_config() before it is kept.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config.schema import load_config  # noqa: E402

PL_BASE = ROOT / "configs/experiments/phase1b/exp1_grid/coverage_eft_eps03/seed41.yaml"
MONO_BASE = ROOT / "configs/experiments/phase1b/exp4_monolithic/mono/seed41.yaml"
OUT = ROOT / "configs/experiments/wave3"
SEEDS = [41, 42, 43, 44, 45, 46]      # full power: triptych + mono anchor
SEEDS3 = [41, 42, 43]                  # thinner: frontier trend + cut-candidates

# Reference bundles (writeup/12 §5; aging λ/τ from the phase-2 reference bundle).
AGING = {"aging_mode": "additive_capped", "aging_lambda": 0.5, "aging_tau_max": 3}

# Each arm: (tier, mode, base, seeds, training-patch).  Lower tier = higher
# priority.  The runner executes serial (timing) arms first within tier 0, then
# concurrent arms tier by tier.
ARMS: dict[str, tuple] = {
    # --- tier 0: timing anchors (SERIAL) ---
    # mono re-anchor: paired-ΔACC baseline under global val + wire-floor timing.
    "mono": (0, "serial", MONO_BASE, SEEDS, {}),
    # W3-E C4 control: whole-layer optimum makespan (== coverage_eft == 0.787s).
    "byte_balanced_eps0": (0, "serial", PL_BASE, [41, 42, 43],
                           {"epsilon_deadline": 0.0, "assignment_strategy": "byte_balanced"}),

    # --- tier 1: the slippage triptych under global val (CONCURRENT) ---
    # recycle+aging is the headline: -1.76pp n=2 -> powered n=6, certified.
    "recycle_aging_eps03": (1, "concurrent", PL_BASE, SEEDS,
                            {"late_layer_policy": "recycle_last_delta", **AGING}),
    "drop_eps03":    (1, "concurrent", PL_BASE, SEEDS, {"late_layer_policy": "drop"}),
    "recycle_eps03": (1, "concurrent", PL_BASE, SEEDS,
                      {"late_layer_policy": "recycle_last_delta"}),

    # --- tier 2: refresh-cost frontier W3-B (CONCURRENT, 3 seeds = trend) ---
    # skip=shed genuinely removes bytes; aging τ_max caps the staleness ratchet.
    "frontier_tau3": (2, "concurrent", PL_BASE, SEEDS3,
                      {"skip_feedback": "shed", "late_layer_policy": "recycle_last_delta",
                       "aging_mode": "additive_capped", "aging_lambda": 0.5, "aging_tau_max": 3}),
    "frontier_tau5": (2, "concurrent", PL_BASE, SEEDS3,
                      {"skip_feedback": "shed", "late_layer_policy": "recycle_last_delta",
                       "aging_mode": "additive_capped", "aging_lambda": 0.5, "aging_tau_max": 5}),
    "frontier_tau2": (2, "concurrent", PL_BASE, SEEDS3,
                      {"skip_feedback": "shed", "late_layer_policy": "recycle_last_delta",
                       "aging_mode": "additive_capped", "aging_lambda": 0.5, "aging_tau_max": 2}),
    "frontier_tau8": (2, "concurrent", PL_BASE, SEEDS3,
                      {"skip_feedback": "shed", "late_layer_policy": "recycle_last_delta",
                       "aging_mode": "additive_capped", "aging_lambda": 0.5, "aging_tau_max": 8}),
    # aging-off: the unbounded starvation ratchet the cap fixes (κ exhibit).
    "frontier_agingoff": (2, "concurrent", PL_BASE, SEEDS3,
                          {"skip_feedback": "shed", "late_layer_policy": "recycle_last_delta",
                           "aging_mode": "none"}),

    # --- tier 3: slippage completion W3-D (CONCURRENT, cut candidate, 3 seeds) ---
    "renormalize_eps03": (3, "concurrent", PL_BASE, SEEDS3,
                          {"late_layer_policy": "renormalize"}),
}


def build(base_path: Path, seed: int, patch: dict) -> dict:
    cfg = yaml.safe_load(base_path.read_text())
    tr = cfg["training"]
    tr["dataset"]["partition"]["seed"] = seed
    tr["total_rounds"] = 20
    tr["momentum"] = 0.0
    tr["global_eval"] = True          # K0 — certify on the shared held-out set
    tr.update(patch)
    return cfg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs: list[tuple] = []
    n_ok = 0
    for arm, (tier, mode, base, seeds, patch) in ARMS.items():
        arm_dir = OUT / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            cfg = build(base, seed, patch)
            path = arm_dir / f"seed{seed}.yaml"
            path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            # Validate: pydantic rejects any bad knob/enum here, not on moltres.
            try:
                v = load_config(str(path))
                assert v.training.global_eval is True
                assert v.training.total_rounds == 20
            except Exception as e:  # noqa: BLE001
                print(f"INVALID {path}: {e}")
                return 1
            specs.append((tier, mode, arm, seed, str(path.relative_to(ROOT))))
            n_ok += 1

    # specs.txt ordered: tier asc, then serial before concurrent, then arm, seed.
    specs.sort(key=lambda r: (r[0], 0 if r[1] == "serial" else 1, r[2], r[3]))
    specs_txt = OUT / "specs.txt"
    with specs_txt.open("w") as f:
        for tier, mode, arm, seed, cfgpath in specs:
            f.write(f"{mode}\t{tier}\t{arm}\t{seed}\t{cfgpath}\n")

    n_serial = sum(1 for r in specs if r[1] == "serial")
    n_conc = len(specs) - n_serial
    print(f"OK: wrote + validated {n_ok} configs across {len(ARMS)} arms")
    print(f"  serial (timing):     {n_serial} runs")
    print(f"  concurrent (accuracy): {n_conc} runs")
    print(f"  specs -> {specs_txt.relative_to(ROOT)}")
    # Rough wall-clock: serial ~7-10min each; concurrent ~11min per 3-way batch.
    est = n_serial * 8 + (n_conc / 3) * 11
    print(f"  rough estimate: ~{est/60:.1f}h "
          f"(serial {n_serial*8}min + concurrent {(n_conc/3)*11:.0f}min @3x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
