#!/usr/bin/env python3
"""Generate the CP2 attribution-control config matrix (2026-08-05).

The make-or-break reviewer controls (writeup/15 item 1-2, handover §5.1):

- ``uniform_eps03`` — importance-blind scheduling control (Claim-A gate).
  ``importance_metric_v2: uniform`` makes coverage-EFT's density ordering
  degenerate to 1/bytes (byte-greedy, "shed the big cheap layers"); the
  ε-trigger stays frozen on delta-sq-norm (G2), so the arm isolates the
  value of importance-aware *ordering* at matched coverage semantics.
- ``cyclic_k7`` — network-and-importance-blind rotation (FedPart-style),
  byte-matched to the coverage arm's fresh-aggregated head: measured mean
  head_bytes in w3/drop_eps03 = 356,577 B/worker-round (47.6% of 749,405 B);
  k=7 rotates 374,702 B/round on average = 1.05x the head (slightly generous
  to the control, i.e. conservative for us).  Claim-A selection gate.
- ``cyclic_k3`` / ``cyclic_k2`` — the naive periodic-refresh sweep
  (refresh period L/k = 14/k rounds ~ a blind tau_max analogue) to place
  against the tau_max frontier on the bytes<->accuracy plane.  Claim-B gate.
- ``sentinel_mono`` / ``sentinel_drop_eps03`` — exact re-runs of w3 arms at
  seed 41 to verify bit-determinism of the reprovisioned box + rebuilt
  fl-node image (TF pin is loose).  If they diverge from w3, all CP2
  comparisons switch to internally re-run baselines.

Everything else mirrors wave-3: 20 rounds, CIFAR-10/deep_cnn/Dirichlet(0.1),
lr 0.1, momentum 0, global_eval (K0).  Deliberately the SAME (weak) baseline
as w3 so the controls pair cleanly against existing arms; the competent-
baseline campaign is a separate axis.

Emits:  configs/experiments/cp2/<arm>/seed<seed>.yaml
        configs/experiments/cp2/specs.txt   (mode<TAB>tier<TAB>arm<TAB>seed<TAB>cfgpath)
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
OUT = ROOT / "configs/experiments/cp2"
SEEDS6 = [41, 42, 43, 44, 45, 46]
SEEDS3 = [41, 42, 43]

# arm: (tier, base, seeds, training-patch).  All concurrent (accuracy-only).
# Tier 0 runs first: the two determinism sentinels + the first uniform seed.
ARMS: dict[str, tuple] = {
    "sentinel_mono":        (0, MONO_BASE, [41], {}),
    "sentinel_drop_eps03":  (0, PL_BASE, [41], {"late_layer_policy": "drop"}),
    "uniform_eps03":        (1, PL_BASE, SEEDS6,
                             {"importance_metric_v2": "uniform",
                              "late_layer_policy": "drop"}),
    "cyclic_k7":            (2, PL_BASE, SEEDS6,
                             {"assignment_strategy": "cyclic", "cyclic_k": 7,
                              "late_layer_policy": "drop"}),
    "cyclic_k3":            (3, PL_BASE, SEEDS3,
                             {"assignment_strategy": "cyclic", "cyclic_k": 3,
                              "late_layer_policy": "drop"}),
    "cyclic_k2":            (3, PL_BASE, SEEDS3,
                             {"assignment_strategy": "cyclic", "cyclic_k": 2,
                              "late_layer_policy": "drop"}),
}


def build(base_path: Path, seed: int, patch: dict) -> dict:
    cfg = yaml.safe_load(base_path.read_text())
    tr = cfg["training"]
    tr["dataset"]["partition"]["seed"] = seed
    tr["total_rounds"] = 20
    tr["momentum"] = 0.0
    tr["global_eval"] = True
    tr.update(patch)
    return cfg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs: list[tuple] = []
    for arm, (tier, base, seeds, patch) in ARMS.items():
        arm_dir = OUT / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            cfg = build(base, seed, patch)
            path = arm_dir / f"seed{seed}.yaml"
            path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            try:
                v = load_config(str(path))
                assert v.training.global_eval is True
                assert v.training.total_rounds == 20
                if patch.get("assignment_strategy") == "cyclic":
                    assert v.training.cyclic_k == patch["cyclic_k"]
                if "importance_metric_v2" in patch:
                    assert v.training.importance_metric_v2 == patch["importance_metric_v2"]
            except Exception as e:  # noqa: BLE001
                print(f"INVALID {path}: {e}")
                return 1
            specs.append((tier, arm, seed, str(path.relative_to(ROOT))))

    specs.sort(key=lambda r: (r[0], r[1], r[2]))
    specs_txt = OUT / "specs.txt"
    with specs_txt.open("w") as f:
        for tier, arm, seed, cfgpath in specs:
            f.write(f"concurrent\t{tier}\t{arm}\t{seed}\t{cfgpath}\n")

    print(f"OK: wrote + validated {len(specs)} configs across {len(ARMS)} arms")
    print(f"  specs -> {specs_txt.relative_to(ROOT)}")
    print(f"  rough estimate: ~{(len(specs) / 3) * 11:.0f} min @3-way")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
