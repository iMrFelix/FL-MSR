"""Generate the C1B isolation campaign: the two MISSING slippage×aging cells.

Campaign C1 (scripts/gen_campaign_c1.py) already supplies two of the four
cells of the slippage (recycle vs drop) × aging (on vs off) 2x2 at ε=0.49,
the largest-effect point — at the SAME code state and seeds:

    a4_ref_eps0p49      = recycle + aging   (additive_capped, τ=3)
    a5_dropctl_eps0p49  = drop    + no-aging

so the a4-vs-a5 contrast moves two knobs at once and cannot separate the
slippage main effect from the aging main effect (dossier limitation #2;
see writeup/05-campaign-c1.md §(e) and the 2026-06-11 dossier correction).
This campaign emits ONLY the two missing cells so that, combined with C1's
a4/a5, the full 2x2 yields the slippage main effect, the aging main effect,
and their interaction:

    b1_recycle_noage_eps0p49 = recycle + no-aging
    b2_drop_age_eps0p49      = drop    + aging   (additive_capped, τ=3)

Everything else (ε=0.49, coverage_eft assignment, delta_sq_norm importance,
50 rounds, seeds 42-46, the 6/3/1 Mbps DeepCNN/CIFAR-10 regime) is held fixed
and matches C1 exactly, so the four cells are directly comparable per seed.

Emits config YAMLs + run-spec JSONs consumed by the validated
scripts/overnight_runner.py (idempotent, hard per-run timeout, docker
cleanup, status.jsonl) — no new run machinery. Mirrors gen_campaign_c1.py's
template handling and robust config_path resolution.

Usage:
  python -m scripts.gen_campaign_c1b --validate         # 10 configs schema-load
  python -m scripts.gen_campaign_c1b --queue-dir results/campaign_c1b/queue \
      --configs-dir configs/experiments/campaign_c1b \
      --output-root results/campaign_c1b/runs
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PER_LAYER_TEMPLATE = REPO / "configs/experiments/prelim/prelim_perlayer_eps0.yaml"

ROUNDS = 50
SEEDS = [42, 43, 44, 45, 46]
TIMEOUT_S = 2400  # ~17 min expected per run at ~20 s/round; generous ceiling

# (tag, base, knob-overrides for the training section).  Both arms are
# per-layer at ε=0.49 with the coverage-EFT scheduler and delta-sq-norm
# importance; the ONLY two knobs that vary across the 2x2 (jointly with C1's
# a4/a5) are late_layer_policy (slippage) and aging_mode (aging).
ARMS = [
    ("b1_recycle_noage_eps0p49", "per_layer", {
        "update_mode": "per_layer", "epsilon_deadline": 0.49,
        "assignment_strategy": "coverage_eft", "importance_metric_v2": "delta_sq_norm",
        "late_layer_policy": "recycle_last_delta",
        "aging_mode": "none",
    }),
    ("b2_drop_age_eps0p49", "per_layer", {
        "update_mode": "per_layer", "epsilon_deadline": 0.49,
        "assignment_strategy": "coverage_eft", "importance_metric_v2": "delta_sq_norm",
        "late_layer_policy": "drop",
        "aging_mode": "additive_capped", "aging_lambda": 0.5, "aging_tau_max": 3,
    }),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-dir", default="results/campaign_c1b/queue")
    ap.add_argument("--configs-dir", default="configs/experiments/campaign_c1b")
    ap.add_argument("--output-root", default="results/campaign_c1b/runs")
    ap.add_argument("--validate", action="store_true",
                    help="Load each emitted config through the schema before writing the spec.")
    args = ap.parse_args()

    queue = REPO / args.queue_dir
    cfgs = REPO / args.configs_dir
    queue.mkdir(parents=True, exist_ok=True)
    cfgs.mkdir(parents=True, exist_ok=True)

    templates = {
        "per_layer": yaml.safe_load(PER_LAYER_TEMPLATE.read_text()),
    }

    n = 0
    for tag, base, knobs in ARMS:
        for seed in SEEDS:
            cfg = copy.deepcopy(templates[base])
            t = cfg["training"]
            t["total_rounds"] = ROUNDS
            t["dataset"]["partition"]["seed"] = seed
            t.update(knobs)
            run_id = f"{tag}_s{seed}"
            cpath = cfgs / f"{run_id}.yaml"
            cpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
            if args.validate:
                from src.config.schema import load_config
                load_config(str(cpath))  # raises on any invalid arm
            try:
                cfg_ref = str(cpath.relative_to(REPO))
            except ValueError:
                cfg_ref = str(cpath)
            spec = {
                "run_id": run_id, "stage": "C1B",
                "config_path": cfg_ref,
                "output_dir": f"{args.output_root}/{run_id}",
                "timeout_s": TIMEOUT_S, "seed": seed,
            }
            (queue / f"{run_id}.json").write_text(json.dumps(spec, indent=1))
            n += 1
    print(f"wrote {n} runs ({len(ARMS)} arms x {len(SEEDS)} seeds, {ROUNDS} rounds) "
          f"to {queue}")


if __name__ == "__main__":
    main()
