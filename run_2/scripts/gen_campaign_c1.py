"""Generate the C1 server campaign: 50-round, 5-seed CIFAR-10/DeepCNN runs.

Closes dossier limitations #1 (10-round horizon) and #2 (2-3 seeds) by
re-running the reference configuration and its controls at a convergence
horizon with enough seeds for the <2pp deltas. Emits config YAMLs + run-spec
JSONs consumed by the validated scripts/overnight_runner.py (idempotent,
hard per-run timeout, docker cleanup, status.jsonl) — no new run machinery.

Arms (reference config = dossier §4: eps=0.2, coverage_eft, delta_sq_norm,
recycle_last_delta, additive_capped aging tau_max=3):
  a1 mono            monolithic baseline
  a2 perlayer_eps0   bug-detector / integrity (must match mono per seed)
  a3 ref_eps0p2      recommended operating point
  a4 ref_eps0p49     below-monolithic-floor point
  a5 dropctl_eps0p49 slippage isolation (recycle vs drop, matched eps)
  a6 cyclic_eps0p49  attribution control (byte-matched k=7, importance-blind)

Usage:
  python -m scripts.gen_campaign_c1 --queue-dir results/campaign_c1/queue \
      --configs-dir configs/experiments/campaign_c1 \
      --output-root results/campaign_c1/runs
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PER_LAYER_TEMPLATE = REPO / "configs/experiments/prelim/prelim_perlayer_eps0.yaml"
MONO_TEMPLATE = REPO / "configs/experiments/prelim/prelim_mono_10mbps.yaml"

ROUNDS = 50
SEEDS = [42, 43, 44, 45, 46]
TIMEOUT_S = 2400  # ~17 min expected per run at ~20 s/round; generous ceiling

# (tag, base, knob-overrides for the training section)
ARMS = [
    ("a1_mono", "monolithic", {"update_mode": "monolithic"}),
    ("a2_perlayer_eps0", "per_layer", {
        "update_mode": "per_layer", "epsilon_deadline": 0.0,
        "assignment_strategy": "coverage_eft", "importance_metric_v2": "delta_sq_norm",
    }),
    ("a3_ref_eps0p2", "per_layer", {
        "update_mode": "per_layer", "epsilon_deadline": 0.2,
        "assignment_strategy": "coverage_eft", "importance_metric_v2": "delta_sq_norm",
        "late_layer_policy": "recycle_last_delta",
        "aging_mode": "additive_capped", "aging_lambda": 0.5, "aging_tau_max": 3,
    }),
    ("a4_ref_eps0p49", "per_layer", {
        "update_mode": "per_layer", "epsilon_deadline": 0.49,
        "assignment_strategy": "coverage_eft", "importance_metric_v2": "delta_sq_norm",
        "late_layer_policy": "recycle_last_delta",
        "aging_mode": "additive_capped", "aging_lambda": 0.5, "aging_tau_max": 3,
    }),
    ("a5_dropctl_eps0p49", "per_layer", {
        "update_mode": "per_layer", "epsilon_deadline": 0.49,
        "assignment_strategy": "coverage_eft", "importance_metric_v2": "delta_sq_norm",
        "late_layer_policy": "drop",
    }),
    ("a6_cyclic_eps0p49", "per_layer", {
        "update_mode": "per_layer", "epsilon_deadline": 0.49,
        "assignment_strategy": "cyclic", "importance_metric_v2": "delta_sq_norm",
        "late_layer_policy": "drop", "cyclic_k": 7,
    }),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-dir", default="results/campaign_c1/queue")
    ap.add_argument("--configs-dir", default="configs/experiments/campaign_c1")
    ap.add_argument("--output-root", default="results/campaign_c1/runs")
    ap.add_argument("--validate", action="store_true",
                    help="Load each emitted config through the schema before writing the spec.")
    args = ap.parse_args()

    queue = REPO / args.queue_dir
    cfgs = REPO / args.configs_dir
    queue.mkdir(parents=True, exist_ok=True)
    cfgs.mkdir(parents=True, exist_ok=True)

    templates = {
        "per_layer": yaml.safe_load(PER_LAYER_TEMPLATE.read_text()),
        "monolithic": yaml.safe_load(MONO_TEMPLATE.read_text()),
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
                "run_id": run_id, "stage": "C1",
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
