"""Generate the Phase-2 FEMNIST campaign: parameterized spec+config emitter.

Phase 2 moves the per-layer ε-trigger evaluation onto the FedLUAR FEMNIST
task (LEAF natural per-writer partition + the 4-layer ``femnist_cnn``,
~25 MiB/update, single-dominant-layer regime) at a convergence horizon.
This generator emits, for a chosen STAGE, both the runner spec JSON files
(scripts/overnight_runner.py contract) and the ExperimentConfig YAMLs they
reference, and validates EVERY emitted YAML through ``src.config.load_config``
at generation time — a typo'd knob fails here, not at 3 a.m. in a container
(same fail-loud discipline as scripts/gen_campaign_c1b.py).

It mirrors gen_campaign_c1.py / gen_campaign_c1b.py: configs are derived from
the FEMNIST template (configs/experiments/phase1/t3_femnist_smoke.yaml) so the
topology, traffic classes, and FedLUAR-matched hyper-parameters stay fixed;
only the ``training`` knobs under evaluation (and, per seed, the partition
seed) change. The per-class bandwidth grid and its monolithic single-class
sum are parameters so the byte-matched mono-vs-per-layer comparison can be
re-pinned without editing code.

----------------------------------------------------------------------------
Reference bundle (the recommended per-layer operating configuration):
  update_mode=per_layer, assignment_strategy=coverage_eft,
  importance_metric_v2=delta_sq_norm, late_layer_policy=recycle_last_delta,
  aging_mode=additive_capped, aging_lambda=0.5, aging_tau_max=3,
  epsilon_warmup_rounds=0, watchdog_factor=3.0.
Dataset: name=femnist, partition.strategy=natural, partition.max_writers=ARG,
  partition.seed=SEED.
FedLUAR-matched training (decisions.md D-T3.3 / DEF-7): epochs_per_round=1,
  batch_size=20, learning_rate=0.01, optimizer=sgd, momentum=0.9 (pinned
  explicitly so the "FedLUAR-matched" claim is provenance-true; the schema
  ignores the extra field and the SGD optimizer is byte-identical, but the
  realized value is recorded in the config that ran).

Arm matrix (select any subset with --arms):
  a1_mono            update_mode=monolithic (byte-matched single-class floor).
  a2_eps0            reference bundle at ε=0 (integrity / bug-detector arm;
                     skip_feedback MUST stay off — DEF-6).
  a3_ref_mid         reference bundle at ε=eps_mid (recommended operating pt).
  a4_ref_high        reference bundle at ε=eps_high.
  a5_drop_high       reference but late_layer_policy=drop at ε=eps_high.
  a6_bytebal         byte_balanced assignment, recycle, aging on, at
                     ε=bytebal_eps (byte-match calibration arm).
  a7_cyclic          cyclic assignment (cyclic_k=ARG), drop, aging none, at
                     ε=eps_high (network-blind attribution control).
  a8_noage           reference but aging_mode=none at ε=eps_high.
  a9_skip_shed       reference + skip_feedback=shed at ε=eps_mid.
  a10_fedluar        skip_feedback=fedluar (skip_fedluar_count=delta), aging
                     off, ε=0 (FedLUAR-native baseline).
  b1_recycle_noage   reference but aging none at ε=eps_high  (2x2 cell).
  b2_drop_age        reference but drop      at ε=eps_high  (2x2 cell).

STAGE semantics:
  P0  Tiny calibration set: a1_mono + a per-layer coverage_eft arm
      (a3_ref_mid), at SHORT --rounds (3-5), 1 seed. Purpose: measure
      round wall-clock and the per-class network share before committing
      the long horizon (protocol §6 P0 outputs; DEF-5).
  P1  Knee-selection manifests: {coverage_eft@ε=0, byte_balanced@ε=0, mono},
      3 rounds, 5 seeds. ε=0 so the per-layer arms emit full manifests for
      the ε-knee selection without coverage shedding.
  P2  Wire checks + byte-match calibration: EACH P3 arm at 3 rounds, 1 seed.
  P3  The selected --arms at --rounds (>=50), --seeds. The confirmatory run.

Usage:
  python -m scripts.gen_phase2_femnist --stage P3 \
      --arms a1_mono a2_eps0 a3_ref_mid a4_ref_high a7_cyclic \
      --rounds 50 --eps-mid 0.2 --eps-high 0.49 --seeds 42 43 44 45 46 \
      --max-writers 20 --cyclic-k 4 --fedluar-delta 2 --bytebal-eps 0.49 \
      --out-queue results/phase2/queue --out-configs configs/experiments/phase2
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]

#: FEMNIST per-writer template: femnist_cnn + natural partition + the
#: 3-class 60/30/10 topology. Everything topology/hyperparam is taken from
#: here; only ``training`` knobs (and bandwidths) are overridden per arm.
FEMNIST_TEMPLATE = REPO / "configs/experiments/phase1/t3_femnist_smoke.yaml"

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_TIMEOUT_S = 5400  # 50-round femnist_cnn (~25 MiB/update); generous ceiling
DEFAULT_BW_MBPS = [60, 30, 10]  # per-class fast/mid/slow (template default)
DEFAULT_MONO_MBPS = 100          # = sum(DEFAULT_BW_MBPS): byte-matched mono

STAGES = ("P0", "P1", "P2", "P3")

#: DSCP code points for the 3-class per-layer regime (template values).
_DSCP_3CLASS = {0: 46, 1: 10, 2: 0}  # EF / AF11 / best-effort


# ---------------------------------------------------------------------------
# Reference bundle + arm matrix
# ---------------------------------------------------------------------------

def _reference_bundle() -> dict[str, Any]:
    """The per-layer reference knobs shared by every per-layer arm.

    epsilon_deadline is intentionally NOT set here — each arm pins its own ε.
    skip_feedback defaults off (schema default) and is only turned on by the
    skip arms; the integrity arm (a2_eps0) therefore keeps it off by
    construction (DEF-6).
    """
    return {
        "update_mode": "per_layer",
        "assignment_strategy": "coverage_eft",
        "importance_metric_v2": "delta_sq_norm",
        "late_layer_policy": "recycle_last_delta",
        "aging_mode": "additive_capped",
        "aging_lambda": 0.5,
        "aging_tau_max": 3,
        "epsilon_warmup_rounds": 0,
        "watchdog_factor": 3.0,
    }


def arm_knobs(arm_id: str, params: "Params") -> dict[str, Any]:
    """Return the ``training`` knob overrides for one arm id.

    The returned dict is layered on top of the template's training section
    (which already carries the FedLUAR-matched epochs/batch/lr/optimizer);
    every arm additionally pins momentum=0.9 so the FedLUAR-matched claim is
    provenance-true (DEF-7). For monolithic the reference bundle is not
    applied (it is a single-blob baseline). ``ref`` arms start from a fresh
    copy of the reference bundle so per-arm edits never leak across arms.
    """
    ref = _reference_bundle

    if arm_id == "a1_mono":
        # Single-blob baseline; per-layer knobs are irrelevant in this mode.
        return {"update_mode": "monolithic"}

    if arm_id == "a2_eps0":
        k = ref()
        k["epsilon_deadline"] = 0.0
        # skip_feedback stays off (DEF-6: ε=0 bug-detector must not skip).
        return k

    if arm_id == "a3_ref_mid":
        k = ref()
        k["epsilon_deadline"] = params.eps_mid
        return k

    if arm_id == "a4_ref_high":
        k = ref()
        k["epsilon_deadline"] = params.eps_high
        return k

    if arm_id == "a5_drop_high":
        k = ref()
        k["epsilon_deadline"] = params.eps_high
        k["late_layer_policy"] = "drop"
        return k

    if arm_id == "a6_bytebal":
        k = ref()
        k["assignment_strategy"] = "byte_balanced"
        # recycle + aging are already the reference defaults.
        k["epsilon_deadline"] = params.bytebal_eps
        return k

    if arm_id == "a7_cyclic":
        k = ref()
        k["assignment_strategy"] = "cyclic"
        k["cyclic_k"] = params.cyclic_k
        k["late_layer_policy"] = "drop"
        k["aging_mode"] = "none"
        k["aging_lambda"] = 0.0
        k["aging_tau_max"] = 0
        k["epsilon_deadline"] = params.eps_high
        return k

    if arm_id == "a8_noage":
        k = ref()
        k["aging_mode"] = "none"
        k["aging_lambda"] = 0.0
        k["aging_tau_max"] = 0
        k["epsilon_deadline"] = params.eps_high
        return k

    if arm_id == "a9_skip_shed":
        k = ref()
        k["skip_feedback"] = "shed"
        k["epsilon_deadline"] = params.eps_mid
        return k

    if arm_id == "a10_fedluar":
        k = ref()
        k["skip_feedback"] = "fedluar"
        k["skip_fedluar_count"] = params.fedluar_delta
        k["aging_mode"] = "none"
        k["aging_lambda"] = 0.0
        k["aging_tau_max"] = 0
        k["epsilon_deadline"] = 0.0
        return k

    if arm_id == "b1_recycle_noage":
        k = ref()
        k["aging_mode"] = "none"
        k["aging_lambda"] = 0.0
        k["aging_tau_max"] = 0
        k["epsilon_deadline"] = params.eps_high
        return k

    if arm_id == "b2_drop_age":
        k = ref()
        k["late_layer_policy"] = "drop"
        k["epsilon_deadline"] = params.eps_high
        return k

    raise ValueError(f"unknown arm id {arm_id!r}; valid arms: {sorted(ALL_ARMS)}")


#: Canonical arm-id set (used for --arms validation and the P2 wire-check
#: superset). Ordered so run_ids sort the controls/baselines first.
ALL_ARMS = (
    "a1_mono", "a2_eps0", "a3_ref_mid", "a4_ref_high", "a5_drop_high",
    "a6_bytebal", "a7_cyclic", "a8_noage", "a9_skip_shed", "a10_fedluar",
    "b1_recycle_noage", "b2_drop_age",
)


class Params:
    """Bag of campaign-wide numeric parameters threaded into arm_knobs."""

    def __init__(
        self,
        eps_mid: float,
        eps_high: float,
        bytebal_eps: float,
        cyclic_k: int,
        fedluar_delta: int,
    ) -> None:
        self.eps_mid = eps_mid
        self.eps_high = eps_high
        self.bytebal_eps = bytebal_eps
        self.cyclic_k = cyclic_k
        self.fedluar_delta = fedluar_delta


# ---------------------------------------------------------------------------
# Stage -> (arm list, rounds, seeds) resolution
# ---------------------------------------------------------------------------

def stage_plan(
    stage: str,
    selected_arms: list[str],
    rounds: int,
    seeds: list[int],
) -> tuple[list[str], int, list[int]]:
    """Resolve the (arms, rounds, seeds) actually emitted for ``stage``.

    P0/P1/P2 have fixed arm/seed/round structure per the protocol; only P3
    honours the user's --arms/--rounds/--seeds directly. P0/P1 still take
    --rounds (P0: short calibration rounds; P1: 3) and --seeds (P1: all)
    from the CLI so the operator controls the budget, with documented
    fallbacks.
    """
    if stage == "P0":
        # Calibration: mono + one per-layer coverage_eft arm, short, 1 seed.
        arms = ["a1_mono", "a3_ref_mid"]
        p0_rounds = rounds if rounds and rounds <= 10 else 3
        return arms, p0_rounds, seeds[:1]

    if stage == "P1":
        # Knee-selection manifests: coverage_eft@ε0, byte_balanced@ε0, mono.
        # ε=0 is forced via dedicated arm ids below (see build_run_id/knobs
        # override in emit); here we name them by their P1 roles.
        arms = ["a1_mono", "p1_coveft_eps0", "p1_bytebal_eps0"]
        return arms, 3, list(seeds)

    if stage == "P2":
        # Wire checks + byte-match calibration: every P3 arm, 3 rounds, 1 seed.
        arms = list(ALL_ARMS)
        return arms, 3, seeds[:1]

    # P3: confirmatory.
    if not selected_arms:
        raise ValueError("stage P3 requires a non-empty --arms list")
    if rounds < 50:
        raise ValueError(
            f"stage P3 is the confirmatory horizon; --rounds must be >= 50 "
            f"(got {rounds}). Use P0/P1/P2 for short pilots."
        )
    return list(selected_arms), rounds, list(seeds)


def p1_arm_knobs(arm_id: str) -> dict[str, Any] | None:
    """Knobs for the P1-only synthetic arm ids (ε=0 manifest emitters)."""
    if arm_id == "p1_coveft_eps0":
        k = _reference_bundle()
        k["epsilon_deadline"] = 0.0
        return k
    if arm_id == "p1_bytebal_eps0":
        k = _reference_bundle()
        k["assignment_strategy"] = "byte_balanced"
        k["epsilon_deadline"] = 0.0
        return k
    return None


# ---------------------------------------------------------------------------
# Config materialization
# ---------------------------------------------------------------------------

def _apply_bandwidths(cfg: dict[str, Any], per_class: list[int], mono_mbps: int,
                      monolithic: bool) -> None:
    """Rewrite the topology bandwidths and traffic-class count in place.

    Per-layer arms keep the 3-class grid (per_class fast/mid/slow). The
    monolithic arm collapses to a single fast class at ``mono_mbps`` (=
    sum(per_class)), so the single shipped blob drains over the same total
    bandwidth — the byte-matched floor, mirroring prelim_mono vs prelim_perlayer.
    """
    edges = cfg["federation"]["topology"]["edges"]
    if monolithic:
        for edge in edges:
            edge["classes"] = {
                0: {"bandwidth_mbps": mono_mbps, "latency_ms": 5, "drop_rate": 0.0}
            }
        cfg["traffic_classes"]["num_classes"] = 1
        cfg["traffic_classes"]["dscp_mapping"] = {0: 46}
    else:
        for edge in edges:
            classes = edge["classes"]
            for ci, bw in enumerate(per_class):
                # Preserve any per-class latency/drop already on the edge.
                cls = dict(classes.get(ci, {"latency_ms": 5, "drop_rate": 0.0}))
                cls["bandwidth_mbps"] = bw
                classes[ci] = cls
        cfg["traffic_classes"]["num_classes"] = len(per_class)
        cfg["traffic_classes"]["dscp_mapping"] = {
            ci: _DSCP_3CLASS.get(ci, 0) for ci in range(len(per_class))
        }


def build_config(
    template: dict[str, Any],
    arm_id: str,
    knobs: dict[str, Any],
    rounds: int,
    seed: int,
    max_writers: int,
    per_class_bw: list[int],
    mono_mbps: int,
    model: str = "femnist_cnn_distributed",
) -> dict[str, Any]:
    """Materialize one ExperimentConfig dict (not yet validated/written).

    Starts from the FEMNIST template and overrides only: total_rounds, the
    partition (strategy=natural, max_writers, seed=SEED), the per-arm training
    knobs, the FedLUAR momentum pin, and the bandwidth grid. Everything else
    (femnist_cnn model, FedLUAR epochs/batch/lr/optimizer, latencies) is
    inherited from the template so arms stay byte-comparable.
    """
    cfg = copy.deepcopy(template)
    t = cfg["training"]

    t["total_rounds"] = rounds
    t["model"] = model   # distributed-byte FEMNIST CNN (option-B redesign, 2026-06-18)

    # Dataset: natural per-writer partition, seeded, capped writers.
    t["dataset"]["name"] = "femnist"
    part = t["dataset"]["partition"]
    part["strategy"] = "natural"
    part["max_writers"] = max_writers
    part["seed"] = seed

    monolithic = knobs.get("update_mode") == "monolithic"

    # FedLUAR-matched hyper-parameters (pin even though inherited, so a
    # template drift cannot silently de-match the claim). momentum=0.9 is the
    # DEF-7 provenance pin (schema ignores it; SGD is byte-identical).
    t["epochs_per_round"] = 1
    t["batch_size"] = 20
    # lr raised 0.01 -> 0.1 to compensate for the schema-dropped momentum=0.9
    # (DEF-7, 2026-06-18): momentum ~ a 10x effective-lr multiplier, so lr-only
    # SGD at 0.01 under-trains (FEMNIST stuck ~0.05); at 0.1 it learns (~0.69 @
    # 20 rounds in P0). Applied to ALL arms, so the relative comparison holds.
    t["learning_rate"] = 0.1
    t["optimizer"] = "sgd"
    t["momentum"] = 0.9

    # Apply the arm knobs last so they win over any template default.
    t.update(knobs)

    # Per-seed: the partition seed IS the training/node seed in this schema
    # (the engine seeds the writer subsample + node RNG off it, as in
    # gen_campaign_c1.py). No separate node-seed field exists; the spec's
    # top-level "seed" carries it for the runner/analysis side.

    _apply_bandwidths(cfg, per_class_bw, mono_mbps, monolithic)

    # Reports land in <output_dir>/results/ — the runner's completion marker.
    cfg.setdefault("monitoring", {})["report_output"] = "./results/"

    return cfg


def _config_header(stage: str, run_id: str, arm_id: str, eps: Any) -> str:
    return "\n".join([
        f"# Generated by scripts/gen_phase2_femnist.py — run {run_id}",
        f"# Phase-2 FEMNIST, stage {stage}, arm {arm_id}, epsilon_deadline={eps}",
        "# Do not hand-edit; regenerate (generation is deterministic).",
        "",
    ])


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def emit(
    stage: str,
    arms: list[str],
    rounds: int,
    seeds: list[int],
    params: Params,
    max_writers: int,
    per_class_bw: list[int],
    mono_mbps: int,
    timeout_s: int,
    out_queue: Path,
    out_configs: Path,
    model: str = "femnist_cnn_distributed",
) -> int:
    """Write configs + specs for the stage; validate every config. Returns n."""
    template = yaml.safe_load(FEMNIST_TEMPLATE.read_text())

    out_queue.mkdir(parents=True, exist_ok=True)
    out_configs.mkdir(parents=True, exist_ok=True)

    # Import the loader lazily (same as gen_campaign_c1b) so --help works
    # without the src package importable. load_config lives in src.config.schema
    # (src.config is the package; the validating loader is the schema module).
    from src.config.schema import load_config

    n = 0
    for arm_id in arms:
        # P1 synthetic arm ids resolve through p1_arm_knobs; all other ids
        # (including the canonical arms used by P2/P3) through arm_knobs.
        p1k = p1_arm_knobs(arm_id)
        knobs = p1k if p1k is not None else arm_knobs(arm_id, params)

        for seed in seeds:
            run_id = f"{stage.lower()}_{arm_id}_s{seed}"
            cfg = build_config(
                template, arm_id, knobs, rounds, seed,
                max_writers, per_class_bw, mono_mbps, model,
            )
            cpath = out_configs / f"{run_id}.yaml"
            eps = cfg["training"].get("epsilon_deadline", "n/a")
            cpath.write_text(
                _config_header(stage, run_id, arm_id, eps)
                + yaml.safe_dump(cfg, sort_keys=False)
            )

            # Fail loud at generation time (like gen_campaign_c1b.py).
            load_config(str(cpath))  # raises pydantic.ValidationError on a bad arm

            try:
                cfg_ref = str(cpath.relative_to(REPO))
            except ValueError:
                cfg_ref = str(cpath)

            spec = {
                "run_id": run_id,
                "stage": stage,
                "config_path": cfg_ref,
                "output_dir": f"results/phase2/runs/{run_id}",
                "timeout_s": timeout_s,
                "seed": seed,
            }
            (out_queue / f"{run_id}.json").write_text(json.dumps(spec, indent=1))
            n += 1
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate Phase-2 FEMNIST run specs + configs (parameterized)."
    )
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--out-queue", required=True,
                    help="Directory for runner spec *.json files.")
    ap.add_argument("--out-configs", required=True,
                    help="Directory for the generated ExperimentConfig YAMLs.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                    help="Seeds (default 42 43 44 45 46). Varies "
                         "dataset.partition.seed and the spec seed.")
    ap.add_argument("--rounds", type=int, default=50,
                    help="Rounds for P3 (>=50); calibration rounds for P0.")
    ap.add_argument("--eps-mid", type=float, default=0.2,
                    help="Mid ε for a3_ref_mid / a9_skip_shed.")
    ap.add_argument("--eps-high", type=float, default=0.49,
                    help="High ε for a4/a5/a7/a8/b1/b2.")
    ap.add_argument("--bytebal-eps", type=float, default=0.49,
                    help="ε for the byte_balanced arm (a6_bytebal).")
    ap.add_argument("--bw-mbps", type=int, nargs=3, default=list(DEFAULT_BW_MBPS),
                    metavar=("FAST", "MID", "SLOW"),
                    help="Per-class bandwidths for per-layer arms (3 ints).")
    ap.add_argument("--mono-mbps", type=int, default=None,
                    help="Single-class bandwidth for a1_mono "
                         "(default: sum of --bw-mbps).")
    ap.add_argument("--max-writers", type=int, default=20,
                    help="FEMNIST seeded writer-subsample cap.")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="Arm ids for P3 (subset of the matrix). "
                         "Ignored by P0/P1/P2, which have fixed arm sets.")
    ap.add_argument("--cyclic-k", type=int, default=4,
                    help="Layers/round for the cyclic arm (a7_cyclic).")
    ap.add_argument("--fedluar-delta", type=int, default=2,
                    help="skip_fedluar_count for a10_fedluar.")
    ap.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S,
                    help="Hard per-run timeout written into every spec.")
    ap.add_argument("--model", type=str, default="femnist_cnn_distributed",
                    help="Model architecture (MODEL_REGISTRY key). Default is the "
                         "distributed-byte redesign; pass femnist_cnn for the "
                         "original single-dominant-layer reference.")
    args = ap.parse_args(argv)

    # --eps-high / --eps-mid feed P2/P3 arms; P2 needs them too (it emits the
    # full P3 arm set as wire checks), so validate them whenever any per-layer
    # ε-bearing arm could be emitted (all stages except a pure-mono case).
    for name, val in (("--eps-mid", args.eps_mid), ("--eps-high", args.eps_high),
                      ("--bytebal-eps", args.bytebal_eps)):
        if not (0.0 <= val < 1.0):
            ap.error(f"{name}={val} must satisfy 0 <= ε < 1 (schema bound).")

    if args.cyclic_k < 1:
        ap.error("--cyclic-k must be >= 1 (cyclic strategy transmits k layers/round).")
    if args.fedluar_delta < 1:
        ap.error("--fedluar-delta must be >= 1 (skip_feedback='fedluar' requires it).")

    bw = list(args.bw_mbps)
    mono_mbps = args.mono_mbps if args.mono_mbps is not None else sum(bw)
    if args.mono_mbps is not None and args.mono_mbps != sum(bw):
        print(
            f"[gen] NOTE: --mono-mbps={args.mono_mbps} != sum(--bw-mbps)={sum(bw)}; "
            "the mono baseline is no longer byte-matched to the per-layer total.",
            file=sys.stderr,
        )

    params = Params(
        eps_mid=args.eps_mid,
        eps_high=args.eps_high,
        bytebal_eps=args.bytebal_eps,
        cyclic_k=args.cyclic_k,
        fedluar_delta=args.fedluar_delta,
    )

    selected = args.arms or []
    if selected:
        unknown = [a for a in selected if a not in ALL_ARMS]
        if unknown:
            ap.error(f"unknown --arms {unknown}; valid: {list(ALL_ARMS)}")

    arms, rounds, seeds = stage_plan(args.stage, selected, args.rounds, args.seeds)

    out_queue = (Path(args.out_queue) if Path(args.out_queue).is_absolute()
                 else REPO / args.out_queue)
    out_configs = (Path(args.out_configs) if Path(args.out_configs).is_absolute()
                   else REPO / args.out_configs)

    n = emit(
        stage=args.stage,
        arms=arms,
        rounds=rounds,
        seeds=seeds,
        params=params,
        max_writers=args.max_writers,
        per_class_bw=bw,
        mono_mbps=mono_mbps,
        model=args.model,
        timeout_s=args.timeout_s,
        out_queue=out_queue,
        out_configs=out_configs,
    )
    print(
        f"stage {args.stage}: wrote {n} runs "
        f"({len(arms)} arm(s) x {len(seeds)} seed(s), {rounds} rounds) — "
        f"specs -> {out_queue}, configs -> {out_configs} "
        f"(all validated through src.config.load_config)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
