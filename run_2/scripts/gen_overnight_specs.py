"""Generate run-spec JSON files + experiment configs for the overnight queue.

Implements the adopted run matrix of ``writeup/01-candidate-selection.md`` §6:

- **Stage A** (ε=0, 15 runs, 3 rounds): 4 metrics x coverage-EFT x 2 repeats,
  byte-balanced x2, gap-based native raw-norm x2, monolithic x3.  Fixed —
  no parameters beyond the seed.
- **Stage B** (ε>0, 25 runs at 3 ε values, 3 rounds): slippage x ε x repeats
  (18) + starvation controls at ε-high (4) + runner-up metric spot-check (2)
  + network-blind cyclic control (1).  Parameterized by the ε list chosen by
  ``scripts/eps_knee_selection.py`` and the Stage-A winners.
- **Stage C** (10 rounds, 16 runs): monolithic x2 seeds, ε=0 bug-detector x2,
  best triplet at ε-low x3 and ε-high x3, drop-control at ε-high x2, warm-up
  ε-schedule x2, cyclic x2.  Parameterized by the Stage-B winning triplet.

Cut-line ordering: within each stage the run_ids of first-cut arms sort
LAST (Stage B: cyclic, then spot-check; Stage C: cyclic), so the supervisor
cuts by deleting spec files from the end of the queue.

Configs are derived from the prelim templates
(``configs/experiments/prelim/prelim_perlayer_eps0.yaml`` for per-layer
arms, ``prelim_mono_10mbps.yaml`` for monolithic) so the topology, traffic
classes, and hyper-parameters stay byte-comparable with the preliminary
experiments; only the ``training`` knobs under evaluation change.  Every
generated config is validated through the pydantic schema before writing —
a typo'd knob must fail at generation time, not at 3 a.m. inside a
container.

Generation is deterministic (no timestamps, sorted keys), so re-running
with identical arguments is a byte-level no-op; differing content for an
existing file aborts unless ``--force`` (protects the provenance of configs
whose runs already completed).

Usage::

    python -m scripts.gen_overnight_specs --stage A
    python -m scripts.gen_overnight_specs --stage B \
        --eps 0.0625 0.2 0.45 --slippage drop recycle renorm \
        --winner-metric delta_sq_norm --runner-up-metric relative
    python -m scripts.gen_overnight_specs --stage C \
        --eps-low 0.0625 --eps-high 0.45 \
        --winner-metric delta_sq_norm --winner-assignment coverage_eft \
        --winner-slippage recycle_last_delta
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config.schema import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

PER_LAYER_TEMPLATE = REPO_ROOT / "configs/experiments/prelim/prelim_perlayer_eps0.yaml"
MONO_TEMPLATE = REPO_ROOT / "configs/experiments/prelim/prelim_mono_10mbps.yaml"

DEFAULT_QUEUE_DIR = "results/overnight/queue"
DEFAULT_CONFIGS_DIR = "configs/experiments/overnight"
DEFAULT_OUTPUT_ROOT = "results/overnight/runs"

#: Screening runs are ~5 min (3 rounds); accuracy runs ~13-15 min (10
#: rounds).  Timeouts give ~4-5x margin, covering the docker image (re)build
#: on the first run of a stage.
DEFAULT_TIMEOUT_S = {"A": 1500, "B": 1500, "C": 3600}

SCREENING_ROUNDS = 3
ACCURACY_ROUNDS = 10
DEFAULT_SEED = 42

#: Admitted metrics, gate §7 — delta_sq_norm is the new primary; raw_norm is
#: the degenerate control.  Order fixes run_id numbering, so do not reorder.
STAGE_A_METRICS = ("delta_sq_norm", "relative", "snr_reweight", "raw_norm")

#: CLI conveniences for the slippage flag (canonical names = config literals).
SLIPPAGE_ALIASES = {
    "drop": "drop",
    "recycle": "recycle_last_delta",
    "recycle_last_delta": "recycle_last_delta",
    "renorm": "renormalize",
    "renormalize": "renormalize",
}

_METRIC_SHORT = {
    "delta_sq_norm": "delta",
    "relative": "rel",
    "snr_reweight": "snr",
    "raw_norm": "raw",
}
_SLIPPAGE_SHORT = {
    "drop": "drop",
    "recycle_last_delta": "recycle",
    "renormalize": "renorm",
}


@dataclass(frozen=True)
class Arm:
    """One run of the matrix: everything needed to emit config + spec."""

    run_id: str
    stage: str
    group: str
    mode: str  # "per_layer" | "monolithic"
    rounds: int
    seed: int = DEFAULT_SEED
    metric: str = "delta_sq_norm"
    assignment: str = "coverage_eft"
    slippage: str = "drop"
    epsilon: float = 0.0
    aging_mode: str = "none"
    aging_lambda: float = 0.0
    aging_tau_max: int = 0
    cyclic_k: int = 0
    epsilon_warmup_rounds: int = 0
    watchdog_factor: float = 3.0
    note: str = ""


def _eps_tag(eps: float) -> str:
    """0.0625 -> 'e0p0625' (filesystem/run_id-safe, unambiguous)."""
    return "e" + f"{eps:g}".replace(".", "p").replace("-", "m")


# ---------------------------------------------------------------------------
# Stage matrices
# ---------------------------------------------------------------------------

def stage_a_arms(seed: int = DEFAULT_SEED) -> list[Arm]:
    """The fixed 15-run Stage-A matrix (ε=0; gate §6).

    Repeats are same-seed by design: Stage A measures the run-to-run noise
    floor of the KPI, not seed variance.
    """
    arms: list[Arm] = []
    seq = 0
    for metric in STAGE_A_METRICS:
        seq += 1
        for rep in (1, 2):
            arms.append(
                Arm(
                    run_id=f"a{seq:02d}_coveft_{metric}_r{rep}",
                    stage="A",
                    group=f"a{seq:02d}_coveft_{metric}",
                    mode="per_layer",
                    rounds=SCREENING_ROUNDS,
                    seed=seed,
                    metric=metric,
                    assignment="coverage_eft",
                    note="coverage-EFT under the fixed receiver-side KPI",
                )
            )
    seq += 1
    for rep in (1, 2):
        arms.append(
            Arm(
                run_id=f"a{seq:02d}_bytebal_delta_sq_norm_r{rep}",
                stage="A",
                group=f"a{seq:02d}_bytebal_delta_sq_norm",
                mode="per_layer",
                rounds=SCREENING_ROUNDS,
                seed=seed,
                metric="delta_sq_norm",
                assignment="byte_balanced",
                note="makespan-optimal control (ε=0 optimum)",
            )
        )
    seq += 1
    for rep in (1, 2):
        arms.append(
            Arm(
                run_id=f"a{seq:02d}_gapbased_raw_norm_r{rep}",
                stage="A",
                group=f"a{seq:02d}_gapbased_raw_norm",
                mode="per_layer",
                rounds=SCREENING_ROUNDS,
                seed=seed,
                metric="raw_norm",
                assignment="gap_based",
                note="naive baseline in its native raw-norm config, "
                "re-measured under the fixed KPI (E2 re-test)",
            )
        )
    seq += 1
    for rep in (1, 2, 3):
        arms.append(
            Arm(
                run_id=f"a{seq:02d}_mono_r{rep}",
                stage="A",
                group=f"a{seq:02d}_mono",
                mode="monolithic",
                rounds=SCREENING_ROUNDS,
                seed=seed,
                note="monolithic floor reference (10 Mbps single class)",
            )
        )
    return arms


def stage_b_arms(
    eps_values: list[float],
    winner_metric: str = "delta_sq_norm",
    winner_assignment: str = "coverage_eft",
    slippage_policies: list[str] | None = None,
    runner_up_metric: str = "relative",
    repeats: int = 2,
    aging_lambda: float = 0.5,
    aging_tau_max: int = 2,
    cyclic_k: int = 5,
    seed: int = DEFAULT_SEED,
) -> list[Arm]:
    """Stage-B matrix: slippage screening at the chosen ε values (gate §6).

    Composition (with the default 3 ε x 3 slippage x 2 repeats = 25 runs):
    18 slippage x ε + 4 starvation controls at ε-high + 2 runner-up metric
    spot-checks + 1 cyclic control.  Cut lines sort last: cyclic, then the
    spot-checks.
    """
    if not eps_values:
        raise ValueError("Stage B needs at least one ε value")
    if any(not (0.0 < e < 1.0) for e in eps_values):
        raise ValueError(f"Stage-B ε values must be in (0, 1); got {eps_values}")
    slippage_policies = [
        SLIPPAGE_ALIASES[s] for s in (slippage_policies or ["drop", "recycle", "renorm"])
    ]
    eps_sorted = sorted(eps_values)
    eps_low, eps_high = eps_sorted[0], eps_sorted[-1]

    arms: list[Arm] = []
    seq = 0
    # 1) slippage x ε x repeats — the core RQ2 screen at the winning
    #    (metric x assignment) from Stage A.
    for eps in eps_sorted:
        for slip in slippage_policies:
            seq += 1
            group = f"b{seq:02d}_slip_{_SLIPPAGE_SHORT[slip]}_{_eps_tag(eps)}"
            for rep in range(1, repeats + 1):
                arms.append(
                    Arm(
                        run_id=f"{group}_r{rep}",
                        stage="B",
                        group=group,
                        mode="per_layer",
                        rounds=SCREENING_ROUNDS,
                        seed=seed,
                        metric=winner_metric,
                        assignment=winner_assignment,
                        slippage=slip,
                        epsilon=eps,
                        note="slippage x ε screen (winner metric/assignment)",
                    )
                )
    # 2) starvation controls at ε-high: additive-capped aging and the
    #    FedLUAR-style stochastic tail (gate §7), slippage pinned to the
    #    drop control so the aging effect is isolated.
    for aging_mode in ("additive_capped", "stochastic_tail"):
        seq += 1
        short = "ageadd" if aging_mode == "additive_capped" else "agestoch"
        group = f"b{seq:02d}_{short}_{_eps_tag(eps_high)}"
        for rep in (1, 2):
            arms.append(
                Arm(
                    run_id=f"{group}_r{rep}",
                    stage="B",
                    group=group,
                    mode="per_layer",
                    rounds=SCREENING_ROUNDS,
                    seed=seed,
                    metric=winner_metric,
                    assignment=winner_assignment,
                    slippage="drop",
                    epsilon=eps_high,
                    aging_mode=aging_mode,
                    aging_lambda=(
                        aging_lambda if aging_mode == "additive_capped" else 0.0
                    ),
                    aging_tau_max=(
                        aging_tau_max if aging_mode == "additive_capped" else 0
                    ),
                    note="starvation control at ε-high",
                )
            )
    # 3) runner-up metric spot-check at ε-low and ε-high (cut line 2).
    seq += 1
    spot_eps = (eps_low,) if eps_low == eps_high else (eps_low, eps_high)
    for eps in spot_eps:
        group = f"b{seq:02d}_spot_{_METRIC_SHORT.get(runner_up_metric, runner_up_metric)}"
        arms.append(
            Arm(
                run_id=f"{group}_{_eps_tag(eps)}_r1",
                stage="B",
                group=f"{group}_{_eps_tag(eps)}",
                mode="per_layer",
                rounds=SCREENING_ROUNDS,
                seed=seed,
                metric=runner_up_metric,
                assignment=winner_assignment,
                slippage="drop",
                epsilon=eps,
                note="runner-up metric spot-check (first cut after cyclic)",
            )
        )
    # 4) network-blind cyclic attribution control (cut line 1 -> sorts last).
    seq += 1
    group = f"b{seq:02d}_cyclic_{_eps_tag(eps_high)}"
    arms.append(
        Arm(
            run_id=f"{group}_r1",
            stage="B",
            group=group,
            mode="per_layer",
            rounds=SCREENING_ROUNDS,
            seed=seed,
            metric=winner_metric,
            assignment="cyclic",
            slippage="drop",
            epsilon=eps_high,
            cyclic_k=cyclic_k,
            note="network-blind FedPart-style control (first cut)",
        )
    )
    return arms


def stage_c_arms(
    eps_low: float,
    eps_high: float,
    winner_metric: str = "delta_sq_norm",
    winner_assignment: str = "coverage_eft",
    winner_slippage: str = "recycle_last_delta",
    seeds: list[int] | None = None,
    warmup_rounds: int = 3,
    cyclic_k: int = 5,
) -> list[Arm]:
    """Stage-C accuracy matrix (10 rounds, accuracy = veto KPI; gate §6).

    ``seeds`` supplies the multi-seed axis: x2 arms use ``seeds[:2]``, the
    best-triplet x3 arms use ``seeds[:3]``.
    """
    if not (0.0 < eps_low < 1.0) or not (0.0 < eps_high < 1.0):
        raise ValueError("Stage-C ε values must be in (0, 1)")
    winner_slippage = SLIPPAGE_ALIASES[winner_slippage]
    seeds = seeds or [42, 43, 44]
    if len(seeds) < 3:
        raise ValueError("Stage C needs >= 3 seeds (best-triplet arms run x3)")
    s2, s3 = seeds[:2], seeds[:3]

    def _arm(seq: int, tag: str, seed: int, **kw: Any) -> Arm:
        group = f"c{seq:02d}_{tag}"
        return Arm(
            run_id=f"{group}_s{seed}",
            stage="C",
            group=group,
            rounds=ACCURACY_ROUNDS,
            seed=seed,
            **kw,
        )

    arms: list[Arm] = []
    for seed in s2:  # monolithic accuracy reference
        arms.append(_arm(1, "mono", seed, mode="monolithic",
                         note="monolithic accuracy reference"))
    for seed in s2:  # ε=0 bug detector: divergence here = implementation bug
        arms.append(_arm(2, "eps0bug", seed, mode="per_layer",
                         metric=winner_metric, assignment=winner_assignment,
                         epsilon=0.0,
                         note="ε=0 bug detector (identical in exact arithmetic)"))
    for seed in s3:  # best triplet at ε-low
        arms.append(_arm(3, f"best_{_eps_tag(eps_low)}", seed, mode="per_layer",
                         metric=winner_metric, assignment=winner_assignment,
                         slippage=winner_slippage, epsilon=eps_low,
                         note="best triplet at ε-low"))
    for seed in s3:  # best triplet at ε-high
        arms.append(_arm(4, f"best_{_eps_tag(eps_high)}", seed, mode="per_layer",
                         metric=winner_metric, assignment=winner_assignment,
                         slippage=winner_slippage, epsilon=eps_high,
                         note="best triplet at ε-high"))
    for seed in s2:  # drop-control at ε-high (isolates the slippage effect)
        arms.append(_arm(5, f"dropctl_{_eps_tag(eps_high)}", seed,
                         mode="per_layer", metric=winner_metric,
                         assignment=winner_assignment, slippage="drop",
                         epsilon=eps_high,
                         note="drop control at ε-high"))
    for seed in s2:  # round-indexed warm-up ε schedule
        arms.append(_arm(6, f"warmup_{_eps_tag(eps_high)}", seed,
                         mode="per_layer", metric=winner_metric,
                         assignment=winner_assignment,
                         slippage=winner_slippage, epsilon=eps_high,
                         epsilon_warmup_rounds=warmup_rounds,
                         note="warm-up ε schedule (ε=0 during critical "
                              "learning period)"))
    for seed in s2:  # cyclic accuracy control (first cut -> sorts last)
        arms.append(_arm(7, f"cyclic_{_eps_tag(eps_high)}", seed,
                         mode="per_layer", metric=winner_metric,
                         assignment="cyclic", slippage="drop",
                         epsilon=eps_high, cyclic_k=cyclic_k,
                         note="network-blind cyclic control (first cut)"))
    return arms


# ---------------------------------------------------------------------------
# Config + spec emission
# ---------------------------------------------------------------------------

def _load_template(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_config(arm: Arm, templates: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Materialize the experiment config dict for one arm.

    Starts from the matching prelim template and overrides only the knobs
    under evaluation, so everything else (topology, 6/3/1 vs 10 Mbps
    classes, LR, batch size) stays identical to the preliminary runs.
    """
    if templates is None:
        templates = {
            "per_layer": _load_template(PER_LAYER_TEMPLATE),
            "monolithic": _load_template(MONO_TEMPLATE),
        }
    cfg = copy.deepcopy(templates[arm.mode])
    training = cfg["training"]
    training["total_rounds"] = arm.rounds
    training["dataset"]["partition"]["seed"] = arm.seed
    training["watchdog_factor"] = arm.watchdog_factor

    if arm.mode == "per_layer":
        training["update_mode"] = "per_layer"
        # Legacy metric field stays on the template value (gradient_norm):
        # the engine consumes importance_metric_v2 for both trigger and
        # sched scores, but the legacy field still gates per-layer manifest
        # emission, so it must remain non-null.
        training["importance_metric_v2"] = arm.metric
        training["assignment_strategy"] = arm.assignment
        training["epsilon_deadline"] = arm.epsilon
        training["late_layer_policy"] = arm.slippage
        training["aging_mode"] = arm.aging_mode
        training["aging_lambda"] = arm.aging_lambda
        training["aging_tau_max"] = arm.aging_tau_max
        training["cyclic_k"] = arm.cyclic_k
        training["epsilon_warmup_rounds"] = arm.epsilon_warmup_rounds

    # Reports land in <output_dir>/results/ — the runner's completion marker.
    cfg.setdefault("monitoring", {})["report_output"] = "./results/"

    # Fail at generation time, not inside a container at 3 a.m.
    ExperimentConfig.model_validate(cfg)
    return cfg


def build_spec(
    arm: Arm,
    config_path: str,
    output_dir: str,
    timeout_s: float,
) -> dict[str, Any]:
    """The runner-facing spec dict (extra keys feed the analysis scripts)."""
    return {
        "run_id": arm.run_id,
        "stage": arm.stage,
        "config_path": config_path,
        "output_dir": output_dir,
        "timeout_s": timeout_s,
        "seed": arm.seed,
        "group": arm.group,
        "arm": {
            "mode": arm.mode,
            "rounds": arm.rounds,
            "metric": arm.metric,
            "assignment": arm.assignment,
            "slippage": arm.slippage,
            "epsilon": arm.epsilon,
            "aging_mode": arm.aging_mode,
            "aging_lambda": arm.aging_lambda,
            "aging_tau_max": arm.aging_tau_max,
            "cyclic_k": arm.cyclic_k,
            "epsilon_warmup_rounds": arm.epsilon_warmup_rounds,
            "watchdog_factor": arm.watchdog_factor,
            "note": arm.note,
        },
    }


def _config_header(arm: Arm) -> str:
    lines = [
        f"# Generated by scripts/gen_overnight_specs.py — run {arm.run_id}",
        f"# Stage {arm.stage}: {arm.note or arm.group}",
        f"# arm: mode={arm.mode} metric={arm.metric} "
        f"assignment={arm.assignment} slippage={arm.slippage} "
        f"epsilon={arm.epsilon:g} rounds={arm.rounds} seed={arm.seed}",
        "# Matrix: writeup/01-candidate-selection.md §6. Do not hand-edit;",
        "# regenerate instead (generation is deterministic).",
        "",
    ]
    return "\n".join(lines)


def _write_if_unchanged(path: Path, content: str, force: bool) -> bool:
    """Write ``content``; refuse to silently change an existing file.

    Returns True when the file was (re)written, False when identical content
    already existed.  Raises on differing content without ``force`` — a
    config whose run already completed must never drift from what ran.
    """
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return False
        if not force:
            raise FileExistsError(
                f"{path} exists with different content; re-run with --force "
                "if the regeneration is intentional"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def write_stage(
    arms: list[Arm],
    queue_dir: Path,
    configs_dir: Path,
    output_root: Path,
    timeout_s: float,
    repo_root: Path = REPO_ROOT,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Emit configs + specs for one stage; returns the spec dicts.

    Paths inside specs are stored repo-root-relative when possible so the
    queue stays valid if the repo moves between generation and execution.
    """
    templates = {
        "per_layer": _load_template(PER_LAYER_TEMPLATE),
        "monolithic": _load_template(MONO_TEMPLATE),
    }
    stage_dir = configs_dir / f"stage_{arms[0].stage.lower()}" if arms else configs_dir

    def _rel(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return str(path)

    specs: list[dict[str, Any]] = []
    for arm in arms:
        cfg = build_config(arm, templates)
        cfg_path = stage_dir / f"{arm.run_id}.yaml"
        cfg_text = _config_header(arm) + yaml.dump(
            cfg, sort_keys=True, default_flow_style=False
        )
        _write_if_unchanged(cfg_path, cfg_text, force)

        spec = build_spec(
            arm,
            config_path=_rel(cfg_path),
            output_dir=_rel(output_root / arm.run_id),
            timeout_s=timeout_s,
        )
        spec_text = json.dumps(spec, sort_keys=True, indent=2) + "\n"
        _write_if_unchanged(queue_dir / f"{arm.run_id}.json", spec_text, force)
        specs.append(spec)
    return specs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate overnight run specs + configs (gate doc §6 matrix)."
    )
    parser.add_argument("--stage", required=True, choices=["A", "B", "C"])
    parser.add_argument("--queue-dir", default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--configs-dir", default=DEFAULT_CONFIGS_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Hard per-run timeout written into the specs "
        f"(defaults: {DEFAULT_TIMEOUT_S}).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Screening seed (stages A/B).")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting existing files with different "
                        "content.")
    # Stage B knobs
    parser.add_argument("--eps", type=float, nargs="+", default=None,
                        help="Stage-B ε values (output of eps_knee_selection).")
    parser.add_argument("--slippage", nargs="+", default=None,
                        help="Slippage policies (aliases ok: drop recycle renorm).")
    parser.add_argument("--winner-metric", default="delta_sq_norm",
                        choices=sorted(_METRIC_SHORT))
    parser.add_argument("--winner-assignment", default="coverage_eft")
    parser.add_argument("--runner-up-metric", default="relative",
                        choices=sorted(_METRIC_SHORT))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--aging-lambda", type=float, default=0.5)
    parser.add_argument("--aging-tau-max", type=int, default=2)
    parser.add_argument("--cyclic-k", type=int, default=5)
    # Stage C knobs
    parser.add_argument("--eps-low", type=float, default=None)
    parser.add_argument("--eps-high", type=float, default=None)
    parser.add_argument("--winner-slippage", default="recycle_last_delta")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the matrix without writing files.")
    args = parser.parse_args(argv)

    if args.stage == "A":
        arms = stage_a_arms(seed=args.seed)
    elif args.stage == "B":
        if not args.eps:
            parser.error("--eps is required for stage B "
                         "(use the eps_knee_selection output)")
        arms = stage_b_arms(
            eps_values=args.eps,
            winner_metric=args.winner_metric,
            winner_assignment=args.winner_assignment,
            slippage_policies=args.slippage,
            runner_up_metric=args.runner_up_metric,
            repeats=args.repeats,
            aging_lambda=args.aging_lambda,
            aging_tau_max=args.aging_tau_max,
            cyclic_k=args.cyclic_k,
            seed=args.seed,
        )
    else:
        if args.eps_low is None or args.eps_high is None:
            parser.error("--eps-low and --eps-high are required for stage C")
        if args.winner_slippage == "drop":
            print("WARNING: winner slippage is 'drop' — the drop-control arm "
                  "duplicates the best-triplet arm; consider deleting the "
                  "dropctl specs.")
        arms = stage_c_arms(
            eps_low=args.eps_low,
            eps_high=args.eps_high,
            winner_metric=args.winner_metric,
            winner_assignment=args.winner_assignment,
            winner_slippage=args.winner_slippage,
            seeds=args.seeds,
            warmup_rounds=args.warmup_rounds,
            cyclic_k=args.cyclic_k,
        )

    timeout_s = args.timeout_s if args.timeout_s is not None else DEFAULT_TIMEOUT_S[args.stage]

    print(f"Stage {args.stage}: {len(arms)} runs")
    for arm in arms:
        detail = (
            "monolithic"
            if arm.mode == "monolithic"
            else f"{arm.metric:<14} {arm.assignment:<13} {arm.slippage}"
        )
        print(f"  {arm.run_id:<42} eps={arm.epsilon:<7g} {detail}")
    if args.dry_run:
        return 0

    specs = write_stage(
        arms,
        queue_dir=Path(args.queue_dir) if Path(args.queue_dir).is_absolute()
        else REPO_ROOT / args.queue_dir,
        configs_dir=Path(args.configs_dir) if Path(args.configs_dir).is_absolute()
        else REPO_ROOT / args.configs_dir,
        output_root=Path(args.output_root) if Path(args.output_root).is_absolute()
        else REPO_ROOT / args.output_root,
        timeout_s=timeout_s,
        force=args.force,
    )
    print(f"Wrote {len(specs)} spec(s) to {args.queue_dir} "
          f"(timeout {timeout_s:.0f}s each)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
