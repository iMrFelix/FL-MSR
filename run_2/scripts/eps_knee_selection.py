"""Offline ε-grid selection from Stage-A telemetry (gate ruling G6).

Pipeline, per the gate doc:

1. **Affine per-class cost fit** ``t_c = alpha_c * n_msgs + bytes / B_hat_c``
   by least squares over the receiver-side arrival intervals logged in the
   uplink telemetry sidecar (falls back to the nominal shaped bandwidths
   when telemetry is missing — flagged in the output).
2. **β(ε) / predicted-t_ε curves**: for every ε on a [0, 0.6]/0.01 grid and
   every logged round manifest ``(u_l, s_l)``, run the *same*
   CoverageEFTStrategy code path the engine deploys (selector consistency —
   an offline re-implementation could place knees where the deployed
   scheduler has none, an E1-style artifact in miniature) and record the
   tail byte fraction β and the affine-model prediction of t_ε.
3. **Knee candidates**: grid points with maximal marginal byte saving per
   unit ε (steps of the mean β curve, i.e. where a large layer crosses into
   the tail).  Chosen ε set = {knee-low, the 1/16 FedLUAR anchor,
   knee-high}, capped at 0.5, at most 3 values; flat curve falls back to
   the pre-registered grid {0.0625, 0.2, 0.5}.
4. **Noise floor** = stddev of ``t_eps_local_receiver_s`` across
   same-config Stage-A repeats (groups differ only in the ``_rN`` suffix);
   pooled as the RMS over groups.  Stage-B arms predicted to sit below this
   floor are cut by the supervisor before launch.

Outputs a JSON summary (``chosen_eps``, ``noise_floor``, fit parameters,
curves) plus a matplotlib (Agg) PNG of both curves.

Usage::

    python -m scripts.eps_knee_selection \
        --runs results/overnight/runs \
        --out results/overnight/analysis/eps_selection.json \
        --plot results/overnight/analysis/eps_curves.png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from scripts.overnight_common import (
    ClassCost,
    ManifestRecord,
    collect_cost_observations,
    find_aggregator,
    find_run_dirs,
    fit_affine_class_costs,
    load_assignment_strategy,
    load_report,
    makespan_of_assignment,
    nominal_uplink_bandwidths,
    repeat_group,
    uplink_observations,
    worker_manifests,
)

logger = logging.getLogger(__name__)

DEFAULT_EPS_MIN = 0.0
DEFAULT_EPS_MAX = 0.6
DEFAULT_EPS_STEP = 0.01
#: FedLUAR's kappa < 1/16 safe-shedding condition, imported as a citable
#: anchor (gate ruling G6).
ANCHOR_EPS = 1.0 / 16.0
DEFAULT_CAP = 0.5
DEFAULT_MAX_VALUES = 3
#: A β step below 1% of total bytes is not a knee worth a run.
DEFAULT_MIN_JUMP = 0.01
FALLBACK_GRID = (0.0625, 0.2, 0.5)


def build_eps_grid(
    eps_min: float = DEFAULT_EPS_MIN,
    eps_max: float = DEFAULT_EPS_MAX,
    step: float = DEFAULT_EPS_STEP,
) -> list[float]:
    """Inclusive ε grid, rounded to the step's precision to avoid fp drift."""
    n_steps = int(round((eps_max - eps_min) / step))
    digits = max(0, -int(math.floor(math.log10(step))))
    return [round(eps_min + i * step, digits + 2) for i in range(n_steps + 1)]


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------

def compute_curves(
    manifests: Sequence[ManifestRecord],
    strategy: Any,
    costs: dict[int, ClassCost],
    eps_grid: Sequence[float],
) -> dict[str, Any]:
    """Mean β(ε) and predicted t_ε(ε) over all logged manifests.

    The strategy is invoked exactly like the engine invokes it (scores,
    sizes, bandwidths, ε, must_receive) on every manifest; t_ε is predicted
    by applying the fitted affine cost to the strategy's own head
    assignment, i.e. ``max_c (alpha_c * n_c + bytes_c / B_hat_c)``.
    """
    bandwidths = {cls: cost.mbps for cls, cost in costs.items()}
    beta_rows: list[list[float]] = []
    t_rows: list[list[float]] = []
    for manifest in manifests:
        total_bytes = manifest.total_bytes
        if total_bytes <= 0:
            continue
        betas: list[float] = []
        t_preds: list[float] = []
        for eps in eps_grid:
            try:
                result = strategy.assign(
                    scores=dict(manifest.scores),
                    sizes=dict(manifest.sizes),
                    bandwidths=dict(bandwidths),
                    epsilon=float(eps),
                    must_receive=set(),
                    ages=None,
                )
            except Exception as exc:  # diagnose, keep the sweep alive
                logger.warning(
                    "strategy.assign failed (run=%s round=%d source=%s "
                    "eps=%.4f): %s",
                    manifest.run_id, manifest.round, manifest.source, eps, exc,
                )
                betas.append(float("nan"))
                t_preds.append(float("nan"))
                continue
            tail_bytes = sum(manifest.sizes[name] for name in result.tail)
            betas.append(tail_bytes / total_bytes)
            t_preds.append(
                makespan_of_assignment(
                    result.assignment, manifest.sizes, result.head, costs
                )
            )
        beta_rows.append(betas)
        t_rows.append(t_preds)

    if not beta_rows:
        raise ValueError("no usable manifests — nothing to select ε from")

    def _nanmean(rows: list[list[float]], idx: int) -> float:
        vals = [r[idx] for r in rows if not math.isnan(r[idx])]
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "eps": list(eps_grid),
        "beta_mean": [_nanmean(beta_rows, i) for i in range(len(eps_grid))],
        "t_pred_mean": [_nanmean(t_rows, i) for i in range(len(eps_grid))],
        "n_manifests": len(beta_rows),
    }


# ---------------------------------------------------------------------------
# Knee selection
# ---------------------------------------------------------------------------

def find_knees(
    eps_grid: Sequence[float],
    beta_curve: Sequence[float],
    anchor: float = ANCHOR_EPS,
    cap: float = DEFAULT_CAP,
    max_values: int = DEFAULT_MAX_VALUES,
    min_jump: float = DEFAULT_MIN_JUMP,
    fallback: Sequence[float] = FALLBACK_GRID,
) -> dict[str, Any]:
    """Pick ≤ ``max_values`` ε values at the knees of the β(ε) curve.

    A knee candidate is a grid point whose β step over the previous point
    (the marginal byte saving per grid step) is at least ``min_jump`` —
    β is a step function of ε, jumping exactly where a layer crosses into
    the tail, so picking ε *at* the jump banks that layer's bytes at the
    minimum coverage sacrifice.

    The chosen set is {knee-low, anchor, knee-high ≤ cap}, deduplicated and
    sorted.  If no candidate jump exists (flat curve), returns the
    pre-registered fallback grid and flags it.
    """
    candidates: list[dict[str, float]] = []
    for i in range(1, len(eps_grid)):
        prev, cur = beta_curve[i - 1], beta_curve[i]
        if math.isnan(prev) or math.isnan(cur):
            continue
        jump = cur - prev
        if jump >= min_jump:
            candidates.append({"eps": float(eps_grid[i]), "jump": float(jump)})

    if not candidates:
        return {
            "chosen_eps": sorted(float(e) for e in fallback if 0.0 < e <= cap),
            "knee_candidates": [],
            "fallback_used": True,
        }

    in_cap = [c for c in candidates if 0.0 < c["eps"] <= cap]
    pool = in_cap if in_cap else candidates  # all knees above cap: cap below
    knee_low = min(c["eps"] for c in pool)
    knee_high = max(c["eps"] for c in pool)

    chosen = {min(knee_low, cap), min(cap, max(anchor, 0.0)), min(knee_high, cap)}
    chosen_sorted = sorted(e for e in chosen if e > 0.0)[:max_values]
    return {
        "chosen_eps": chosen_sorted,
        "knee_candidates": sorted(
            candidates, key=lambda c: (-c["jump"], c["eps"])
        ),
        "fallback_used": False,
    }


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------

def compute_noise_floor(
    per_run_t_eps: dict[str, list[float]],
) -> dict[str, Any]:
    """Pooled stddev of t_eps_local_receiver_s across same-config repeat groups.

    Args:
        per_run_t_eps: run_id -> all ``t_eps_local_receiver_s``
            samples of that run (one per round x source).

    Returns:
        dict with per-group stats and the pooled (RMS over groups) floor in
        seconds, or ``noise_floor=None`` when no group has >= 2 repeats with
        telemetry (e.g. the engine-side sidecar is not integrated yet).
    """
    groups: dict[str, dict[str, float]] = {}
    by_group: dict[str, list[float]] = {}
    for run_id, samples in sorted(per_run_t_eps.items()):
        if not samples:
            continue
        by_group.setdefault(repeat_group(run_id), []).append(
            sum(samples) / len(samples)
        )
    variances: list[float] = []
    for group, means in sorted(by_group.items()):
        if len(means) < 2:
            continue
        stdev = statistics.stdev(means)
        groups[group] = {
            "n_repeats": len(means),
            "mean_t_eps_s": sum(means) / len(means),
            "stdev_t_eps_s": stdev,
        }
        variances.append(stdev**2)
    if not variances:
        return {"noise_floor": None, "per_group": groups}
    return {
        "noise_floor": math.sqrt(sum(variances) / len(variances)),
        "per_group": groups,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_curves(
    curves: dict[str, list[float]],
    chosen_eps: Sequence[float],
    anchor: float,
    noise_floor: float | None,
    out_path: Path,
) -> None:
    """Two-panel β(ε) / predicted-t_ε(ε) figure (Agg backend, headless)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_beta, ax_t) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    eps = curves["eps"]

    ax_beta.step(eps, curves["beta_mean"], where="post", color="tab:blue")
    ax_beta.set_ylabel("β(ε) — tail byte fraction")
    ax_beta.set_title("Stage-A manifests: shed-byte fraction and predicted t_ε")
    ax_beta.grid(True, alpha=0.3)

    ax_t.plot(eps, curves["t_pred_mean"], color="tab:red")
    ax_t.set_ylabel("predicted t_ε (s, affine fit)")
    ax_t.set_xlabel("ε")
    ax_t.grid(True, alpha=0.3)
    if noise_floor is not None:
        ax_t.axhspan(0, noise_floor, color="gray", alpha=0.25,
                     label=f"noise floor {noise_floor:.3f}s")
        ax_t.legend(loc="upper right")

    for ax in (ax_beta, ax_t):
        ax.axvline(anchor, color="green", linestyle=":", alpha=0.8)
        for e in chosen_eps:
            ax.axvline(e, color="black", linestyle="--", alpha=0.6)
    ax_beta.text(anchor, ax_beta.get_ylim()[1] * 0.95, " 1/16 anchor",
                 color="green", fontsize=8, va="top")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Choose the Stage-B ε grid from Stage-A telemetry (G6)."
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="Stage-A run output dirs (or parents thereof, e.g. "
        "results/overnight/runs).",
    )
    parser.add_argument(
        "--groups", nargs="*", default=None,
        help="Only use manifests from runs whose repeat-group name contains "
        "one of these substrings (e.g. 'coveft_delta'). Default: every "
        "per-layer run found.",
    )
    parser.add_argument("--strategy", default="coverage_eft",
                        help="Registered assignment strategy to sweep "
                        "(must be the deployed one for selector consistency).")
    parser.add_argument("--out", default="results/overnight/analysis/eps_selection.json")
    parser.add_argument("--plot", default="results/overnight/analysis/eps_curves.png")
    parser.add_argument("--eps-min", type=float, default=DEFAULT_EPS_MIN)
    parser.add_argument("--eps-max", type=float, default=DEFAULT_EPS_MAX)
    parser.add_argument("--eps-step", type=float, default=DEFAULT_EPS_STEP)
    parser.add_argument("--anchor", type=float, default=ANCHOR_EPS)
    parser.add_argument("--cap", type=float, default=DEFAULT_CAP)
    parser.add_argument("--max-values", type=int, default=DEFAULT_MAX_VALUES)
    parser.add_argument("--min-jump", type=float, default=DEFAULT_MIN_JUMP)
    parser.add_argument("--min-round", type=int, default=0,
                        help="Ignore manifests from rounds below this "
                        "(round 0 deltas can be degenerate).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dirs = find_run_dirs(args.runs)
    if not run_dirs:
        print("ERROR: no run dirs with results/report.json found", file=sys.stderr)
        return 2

    manifests: list[ManifestRecord] = []
    per_run_t_eps: dict[str, list[float]] = {}
    cost_obs: dict[int, list[tuple[int, int, float]]] = {}
    nominal_bw: dict[int, float] = {}
    used_runs: list[str] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        if args.groups and not any(g in repeat_group(run_id) for g in args.groups):
            continue
        report = load_report(run_dir)
        if report is None:
            continue
        aggregator = find_aggregator(run_dir)
        run_manifests = [
            m for m in worker_manifests(run_id, report, aggregator)
            if m.round >= args.min_round
        ]
        telemetry = uplink_observations(run_id, report, aggregator)
        per_run_t_eps[run_id] = [
            o.t_eps_local_receiver_s for o in telemetry
            if o.t_eps_local_receiver_s is not None
        ]
        for cls, obs in collect_cost_observations(run_manifests, telemetry).items():
            cost_obs.setdefault(cls, []).extend(obs)
        if run_manifests:
            manifests.extend(run_manifests)
            used_runs.append(run_id)
            if not nominal_bw:
                nominal_bw = {
                    cls: bw
                    for cls, bw in nominal_uplink_bandwidths(run_dir).items()
                    if math.isfinite(bw)
                }

    if not manifests:
        print("ERROR: no per-layer manifests in the given runs", file=sys.stderr)
        return 2
    print(f"Using {len(manifests)} manifests from {len(used_runs)} runs")

    costs = fit_affine_class_costs(cost_obs, nominal_bw)
    for cls, cost in sorted(costs.items()):
        print(f"  class {cls}: alpha={cost.alpha_s * 1e3:.2f} ms/msg, "
              f"B_hat={cost.mbps:.3f} Mbps ({cost.source})")

    strategy = load_assignment_strategy(args.strategy)
    eps_grid = build_eps_grid(args.eps_min, args.eps_max, args.eps_step)
    curves = compute_curves(manifests, strategy, costs, eps_grid)
    knees = find_knees(
        eps_grid,
        curves["beta_mean"],
        anchor=args.anchor,
        cap=args.cap,
        max_values=args.max_values,
        min_jump=args.min_jump,
    )
    noise = compute_noise_floor(per_run_t_eps)

    plot_path = Path(args.plot)
    plot_curves(curves, knees["chosen_eps"], args.anchor,
                noise["noise_floor"], plot_path)

    summary = {
        "chosen_eps": knees["chosen_eps"],
        "fallback_used": knees["fallback_used"],
        "knee_candidates": knees["knee_candidates"],
        "anchor_eps": args.anchor,
        "cap": args.cap,
        "noise_floor": noise["noise_floor"],
        "noise_per_group": noise["per_group"],
        "cost_fit": {
            str(cls): {
                "alpha_s": cost.alpha_s,
                "B_hat_bytes_per_s": cost.bytes_per_s,
                "B_hat_mbps": cost.mbps,
                "source": cost.source,
            }
            for cls, cost in sorted(costs.items())
        },
        "n_manifests": len(manifests),
        "runs_used": used_runs,
        "strategy": args.strategy,
        "curves": {
            "eps": curves["eps"],
            "beta_mean": curves["beta_mean"],
            "t_pred_mean": curves["t_pred_mean"],
        },
        "curves_png": str(plot_path),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(f"chosen_eps = {knees['chosen_eps']}"
          + (" (FALLBACK grid — flat β curve)" if knees["fallback_used"] else ""))
    nf = noise["noise_floor"]
    print(f"noise_floor = {nf:.4f}s" if nf is not None
          else "noise_floor = n/a (no repeat telemetry)")
    print(f"wrote {out_path} and {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
