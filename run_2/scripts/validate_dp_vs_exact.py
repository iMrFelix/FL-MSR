"""DP-fallback validation: quantized-utility DP vs exact tail enumeration.

Phase-1 plan T2 validation evidence (``writeup/04-phase1/plan.md``):

(a) On every logged L=14 Stage-A manifest (12 per-layer runs x 3 workers x
    3 rounds = 108), compare the auto-selected exact enumeration against the
    force-selected >20-layer fallback (greedy density + quantized-utility
    DP, utility grid 1e-4 of the eps*U budget) across an epsilon sweep.
    PASS per (manifest, eps): identical shed sets OR shed-byte delta
    < 1% of the manifest's total bytes.  The DP can never shed *more* than
    the exact optimum (both are budget-feasible integral sheds), so a
    negative delta is reported as a hard exactness failure.

(b) On synthetic L=65 manifests with the ResNet-20 layer-size profile
    (the regime the fallback exists for), run the *default* strategy —
    which must auto-select the DP fallback at 65 > 20 candidates — and
    report the DP's realized tail byte fraction beta against the LP
    (fractional-knapsack) upper bound, plus the assign() runtime against
    the plan's 50 ms ceiling.

Selector consistency (gate ruling G6 ethos): both parts call the deployed
``CoverageEFTStrategy.assign`` code path, never an offline re-implementation;
the LP bound reuses the module's own ``_fractional_shed_bytes`` diagnostic
helper.

Exit code 0 iff part (a) passes on every pair AND part (b) meets the
runtime ceiling (the beta-vs-LP gap is *reported*, per the plan wording).

Usage::

    python -m scripts.validate_dp_vs_exact \
        --runs results/overnight/runs --prefix a \
        --out results/phase1/dp_vs_exact_validation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from scripts.overnight_common import (
    ManifestRecord,
    find_aggregator,
    find_run_dirs,
    load_report,
    nominal_uplink_bandwidths,
    worker_manifests,
)
from src.importance.assignment import (
    CoverageEFTStrategy,
    _fractional_shed_bytes,
)

logger = logging.getLogger(__name__)

#: The 1% criterion of plan T2(a), expressed as a fraction of total
#: manifest bytes (i.e. a delta in beta percentage points).
SHED_DELTA_TOLERANCE = 0.01

#: Plan T2(b) runtime ceiling for one assign() call at L=65.
RUNTIME_BUDGET_MS = 50.0

#: FedLUAR 1/16 anchor — always included in the sweep (gate ruling G6).
ANCHOR_EPS = 1.0 / 16.0

#: Default bandwidth map when a run dir carries no usable config and for
#: the synthetic part: the testbed's standard 3-class uplink shape (Mbps).
DEFAULT_BANDWIDTHS = {0: 6.0, 1: 3.0, 2: 1.0}

#: Acceptance-side fp bias the strategy itself documents (BUDGET_RTOL);
#: coverage violations are only flagged beyond twice that.
_COVERAGE_SLACK = 2e-9


def build_eps_sweep(
    eps_min: float = 0.01,
    eps_max: float = 0.60,
    step: float = 0.01,
    anchor: float = ANCHOR_EPS,
) -> list[float]:
    """Epsilon sweep: inclusive grid plus the 1/16 anchor, deduplicated.

    eps = 0 is excluded by design: both code paths return the empty tail
    without entering any knapsack solver, so it validates nothing.
    """
    n_steps = int(round((eps_max - eps_min) / step))
    grid = {round(eps_min + i * step, 6) for i in range(n_steps + 1)}
    grid.add(round(anchor, 6))
    return sorted(e for e in grid if e > 0.0)


# ---------------------------------------------------------------------------
# Synthetic ResNet-20 profile (part b)
# ---------------------------------------------------------------------------

def resnet20_layer_profile() -> list[tuple[str, int]]:
    """The 65-variable ResNet-20 (CIFAR) layer-size profile, in depth order.

    Composition (He et al.'s CIFAR ResNet-20 with option-B projection
    shortcuts so the variable count lands at the plan's "~65 vars"):

    - stem: 3x3x3x16 conv + BN(gamma, beta)
    - 3 stages x 3 blocks x 2 convs (3x3), widths 16/32/64, each conv
      followed by BN(gamma, beta); first conv of stages 2/3 strided with a
      1x1 projection shortcut (+BN)
    - dense head: 64x10 kernel + bias

    21 conv kernels + 21 BN gamma + 21 BN beta + dense kernel + dense bias
    = 65 variables, 272,474 parameters (~0.27 M — the canonical ResNet-20
    figure), spanning 40 B BN vectors to 147,456 B kernels at float32: the
    byte/importance anti-correlation regime the scheduler targets.
    """
    profile: list[tuple[str, int]] = []

    def conv(name: str, k: int, c_in: int, c_out: int) -> None:
        profile.append((f"{name}/kernel", k * k * c_in * c_out))
        profile.append((f"{name}/bn_gamma", c_out))
        profile.append((f"{name}/bn_beta", c_out))

    conv("conv_in", 3, 3, 16)
    widths = (16, 32, 64)
    for stage, width in enumerate(widths):
        in_width = 16 if stage == 0 else widths[stage - 1]
        for block in range(3):
            first_in = in_width if block == 0 else width
            conv(f"s{stage}b{block}_conv0", 3, first_in, width)
            conv(f"s{stage}b{block}_conv1", 3, width, width)
        if stage > 0:  # strided projection shortcut into the new width
            conv(f"s{stage}_shortcut", 1, in_width, width)
    profile.append(("dense/kernel", 64 * 10))
    profile.append(("dense/bias", 10))
    return profile


def resnet20_sizes(bytes_per_param: int = 4) -> dict[str, int]:
    """Byte sizes of the profile at float32 (serialized payload approx)."""
    return {
        name: params * bytes_per_param
        for name, params in resnet20_layer_profile()
    }


def synthetic_scores(
    model: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    """One synthetic utility manifest over the ResNet-20 profile.

    Models:
        ``iid_lognormal`` — heavy-tailed utilities independent of size
            (lognormal sigma=1.5, the shape of logged delta-norm masses).
        ``anti_correlated`` — lognormal scaled by 1/sqrt(params): small
            vectors dense, big kernels cheap per byte (the E2/E6 pathology
            that makes shed tails byte-heavy).
    """
    profile = resnet20_layer_profile()
    if model == "iid_lognormal":
        draws = rng.lognormal(mean=0.0, sigma=1.5, size=len(profile))
        return {name: float(d) for (name, _), d in zip(profile, draws)}
    if model == "anti_correlated":
        draws = rng.lognormal(mean=0.0, sigma=1.0, size=len(profile))
        return {
            name: float(d) / math.sqrt(params)
            for (name, params), d in zip(profile, draws)
        }
    raise ValueError(f"unknown synthetic utility model {model!r}")


# ---------------------------------------------------------------------------
# Part (a): logged manifests, DP vs exact
# ---------------------------------------------------------------------------

def compare_dp_vs_exact(
    manifest: ManifestRecord,
    bandwidths: dict[int, float],
    eps_sweep: Sequence[float],
    exact_strategy: CoverageEFTStrategy,
    dp_strategy: CoverageEFTStrategy,
) -> list[dict[str, Any]]:
    """One comparison row per epsilon for a single logged manifest."""
    rows: list[dict[str, Any]] = []
    total_bytes = manifest.total_bytes
    total_score = manifest.total_score
    for eps in eps_sweep:
        kwargs = dict(
            scores=dict(manifest.scores),
            sizes=dict(manifest.sizes),
            bandwidths=dict(bandwidths),
            epsilon=float(eps),
            must_receive=set(),
            ages=None,
        )
        res_exact = exact_strategy.assign(**kwargs)
        res_dp = dp_strategy.assign(**kwargs)

        bytes_exact = sum(manifest.sizes[n] for n in res_exact.tail)
        bytes_dp = sum(manifest.sizes[n] for n in res_dp.tail)
        shed_util_dp = sum(manifest.scores[n] for n in res_dp.tail)
        delta = bytes_exact - bytes_dp
        identical = res_exact.tail == res_dp.tail
        delta_frac_total = (delta / total_bytes) if total_bytes else 0.0
        coverage_ok = shed_util_dp <= (
            eps * total_score + _COVERAGE_SLACK * max(1.0, total_score)
        )
        rows.append({
            "run_id": manifest.run_id,
            "round": manifest.round,
            "source": manifest.source,
            "eps": float(eps),
            "n_layers": len(manifest.scores),
            "exact_method": res_exact.diagnostics.get("tail_method"),
            "dp_method": res_dp.diagnostics.get("tail_method"),
            "identical_sets": bool(identical),
            "shed_bytes_exact": int(bytes_exact),
            "shed_bytes_dp": int(bytes_dp),
            "delta_bytes": int(delta),
            "delta_frac_total_bytes": float(delta_frac_total),
            "delta_frac_exact_shed": (
                float(delta / bytes_exact) if bytes_exact > 0 else 0.0
            ),
            "dp_exceeds_exact": bool(delta < 0),
            "coverage_ok": bool(coverage_ok),
            "pass": bool(
                (identical or delta_frac_total < SHED_DELTA_TOLERANCE)
                and delta >= 0
                and coverage_ok
            ),
        })
    return rows


def load_logged_manifests(
    run_paths: Iterable[str],
    prefix: str,
) -> tuple[list[ManifestRecord], dict[str, dict[int, float]]]:
    """Stage manifests + per-run nominal uplink bandwidths.

    ``prefix`` filters run dirs by name (default 'a' selects the Stage-A
    runs, whose 12 per-layer members carry the canonical 108 L=14
    manifests).  Empty prefix admits every run found.
    """
    manifests: list[ManifestRecord] = []
    bandwidths_by_run: dict[str, dict[int, float]] = {}
    for run_dir in find_run_dirs(list(run_paths)):
        run_id = run_dir.name
        if prefix and not run_id.startswith(prefix):
            continue
        report = load_report(run_dir)
        if report is None:
            continue
        aggregator = find_aggregator(run_dir)
        records = worker_manifests(run_id, report, aggregator)
        if not records:
            continue
        manifests.extend(records)
        bw = {
            cls: b
            for cls, b in nominal_uplink_bandwidths(run_dir).items()
            if b > 0
        }
        bandwidths_by_run[run_id] = bw or dict(DEFAULT_BANDWIDTHS)
    return manifests, bandwidths_by_run


def run_part_a(
    run_paths: Iterable[str],
    prefix: str,
    eps_sweep: Sequence[float],
) -> dict[str, Any]:
    manifests, bandwidths_by_run = load_logged_manifests(run_paths, prefix)
    if not manifests:
        return {"status": "no_manifests", "pass": False}

    exact_strategy = CoverageEFTStrategy()  # L=14 <= 20: exact enumeration
    dp_strategy = CoverageEFTStrategy(exact_tail_limit=0)  # force fallback

    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        rows.extend(
            compare_dp_vs_exact(
                manifest,
                bandwidths_by_run[manifest.run_id],
                eps_sweep,
                exact_strategy,
                dp_strategy,
            )
        )

    failures = [r for r in rows if not r["pass"]]
    non_identical = [r for r in rows if not r["identical_sets"]]
    layer_hist: dict[int, int] = {}
    for m in manifests:
        layer_hist[len(m.scores)] = layer_hist.get(len(m.scores), 0) + 1

    summary = {
        "status": "ok",
        "n_manifests": len(manifests),
        "layer_count_histogram": {str(k): v for k, v in sorted(layer_hist.items())},
        "n_runs": len(bandwidths_by_run),
        "eps_sweep": [float(e) for e in eps_sweep],
        "n_pairs": len(rows),
        "n_identical": sum(r["identical_sets"] for r in rows),
        "n_pass": sum(r["pass"] for r in rows),
        "n_fail": len(failures),
        "n_dp_exceeds_exact": sum(r["dp_exceeds_exact"] for r in rows),
        "n_coverage_violations": sum(not r["coverage_ok"] for r in rows),
        "max_delta_frac_total_bytes": max(
            (r["delta_frac_total_bytes"] for r in rows), default=0.0
        ),
        "max_delta_frac_exact_shed": max(
            (r["delta_frac_exact_shed"] for r in rows), default=0.0
        ),
        "tolerance_frac_total_bytes": SHED_DELTA_TOLERANCE,
        "non_identical_pairs": non_identical[:50],
        "failing_pairs": failures[:50],
        "pass": not failures,
    }
    return summary


# ---------------------------------------------------------------------------
# Part (b): synthetic L=65, beta vs LP bound + runtime
# ---------------------------------------------------------------------------

def run_part_b(
    eps_sweep: Sequence[float],
    n_per_model: int,
    seed: int,
    runtime_budget_ms: float,
    bandwidths: dict[int, float] | None = None,
) -> dict[str, Any]:
    sizes = resnet20_sizes()
    names = list(sizes)
    bandwidths = dict(bandwidths or DEFAULT_BANDWIDTHS)
    strategy = CoverageEFTStrategy()  # default limit: 65 > 20 auto-selects DP

    # One untimed warmup call so the timing reflects the steady-state
    # per-round cost (module/numpy buffers warm), which is what the engine
    # pays every round; the cold first call is recorded separately.
    warm_scores = synthetic_scores("iid_lognormal", np.random.default_rng(seed))
    t0 = time.perf_counter()
    strategy.assign(
        scores=warm_scores, sizes=sizes, bandwidths=bandwidths,
        epsilon=float(eps_sweep[0]), must_receive=set(), ages=None,
    )
    cold_call_ms = (time.perf_counter() - t0) * 1e3

    rows: list[dict[str, Any]] = []
    runtimes_ms: list[float] = []
    methods: set[str] = set()
    for model_idx, model in enumerate(("iid_lognormal", "anti_correlated")):
        for k in range(n_per_model):
            rng = np.random.default_rng(seed + 1000 * model_idx + k)
            scores = synthetic_scores(model, rng)
            total_bytes = float(sum(sizes.values()))
            total_score = float(sum(scores.values()))
            for eps in eps_sweep:
                t0 = time.perf_counter()
                result = strategy.assign(
                    scores=scores, sizes=sizes, bandwidths=bandwidths,
                    epsilon=float(eps), must_receive=set(), ages=None,
                )
                dt_ms = (time.perf_counter() - t0) * 1e3
                runtimes_ms.append(dt_ms)
                methods.add(result.diagnostics.get("tail_method", "?"))

                # LP (fractional-knapsack) shed upper bound on the same
                # candidates and budget formula the strategy used.
                budget_tol = eps * total_score + (
                    CoverageEFTStrategy.BUDGET_RTOL
                    * max(1.0, abs(total_score))
                )
                candidates = sorted(names)
                utilities = np.array(
                    [scores[n] for n in candidates], dtype=np.float64,
                )
                byte_arr = np.array(
                    [float(sizes[n]) for n in candidates], dtype=np.float64,
                )
                lp_shed = _fractional_shed_bytes(
                    candidates, utilities, byte_arr, budget_tol,
                )
                beta_dp = float(result.diagnostics["beta_realized"])
                beta_lp = lp_shed / total_bytes
                shed_util = float(result.diagnostics["shed_utility"])
                rows.append({
                    "model": model,
                    "manifest": k,
                    "eps": float(eps),
                    "beta_dp": beta_dp,
                    "beta_lp_bound": float(beta_lp),
                    "beta_gap": float(beta_lp - beta_dp),
                    "tail_method": result.diagnostics.get("tail_method"),
                    "fallback_winner": result.diagnostics.get("fallback_winner"),
                    "runtime_ms": dt_ms,
                    "coverage_ok": bool(
                        shed_util
                        <= eps * total_score
                        + _COVERAGE_SLACK * max(1.0, total_score)
                    ),
                    "lp_fluid_bound_s": result.diagnostics.get("lp_fluid_bound_s"),
                    "predicted_t_eps_s": result.predicted_t_eps,
                })

    gaps = [r["beta_gap"] for r in rows]

    def _aggregate(rs: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "mean_beta_dp": statistics.fmean(r["beta_dp"] for r in rs),
            "mean_beta_lp": statistics.fmean(r["beta_lp_bound"] for r in rs),
            "mean_gap": statistics.fmean(r["beta_gap"] for r in rs),
            "max_gap": max(r["beta_gap"] for r in rs),
        }

    by_eps: dict[float, list[dict[str, Any]]] = {}
    by_model_eps: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for r in rows:
        by_eps.setdefault(r["eps"], []).append(r)
        by_model_eps.setdefault((r["model"], r["eps"]), []).append(r)
    per_eps = {
        f"{eps:.4f}": _aggregate(rs) for eps, rs in sorted(by_eps.items())
    }
    # Per-model split: the anti-correlated model is *built* to make tails
    # byte-heavy (large beta at small eps is its win condition, not a bug),
    # so pooled means alone would mislead.
    per_model: dict[str, dict[str, dict[str, float]]] = {}
    for (model, eps), rs in sorted(by_model_eps.items()):
        per_model.setdefault(model, {})[f"{eps:.4f}"] = _aggregate(rs)
    runtimes_sorted = sorted(runtimes_ms)
    runtime_max = max(runtimes_ms)
    negative_gaps = [g for g in gaps if g < -1e-9]
    summary = {
        "status": "ok",
        "n_layers": len(names),
        "total_bytes": int(sum(sizes.values())),
        "n_manifests": 2 * n_per_model,
        "n_calls_timed": len(runtimes_ms),
        "tail_methods_observed": sorted(methods),
        "auto_selected_dp": methods == {"quantized_dp_fallback"},
        "beta_gap_mean": statistics.fmean(gaps) if gaps else 0.0,
        "beta_gap_max": max(gaps) if gaps else 0.0,
        "n_negative_gaps": len(negative_gaps),
        "per_eps": per_eps,
        "per_model": per_model,
        "runtime_ms": {
            "cold_first_call": cold_call_ms,
            "mean": statistics.fmean(runtimes_ms),
            "p95": runtimes_sorted[int(0.95 * (len(runtimes_sorted) - 1))],
            "max": runtime_max,
            "budget": runtime_budget_ms,
        },
        "n_coverage_violations": sum(not r["coverage_ok"] for r in rows),
        "runtime_pass": runtime_max < runtime_budget_ms,
        "pass": (
            runtime_max < runtime_budget_ms
            and methods == {"quantized_dp_fallback"}
            and not negative_gaps
            and not any(not r["coverage_ok"] for r in rows)
        ),
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the quantized-DP tail fallback against the "
        "exact enumeration (plan T2 criteria a+b)."
    )
    parser.add_argument(
        "--runs", nargs="+", default=["results/overnight/runs"],
        help="Run output dirs (or parents). Default: the overnight runs.",
    )
    parser.add_argument(
        "--prefix", default="a",
        help="Run-dir name prefix filter; 'a' = Stage A (the canonical 108 "
        "L=14 manifests). Empty string admits every run.",
    )
    parser.add_argument("--out", default="results/phase1/dp_vs_exact_validation.json")
    parser.add_argument("--eps-min", type=float, default=0.01)
    parser.add_argument("--eps-max", type=float, default=0.60)
    parser.add_argument("--eps-step", type=float, default=0.01)
    parser.add_argument(
        "--synthetic-per-model", type=int, default=10,
        help="Synthetic L=65 manifests per utility model (2 models).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runtime-budget-ms", type=float,
                        default=RUNTIME_BUDGET_MS)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    eps_sweep = build_eps_sweep(args.eps_min, args.eps_max, args.eps_step)

    part_a = run_part_a(args.runs, args.prefix, eps_sweep)
    if part_a["status"] == "no_manifests":
        print("ERROR: no logged manifests found under "
              f"{args.runs} (prefix {args.prefix!r})", file=sys.stderr)
        return 2

    part_b = run_part_b(
        eps_sweep,
        n_per_model=args.synthetic_per_model,
        seed=args.seed,
        runtime_budget_ms=args.runtime_budget_ms,
    )

    overall_pass = bool(part_a["pass"] and part_b["pass"])
    summary = {
        "criteria": {
            "a": "logged manifests: identical shed sets or shed-byte delta "
                 f"< {SHED_DELTA_TOLERANCE:.0%} of total bytes",
            "b": "synthetic L=65 ResNet-20 profile: beta vs LP bound gap "
                 f"reported; assign() runtime < {args.runtime_budget_ms} ms",
        },
        "part_a_logged_manifests": part_a,
        "part_b_synthetic_l65": part_b,
        "pass": overall_pass,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(
        f"(a) logged manifests: {part_a['n_manifests']} manifests "
        f"(L histogram {part_a['layer_count_histogram']}), "
        f"{part_a['n_pairs']} (manifest, eps) pairs, "
        f"{part_a['n_identical']} identical, {part_a['n_fail']} failing, "
        f"max byte delta {part_a['max_delta_frac_total_bytes']:.3%} of total "
        f"-> {'PASS' if part_a['pass'] else 'FAIL'}"
    )
    rt = part_b["runtime_ms"]
    print(
        f"(b) synthetic L=65: beta gap mean {part_b['beta_gap_mean']:.5f} / "
        f"max {part_b['beta_gap_max']:.5f} (LP bound minus DP, in beta); "
        f"runtime mean {rt['mean']:.2f} ms, p95 {rt['p95']:.2f} ms, "
        f"max {rt['max']:.2f} ms (cold first call {rt['cold_first_call']:.2f} "
        f"ms) vs budget {rt['budget']:.0f} ms -> "
        f"{'PASS' if part_b['pass'] else 'FAIL'}"
    )
    print(f"wrote {out_path}")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
