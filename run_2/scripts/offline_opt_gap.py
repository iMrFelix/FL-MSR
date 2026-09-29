"""Offline per-round optimality-gap reference (gate §7, assignment item 9).

For every logged round manifest ``(u_l, s_l)`` and every evaluated ε this
computes, under the affine per-class cost model:

- ``t_fluid_lb`` — the LP/fluid lower bound: best over *exactly enumerated*
  feasible tail subsets of ``max(head_bytes/ΣB, max_layer/B_fastest)``.
  No discrete schedule can beat this.
- ``t_opt_eft`` — exact tail-subset enumeration (all 2^L subsets with tail
  utility ≤ ε·U, L=14 for DeepCNN) with the head placed by size-descending
  earliest-finish-time.  The subset choice — the complement-knapsack part,
  where the gate's theory repair says the density prefix is only the LP
  relaxation — is exact; head placement carries the GIS-1977 ≤1.38x factor.
- ``t_deployed_pred`` — the deployed strategy re-run on the same manifest
  (same code path the engine used), its head priced by the same cost model.
- ``t_realized`` — the receiver-measured ``t_eps_local_receiver_s`` from the uplink
  telemetry sidecar, when the evaluated ε equals the run's configured ε.

Gap columns divide realized/deployed by the OPT references.  One CSV row
per (run, round, source, ε).

``must_receive`` sets are not reconstructable from current logs and are
treated as empty; rows therefore lower-bound OPT for aging arms (a
must_receive constraint can only raise OPT), flagged in the column docs.

Usage::

    python -m scripts.offline_opt_gap --runs results/overnight/runs \
        --eps 0.0625 0.2 0.45 \
        --out results/overnight/analysis/opt_gap.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any

from scripts.overnight_common import (
    ClassCost,
    ManifestRecord,
    collect_cost_observations,
    find_aggregator,
    find_run_dirs,
    fit_affine_class_costs,
    load_assignment_strategy,
    load_node_config,
    load_report,
    makespan_of_assignment,
    nominal_uplink_bandwidths,
    run_epsilon,
    uplink_observations,
    worker_manifests,
)

logger = logging.getLogger(__name__)

#: 2^L subset enumeration is exact but exponential; DeepCNN has L=14.
MAX_ENUMERABLE_LAYERS = 20

#: Float-tolerance on the coverage budget so a tail whose utility equals
#: ε·U exactly (common with round numbers) is not excluded by fp noise.
_BUDGET_TOL = 1e-9


def enumerate_opt(
    scores: dict[str, float],
    sizes: dict[str, int],
    eps: float,
    costs: dict[int, ClassCost],
    must_receive: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Exact tail-subset enumeration of the ε-coverage scheduling problem.

    Args:
        scores: layer -> trigger-accounting utility (frozen metric, G2).
        sizes: layer -> payload bytes.
        eps: coverage tolerance; a subset T is a feasible tail iff
            ``Σ_{l∈T} u_l ≤ eps·Σu`` and ``T ∩ must_receive = ∅``.
        costs: affine per-class cost model.
        must_receive: layers that may never be shed.

    Returns:
        dict with ``t_fluid_lb``, ``t_opt_eft``, ``beta_opt`` (tail byte
        fraction at the EFT optimum), and enumeration stats.
    """
    if not costs:
        raise ValueError("enumerate_opt needs at least one traffic class")
    names = sorted(scores)
    n = len(names)
    if n > MAX_ENUMERABLE_LAYERS:
        raise ValueError(
            f"{n} layers exceed the enumeration bound "
            f"({MAX_ENUMERABLE_LAYERS}); use the fluid bound only"
        )
    u = [float(scores[name]) for name in names]
    s = [int(sizes[name]) for name in names]
    total_u = sum(u)
    total_b = sum(s)
    budget = eps * total_u + _BUDGET_TOL
    must_mask = 0
    for i, name in enumerate(names):
        if name in must_receive:
            must_mask |= 1 << i

    # Subset-sum DPs over bitmasks: drop the lowest set bit, add its weight.
    # These make fluid bounds O(1) per subset, leaving only the EFT pass in
    # the 2^L loop.
    size_count = 1 << n
    score_sum = [0.0] * size_count
    byte_sum = [0] * size_count
    max_byte = [0] * size_count
    for mask in range(1, size_count):
        low = mask & (-mask)
        idx = low.bit_length() - 1
        rest = mask ^ low
        score_sum[mask] = score_sum[rest] + u[idx]
        byte_sum[mask] = byte_sum[rest] + s[idx]
        max_byte[mask] = max(max_byte[rest], s[idx])

    cls_ids = sorted(costs)
    cls_alpha = [costs[c].alpha_s for c in cls_ids]
    cls_rate = [costs[c].bytes_per_s for c in cls_ids]
    n_cls = len(cls_ids)
    agg_rate = sum(cls_rate)
    top_rate = max(cls_rate)
    size_desc = sorted(range(n), key=lambda i: (-s[i], i))

    best_fluid = float("inf")
    best_eft = float("inf")
    best_eft_tail_bytes = 0
    n_feasible = 0
    full = size_count - 1
    for tail_mask in range(size_count):
        if tail_mask & must_mask:
            continue
        if score_sum[tail_mask] > budget:
            continue
        n_feasible += 1
        head_mask = full ^ tail_mask
        if head_mask:
            fluid = max(
                byte_sum[head_mask] / agg_rate, max_byte[head_mask] / top_rate
            )
        else:
            fluid = 0.0
        if fluid < best_fluid:
            best_fluid = fluid
        # Every feasible subset gets the EFT evaluation (not only maximal
        # tails): greedy list scheduling is not monotone under job removal
        # (Graham anomalies), so restricting to maximal tails could miss
        # the EFT-optimal subset.  Inlined size-descending EFT — identical
        # rule to overnight_common.eft_makespan, kept loop-local because
        # this runs 2^L times per (manifest, ε).
        loads = [0.0] * n_cls
        for i in size_desc:
            if not (head_mask >> i & 1):
                continue
            size_i = s[i]
            best_c = 0
            best_t = loads[0] + cls_alpha[0] + size_i / cls_rate[0]
            for c in range(1, n_cls):
                t = loads[c] + cls_alpha[c] + size_i / cls_rate[c]
                if t < best_t:
                    best_t = t
                    best_c = c
            loads[best_c] = best_t
        eft = max(loads) if loads else 0.0
        if eft < best_eft:
            best_eft = eft
            best_eft_tail_bytes = byte_sum[tail_mask]

    return {
        "t_fluid_lb": best_fluid,
        "t_opt_eft": best_eft,
        "beta_opt": (best_eft_tail_bytes / total_b) if total_b else 0.0,
        "n_layers": n,
        "n_feasible_tails": n_feasible,
        "total_bytes": total_b,
    }


def deployed_prediction(
    manifest: ManifestRecord,
    strategy_name: str | None,
    eps: float,
    costs: dict[int, ClassCost],
) -> tuple[float | None, float | None]:
    """Re-run the deployed strategy on the manifest; price its head.

    Returns ``(t_deployed_pred, beta_deployed)`` or ``(None, None)`` when
    the strategy is unavailable (not registered yet) or needs constructor
    arguments we cannot reconstruct from logs (cyclic state).
    """
    if not strategy_name or strategy_name == "cyclic":
        return None, None
    try:
        strategy = load_assignment_strategy(strategy_name)
    except (ValueError, TypeError) as exc:
        logger.debug("Strategy %r unavailable: %s", strategy_name, exc)
        return None, None
    bandwidths = {cls: cost.mbps for cls, cost in costs.items()}
    try:
        result = strategy.assign(
            scores=dict(manifest.scores),
            sizes=dict(manifest.sizes),
            bandwidths=bandwidths,
            epsilon=float(eps),
            must_receive=set(),
            ages=None,
        )
    except Exception as exc:
        logger.warning(
            "Deployed strategy %r failed on run=%s round=%d: %s",
            strategy_name, manifest.run_id, manifest.round, exc,
        )
        return None, None
    t_pred = makespan_of_assignment(
        result.assignment, manifest.sizes, result.head, costs
    )
    total_b = manifest.total_bytes
    beta = (
        sum(manifest.sizes[name] for name in result.tail) / total_b
        if total_b else 0.0
    )
    return t_pred, beta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-round optimality-gap CSV via exact subset enumeration."
    )
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run output dirs (or parents thereof).")
    parser.add_argument("--eps", type=float, nargs="*", default=[],
                        help="Extra ε values to evaluate besides each run's "
                        "own configured ε.")
    parser.add_argument("--out", default="results/overnight/analysis/opt_gap.csv")
    parser.add_argument("--min-round", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dirs = find_run_dirs(args.runs)
    if not run_dirs:
        print("ERROR: no run dirs with results/report.json found", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(run_dirs):
        report = load_report(run_dir)
        if report is None:
            continue
        aggregator = find_aggregator(run_dir)
        manifests = [
            m for m in worker_manifests(run_dir.name, report, aggregator)
            if m.round >= args.min_round
        ]
        if not manifests:
            continue  # monolithic arms have no manifests, hence no gap rows
        telemetry = {
            (o.round, o.source): o
            for o in uplink_observations(run_dir.name, report, aggregator)
        }
        # Per-run cost model: fit from this run's own telemetry where
        # possible so the gap reflects this run's wire reality.
        nominal = {
            cls: bw
            for cls, bw in nominal_uplink_bandwidths(run_dir).items()
            if bw != float("inf")
        }
        cost_obs = collect_cost_observations(manifests, list(telemetry.values()))
        costs = fit_affine_class_costs(cost_obs, nominal)
        if not costs:
            logger.warning("%s: no cost model derivable; skipping", run_dir.name)
            continue

        node_cfg = load_node_config(run_dir, "node-1") or {}
        strategy_name = (node_cfg.get("training") or {}).get("assignment_strategy")
        eps_run = run_epsilon(run_dir)
        eps_values = sorted({*(args.eps or [])}
                            | ({eps_run} if eps_run is not None else set()))
        if not eps_values:
            eps_values = [0.0]

        for manifest in manifests:
            obs = telemetry.get((manifest.round, manifest.source))
            for eps in eps_values:
                opt = enumerate_opt(manifest.scores, manifest.sizes, eps, costs)
                t_dep, beta_dep = deployed_prediction(
                    manifest, strategy_name, eps, costs
                )
                is_run_eps = eps_run is not None and abs(eps - eps_run) < 1e-9
                t_realized = (
                    obs.t_eps_local_receiver_s
                    if (is_run_eps and obs is not None) else None
                )
                rows.append({
                    "run_id": manifest.run_id,
                    "round": manifest.round,
                    "source": manifest.source,
                    "eps": eps,
                    "eps_is_run_eps": is_run_eps,
                    "strategy": strategy_name or "",
                    "n_layers": opt["n_layers"],
                    "total_bytes": opt["total_bytes"],
                    "n_feasible_tails": opt["n_feasible_tails"],
                    "t_fluid_lb": _r(opt["t_fluid_lb"]),
                    "t_opt_eft": _r(opt["t_opt_eft"]),
                    "beta_opt": _r(opt["beta_opt"]),
                    "t_deployed_pred": _r(t_dep),
                    "beta_deployed": _r(beta_dep),
                    "t_realized": _r(t_realized),
                    "gap_deployed_vs_opt": _ratio(t_dep, opt["t_opt_eft"]),
                    "gap_deployed_vs_fluid": _ratio(t_dep, opt["t_fluid_lb"]),
                    "gap_realized_vs_opt": _ratio(t_realized, opt["t_opt_eft"]),
                })

    if not rows:
        print("ERROR: no per-layer manifests found in the given runs",
              file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


def _r(value: float | None, digits: int = 6) -> float | str:
    """Round for CSV friendliness; empty string for missing values."""
    return "" if value is None else round(float(value), digits)


def _ratio(num: float | None, den: float | None) -> float | str:
    if num is None or den is None or not den > 0:
        return ""
    return round(float(num) / float(den), 4)


if __name__ == "__main__":
    sys.exit(main())
