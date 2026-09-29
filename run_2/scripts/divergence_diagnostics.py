"""Protocol heterogeneity diagnostics for H-P5 (protocol §2.5).

Computes the two pre-registered heterogeneity diagnostics from logged
manifests + aggregator telemetry, per ``(arm, seed)`` run:

- **Client-importance divergence** ``D_imp(s)`` — per round, the mean
  pairwise Jensen-Shannon divergence between workers' *normalized*
  per-layer utility vectors (utility = the manifest per-layer
  ``raw_score``, i.e. each worker's ``layer_comm_metrics.importance``,
  the G2-frozen ``delta_sq_norm`` trigger metric).  ``D_imp(s)`` is the
  mean over rounds 2..R (round 1 / the run's first round excluded — cold
  start, protocol §2.5).

- **Selection differentiation** ``D_sel(a, s)`` — per round, the mean
  pairwise Jaccard *distance* between workers' withheld sets
  ``M(i, r)``, where a layer ``L`` is in ``M(i, r)`` iff worker ``i`` is
  not in ``_aggregation.layers[L].arrived_sources`` for round ``r``
  (protocol §2.3 source-of-truth, §2.5).  For a global-selection arm
  (``skip_feedback: fedluar``) every worker withholds the same set, so
  ``D_sel`` is ~0 *by construction* — this script reports it, it does
  not assume it (H-P5 differentiation check (iii)).

Both divergences are computed numpy-only (no scipy dependency — scipy may
be absent locally; the optional self-test import below is wrapped
defensively).  Jensen-Shannon is base-2, ``JS(p,q) = 0.5 KL(p||m) +
0.5 KL(q||m)`` with ``m = (p+q)/2``, zero-guarded.

Reuses ``scripts/overnight_common.py`` for all report parsing
(``load_report``, ``find_aggregator``, ``worker_manifests``,
``parse_uplink_telemetry``, ``iter_round_nodes``); this script adds only
the divergence math and the ``(arm, seed)`` run-identity helpers.

Usage::

    python -m scripts.divergence_diagnostics --runs results/overnight/runs \
        --out results/overnight/analysis/divergence.csv

    # JSON output (carries the per-round arrays):
    python -m scripts.divergence_diagnostics --runs <dirs> \
        --out results/overnight/analysis/divergence.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make the sibling helper importable both as ``python -m scripts.x`` and as
# a direct ``python scripts/divergence_diagnostics.py`` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from overnight_common import (  # noqa: E402  (after sys.path shim)
    ManifestRecord,
    find_aggregator,
    find_run_dirs,
    iter_round_nodes,
    load_report,
    parse_uplink_telemetry,
    repeat_group,
    worker_manifests,
)

logger = logging.getLogger(__name__)

#: Seed suffix the spec generator appends to a run-dir / repeat-group name,
#: e.g. ``a09_skipv2_s44`` -> seed 44 (``dataset.partition.seed`` axis,
#: protocol §4.2).  ``_r2`` repeat suffixes are stripped by
#: ``overnight_common.repeat_group`` and carry no seed.
_SEED_SUFFIX = re.compile(r"_s(\d+)$")


# ---------------------------------------------------------------------------
# Divergence math (numpy-only)
# ---------------------------------------------------------------------------

def _kl_base2(p: np.ndarray, q: np.ndarray) -> float:
    """``KL(p||q)`` in bits, summed only where ``p>0`` (0 log 0 := 0).

    Assumes ``p`` and ``q`` are non-negative and ``q>0`` wherever ``p>0``
    (guaranteed when ``q`` is the mixture ``(p+other)/2``).
    """
    mask = p > 0.0
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits between two distributions.

    ``JS = 0.5 KL(p||m) + 0.5 KL(q||m)``, ``m = (p+q)/2``.  Inputs are
    L1-normalized first (so callers may pass un-normalized weight vectors);
    a degenerate all-zero vector yields ``nan`` (no distribution to
    compare) — the caller filters such pairs out.  Range ``[0, 1]`` for
    base-2 JS: 0 for identical, 1 for disjoint support.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    sp, sq = p.sum(), q.sum()
    if sp <= 0.0 or sq <= 0.0:
        return float("nan")
    p = p / sp
    q = q / sq
    m = 0.5 * (p + q)
    js = 0.5 * _kl_base2(p, m) + 0.5 * _kl_base2(q, m)
    # Clamp tiny negative/over-unity fp drift into the analytic range.
    if js < 0.0:
        return 0.0
    if js > 1.0:
        return 1.0
    return js


def jaccard_distance(a: set[str], b: set[str]) -> float:
    """Jaccard distance ``1 - |a∩b|/|a∪b|`` between two sets.

    Two empty sets (neither worker withheld anything) have distance 0 —
    they are identical (no differentiation), the correct reading for the
    global-selection by-construction case.
    """
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def _mean_pairwise(values: list[float]) -> float | None:
    """Mean of a pairwise list, or None when no valid pairs exist."""
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))


# ---------------------------------------------------------------------------
# Per-round diagnostics from a single report
# ---------------------------------------------------------------------------

def d_imp_per_round(
    manifests: list[ManifestRecord],
) -> dict[int, float]:
    """Mean pairwise JS divergence of worker utility vectors, per round.

    Each worker's vector is its manifest per-layer ``raw_score`` (==
    ``layer_comm_metrics.importance``), aligned over the round's union of
    layer names and L1-normalized to sum 1 inside ``js_divergence``.
    Negative scores (none expected for ``delta_sq_norm``) are floored at 0
    so the vector is a valid distribution.  Rounds with <2 usable workers
    are omitted from the returned map.
    """
    by_round: dict[int, list[ManifestRecord]] = {}
    for m in manifests:
        by_round.setdefault(m.round, []).append(m)

    out: dict[int, float] = {}
    for rnd, recs in by_round.items():
        recs = sorted(recs, key=lambda r: r.source)
        layers = sorted({name for r in recs for name in r.scores})
        if not layers:
            continue
        vectors: dict[str, np.ndarray | None] = {}
        for r in recs:
            vec = np.array(
                [max(r.scores.get(name, 0.0), 0.0) for name in layers],
                dtype=float,
            )
            vectors[r.source] = vec if vec.sum() > 0.0 else None
        sources = [r.source for r in recs]
        js_vals = [
            js_divergence(vectors[a], vectors[b])
            for a, b in itertools.combinations(sources, 2)
            if vectors[a] is not None and vectors[b] is not None
        ]
        mean = _mean_pairwise(js_vals)
        if mean is not None:
            out[rnd] = mean
    return out


def withheld_sets_per_round(
    report: dict[str, Any],
    aggregator: str,
    workers: list[str],
) -> dict[int, dict[str, set[str]]]:
    """``round -> {worker -> withheld layer set M(i, r)}`` from telemetry.

    Source of truth is the aggregator's ``_aggregation.layers`` block:
    a layer ``L`` is in ``M(i, r)`` iff worker ``i`` is absent from
    ``arrived_sources(L, r)`` (protocol §2.3).  The layer universe per
    round is exactly the set of aggregated layers (the
    ``_aggregation.layers`` keys), so a layer a worker never manifested is
    correctly counted as withheld for that worker (omitted-by-strategy is
    one of the §2.3 withheld categories).
    """
    out: dict[int, dict[str, set[str]]] = {}
    for rnum, node_id, entry in iter_round_nodes(report):
        if node_id != aggregator:
            continue
        telemetry = parse_uplink_telemetry(entry)
        if not telemetry:
            continue
        agg_block = telemetry.get("_aggregation") or {}
        layers = agg_block.get("layers") or {}
        if not layers:
            # No aggregation bookkeeping this round (e.g. monolithic round):
            # nothing withheld is observable; record empties so the round
            # still contributes a (degenerate) D_sel = 0 rather than being
            # silently dropped.
            out[rnum] = {w: set() for w in workers}
            continue
        withheld: dict[str, set[str]] = {w: set() for w in workers}
        for layer_name, layer_val in layers.items():
            arrived = set((layer_val or {}).get("arrived_sources") or [])
            for w in workers:
                if w not in arrived:
                    withheld[w].add(layer_name)
        out[rnum] = withheld
    return out


def d_sel_per_round(
    withheld_by_round: dict[int, dict[str, set[str]]],
) -> dict[int, float]:
    """Mean pairwise Jaccard distance of worker withheld sets, per round."""
    out: dict[int, float] = {}
    for rnd, withheld in withheld_by_round.items():
        workers = sorted(withheld)
        if len(workers) < 2:
            continue
        dists = [
            jaccard_distance(withheld[a], withheld[b])
            for a, b in itertools.combinations(workers, 2)
        ]
        mean = _mean_pairwise(dists)
        if mean is not None:
            out[rnd] = mean
    return out


# ---------------------------------------------------------------------------
# Cold-start round filtering
# ---------------------------------------------------------------------------

def _confirmatory_rounds(
    all_rounds: list[int],
    min_round: int | None,
    keep_cold_start: bool,
) -> set[int]:
    """Rounds that count toward the mean (protocol §2.5: drop cold start).

    The protocol excludes "round 1 (cold start)".  Reports may be 0- or
    1-indexed depending on the runner; the run-agnostic reading is "drop
    the run's first round".  Resolution order:

    - ``keep_cold_start`` -> keep every round.
    - explicit ``min_round`` -> keep rounds ``>= min_round``.
    - default -> keep every round except the smallest index present.
    """
    rounds = set(all_rounds)
    if not rounds or keep_cold_start:
        return rounds
    if min_round is not None:
        return {r for r in rounds if r >= min_round}
    return rounds - {min(rounds)}


# ---------------------------------------------------------------------------
# Run identity: (arm, seed)
# ---------------------------------------------------------------------------

def parse_arm_seed(run_id: str) -> tuple[str, int | None]:
    """Split a run-dir name into ``(arm, seed)``.

    The spec generator names runs ``<arm>_s<seed>`` (seed = the
    ``dataset.partition.seed`` axis, protocol §4.2), optionally with a
    trailing ``_r<n>`` repeat tag.  The seed must be read from the *raw*
    name first — ``overnight_common.repeat_group`` strips the ``_s<seed>``
    suffix as part of its repeat-group normalization, so calling it first
    would discard the seed.  ``arm`` is then the repeat-group name with the
    seed removed.  When no ``_s<seed>`` suffix is present (e.g. a one-off
    smoke dir), the seed is ``None`` and the arm is the full repeat-group
    name.
    """
    # repeat_group drops a single trailing ``_r<n>`` OR ``_s<n>``; for a
    # combined ``..._s44_r2`` name the seed sits one tag in, so search the
    # raw id for the ``_s<seed>`` token directly.
    seed_match = re.search(r"_s(\d+)(?:_r\d+)?$", run_id)
    seed = int(seed_match.group(1)) if seed_match else None
    arm = repeat_group(run_id)
    if seed is not None:
        arm = _SEED_SUFFIX.sub("", arm)
    return arm, seed


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------

def diagnose_run(
    run_dir: Path,
    min_round: int | None,
    keep_cold_start: bool,
) -> dict[str, Any] | None:
    """Compute D_imp / D_sel for one run dir; None when unreadable.

    Returns a record carrying the scalar means, the per-round arrays, and
    the run identity (arm, seed, run_id).  D_imp uses the cold-start
    filter; D_sel is reported on the same confirmatory round set *and*
    over all rounds (the by-construction differentiation check, §2.5
    item iii, is a property of every round).
    """
    report = load_report(run_dir)
    if report is None:
        logger.warning("skip %s: no readable report.json", run_dir)
        return None
    run_id = run_dir.name
    aggregator = find_aggregator(run_dir)

    manifests = worker_manifests(run_id, report, aggregator)
    workers = sorted({m.source for m in manifests})

    d_imp_rounds = d_imp_per_round(manifests)
    withheld = withheld_sets_per_round(report, aggregator, workers)
    d_sel_rounds = d_sel_per_round(withheld)

    # Confirmatory round set is defined from the rounds where the metric is
    # actually computable; cold-start drop keys off the union so both
    # metrics exclude the same first round.
    observed_rounds = sorted(set(d_imp_rounds) | set(d_sel_rounds))
    conf = _confirmatory_rounds(observed_rounds, min_round, keep_cold_start)

    def _series(per_round: dict[int, float], rounds: set[int]):
        ordered = sorted(r for r in per_round if r in rounds)
        return ordered, [per_round[r] for r in ordered]

    imp_rds, imp_vals = _series(d_imp_rounds, conf)
    sel_rds, sel_vals = _series(d_sel_rounds, conf)
    # D_sel "all rounds" view (the by-construction check is round-wise):
    sel_all_rds = sorted(d_sel_rounds)
    sel_all_vals = [d_sel_rounds[r] for r in sel_all_rds]

    mean_d_imp = float(np.mean(imp_vals)) if imp_vals else None
    mean_d_sel = float(np.mean(sel_vals)) if sel_vals else None
    mean_d_sel_all = float(np.mean(sel_all_vals)) if sel_all_vals else None
    max_d_sel_all = float(np.max(sel_all_vals)) if sel_all_vals else None

    arm, seed = parse_arm_seed(run_id)
    return {
        "run_id": run_id,
        "arm": arm,
        "seed": seed,
        "aggregator": aggregator,
        "n_workers": len(workers),
        "workers": workers,
        "mean_D_imp": mean_d_imp,
        "mean_D_sel": mean_d_sel,
        "mean_D_sel_all_rounds": mean_d_sel_all,
        "max_D_sel_all_rounds": max_d_sel_all,
        "n_rounds_observed": len(observed_rounds),
        "n_rounds_used": len(conf),
        "cold_start_excluded": (
            sorted(set(observed_rounds) - conf) if observed_rounds else []
        ),
        "D_imp_rounds": imp_rds,
        "D_imp_per_round": imp_vals,
        "D_sel_rounds": sel_rds,
        "D_sel_per_round": sel_vals,
        "D_sel_rounds_all": sel_all_rds,
        "D_sel_per_round_all": sel_all_vals,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

#: Scalar columns for CSV output (per-round arrays go to JSON only — CSV
#: stays flat, matching ``posthoc_cost.py``).
_CSV_FIELDS = [
    "run_id",
    "arm",
    "seed",
    "n_workers",
    "n_rounds_observed",
    "n_rounds_used",
    "mean_D_imp",
    "mean_D_sel",
    "mean_D_sel_all_rounds",
    "max_D_sel_all_rounds",
    "cold_start_excluded",
]


def _write_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["cold_start_excluded"] = json.dumps(rec["cold_start_excluded"])
            writer.writerow(row)


def _write_json(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "divergence_diagnostics/v1",
        "metric": "raw_score (delta_sq_norm, G2-frozen trigger utility)",
        "runs": records,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Protocol heterogeneity diagnostics D_imp / D_sel for "
        "H-P5 (writeup/04-phase1/protocol.md §2.5).",
    )
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="Run output dirs (or parents thereof, e.g. "
        "results/overnight/runs).",
    )
    parser.add_argument(
        "--out", default="results/overnight/analysis/divergence.csv",
        help="Output path; .json emits the per-round arrays, anything else "
        "(.csv) emits the flat scalar table.",
    )
    parser.add_argument(
        "--min-round", type=int, default=None,
        help="Keep only rounds >= this when averaging (cold-start drop). "
        "Default: drop each run's first observed round (protocol §2.5).",
    )
    parser.add_argument(
        "--keep-cold-start", action="store_true",
        help="Average over every round, including the run's first "
        "(disables the §2.5 cold-start exclusion).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dirs = find_run_dirs(args.runs)
    if not run_dirs:
        print("ERROR: no run dirs with results/report.json found",
              file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rec = diagnose_run(run_dir, args.min_round, args.keep_cold_start)
        if rec is not None:
            records.append(rec)

    if not records:
        print("ERROR: no readable reports", file=sys.stderr)
        return 2

    records.sort(key=lambda r: (r["arm"], (r["seed"] is None, r["seed"]),
                                r["run_id"]))

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".json":
        _write_json(records, out_path)
    else:
        _write_csv(records, out_path)

    for rec in records:
        di = rec["mean_D_imp"]
        ds = rec["mean_D_sel"]
        print(
            f"{rec['run_id']:<28s} arm={rec['arm']} seed={rec['seed']} "
            f"D_imp={'n/a' if di is None else f'{di:.5f}'} "
            f"D_sel={'n/a' if ds is None else f'{ds:.5f}'} "
            f"(rounds used={rec['n_rounds_used']}/"
            f"{rec['n_rounds_observed']})"
        )
    print(f"wrote {len(records)} runs to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
