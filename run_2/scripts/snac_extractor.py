"""Shed-Normalized Accuracy Cost (SNAC) extractor — protocol §2.3.

The Phase-2 protocol (``writeup/04-phase1/protocol.md`` §2.3) defines SNAC to
de-confound the RQ3 accuracy comparison (dossier limitation 4): different
schedulers shed different byte mass at the same epsilon, so a raw accuracy
delta conflates *how much* fresh signal was withheld with *which* signal.
SNAC divides the accuracy cost by the realized withheld-fresh-mass fraction.

Per arm ``a`` (one per non-mono run dir), seed ``s``:

- ``mu_bar(a,s)``  — withheld-fresh-mass fraction: mean over (sender i,
  round r) of ``sum_{L in M(i,r)} size_L / sum_{L in model} size_L``, where
  ``M(i,r)`` is the set of layers whose *fresh* round-r update from sender i
  did **not** enter the aggregate.  Source of truth is the per-round
  ``_aggregation.layers[L].arrived_sources`` block: a layer L is in M(i,r)
  iff ``i not in arrived_sources(L, r)``.  This single rule covers every
  reason a fresh update is missing — trigger-shed, skip-advised,
  strategy-omitted (cyclic), and watchdog-missing — because all four end the
  same way: i is absent from that layer's arrived set (protocol §2.3).
- ``ACC(a,s)``      — mean val_accuracy [pp] over the final 5 evaluation
  rounds, averaged across worker nodes (protocol §2.2).
- ``Delta_ACC(a,s) = ACC(mono,s) - ACC(a,s)`` [pp], seed-paired.
- ``SNAC(a,s)     = Delta_ACC(a,s) / mu_bar(a,s)``; ``SNAC(a)`` = mean over s.
- ``beta_bar``     — trigger-shed byte fraction from ``shed_layers`` only
  (secondary normalizer, reported not hypothesis-bearing).
- ``kappa_bar``    — mean TOTAL shed mass over flows (secondary
  normalizer): slippage plus deliberately-omitted, recycle-filled mass.
- ``kappa_slip_bar`` — the slippage half alone (audit BYTE-04).  ``kappa_bar``
  is NOT comparable against epsilon or FedLUAR's kappa<1/16; this one is.

Layer sizes are static per model, so the model-wide ``{layer: size}`` map is
built once from ``layer_comm_metrics.bytes_sent`` across every (round, sender)
in the run: a layer a given sender did not transmit a given round (it was
shed) gets its size from any round/sender that did transmit it.

Reuses ``scripts/overnight_common.py`` for report loading, node iteration,
telemetry decoding, and aggregator resolution.  This script owns only the
SNAC-specific aggregation (M-set construction, mu_bar, seed pairing) and the
no-dependency stats — numpy-only; scipy is wrapped defensively because it is
not guaranteed present (the n=5 paired contrasts in the protocol use exact
permutation/sign tests, not implemented here — this script produces the
per-(arm,seed) estimands the paired-test driver consumes).

Usage::

    python -m scripts.snac_extractor \
        --runs results/p3/runs \
        --mono-runs results/p3/runs/mono_s42 ... \
        --out results/p3/analysis/snac.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from scripts import analysis_common as ac
from scripts.overnight_common import (
    find_aggregator,
    find_run_dirs,
    iter_round_nodes,
    load_node_config,
    load_report,
    parse_uplink_telemetry,
)

# scipy is optional and frequently absent on this host; nothing below needs
# it (the protocol's n=5 contrasts use exact permutation tests, computed by
# the separate paired-test driver), but guard the import so a future addition
# degrades instead of crashing the extractor.
try:  # pragma: no cover - defensive
    import scipy  # noqa: F401

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - defensive
    _HAVE_SCIPY = False

logger = logging.getLogger(__name__)

MONO_ARM = "mono"
#: Number of trailing evaluation rounds averaged into ACC (protocol §2.2).
FINAL_EVAL_ROUNDS = 5
_SEED_SUFFIX = re.compile(r"_s(\d+)$")


# ---------------------------------------------------------------------------
# Per-run primitives
# ---------------------------------------------------------------------------


def run_seed(run_dir: Path, report: dict[str, Any], aggregator: str) -> int | None:
    """Recover the run-level seed used for pairing.

    Prefer the generated node config (``seed`` / ``training.seed``, the axis
    the spec generators vary — ``dataset.partition.seed`` forwarded to the
    node RNGs), since that is the authoritative value.  Fall back to the
    ``_s<NN>`` run-dir suffix (generator naming convention,
    ``scripts/gen_campaign_c1.py``) so the extractor still pairs runs whose
    configs were not retained.
    """
    for node in (_first_worker_id(report, aggregator), "node-1", "node-0"):
        if node is None:
            continue
        cfg = load_node_config(run_dir, node)
        if not cfg:
            continue
        for path in (("seed",), ("training", "seed"), ("dataset", "partition", "seed")):
            cur: Any = cfg
            for key in path:
                cur = cur.get(key) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if cur is not None:
                try:
                    return int(cur)
                except (TypeError, ValueError):
                    pass
    m = _SEED_SUFFIX.search(run_dir.name)
    return int(m.group(1)) if m else None


def _first_worker_id(report: dict[str, Any], aggregator: str) -> str | None:
    for _round, node_id, _entry in iter_round_nodes(report):
        if node_id != aggregator:
            return node_id
    return None


def worker_nodes(report: dict[str, Any], aggregator: str) -> list[str]:
    """All non-aggregator node ids that ever report val_accuracy."""
    nodes: set[str] = set()
    for _round, node_id, entry in iter_round_nodes(report):
        if node_id == aggregator:
            continue
        if entry.get("val_accuracy") is not None:
            nodes.add(node_id)
    return sorted(nodes)


def model_layer_sizes(report: dict[str, Any], aggregator: str) -> dict[str, int]:
    """Model-wide ``{layer: byte_size}`` map (sizes are static per model).

    Built from every worker's ``layer_comm_metrics.bytes_sent`` across all
    rounds: the first strictly-positive ``bytes_sent`` seen for a layer wins
    (a shed layer in one (round, sender) gets its size from another that
    transmitted it).  Aggregator entries are skipped — their metrics describe
    the downlink broadcast, not an uplink manifest.

    Layers that appear only in ``_aggregation.layers`` (never transmitted by
    any logged sender) are added with size 0 and logged: they still belong to
    the model denominator's layer set, but contribute no measurable mass.
    """
    sizes: dict[str, int] = {}
    for _round, node_id, entry in iter_round_nodes(report):
        if node_id == aggregator:
            continue
        for lm in entry.get("layer_comm_metrics") or []:
            name = lm.get("layer_name")
            if not name:
                continue
            nbytes = int(lm.get("bytes_sent", 0) or 0)
            if nbytes > 0 and sizes.get(name, 0) <= 0:
                sizes[name] = nbytes
            sizes.setdefault(name, 0)
    # Fold in any layer the aggregation block names but no sender transmitted.
    for _round, agg in _aggregation_blocks(report, aggregator):
        for name in (agg.get("layers") or {}):
            if name not in sizes:
                logger.warning(
                    "layer %r seen in _aggregation but never transmitted; "
                    "size defaults to 0",
                    name,
                )
                sizes[name] = 0
    return sizes


def _aggregation_blocks(
    report: dict[str, Any], aggregator: str
) -> list[tuple[int, dict[str, Any]]]:
    """Yield ``(round, _aggregation_block)`` from the aggregator telemetry."""
    out: list[tuple[int, dict[str, Any]]] = []
    for round_num, node_id, entry in iter_round_nodes(report):
        if node_id != aggregator:
            continue
        telemetry = parse_uplink_telemetry(entry)
        if not telemetry:
            continue
        agg = telemetry.get("_aggregation")
        if isinstance(agg, dict):
            out.append((round_num, agg))
    return out


def _arrived_sources_by_round(
    report: dict[str, Any], aggregator: str
) -> dict[int, dict[str, set[str]]]:
    """``round -> {layer: set(arrived source ids)}`` from ``_aggregation``."""
    by_round: dict[int, dict[str, set[str]]] = {}
    for round_num, agg in _aggregation_blocks(report, aggregator):
        layers = agg.get("layers") or {}
        per_layer: dict[str, set[str]] = {}
        for name, info in layers.items():
            arrived = (info or {}).get("arrived_sources") or []
            per_layer[name] = {str(s) for s in arrived}
        by_round[round_num] = per_layer
    return by_round


def _shed_layers_by_round(
    report: dict[str, Any], aggregator: str
) -> dict[int, dict[str, list[str]]]:
    """``round -> {source: shed_layers list}`` from per-source telemetry."""
    by_round: dict[int, dict[str, list[str]]] = {}
    for round_num, node_id, entry in iter_round_nodes(report):
        if node_id != aggregator:
            continue
        telemetry = parse_uplink_telemetry(entry)
        if not telemetry:
            continue
        per_source: dict[str, list[str]] = {}
        for source, td in telemetry.items():
            if source.startswith("_") or not isinstance(td, dict):
                continue
            per_source[str(source)] = [str(x) for x in (td.get("shed_layers") or [])]
        by_round[round_num] = per_source
    return by_round


def _kappa_values(
    report: dict[str, Any], aggregator: str,
) -> tuple[list[float], list[float]]:
    """(total, slippage) shed-mass values over (round, source) flows.

    Two lists, not one: ``kappa_realized`` is TOTAL shed mass — slippage plus
    the mass the sender was advised to omit and the aggregator recycle-fills —
    so it is not the quantity ε bounds and must not be read as such (audit
    TRIG-2/BYTE-04).  The slippage list goes through
    ``analysis_common.kappa_split``, which reads the engine's ``kappa_slip``
    when present, then the exact layer-count ends, then the reconstruction from
    the sender's own manifest scores; flows none of the three settles contribute
    to the total only.
    """
    totals: list[float] = []
    slips: list[float] = []
    for round_entry in report.get("per_round", []):
        nodes = round_entry.get("nodes", {})
        entry = nodes.get(aggregator)
        if not isinstance(entry, dict):
            continue
        telemetry = parse_uplink_telemetry(entry)
        if not telemetry:
            continue
        for source, td in telemetry.items():
            if source.startswith("_") or not isinstance(td, dict):
                continue
            # The sender's importance log is on ITS entry in the same round.
            split = ac.kappa_split(
                td, ac.sender_importance(nodes.get(source) or {}))
            if split is None:
                continue
            if split.conflated is not None:
                totals.append(float(split.conflated))
            if split.resolved and split.slip is not None:
                slips.append(float(split.slip))
    return totals, slips


# ---------------------------------------------------------------------------
# Estimands
# ---------------------------------------------------------------------------


def acc_final(report: dict[str, Any], aggregator: str) -> float | None:
    """Mean val_accuracy [pp] over the final ``FINAL_EVAL_ROUNDS`` evals.

    Per round, average val_accuracy across worker nodes (the protocol
    averages over worker nodes; the aggregator's own val row is excluded so
    monolithic and per-layer arms use the identical estimator).  Then average
    over the last few rounds that actually carry a worker eval.  Returned in
    percentage points.
    """
    per_round: dict[int, list[float]] = {}
    for round_num, node_id, entry in iter_round_nodes(report):
        if node_id == aggregator:
            continue
        acc = entry.get("val_accuracy")
        if acc is not None:
            per_round.setdefault(round_num, []).append(float(acc))
    if not per_round:
        return None
    round_means = [
        (r, sum(vals) / len(vals)) for r, vals in sorted(per_round.items()) if vals
    ]
    tail = round_means[-FINAL_EVAL_ROUNDS:]
    return 100.0 * sum(v for _r, v in tail) / len(tail)


def mu_bar(
    report: dict[str, Any], aggregator: str, sizes: dict[str, int]
) -> tuple[float, dict[str, Any]]:
    """Withheld-fresh-mass fraction averaged over (sender, round) flows.

    For each round and worker i, ``M(i,r)`` is the set of model layers for
    which i is absent from ``arrived_sources``; a model layer with no
    aggregation entry that round counts as withheld (no source arrived).  The
    flow value is ``sum_{L in M} size_L / total_model_bytes``; mu_bar is the
    mean of that value over all (i, r) flows.

    Returns ``(mu_bar, diagnostics)`` where diagnostics records the flow count
    and total model bytes (0-byte models would make every fraction undefined;
    handled by returning 0.0 with a flag).
    """
    total = float(sum(sizes.values()))
    arrived = _arrived_sources_by_round(report, aggregator)
    workers = worker_nodes(report, aggregator)
    model_layers = list(sizes)

    flow_fractions: list[float] = []
    for round_num in sorted(arrived):
        per_layer = arrived[round_num]
        for i in workers:
            withheld = sum(
                sizes[L]
                for L in model_layers
                if i not in per_layer.get(L, set())
            )
            if total > 0:
                flow_fractions.append(withheld / total)
            else:
                flow_fractions.append(0.0)
    value = sum(flow_fractions) / len(flow_fractions) if flow_fractions else 0.0
    diag = {
        "n_flows": len(flow_fractions),
        "total_model_bytes": int(total),
        "n_model_layers": len(model_layers),
        "n_rounds_with_aggregation": len(arrived),
    }
    return value, diag


def beta_bar(report: dict[str, Any], aggregator: str, sizes: dict[str, int]) -> float:
    """Trigger-shed byte fraction (``shed_layers`` only), mean over flows.

    Distinct from mu_bar: beta counts only layers a sender's *trigger* shed
    (the secondary normalizer of protocol §2.3), whereas mu_bar counts every
    layer that failed to enter the aggregate for any reason.
    """
    total = float(sum(sizes.values()))
    shed = _shed_layers_by_round(report, aggregator)
    fractions: list[float] = []
    for round_num in sorted(shed):
        for _source, layers in shed[round_num].items():
            if total > 0:
                fractions.append(sum(sizes.get(L, 0) for L in layers) / total)
            else:
                fractions.append(0.0)
    return sum(fractions) / len(fractions) if fractions else 0.0


def kappa_bar(
    report: dict[str, Any], aggregator: str,
) -> tuple[float | None, float | None]:
    """Mean TOTAL and mean SLIPPAGE shed mass over flows; None if never logged.

    Reported as a pair because the two answer different questions: the total is
    the withheld-utility normalizer, the slippage is the only one that may be
    put beside ε or FedLUAR's κ<1/16 (BYTE-04).
    """
    totals, slips = _kappa_values(report, aggregator)
    return (_mean(totals), _mean(slips))


# ---------------------------------------------------------------------------
# Per-run record
# ---------------------------------------------------------------------------


def arm_name(run_dir: Path, is_mono: bool) -> str:
    """Arm label = seed-stripped run-dir name (``mono`` for the baseline)."""
    if is_mono:
        return MONO_ARM
    return _SEED_SUFFIX.sub("", run_dir.name)


def extract_run(run_dir: Path, is_mono: bool) -> dict[str, Any] | None:
    """Compute every per-run estimand for one run dir.

    Returns None when the report is missing/unreadable so the caller can skip
    it without aborting the batch.
    """
    report = load_report(run_dir)
    if report is None:
        logger.warning("no readable report under %s; skipping", run_dir)
        return None
    aggregator = find_aggregator(run_dir)
    sizes = model_layer_sizes(report, aggregator)
    mu, diag = mu_bar(report, aggregator, sizes)
    acc = acc_final(report, aggregator)
    kappa_total, kappa_slip = kappa_bar(report, aggregator)
    return {
        "run_id": run_dir.name,
        "arm": arm_name(run_dir, is_mono),
        "is_mono": is_mono,
        "seed": run_seed(run_dir, report, aggregator),
        "mu_bar": mu,
        "beta_bar": beta_bar(report, aggregator, sizes),
        "kappa_bar": kappa_total,
        "kappa_slip_bar": kappa_slip,
        "acc": acc,
        "_diag": diag,
    }


# ---------------------------------------------------------------------------
# Pairing and per-arm aggregation
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pair_and_score(
    arm_records: list[dict[str, Any]], mono_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Seed-pair each arm record against the mono baseline and aggregate.

    ``Delta_ACC(a,s) = ACC(mono,s) - ACC(a,s)``; ``SNAC = Delta_ACC / mu_bar``.
    A seed present in an arm but absent from the mono set (or vice versa)
    yields a row with null Delta_ACC/SNAC and a note — pairing is never
    silently dropped, matching the protocol's P-4 seed-composition discipline
    (the paired-test driver is responsible for refusing mixed compositions;
    here we surface, not hide, the gap).

    Returns ``(per_seed_rows, per_arm_rows)``.
    """
    mono_acc_by_seed: dict[int, float] = {
        r["seed"]: r["acc"]
        for r in mono_records
        if r["seed"] is not None and r["acc"] is not None
    }

    per_seed: list[dict[str, Any]] = []
    for r in sorted(arm_records, key=lambda r: (r["arm"], r["seed"] is None, r["seed"])):
        seed = r["seed"]
        mono_acc = mono_acc_by_seed.get(seed) if seed is not None else None
        delta = mu = snac = None
        note = ""
        if r["acc"] is None:
            note = "no val_accuracy in arm run"
        elif mono_acc is None:
            note = f"no seed-matched mono (seed={seed})"
        else:
            delta = mono_acc - r["acc"]
            mu = r["mu_bar"]
            if mu and mu > 0:
                snac = delta / mu
            else:
                note = "mu_bar==0 (no withheld mass); SNAC undefined"
        per_seed.append({
            "arm": r["arm"],
            "run_id": r["run_id"],
            "seed": seed,
            "mu_bar": r["mu_bar"],
            "beta_bar": r["beta_bar"],
            "kappa_bar": r["kappa_bar"],
            "kappa_slip_bar": r["kappa_slip_bar"],
            "acc": r["acc"],
            "acc_mono": mono_acc,
            "delta_acc": delta,
            "snac": snac,
            "note": note,
        })

    # Mono arm itself, reported for completeness (Delta_ACC == 0 by definition).
    for r in sorted(mono_records, key=lambda r: (r["seed"] is None, r["seed"])):
        per_seed.append({
            "arm": MONO_ARM,
            "run_id": r["run_id"],
            "seed": r["seed"],
            "mu_bar": r["mu_bar"],
            "beta_bar": r["beta_bar"],
            "kappa_bar": r["kappa_bar"],
            "kappa_slip_bar": r["kappa_slip_bar"],
            "acc": r["acc"],
            "acc_mono": r["acc"],
            "delta_acc": 0.0 if r["acc"] is not None else None,
            "snac": None,
            "note": "" if r["acc"] is not None else "no val_accuracy in mono run",
        })

    # Per-arm means over seeds.
    per_arm: list[dict[str, Any]] = []
    arms = sorted({row["arm"] for row in per_seed})
    for arm in arms:
        rows = [row for row in per_seed if row["arm"] == arm]
        kappas = [row["kappa_bar"] for row in rows if row["kappa_bar"] is not None]
        slips = [row["kappa_slip_bar"] for row in rows
                 if row["kappa_slip_bar"] is not None]
        per_arm.append({
            "arm": arm,
            "n_seeds": len({row["seed"] for row in rows if row["seed"] is not None}),
            "mu_bar_mean": _mean([row["mu_bar"] for row in rows if row["mu_bar"] is not None]),
            "beta_bar_mean": _mean([row["beta_bar"] for row in rows if row["beta_bar"] is not None]),
            "kappa_bar_mean": _mean(kappas) if kappas else None,
            "kappa_slip_bar_mean": _mean(slips) if slips else None,
            "acc_mean": _mean([row["acc"] for row in rows if row["acc"] is not None]),
            "delta_acc_mean": _mean([row["delta_acc"] for row in rows if row["delta_acc"] is not None]),
            "snac_mean": _mean([row["snac"] for row in rows if row["snac"] is not None]),
        })
    return per_seed, per_arm


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_PER_SEED_FIELDS = [
    "arm", "run_id", "seed", "mu_bar", "beta_bar", "kappa_bar", "kappa_slip_bar",
    "acc", "acc_mono", "delta_acc", "snac", "note",
]
_PER_ARM_FIELDS = [
    "arm", "n_seeds", "mu_bar_mean", "beta_bar_mean", "kappa_bar_mean",
    "kappa_slip_bar_mean", "acc_mean", "delta_acc_mean", "snac_mean",
]


def _round_opt(x: Any, ndigits: int) -> Any:
    return round(x, ndigits) if isinstance(x, (int, float)) else x


def write_outputs(
    out_path: Path,
    per_seed: list[dict[str, Any]],
    per_arm: list[dict[str, Any]],
) -> list[Path]:
    """Write per-(arm,seed) and per-arm tables as CSV and/or JSON.

    If ``out_path`` ends in ``.json`` a single JSON object is written; if it
    ends in ``.csv`` the per-seed table is the main CSV and the per-arm means
    go to a sibling ``<stem>_per_arm.csv`` plus a ``<stem>.json`` with both
    tables (so downstream readers get the structured form for free).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    payload = {"per_seed": per_seed, "per_arm": per_arm}

    if out_path.suffix == ".json":
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return [out_path]

    # CSV main + per-arm sibling + JSON companion.
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PER_SEED_FIELDS)
        writer.writeheader()
        for row in per_seed:
            writer.writerow({k: _round_opt(row.get(k), 6) for k in _PER_SEED_FIELDS})
    written.append(out_path)

    arm_path = out_path.with_name(f"{out_path.stem}_per_arm.csv")
    with arm_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PER_ARM_FIELDS)
        writer.writeheader()
        for row in per_arm:
            writer.writerow({k: _round_opt(row.get(k), 6) for k in _PER_ARM_FIELDS})
    written.append(arm_path)

    json_path = out_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    written.append(json_path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract Shed-Normalized Accuracy Cost (SNAC) and its "
        "components per (arm, seed) from completed run reports "
        "(protocol writeup/04-phase1/protocol.md §2.3).",
    )
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Treatment-arm run dirs (or parents thereof), each containing "
        "results/report.json. Required unless --selftest.",
    )
    parser.add_argument(
        "--mono-runs", nargs="+", default=None,
        help="Seed-matched monolithic baseline run dirs (or parents). "
        "Required unless --selftest.",
    )
    parser.add_argument(
        "--out", default="results/p3/analysis/snac.csv",
        help="Output path; .csv writes per-seed CSV + per-arm sibling + JSON "
        "companion, .json writes a single JSON object.",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run the bundled self-test against run_output_t3_resmoke and exit.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.selftest:
        return _selftest()

    if not args.runs or not args.mono_runs:
        parser.error("--runs and --mono-runs are required unless --selftest")

    arm_dirs = find_run_dirs(args.runs)
    mono_dirs = find_run_dirs(args.mono_runs)
    if not arm_dirs:
        print("ERROR: no treatment run dirs with results/report.json", file=sys.stderr)
        return 2
    if not mono_dirs:
        print("ERROR: no mono run dirs with results/report.json", file=sys.stderr)
        return 2

    arm_records = [r for d in arm_dirs if (r := extract_run(d, is_mono=False))]
    mono_records = [r for d in mono_dirs if (r := extract_run(d, is_mono=True))]
    if not arm_records or not mono_records:
        print("ERROR: no readable reports", file=sys.stderr)
        return 2

    per_seed, per_arm = pair_and_score(arm_records, mono_records)
    written = write_outputs(Path(args.out), per_seed, per_arm)

    for row in per_arm:
        logger.info(
            "arm=%s n=%d mu_bar=%s beta_bar=%s kappa_bar=%s kappa_slip=%s "
            "ACC=%s dACC=%s SNAC=%s",
            row["arm"], row["n_seeds"],
            _fmt(row["mu_bar_mean"]), _fmt(row["beta_bar_mean"]),
            _fmt(row["kappa_bar_mean"]), _fmt(row["kappa_slip_bar_mean"]),
            _fmt(row["acc_mean"]),
            _fmt(row["delta_acc_mean"]), _fmt(row["snac_mean"]),
        )
    print(f"wrote {len(per_seed)} per-seed rows, {len(per_arm)} arms to "
          + ", ".join(str(p) for p in written))
    return 0


def _fmt(x: Any) -> str:
    return "n/a" if x is None else f"{x:.4g}"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_SELFTEST_RUN = "run_output_t3_resmoke"


def _selftest() -> int:
    """Validate parsing against the committed FEMNIST skip-feedback smoke.

    The smoke is a 2-round run where every worker arrived for every layer, so
    M(i,r) is empty and mu_bar must be exactly 0.0 — the cleanest invariant
    check that arrived_sources/size parsing is wired correctly (a parsing bug
    that misread arrived_sources or layer sizes would push mu_bar off 0 or out
    of [0,1]).  Also asserts the size map is non-empty and positive and that
    ACC is a sane probability in pp.
    """
    from scripts.overnight_common import REPO_ROOT

    run_dir = REPO_ROOT / _SELFTEST_RUN
    report = load_report(run_dir)
    assert report is not None, f"no report under {run_dir}"
    aggregator = find_aggregator(run_dir)

    sizes = model_layer_sizes(report, aggregator)
    assert sizes, "empty model layer-size map"
    assert all(s >= 0 for s in sizes.values()), "negative layer size"
    total = sum(sizes.values())
    assert total > 0, "zero total model bytes"

    workers = worker_nodes(report, aggregator)
    arrived = _arrived_sources_by_round(report, aggregator)

    mu, diag = mu_bar(report, aggregator, sizes)
    assert 0.0 <= mu <= 1.0, f"mu_bar {mu} outside [0,1]"

    # Cross-check the arrived_sources reading: this smoke has full arrival.
    full_arrival = all(
        set(workers) <= per_layer.get(L, set())
        for per_layer in arrived.values()
        for L in sizes
    )
    if full_arrival:
        assert mu == 0.0, f"full-arrival smoke must give mu_bar 0, got {mu}"

    beta = beta_bar(report, aggregator, sizes)
    assert 0.0 <= beta <= 1.0, f"beta_bar {beta} outside [0,1]"
    kappa, kappa_slip = kappa_bar(report, aggregator)

    acc = acc_final(report, aggregator)
    assert acc is not None and 0.0 <= acc <= 100.0, f"ACC {acc} not a pp prob"
    seed = run_seed(run_dir, report, aggregator)

    largest = max(sizes, key=lambda k: sizes[k])
    print("SELF-TEST PASSED")
    print(f"  run                 : {run_dir.name}")
    print(f"  aggregator          : {aggregator}")
    print(f"  workers             : {workers}")
    print(f"  rounds w/ aggregation: {sorted(arrived)}")
    print(f"  model layers        : {len(sizes)}  total_bytes={total:,}")
    print(f"  largest layer       : {largest} ({sizes[largest]:,} bytes, "
          f"{100*sizes[largest]/total:.1f}% of model)")
    print(f"  full arrival        : {full_arrival}")
    print(f"  mu_bar              : {mu}  (in [0,1]: {0.0 <= mu <= 1.0})")
    print(f"  beta_bar            : {beta}")
    print(f"  kappa_bar (total)   : {kappa}")
    print(f"  kappa_slip (vs eps) : {kappa_slip}")
    print(f"  ACC (pp)            : {acc:.4f}")
    print(f"  seed                : {seed}")
    print(f"  diag                : {json.dumps(diag)}")
    print(f"  scipy available     : {_HAVE_SCIPY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
