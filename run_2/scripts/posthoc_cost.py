"""Post-hoc $-scoring of completed runs under per-class price vectors (RQ4).

Design doc §7: run cost ``$ = Σ_c bytes_c · κ_c`` is computable from the
already-logged ``layer_comm_metrics`` for any price vector κ without
re-running experiments.  This script scores every given run under

- synthetic ratio vectors ``--ratios R...``: κ_c = R^((C−1−c)/(C−1)) $/GB,
  i.e. geometric interpolation from R on class 0 (the premium/EF pipe) down
  to 1 on the slowest class — for C=1 (monolithic) the single class is
  priced at R;
- and/or explicit vectors ``--kappa name=v0,v1,v2`` (e.g. real cloud egress
  prices when the industry collaborator's numbers arrive).

Cancelled zombie-tail bytes (pre-run fix 3 in the gate doc — queued for
round r but cancelled when round r+1 opened) are read from the report when
present and priced separately: they were *not* transmitted, so they appear
as ``dollars_cancelled`` (the saving the cancellation banked), never inside
``dollars_sent``.  Field shapes tolerated: a per-node ``cancelled_bytes``
scalar, a ``cancelled_bytes_by_class`` dict, or a JSON-string variant of
either under ``cancelled_telemetry_json``.

Output: one CSV row per (run, κ).

Usage::

    python -m scripts.posthoc_cost --runs results/overnight/runs \
        --out results/overnight/analysis/costs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

from scripts.overnight_common import (
    find_aggregator,
    find_run_dirs,
    iter_round_nodes,
    load_node_config,
    load_report,
)

logger = logging.getLogger(__name__)

DEFAULT_RATIOS = (1.0, 2.0, 5.0, 10.0)
_GB = 1e9


def ratio_kappa(ratio: float, num_classes: int) -> list[float]:
    """Geometric per-class price vector in $/GB for a top-to-bottom ratio.

    Class 0 (highest QoS) costs ``ratio``, the last class costs 1.0, and
    intermediate classes interpolate geometrically — the simplest monotone
    family with a single knob, matching the design doc's {1x,2x,5x,10x}
    synthetic-ratio sweep.
    """
    if num_classes <= 1:
        return [float(ratio)]
    return [
        float(ratio) ** ((num_classes - 1 - c) / (num_classes - 1))
        for c in range(num_classes)
    ]


def parse_kappa_arg(raw: str) -> tuple[str, list[float]]:
    """Parse ``name=v0,v1,v2`` into ``(name, [v0, v1, v2])``."""
    if "=" not in raw:
        raise ValueError(f"--kappa expects name=v0,v1,...; got {raw!r}")
    name, _, values = raw.partition("=")
    vec = [float(v) for v in values.split(",") if v.strip()]
    if not name or not vec:
        raise ValueError(f"--kappa expects name=v0,v1,...; got {raw!r}")
    return name, vec


# ---------------------------------------------------------------------------
# Byte extraction
# ---------------------------------------------------------------------------

def sent_bytes_by_class(
    report: dict[str, Any],
    aggregator: str,
    scope: str = "all",
) -> dict[int, int]:
    """Sum ``layer_comm_metrics`` bytes per traffic class across the run.

    ``scope='uplink'`` counts worker nodes only (the aggregator's metrics
    describe the downlink broadcast); ``'all'`` prices total traffic.
    """
    totals: dict[int, int] = {}
    for _round, node_id, entry in iter_round_nodes(report):
        if scope == "uplink" and node_id == aggregator:
            continue
        for lm in entry.get("layer_comm_metrics") or []:
            cls = int(lm.get("traffic_class", 0))
            totals[cls] = totals.get(cls, 0) + int(lm.get("bytes_sent", 0))
    return totals


def cancelled_bytes_by_class(report: dict[str, Any]) -> dict[int, int]:
    """Extract cancelled zombie-tail bytes per class, tolerating shapes.

    Unknown-class scalars are accounted under class ``-1`` so the totals
    stay correct even when the telemetry does not attribute a class.
    """
    totals: dict[int, int] = {}

    def _add(cls: int, n: int) -> None:
        totals[cls] = totals.get(cls, 0) + int(n)

    for _round, _node, entry in iter_round_nodes(report):
        for key in ("cancelled_bytes_by_class", "cancelled_telemetry_json",
                    "cancelled_bytes"):
            value = entry.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                try:
                    value = json.loads(value) if value.strip() else None
                except json.JSONDecodeError:
                    logger.warning("Undecodable %s entry; skipping", key)
                    continue
                if value is None:
                    continue
            if isinstance(value, dict):
                for cls, n in value.items():
                    try:
                        _add(int(cls), int(n))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _add(int(item.get("traffic_class", -1)),
                             int(item.get("bytes", item.get("bytes_cancelled", 0))))
            elif isinstance(value, (int, float)):
                _add(-1, int(value))
            break  # first matching key wins per node entry
    return totals


def dollars(bytes_by_class: dict[int, int], kappa: list[float]) -> float:
    """``Σ_c bytes_c · κ_c`` in $ with bytes priced per GB.

    Classes beyond the κ vector are priced at the *last* (cheapest) rate
    rather than dropped — undercounting silently would bias rankings.  The
    unknown-class bucket (-1) is priced at the cheapest rate too.
    """
    if not kappa:
        return 0.0
    total = 0.0
    for cls, n_bytes in bytes_by_class.items():
        rate = kappa[cls] if 0 <= cls < len(kappa) else kappa[-1]
        total += (n_bytes / _GB) * rate
    return total


def run_num_classes(run_dir: Path, observed: dict[int, int]) -> int:
    """Traffic-class count from the generated node config, else observed."""
    cfg = load_node_config(run_dir, "node-1") or load_node_config(run_dir, "node-0")
    if cfg and cfg.get("num_traffic_classes"):
        return int(cfg["num_traffic_classes"])
    valid = [c for c in observed if c >= 0]
    return max(valid) + 1 if valid else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score completed runs under per-class $/GB price vectors."
    )
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run output dirs (or parents thereof).")
    parser.add_argument("--ratios", type=float, nargs="*",
                        default=list(DEFAULT_RATIOS),
                        help="Synthetic top/bottom price ratios "
                        "(geometric per-class interpolation).")
    parser.add_argument("--kappa", action="append", default=[],
                        metavar="NAME=V0,V1,...",
                        help="Explicit price vector in $/GB, class order; "
                        "repeatable.")
    parser.add_argument("--scope", choices=["all", "uplink"], default="all",
                        help="Which traffic to price (default: all nodes).")
    parser.add_argument("--out", default="results/overnight/analysis/costs.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_dirs = find_run_dirs(args.runs)
    if not run_dirs:
        print("ERROR: no run dirs with results/report.json found", file=sys.stderr)
        return 2
    explicit = [parse_kappa_arg(raw) for raw in args.kappa]

    max_classes = 0
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(run_dirs):
        report = load_report(run_dir)
        if report is None:
            continue
        aggregator = find_aggregator(run_dir)
        sent = sent_bytes_by_class(report, aggregator, scope=args.scope)
        cancelled = cancelled_bytes_by_class(report)
        num_classes = run_num_classes(run_dir, sent)
        max_classes = max(max_classes, num_classes)

        kappas: list[tuple[str, list[float]]] = [
            (f"ratio_{ratio:g}x", ratio_kappa(ratio, num_classes))
            for ratio in args.ratios
        ] + explicit

        note = "" if sent else "no layer_comm_metrics (monolithic or empty run)"
        for name, kappa in kappas:
            d_sent = dollars(sent, kappa)
            d_cancelled = dollars(cancelled, kappa)
            rows.append({
                "run_id": run_dir.name,
                "scope": args.scope,
                "kappa_name": name,
                "kappa_per_class": json.dumps(kappa),
                "bytes_total": sum(sent.values()),
                "cancelled_bytes_total": sum(cancelled.values()),
                "dollars_sent": round(d_sent, 9),
                "dollars_cancelled": round(d_cancelled, 9),
                "dollars_sent_plus_cancelled": round(d_sent + d_cancelled, 9),
                "note": note,
                "_sent": sent,
                "_cancelled": cancelled,
            })

    if not rows:
        print("ERROR: no readable reports", file=sys.stderr)
        return 2

    fieldnames = [
        "run_id", "scope", "kappa_name", "kappa_per_class",
        "bytes_total", "cancelled_bytes_total",
        "dollars_sent", "dollars_cancelled", "dollars_sent_plus_cancelled",
    ]
    fieldnames += [f"bytes_c{c}" for c in range(max_classes)]
    fieldnames += [f"cancelled_c{c}" for c in range(max_classes)]
    fieldnames.append("note")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sent = row.pop("_sent")
            cancelled = row.pop("_cancelled")
            for c in range(max_classes):
                row[f"bytes_c{c}"] = sent.get(c, 0)
                row[f"cancelled_c{c}"] = cancelled.get(c, 0)
            writer.writerow(row)

    print(f"wrote {len(rows)} rows ({len(run_dirs)} runs) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
