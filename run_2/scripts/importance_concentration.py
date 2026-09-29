#!/usr/bin/env python3
"""Measure per-layer importance (‖Δ‖²) concentration — Claim A's load-bearing fact.

The coverage primitive can shed bytes at all only because importance mass
concentrates in a few cheap layers (novelty audit, writeup/15 §1/§3).  This
quantifies that concentration the way the scheduler actually sees it: per
(seed, worker, round) cell, ranking layers by importance-per-byte and reading
off the byte fraction that covers the threshold mass.

Rewritten after writeup/19 S3 / BYTE-06, which found the published "70% of the
mass in 41% of the bytes" to be:
  * one arbitrary sample — the old `worker_layers` returned whichever worker
    came first in JSON dict order, re-picked every round, so the number was a
    chimera over a nondeterministic worker mixture, not reproducible in
    principle;
  * a mean-then-rank aggregation the scheduler never performs;
  * a point estimate with no dispersion, over a grid whose per-cell values
    span 20-61% (the cumulative curve crosses the threshold inside one of five
    identical 147.5 kB kernels, so the statistic is a coarse step function); and
  * non-stationary — it rises steeply through ~r15 and then sits near 53%
    (ratio 1.31×), so the ~1.7× quoted as a constant is a short-horizon value.

So: every report in the arm, every worker, every round; pooled distribution with
its spread, the per-(seed,node) steady means, the per-round trajectory, and the
whole cumulative importance-vs-bytes CURVE per round bucket, so any threshold
can be read off instead of one knife-edge number.

Usage:
    .venv/bin/python scripts/importance_concentration.py campaigns/w3/byte_balanced_eps0
    .venv/bin/python scripts/importance_concentration.py campaigns/h50min/drop_eps03 --r0 0
    (an explicit .../results/report.json path also works — one run, still all workers)
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/x.py` as well as `-m scripts.x`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import analysis_common as ac


def cell_curve(layers):
    """Cumulative (importance-fraction, byte-fraction) curve for one sender-round.

    Layers are ranked by importance-per-byte DESC — the coverage send order.
    Returns [] when the cell carries no importance or no bytes.
    """
    rows = []
    for l in layers:
        im, b = l.get("importance"), l.get("bytes_sent")
        if im is None or not b:
            continue
        rows.append((str(l.get("layer_name")), float(im), float(b)))
    tot_i = sum(r[1] for r in rows)
    tot_b = sum(r[2] for r in rows)
    if not rows or tot_i <= 0 or tot_b <= 0:
        return []
    rows.sort(key=lambda r: -(r[1] / r[2]))
    curve, ci, cb = [], 0.0, 0.0
    for name, im, b in rows:
        ci += im
        cb += b
        curve.append((name, ci / tot_i, cb / tot_b))
    return curve


def cover_fraction(curve, threshold):
    """Byte fraction at which the cumulative importance first reaches threshold."""
    for _name, ci, cb in curve:
        if ci >= threshold - 1e-12:
            return cb
    return None


def collect(paths, r0, threshold):
    """[(seed, node, round, byte_fraction, curve)] over every cell in the arm.

    Runs that failed the health gate are skipped and returned separately: after
    a divergence the importance vector is a readout of the blow-up, not of the
    mechanism, so pooling it into the concentration distribution would repeat
    S11's mistake on the importance axis.
    """
    cells, skipped = [], []
    for p in paths:
        seed_dir = Path(p).parent.parent.name
        seed = int(seed_dir[4:]) if seed_dir.startswith("seed") else -1
        rep = json.load(open(p))
        health = ac.read_run(rep, seed=seed).health
        if not health.healthy:
            skipped.append((seed, health.reason))
            continue
        _agg, workers = ac.report_roles(rep)
        for rd in rep.get("per_round", []):
            if rd.get("round", 0) < r0:
                continue
            for nid, nd in rd.get("nodes", {}).items():
                if nid not in workers:
                    continue
                curve = cell_curve(nd.get("layer_comm_metrics") or [])
                cf = cover_fraction(curve, threshold)
                if cf is not None:
                    cells.append((seed, nid, rd["round"], cf, curve))
    return cells, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", help="campaign arm directory (or one results/report.json)")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="importance mass to cover (default 0.70 = 1−ε at ε=0.3)")
    ap.add_argument("--r0", type=int, default=3, help="first round to include (default 3)")
    ap.add_argument("--bucket", type=int, default=10,
                    help="round-bucket width for the cumulative curves (default 10)")
    a = ap.parse_args(argv)

    arm = Path(a.arm)
    paths = ([str(arm)] if arm.name == "report.json"
             else sorted(glob.glob(str(arm / "seed*/results/report.json"))))
    if not paths:
        print(f"no report.json under {arm}")
        return 1
    cells, skipped = collect(paths, a.r0, a.threshold)
    for seed, why in skipped:
        print(f"  !! seed{seed} EXCLUDED from the grid ({why}) — a diverged run's "
              f"importance vector is not a mechanism measurement")
    if not cells:
        print(f"no usable per-layer importance in {arm} (r>={a.r0}; "
              f"{len(skipped)}/{len(paths)} runs collapsed)")
        return 1

    fr = [c[3] for c in cells]
    mean, ci, n = ac.mean_ci([100 * x for x in fr])
    print(f"{arm} · {len(paths) - len(skipped)}/{len(paths)} healthy runs · {n} "
          f"(seed×worker×round) cells, r>={a.r0} · "
          f"threshold {100 * a.threshold:.0f}% of ‖Δ‖²\n")
    print(f"  byte fraction covering {100 * a.threshold:.0f}% of the mass:")
    print(f"    pooled   mean {mean:.1f}%   median {100 * st.median(fr):.1f}%   "
          f"sd {st.stdev([100 * x for x in fr]) if n > 1 else float('nan'):.1f}   "
          f"range {100 * min(fr):.1f}-{100 * max(fr):.1f}%")

    # per-(seed,node) steady means: the independent replicate unit for a CI
    per_cell = defaultdict(list)
    for seed, nid, _r, cf, _c in cells:
        per_cell[(seed, nid)].append(100 * cf)
    reps = {k: st.mean(v) for k, v in sorted(per_cell.items())}
    m2, ci2, n2 = ac.mean_ci(list(reps.values()))
    print(f"    per-(seed,node) mean {m2:.1f} ± {ci2:.1f}% (t-CI, n={n2} replicates; "
          f"spread {min(reps.values()):.1f}-{max(reps.values()):.1f}%)")
    print(f"    concentration ratio {a.threshold / (m2 / 100):.2f}× "
          f"(1.00× = no concentration)\n")
    print("  per-(seed,node) steady means:")
    for (seed, nid), v in reps.items():
        print(f"    seed{seed} {nid}: {v:.1f}%")

    # per-round trajectory — the statistic is NOT stationary (S3/BYTE-06)
    by_round = defaultdict(list)
    for _s, _n, r, cf, _c in cells:
        by_round[r].append(100 * cf)
    print(f"\n  per-round trajectory (n cells per round, byte% covering "
          f"{100 * a.threshold:.0f}%):")
    print(f"    {'round':>6}{'n':>5}{'mean%':>9}{'median%':>9}{'min%':>7}{'max%':>7}"
          f"{'ratio×':>9}")
    for r in sorted(by_round):
        v = by_round[r]
        mu = st.mean(v)
        print(f"    {r:>6}{len(v):>5}{mu:>9.1f}{st.median(v):>9.1f}{min(v):>7.1f}"
              f"{max(v):>7.1f}{a.threshold / (mu / 100):>9.2f}")

    # the whole cumulative curve per round bucket — read off any threshold
    print(f"\n  cumulative importance-vs-bytes curve, mean over cells per "
          f"{a.bucket}-round bucket:")
    buckets = defaultdict(list)
    for _s, _n, r, _cf, curve in cells:
        buckets[(r // a.bucket) * a.bucket].append(curve)
    for b in sorted(buckets):
        curves = [c for c in buckets[b] if c]
        depth = min(len(c) for c in curves)
        pts = []
        for i in range(depth):
            pts.append((100 * st.mean([c[i][2] for c in curves]),
                        100 * st.mean([c[i][1] for c in curves])))
        print(f"    r{b}-{b + a.bucket - 1} (n={len(curves)}): " +
              "  ".join(f"{by:.0f}%B→{im:.0f}%I" for by, im in pts))
    print("\n  (Each conv kernel is ~19.7% of the byte axis, so this statistic is a "
          "coarse\n   step function — quote the curve, not one threshold crossing. On a "
          "SHEDDING\n   arm the byte axis counts only what was sent, so the concentration "
          "of the\n   model itself is best read off an eps=0 arm.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
