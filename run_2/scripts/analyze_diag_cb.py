#!/usr/bin/env python3
"""Analyze the collapse-diagnostic (diag) + competent-baseline (cb) campaigns.

Pull first (single ssh):
    rsync -az -e ssh moltres-claude:fl-framework/campaigns/diag/ campaigns/diag/
    rsync -az -e ssh moltres-claude:fl-framework/campaigns/cb/ campaigns/cb/

Sections:
  1. diag — did the two collapsed 50r seeds survive with exactly one knob
     changed?  (clipnorm=1.0 | lr=0.05; collapse originally at r39 / r44-45.)
     Reports NaN round (if any), final/best val_acc, val_loss trajectory tail.
  2. cb 20r — the triptych under the fixed baseline (clipnorm + FedAvgM 0.9 +
     cosine + workers_only): paired dACC vs cb-mono (same campaign, same
     stack), n=6, with BOTH endpoint and last-3-mean reads.
  3. cb 50r — horizon trend + collapse census under the fixed baseline.
  4. Cross-references: cb-mono vs w3-mono (how much does the fixed baseline
     lift the baseline itself?) and vs the central ceiling
     (campaigns/central/*.json).

Reads and statistics come from scripts/analysis_common so every analyzer shares
one estimator (writeup/19 S6/S7/S15): t-based CI with the sample sd, last-k
window reads, and the run-health gate — a collapsed run is excluded from the
paired means and listed in the census rather than averaged in.

Outputs -> campaigns/cb/derived/*.csv + printed summary.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/x.py` as well as `-m scripts.x`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import analysis_common as ac

ROOT = Path(__file__).resolve().parent.parent
WORKERS = {"node-1", "node-2", "node-3"}


def load(p):
    return json.load(open(p))


def seed_of(p):
    return int(Path(p).parent.parent.name.replace("seed", ""))


def val_series(rep):
    """round -> worker-mean val_accuracy (and val_loss)."""
    _rounds, acc, loss = ac.val_series(rep, WORKERS)
    return acc, loss


def nan_round(loss):
    return ac.first_bad_round(loss)


def endpoint(acc):
    xs = [a for a in acc if not math.isnan(a)]
    return xs[-1] if xs else None


def last3(acc):
    return ac.last_k(acc, 3)


def health(acc, loss, k=3):
    """Run-health verdict; a failing run is excluded from every paired mean."""
    return ac.run_health(acc, loss, k)


#: t-based 95% CI half-width, SAMPLE sd (exact Student-t, not a lookup table)
tci95 = ac.tci95


def main(root=ROOT, der=None):
    root = Path(root)
    # ---- 1. diag ----
    print("=== 1. Collapse diagnostics (50r, single-knob reruns) ===")
    print("  original collapses: mono/seed41 NaN@r39, recycle_aging/seed42 NaN@r44-45")
    for arm in ["clip_mono", "clip_aging", "lr005_mono", "lr005_aging"]:
        ps = sorted(glob.glob(str(root / f"campaigns/diag/{arm}/seed*/results/report.json")))
        if not ps:
            print(f"  {arm:<12} (no report yet)")
            continue
        acc, loss = val_series(load(ps[0]))
        nr = nan_round(loss)
        verdict = f"COLLAPSED @r{nr}" if nr is not None else "SURVIVED"
        print(f"  {arm:<12} {verdict}  final={endpoint(acc):.4f} "
              f"best={max(a for a in acc if not math.isnan(a)):.4f} "
              f"last3={last3(acc):.4f}")

    # ---- 2. cb 20r triptych ----
    print("\n=== 2. Competent-baseline triptych (20r, n=6, paired vs cb-mono) ===")
    mono20, rows, census = {}, [], []
    for p in sorted(glob.glob(str(root / "campaigns/cb/mono_r20/seed*/results/report.json"))):
        acc, loss = val_series(load(p))
        h = health(acc, loss)
        census.append(("mono_r20", seed_of(p), h.healthy, h.reason))
        if h.healthy:
            mono20[seed_of(p)] = (endpoint(acc), last3(acc))
        else:
            print(f"  !! cb-mono/seed{seed_of(p)} EXCLUDED ({h.reason}) — its pairs drop")
    print(f"  cb-mono 20r seeds: {sorted(mono20)}  "
          f"mean acc={st.mean([v[0] for v in mono20.values()]):.4f}"
          if mono20 else "  cb-mono 20r: no data yet")
    for arm in ["drop_eps03_r20", "recycle_eps03_r20"]:
        d_end, d_l3 = [], []
        for p in sorted(glob.glob(str(root / f"campaigns/cb/{arm}/seed*/results/report.json"))):
            s = seed_of(p)
            if s not in mono20:
                continue
            acc, loss = val_series(load(p))
            h = health(acc, loss)
            census.append((arm, s, h.healthy, h.reason))
            if not h.healthy:
                print(f"  !! {arm}/seed{s} EXCLUDED ({h.reason})")
                continue
            d_end.append((endpoint(acc) - mono20[s][0]) * 100)
            d_l3.append((last3(acc) - mono20[s][1]) * 100)
        if d_end:
            rows.append((arm, len(d_l3), round(st.mean(d_l3), 2),
                         round(tci95(d_l3), 2), round(st.mean(d_end), 2),
                         round(tci95(d_end), 2)))
            print(f"  {arm:<20} n={len(d_l3)}  dACC(last3)={st.mean(d_l3):+.2f}"
                  f"±{tci95(d_l3):.2f}pp  [sensitivity: endpoint "
                  f"{st.mean(d_end):+.2f}±{tci95(d_end):.2f}pp]")
    der = Path(der) if der is not None else root / "campaigns/cb/derived"
    der.mkdir(parents=True, exist_ok=True)
    with open(der / "cb_triptych.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n", "dACC_last3_pp", "t_ci95_pp",
                    "dACC_end_pp", "t_ci95_end_pp"])
        w.writerows(rows)
    with open(der / "cb_health.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "healthy", "reason"])
        w.writerows(census)

    # ---- 3. cb 50r ----
    print("\n=== 3. Competent-baseline 50r (n=3): horizon + collapse census ===")
    for arm in ["mono_r50", "drop_eps03_r50", "recycle_eps03_r50"]:
        for p in sorted(glob.glob(str(root / f"campaigns/cb/{arm}/seed*/results/report.json"))):
            acc, loss = val_series(load(p))
            nr = nan_round(loss)
            tag = f"COLLAPSED @r{nr}" if nr is not None else "ok"
            print(f"  {arm:<20} seed{seed_of(p)}  {tag}  "
                  f"final={endpoint(acc):.4f}  last3={last3(acc):.4f}")

    # ---- 4. cross-references ----
    print("\n=== 4. Baseline lift + ceiling ===")
    # BUGFIX 2026-08-05 (KB verifier E1): this contrast used the ENDPOINT read
    # while every other contrast in this script uses last-k — the estimator the
    # audit retired (S7).  It understated the lift by ~4x (+1.29 -> +5.23) and
    # was the sole quantitative basis for "the competent bundle is mis-tuned".
    # CAVEAT (audit S25): w3 has no `workers_only`, cb has it true, so the two
    # campaigns do not share a partition — this pairing is INDICATIVE ONLY.
    w3mono = {}
    for p in sorted(glob.glob(str(root / "campaigns/w3/mono/seed*/results/report.json"))):
        acc, _ = val_series(load(p))
        w3mono[seed_of(p)] = last3(acc)
    common = sorted(set(mono20) & set(w3mono))
    if common:
        lift = [(mono20[s][1] - w3mono[s]) * 100 for s in common]      # [1] = last3
        lift_end = [(mono20[s][0] - w3mono[s]) * 100 for s in common]  # [0] = endpoint
        print(f"  fixed-baseline mono lift vs w3 mono (20r, {len(common)} seeds, last-3): "
              f"{st.mean(lift):+.2f}±{tci95(lift):.2f}pp "
              f"[endpoint sensitivity: {st.mean(lift_end):+.2f}±{tci95(lift_end):.2f}pp]")
        print("    NB indicative only — w3 and cb differ in `workers_only`, so the "
              "partitions differ (audit S25); this is not a clean paired contrast.")
    for arm in ["fl_matched", "competent"]:
        p = root / f"campaigns/central/{arm}.json"
        if p.exists():
            h = json.load(open(p))["history"]
            best = max(h, key=lambda r: r["test_acc"])
            # Report BOTH: the peak (a legitimate ceiling estimate) and a
            # converged-window mean.  Comparing two recipes on their PEAKS is
            # the S7 failure mode — fl_matched's tail oscillates several points
            # while a cosine-annealed run sits on a flat plateau, so the peak
            # flatters the noisier recipe (KB verifier E2).
            last5 = st.mean([r["test_acc"] for r in h[-5:]])
            print(f"  central {arm:<10} best={best['test_acc']:.4f} @epoch{best['epoch']}"
                  f"   last-5={last5:.4f}   (n=1 run, {len(h)} epochs)")

    print(f"\nDerived -> {der}/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT), help="repo root holding campaigns/")
    ap.add_argument("--derived", default=None)
    a = ap.parse_args()
    main(a.root, a.derived)
