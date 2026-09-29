#!/usr/bin/env python3
"""Per-round paired-ΔACC trajectory + wash-out verdict for a horizon campaign.

Answers the #1 open question (writeup/15 §4): do the ε=0.3 accuracy gaps narrow as
training converges?  Computes, per arm, the paired ΔACC vs same-seed mono, and
prints a NARROWING / FLAT / WIDENING verdict comparing an early to a late window.

Three audited corrections over the first version of this tool (writeup/19 S1):

  * **health gate.** A seed is dropped from EVERY arm when any run at that seed
    has a non-finite loss or reads at/below chance.  The old tool emitted
    "drop_eps03 Δ=+13.66pp NARROWING" purely because mono/seed41 — the paired
    DENOMINATOR — had collapsed to 10.00%.  Seeds missing a run are reported as
    *unavailable*, which is not the same thing as excluded.
  * **windowed read.** Gaps are last-k window means (k=5 at a 50-round horizon),
    not one arbitrary round on a noisy plateau.
  * **inference, not a threshold.** The narrowing verdict is a one-sided paired
    t on the per-seed (late − early) gap changes with Holm across arms and the
    frozen 0.5pp minimum effect (scripts/hypothesis_tests), replacing a ±0.3pp
    threshold applied to a mean of opposite-signed seeds.

Usage:  .venv/bin/python scripts/horizon_trajectory.py [campaigns/h50] [early_round] [-k 5]
        (defaults: campaigns/h50, early=20 — the wave-3 horizon, for a clean 20r→50r read)
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/x.py` as well as `-m scripts.x`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import analysis_common as ac
from scripts import hypothesis_tests as ht

WORKERS = {"node-1", "node-2", "node-3"}
ARMS = ["drop_eps03", "recycle_eps03", "recycle_aging_eps03"]  # vs mono


def per_round(path):
    """(round -> worker-mean global val_accuracy %, health verdict) for one run."""
    rep = json.load(open(path))
    rounds, acc, loss = ac.val_series(rep, WORKERS)
    series = {r: 100 * a for r, a in zip(rounds, acc) if a == a}
    k = ac.default_k(len(acc))
    return series, ac.run_health(acc, loss, k)


def bag(root, arm):
    """seed -> (per-round series, health) for every run of one arm."""
    b = {}
    for f in glob.glob(str(Path(root) / arm / "seed*/results/report.json")):
        b[int(Path(f).parent.parent.name[4:])] = per_round(f)
    return b


def window_gap(arm_series, mono_series, r, k):
    """Mean paired gap over the last-k window ending at round r (None if short)."""
    rs = [rr for rr in range(r - k + 1, r + 1)
          if rr in arm_series and rr in mono_series]
    return st.mean([arm_series[rr] - mono_series[rr] for rr in rs]) if rs else None


def main(root_s="campaigns/h50", early=20, k=None):
    root = Path(root_s)
    mono = bag(root, "mono")
    if not mono:
        print(f"no mono reports under {root} yet — nothing to pair against")
        return 1
    arms = {a: bag(root, a) for a in ARMS}
    maxr = max((r for s, _h in mono.values() for r in s), default=0)
    k = k or (5 if maxr > 20 else 3)

    # ---- health gate: a collapsed run at a seed poisons that seed everywhere ----
    all_seeds = sorted(set(mono) | {s for b in arms.values() for s in b})
    unhealthy = {}
    for s in all_seeds:
        unhealthy[s] = [f"{name}({b[s][1].reason})"
                        for name, b in [("mono", mono)] + list(arms.items())
                        if s in b and not b[s][1].healthy]
    seeds = [s for s in all_seeds if s in mono and not unhealthy[s]]

    print(f"{root} · seeds {all_seeds} · max round {maxr} · window k={k}")
    for s in all_seeds:
        bags = {"mono": mono, **arms}
        miss = [a for a in ["mono"] + ARMS if s not in bags[a]]
        if unhealthy[s]:
            print(f"  !! seed{s} DROPPED from every arm — collapsed runs: "
                  f"{', '.join(unhealthy[s])}")
        elif miss:
            print(f"  .. seed{s} has no run for {miss} (data availability, "
                  f"NOT an outcome-dependent exclusion)")
    print(f"  healthy paired seed set: {seeds} (n={len(seeds)})\n")
    if not seeds:
        print("no healthy seed survives the gate — no verdict is computable")
        return 1

    # ---- per-round trajectory table ----
    rounds = sorted({r for r in [5, 10, 20, 30, 40, maxr] if r <= maxr})
    print(f"Per-round paired ΔACC vs mono (pp), mean over the {len(seeds)} healthy seeds")

    def lbl(a):
        return a.replace("_eps03", "").replace("recycle_aging", "rec+aging")
    print("round " + "".join(f"{lbl(a):>11}" for a in ARMS))
    for r in rounds:
        cells = []
        for a in ARMS:
            ds = [arms[a][s][0][r] - mono[s][0][r] for s in seeds
                  if s in arms[a] and r in arms[a][s][0] and r in mono[s][0]]
            cells.append(f"{st.mean(ds):>11.2f}" if ds else f"{'--':>11}")
        print(f"{r:>5} " + "".join(cells))

    # ---- wash-out verdict on the windowed gaps ----
    print(f"\nPRIMARY wash-out verdict: last-{k} gap at r{early} → r{maxr}, paired, "
          f"strict seed gate (a collapse anywhere drops the seed everywhere)")
    verdicts(arms, mono, {a: seeds for a in ARMS}, early, maxr, k)

    # Sensitivity: the weaker PAIRWISE gate (mono + this arm healthy). It keeps
    # more seeds but the seed set differs per arm, so a cross-arm comparison
    # under it is not paired — report it, never headline it (S1).
    pairwise = {a: [s for s in all_seeds
                    if s in mono and s in arms[a]
                    and mono[s][1].healthy and arms[a][s][1].healthy]
                for a in ARMS}
    if any(sorted(pairwise[a]) != sorted([s for s in seeds if s in arms[a]]) for a in ARMS):
        print(f"\nSENSITIVITY (pairwise gate — arm-dependent seed sets, NOT cross-arm paired):")
        verdicts(arms, mono, pairwise, early, maxr, k)

    print("\n(NARROWING ⇒ accuracy cost is a transient the horizon washes out — rescues the "
          "non-inferiority framing. WIDENING/INCONCLUSIVE ⇒ a real standing cost; lean on "
          "time+certification. A mean of opposite-signed seeds is INCONCLUSIVE, not a direction.)")
    return 0


def verdicts(arms, mono, seeds_by_arm, early, maxr, k):
    """Per-arm narrowing verdict over one seed selection, Holm across the arms."""
    changes, gaps = {}, {}
    for a in ARMS:
        ch, ge, gl, used = [], [], [], []
        for s in seeds_by_arm.get(a, []):
            if s not in arms[a] or s not in mono:
                continue
            e = window_gap(arms[a][s][0], mono[s][0], early, k)
            l = window_gap(arms[a][s][0], mono[s][0], maxr, k)
            if e is None or l is None:
                continue
            ch.append(l - e); ge.append(e); gl.append(l); used.append(s)
        if ch:
            changes[a] = ch
            gaps[a] = (st.mean(ge), st.mean(gl), used)

    # Holm within each direction's family of arms — the widening verdict is a
    # test too, so it carries the same multiplicity cost as the narrowing one.
    testable = [a for a in changes if len(changes[a]) >= 2]
    adj, adj_w = {}, {}
    if testable:
        for direction, into in (("greater", adj), ("less", adj_w)):
            ps = [ht.paired_t_one_sided(changes[a], direction).p for a in testable]
            into.update(zip(testable, ht.holm(ps, testable).adj_p))
    for a in ARMS:
        if a not in changes:
            print(f"  {a:22s} insufficient data")
            continue
        ge, gl, used = gaps[a]
        ch = changes[a]
        sign = sum(1 for x in ch if x > 0)
        head = (f"  {a:22s} n={len(ch)} seeds={used}  r{early}={ge:+.2f}  "
                f"r{maxr}={gl:+.2f}  Δ={st.mean(ch):+.2f}")
        if a not in adj:
            print(f"{head}pp  NOT EVALUABLE (n<2 healthy pairs — no interval, no verdict)")
            continue
        v_nar = ht.directional_verdict(ch, "greater", ac.MIN_EFFECT_PP, label=a,
                                       p_for_verdict=adj[a])
        v_wid = ht.directional_verdict(ch, "less", ac.MIN_EFFECT_PP, label=a,
                                       p_for_verdict=adj_w[a])
        verdict = ("NARROWING" if v_nar.verdict != ht.NOT_SUPPORTED else
                   "WIDENING" if v_wid.verdict != ht.NOT_SUPPORTED else "INCONCLUSIVE")
        print(f"{head} ±{ac.tci95(ch):.2f}pp  holm p(narrow/widen)="
              f"{adj[a]:.4f}/{adj_w[a]:.4f}  {sign}/{len(ch)} seeds narrow  {verdict}")
        print(f"      per-seed Δ: {[round(x, 2) for x in ch]}")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "campaigns/h50"
    e = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    kk = int(sys.argv[3]) if len(sys.argv) > 3 else None
    raise SystemExit(main(a, e, kk))
