#!/usr/bin/env python3
"""n10pilot analyzer — does the shedding accuracy cost shrink at 10 nodes?

4 arms (mono, eps0, drop_eps03, cyclic_k7) x 5 seeds (41-45), 20 rounds,
SERIAL seed-blocked, ACCURACY ONLY (egress-only shaping: no timing claims).

Statistics come exclusively from scripts/analysis_common.py (frozen
conventions: read_run health gate, last-3 at 20 rounds, tci95 sample-SD
intervals, contrast_family = paired t + exact sign-flip + Holm + the
pre-registered non-inferiority UCB).

Integrity is IN-BAND, three channels, printed per seed:
  * canary — eps0's per-node val_accuracy must equal mono's EXACTLY at
    every node and round (equivalence + data-integrity in one bit);
  * hashes — the 4 arms' results/PARTITION_SHA256.txt must be identical
    (byte-identical partitions per seed by construction);
  * zeroscan — campaigns/n10pilot_zeroscan.json (mirror-residue shards,
    detector of record; reclaimed cells are digest-verified-not-shard-
    verified and say so).

The 3-worker anchor comparison is CAMPAIGN-LEVEL (mean±CI vs mean±CI):
partitions differ across scales by construction, so no per-seed pairing
against w3/cb is computed or implied. w3 mono per-seed last-3 values are
printed as operating points next to the pilot's mono absolutes.

Usage:  .venv/bin/python -m scripts.analyze_n10pilot
        (writes campaigns/n10pilot/derived/pilot_table.csv)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import analysis_common as ac  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CAMP = ROOT / "campaigns" / "n10pilot"
W3 = ROOT / "campaigns" / "w3"
ARMS = ("mono", "eps0", "drop_eps03", "cyclic_k7")
SEEDS = (41, 42, 43, 44, 45)

# 3-worker anchors (verified against writeup/23 §1; campaign-level only).
ANCHORS = {
    "drop_eps03": (("w3 drop", -5.27, 2.16), ("cb drop", -5.02, 2.06)),
    "cyclic_k7": (("cp2fix k7", -7.48, 1.29),),
}


def load(arm: str, seed: int) -> dict | None:
    p = CAMP / arm / f"seed{seed}" / "results" / "report.json"
    return json.loads(p.read_text()) if p.exists() else None


def hashes(arm: str, seed: int) -> str | None:
    p = CAMP / arm / f"seed{seed}" / "results" / "PARTITION_SHA256.txt"
    return p.read_text().strip() if p.exists() else None


def canary_exact(mono: dict, eps0: dict) -> bool:
    """eps0 identical to mono at EVERY node and round — no tolerance."""
    for rm, re_ in zip(mono["per_round"], eps0["per_round"]):
        for node, nd in rm["nodes"].items():
            if nd["val_accuracy"] != re_["nodes"][node]["val_accuracy"]:
                return False
    return True


def main() -> int:
    reps = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    missing = [k for k, v in reps.items() if v is None]
    if missing:
        print(f"FATAL: missing reports: {missing}")
        return 1

    print("=" * 74)
    print("n10pilot — 4 arms x 5 seeds, 20 rounds, 10 nodes, serial seed-blocked")
    print("  stats: scripts/analysis_common.py (frozen); accuracy only")
    print("=" * 74)

    # --- integrity, in-band, per seed -------------------------------------
    print("\n§0 INTEGRITY (canary / hash-identity / zeroscan)")
    integrity_ok = True
    for s in SEEDS:
        can = canary_exact(reps[("mono", s)], reps[("eps0", s)])
        hs = {a: hashes(a, s) for a in ARMS}
        hid = all(h is not None and h == hs["mono"] for h in hs.values())
        nn = len(hs["mono"].splitlines()) if hs["mono"] else 0
        integrity_ok &= can and hid
        print(f"  seed{s}: canary={'EXACT' if can else '*** FAIL ***'}  "
              f"hashes={'identical' if hid else '*** MISMATCH ***'} ({nn} nodes)")
    zs = ROOT / "campaigns" / "n10pilot_zeroscan.json"
    if zs.exists():
        scans = json.loads(zs.read_text())["scans"]
        n_corrupt = sum(1 for e in scans.values() if e.get("verdict") == "corrupt")
        print(f"  zeroscan: {len(scans)} mirror-residue shards, {n_corrupt} corrupt "
              f"({zs.name}); reclaimed cells are digest-verified-not-shard-verified")
    if not integrity_ok:
        print("  *** integrity failure above — paired reads below are suspect ***")

    # --- per-run reads, health-gated ---------------------------------------
    reads = {(a, s): ac.read_run(reps[(a, s)], seed=s) for a in ARMS for s in SEEDS}
    unhealthy = [(a, s, r.health.reason) for (a, s), r in reads.items()
                 if not r.health.healthy]
    print(f"\n§1 HEALTH GATE — {20 - len(unhealthy)}/20 healthy"
          + (f"; excluded: {unhealthy}" if unhealthy else ""))

    # --- per-seed table -----------------------------------------------------
    print("\n§2 PER-SEED (last-3 worker-mean val acc, %, healthy runs only)")
    print(f"  {'seed':<6}{'mono':>8}{'eps0':>8}{'drop':>8}{'cyclic':>8}"
          f"{'Δdrop':>8}{'Δcyc':>8}{'Δeps0':>8}")
    rows = []
    diffs: dict[str, list[float]] = {"drop_eps03": [], "cyclic_k7": []}
    seeds_used: dict[str, list[int]] = {"drop_eps03": [], "cyclic_k7": []}
    eps0_diffs = []
    for s in SEEDS:
        vals = {}
        for a in ARMS:
            r = reads[(a, s)]
            vals[a] = 100 * r.lastk if (r.health.healthy and r.lastk is not None) else None
        d_drop = (vals["drop_eps03"] - vals["mono"]
                  if vals["drop_eps03"] is not None and vals["mono"] is not None else None)
        d_cyc = (vals["cyclic_k7"] - vals["mono"]
                 if vals["cyclic_k7"] is not None and vals["mono"] is not None else None)
        d_eps = (vals["eps0"] - vals["mono"]
                 if vals["eps0"] is not None and vals["mono"] is not None else None)
        if d_drop is not None:
            diffs["drop_eps03"].append(d_drop); seeds_used["drop_eps03"].append(s)
        if d_cyc is not None:
            diffs["cyclic_k7"].append(d_cyc); seeds_used["cyclic_k7"].append(s)
        if d_eps is not None:
            eps0_diffs.append(d_eps)
        fmt = lambda v: f"{v:>8.2f}" if v is not None else f"{'--':>8}"
        print(f"  {s:<6}" + "".join(fmt(vals[a]) for a in ARMS)
              + fmt(d_drop) + fmt(d_cyc) + fmt(d_eps))
        rows.append({"seed": s, **{f"acc_{a}": vals[a] for a in ARMS},
                     "d_drop": d_drop, "d_cyclic": d_cyc, "d_eps0": d_eps})

    # --- eps0 identity check (integrity channel, not an inference) ----------
    eps_exact = all(d == 0.0 for d in eps0_diffs) and len(eps0_diffs) == len(SEEDS)
    print(f"\n§3 eps0 IDENTITY: {'+0.00 exactly at all '
          f'{len(eps0_diffs)} seeds — PASS' if eps_exact else '*** NONZERO — FAIL ***'}")

    # --- the pre-registered contrast family ---------------------------------
    print("\n§4 PAIRED CONTRASTS vs mono (Holm family: drop_eps03, cyclic_k7)")
    fam = ac.contrast_family(diffs, seeds_used, direction="less")
    print(ac.format_family(fam))

    # --- campaign-level anchor comparison ------------------------------------
    print("\n§5 CROSS-SCALE (CAMPAIGN-LEVEL, mean±CI vs mean±CI — partitions "
          "differ across scales by construction; NOT a paired contrast)")
    for arm, anchors in ANCHORS.items():
        m, ci, n = ac.mean_ci(diffs[arm])
        for name, am, aci in anchors:
            print(f"  {arm:<12} 10-node {m:+.2f}±{ci:.2f} (n={n})   vs   "
                  f"{name} 3-worker {am:+.2f}±{aci:.2f}")

    # --- operating points -----------------------------------------------------
    print("\n§6 OPERATING POINTS (mono absolute last-3, %; w3 mono = 3-worker)")
    for s in SEEDS:
        w3p = W3 / "mono" / f"seed{s}" / "results" / "report.json"
        w3v = None
        if w3p.exists():
            w3r = ac.read_run(json.loads(w3p.read_text()), seed=s)
            w3v = 100 * w3r.lastk if w3r.lastk is not None else None
        pv = rows[SEEDS.index(s)]["acc_mono"]
        print(f"  seed{s}: pilot {pv:.2f}%" + (f"   w3 {w3v:.2f}%   gap {pv - w3v:+.2f}pp"
              if w3v is not None and pv is not None else "   w3 --"))

    # --- CSV -------------------------------------------------------------------
    out = CAMP / "derived"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "pilot_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path.relative_to(ROOT)}")
    print("provenance: scripts/analyze_n10pilot.py; reads via analysis_common "
          "(last-3 @20r, tci95, health gate); integrity: canary+hashes in-band, "
          "campaigns/n10pilot_zeroscan.json, per-run PARTITION_SHA256.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
