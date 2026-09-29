#!/usr/bin/env python3
"""Analyze the comm-bound regime campaigns (2026-08-05).

WHY THIS CAMPAIGN EXISTS
------------------------
On the standard testbed communication is ~2.5% of a round (wire ~0.8s vs a
~32s round; comm_duration is almost entirely barrier-wait on straggler CPU),
so NO communication-scheduling mechanism can move end-to-end time by more
than ~2.5% — the time claim was unmeasurable, not disproven.  `commbound`
scales every link 30x down (mono 10 -> 0.333 Mbps; per-layer 6/3/1 ->
0.2/0.1/0.033), leaving everything else identical, which puts communication
at ~64% of the round.

WHAT IT FOUND (n=3, paired per seed, t-CIs)
-------------------------------------------
1. GRANULARITY TAX.  Per-layer scheduling at eps=0 is +26.8% comm time vs
   monolithic.  The recorded assignment diagnostics give whole-layer EFT
   makespan / fluid (perfectly-splittable) bound = 1.312 — reproducing, on
   the wire, the 1.31x bound the audit derived analytically from the layer
   size vector (NT-02).
2. THE TRIGGER WORKS.  eps=0.3 vs eps=0 (same machinery, only the deadline
   differs) is -22.4% +- 3.9 comm time.
3. THEY CANCEL.  eps=0.3 vs mono is -1.6% +- 2.8 comm and -1.4% +- 4.8 round
   — indistinguishable from parity — at -5.70 +- 3.54 pp accuracy.
4. THE KNOB SATURATES, AND WE MEASURED THE QUANTUM.  eps=0.3 and eps=0.5 have
   IDENTICAL head makespan (11.80s) though eps=0.5 ships 34% fewer head bytes
   (295.5 vs 445.4 KB); 11.80s = exactly 2.000x the 5.90s needed to push one
   147,536 B conv kernel over the 0.2 Mbps class.  Coverage time is set by how
   many INDIVISIBLE large layers land on the critical class, not by bytes, so
   eps only pays when it removes a whole large layer from the critical path;
   between thresholds it is pure accuracy loss.  This is the same phenomenon
   the audit saw as "t_eps quantized to k x 0.2005s" (NT-04) at 30x higher
   link rates: 5.90s here <-> 0.197s there.

CAVEATS THIS SCRIPT ENFORCES
----------------------------
- Paired per seed, health-gated, t-based CIs (audited estimators).
- `commbound` batches ONE ARM AT A TIME, so measurement occasion is aliased
  onto arm; `commbound2` re-runs seed-blocked (all arms in one batch) and
  `cbserial` runs one stack at a time to remove co-tenancy compute skew
  (train-time spread across identical workers reaches 18s).  Read all three
  before quoting a timing number.

Usage: .venv/bin/python scripts/analyze_commbound.py [--campaign commbound]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import analysis_common as ac

ROOT = Path(__file__).resolve().parent.parent
WORKERS = {"node-1", "node-2", "node-3"}
#: bytes of one conv kernel and the fastest per-layer class, for the quantum
KERNEL_B, FAST_MBPS = 147_536, 0.2


def seed_of(p: str) -> int:
    return int(Path(p).parent.parent.name.replace("seed", ""))


def per_seed(root: Path, arm: str, key: str, r0: int = 3) -> dict[int, float]:
    """Median over worker-rounds (>= r0) of a per-node timing field, per seed."""
    out: dict[int, float] = {}
    for p in sorted(glob.glob(str(root / arm / "seed*/results/report.json"))):
        rep = json.load(open(p))
        v = [float(d[key]) for rd in rep["per_round"] if rd["round"] >= r0
             for n, d in rd["nodes"].items() if n in WORKERS and d.get(key)]
        if v:
            out[seed_of(p)] = st.median(v)
    return out


def accuracy(root: Path, arm: str) -> dict[int, float]:
    """Health-gated last-k accuracy read per seed."""
    out: dict[int, float] = {}
    for p in sorted(glob.glob(str(root / arm / "seed*/results/report.json"))):
        r = ac.read_run(json.load(open(p)), seed=seed_of(p), workers=WORKERS)
        if r.health.healthy:
            out[r.seed] = r.lastk
    return out


def packing(root: Path, arm: str, r0: int = 3):
    """(whole-layer EFT makespan, fluid bound, ratio, head bytes) medians."""
    mk, fl, hb = [], [], []
    for p in sorted(glob.glob(str(root / arm / "seed*/results/report.json"))):
        rep = json.load(open(p))
        for rd in rep["per_round"]:
            if rd["round"] < r0:
                continue
            for n, nd in rd["nodes"].items():
                if n not in WORKERS:
                    continue
                d = ((nd.get("uplink_telemetry") or {}).get("_sender")
                     or {}).get("assignment_diagnostics") or {}
                e, f = d.get("eft_head_loads_s"), d.get("fluid_bound_head_s")
                if e:
                    mk.append(max(float(x) for x in e.values()))
                if f:
                    fl.append(float(f))
                if d.get("head_bytes"):
                    hb.append(float(d["head_bytes"]))
    if not mk:
        return None
    ratio = st.median([a / b for a, b in zip(mk, fl) if b > 0]) if fl else float("nan")
    return st.median(mk), (st.median(fl) if fl else float("nan")), ratio, st.median(hb)


def rel(a: dict[int, float], b: dict[int, float]) -> list[float]:
    return [(a[s] - b[s]) / b[s] * 100 for s in sorted(set(a) & set(b))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="commbound")
    ap.add_argument("--arms", default="mono,eps0,eps03,eps05")
    args = ap.parse_args()
    root = ROOT / "campaigns" / args.campaign
    arms = args.arms.split(",")
    base = arms[0]

    mr, mc, ma = (per_seed(root, base, "round_duration_s"),
                  per_seed(root, base, "comm_duration_s"),
                  accuracy(root, base))
    if not mr:
        print(f"no data for baseline arm {base!r} in {root}")
        return 1

    print(f"=== {args.campaign}: paired vs {base}, health-gated, t-CIs ===")
    print(f"{'arm':<8} {'round vs base':>18} {'comm vs base':>18} {'dACC (pp)':>17}")
    for arm in arms[1:]:
        R, C, A = (per_seed(root, arm, "round_duration_s"),
                   per_seed(root, arm, "comm_duration_s"), accuracy(root, arm))
        if not R:
            print(f"{arm:<8} (no data)")
            continue
        dr, dc = rel(R, mr), rel(C, mc)
        da = [(A[s] - ma[s]) * 100 for s in sorted(set(A) & set(ma))]
        print(f"{arm:<8} {st.mean(dr):+8.1f}% ± {ac.tci95(dr):4.1f}"
              f" {st.mean(dc):+8.1f}% ± {ac.tci95(dc):4.1f}"
              f" {st.mean(da):+8.2f} ± {ac.tci95(da):4.2f}")

    print(f"\n=== packing: whole-layer EFT makespan vs perfectly-splittable bound ===")
    quantum = KERNEL_B * 8 / (FAST_MBPS * 1e6)
    print(f"{'arm':<8} {'makespan':>10} {'fluid':>9} {'tax':>7} {'head KB':>9} {'makespan/quantum':>18}")
    for arm in arms:
        pk = packing(root, arm)
        if not pk:
            continue
        mk, fl, ratio, hb = pk
        print(f"{arm:<8} {mk:9.2f}s {fl:8.2f}s {ratio:7.3f} {hb/1e3:8.1f} {mk/quantum:18.3f}")
    print(f"  quantum = one {KERNEL_B} B kernel over the {FAST_MBPS} Mbps class "
          f"= {quantum:.2f}s")
    print("  makespan/quantum near an INTEGER => coverage time is set by how many\n"
          "  indivisible large layers sit on the critical class, not by bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
