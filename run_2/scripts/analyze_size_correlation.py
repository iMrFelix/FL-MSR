#!/usr/bin/env python3
"""Is a layer-selection rule a SIZE heuristic in disguise?

MOTIVATION.  At matched skip COUNT (δ=7 of 14 tensors), `fedluar_hh`'s arms land
at very different byte budgets: `fedluar_d7` at Comm_up 0.2701 against the blind
arms' 0.50.  Since all three skip the same NUMBER of tensors, the only way that
happens is if FedLUAR's ‖Δ_l‖/‖θ_l‖ metric preferentially selects the *large*
layers.  This script measures that directly.

METHOD.  `layer_comm_metrics` records only layers that were SENT, so a layer
absent from a round's metrics was skipped.  Per layer we compute skip frequency
over rounds ≥ r0, WORKERS ONLY (the aggregator broadcasts the full model and
never skips, so including it dilutes every frequency by the worker fraction —
a first pass of this analysis reported 0.375 instead of 0.500 for exactly that
reason).  Then Spearman ρ between layer size in bytes and skip frequency.

TWO BUILT-IN CORRECTNESS CHECKS, both of which must pass before the ρ means
anything:
  * mean skip frequency must equal δ/L for every arm (0.500 at δ=7, L=14).
    If it does not, the sent/skipped reconstruction is wrong.
  * importance-BLIND arms must show ρ ≈ 0. They are the control; a large |ρ|
    there means the method is measuring something other than what it claims.

INDEPENDENCE NOTE.  This contrast was designed and first run by the same agent,
which violates the project's fresh-eyes rule.  The result is arithmetically
self-checking (see the predicted-vs-measured line the script prints), but it
should be reproduced by someone who did not design it before it is cited.

Usage:  python -m scripts.analyze_size_correlation [campaigns/fedluar_hh] \\
            [--arms fedluar_d7,luarand_d7,luarcyc_d7] [--payload-share 0.984]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import analysis_common as ac  # noqa: E402

LARGE_BYTES = 100_000        # deep_cnn: conv kernels 147.5 kB vs biases ≤ 7 kB


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def analyse(campaign: Path, arm: str, r0: int = 3):
    files = sorted(glob.glob(str(campaign / arm / "seed*/results/report.json")))
    sizes: dict[str, int] = {}
    for p in files:                       # pass 1 — learn the FULL layer set
        for rd in json.loads(Path(p).read_text()).get("per_round", []):
            for nd in rd.get("nodes", {}).values():
                for e in nd.get("layer_comm_metrics") or ():
                    if e.get("layer_name") and e.get("bytes_sent"):
                        sizes[e["layer_name"]] = int(e["bytes_sent"])
    skip: dict[str, int] = defaultdict(int)
    tot: dict[str, int] = defaultdict(int)
    for p in files:                       # pass 2 — count, workers only
        for rd in json.loads(Path(p).read_text()).get("per_round", []):
            if (rd.get("round") or 0) < r0:
                continue
            nodes = rd.get("nodes", {})
            _agg, workers = ac.split_roles(nodes)
            for nid, nd in nodes.items():
                if nid not in workers:
                    continue
                lcm = nd.get("layer_comm_metrics")
                if not lcm:
                    continue
                sent = {e.get("layer_name") for e in lcm}
                for nm in sizes:
                    tot[nm] += 1
                    if nm not in sent:
                        skip[nm] += 1
    return sizes, {n: skip[n] / tot[n] for n in sizes if tot[n]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", nargs="?", default=str(ROOT / "campaigns/fedluar_hh"))
    ap.add_argument("--arms", default="fedluar_d7,luarand_d7,luarcyc_d7")
    ap.add_argument("--payload-share", type=float, default=0.984,
                    help="fraction of uplink bytes held by the LARGE layers")
    a = ap.parse_args()
    campaign = Path(a.campaign)

    print(f"{'arm':<16}{'rho(bytes,skip)':>18}{'skip LARGE':>12}{'skip SMALL':>12}"
          f"{'mean':>8}  layers")
    rows = {}
    for arm in a.arms.split(","):
        sizes, sk = analyse(campaign, arm)
        if not sk:
            print(f"{arm:<16}{'(no data)':>18}")
            continue
        names = list(sk)
        rho = spearman([sizes[n] for n in names], [sk[n] for n in names])
        big = [n for n in names if sizes[n] > LARGE_BYTES]
        sml = [n for n in names if sizes[n] <= LARGE_BYTES]
        mb = st.mean([sk[n] for n in big]) if big else float("nan")
        ms = st.mean([sk[n] for n in sml]) if sml else float("nan")
        mean = st.mean(list(sk.values()))
        rows[arm] = (rho, mb, ms, mean)
        print(f"{arm:<16}{rho:>+18.3f}{mb:>12.3f}{ms:>12.3f}{mean:>8.3f}  {len(names)}")

    print("\nCORRECTNESS CHECKS")
    print("  * mean must equal delta/L (0.500 at delta=7, L=14) for EVERY arm —"
          "\n    otherwise the sent/skipped reconstruction is broken.")
    print("  * blind arms (luarand/luarcyc) must show rho ~ 0 — they are the "
          "control.")

    print("\nDOES SIZE-CORRELATED SKIPPING EXPLAIN THE BYTE BUDGET?")
    for arm, (rho, mb, ms, _mean) in rows.items():
        pred = a.payload_share * (1 - mb) + (1 - a.payload_share) * (1 - ms)
        print(f"  {arm:<16} predicted Comm_up = {a.payload_share:.3f}*(1-{mb:.3f})"
              f" + {1 - a.payload_share:.3f}*(1-{ms:.3f}) = {pred:.4f}")
    print("  Compare against the MEASURED Comm_up in the campaign's analyzer "
          "output (§1).\n  Agreement means the byte asymmetry is fully "
          "accounted for by WHICH layers\n  the rule selects, with nothing left "
          "to explain.")
    print("\nSCOPE: architecture-specific. On deep_cnn five identically-sized "
          "conv kernels\nhold ~98.4% of the payload, which is exactly the "
          "geometry that lets a\nsize-correlated metric look like a byte-saving "
          "one. Do not generalise without\nre-running on the target "
          "architecture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
