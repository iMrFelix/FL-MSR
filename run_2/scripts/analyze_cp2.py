#!/usr/bin/env python3
"""Analyze the CP2 attribution-control campaign (2026-08-05).

Pull first (single ssh, no polling):
    rsync -az -e ssh moltres-claude:fl-framework/campaigns/cp2/ campaigns/cp2/

Sections:
  0. Determinism sentinels: per-round val trajectories of sentinel_mono/41 and
     sentinel_drop_eps03/41 vs the wave-3 runs of the same configs.  If NOT
     bit-identical, every cross-campaign paired comparison below is flagged
     (different software stack; use within-campaign contrasts only).
  1. Claim-A ordering control (uniform_eps03): paired dACC vs w3 mono and vs
     w3 drop_eps03 + the mechanism metrics (t_eps, head bytes) — blind
     ordering should need MORE bytes/time for the same coverage.
  2. Claim-A selection control (cyclic_k7, byte-matched): paired dACC vs mono
     and drop — if blind rotation matches coverage-scheduling accuracy at the
     same fresh-byte budget, importance selection buys nothing.
  3. Claim-B periodic-refresh sweep (cyclic k2/k3/k7): (uplink MB, dACC)
     points overlaid on the w3 tau_max frontier, which is RECOMPUTED here from
     campaigns/w3 through the same code path (analyze_wave3.frontier_table) —
     never read back from the committed derived/frontier.csv, which predates
     the audit fixes and would put an endpoint-read, un-gated, aliased-window
     dACC in the same table as a last-3, health-gated, cycle-aligned one
     (S6/S11).

Read semantics are the audited ones (writeup/19), from scripts/analysis_common:
last-3 reads not the endpoint round (S7), collapsed runs excluded from EVERY
mean — accuracy, bytes, traffic and timing alike — and reported in the census
(S1/S11), t-based CIs with the sample sd (S6/ML-05), Holm + the frozen
non-inferiority UCB via scripts/hypothesis_tests (S5/S9/S15), cyclic byte
windows aligned to the arm's own period k (BYTE-02) with the downlink reported
alongside (BYTE-03), and kappa split into coverage slippage vs deliberately
shed mass (BYTE-04).

Outputs -> campaigns/cp2/derived/*.csv + printed verdict.
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
from scripts import analyze_wave3 as w3a

ROOT = Path(__file__).resolve().parent.parent
CP2 = ROOT / "campaigns/cp2"
W3 = ROOT / "campaigns/w3"
WORKERS = {"node-1", "node-2", "node-3"}
#: refresh period of each control arm — the byte window must be a whole
#: multiple of it (BYTE-02); coverage arms have no cyclic period.
ARM_PERIOD = {"uniform_eps03": None, "cyclic_k7": 7, "cyclic_k3": 3, "cyclic_k2": 2}


def load(path):
    return json.load(open(path))


def reports(base, arm):
    return sorted(glob.glob(str(Path(base) / arm / "seed*/results/report.json")))


def seed_of(path):
    return int(Path(path).parent.parent.name.replace("seed", ""))


#: One reader for both campaigns: `w3a.arm_runs` parses each report once and
#: keeps it beside its RunRead, so every accumulator below — ΔACC, bytes,
#: traffic, t_eps, κ — can see the same health verdict (S11).
arm_runs = w3a.arm_runs


def trajectory(rep):
    """[(round, node, val_acc)] over all rounds/nodes."""
    out = []
    for rd in rep["per_round"]:
        for n, d in sorted(rd["nodes"].items()):
            if d.get("val_accuracy") is not None:
                out.append((rd["round"], n, float(d["val_accuracy"])))
    return out


def uplink_by_round(rep):
    """round -> summed WORKER uplink (MB); the aggregator broadcast is excluded."""
    out = {}
    for rd in rep["per_round"]:
        out[rd["round"]] = sum(
            sum((l.get("bytes_sent", 0) or 0)
                for l in (nd.get("layer_comm_metrics") or []))
            for n, nd in rd["nodes"].items() if n in WORKERS
        ) / 1e6
    return out


def downlink_by_round(rep):
    """round -> aggregator broadcast (MB) — the direction with no shed lever."""
    agg, _ = ac.report_roles(rep)
    out = {}
    for rd in rep["per_round"]:
        nd = rd["nodes"].get(agg or "node-0") or {}
        out[rd["round"]] = sum((l.get("bytes_sent", 0) or 0)
                               for l in (nd.get("layer_comm_metrics") or [])) / 1e6
    return out


def max_t_eps(rep):
    """Round coverage completes when the LAST sender completes (NT-04): the
    round statistic is the MAX over sources, not their mean — a barrier-
    synchronised round cannot finish earlier than its slowest sender."""
    vals = []
    agg_id, _ = ac.report_roles(rep)
    for rd in rep["per_round"]:
        agg = rd["nodes"].get(agg_id or "node-0") or {}
        ut = agg.get("uplink_telemetry") or {}
        per = [float(v) for s in ut.values() if isinstance(s, dict)
               for v in [ac.t_eps_receiver_s(s)] if v is not None]
        if per:
            vals.append(max(per))
    return st.mean(vals) if vals else None


def mean_head_bytes(rep):
    """Mean sender head_bytes (coverage arms) or transmit_bytes (cyclic)."""
    vals = []
    for rd in rep["per_round"]:
        for n, nd in rd["nodes"].items():
            if n not in WORKERS:
                continue
            s = (nd.get("uplink_telemetry") or {}).get("_sender") or {}
            d = s.get("assignment_diagnostics") or {}
            v = d.get("head_bytes", d.get("transmit_bytes"))
            if v is not None:
                vals.append(float(v))
    return st.mean(vals) if vals else None


def paired(base_reads, runs):
    """Per-seed paired ΔACC (pp) on HEALTHY pairs only + the census rows."""
    diffs, seeds, census = [], [], []
    for r in runs:
        census.append((r.seed, r.healthy, r.read.health.reason, r.read.lastk))
        b = base_reads.get(r.seed)
        if b is None or not b.health.healthy or not r.healthy:
            continue
        diffs.append((r.read.lastk - b.lastk) * 100)
        seeds.append(r.seed)
    return diffs, seeds, census


def main(cp2_root=CP2, w3_root=W3, der=None):
    cp2_root, w3_root = Path(cp2_root), Path(w3_root)
    der = Path(der) if der is not None else cp2_root / "derived"
    der.mkdir(parents=True, exist_ok=True)

    # ---- 0. determinism sentinels ----
    print("=== 0. Determinism sentinels (rebuilt stack vs wave-3) ===")
    identical = True
    for cp2_arm, w3_arm in [
        ("sentinel_mono", "mono"), ("sentinel_drop_eps03", "drop_eps03"),
    ]:
        a = reports(cp2_root, cp2_arm)
        b = [p for p in reports(w3_root, w3_arm) if seed_of(p) == 41]
        if not a or not b:
            print(f"  {cp2_arm}: MISSING data (cp2={len(a)} w3={len(b)}) — skip")
            identical = False
            continue
        ta, tb = trajectory(load(a[0])), trajectory(load(b[0]))
        n = min(len(ta), len(tb))
        diffs = [abs(x[2] - y[2]) for x, y in zip(ta[:n], tb[:n])]
        mx = max(diffs) if diffs else float("nan")
        same = mx == 0.0 and len(ta) == len(tb)
        identical = identical and same
        print(f"  {cp2_arm:<22} points={n}  max|dval|={mx:.6f}  "
              f"{'BIT-IDENTICAL' if same else 'DIVERGED'}")
    if not identical:
        print("  *** SENTINELS DIVERGED: cross-campaign pairing vs w3 is "
              "cross-stack — prefer within-campaign contrasts / rerun "
              "baselines under the new stack. ***")

    # ---- reference maps from w3 (same seeds), health-gated ----
    w3_mono_runs = arm_runs(w3_root, "mono")
    w3_drop_runs = arm_runs(w3_root, "drop_eps03")
    mono = {r.seed: r.read for r in w3_mono_runs}
    drop = {r.seed: r.read for r in w3_drop_runs}
    for name, m in (("w3 mono", mono), ("w3 drop_eps03", drop)):
        bad = sorted(s for s, r in m.items() if not r.health.healthy)
        if bad:
            print(f"  !! {name} baseline collapsed at seeds {bad} — those pairs "
                  f"are dropped from every contrast below")

    # the mechanism reference is a mean too, so it is gated like every other mean
    w3_ok = [r for r in w3_drop_runs if r.healthy]
    w3_drop_teps = [v for v in (max_t_eps(r.rep) for r in w3_ok) if v]
    w3_drop_head = [v for v in (mean_head_bytes(r.rep) for r in w3_ok) if v]

    rows, dm_all, seeds_all, census = [], {}, {}, []
    print("\n=== 1+2. Claim-A controls: paired dACC + mechanism metrics ===")
    for arm, period in ARM_PERIOD.items():
        runs = arm_runs(cp2_root, arm)
        if not runs:
            print(f"  {arm}: no reports yet")
            continue
        dm, sm, cen = paired(mono, runs)
        dd, _sd, _ = paired(drop, runs)
        dm_all[arm], seeds_all[arm] = dm, sm
        census += [(arm, *c) for c in cen]
        # S11: bytes, traffic, t_eps and head bytes are per-arm MEANS, so they
        # take the same health gate as ΔACC — a diverged run's byte footprint
        # measures the divergence, not the operating point.
        ok = [r for r in runs if r.healthy]
        teps = [v for v in (max_t_eps(r.rep) for r in ok) if v]
        head = [v for v in (mean_head_bytes(r.rep) for r in ok) if v]
        ups, bands, downs, splits = [], [], [], []
        for r in ok:
            series = uplink_by_round(r.rep)
            cs = ac.cycle_stats(series, period, r0=3)
            if cs.mean is not None:
                ups.append(cs.mean)
                bands.append(ac.window_band(
                    lambda rr: ac.cycle_stats(series, period, r0=rr).mean))
            dn = list(downlink_by_round(r.rep).values())
            if dn:
                downs.append(st.mean(dn))
            splits += ac.flow_kappas(r.rep)
        up = round(st.mean(ups), 3) if ups else None
        dn_mb = round(st.mean(downs), 3) if downs else None
        rows.append(dict(
            arm=arm, refresh_period=period, n_runs=len(cen),
            n_collapsed=sum(1 for c in cen if not c[1]), n_healthy=len(ok),
            n_paired=len(dm), bytes_evaluable=bool(ok),
            dacc_vs_drop_pp=round(st.mean(dd), 2) if dd else None,
            uplink_MB=up,
            uplink_band_lo_MB=round(min(b[0] for b in bands), 3) if bands else None,
            uplink_band_hi_MB=round(max(b[1] for b in bands), 3) if bands else None,
            agg_downlink_MB=dn_mb,
            total_traffic_MB=round(up + dn_mb, 3) if (up is not None and dn_mb is not None) else None,
            mean_t_eps_max_s=round(st.mean(teps), 3) if teps else None,
            mean_head_kb=round(st.mean(head) / 1e3, 1) if head else None,
            clock_domain=ac.CLOCK_RECEIVER,
            **ac.kappa_summary(splits),
        ))
    fam = ac.contrast_family(dm_all, seeds_all)   # one Holm family: controls vs w3 mono
    byrow = {r.label: r for r in fam}
    print(ac.format_family(fam))
    for row in rows:
        r = byrow.get(row["arm"])
        if r is not None:
            row.update({k: v for k, v in r.as_dict().items() if k != "label"})
        if not row["bytes_evaluable"]:
            print(f"  {row['arm']:<15} n={row['n_paired']}  NOT EVALUABLE — "
                  f"{row['n_collapsed']}/{row['n_runs']} runs collapsed, no healthy run "
                  f"to measure bytes/t_eps on")
            continue
        band = (f"[{row['uplink_band_lo_MB']}, {row['uplink_band_hi_MB']}]"
                if row["uplink_band_lo_MB"] is not None else "--")
        print(f"  {row['arm']:<15} n={row['n_paired']}  vs drop {row['dacc_vs_drop_pp']}pp  "
              f"t_eps(max-over-senders)={row['mean_t_eps_max_s']}s  head={row['mean_head_kb']}KB  "
              f"uplink={row['uplink_MB']}MB {band}  down={row['agg_downlink_MB']}MB  "
              f"max κ_slip={w3a._num(row['max_kappa_slip'])}"
              f"{ac.kappa_bound_mark(row)}")
    if w3_drop_teps and w3_drop_head:
        print(f"  [w3 drop_eps03 reference (healthy runs only, n={len(w3_ok)}): "
              f"t_eps(max)={st.mean(w3_drop_teps):.3f}s  "
              f"head={st.mean(w3_drop_head) / 1e3:.1f}KB]")
    else:
        print("  [w3 drop reference missing or fully collapsed]")
    print("  (every column is over HEALTHY runs only — bytes, t_eps and head bytes "
          "included; S11.)")

    bad = [c for c in census if not c[2]]
    print(f"\n=== run health: {len(census) - len(bad)}/{len(census)} healthy ===")
    for arm, seed, _h, why, rd in bad:
        print(f"  COLLAPSED  {arm:<18} seed{seed}  {why}  read={rd}")
    with open(der / "cp2_health.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "healthy", "reason", "last3_val_acc"])
        w.writerows(census)

    with open(der / "cp2_controls.csv", "w", newline="") as f:
        if rows:
            keys = sorted({k for row in rows for k in row})
            w = csv.DictWriter(f, fieldnames=["arm"] + [k for k in keys if k != "arm"])
            w.writeheader()
            w.writerows(rows)

    # ---- 3. Claim-B: overlay points ----
    print("\n=== 3. Claim-B: cyclic points vs w3 tau_max frontier ===")
    frows, front, _cen = w3a.frontier_table(w3_root)
    if frows:
        print("  (w3 frontier RECOMPUTED from campaigns/w3 reports through the same "
              "reader — last-3,")
        print("   health-gated, cycle-aligned — so both halves of this exhibit have one "
              "read semantics)")
        print(ac.format_family(front))
        print(w3a.format_frontier(frows))
    else:
        # Refusing to substitute the committed artifact: it is an endpoint-read,
        # un-gated, aliased-window, conflated-κ table (writeup/19 §C quarantines
        # all four), and printing it above post-fix numbers is precisely the
        # silent mixing S6/S11 flag.
        print(f"  *** NO w3 REPORTS under {w3_root} — the frontier half of this "
              f"exhibit is UNAVAILABLE. ***")
        stale = w3_root / "derived/frontier.csv"
        if stale.exists():
            print(f"  *** {stale} exists but is a PRE-FIX artifact (endpoint read, "
                  f"un-gated ΔACC,")
            print("      aliased byte window, conflated κ — writeup/19 §C) and is NOT "
                  "printed here.")
            print("      Pull the reports and regenerate:")
            print("        rsync -az -e ssh moltres-claude:fl-framework/campaigns/w3/ "
                  "campaigns/w3/")
            print("        .venv/bin/python scripts/analyze_wave3.py ***")
    for r in rows:
        if r["arm"].startswith("cyclic"):
            m = r.get("mean")
            dacc = ("NOT EVALUABLE" if m is None or not math.isfinite(m)
                    else f"{round(m, 2)}pp")
            up = f"{r['uplink_MB']}MB" if r["bytes_evaluable"] else "NOT EVALUABLE"
            print(f"  cyclic point: {r['arm']} uplink={up} dACC={dacc} "
                  f"(healthy pairs, last-3)")

    print(f"\nDerived -> {der}/cp2_controls.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(CP2))
    ap.add_argument("--w3", default=str(W3))
    ap.add_argument("--derived", default=None)
    a = ap.parse_args()
    main(a.root, a.w3, a.derived)
