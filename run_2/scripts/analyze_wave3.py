#!/usr/bin/env python3
"""Analyze the wave-3 campaign once results are pulled to campaigns/w3/.

Turnkey morning tool. Robust to partial results (processes whatever report.json
exist). All wave-3 arms ran with global_eval=true (K0), so val_accuracy is the
seed-independent global-held-out number and paired ΔACC vs same-seed mono is
low-variance.

Read semantics are the audited ones (writeup/19), all from scripts/analysis_common:
  * last-3-round mean, not the single endpoint round (S7 — the endpoint noise
    flipped the sign of the aging−recycle null); the endpoint is kept as a
    sensitivity column;
  * a run with a non-finite loss or a read pinned at chance is EXCLUDED from
    EVERY mean — accuracy, bytes, traffic, timing and κ alike — and reported in
    the collapse census (S1/S11 — the τ frontier averaged chance-level runs into
    a "graded tradeoff", and a diverged run's byte column is a measurement of
    the divergence, not of the mechanism); an arm with no healthy run publishes
    NOT EVALUABLE, never a number;
  * t-based CIs with the sample sd (S6/ML-05), never 1.96·pstdev;
  * paired t + exact sign-flip + Holm within each pre-registered family, and
    the frozen ≤1pp non-inferiority UCB, from scripts/hypothesis_tests (S5/S9/S15);
  * byte windows averaged over an INTEGER number of refresh cycles, with the
    window-start sensitivity band (BYTE-02);
  * downlink and total traffic alongside the uplink column, and every saving
    labelled with its direction (BYTE-03);
  * κ reported as TWO columns — coverage slippage (the quantity ε bounds) and
    deliberately-shed recycled mass — never as one conflated number (BYTE-04).

Pull first (single ssh, no polling):
    rsync -az -e ssh moltres-claude:fl-framework/campaigns/w3/ campaigns/w3/

Then:
    .venv/bin/python scripts/analyze_wave3.py

Outputs -> campaigns/w3/derived/{deltacc,frontier,health,timing}.csv (+ PNGs)
"""
from __future__ import annotations
import csv, glob, json, math, statistics as st, sys
from pathlib import Path
from typing import NamedTuple

if __package__ in (None, ""):  # allow `python scripts/x.py` as well as `-m scripts.x`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import analysis_common as ac

ROOT = Path(__file__).resolve().parent.parent
W3 = ROOT / "campaigns/w3"
DER = W3 / "derived"
WORKERS = {"node-1", "node-2", "node-3"}
REF_ARM = "byte_balanced_eps0"   # full-model uplink = the mono-equivalent byte reference

def reports(root, arm):
    return sorted(glob.glob(str(Path(root) / arm / "seed*/results/report.json")))

def seed_of(path): return int(Path(path).parent.parent.name.replace("seed", ""))

class Run(NamedTuple):
    """One run: the parsed report kept beside its read, so the health verdict
    is available to every accumulator that touches the report (S11)."""

    path: str
    rep: dict
    read: ac.RunRead

    @property
    def healthy(self): return self.read.health.healthy

    @property
    def seed(self): return self.read.seed


def arm_runs(root, arm):
    """Every run of one arm, parsed once. Callers MUST filter on `.healthy`."""
    out = []
    for p in reports(root, arm):
        rep = json.load(open(p))
        out.append(Run(p, rep, ac.read_run(rep, seed=seed_of(p), workers=WORKERS)))
    return out


def census_of(arm, runs):
    """Health-census rows for one arm (every run, healthy or not)."""
    return [(arm, r.seed, r.read.health.healthy, r.read.health.reason,
             round(r.read.health.peak, 4) if r.read.health.peak is not None else None,
             round(r.read.lastk, 4) if r.read.lastk is not None else None)
            for r in runs]

def uplink_by_round(rep, r0=0):
    """round -> summed WORKER uplink (MB); the aggregator broadcast is excluded."""
    out = {}
    for rd in rep["per_round"]:
        if rd["round"] < r0: continue
        out[rd["round"]] = sum(
            sum((l.get("bytes_sent", 0) or 0) for l in (nd.get("layer_comm_metrics") or []))
            for n, nd in rd["nodes"].items() if n in WORKERS) / 1e6
    return out

def downlink_by_round(rep):
    """round -> aggregator broadcast (MB). Constant in every arm: the mechanism
    has no downlink lever (_downlink_plan broadcasts every layer) — BYTE-03."""
    agg, _ = ac.report_roles(rep)
    out = {}
    for rd in rep["per_round"]:
        nd = rd["nodes"].get(agg or "node-0") or {}
        out[rd["round"]] = sum((l.get("bytes_sent", 0) or 0)
                               for l in (nd.get("layer_comm_metrics") or [])) / 1e6
    return out

def paired(mono_reads, runs):
    """Per-seed paired ΔACC (pp) on HEALTHY pairs only."""
    ok, ends, seeds = [], [], []
    for r in runs:
        m = mono_reads.get(r.seed)
        if m is None or not m.health.healthy or not r.healthy:
            continue
        ok.append((r.read.lastk - m.lastk) * 100)
        ends.append((r.read.endpoint - m.endpoint) * 100)
        seeds.append(r.seed)
    return ok, ends, seeds


def byte_reference(root):
    """(uplink, downlink) MB/round of the full-model arm over HEALTHY runs.

    The denominator of every saving percentage — so it is gated exactly like
    the numerator; a collapsed reference would rescale the whole column.
    """
    ups, downs = [], []
    for r in arm_runs(root, REF_ARM):
        if not r.healthy:
            continue
        ups.append(st.mean(list(uplink_by_round(r.rep, 3).values())))
        downs.append(st.mean(list(downlink_by_round(r.rep).values())))
    return (st.mean(ups) if ups else None, st.mean(downs) if downs else None)


def arm_bytes(runs, tau, ref_up=None, ref_down=None):
    """Byte/traffic/κ columns for one arm over its HEALTHY runs only (S11).

    The τ frontier's whole point is a bytes↔accuracy exchange rate, so a byte
    mean taken over diverged runs is not a cheaper operating point — it is the
    footprint of a model that stopped learning (frontier_agingoff published a
    99.4% "saving" computed entirely from its three collapsed runs). With no
    healthy run the columns are None and the arm reads NOT EVALUABLE.
    """
    ok = [r for r in runs if r.healthy]
    cols = {"n_healthy": len(ok), "bytes_evaluable": bool(ok)}
    ups, spreads, cycles, bands, downs, splits = [], [], [], [], [], []
    for r in ok:
        series = uplink_by_round(r.rep)
        cs = ac.cycle_stats(series, tau, r0=3)
        if cs.mean is not None:
            ups.append(cs.mean); spreads.append(cs.cycle_spread); cycles.append(cs.n_cycles)
            bands.append(ac.window_band(lambda rr: ac.cycle_stats(series, tau, r0=rr).mean))
        dn = list(downlink_by_round(r.rep).values())
        if dn:
            downs.append(st.mean(dn))
        splits += ac.flow_kappas(r.rep)
    up = round(st.mean(ups), 3) if ups else None
    dn_mb = round(st.mean(downs), 3) if downs else None
    cols.update(
        steady_uplink_MB=up,
        uplink_band_lo_MB=round(min(b[0] for b in bands), 3) if bands else None,
        uplink_band_hi_MB=round(max(b[1] for b in bands), 3) if bands else None,
        n_cycles=min(cycles) if cycles else 0,
        cycle_spread_MB=(round(st.mean([x for x in spreads if math.isfinite(x)]), 3)
                         if any(math.isfinite(x) for x in spreads) else None),
        agg_downlink_MB=dn_mb,
        total_traffic_MB=round(up + dn_mb, 3) if (up is not None and dn_mb is not None) else None,
        uplink_saving_pct=round(100 * (1 - up / ref_up), 1) if (up is not None and ref_up) else None,
        total_saving_pct=(round(100 * (1 - (up + dn_mb) / (ref_up + ref_down)), 1)
                          if (up is not None and dn_mb is not None
                              and ref_up and ref_down is not None) else None),
        **ac.kappa_summary(splits))
    return cols


#: τ_max per frontier arm — also the byte window's cycle length (BYTE-02).
TAU = {"frontier_tau2": 2, "frontier_tau3": 3, "frontier_tau5": 5,
       "frontier_tau8": 8, "frontier_agingoff": None}


def frontier_table(root, monos=None):
    """Recompute the τ frontier from the reports with the audited semantics.

    Public because analyze_cp2 overlays its cyclic points on this frontier: it
    must recompute it here rather than print the committed derived/frontier.csv,
    which is a pre-fix artifact (endpoint read, un-gated ΔACC, aliased byte
    window, conflated κ) and would put two read semantics in one table (S6/S11).

    Returns (rows, contrast_rows, census) or (None, None, None) when the wave-3
    reports are not on disk — the caller must then say so, not fall back.
    """
    root = Path(root)
    monos = mono_reads(root) if monos is None else monos
    ref_up, ref_down = byte_reference(root)
    fdiffs, fseeds, fextra, census = {}, {}, {}, []
    seen = False
    for arm, tau in TAU.items():
        runs = arm_runs(root, arm)
        if not runs:
            continue
        seen = True
        census += census_of(arm, runs)
        d, _e, s = paired(monos, runs)
        fdiffs[arm], fseeds[arm] = d, s   # empty ⇒ NOT EVALUABLE row (all pairs collapsed)
        n_bad = sum(1 for r in runs if not r.healthy)
        fextra[arm] = dict(tau_max=tau, n_runs=len(runs), n_collapsed=n_bad,
                           collapse_rate=round(n_bad / len(runs), 3),
                           **arm_bytes(runs, tau, ref_up, ref_down))
    if not seen:
        return None, None, None
    front = ac.contrast_family(fdiffs, fseeds)   # one Holm family: τ frontier vs mono
    byrow = {r.label: r for r in front}
    rows = []
    for arm in fextra:
        r = byrow.get(arm)
        rows.append({"arm": arm, **fextra[arm],
                     **({k: v for k, v in r.as_dict().items() if k != "label"} if r else {}),
                     "read": "last3", "family": "frontier",
                     "ref_uplink_MB": round(ref_up, 3) if ref_up else None,
                     "ref_downlink_MB": round(ref_down, 3) if ref_down else None})
    return rows, front, census


def format_frontier(rows):
    """Frontier table with NOT EVALUABLE where the health gate emptied an arm."""
    def cell(row, key, width, suffix=""):
        w = width + len(suffix)
        if not row["bytes_evaluable"]:
            return f"{'NOT EVAL':>{w}}"
        v = row.get(key)
        return f"{'--' if v is None else str(v) + suffix:>{w}}"

    out = [f"  {'arm':<20}{'τ':>3}{'collapse':>10}{'uplinkMB':>10}{'[band]':>16}"
           f"{'cyc':>5}{'downMB':>9}{'totalMB':>9}{'up-save':>9}{'tot-save':>9}"]
    for row in rows:
        band = (f"[{row['uplink_band_lo_MB']}, {row['uplink_band_hi_MB']}]"
                if row.get("uplink_band_lo_MB") is not None else "--")
        coll = "{}/{}".format(row["n_collapsed"], row["n_runs"])
        out.append(f"  {row['arm']:<20}{str(row['tau_max']):>3}{coll:>10}"
                   f"{cell(row, 'steady_uplink_MB', 10)}"
                   f"{(band if row['bytes_evaluable'] else '--'):>16}"
                   f"{row['n_cycles']:>5}{cell(row, 'agg_downlink_MB', 9)}"
                   f"{cell(row, 'total_traffic_MB', 9)}"
                   f"{cell(row, 'uplink_saving_pct', 8, '%')}"
                   f"{cell(row, 'total_saving_pct', 9, '%')}")
    out.append("  (byte/traffic/κ columns are over HEALTHY runs only — an arm whose runs"
               " all collapsed")
    out.append("   has no operating point to report, so it reads NOT EVAL rather than a"
               " cheap-looking number; S11.")
    out.append("   savings are UPLINK unless the total column says otherwise: the downlink"
               " broadcast has no shed lever — BYTE-03.)")
    return "\n".join(out)


def _num(v, fmt="{:.4f}"):
    return "NA" if v is None else fmt.format(v)


def format_kappa(rows, eps=0.3):
    """Coverage accounting: slippage and recycled mass as SEPARATE columns.

    The single κ symbol counted deliberate omission as coverage failure, so the
    frontier table appeared to violate its own "κ ≤ ε" invariant by 3× at τ=8,
    where five of six shed layers had never been sent (BYTE-04). Split, the
    slippage column is the bound check and the recycled column is a mechanism
    statistic that ε says nothing about.

    The bound check is only as good as its coverage: `max_kappa_slip` maximizes
    over the flows whose split resolved, so it is a LOWER bound on the arm's max
    and the ≤ ε mark comes from `ac.kappa_bound_mark`, which withholds the
    verdict while anything is unresolved.
    """
    out = [f"  {'arm':<20}{'max κ_slip':>11}{'ub':>8}{'recycled':>10}"
           f"{'shed/rcy/slip layers':>22}{'unres/n':>12}  source",
           f"  (κ_slip ≤ ε={eps} is the Claim-A bound; 'recycled' is deliberate,"
           f" ε-exempt shed mass)"]
    for row in rows:
        if not row.get("n_kappa_flows"):
            out.append(f"  {row['arm']:<20}{'NOT EVAL':>11}   (no healthy run with κ telemetry)")
            continue
        layers = (f"{row['mean_shed_layers']}/{row['mean_recycled_layers']}"
                  f"/{row['mean_slip_layers']}")
        unres = f"{row['n_kappa_unresolved']}/{row['n_kappa_flows']}"
        out.append(f"  {row['arm']:<20}{_num(row['max_kappa_slip']):>11}"
                   f"{_num(row['max_kappa_slip_ub'], '{:.2f}'):>8}"
                   f"{_num(row['max_recycled_mass_fraction']):>10}{layers:>22}"
                   f"{unres:>12}  {row['kappa_source']}"
                   f"{ac.kappa_bound_mark(row, eps)}")
    checks = [r for r in rows if r.get("n_kappa_recon_checks")]
    if checks:
        err = max(r["kappa_recon_max_err"] for r in checks)
        n = sum(r["n_kappa_recon_checks"] for r in checks)
        recon = sum(r.get("n_kappa_reconstructed") or 0 for r in rows)
        out.append(f"  ({recon} flows' slippage reconstructed from the sender's manifest"
                   f" scores; on the {n} flows")
        out.append(f"   set logic settles independently the reconstruction agrees to"
                   f" {err:.1e}.)")
    out.append("  (κ_slip is a max over the RESOLVED flows — a lower bound on the arm's"
               " true max — so the ≤ε")
    out.append("   verdict is minted only where every flow resolved; an unresolved flow"
               " suspends it, never")
    out.append("   passes it. 'ub' bounds κ_slip over the unresolved flows too, so it is"
               " vacuous wherever")
    out.append("   recycled mass dominates — it is the old single-κ column, which"
               " published exactly those")
    out.append("   numbers as slippage. Post-audit reports carry `kappa_slip` beside the"
               " unchanged total.)")
    return "\n".join(out)


def mono_reads(root):
    """seed -> RunRead for the mono baseline; a collapsed baseline kills its pair."""
    return {r.seed: r.read for r in arm_runs(Path(root), "mono")}


def main(root=W3, der=None):
    root = Path(root)
    der = Path(der) if der is not None else root / "derived"
    der.mkdir(parents=True, exist_ok=True)

    # mono baseline per seed (paired); a collapsed baseline invalidates its pair
    mono_runs = arm_runs(root, "mono")
    monos = {r.seed: r.read for r in mono_runs}
    census = census_of("mono", mono_runs)
    healthy_mono = sorted(s for s, r in monos.items() if r.health.healthy)
    print(f"mono seeds: {sorted(monos)} (n={len(monos)}); "
          f"healthy: {healthy_mono} (n={len(healthy_mono)})")
    for s, r in sorted(monos.items()):
        if not r.health.healthy:
            print(f"  !! mono/seed{s} EXCLUDED ({r.health.reason}, peak="
                  f"{r.health.peak}, read={r.health.read}) — its pairs are dropped")

    # ---- paired ΔACC per arm (triptych + aging n=6) ----
    ACC_ARMS = ["drop_eps03", "recycle_eps03", "recycle_aging_eps03", "renormalize_eps03"]
    diffs, ends, seedmap, drows = {}, {}, {}, []
    for arm in ACC_ARMS:
        runs = arm_runs(root, arm)
        census += census_of(arm, runs)
        d, e, s = paired(monos, runs)
        diffs[arm], ends[arm], seedmap[arm] = d, e, s   # empty ⇒ NOT EVALUABLE row
    trip = ac.contrast_family(diffs, seedmap)          # family 1: triptych vs mono
    print("\n=== paired ΔACC vs mono (global val, last-3 read, healthy pairs) ===")
    print(ac.format_family(trip))
    for r in trip:
        ep = st.mean(ends[r.label]) if ends.get(r.label) else float("nan")
        print(f"  {r.label:<22} seeds={list(r.seeds)}  per-seed ΔACC="
              f"{[round(x, 2) for x in diffs.get(r.label, [])]}  (endpoint read: {ep:+.2f}pp)")
        drows.append({**r.as_dict(), "dACC_endpoint_pp": round(ep, 2),
                      "read": "last3", "family": "triptych"})
    with open(der / "deltacc.csv", "w", newline="") as f:
        if drows:
            w = csv.DictWriter(f, fieldnames=list(drows[0])); w.writeheader(); w.writerows(drows)

    # ---- refresh-cost frontier (same code path analyze_cp2 overlays onto) ----
    frows, front, fcensus = frontier_table(root, monos)
    frows = frows or []
    census += fcensus or []
    if frows:
        with open(der / "frontier.csv", "w", newline="") as f:
            keys = sorted({k for row in frows for k in row})
            w = csv.DictWriter(f, fieldnames=["arm"] + [k for k in keys if k != "arm"])
            w.writeheader(); w.writerows(frows)
        ref_up = frows[0].get("ref_uplink_MB")
        ref_down = frows[0].get("ref_downlink_MB")
        print("\n=== refresh-cost frontier (skip=shed; cycle-aligned byte window) ===")
        print(ac.format_family(front))
        print(f"  byte reference (mono-equivalent) = {ref_up} MB uplink + "
              f"{ref_down} MB downlink/round\n" if ref_up else "  (no healthy byte reference)")
        print(format_frontier(frows))
        print("\n=== coverage accounting: κ_slip vs recycled mass (BYTE-04) ===")
        print(format_kappa(frows))

    # ---- health / collapse census ----
    with open(der / "health.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "healthy", "reason", "peak_val_acc", "last3_val_acc"])
        w.writerows(census)
    bad = [c for c in census if not c[2]]
    print(f"\n=== run health: {len(census) - len(bad)}/{len(census)} healthy ===")
    for arm, s, _h, why, peak, rd in bad:
        print(f"  COLLAPSED  {arm:<22} seed{s}  {why:<20} peak={peak} read={rd}")

    # ---- timing (serial arms) — SENDER-SIDE, quarantined, health-gated ----
    trows = []
    for arm in ["mono", REF_ARM]:
        for r in arm_runs(root, arm):
            sends = [float(v) for rd in r.rep["per_round"][-5:]
                     for n, nd in rd["nodes"].items() if n in WORKERS
                     for v in [ac.enqueue_duration_s(nd)] if v is not None]
            trows.append((arm, r.seed, r.healthy,
                          round(st.median(sends), 3) if (sends and r.healthy) else None))
    with open(der / "timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "healthy", f"median_{ac.CLOCK_SENDER_ENQUEUE}"])
        w.writerows(trows)
    print(f"\n=== timing ({ac.CLOCK_SENDER_ENQUEUE}; buffer-accept, NOT wire time) ===")
    for arm, s, ok, t in trows:
        print(f"  {arm:<20} seed{s}: {t}s" if ok
              else f"  {arm:<20} seed{s}: NOT EVALUABLE (run collapsed)")
    print("  (QUARANTINED as a wall-clock claim — this clock domain must never be"
          f" plotted against {ac.CLOCK_RECEIVER}; NT-01/NT-03.)")

    figures(trip, frows, der)
    print(f"\nDerived -> {der}/  (paste ΔACC + frontier into writeup/14 §New data)")


def figures(trip, frows, der=None):
    der = Path(der) if der is not None else DER
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(no matplotlib: {e})"); return
    ok = [r for r in trip if r.n >= 2]
    if ok:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [r.label.replace("_eps03", "") for r in ok]
        ax.bar(names, [r.mean for r in ok], yerr=[r.ci95 for r in ok], capsize=4,
               color=["#7f8c8d", "#2980b9", "#27ae60", "#8e44ad"][:len(names)])
        ax.axhline(-1.0, ls="--", color="red", alpha=0.6, label="−1pp non-inferiority margin")
        ax.set_ylabel("ΔACC vs mono (pp, global val, last-3)")
        ax.set_title("Slippage triptych + aging under global val (t-CI, healthy pairs)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
        fig.savefig(der / "triptych.png", dpi=130); plt.close(fig)
    # only arms with a healthy operating point are plottable — a collapsed arm
    # has neither a byte point nor a ΔACC to place it at (S11)
    fr = [r for r in frows if r.get("tau_max") is not None and r.get("bytes_evaluable")
          and r.get("steady_uplink_MB") is not None and r.get("uplink_band_lo_MB") is not None
          and r.get("mean") is not None and math.isfinite(r["mean"])]
    if fr:
        fr.sort(key=lambda r: r["tau_max"])
        fig, ax1 = plt.subplots(figsize=(6, 4))
        taus = [r["tau_max"] for r in fr]
        ax1.errorbar(taus, [r["steady_uplink_MB"] for r in fr],
                     yerr=[[r["steady_uplink_MB"] - r["uplink_band_lo_MB"] for r in fr],
                           [r["uplink_band_hi_MB"] - r["steady_uplink_MB"] for r in fr]],
                     fmt="o-", color="#2980b9", capsize=3)
        ax1.set_xlabel("τ_max (refresh period)")
        ax1.set_ylabel("steady-state UPLINK (MB, cycle-aligned ± window band)", color="#2980b9")
        ax2 = ax1.twinx()
        ax2.errorbar(taus, [r["mean"] for r in fr], yerr=[r["ci95"] for r in fr],
                     fmt="s-", color="#c0392b", capsize=3)
        for r in fr:
            if r["n_collapsed"]:
                ax2.annotate(f"{r['n_collapsed']}/{r['n_runs']} collapsed",
                             (r["tau_max"], r["mean"]), fontsize=6, xytext=(3, -9),
                             textcoords="offset points", color="#c0392b")
        ax2.set_ylabel("ΔACC (pp, healthy pairs, t-CI)", color="#c0392b")
        ax1.set_title("Refresh-cost frontier: uplink vs accuracy vs τ_max"); fig.tight_layout()
        fig.savefig(der / "frontier.png", dpi=130); plt.close(fig)
    print(f"  figures -> {der}/triptych.png , frontier.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(W3), help="campaign root (default campaigns/w3)")
    ap.add_argument("--derived", default=None, help="output dir (default <root>/derived)")
    a = ap.parse_args()
    main(a.root, a.derived)
