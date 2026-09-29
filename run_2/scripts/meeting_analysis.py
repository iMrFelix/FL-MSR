#!/usr/bin/env python3
"""Pre-meeting analysis from the existing phase-1b campaign (campaigns/p1).

Reproduces, from raw report.json + summary.csv, the three exhibits that answer
the advisor comments without any new runs:

  C4  the 2.16x overhead, decomposed against the two floors the framework
      already computes  ->  1.04x protocol overhead  x  1.31x whole-layer
      granularity  =  1.37x vs the (unreachable) fluid ideal
  C1  plot-6 re-cut as honest sender uplink (workers only, no broadcast dilution)
  --  a time-vs-accuracy Pareto v1 (local-val; wave-3 supplies global-val)

Ground truth (verified against raw reports, 2026-07-23):
  * node-0 is the aggregator (broadcasts 3x the model); node-1/2/3 are workers.
  * each worker's uplink is 749,405 B (14 layers), CONSTANT across trigger-path
    arms (drop/recycle/aging) and equal to mono -> "deferred, not saved".
  * summary.csv mean_fresh_bytes (~1.124 MB) is the per-node mean INCLUDING the
    aggregator broadcast = (2,248,191 + 3x749,405)/4 -- the dilution behind plot 6.

Audit corrections (writeup/19):
  * NT-02 — the 0.600 s "analytic wire floor" is a FLUID split of the bytes
    across the three classes, and deep_cnn's five indivisible 147.5 kB kernels
    can never reach it.  The floor a measured t_eps must be compared against is
    the achievable EFT makespan the sender already records
    (realized_makespan_s / predicted_t_eps ~ 0.787 s).  Both are reported.
  * NT-03 — pareto_v1 put monolithic's SENDER buffer-accept time on the same
    axis as every other arm's RECEIVER-side t_eps: the very C4 error, inside
    the script that fixes C4.  Every time column now carries a clock-domain
    tag, the axis refuses two tags, and mono is annotated in text rather than
    plotted until a receiver-side monolithic stamp exists (NT-05).
  * NT-04 — t_eps is quantized at ~0.2005 s by the kernel granularity, and a
    barrier-synchronised round completes when the LAST sender does.  This
    script prefers summary.csv's max-over-senders column and refuses to
    silently pass off the pre-registered worker-MEAN as it.
  * S6 — the pareto spread is a t-CI with the sample sd, not a population sd.

Outputs -> campaigns/p1/derived/{c4_decomposition,plot6_recut,pareto_v1}.csv (+ .png)
"""
from __future__ import annotations
import csv, glob, json, statistics as st, sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/x.py` as well as `-m scripts.x`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import analysis_common as ac

ROOT = Path(__file__).resolve().parent.parent
P1 = ROOT / "campaigns/p1"
DER = P1 / "derived"
AGG, WORKERS = "node-0", {"node-1", "node-2", "node-3"}
AGG_BW_MBPS = 6 + 3 + 1          # the three traffic classes on each worker->agg edge

def load(p): return json.load(open(p))

def worker_uplink_by_round(rep):
    """sum(layer_comm_metrics.bytes_sent) over WORKER nodes only, per round."""
    rows = []
    for rd in rep["per_round"]:
        s = 0
        for nid, nd in rd["nodes"].items():
            if nid in WORKERS:
                s += sum((l.get("bytes_sent", 0) or 0) for l in (nd.get("layer_comm_metrics") or []))
        rows.append((rd["round"], s))
    return rows

def summary_rows():
    """summary.csv rows, collapsed runs dropped (S11).

    extract_phase1b marks each run `healthy`; a pre-fix summary.csv has no such
    column, and every row is kept (with the NT-04 warning below already telling
    the reader the file predates the audit fixes).
    """
    with open(P1 / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    bad = [r for r in rows if r.get("healthy") == "False"]
    for r in bad:
        print(f"  !! {r['exp']}/{r['arm']}/seed{r['seed']} EXCLUDED from every median "
              f"({r.get('health_reason')})")
    return [r for r in rows if r.get("healthy") != "False"]

def med(xs): return st.median([float(x) for x in xs if x not in ("", None)])


def teps_column(S, exp, arm=None):
    """Median t_eps for an (exp, arm), preferring the max-over-senders column.

    NT-04: the round statistic must be the MAX over sources (a barrier-
    synchronised round completes when the last sender does); the pre-registered
    worker-MEAN mixes three discrete quantization levels and manufactures the
    sweep.  extract_phase1b emits `median_t_eps_max`; when reading a summary.csv
    generated before that fix we fall back to the mean and say so loudly rather
    than passing one statistic off as the other.
    """
    rows = [r for r in S if r["exp"] == exp and (arm is None or r["arm"] == arm)]
    col = "median_t_eps_max" if rows and rows[0].get("median_t_eps_max") else "median_t_eps"
    vals = [r[col] for r in rows if r.get(col) not in ("", None)]
    return (med(vals) if vals else None), col


def wire_bounds():
    """(achievable EFT floor, fluid floor) medians over the eps=0 sender-rounds.

    NT-02: `head_bytes x 8 / 10 Mbps` is a perfect-FLUID split of the bytes
    across the 6/3/1 Mbps classes.  deep_cnn's layers are indivisible (five
    conv kernels of ~147.5 kB plus nine layers under 7 kB), so nothing
    meaningful can ride the 1 Mbps class and the fluid value is unreachable.
    Both floors are already in every sender's assignment_diagnostics.
    """
    ach, fluid = [], []
    for p in sorted(glob.glob(str(P1 / "exp6/eps0/seed*/results/report.json"))):
        for b in ac.sender_bounds(load(p)):
            if b.achievable: ach.append(b.achievable)
            if b.fluid: fluid.append(b.fluid)
    return (st.median(ach) if ach else None), (st.median(fluid) if fluid else None)


def c4_decomposition():
    S = summary_rows()
    mono_send = med([v for r in S if r["exp"] == "exp4"                    # SENDER domain
                     for v in [ac.summary_enqueue_s(r)] if v is not None])
    eps0_teps, teps_col = teps_column(S, "exp6")                           # RECEIVER domain
    # per-worker model bytes from a real eps0 worker report
    rep0 = load(P1 / "exp6/eps0/seed41/results/report.json")
    wbytes = int(worker_uplink_by_round(rep0)[5][1] / 3)  # /3 workers -> per worker
    achievable, fluid = wire_bounds()
    if fluid is None:                       # pre-diagnostics reports: recompute it
        fluid = wbytes * 8 / (AGG_BW_MBPS * 1e6)
    if achievable is None:
        achievable = fluid
    naive = eps0_teps / mono_send                  # the retired 2.16x
    protocol = eps0_teps / achievable              # what the scheduler actually costs
    granularity = achievable / fluid               # irreducible whole-layer penalty
    vs_fluid = eps0_teps / fluid
    sweep = {e: teps_column(S, "exp1", a)[0]
             for e, a in [(0.2, "eps02"), (0.3, "eps03"), (0.4, "eps04")]}
    rows = [
        (f"mono send [{ac.CLOCK_SENDER_ENQUEUE}] (exp4, n=6)", f"{mono_send:.3f}s"),
        ("FLUID floor = %d B x8 / %d Mbps (UNREACHABLE: layers are indivisible)"
         % (wbytes, AGG_BW_MBPS), f"{fluid:.3f}s"),
        ("ACHIEVABLE floor = EFT makespan over whole layers (realized_makespan_s)",
         f"{achievable:.3f}s"),
        (f"eps=0 t_eps [{ac.CLOCK_RECEIVER}] (exp6, n=6, {teps_col})", f"{eps0_teps:.3f}s"),
        ("RETIRED naive ratio t_eps / mono_send -- MIXED CLOCK DOMAINS, do not quote",
         f"{naive:.2f}x"),
        ("PROTOCOL overhead  t_eps / achievable", f"{protocol:.2f}x"),
        ("GRANULARITY penalty  achievable / fluid", f"{granularity:.2f}x"),
        ("= total vs the fluid ideal  (protocol x granularity)",
         f"{protocol:.2f} x {granularity:.2f} = {vs_fluid:.2f}x"),
    ]
    for e, t in sweep.items():
        if t is None: continue
        rows.append((f"t_eps at eps={e} (vs achievable {achievable:.3f}s)",
                     f"{t:.3f}s  {'BEATS achievable floor' if t < achievable else 'above achievable floor'}"))
    with open(DER / "c4_decomposition.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["quantity", "value"]); w.writerows(rows)
    print("\n=== C4: the 2.16x, decomposed (answers advisor comment C4) ===")
    for k, v in rows: print(f"  {k:<74} {v}")
    print(f"  -> Story: the 2.16x compared two clock domains and is retired. Against the")
    print(f"     ACHIEVABLE floor the eps=0 no-shed protocol overhead is {protocol:.2f}x; the")
    print(f"     remaining {granularity:.2f}x is the irreducible whole-layer granularity gap to the")
    print(f"     fluid ideal, not a scheduler defect. No receiver-clock caveat is needed:")
    print(f"     both bounds are recorded per sender-round in the reports already.")
    if teps_col != "median_t_eps_max":
        print(f"  !! WARNING: summary.csv predates the NT-04 fix — t_eps here is the")
        print(f"     pre-registered worker-MEAN, which mixes the ~0.2005s quantization")
        print(f"     levels. Re-run scripts/extract_phase1b.py for the max-over-senders read.")
    return dict(mono_send=mono_send, fluid=fluid, achievable=achievable,
                eps0_teps=eps0_teps, naive=naive, protocol=protocol,
                granularity=granularity, vs_fluid=vs_fluid, sweep=sweep,
                wbytes=wbytes, teps_col=teps_col)


def plot6_recut():
    on = load(P1 / "exp10_recycle/skip_aging_smoke/results/report.json")
    off = load(P1 / "exp10_recycle/skip_noaging_smoke_report.json")
    ron = worker_uplink_by_round(on)
    roff = worker_uplink_by_round(off)
    n = min(len(ron), len(roff))
    rows = []
    for i in range(n):
        rnd = ron[i][0]
        von, voff = ron[i][1] / 1e6, roff[i][1] / 1e6
        # the diluted per-node-mean-incl-broadcast the advisors actually saw:
        diluted_on = (ron[i][1] + 2.248e6) / 4 / 1e6
        diluted_off = (roff[i][1] + 2.248e6) / 4 / 1e6
        rows.append((rnd, round(von, 3), round(voff, 3),
                     round(von / voff, 2) if voff else None,
                     round(diluted_on, 3), round(diluted_off, 3)))
    with open(DER / "plot6_recut.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "aging_on_MB", "aging_off_MB", "honest_ratio",
                    "diluted_on_MB(incl_broadcast)", "diluted_off_MB(incl_broadcast)"])
        w.writerows(rows)
    print("\n=== C1: plot-6 re-cut as honest sender uplink (answers advisor comment C1) ===")
    print("  round | aging-ON | aging-OFF | honest x | (diluted-ON | diluted-OFF)")
    for rnd, on_, off_, rat, d_on, d_off in rows:
        print(f"   r{rnd}   |  {on_:5.3f}  |  {off_:5.3f}   |  {rat}x  | ({d_on:.3f} | {d_off:.3f})")
    r3 = rows[3] if len(rows) > 3 else rows[-1]
    print(f"  -> At r3 the HONEST worker-uplink contrast is {r3[1]:.3f} vs {r3[2]:.3f} MB "
          f"= {r3[3]}x;\n     the plotted per-node-mean-incl-broadcast diluted it to "
          f"{r3[4]:.3f} vs {r3[5]:.3f} ({r3[4]/r3[5]:.1f}x). The aging cap fires at r3 "
          f"(refresh);\n     aging-off keeps ratcheting down (unbounded staleness).")
    return rows


def pareto_v1(c4):
    """Time-vs-accuracy frontier on ONE clock domain (NT-03).

    Monolithic has no receiver-side completion stamp, only a sender-side
    buffer-accept time, so it is NOT plotted: it is annotated in text with its
    achievable wire floor until NT-05 adds the stamp.  The previous version
    placed it at 0.379 s — the very quantity this script identifies as
    undercounting the wire by 0.221 s — which put it left of every arm but
    eps=0.4 and inverted which arms looked Pareto-dominant.
    """
    S = summary_rows()
    mono = {r["seed"]: float(r["final_val_acc"]) for r in S if r["exp"] == "exp4"}
    pts, dropped = {}, []
    for name, exp, a in [("eps0", "exp6", "eps0"), ("eps0.2", "exp1", "eps02"),
                         ("eps0.3", "exp1", "eps03"), ("eps0.4", "exp1", "eps04")]:
        accs = [100 * (float(r["final_val_acc"]) - mono[r["seed"]])
                for r in S if r["exp"] == exp and r["arm"] == a and r["seed"] in mono]
        t, col = teps_column(S, exp, a)
        if t is None:                       # no receiver-side time -> off the axis
            dropped.append(name); continue
        pts[name] = (t, st.mean(accs), ac.tci95(accs), len(accs), ac.CLOCK_RECEIVER)
    ac.require_one_clock([p[4] for p in pts.values()], "pareto time axis")
    with open(DER / "pareto_v1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", ac.CLOCK_RECEIVER, "dACC_localval_pp", "ci95_pp", "n",
                    "clock_domain"])
        for k, (t, d, ci, n, dom) in pts.items():
            w.writerow([k, round(t, 3), round(d, 2), round(ci, 2), n, dom])
    print("\n=== Pareto v1: time vs accuracy (LOCAL-val, PRELIMINARY; wave-3 = global-val) ===")
    print(f"  arm    | {ac.CLOCK_RECEIVER} | dACC (local, pp) +- t-CI")
    for k, (t, d, ci, n, _dom) in pts.items():
        print(f"  {k:<7}| {t:5.3f}s | {d:+5.2f} +- {ci:.2f}  (n={n})")
    print(f"  [mono is OFF this axis: it has only a {ac.CLOCK_SENDER_ENQUEUE} value "
          f"({c4['mono_send']:.3f}s),")
    print(f"   which is not comparable with a receiver-side completion time. Its "
          f"achievable")
    print(f"   wire floor is {c4['achievable']:.3f}s (fluid ideal {c4['fluid']:.3f}s) "
          f"— quote that instead until NT-05.]")
    if dropped:
        print(f"  (dropped, no receiver-side time: {dropped})")
    ts = [v[0] for v in pts.values()]
    print("  -> t_eps falls with eps, but it is QUANTIZED at ~0.2005s by the 147.5 kB")
    print("     kernel granularity, so eps moves the MIXTURE over k=1,2,3 levels, not a")
    print(f"     continuum: observed levels {sorted({round(t, 2) for t in ts})}.")
    return pts


def figures(c4, p6):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n(matplotlib unavailable, skipping PNGs: {e})"); return
    # plot-6 re-cut
    fig, ax = plt.subplots(figsize=(7, 4))
    rr = [r[0] for r in p6]
    ax.plot(rr, [r[1] for r in p6], "o-", color="#c0392b", label="aging ON (honest worker uplink)")
    ax.plot(rr, [r[2] for r in p6], "s-", color="#2980b9", label="aging OFF (honest worker uplink)")
    ax.plot(rr, [r[4] for r in p6], "o--", color="#c0392b", alpha=0.35, label="aging ON (diluted, incl broadcast)")
    ax.plot(rr, [r[5] for r in p6], "s--", color="#2980b9", alpha=0.35, label="aging OFF (diluted, incl broadcast)")
    ax.set_xlabel("round"); ax.set_ylabel("sender uplink (MB/round, summed over 3 workers)")
    ax.set_title("Plot-6 re-cut: honest sender uplink vs broadcast-diluted metric")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(DER / "plot6_recut.png", dpi=130); plt.close(fig)
    # C4 waterfall
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = [("fluid ideal\n(unreachable)", c4["fluid"], "#27ae60"),
            ("achievable EFT\nfloor", c4["achievable"], "#16a085"),
            ("eps=0 t_eps\n(receiver)", c4["eps0_teps"], "#c0392b")]
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
    ax.axhline(c4["achievable"], ls="--", color="#16a085", alpha=0.6)
    ax.axhline(c4["fluid"], ls=":", color="#27ae60", alpha=0.6)
    for e, t in c4["sweep"].items():
        ax.scatter(["eps=0 t_eps\n(receiver)"], [t], color="black", zorder=5)
        ax.annotate(f"eps={e}: {t:.2f}s", ("eps=0 t_eps\n(receiver)", t), fontsize=6,
                    xytext=(3, 0), textcoords="offset points")
    ax.set_ylabel("seconds"); ax.set_title(
        f"C4: {c4['protocol']:.2f}x protocol x {c4['granularity']:.2f}x layer granularity "
        f"= {c4['vs_fluid']:.2f}x vs fluid")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(DER / "c4_decomposition.png", dpi=130); plt.close(fig)
    print(f"\n  figures -> {DER}/plot6_recut.png , c4_decomposition.png")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="phase-1b meeting exhibits")
    _ap.add_argument("--derived", default=None, help="output dir (default campaigns/p1/derived)")
    _a = _ap.parse_args()
    if _a.derived:
        DER = Path(_a.derived)
    DER.mkdir(parents=True, exist_ok=True)
    c4 = c4_decomposition()
    p6 = plot6_recut()
    pareto_v1(c4)
    figures(c4, p6)
    print(f"\nAll derived artifacts in {DER}/")
