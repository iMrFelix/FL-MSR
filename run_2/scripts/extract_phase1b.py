"""Extract phase-1b metrics from campaigns/p1/<exp>/<arm>/seed<N>/results/report.json.

Frozen aggregation recipe (writeup/11 §2): per metric, worker-MEAN across
sender/worker nodes per round, then MEDIAN across rounds per seed. Watchdog-fired
sender-rounds are censored from clean t_eps. Emits:
  - campaigns/p1/summary.csv     (one row per exp,arm,seed)
  - campaigns/p1/perround.csv    (one row per exp,arm,seed,round)
and prints a headline table (across-seed) with the key contrasts vs the
monolithic floor.

Four audited corrections (writeup/19):

  * **NT-04 — max over senders.** t_eps is quantized at k x 0.2005 s (the wire
    time of one 147.5 kB conv kernel on the 6 Mbps class), so its support is
    three discrete points and a MEAN over three senders manufactures a smooth
    sweep out of a moving mixture. A barrier-synchronised round's coverage
    completes when the LAST sender completes, so the round statistic is the
    MAX. `t_eps_max` / `median_t_eps_max` are the primary columns; the
    pre-registered worker-mean is kept beside them as `t_eps_mean` /
    `median_t_eps` and disclosed, never silently replaced. A per-flow level
    census is printed so the quantization is visible.
  * **BYTE-05 — no broadcast dilution.** The old `fresh_bytes_mean` averaged
    the aggregator's 2,248,191 B downlink broadcast together with the three
    workers' 749,405 B uplinks, giving an arm-INSENSITIVE 1,124,101 B for
    every arm and compressing the dynamic range of every byte statement
    sourced from it by ~200x. The aggregator is now identified from the
    telemetry (not hardcoded as node-0) and excluded, giving
    `worker_uplink_bytes_mean`, with `agg_downlink_bytes_mean` reported
    separately and `fresh_aggregated_bytes_mean` counting only the bytes of
    layers that were NOT shed — so "fresh" means fresh.
  * **BYTE-04 — κ is two quantities.** `max_kappa` conflated coverage slippage
    (what ε bounds) with mass the sender was advised to skip. The summary now
    carries `max_kappa_slip`, `max_kappa_slip_ub` and
    `max_recycled_mass_fraction` separately, and the headline column is the
    slippage one. On the p1 arms (no skip-feedback) the two coincide, which is
    why κ there always read cleanly ≤ ε. Where they do not, the split comes
    from the sender's own `layer_comm_metrics` importances (the frozen manifest
    scores), so `n_kappa_unresolved` is what licenses the ≤ ε verdict.
  * **S11 — health-gated medians.** A run with a non-finite loss or a read
    pinned at chance is written to summary.csv with `healthy=False` and its
    reason, and is EXCLUDED from the per-arm medians and the t_eps census — a
    diverged run's timings measure the divergence, not the mechanism.

Usage: python scripts/extract_phase1b.py [--root campaigns/p1]
Pinned report.json fields: round_duration_s, val_accuracy, layer_comm_metrics
(bytes_sent), uplink_telemetry.<sender>.{t_eps_local_receiver_s,kappa_realized,watchdog_fired,
shed_layers}.
"""
from __future__ import annotations
import argparse, csv, json, os, glob, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/x.py` as well as `-m scripts.x`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import analysis_common as ac


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _find_uplink(nodes):
    """Return the uplink_telemetry dict (sender -> metrics), else {}."""
    for n in nodes.values():
        ut = n.get("uplink_telemetry")
        if isinstance(ut, dict) and any(k.startswith("node-") for k in ut):
            return ut
    return {}


def parse_report(path):
    d = json.load(open(path))
    pr = d.get("per_round", [])
    agg_id, worker_ids = ac.report_roles(d)   # BYTE-05: never hardcode node-0
    rows = []  # per round
    flows = []  # every per-sender t_eps, for the NT-04 quantization census
    splits = []  # every per-sender κ split, for the BYTE-04 columns
    for r in pr:
        rnd = r.get("round")
        nodes = r.get("nodes", {})
        # workers = nodes reporting a val_accuracy (exclude pure aggregator)
        wv = [n.get("val_accuracy") for n in nodes.values() if n.get("val_accuracy") is not None]
        wd = [n.get("round_duration_s") for n in nodes.values() if n.get("round_duration_s") is not None]
        val_mean = _mean(wv)
        round_dur = max(wd) if wd else None
        # per-sender uplink metrics
        ut = _find_uplink(nodes)
        senders = {k: v for k, v in ut.items() if k.startswith("node-")}
        teps, rsplits, wfired, shed_by = [], [], 0, {}
        for sname, s in senders.items():
            shed_by[sname] = set(s.get("shed_layers") or [])
            wf = bool(s.get("watchdog_fired"))
            if wf:
                wfired += 1
            else:
                t = ac.t_eps_receiver_s(s)
                teps.append(t)
                if t is not None:
                    flows.append(float(t))
            # BYTE-04: slippage and shed mass are two things; the sender's own
            # importance log is what resolves the split on pre-audit reports.
            ks = ac.kappa_split(s, ac.sender_importance(nodes.get(sname) or {}))
            if ks is not None:
                rsplits.append(ks)
        splits += rsplits
        clean = [t for t in teps if t is not None]
        # BYTE-05: worker uplink and aggregator downlink are different quantities
        up, agg_down, fresh_agg = [], None, []
        for nid, n in nodes.items():
            lcm = n.get("layer_comm_metrics")
            if not (isinstance(lcm, list) and lcm):
                continue
            total = sum(l.get("bytes_sent", 0) or 0 for l in lcm)
            if nid == agg_id:
                agg_down = total
                continue
            if worker_ids and nid not in worker_ids:
                continue
            up.append(total)
            shed = shed_by.get(nid, set())
            fresh_agg.append(sum((l.get("bytes_sent", 0) or 0) for l in lcm
                                 if l.get("layer_name") not in shed))
        wsend = [v for v in (ac.enqueue_duration_s(n) for n in nodes.values())
                 if v is not None]
        wbar = [n.get("barrier_wait_duration_s") for n in nodes.values() if n.get("barrier_wait_duration_s") is not None]
        wtrain = [n.get("train_duration_s") for n in nodes.values() if n.get("train_duration_s") is not None]
        rows.append({
            "round": rnd, "round_dur": round_dur, "val_mean": val_mean,
            # NT-04: max is the round statistic; the frozen mean is kept beside it
            "t_eps_max": max(clean) if clean else None,
            "t_eps_mean": _mean(clean) if clean else None,
            "kappa_slip_max": max((k.slip for k in rsplits
                                   if k.resolved and k.slip is not None), default=None),
            "recycled_mass_max": max((k.recycled for k in rsplits
                                      if k.resolved and k.recycled is not None), default=None),
            "watchdog": wfired,
            "worker_uplink_bytes_mean": _mean(up) if up else None,
            "agg_downlink_bytes": agg_down,
            "fresh_aggregated_bytes_mean": _mean(fresh_agg) if fresh_agg else None,
            "send_enqueue_mean": _mean(wsend), "barrier_mean": _mean(wbar),
            "train_mean": _mean(wtrain), "n_senders": len(senders),
        })
    # S11: the run-health verdict travels with the run so the per-arm medians
    # below can exclude a diverged run instead of averaging its timings in.
    return d.get("experiment", {}), rows, flows, splits, ac.read_run(d)


def achievable_floor(root):
    """Median EFT makespan over the eps=0 sender-rounds — the realizable floor.

    NT-02: monolithic's `send_duration_s` is a SENDER buffer-accept time and is
    not comparable with the receiver-side t_eps; the fluid `bytes*8/10 Mbps`
    value is unreachable for indivisible layers. This is the bound to compare
    a measured t_eps against.
    """
    ach = []
    for p in sorted(glob.glob(os.path.join(root, "exp6/eps0/seed*/results/report.json"))):
        ach += [b.achievable for b in ac.sender_bounds(json.load(open(p))) if b.achievable]
    return statistics.median(ach) if ach else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="campaigns/p1")
    ap.add_argument("--out", default=None, help="output dir for the CSVs (default --root)")
    args = ap.parse_args()
    out_dir = args.out or args.root
    reports = sorted(glob.glob(os.path.join(args.root, "*/*/seed*/results/report.json")))
    perround_rows, summary_rows = [], []
    by_arm = defaultdict(list)   # (exp,arm) -> list of summary dicts
    flows_by_arm = defaultdict(list)
    for rp in reports:
        parts = rp.split(os.sep)
        # .../campaigns/p1/<exp>/<arm>/seed<N>/results/report.json
        exp, arm, seedd = parts[-5], parts[-4], parts[-3]
        seed = int(seedd.replace("seed", ""))
        exper, rows, flows, splits, rr = parse_report(rp)
        if not rows:
            continue
        for r in rows:
            perround_rows.append({"exp": exp, "arm": arm, "seed": seed, **r})
        if rr.health.healthy:
            flows_by_arm[(exp, arm)] += flows
        last = rows[-1]
        s = {
            "exp": exp, "arm": arm, "seed": seed, "rounds": len(rows),
            "healthy": rr.health.healthy, "health_reason": rr.health.reason,
            "median_round_dur": _median([r["round_dur"] for r in rows]),
            "median_send_enqueue": _median([r["send_enqueue_mean"] for r in rows]),
            "median_barrier": _median([r["barrier_mean"] for r in rows]),
            "median_train": _median([r["train_mean"] for r in rows]),
            "final_val_acc": last["val_mean"],
            "median_t_eps_max": _median([r["t_eps_max"] for r in rows]),
            "median_t_eps": _median([r["t_eps_mean"] for r in rows]),  # frozen recipe
            "watchdog_fires": sum(r["watchdog"] for r in rows),
            "worker_uplink_bytes_mean": _mean([r["worker_uplink_bytes_mean"] for r in rows]),
            "agg_downlink_bytes_mean": _mean([r["agg_downlink_bytes"] for r in rows]),
            "fresh_aggregated_bytes_mean": _mean([r["fresh_aggregated_bytes_mean"] for r in rows]),
            **ac.kappa_summary(splits),   # BYTE-04: κ_slip and recycled mass, split
        }
        summary_rows.append(s)
        if rr.health.healthy:
            by_arm[(exp, arm)].append(s)   # S11: only healthy runs enter a median
        else:
            print(f"  !! {exp}/{arm}/seed{seed} EXCLUDED from the arm medians "
                  f"({rr.health.reason}, read={rr.health.read})")

    os.makedirs(out_dir, exist_ok=True)
    if perround_rows:
        with open(os.path.join(out_dir, "perround.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(perround_rows[0].keys())); w.writeheader(); w.writerows(perround_rows)
    if summary_rows:
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys())); w.writeheader(); w.writerows(summary_rows)

    # ---- headline table (across-seed) ----
    print(f"\n=== phase-1b headline ({len(summary_rows)} runs, {len(by_arm)} arms) ===")
    mono = by_arm.get(("exp4", "mono"), [])
    floor = achievable_floor(args.root)
    mono_send = _median([s["median_send_enqueue"] for s in mono]) if mono else None
    floor_round = _median([s["median_round_dur"] for s in mono]) if mono else None
    if floor is not None:
        print(f"ACHIEVABLE wire floor (EFT makespan over whole layers) = {floor:.3f}s"
              f"  <- compare t_eps to THIS")
    if mono_send is not None:
        print(f"monolithic {ac.CLOCK_SENDER_ENQUEUE} = {mono_send:.3f}s "
              f"(buffer-accept, NOT wire time — never compare it with t_eps)")
    if floor_round is not None:
        print(f"monolithic round_duration = {floor_round:.2f}s (train+barrier dominated)")
    hdr = (f"{'exp/arm':30} {'n':>2} {'tEps_max':>9} {'tEps_mean':>10} {'sendEnq':>8} "
           f"{'barrier':>8} {'train':>7} {'round':>7} {'acc':>7} {'κ_slip':>7} "
           f"{'upB/wrkr':>10} {'downB':>10} {'wd':>3}")
    print(hdr); print("-" * len(hdr))
    def fmt(x, p=2): return f"{x:.{p}f}" if isinstance(x, (int, float)) else "   -"
    for (exp, arm), ss in sorted(by_arm.items()):
        n = len(ss)
        mtx = _median([s["median_t_eps_max"] for s in ss])
        mtm = _median([s["median_t_eps"] for s in ss])
        msd = _median([s["median_send_enqueue"] for s in ss])
        mb = _median([s["median_barrier"] for s in ss])
        mtr = _median([s["median_train"] for s in ss])
        mr = _median([s["median_round_dur"] for s in ss])
        fa = _mean([s["final_val_acc"] for s in ss])
        mk = max([s["max_kappa_slip"] for s in ss
                  if s["max_kappa_slip"] is not None], default=None)
        # A κ_slip max over resolved flows only is a LOWER bound on the arm's
        # true max, so an unresolved flow must be visible beside it (BYTE-04).
        unres = sum(s["n_kappa_unresolved"] for s in ss)
        upb = _mean([s["worker_uplink_bytes_mean"] for s in ss])
        dnb = _mean([s["agg_downlink_bytes_mean"] for s in ss])
        wd = sum(s["watchdog_fires"] for s in ss)
        tag = ""
        if mtx is not None and floor is not None:
            tag = "  t_eps<floor" if mtx < floor else "  (>=floor)"
        if unres:
            tag += f"  κ_slip: {unres} flows unresolved (lower bound)"
        print(f"{exp+'/'+arm:30} {n:>2} {fmt(mtx,3):>9} {fmt(mtm,3):>10} {fmt(msd,3):>8} "
              f"{fmt(mb):>8} {fmt(mtr):>7} {fmt(mr):>7} {fmt(fa,4):>7} {fmt(mk,3):>7} "
              f"{fmt(upb,0):>10} {fmt(dnb,0):>10} {wd:>3}{tag}")

    # ---- NT-04: per-flow t_eps level census (10 ms bins) ----
    print("\n=== t_eps quantization census (per-flow, 10 ms bins) ===")
    print("  t_eps takes k x ~0.2005 s levels = whole 147.5 kB kernels on the 6 Mbps")
    print("  class, so eps moves the MIXTURE over k, not a continuum:")
    for (exp, arm), fl in sorted(flows_by_arm.items()):
        if not fl:
            continue
        levels = Counter(round(t, 2) for t in fl)
        top = ", ".join(f"{lv:.2f}s x{c}" for lv, c in sorted(levels.items()))
        print(f"  {exp+'/'+arm:24} n={len(fl):>4}  median={statistics.median(fl):.3f}s  {{{top}}}")

    print("\nReading guide: the PRIMITIVE lives in t_eps (coverage-completion, receiver-side,")
    print("MAX over senders) vs the ACHIEVABLE EFT floor. round=train+barrier+send; barrier-")
    print("wait is sync, not wire time (the measurement trap the paper flags). The kappa")
    print("column is SLIPPAGE only and must be <= eps on every coverage arm; it maxes over")
    print("the RESOLVED flows, so an arm tagged 'flows unresolved' reads as a lower bound.")
    print("Deliberately skipped mass is `max_recycled_mass_fraction` in summary.csv and eps")
    print("says nothing about it (BYTE-04). Byte columns are per-direction: upB/wrkr is UPLINK,")
    print("downB is the aggregator broadcast — never average the two (BYTE-05). All arm")
    print("medians are over HEALTHY runs only; excluded runs are listed above (S11).")


if __name__ == "__main__":
    main()
