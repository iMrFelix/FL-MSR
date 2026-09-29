"""Phase-1b headline contrasts: paired accuracy deltas vs monolithic, t_eps
reductions, and the EXP-8 deployed/opt scheduler-optimality summary.

Audited (writeup/19): the accuracy read is the last-3 window with the health
gate, not the single final round (S7/S11); the interval is the pre-registered
one-sided t-UCB from scripts/hypothesis_tests, not a hand-rolled 1.645·SE
(S6/S15); and t_eps is compared against the ACHIEVABLE EFT floor read from the
reports, not against monolithic's 0.379 s sender buffer-accept time, which is
a different clock domain (NT-02/NT-03).
"""
from __future__ import annotations
import json, os, math, statistics, csv, glob, sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/x.py` as well as `-m scripts.x`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import analysis_common as ac
from scripts import hypothesis_tests as ht

ROOT = "campaigns/p1"
SEEDS = [41, 42, 43, 44, 45, 46]
ARMS = [("eps0", "exp6/eps0"), ("eps02", "exp1/eps02"), ("eps03", "exp1/eps03"),
        ("eps04", "exp1/eps04"), ("c1_eps03", "exp5/c1_eps03")]
#: max-over-senders t_eps per arm (NT-04); the worker-MEAN values this table
#: used to carry (0.511/0.409/0.374/0.315) mixed the 0.2005 s quantization
#: levels.  Regenerate with scripts/extract_phase1b.py.
TE = {"eps0": 0.820, "eps02": 0.615, "eps03": 0.410, "eps04": 0.410, "c1_eps03": 0.378}


def read(p):
    """Last-3 val-accuracy read + health verdict for one run (S7/S11)."""
    return ac.read_run(json.load(open(p)))


def achievable_floor():
    """Median realizable EFT makespan over the eps=0 sender-rounds (NT-02)."""
    ach = []
    for p in sorted(glob.glob("%s/exp6/eps0/seed*/results/report.json" % ROOT)):
        ach += [b.achievable for b in ac.sender_bounds(json.load(open(p))) if b.achievable]
    return statistics.median(ach) if ach else None


def main():
    print("PAIRED last-3 accuracy delta vs monolithic (worker-mean, pp):")
    hdr = "%-12s %8s %6s %8s   per-seed" % ("arm", "mean", "SE", "95%UCB")
    print(hdr)
    for arm, path in ARMS:
        ds = []
        for s in SEEDS:
            mp = "%s/exp4/mono/seed%d/results/report.json" % (ROOT, s)
            cp = "%s/%s/seed%d/results/report.json" % (ROOT, path, s)
            if not (os.path.exists(mp) and os.path.exists(cp)):
                continue
            rm, rc = read(mp), read(cp)
            if not (rm.health.healthy and rc.health.healthy):
                print("  (seed%d dropped: mono=%s arm=%s)" % (
                    s, rm.health.reason or "ok", rc.health.reason or "ok"))
                continue
            ds.append((rc.lastk - rm.lastk) * 100)
        if len(ds) < 2:
            continue
        # cost framing (mono - arm) so the UCB is an upper bound on the LOSS
        ni = ht.noninferiority_ucb([-d for d in ds], margin=ac.NONINF_MARGIN_PP)
        print("%-12s %+8.2f %6.2f %+8.2f   %s" % (
            arm, statistics.mean(ds), ni.se, -ni.ucb, [round(x, 1) for x in ds]))
        print("%-12s   non-inferiority: cost %+.2f pp, one-sided 95%% UCB %+.2f vs "
              "%.1f pp margin -> %s" % ("", ni.mean, ni.ucb, ni.margin,
                                        "PASS" if ni.non_inferior else "FAIL"))

    floor = achievable_floor()
    print("\nt_eps (coverage-completion, MAX over senders) vs per-layer wait-for-all "
          "(eps0=%.3fs) and the ACHIEVABLE EFT wire floor (%s):"
          % (TE["eps0"], "%.3fs" % floor if floor else "unavailable"))
    for arm, te in TE.items():
        tag = "" if floor is None else (
            "   BELOW achievable floor" if te < floor else "   at/above achievable floor")
        print("  %-10s t_eps=%.3f = %3.0f%% of eps0%s" % (
            arm, te, te / TE["eps0"] * 100, tag))

    # EXP-8 deployed/opt
    cs = sorted(glob.glob("/tmp/optgap.csv"))
    if cs:
        rows = list(csv.DictReader(open(cs[0])))
        col = None
        for c in ("deployed_over_opt", "deployed/opt", "ratio_deployed_opt", "deployed_opt_ratio"):
            if rows and c in rows[0]:
                col = c
                break
        print("\nEXP-8 scheduler optimality (deep_cnn real manifests, %d instances):" % len(rows))
        print("  columns:", list(rows[0].keys()) if rows else "none")
        if col:
            vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
            print("  %s: min=%.4f max=%.4f mean=%.4f  (=1.0000 => deployed schedule is optimal)" % (
                col, min(vals), max(vals), sum(vals) / len(vals)))


if __name__ == "__main__":
    main()
