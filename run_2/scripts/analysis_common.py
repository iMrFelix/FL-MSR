#!/usr/bin/env python3
"""Shared read / health / statistics primitives for the campaign analyzers.

Single home for the estimators the pre-registration freezes (writeup/11 §2:
"Statistics (reuse only): scripts/hypothesis_tests.py … Never hand-roll"). The
measurement-validity audit (writeup/19) found four analyzers hand-rolling all
three of them, so this module is the one place they now live:

  * **intervals** — `tci95` (Student-t critical value, SAMPLE sd) instead of
    `1.96 * pstdev / √n`, which understated every published ± by 1.44× at n=6
    and 2.69× at n=3 (S6 / ML-05);
  * **endpoint reads** — `last_k` (k=3 at 20r, k=5 at 50r+) instead of the
    single final round, whose noise flipped the sign of a headline null (S7);
  * **run health** — `run_health` flags a run with any non-finite loss or a
    read-window accuracy pinned at chance, so collapsed runs are reported
    separately instead of being averaged into a "graded tradeoff" (S1/S11);
  * **inference** — `contrast_family` wires hypothesis_tests' paired t, exact
    sign-flip, sign test, Holm and the pre-registered non-inferiority UCB into
    the analyzers (S5/S9/S15);
  * **byte windows** — `cycle_stats` averages an INTEGER number of refresh
    cycles, so the frontier byte column stops aliasing the refresh period
    (BYTE-02);
  * **coverage accounting** — `flow_kappas` splits the one overloaded κ symbol
    into coverage slippage (what ε bounds) and deliberately-shed recycled mass,
    reconstructing the split on pre-audit reports from the sender's manifest
    scores, and `kappa_bound_mark` refuses a ≤ ε verdict while any flow is
    unresolved (BYTE-04);
  * **clock domains** — `require_one_clock` refuses a sender-side and a
    receiver-side time on the same axis (NT-03).

The health gate is not decorative: EVERY per-arm mean an analyzer publishes —
accuracy, bytes, traffic, timing, κ — is taken over healthy runs only, and an
arm with no healthy run publishes `NOT_EVALUABLE`, never a number.

tests/test_analysis_stats.py exercises all of it and fails if a script under
scripts/ starts hand-rolling an interval or an endpoint read again.
"""
from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from scripts import hypothesis_tests as ht

# --- frozen constants (writeup/11 §3) --------------------------------------
ALPHA = ht.ALPHA                      # 0.05
MIN_EFFECT_PP = 0.5                   # minimum ΔACC effect for a directional verdict
NONINF_MARGIN_PP = ht.NONINF_MARGIN_PP  # 1.0 pp non-inferiority margin

# --- run-health gate (CIFAR-10 defaults; pass explicit values for FEMNIST) --
CHANCE_ACC = 0.10                     # 10-class chance level
CHANCE_TOL = 0.005                    # "pinned at chance" tolerance
HEALTH_FLOOR = 0.12                   # a healthy run reads above chance+2pp
LOSS_BLOWUP = 1e6                     # a finite-but-absurd loss is a divergence too

# --- clock-domain tags (NT-03) ---------------------------------------------
CLOCK_RECEIVER = "t_cover_receiver_s"        # aggregator-side coverage completion
CLOCK_SENDER_ENQUEUE = "t_send_enqueue_sender_s"  # sender buffer-accept (NOT wire time)


# ===========================================================================
# Report reading: roles, series, endpoint estimators
# ===========================================================================

def split_roles(nodes: Mapping[str, Any]) -> tuple[str | None, set[str]]:
    """(aggregator_id, worker_ids) inferred from the telemetry, not hardcoded.

    The aggregator is the node whose ``uplink_telemetry`` holds per-source
    entries (``node-*`` keys) or the ``_aggregation`` block; workers carry a
    ``_sender`` block. BYTE-05: the aggregator's broadcast must never be
    averaged into a worker-uplink figure, and node-0 is a convention, not a
    guarantee.
    """
    agg = None
    workers: set[str] = set()
    for nid, nd in nodes.items():
        ut = nd.get("uplink_telemetry") if isinstance(nd, Mapping) else None
        if isinstance(ut, Mapping):
            if "_aggregation" in ut or any(k.startswith("node-") for k in ut):
                agg = nid
                continue
            if "_sender" in ut:
                workers.add(nid)
    if agg is None:
        return None, set(nodes) - {"node-0"} if "node-0" in nodes else set(nodes)
    return agg, (workers or (set(nodes) - {agg}))


def report_roles(rep: Mapping[str, Any]) -> tuple[str | None, set[str]]:
    """Roles for a whole report (first round that carries telemetry wins)."""
    for rd in rep.get("per_round", []):
        agg, workers = split_roles(rd.get("nodes", {}))
        if agg is not None and workers:
            return agg, workers
    nodes = rep.get("per_round", [{}])[0].get("nodes", {}) if rep.get("per_round") else {}
    return None, set(nodes) - {"node-0"}


def val_series(rep: Mapping[str, Any], workers: Iterable[str] | None = None
               ) -> tuple[list[int], list[float], list[float]]:
    """(rounds, worker-mean val_accuracy, worker-mean val_loss) per round."""
    ws = set(workers) if workers is not None else report_roles(rep)[1]
    rounds, acc, loss = [], [], []
    for rd in rep.get("per_round", []):
        nodes = rd.get("nodes", {})
        vs = [float(d["val_accuracy"]) for n, d in nodes.items()
              if n in ws and d.get("val_accuracy") is not None]
        ls = [float(d["val_loss"]) for n, d in nodes.items()
              if n in ws and d.get("val_loss") is not None]
        rounds.append(rd.get("round"))
        acc.append(st.mean(vs) if vs else float("nan"))
        loss.append(st.mean(ls) if ls else float("nan"))
    return rounds, acc, loss


#: Field spellings for the two timing quantities, newest first. NT-01 renamed
#: the buffer-accept duration (it is not a wire time) and NT-03 added the
#: clock-domain suffix; reports predating either change keep the older keys.
ENQUEUE_KEYS = ("send_enqueue_duration_sender_s", "send_enqueue_duration_s",
                "send_duration_s")
TEPS_KEYS = ("t_eps_local_receiver_s", "t_eps_local")
#: summary.csv enqueue-time column, new spelling first.  Campaign CSVs on
#: disk predate the NT-01 rename and still carry `median_send`; regenerating
#: one in place must not break the C4/Pareto exhibits that read it.
SUMMARY_ENQUEUE_KEYS = ("median_send_enqueue", "median_send")


def _first_present(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def enqueue_duration_s(node: Mapping[str, Any]) -> float | None:
    """Sender buffer-accept duration (CLOCK_SENDER_ENQUEUE domain), any spelling."""
    return _first_present(node, ENQUEUE_KEYS)


def t_eps_receiver_s(flow: Mapping[str, Any]) -> float | None:
    """Receiver-side coverage-completion time (CLOCK_RECEIVER domain)."""
    return _first_present(flow, TEPS_KEYS)


def summary_enqueue_s(row: Mapping[str, Any]) -> float | None:
    """Enqueue-time column of a summary.csv row, either spelling.

    CLOCK_SENDER_ENQUEUE domain — a buffer-accept duration, never a wire
    time (audit NT-01); callers must not place it on a receiver-clock axis.
    """
    v = _first_present(row, SUMMARY_ENQUEUE_KEYS)
    return None if v in (None, "") else float(v)


def last_k(xs: Sequence[float], k: int = 3) -> float | None:
    """Mean of the last k finite values — the S7 read (k=3 at 20r, 5 at 50r+)."""
    vals = [x for x in xs[-k:] if x is not None and math.isfinite(x)]
    return st.mean(vals) if vals else None


def default_k(n_rounds: int) -> int:
    """Pre-registered window: 3 rounds at a 20-round horizon, 5 beyond it."""
    return 3 if n_rounds <= 20 else 5


# ===========================================================================
# Run-health gate  (S1 / S11 / S13)
# ===========================================================================

@dataclass(frozen=True)
class RunHealth:
    """Verdict on one run: may its numbers enter a mean?"""

    healthy: bool
    reason: str            # "" when healthy
    nan_round: int | None  # index of the first non-finite/blown-up round
    peak: float | None     # best worker-mean val accuracy
    read: float | None     # last-k read the gate was applied to

    def as_dict(self) -> dict[str, Any]:
        return {"healthy": self.healthy, "reason": self.reason,
                "nan_round": self.nan_round, "peak": self.peak, "read": self.read}


def first_bad_round(loss: Sequence[float], acc: Sequence[float] = ()) -> int | None:
    """Index of the first non-finite (or > LOSS_BLOWUP) loss / non-finite acc."""
    for i, v in enumerate(loss):
        if v is None or not math.isfinite(v) or v > LOSS_BLOWUP:
            return i
    for i, v in enumerate(acc):
        if v is None or not math.isfinite(v):
            return i
    return None


def run_health(acc: Sequence[float], loss: Sequence[float], k: int = 3,
               chance: float = CHANCE_ACC, floor: float = HEALTH_FLOOR) -> RunHealth:
    """Pre-registered health criterion, applied to the last-k read window.

    A run is UNHEALTHY when (in priority order) it has a non-finite/blown-up
    val_loss at any round, its read sits at chance (±0.5pp), or the read is
    below ``floor`` — distinguishing a post-peak collapse from a run that
    never learned. Unhealthy runs are excluded from means and reported
    separately; they are not silently averaged (S1 set the sign of a headline
    from a collapsed *baseline*; S11 averaged three chance-level runs into a
    "graded tradeoff").
    """
    finite = [a for a in acc if a is not None and math.isfinite(a)]
    peak = max(finite) if finite else None
    read = last_k(acc, k)
    bad = first_bad_round(loss, acc)
    if bad is not None:
        return RunHealth(False, f"nonfinite_loss@r{bad}", bad, peak, read)
    if read is None:
        return RunHealth(False, "no_val_data", None, peak, None)
    if abs(read - chance) <= CHANCE_TOL:
        return RunHealth(False, "pinned_at_chance", None, peak, read)
    if read < floor:
        why = "post_peak_collapse" if (peak or 0.0) >= floor else "never_learned"
        return RunHealth(False, why, None, peak, read)
    return RunHealth(True, "", None, peak, read)


@dataclass(frozen=True)
class RunRead:
    """One run reduced to the two reads plus its health verdict."""

    seed: int | None
    lastk: float | None    # primary read (S7)
    endpoint: float | None # single final round, kept as a sensitivity column
    k: int
    health: RunHealth


def read_run(rep: Mapping[str, Any], k: int | None = None, seed: int | None = None,
             workers: Iterable[str] | None = None, **health_kw: float) -> RunRead:
    """Read a report with the fixed semantics: last-k mean + health gate."""
    _, acc, loss = val_series(rep, workers)
    kk = k if k is not None else default_k(len(acc))
    end = next((a for a in reversed(acc) if a is not None and math.isfinite(a)), None)
    return RunRead(seed, last_k(acc, kk), end, kk, run_health(acc, loss, kk, **health_kw))


# ===========================================================================
# Intervals  (S6 / ML-05)
# ===========================================================================

def tci95(xs: Sequence[float], conf: float = 0.95) -> float:
    """t-based CI half-width with the SAMPLE sd: t_{conf,n-1}·s/√n.

    Replaces `1.96 * pstdev / √n`, whose actual coverage at the published n
    is 87% (n=6) / 75% (n=3), not 95%.
    """
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    n = len(xs)
    if n < 2:
        return float("nan")
    t = ht.student_t_ppf(0.5 * (1.0 + conf), n - 1)
    return float(t * st.stdev(xs) / math.sqrt(n))


def mean_ci(xs: Sequence[float], conf: float = 0.95) -> tuple[float | None, float, int]:
    """(mean, t-CI half-width, n) over the finite values."""
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:
        return None, float("nan"), 0
    return st.mean(xs), tci95(xs, conf), len(xs)


# ===========================================================================
# Inference: the pre-registered ladder, wired  (S5 / S9 / S15)
# ===========================================================================

@dataclass(frozen=True)
class ContrastRow:
    """One arm-vs-baseline contrast with its full pre-registered verdict."""

    label: str
    n: int
    seeds: tuple[int, ...]
    mean: float            # ΔACC (arm − baseline), pp
    ci95: float            # honest t-CI half-width
    t_p: float
    perm_p: float
    sign_p: float
    holm_p: float          # Holm-adjusted within the family
    perm_floor: float      # 1/2^n — the smallest p the exact test can return
    verdict: str
    ni_cost: float         # baseline − arm, pp (positive ⇒ the arm lost accuracy)
    ni_ucb: float          # one-sided 95% upper bound on the cost
    ni_pass: bool          # UCB < margin
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["seeds"] = list(self.seeds)
        d["notes"] = list(self.notes)
        return d


def contrast_family(diffs_by_label: Mapping[str, Sequence[float]],
                    seeds_by_label: Mapping[str, Sequence[int]] | None = None,
                    direction: str = "less",
                    min_effect: float = MIN_EFFECT_PP,
                    margin: float = NONINF_MARGIN_PP,
                    alpha: float = ALPHA) -> list[ContrastRow]:
    """Run one Holm family of paired contrasts through hypothesis_tests.

    ``diffs_by_label`` maps an arm label to its per-seed paired ΔACC in pp
    (arm − baseline, same seed). Every contrast gets the paired one-sided t,
    the exact 2^n sign-flip test, the sign test, the Holm-adjusted p WITHIN
    this family (S9: nine uncorrected arm-vs-mono contrasts were published),
    the §3 verdict computed on the adjusted p, and the pre-registered
    non-inferiority UCB on the cost framing d = baseline − arm (S5: the
    frozen ≤1pp rule was never applied to the data it was frozen for).

    ``direction='less'`` is the shedding-costs-accuracy hypothesis on ΔACC.
    """
    usable = {lab: [float(x) for x in ds] for lab, ds in diffs_by_label.items()
              if len([x for x in ds if x is not None]) >= 2}
    labels = list(usable)
    raw_p = [ht.paired_t_one_sided(usable[lab], direction=direction).p for lab in labels]
    adj = dict(zip(labels, ht.holm(raw_p, labels, alpha=alpha).adj_p)) if labels else {}

    rows: list[ContrastRow] = []
    for lab, ds in diffs_by_label.items():
        sd = tuple(seeds_by_label.get(lab, ())) if seeds_by_label else ()
        if lab not in usable:
            rows.append(ContrastRow(lab, len(list(ds)), sd, float("nan"), float("nan"),
                                    float("nan"), float("nan"), float("nan"),
                                    float("nan"), float("nan"), ht.NOT_EVALUABLE,
                                    float("nan"), float("nan"), False,
                                    ("n<2 healthy pairs",)))
            continue
        d = usable[lab]
        v = ht.directional_verdict(d, direction, min_effect, label=lab,
                                   alpha=alpha, p_for_verdict=adj[lab])
        ni = ht.noninferiority_ucb([-x for x in d], margin=margin)
        notes = list(v.notes)
        if v.perm_min_p > alpha:
            # At n=3 the exact sign-flip floor is 1/8 = 0.125 > α, so the ROBUST
            # tier is unreachable and the "robust" label would be vacuous (S9/S11:
            # no frontier point has inferential support at n=3).
            notes.append(f"exact test cannot reach α at n={len(d)} "
                         f"(sign-flip floor {v.perm_min_p:.3f})")
        rows.append(ContrastRow(
            label=lab, n=len(d), seeds=sd, mean=st.mean(d), ci95=tci95(d),
            t_p=v.t_p, perm_p=v.perm_p, sign_p=v.sign_p, holm_p=adj[lab],
            perm_floor=v.perm_min_p, verdict=v.verdict, ni_cost=ni.mean,
            ni_ucb=ni.ucb, ni_pass=ni.non_inferior, notes=tuple(notes)))
    return rows


def format_family(rows: Sequence[ContrastRow], margin: float = NONINF_MARGIN_PP) -> str:
    """Fixed-width verdict table: ΔACC ± honest CI, Holm p, NI verdict."""
    hdr = (f"  {'arm':<22}{'n':>3}{'ΔACC±CI (pp)':>18}{'t p':>9}{'perm p':>9}"
           f"{'holm p':>9}  {'verdict':<18}{'NI cost/UCB':>14}  NI")
    out = [hdr, "  " + "-" * (len(hdr) - 2)]
    for r in rows:
        if r.verdict == ht.NOT_EVALUABLE:
            out.append(f"  {r.label:<22}{r.n:>3}{'--':>18}  (not evaluable: "
                       f"{'; '.join(r.notes)})")
            continue
        flag = "†" if r.perm_floor > ALPHA else " "
        out.append(
            f"  {r.label:<22}{r.n:>3}{r.mean:>+10.2f} ±{r.ci95:>6.2f}{r.t_p:>9.4f}"
            f"{r.perm_p:>9.4f}{r.holm_p:>9.4f}  {r.verdict:<17}{flag}"
            f"{r.ni_cost:>+7.2f}/{r.ni_ucb:>+6.2f}  "
            f"{'PASS' if r.ni_pass else 'FAIL'}")
    out.append(f"  (NI = non-inferiority vs the frozen {margin:.1f}pp margin, "
               f"cost framing baseline−arm; Holm within this family only.")
    out.append("   † n too small for the exact sign-flip test to reach α — the "
               "ROBUST tier is vacuous there.)")
    return "\n".join(out)


# ===========================================================================
# Byte windows: refresh-cycle alignment  (BYTE-02)
# ===========================================================================

def cycle_window(rounds: Sequence[int], tau: int | None, r0: int = 3) -> list[int]:
    """Rounds >= r0 truncated to an INTEGER number of τ-length refresh cycles.

    A fixed r0..R truncation contains a fractional number of refresh cycles,
    so the mean is phase-dependent — worst at large τ, where it collapsed two
    frontier points into one (BYTE-02). With τ unknown (aging OFF: no cycle)
    every round from r0 is used.
    """
    rs = sorted(r for r in rounds if r is not None and r >= r0)
    if not rs or not tau or tau < 1:
        return rs
    ncyc = len(rs) // tau
    return rs[:ncyc * tau] if ncyc >= 1 else rs


@dataclass(frozen=True)
class CycleStats:
    """Cycle-aligned mean of a per-round series plus its honest dispersion."""

    mean: float | None
    cycle_spread: float    # max−min over whole-cycle means (nan if <2 cycles)
    n_cycles: int
    rounds_used: tuple[int, ...]
    aligned: bool          # False ⇒ fewer than one full cycle, mean is phase-dependent


def cycle_stats(series: Mapping[int, float], tau: int | None, r0: int = 3) -> CycleStats:
    """Mean over whole refresh cycles + the cycle-to-cycle spread."""
    rs = cycle_window(list(series), tau, r0)
    vals = [series[r] for r in rs if series.get(r) is not None]
    if not vals:
        return CycleStats(None, float("nan"), 0, (), False)
    ncyc = (len(rs) // tau) if (tau and tau >= 1) else 0
    spread = float("nan")
    if ncyc >= 2:
        cyc = [st.mean([series[r] for r in rs[i * tau:(i + 1) * tau]])
               for i in range(ncyc)]
        spread = max(cyc) - min(cyc)
    return CycleStats(st.mean(vals), spread, ncyc, tuple(rs), bool(ncyc >= 1))


def window_band(fn: Callable[[int], float | None],
                r0s: Sequence[int] = (3, 5, 8, 10)) -> tuple[float, float]:
    """(min, max) of a windowed statistic over the window START — the
    instrument BYTE-02 shows is the right one (a first/second-half split
    probes stationarity *within* a window, not the aliasing across it)."""
    vals = [v for v in (fn(r0) for r0 in r0s) if v is not None and math.isfinite(v)]
    return (min(vals), max(vals)) if vals else (float("nan"), float("nan"))


# ===========================================================================
# Coverage accounting: slippage vs deliberately-shed mass  (BYTE-04)
# ===========================================================================

#: Printed instead of a number wherever the health gate leaves an arm with no
#: usable run — a missing mean must not read as a small one.
NOT_EVALUABLE = ht.NOT_EVALUABLE


@dataclass(frozen=True)
class KappaSplit:
    """One uplink flow's coverage accounting with the two masses separated.

    ``kappa_realized`` carries both meanings at once, which is how the frontier
    table came to print κ up to 0.998 next to the invariant "κ ≤ ε" (BYTE-04):
    at τ=8 five of six shed layers had never been sent at all.  It still means
    the TOTAL in every report, old and new — the audit fix added `kappa_slip`
    beside it rather than redefining it, so nothing here has to guess which
    definition a given file was written under.
    """

    slip: float | None       # coverage SLIPPAGE — the mass the ε bound speaks about
    recycled: float | None   # deliberately skip-omitted mass, recycle-filled by design
    conflated: float | None  # slip + recycled = `kappa_realized`, in every report
    resolved: bool           # False ⇒ the mass split is not recoverable from this file
    n_shed: int              # shed layers at completion
    n_recycled: int          # of those, the ones the sender was advised to skip
    n_slip: int              # of those, the genuinely-late ones
    source: str              # "engine_split"|"layer_count"|"sender_importance"|"unresolved"
    recon_err: float | None = None   # |reconstruction − exact| where both routes apply


#: Reconstructed and exactly-known slippage may differ only by float noise; the
#: manifest scores are float32 on the wire, so this is generous by ~7 decades.
KAPPA_RECON_TOL = 1e-6


def sender_importance(node_entry: Mapping[str, Any]) -> dict[str, float]:
    """One sender-round's transmitted layer → manifest raw_score.

    ``layer_comm_metrics[].importance`` IS the frozen manifest score on the
    uplink: `_send_updates` feeds the same ``plan.trigger_scores`` into
    `build_manifest` and into `serialize_layer_updates` (engine.py §_dispatch),
    so the sender's own log carries the numerator terms the receiver's flow
    record does not.  A layer sent to several destinations appears once per
    destination with the same score; a name whose entries disagree is dropped
    rather than averaged, so an ambiguous score can only ever cost resolution.
    """
    out: dict[str, float] = {}
    bad: set[str] = set()
    for lm in node_entry.get("layer_comm_metrics") or ():
        if not isinstance(lm, Mapping):
            continue
        name, imp = lm.get("layer_name"), lm.get("importance")
        if name is None or imp is None:
            continue
        imp = float(imp)
        if name in out and abs(out[name] - imp) > KAPPA_RECON_TOL:
            bad.add(name)                # same layer, two scores: unusable
        out[name] = imp
    for name in bad:
        out.pop(name, None)
    return out


def _reconstruct_slip(
    kappa: float, shed: Sequence[str], skipped: Iterable[str],
    scores: Mapping[str, float],
) -> float | None:
    """Slippage mass fraction from the SENDER's importance log, or None.

    The manifest denominator is gone from the receiver's flow record, but it is
    implied: everything the sender transmitted and the receiver did not shed is
    covered mass, so

        total   = received / (1 − κ)          (κ < 1, received > 0)
        κ_slip  = Σ importance[shed ∧ ¬skip-omitted] / total

    The late layers are exactly the ones that WERE sent, so their mass is on
    disk; only the skip-omitted layers are absent, and they never enter this
    numerator.  Returns None whenever the identity is not usable — κ ≥ 1 (no
    covered mass to scale from), nothing received, a late layer missing from
    the sender's log (a straggler tail that completed after the report shipped),
    or a result outside [0, κ], which would mean the two sides of the join do
    not describe the same round.
    """
    skipped = set(skipped)
    late = [name for name in shed if name not in skipped]
    if kappa >= 1.0 - KAPPA_RECON_TOL or any(name not in scores for name in late):
        return None
    shed_set = set(shed)
    received = sum(v for name, v in scores.items() if name not in shed_set)
    if received <= 0.0:
        return None
    slip = sum(scores[name] for name in late) * (1.0 - kappa) / received
    if slip < -KAPPA_RECON_TOL or slip > kappa + KAPPA_RECON_TOL:
        return None                      # join mismatch — refuse, do not publish
    return min(max(slip, 0.0), kappa)


def kappa_split(
    flow: Mapping[str, Any],
    scores: Mapping[str, float] | None = None,
) -> KappaSplit | None:
    """Split one flow's shed mass; ``None`` when the flow reports no κ at all.

    THE ONLY κ READ in this codebase: every consumer goes through here, because
    reading `kappa_realized` on its own gives the conflated total in both
    generations of report and comparing that against ε is precisely BYTE-04.

    Three resolution sources, in order of directness:

    * ``engine_split`` — post-audit reports carry the split explicitly
      (`kappa_slip` slippage, `shed_mass_fraction` recycled, `kappa_realized`
      their sum), and the presence of `kappa_slip` is the discriminator;
    * ``layer_count`` — set logic settles both ends of a pre-audit flow: no
      skip-omitted layer in the shed set makes the whole number slippage, a
      wholly skip-omitted shed set makes the slippage exactly zero;
    * ``sender_importance`` — everything in between is recovered from the
      sender's own `layer_comm_metrics` (pass ``scores`` from
      `sender_importance`), which carries the manifest score of every layer it
      transmitted; see `_reconstruct_slip` for the identity.

    Only a flow none of the three can settle is ``unresolved``, and it is
    reported as such rather than guessed.  Where both set logic and the
    reconstruction apply, the exact value is published and the discrepancy is
    kept in ``recon_err`` — that self-check is what licenses the reconstructed
    numbers elsewhere in the same campaign (on wave-3: ≤ 1.1e-16 over the 1908
    flows set logic settles, and every one of the remaining 432 resolved, where
    before the reconstruction each frontier arm's bound check ran on a minority
    of its flows).
    """
    if not isinstance(flow, Mapping) or "kappa_realized" not in flow:
        return None
    kappa = flow.get("kappa_realized")
    kappa = float(kappa) if kappa is not None else None
    shed = list(flow.get("shed_layers") or [])
    skipped = set(flow.get("skip_omitted_layers") or ())
    n_shed = len(shed)
    n_rec = sum(1 for name in shed if name in skipped)
    n_slip = n_shed - n_rec

    if flow.get("kappa_slip") is not None:              # engine already split it
        slip = float(flow["kappa_slip"])
        rec = flow.get("shed_mass_fraction")
        rec = float(rec) if rec is not None else (
            None if kappa is None else kappa - slip)
        tot = kappa if kappa is not None else (
            None if rec is None else slip + rec)
        return KappaSplit(slip, rec, tot, True, n_shed, n_rec, n_slip, "engine_split")
    if kappa is None:
        return KappaSplit(None, None, None, False, n_shed, n_rec, n_slip, "unresolved")
    recon = (_reconstruct_slip(kappa, shed, skipped, scores)
             if scores else None)
    if n_rec == 0 or n_slip == 0:
        # Set logic is exact at both ends; the reconstruction is graded against
        # it here rather than trusted on its own word.
        exact = kappa if n_rec == 0 else 0.0
        err = None if recon is None else abs(recon - exact)
        return KappaSplit(exact, kappa - exact, kappa, True,
                          n_shed, n_rec, n_slip, "layer_count", err)
    if recon is not None:
        return KappaSplit(recon, kappa - recon, kappa, True,
                          n_shed, n_rec, n_slip, "sender_importance")
    return KappaSplit(None, None, kappa, False, n_shed, n_rec, n_slip, "unresolved")


def flow_kappas(rep: Mapping[str, Any]) -> list[KappaSplit]:
    """Every uplink flow's κ split, over every round and every receiver.

    The sender's importance log lives on ITS node entry in the same round, so
    the flows are read against the round's node map (BYTE-04 reconstruction).
    """
    out: list[KappaSplit] = []
    for rd in rep.get("per_round", []):
        nodes = rd.get("nodes", {})
        scores = {nid: sender_importance(nd) for nid, nd in nodes.items()
                  if isinstance(nd, Mapping)}
        for _, nd in nodes.items():
            ut = nd.get("uplink_telemetry") if isinstance(nd, Mapping) else None
            if not isinstance(ut, Mapping):
                continue
            for key, flow in ut.items():
                if key.startswith("_"):     # reserved blocks, not source flows
                    continue
                ks = kappa_split(flow, scores.get(key))
                if ks is not None:
                    out.append(ks)
    return out


def kappa_summary(splits: Sequence[KappaSplit]) -> dict[str, Any]:
    """Arm-level κ columns: slippage and recycled mass as SEPARATE quantities.

    ``max_kappa_slip`` is the Claim-A bound check.  It is a LOWER bound on the
    arm's true maximum whenever any flow is unresolved, so it may be compared
    against ε only when ``n_kappa_unresolved`` is 0 — see `kappa_bound_mark`,
    which is the only place the ≤ ε verdict is minted.  ``max_kappa_slip_ub``
    covers the unresolved flows too (it degenerates to the old conflated column
    there); ``kappa_recon_max_err`` is the reconstruction's self-check against
    the flows set logic settles independently.
    """
    if not splits:
        # Stable schema: a fully-collapsed arm still owns every column, empty —
        # a missing header would silently drop the arm out of the CSV instead of
        # marking it NOT EVALUABLE.
        return {"kappa_source": None, "n_kappa_flows": 0, "n_kappa_unresolved": 0,
                "n_kappa_reconstructed": 0, "n_kappa_recon_checks": 0,
                "kappa_recon_max_err": None,
                "max_kappa_slip": None, "max_kappa_slip_ub": None,
                "max_recycled_mass_fraction": None, "mean_shed_layers": None,
                "mean_recycled_layers": None, "mean_slip_layers": None}
    slips = [s.slip for s in splits if s.resolved and s.slip is not None]
    recs = [s.recycled for s in splits if s.resolved and s.recycled is not None]
    ubs = [s.slip if (s.resolved and s.slip is not None) else s.conflated
           for s in splits if (s.conflated is not None or s.slip is not None)]
    unresolved = [s for s in splits if not s.resolved]
    errs = [s.recon_err for s in splits if s.recon_err is not None]
    sources = sorted({s.source for s in splits})
    return {
        "kappa_source": sources[0] if len(sources) == 1 else "mixed",
        "n_kappa_flows": len(splits),
        "n_kappa_unresolved": len(unresolved),
        "n_kappa_reconstructed": sum(1 for s in splits
                                     if s.source == "sender_importance"),
        "n_kappa_recon_checks": len(errs),
        "kappa_recon_max_err": max(errs) if errs else None,
        "max_kappa_slip": round(max(slips), 4) if slips else None,
        "max_kappa_slip_ub": round(max(x for x in ubs if x is not None), 4) if ubs else None,
        "max_recycled_mass_fraction": round(max(recs), 4) if recs else None,
        "mean_shed_layers": round(st.mean([s.n_shed for s in splits]), 2),
        "mean_recycled_layers": round(st.mean([s.n_recycled for s in splits]), 2),
        "mean_slip_layers": round(st.mean([s.n_slip for s in splits]), 2),
    }


def kappa_bound_mark(row: Mapping[str, Any], eps: float = 0.3) -> str:
    """The κ_slip ≤ ε verdict, minted only where the data can support it.

    ``max_kappa_slip`` is a maximum over the RESOLVED flows, i.e. a lower bound
    on the arm's true maximum: stamping "≤ ε" on it while some flow's slippage
    is unknown asserts an upper-bound constraint from a lower bound, which is
    the same species of error BYTE-04 exists to fix.  So an arm with any
    unresolved flow gets no verdict at all, and one whose lower bound already
    exceeds ε gets the violation (that direction IS sound from below).
    """
    slip, n_unres = row.get("max_kappa_slip"), row.get("n_kappa_unresolved") or 0
    if slip is None:
        return "  no κ_slip"
    if slip > eps + 1e-9:
        return "  ✗ >ε"
    if n_unres:
        return f"  ? {n_unres} unresolved — no verdict"
    return "  ✓ ≤ε"


# ===========================================================================
# Clock domains  (NT-03)
# ===========================================================================

def require_one_clock(tags: Iterable[str], axis: str = "time axis") -> str:
    """Refuse to put two clock domains on one axis; return the single tag.

    The C4 error (a sender buffer-accept time compared against receiver-side
    coverage completion) reappeared inside pareto_v1 precisely because both
    quantities were bare floats named ``*_s``.
    """
    seen = sorted({t for t in tags if t})
    if len(seen) > 1:
        raise ValueError(
            f"mixed clock domains on the {axis}: {seen} — a sender-side "
            "buffer-accept time is not comparable with a receiver-side "
            "completion time (NT-03); drop the odd one out or add a "
            "receiver-side stamp for it (NT-05)")
    return seen[0] if seen else ""


# ===========================================================================
# Wire bounds  (NT-02)
# ===========================================================================

@dataclass(frozen=True)
class WireBounds:
    """The two floors the framework already computes per sender-round."""

    achievable: float | None  # EFT makespan over indivisible layers (realizable)
    fluid: float | None       # LP/fluid relaxation (NOT realizable)

    @property
    def granularity(self) -> float:
        """achievable / fluid — the irreducible whole-layer penalty."""
        if not self.achievable or not self.fluid:
            return float("nan")
        return self.achievable / self.fluid


def sender_bounds(rep: Mapping[str, Any]) -> list[WireBounds]:
    """Per sender-round achievable (EFT) and fluid bounds from the reports.

    NT-02: the published "0.600 s analytic wire floor" is the fluid split of
    the bytes across the three traffic classes, which indivisible 147.5 kB
    kernels can never reach. The achievable bound (``realized_makespan_s`` /
    ``predicted_t_eps`` / max ``eft_head_loads_s``) is recorded next to it in
    every report and is the floor a measured t_ε must be compared against.
    """
    out: list[WireBounds] = []
    for rd in rep.get("per_round", []):
        for _, nd in rd.get("nodes", {}).items():
            s = (nd.get("uplink_telemetry") or {}).get("_sender")
            if not isinstance(s, Mapping):
                continue
            diag = s.get("assignment_diagnostics") or {}
            loads = diag.get("eft_head_loads_s") or {}
            ach = diag.get("realized_makespan_s")
            if ach is None:
                # Post-NT-03 key first, pre-fix campaigns second.
                ach = s.get("predicted_t_eps_model_s", s.get("predicted_t_eps"))
            if ach is None and loads:
                ach = max(float(v) for v in loads.values())
            fluid = diag.get("fluid_bound_head_s", diag.get("lp_fluid_bound_s"))
            if ach is None and fluid is None:
                continue
            out.append(WireBounds(float(ach) if ach is not None else None,
                                  float(fluid) if fluid is not None else None))
    return out
