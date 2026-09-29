"""Tests for the audited analysis/statistics layer (writeup/19 track T3).

Every check here pins one finding from the measurement-validity audit so the
defect cannot silently return:

- **S6 / ML-05** — `tci95` uses the Student-t critical value and the SAMPLE sd.
  Pinned against the audit's own regenerated numbers (drop ±1.42 -> ±2.04 at
  n=6; the n=3 understatement factor 2.69x).
- **S7** — `last_k` is the endpoint read; the single final round is only a
  sensitivity value.
- **S1 / S11** — `run_health` flags non-finite losses, chance-pinned reads and
  post-peak collapses, and the analyzers exclude those runs from EVERY mean —
  ΔACC (including when the collapsed run is the paired DENOMINATOR), bytes,
  traffic, savings denominators, t_eps, timing, κ and the importance grid. An
  arm with no healthy run publishes NOT EVALUABLE, never a number.
- **BYTE-04** — κ is published as two columns: coverage slippage (which ε
  bounds) and deliberately-shed recycled mass (which it does not). Pre-fix
  reports resolve exactly at both ends and, in between, from the sender's own
  manifest scores — never relabelled as slippage, and never bounded away as
  unresolvable while the mass is still on disk. The ≤ ε verdict is withheld
  while any flow is unresolved, because a max over resolved flows is a LOWER
  bound on the arm's true max.
- **S6 / S11 (provenance)** — no analyzer reads a derived CSV back in, so a
  pre-fix artifact cannot land in the same table as a post-fix computation;
  analyze_cp2 recomputes the wave-3 frontier through the shared reader and
  warns loudly when the reports are absent.
- **S5 / S9 / S15** — `contrast_family` really calls hypothesis_tests: Holm
  within the family, the exact sign-flip p, and the frozen 1.0pp
  non-inferiority UCB.
- **BYTE-02** — byte windows cover an integer number of refresh cycles and the
  window-start band exposes the aliasing.
- **BYTE-03 / BYTE-05** — worker uplink and aggregator downlink stay separate
  quantities; the aggregator is found from telemetry, not hardcoded.
- **NT-02** — the achievable EFT floor is read from the reports, not a fluid
  relaxation.
- **NT-03** — one axis, one clock domain.
- **NT-04** — the round t_eps statistic is the max over senders.
- **S15 (enforcement)** — a repo guard that FAILS if any script under scripts/
  hand-rolls an interval or an endpoint estimator again.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
from pathlib import Path

import pytest

from scripts import analysis_common as ac
from scripts import hypothesis_tests as ht

REPO = Path(__file__).resolve().parent.parent


# ===========================================================================
# Synthetic campaign fixtures
# ===========================================================================

LAYERS = [  # deep_cnn-shaped: five fat kernels + small biases
    ("conv_0/kernel", 147_536, 0.05), ("conv_1/kernel", 147_536, 0.10),
    ("conv_2/kernel", 147_536, 0.20), ("conv_3/kernel", 147_536, 0.30),
    ("conv_4/kernel", 147_536, 0.30), ("head/bias", 1_000, 0.05),
]


def make_report(acc_by_round, loss_by_round=None, workers=("node-1", "node-2", "node-3"),
                agg="node-0", uplink_by_round=None, downlink=2_248_191,
                teps_by_round=None, sender_diag=None, shed=(),
                kappa=0.0, skip_omitted=None, shed_mass=None):
    """A minimal report.json with the fields the analyzers actually read.

    ``kappa``/``skip_omitted``/``shed_mass`` shape the per-flow coverage block.
    ``kappa`` is always ``kappa_realized`` — total shed mass, the meaning that
    key has in every generation of report.  With ``shed_mass`` set the flow is
    a post-audit one: the engine's split ships beside the total as
    ``kappa_slip`` (= ``kappa`` − ``shed_mass``) and ``shed_mass_fraction``.
    Without it the flow is pre-audit and carries only the total.
    """
    per_round = []
    for i, acc in enumerate(acc_by_round):
        loss = (loss_by_round[i] if loss_by_round else 1.0)
        nodes = {}
        up = (uplink_by_round[i] if uplink_by_round else 749_405)
        for w in workers:
            lcm = [{"layer_name": n, "bytes_sent": int(b * up / 749_405),
                    "importance": imp, "traffic_class": 0} for n, b, imp in LAYERS]
            nodes[w] = {
                "val_accuracy": acc, "val_loss": loss, "round_duration_s": 30.0,
                "send_duration_s": 0.379, "barrier_wait_duration_s": 14.0,
                "train_duration_s": 13.0, "layer_comm_metrics": lcm,
                "uplink_telemetry": {"_sender": {
                    "predicted_t_eps": 0.786944,
                    "assignment_diagnostics": dict(sender_diag or {
                        "realized_makespan_s": 0.786944,
                        "fluid_bound_head_s": 0.5997376,
                        "head_bytes": up}),
                }},
            }
        srcs = {}
        for j, w in enumerate(workers):
            t = (teps_by_round[i][j] if teps_by_round else 0.41)
            srcs[w] = {"t_eps_local": t, "kappa_realized": kappa,
                       "watchdog_fired": False, "shed_layers": list(shed)}
            if skip_omitted is not None:
                srcs[w]["skip_omitted_layers"] = list(skip_omitted)
            if shed_mass is not None:
                srcs[w]["shed_mass_fraction"] = shed_mass
                srcs[w]["kappa_slip"] = kappa - shed_mass
        nodes[agg] = {
            "val_accuracy": acc, "val_loss": loss, "round_duration_s": 30.0,
            "layer_comm_metrics": [{"layer_name": "all", "bytes_sent": downlink}],
            "uplink_telemetry": {**srcs, "_aggregation": {}},
        }
        per_round.append({"round": i, "nodes": nodes})
    return {"experiment": {"num_nodes": len(workers) + 1}, "per_round": per_round}


def write_run(root, arm, seed, report):
    d = Path(root) / arm / f"seed{seed}" / "results"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(report))
    return d / "report.json"


def flat(value, n=20):
    return [value] * n


def up_mb(budget, n_workers=3):
    """Total worker uplink (MB/round) a make_report uplink budget produces."""
    return n_workers * sum(int(b * budget / 749_405) for _n, b, _i in LAYERS) / 1e6


# ===========================================================================
# S6 / ML-05 — t-based CI with the sample sd
# ===========================================================================

class TestIntervals:
    def test_t_and_sample_sd(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
        n = len(xs)
        want = ht.student_t_ppf(0.975, n - 1) * st.stdev(xs) / math.sqrt(n)
        assert ac.tci95(xs) == pytest.approx(want, rel=1e-12)

    def test_wider_than_the_buggy_normal_pstdev_form(self):
        """The audited inflation factors: 1.44x at n=6, 2.69x at n=3 (S6)."""
        for n, factor in ((6, 1.44), (3, 2.69)):
            xs = [1.0 * i for i in range(n)]
            buggy = 1.96 * st.pstdev(xs) / math.sqrt(n)
            assert ac.tci95(xs) / buggy == pytest.approx(factor, abs=0.01)

    def test_reproduces_the_published_correction(self):
        """drop_eps03: sample sd 1.94pp over n=6 -> half-width 2.04pp, not 1.42."""
        sd, n = 1.94, 6
        half = ht.student_t_ppf(0.975, n - 1) * sd / math.sqrt(n)
        assert half == pytest.approx(2.04, abs=0.01)
        assert 1.96 * sd * math.sqrt((n - 1) / n) / math.sqrt(n) == pytest.approx(1.42, abs=0.01)

    def test_degenerate_inputs(self):
        assert math.isnan(ac.tci95([]))
        assert math.isnan(ac.tci95([1.0]))
        assert ac.tci95([2.0, 2.0, 2.0]) == pytest.approx(0.0)
        m, ci, n = ac.mean_ci([1.0, float("nan"), 3.0])
        assert (m, n) == (2.0, 2)


# ===========================================================================
# S7 — last-k endpoint read
# ===========================================================================

class TestEndpointRead:
    def test_last_k_mean_ignores_nans(self):
        assert ac.last_k([0.1, 0.2, 0.3, 0.4], 3) == pytest.approx(0.3)
        assert ac.last_k([0.5, float("nan"), 0.3], 3) == pytest.approx(0.4)
        assert ac.last_k([], 3) is None

    def test_default_window_widens_at_long_horizons(self):
        assert ac.default_k(20) == 3
        assert ac.default_k(50) == 5

    def test_endpoint_noise_does_not_drive_the_read(self):
        """A single noisy final round moves the endpoint by 10pp, the read by 3."""
        acc = [0.40, 0.41, 0.42, 0.43, 0.33]
        assert ac.last_k(acc, 3) == pytest.approx((0.42 + 0.43 + 0.33) / 3)
        r = ac.read_run(make_report(acc), k=3)
        assert r.endpoint == pytest.approx(0.33)
        assert r.lastk == pytest.approx(0.3933, abs=1e-3)


# ===========================================================================
# S1 / S11 — the divergence / health gate
# ===========================================================================

class TestHealthGate:
    def test_nonfinite_loss_is_a_divergence(self):
        h = ac.run_health([0.4] * 5, [1.0, 1.0, float("nan"), 1.0, 1.0])
        assert not h.healthy and h.reason == "nonfinite_loss@r2" and h.nan_round == 2

    def test_blown_up_but_finite_loss_also_counts(self):
        h = ac.run_health([0.4] * 4, [1.0, 1.0, 1e12, 1.0])
        assert not h.healthy and h.nan_round == 2

    def test_val_pinned_at_chance(self):
        h = ac.run_health([0.20, 0.15, 0.100, 0.100, 0.100], [1.0] * 5)
        assert not h.healthy and h.reason == "pinned_at_chance"

    def test_post_peak_collapse_vs_never_learned(self):
        collapsed = ac.run_health([0.30, 0.31, 0.11, 0.11, 0.11], [1.0] * 5)
        assert not collapsed.healthy and collapsed.reason == "post_peak_collapse"
        never = ac.run_health([0.09, 0.10, 0.11, 0.11, 0.115], [1.0] * 5)
        assert not never.healthy and never.reason == "never_learned"

    def test_healthy_run_passes(self):
        h = ac.run_health([0.2, 0.3, 0.4, 0.42, 0.44], [1.0] * 5)
        assert h.healthy and h.reason == "" and h.peak == pytest.approx(0.44)

    def test_chance_level_is_configurable_for_other_datasets(self):
        """FEMNIST has 62 classes: 10% is a fine accuracy there, not a collapse."""
        h = ac.run_health([0.10] * 5, [1.0] * 5, chance=1 / 62, floor=0.03)
        assert h.healthy

    def test_collapsed_denominator_drops_the_pair(self, tmp_path):
        """S1's exact failure: mono dies, so the arm's paired delta is +33pp."""
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        # seed41: mono collapses to chance; seed42/43 healthy
        write_run(root, "mono", 41, make_report(flat(0.45, 19) + [0.100],
                                                flat(1.0, 19) + [float("nan")]))
        write_run(root, "mono", 42, make_report(flat(0.45)))
        write_run(root, "mono", 43, make_report(flat(0.45)))
        for s in (41, 42, 43):
            write_run(root, "drop_eps03", s, make_report(flat(0.40)))
        aw3.main(root, tmp_path / "der")
        rows = list(_csv(tmp_path / "der" / "deltacc.csv"))
        drop = [r for r in rows if r["label"] == "drop_eps03"][0]
        assert int(drop["n"]) == 2, "the collapsed-mono pair must not be counted"
        assert float(drop["mean"]) == pytest.approx(-5.0, abs=0.01)
        health = list(_csv(tmp_path / "der" / "health.csv"))
        bad = [r for r in health if r["healthy"] == "False"]
        assert [(r["arm"], r["seed"]) for r in bad] == [("mono", "41")]

    def test_frontier_reports_collapse_rate_and_excludes_the_collapses(self, tmp_path):
        """S11: a chance-level run is a regime change, not a graded tradeoff."""
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        for s in (41, 42, 43):
            write_run(root, "mono", s, make_report(flat(0.45)))
            write_run(root, "byte_balanced_eps0", s, make_report(flat(0.44)))
        write_run(root, "frontier_tau8", 41, make_report(flat(0.100)))   # collapsed
        write_run(root, "frontier_tau8", 42, make_report(flat(0.30)))
        write_run(root, "frontier_tau8", 43, make_report(flat(0.30)))
        aw3.main(root, tmp_path / "der")
        rows = {r["arm"]: r for r in _csv(tmp_path / "der" / "frontier.csv")}
        tau8 = rows["frontier_tau8"]
        assert tau8["n_collapsed"] == "1" and tau8["collapse_rate"] == "0.333"
        assert int(tau8["n"]) == 2
        assert float(tau8["mean"]) == pytest.approx(-15.0, abs=0.01)


def _csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


# ===========================================================================
# S5 / S9 / S15 — the pre-registered ladder is actually wired in
# ===========================================================================

class TestHolmWiring:
    def test_holm_matches_the_library(self):
        fam = {"a": [-3.0, -2.0, -4.0, -3.5, -2.5, -3.2],
               "b": [-1.0, +0.5, -2.0, -0.5, +0.2, -1.5],
               "c": [-0.2, -0.1, +0.3, -0.4, +0.1, -0.2]}
        rows = {r.label: r for r in ac.contrast_family(fam)}
        raw = [ht.paired_t_one_sided(fam[k], "less").p for k in ("a", "b", "c")]
        want = ht.holm(raw, ["a", "b", "c"])
        for lab, adj in zip(want.labels, want.adj_p):
            assert rows[lab].holm_p == pytest.approx(adj)
        assert rows["a"].holm_p > rows["a"].t_p, "adjusted p must not be smaller"

    def test_holm_is_monotone_in_the_family_size(self):
        d = [-3.0, -2.0, -4.0, -3.5, -2.5, -3.2]
        alone = ac.contrast_family({"a": d})[0]
        with_family = {r.label: r for r in ac.contrast_family(
            {"a": d, "b": d, "c": d, "d": d})}["a"]
        assert with_family.holm_p > alone.holm_p

    def test_noninferiority_ucb_is_the_frozen_one(self):
        """S5: the ≤1pp rule applied to the cost framing d = baseline − arm."""
        d = [-4.0, -5.0, -6.0, -4.5, -5.5, -3.5]      # ΔACC, arm loses accuracy
        row = ac.contrast_family({"drop": d})[0]
        want = ht.noninferiority_ucb([-x for x in d], margin=1.0)
        assert row.ni_cost == pytest.approx(want.mean)
        assert row.ni_ucb == pytest.approx(want.ucb)
        assert row.ni_pass is False and row.ni_ucb > 1.0

    def test_noninferiority_can_pass(self):
        row = ac.contrast_family({"tiny": [-0.1, 0.0, -0.2, 0.1, -0.05, 0.05]})[0]
        assert row.ni_pass is True and row.ni_ucb < 1.0

    def test_small_n_is_not_evaluable(self):
        rows = {r.label: r for r in ac.contrast_family({"a": [-1.0], "b": [-1.0, -2.0]})}
        assert rows["a"].verdict == ht.NOT_EVALUABLE
        assert rows["b"].verdict != ht.NOT_EVALUABLE

    def test_exact_test_floor_is_flagged_at_small_n(self):
        """At n=3 the sign-flip floor is 1/8 > α, so 'robust' is vacuous."""
        row = ac.contrast_family({"a": [-3.0, -4.0, -5.0]})[0]
        assert row.perm_floor == pytest.approx(0.125)
        assert any("cannot reach" in n for n in row.notes)

    def test_format_family_is_printable(self):
        rows = ac.contrast_family({"a": [-3.0, -2.0, -4.0, -3.5, -2.5, -3.2]})
        text = ac.format_family(rows)
        assert "holm p" in text and "non-inferiority" in text


# ===========================================================================
# BYTE-02 — refresh-cycle-aligned byte windows
# ===========================================================================

class TestByteWindows:
    def test_window_is_a_whole_number_of_cycles(self):
        rounds = list(range(20))
        assert ac.cycle_window(rounds, 8, r0=3) == list(range(3, 19))   # 2 cycles
        assert ac.cycle_window(rounds, 5, r0=3) == list(range(3, 18))   # 3 cycles
        assert ac.cycle_window(rounds, None, r0=3) == list(range(3, 20))

    def test_cycle_mean_is_phase_independent_for_a_periodic_series(self):
        series = {r: (10.0 if r % 4 == 0 else 1.0) for r in range(20)}
        means = {r0: ac.cycle_stats(series, 4, r0).mean for r0 in (0, 1, 2, 3, 4)}
        assert all(v == pytest.approx(3.25) for v in means.values())

    def test_unaligned_window_is_phase_dependent(self):
        """The defect BYTE-02 measured: a fixed r0..R truncation aliases."""
        series = {r: (10.0 if r % 4 == 0 else 1.0) for r in range(20)}
        naive = {r0: st.mean([series[r] for r in range(r0, 20)]) for r0 in (3, 4, 5)}
        aligned = {r0: ac.cycle_stats(series, 4, r0).mean for r0 in (3, 4, 5)}
        assert max(naive.values()) - min(naive.values()) > 0.1
        assert max(aligned.values()) - min(aligned.values()) == pytest.approx(0.0)

    def test_cycle_spread_and_count(self):
        series = {r: float(r) for r in range(20)}
        cs = ac.cycle_stats(series, 4, r0=0)
        assert cs.n_cycles == 5 and cs.aligned
        assert cs.cycle_spread == pytest.approx(16.0)   # cycle means 1.5 .. 17.5

    def test_window_band_reports_the_sensitivity(self):
        series = {r: float(r) for r in range(20)}
        lo, hi = ac.window_band(lambda r0: ac.cycle_stats(series, None, r0).mean)
        assert lo < hi and lo == pytest.approx(11.0)

    def test_analyzer_emits_the_band_and_cycle_count(self, tmp_path):
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        # a strongly periodic uplink series with period 4
        up = [2_248_191 if r % 4 == 0 else 10_000 for r in range(20)]
        for s in (41, 42, 43):
            write_run(root, "mono", s, make_report(flat(0.45)))
            write_run(root, "byte_balanced_eps0", s, make_report(flat(0.44)))
            write_run(root, "frontier_tau2", s,
                      make_report(flat(0.40), uplink_by_round=up))
        aw3.main(root, tmp_path / "der")
        row = {r["arm"]: r for r in _csv(tmp_path / "der" / "frontier.csv")}["frontier_tau2"]
        assert int(row["n_cycles"]) >= 1
        assert float(row["uplink_band_lo_MB"]) <= float(row["steady_uplink_MB"])
        assert float(row["uplink_band_hi_MB"]) >= float(row["steady_uplink_MB"])


# ===========================================================================
# BYTE-03 / BYTE-05 — directions stay separate, aggregator is inferred
# ===========================================================================

class TestByteDirections:
    def test_aggregator_is_found_from_telemetry_not_hardcoded(self):
        rep = make_report(flat(0.4, 3), workers=("node-7", "node-8"), agg="node-9")
        agg, workers = ac.report_roles(rep)
        assert agg == "node-9" and workers == {"node-7", "node-8"}

    def test_extractor_separates_uplink_from_downlink(self, tmp_path):
        from scripts import extract_phase1b as ex
        p = write_run(tmp_path / "p1" / "exp1" / "eps03", ".", 41,
                      make_report(flat(0.4, 4), downlink=2_248_191))
        _exp, rows, *_ = ex.parse_report(str(p))
        r = rows[0]
        assert r["worker_uplink_bytes_mean"] == pytest.approx(738_680)
        assert r["agg_downlink_bytes"] == 2_248_191
        # the audited dilution would have produced ~1.12 MB for every arm
        diluted = (r["agg_downlink_bytes"] + 3 * r["worker_uplink_bytes_mean"]) / 4
        assert diluted == pytest.approx(1_116_057, rel=1e-3)
        assert r["worker_uplink_bytes_mean"] != pytest.approx(diluted)

    def test_fresh_means_fresh(self, tmp_path):
        """fresh_aggregated_bytes counts only layers that were not shed."""
        from scripts import extract_phase1b as ex
        p = write_run(tmp_path / "p1" / "exp1" / "eps03", ".", 41,
                      make_report(flat(0.4, 3), shed=("conv_3/kernel", "conv_4/kernel")))
        _exp, rows, *_ = ex.parse_report(str(p))
        r = rows[0]
        assert r["fresh_aggregated_bytes_mean"] < r["worker_uplink_bytes_mean"]
        assert r["fresh_aggregated_bytes_mean"] == pytest.approx(738_680 - 2 * 147_536)

    def test_frontier_carries_downlink_and_total_traffic(self, tmp_path):
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        for s in (41, 42, 43):
            write_run(root, "mono", s, make_report(flat(0.45)))
            write_run(root, "byte_balanced_eps0", s, make_report(flat(0.44)))
            write_run(root, "frontier_tau8", s,
                      make_report(flat(0.35), uplink_by_round=[74_940] * 20))
        aw3.main(root, tmp_path / "der")
        row = {r["arm"]: r for r in _csv(tmp_path / "der" / "frontier.csv")}["frontier_tau8"]
        up, down = float(row["steady_uplink_MB"]), float(row["agg_downlink_MB"])
        assert down == pytest.approx(2.248, abs=1e-3)
        assert float(row["total_traffic_MB"]) == pytest.approx(up + down, abs=1e-3)
        # uplink saving is ~90%, total saving roughly half of it — the BYTE-03 point
        assert float(row["uplink_saving_pct"]) == pytest.approx(90.0, abs=0.5)
        assert float(row["total_saving_pct"]) < 0.6 * float(row["uplink_saving_pct"])


# ===========================================================================
# NT-02 / NT-03 / NT-04 — bounds, clock domains, max-over-senders
# ===========================================================================

class TestTimingSemantics:
    def test_achievable_and_fluid_bounds_are_both_read(self):
        rep = make_report(flat(0.4, 3))
        bounds = ac.sender_bounds(rep)
        assert bounds and bounds[0].achievable == pytest.approx(0.786944)
        assert bounds[0].fluid == pytest.approx(0.5997376)
        assert bounds[0].granularity == pytest.approx(1.3121, abs=1e-3)

    def test_achievable_falls_back_to_the_eft_loads(self):
        rep = make_report(flat(0.4, 2), sender_diag={
            "eft_head_loads_s": {"0": 0.786944, "1": 0.393472, "2": 0.095296},
            "fluid_bound_head_s": 0.5997376})
        b = ac.sender_bounds(rep)[0]
        assert b.achievable == pytest.approx(0.786944)

    def test_one_axis_one_clock(self):
        assert ac.require_one_clock([ac.CLOCK_RECEIVER] * 3) == ac.CLOCK_RECEIVER
        with pytest.raises(ValueError, match="mixed clock domains"):
            ac.require_one_clock([ac.CLOCK_RECEIVER, ac.CLOCK_SENDER_ENQUEUE])

    def test_round_teps_is_the_max_over_senders(self, tmp_path):
        """NT-04: the mean over senders mixes quantization levels (0.41/0.41/0.61)."""
        from scripts import extract_phase1b as ex
        teps = [[0.409, 0.409, 0.614]] * 4
        p = write_run(tmp_path / "p1" / "exp1" / "eps02", ".", 41,
                      make_report(flat(0.4, 4), teps_by_round=teps))
        _exp, rows, flows, *_ = ex.parse_report(str(p))
        assert rows[0]["t_eps_max"] == pytest.approx(0.614)
        assert rows[0]["t_eps_mean"] == pytest.approx(0.477, abs=1e-3)
        assert len(flows) == 12   # every per-sender flow, for the level census

    def test_cp2_max_teps_helper(self, tmp_path):
        from scripts import analyze_cp2 as cp2
        rep = make_report(flat(0.4, 3), teps_by_round=[[0.2, 0.4, 0.6]] * 3)
        assert cp2.max_t_eps(rep) == pytest.approx(0.6)


# ===========================================================================
# S3 / BYTE-06 — concentration over the whole grid
# ===========================================================================

class TestConcentration:
    def test_cell_curve_is_ranked_by_importance_per_byte(self):
        from scripts import importance_concentration as ic
        layers = [{"layer_name": "big", "bytes_sent": 100, "importance": 0.5},
                  {"layer_name": "cheap", "bytes_sent": 1, "importance": 0.5}]
        curve = ic.cell_curve(layers)
        assert [c[0] for c in curve] == ["cheap", "big"]
        assert ic.cover_fraction(curve, 0.5) == pytest.approx(1 / 101)
        assert ic.cover_fraction(curve, 1.0) == pytest.approx(1.0)

    def test_every_worker_and_round_enters_the_grid(self, tmp_path):
        from scripts import importance_concentration as ic
        root = tmp_path / "arm"
        for s in (41, 42):
            write_run(root, ".", s, make_report(flat(0.4, 10)))
        paths = sorted(str(p) for p in root.glob("seed*/results/report.json"))
        cells, skipped = ic.collect(paths, r0=3, threshold=0.70)
        # 2 seeds x 3 workers x 7 rounds — no first-worker-in-dict-order pick
        assert len(cells) == 2 * 3 * 7
        assert {c[1] for c in cells} == {"node-1", "node-2", "node-3"}
        assert {c[0] for c in cells} == {41, 42}
        assert skipped == []

    def test_collapsed_run_leaves_the_grid(self, tmp_path):
        """S11 on the importance axis: after divergence the importance vector
        reads out the blow-up, not the mechanism."""
        from scripts import importance_concentration as ic
        root = tmp_path / "arm"
        write_run(root, ".", 41, make_report(flat(0.100, 10)))    # collapsed
        write_run(root, ".", 42, make_report(flat(0.40, 10)))
        paths = sorted(str(p) for p in root.glob("seed*/results/report.json"))
        cells, skipped = ic.collect(paths, r0=3, threshold=0.70)
        assert skipped == [(41, "pinned_at_chance")]
        assert {c[0] for c in cells} == {42}


# ===========================================================================
# S1 — horizon verdicts on healthy pairs only
# ===========================================================================

class TestHorizonVerdict:
    def _campaign(self, root, mono_late=0.50, drop_late=0.44):
        early = [0.30 + 0.005 * r for r in range(21)]
        for s in (41, 42, 43):
            write_run(root, "mono", s, make_report(early + [mono_late] * 29))
            write_run(root, "drop_eps03", s, make_report(early + [drop_late] * 29))

    def test_collapsed_baseline_no_longer_manufactures_narrowing(self, tmp_path, capsys):
        from scripts import horizon_trajectory as hz
        root = tmp_path / "h50"
        self._campaign(root)
        # seed41's mono collapses at r39 — S1's exact scenario
        write_run(root, "mono", 41, make_report(
            [0.30 + 0.005 * r for r in range(21)] + [0.50] * 18 + [0.100] * 11,
            [1.0] * 39 + [float("nan")] * 11))
        hz.main(str(root), 20)
        out = capsys.readouterr().out
        assert "seed41 DROPPED from every arm" in out
        assert "NARROWING" not in out.split("(NARROWING")[0]

    def test_verdict_uses_a_window_and_a_test(self, tmp_path, capsys):
        from scripts import horizon_trajectory as hz
        root = tmp_path / "h50"
        self._campaign(root)
        hz.main(str(root), 20)
        out = capsys.readouterr().out
        assert "window k=5" in out and "holm p(narrow/widen)" in out


# ===========================================================================
# S11 — the health gate covers EVERY mean, not only ΔACC
# ===========================================================================

def _w3_baseline(root, acc=0.45, ref_up=None):
    """mono + the byte-reference arm, all healthy — the paired denominator."""
    for s in (41, 42, 43):
        write_run(root, "mono", s, make_report(flat(acc)))
        write_run(root, "byte_balanced_eps0", s,
                  make_report(flat(acc - 0.01), uplink_by_round=ref_up))


class TestHealthGateOnBytesAndTiming:
    """S11 on the byte axis: frontier_agingoff published a 99.4% uplink saving
    computed entirely from its three collapsed runs, and τ8's byte mean was
    pulled by a run whose ΔACC the same table excluded. A byte mean over a
    diverged run measures the divergence, not a cheaper operating point."""

    def test_byte_mean_excludes_the_collapsed_run(self, tmp_path):
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        _w3_baseline(root)
        # the collapsed run sends almost nothing — exactly the bias S11 found
        write_run(root, "frontier_tau8", 41,
                  make_report(flat(0.100), uplink_by_round=flat(10_000)))
        for s in (42, 43):
            write_run(root, "frontier_tau8", s,
                      make_report(flat(0.30), uplink_by_round=flat(600_000)))
        aw3.main(root, tmp_path / "der")
        row = {r["arm"]: r for r in _csv(tmp_path / "der" / "frontier.csv")}["frontier_tau8"]
        assert row["n_healthy"] == "2" and row["n_collapsed"] == "1"
        assert float(row["steady_uplink_MB"]) == pytest.approx(up_mb(600_000), abs=1e-3)
        ungated = st.mean([up_mb(10_000), up_mb(600_000), up_mb(600_000)])
        assert float(row["steady_uplink_MB"]) != pytest.approx(ungated, abs=1e-2)

    def test_arm_with_no_healthy_run_is_not_evaluable(self, tmp_path):
        """agingoff's own column says 3/3 collapsed; it must not also publish
        a byte number, a saving percentage or a κ."""
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        _w3_baseline(root)
        for s in (41, 42, 43):
            write_run(root, "frontier_agingoff", s,
                      make_report(flat(0.100), uplink_by_round=flat(10_000),
                                  kappa=0.998, shed=[n for n, _b, _i in LAYERS]))
        aw3.main(root, tmp_path / "der")
        row = {r["arm"]: r for r in _csv(tmp_path / "der" / "frontier.csv")}["frontier_agingoff"]
        assert row["bytes_evaluable"] == "False" and row["n_healthy"] == "0"
        for col in ("steady_uplink_MB", "agg_downlink_MB", "total_traffic_MB",
                    "uplink_saving_pct", "total_saving_pct", "max_kappa_slip",
                    "max_kappa_slip_ub"):
            assert row[col] == "", f"{col} was published for a fully collapsed arm"

    def test_not_evaluable_is_printed_not_silently_dropped(self, tmp_path, capsys):
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        _w3_baseline(root)
        for s in (41, 42, 43):
            write_run(root, "frontier_agingoff", s, make_report(flat(0.100)))
        aw3.main(root, tmp_path / "der")
        out = capsys.readouterr().out
        line = [ln for ln in out.splitlines() if "frontier_agingoff" in ln and "3/3" in ln]
        assert line and "NOT EVAL" in line[0]

    def test_saving_denominator_is_health_gated_too(self, tmp_path):
        """A collapsed reference run would rescale every saving percentage."""
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        for s in (41, 42, 43):
            write_run(root, "mono", s, make_report(flat(0.45)))
        write_run(root, "byte_balanced_eps0", 41,
                  make_report(flat(0.100), uplink_by_round=flat(10_000)))  # collapsed
        for s in (42, 43):
            write_run(root, "byte_balanced_eps0", s,
                      make_report(flat(0.44), uplink_by_round=flat(1_000_000)))
        for s in (41, 42, 43):
            write_run(root, "frontier_tau2", s,
                      make_report(flat(0.40), uplink_by_round=flat(500_000)))
        aw3.main(root, tmp_path / "der")
        row = {r["arm"]: r for r in _csv(tmp_path / "der" / "frontier.csv")}["frontier_tau2"]
        assert float(row["ref_uplink_MB"]) == pytest.approx(up_mb(1_000_000), abs=1e-3)
        assert float(row["uplink_saving_pct"]) == pytest.approx(50.0, abs=0.1)

    def test_timing_rows_carry_the_health_verdict(self, tmp_path):
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        write_run(root, "mono", 41, make_report(flat(0.100)))     # collapsed
        write_run(root, "mono", 42, make_report(flat(0.45)))
        aw3.main(root, tmp_path / "der")
        rows = {r["seed"]: r for r in _csv(tmp_path / "der" / "timing.csv")
                if r["arm"] == "mono"}
        assert rows["41"]["healthy"] == "False"
        assert rows["41"]["median_" + ac.CLOCK_SENDER_ENQUEUE] == ""
        assert rows["42"]["median_" + ac.CLOCK_SENDER_ENQUEUE] != ""

    def test_cp2_byte_and_teps_means_are_health_gated(self, tmp_path):
        """Inert on today's cp2 data (18/18 healthy) — pinned so it stays so."""
        from scripts import analyze_cp2 as acp2
        w3, cp2 = tmp_path / "w3", tmp_path / "cp2"
        _w3_baseline(w3)
        for s in (41, 42, 43):
            write_run(w3, "drop_eps03", s, make_report(flat(0.40)))
        write_run(cp2, "cyclic_k7", 41,
                  make_report(flat(0.100), uplink_by_round=flat(10_000),
                              teps_by_round=[[9.0] * 3] * 20))       # collapsed
        for s in (42, 43):
            write_run(cp2, "cyclic_k7", s,
                      make_report(flat(0.35), uplink_by_round=flat(600_000),
                                  teps_by_round=[[0.4] * 3] * 20))
        acp2.main(cp2, w3, tmp_path / "der")
        row = {r["arm"]: r for r in _csv(tmp_path / "der" / "cp2_controls.csv")}["cyclic_k7"]
        assert row["n_healthy"] == "2"
        assert float(row["uplink_MB"]) == pytest.approx(up_mb(600_000), abs=1e-3)
        assert float(row["mean_t_eps_max_s"]) == pytest.approx(0.4, abs=1e-6)


# ===========================================================================
# BYTE-04 — κ is two quantities, never one column
# ===========================================================================

class TestKappaSplit:
    """The frontier's `max_kappa` column published 0.71/0.96/0.98/1.00 next to
    the invariant "κ ≤ ε=0.3", because deliberate skip-omission was counted as
    coverage slippage. Split, the slippage column supports the bound instead."""

    def test_post_fix_report_resolves_exactly(self):
        f = {"kappa_realized": 0.82, "kappa_slip": 0.12, "shed_mass_fraction": 0.70,
             "shed_layers": ["a", "b", "c"], "skip_omitted_layers": ["b", "c"]}
        ks = ac.kappa_split(f)
        assert ks.resolved and ks.source == "engine_split"
        assert (ks.slip, ks.recycled) == (0.12, 0.70)
        assert ks.conflated == pytest.approx(0.82)
        assert (ks.n_shed, ks.n_recycled, ks.n_slip) == (3, 2, 1)

    def test_kappa_realized_is_the_total_in_both_generations(self):
        """TRIG-2: the split is a new key, so one name never means two things.

        The same flow, pre- and post-audit, must agree on `kappa_realized`;
        only the post-audit one can additionally resolve the slippage.
        """
        pre = {"kappa_realized": 0.82, "shed_layers": ["a", "b", "c"],
               "skip_omitted_layers": ["b", "c"]}
        post = dict(pre, kappa_slip=0.12, shed_mass_fraction=0.70)
        assert ac.kappa_split(pre).conflated == ac.kappa_split(post).conflated
        assert ac.kappa_split(pre).slip is None          # not recoverable, not guessed
        assert ac.kappa_split(post).slip == pytest.approx(0.12)

    def test_pre_fix_wholly_deliberate_shed_is_zero_slippage(self):
        """The τ8 case: every shed layer had never been sent."""
        f = {"kappa_realized": 0.998, "shed_layers": ["a", "b"],
             "skip_omitted_layers": ["a", "b"]}
        ks = ac.kappa_split(f)
        assert ks.resolved and ks.slip == 0.0 and ks.recycled == 0.998
        assert ks.source == "layer_count" and ks.n_slip == 0

    def test_pre_fix_no_skip_omission_is_all_slippage(self):
        """The trigger-path arms: κ is clean there and reads ≤ ε."""
        f = {"kappa_realized": 0.2999, "shed_layers": ["a", "b"]}
        ks = ac.kappa_split(f)
        assert ks.resolved and ks.slip == pytest.approx(0.2999) and ks.recycled == 0.0

    def test_mixed_shed_without_the_sender_log_is_unresolved_not_guessed(self):
        """The flow alone does not carry the split — and is not made to."""
        f = {"kappa_realized": 0.80, "shed_layers": ["a", "b", "c"],
             "skip_omitted_layers": ["a"]}
        ks = ac.kappa_split(f)
        assert not ks.resolved and ks.slip is None and ks.recycled is None
        assert ks.conflated == 0.80 and (ks.n_recycled, ks.n_slip) == (1, 2)

    def test_summary_separates_the_two_masses(self):
        splits = [
            ac.kappa_split({"kappa_realized": 0.25, "shed_layers": ["a"]}),
            ac.kappa_split({"kappa_realized": 0.99, "shed_layers": ["a", "b"],
                            "skip_omitted_layers": ["a", "b"]}),
            ac.kappa_split({"kappa_realized": 0.90, "shed_layers": ["a", "b"],
                            "skip_omitted_layers": ["a"]}),
        ]
        s = ac.kappa_summary(splits)
        assert s["max_kappa_slip"] == pytest.approx(0.25)     # NOT 0.99, NOT 0.90
        assert s["max_recycled_mass_fraction"] == pytest.approx(0.99)
        assert s["max_kappa_slip_ub"] == pytest.approx(0.90)  # the unresolved flow
        assert s["n_kappa_unresolved"] == 1
        assert (s["mean_shed_layers"], s["mean_slip_layers"]) == (1.67, 0.67)

    def test_summary_counts_the_reconstructed_flows(self):
        scores = {"a": 6.0, "b": 3.0, "c": 1.0}
        splits = [
            ac.kappa_split({"kappa_realized": 0.4, "shed_layers": ["b", "c"],
                            "skip_omitted_layers": ["b"]}, scores),
            ac.kappa_split({"kappa_realized": 0.1, "shed_layers": ["c"]}, scores),
        ]
        s = ac.kappa_summary(splits)
        assert (s["n_kappa_reconstructed"], s["n_kappa_unresolved"]) == (1, 0)
        assert s["kappa_source"] == "mixed"          # layer_count + reconstruction
        assert s["n_kappa_recon_checks"] == 1 and s["kappa_recon_max_err"] < 1e-12
        assert s["max_kappa_slip"] == s["max_kappa_slip_ub"] == pytest.approx(0.1)

    def test_frontier_publishes_the_split_not_the_conflated_number(self, tmp_path):
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        _w3_baseline(root)
        names = [n for n, _b, _i in LAYERS]
        for s in (41, 42, 43):
            write_run(root, "frontier_tau8", s,
                      make_report(flat(0.30), kappa=0.998, shed=names,
                                  skip_omitted=names))
        aw3.main(root, tmp_path / "der")
        row = {r["arm"]: r for r in _csv(tmp_path / "der" / "frontier.csv")}["frontier_tau8"]
        assert "max_kappa" not in row, "the conflated column must be gone"
        assert float(row["max_kappa_slip"]) == 0.0
        assert float(row["max_recycled_mass_fraction"]) == pytest.approx(0.998)
        assert row["kappa_source"] == "layer_count"

    def test_phase1b_summary_carries_the_split_not_the_conflated_column(
            self, tmp_path, monkeypatch):
        from scripts import extract_phase1b as ep
        names = [n for n, _b, _i in LAYERS]
        d = tmp_path / "p1" / "exp1" / "eps03" / "seed41" / "results"
        d.mkdir(parents=True)
        (d / "report.json").write_text(json.dumps(
            make_report(flat(0.32), kappa=0.85, shed=names[:2], shed_mass=0.55)))
        monkeypatch.setattr("sys.argv", ["x", "--root", str(tmp_path / "p1"),
                                         "--out", str(tmp_path / "out")])
        ep.main()
        row = _csv(tmp_path / "out" / "summary.csv")[0]
        assert "max_kappa" not in row or "max_kappa_slip" in row
        assert float(row["max_kappa_slip"]) == pytest.approx(0.30)
        assert float(row["max_recycled_mass_fraction"]) == pytest.approx(0.55)
        assert row["kappa_source"] == "engine_split" and row["healthy"] == "True"

    def test_phase1b_excludes_a_collapsed_run_from_the_arm_medians(
            self, tmp_path, monkeypatch, capsys):
        from scripts import extract_phase1b as ep
        for seed, acc, teps in ((41, 0.100, 9.0), (42, 0.32, 0.41), (43, 0.33, 0.41)):
            d = tmp_path / "p1" / "exp1" / "eps03" / f"seed{seed}" / "results"
            d.mkdir(parents=True)
            (d / "report.json").write_text(json.dumps(
                make_report(flat(acc), teps_by_round=[[teps] * 3] * 20)))
        monkeypatch.setattr("sys.argv", ["x", "--root", str(tmp_path / "p1"),
                                         "--out", str(tmp_path / "out")])
        ep.main()
        out = capsys.readouterr().out
        assert "seed41 EXCLUDED" in out
        rows = {r["seed"]: r for r in _csv(tmp_path / "out" / "summary.csv")}
        assert rows["41"]["healthy"] == "False"     # still written, just not averaged
        line = [ln for ln in out.splitlines() if ln.startswith("exp1/eps03")]
        assert line and line[0].split()[1] == "2", "the collapsed run must not count"
        assert "9.0" not in line[0], "its t_eps must not reach the arm median"

    def test_epsilon_bound_check_uses_the_slip_column(self, tmp_path, capsys):
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        _w3_baseline(root)
        names = [n for n, _b, _i in LAYERS]
        for s in (41, 42, 43):
            write_run(root, "frontier_tau5", s,
                      make_report(flat(0.30), kappa=0.98, shed=names, skip_omitted=names))
        aw3.main(root, tmp_path / "der")
        out = capsys.readouterr().out
        assert "κ_slip vs recycled mass" in out
        line = [ln for ln in out.splitlines()
                if ln.strip().startswith("frontier_tau5") and "layer_count" in ln]
        assert line and "≤ε" in line[0], "the bound check must read on the slip column"


# ===========================================================================
# BYTE-04 (cont.) — the split is RECOVERED from stored data, not bounded away
# ===========================================================================

class TestKappaReconstruction:
    """A mixed shed set is not the end of the road: the late layers are exactly
    the ones that WERE sent, so their manifest mass is on disk in the sender's
    `layer_comm_metrics`, and the denominator follows from κ itself. Declaring
    those flows unresolved left `max_kappa_slip` a max over a minority of flows
    — a LOWER bound — with an "OK ≤ ε" stamped on it, which is the same species
    of error BYTE-04 exists to fix."""

    # manifest total 10.0; `a` covered, `b` skip-omitted, `c` genuinely late
    SCORES = {"a": 6.0, "b": 3.0, "c": 1.0}
    MIXED = {"kappa_realized": 0.4, "shed_layers": ["b", "c"],
             "skip_omitted_layers": ["b"]}

    def test_mixed_shed_resolves_from_the_senders_importance_log(self):
        ks = ac.kappa_split(self.MIXED, {"a": 6.0, "c": 1.0})   # `b` never sent
        assert ks.resolved and ks.source == "sender_importance"
        assert ks.slip == pytest.approx(0.1)        # 1.0 of 10.0, not 0.4
        assert ks.recycled == pytest.approx(0.3)
        assert ks.conflated == pytest.approx(0.4)   # the total is untouched

    def test_reconstruction_reproduces_the_exactly_known_ends(self):
        """The calibration that licenses the reconstructed numbers: where set
        logic settles the answer independently, the two must agree."""
        all_late = {"kappa_realized": 0.1, "shed_layers": ["c"]}
        all_deliberate = {"kappa_realized": 0.4, "shed_layers": ["b", "c"],
                          "skip_omitted_layers": ["b", "c"]}
        for flow, exact in ((all_late, 0.1), (all_deliberate, 0.0)):
            ks = ac.kappa_split(flow, self.SCORES)
            assert ks.source == "layer_count" and ks.slip == pytest.approx(exact)
            assert ks.recon_err is not None and ks.recon_err < 1e-12

    def test_a_late_layer_missing_from_the_log_stays_unresolved(self):
        """A tail that completed after the report shipped: its mass is genuinely
        absent, so the flow is unresolved — the fallback still exists."""
        flow = dict(self.MIXED, shed_layers=["b", "c", "d"])
        ks = ac.kappa_split(flow, {"a": 6.0, "c": 1.0})
        assert not ks.resolved and ks.slip is None and ks.source == "unresolved"

    def test_total_shed_leaves_nothing_to_scale_from(self):
        flow = {"kappa_realized": 1.0, "shed_layers": ["b", "c"],
                "skip_omitted_layers": ["b"]}
        assert not ac.kappa_split(flow, self.SCORES).resolved

    def test_a_join_mismatch_is_refused_rather_than_published(self):
        """Sender log and flow record must describe the same round; a slippage
        exceeding the total shed mass proves they do not."""
        ks = ac.kappa_split(self.MIXED, {"a": 1.0, "c": 9.0})
        assert not ks.resolved and ks.source == "unresolved"

    def test_sender_importance_survives_multi_destination_duplication(self):
        entry = {"layer_comm_metrics": [
            {"layer_name": "a", "importance": 6.0},     # to dest 1
            {"layer_name": "a", "importance": 6.0},     # same layer, dest 2
            {"layer_name": "c", "importance": 1.0},
            {"layer_name": "z", "importance": 2.0},
            {"layer_name": "z", "importance": 5.0},     # contradictory: unusable
        ]}
        assert ac.sender_importance(entry) == {"a": 6.0, "c": 1.0}

    def test_bound_verdict_is_withheld_while_a_flow_is_unresolved(self):
        """A max over resolved flows bounds the arm from BELOW; ✓ ≤ ε asserts an
        upper bound and may not be minted from it."""
        clean = {"max_kappa_slip": 0.29, "n_kappa_unresolved": 0}
        partial = {"max_kappa_slip": 0.29, "n_kappa_unresolved": 7}
        assert ac.kappa_bound_mark(clean) == "  ✓ ≤ε"
        assert "✓" not in ac.kappa_bound_mark(partial)
        assert "unresolved" in ac.kappa_bound_mark(partial)
        # a violation IS sound from below, so it still speaks
        assert "✗" in ac.kappa_bound_mark({"max_kappa_slip": 0.41,
                                           "n_kappa_unresolved": 7})

    @staticmethod
    def _kappa_line(out, arm):
        """The arm's row of the κ table (the run also prints ΔACC rows)."""
        section = out.split("coverage accounting")[-1]
        return [ln for ln in section.splitlines() if ln.strip().startswith(arm)]

    def _mixed_arm(self, root, strip_sender_log=False):
        """frontier_tau5 with a mixed shed set: 0.20 late + 0.05 skip-omitted."""
        _w3_baseline(root)
        late, deliberate = "conv_2/kernel", "conv_0/kernel"
        for s in (41, 42, 43):
            rep = make_report(flat(0.30), kappa=0.25, shed=[deliberate, late],
                              skip_omitted=[deliberate])
            if strip_sender_log:
                for rd in rep["per_round"]:
                    for nd in rd["nodes"].values():
                        nd.pop("layer_comm_metrics", None)
            write_run(root, "frontier_tau5", s, rep)

    def test_frontier_resolves_every_flow_and_earns_its_verdict(
            self, tmp_path, capsys):
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        self._mixed_arm(root)
        aw3.main(root, tmp_path / "der")
        out = capsys.readouterr().out
        row = {r["arm"]: r for r in _csv(tmp_path / "der" / "frontier.csv")}["frontier_tau5"]
        assert row["n_kappa_unresolved"] == "0"
        assert int(row["n_kappa_reconstructed"]) == int(row["n_kappa_flows"]) > 0
        assert float(row["max_kappa_slip"]) == pytest.approx(0.20)
        assert float(row["max_recycled_mass_fraction"]) == pytest.approx(0.05)
        # the vacuous bound collapses onto the real one once nothing is unresolved
        assert float(row["max_kappa_slip_ub"]) == pytest.approx(0.20)
        line = self._kappa_line(out, "frontier_tau5")
        assert line and "✓ ≤ε" in line[0]

    def test_without_the_sender_log_the_verdict_is_suspended_not_passed(
            self, tmp_path, capsys):
        from scripts import analyze_wave3 as aw3
        root = tmp_path / "w3"
        self._mixed_arm(root, strip_sender_log=True)
        aw3.main(root, tmp_path / "der")
        out = capsys.readouterr().out
        row = {r["arm"]: r for r in _csv(tmp_path / "der" / "frontier.csv")}["frontier_tau5"]
        assert int(row["n_kappa_unresolved"]) == int(row["n_kappa_flows"]) > 0
        assert row["max_kappa_slip"] == ""           # nothing resolved, nothing claimed
        line = self._kappa_line(out, "frontier_tau5")
        assert line and "✓" not in line[0] and "unresolved" in line[0]


# ===========================================================================
# S6 / S11 — one exhibit, one read semantics (no pre-fix artifact reuse)
# ===========================================================================

class TestFrontierProvenance:
    """analyze_cp2 printed the committed campaigns/w3/derived/frontier.csv
    verbatim above its own freshly computed rows: an endpoint-read, un-gated,
    aliased-window, conflated-κ table beside a last-3, health-gated one."""

    def _cp2_fixture(self, tmp_path):
        w3, cp2 = tmp_path / "w3", tmp_path / "cp2"
        _w3_baseline(w3)
        for s in (41, 42, 43):
            write_run(w3, "drop_eps03", s, make_report(flat(0.40)))
            write_run(w3, "frontier_tau5", s,
                      make_report(flat(0.35), uplink_by_round=flat(300_000)))
            write_run(cp2, "cyclic_k3", s,
                      make_report(flat(0.33), uplink_by_round=flat(200_000)))
        return w3, cp2

    def test_committed_csv_is_never_printed(self, tmp_path, capsys):
        from scripts import analyze_cp2 as acp2
        w3, cp2 = self._cp2_fixture(tmp_path)
        stale = w3 / "derived"
        stale.mkdir(parents=True, exist_ok=True)
        (stale / "frontier.csv").write_text(
            "arm,dACC_pp,max_kappa\nfrontier_tau5,-18.68,0.983\n")
        acp2.main(cp2, w3, tmp_path / "der")
        out = capsys.readouterr().out
        assert "-18.68" not in out and "0.983" not in out

    def test_frontier_is_recomputed_from_the_reports(self, tmp_path, capsys):
        from scripts import analyze_cp2 as acp2
        from scripts import analyze_wave3 as aw3
        w3, cp2 = self._cp2_fixture(tmp_path)
        acp2.main(cp2, w3, tmp_path / "der")
        out = capsys.readouterr().out
        rows, _f, _c = aw3.frontier_table(w3)
        want = {r["arm"]: r for r in rows}["frontier_tau5"]
        assert str(want["steady_uplink_MB"]) in out
        assert "RECOMPUTED" in out

    def test_missing_w3_reports_warn_loudly_instead_of_falling_back(self, tmp_path, capsys):
        from scripts import analyze_cp2 as acp2
        _w3, cp2 = self._cp2_fixture(tmp_path)
        empty = tmp_path / "now3"
        (empty / "derived").mkdir(parents=True)
        (empty / "derived" / "frontier.csv").write_text("arm,dACC_pp\nfrontier_tau5,-18.68\n")
        acp2.main(cp2, empty, tmp_path / "der2")
        out = capsys.readouterr().out
        assert "-18.68" not in out
        assert "PRE-FIX artifact" in out and "analyze_wave3.py" in out

    def test_no_analyzer_reads_a_derived_csv_back_in(self):
        """Structural guard: derived/*.csv is an OUTPUT of this layer. Reading
        one back is how two read semantics end up in one table (S6/S11)."""
        pat = re.compile(r"derived/[a-z_]*\.csv[\"']\s*\)?\s*\.?\s*(read_text|open)|"
                         r"(read_text|DictReader|reader)\s*\([^)]*frontier")
        offenders = [p.name for p in sorted((REPO / "scripts").glob("*.py"))
                     if pat.search(p.read_text())]
        assert not offenders, f"analyzers reading their own derived CSVs back: {offenders}"


# ===========================================================================
# S15 — enforcement: no script may hand-roll these estimators again
# ===========================================================================

BANNED = [
    (re.compile(r"pstdev\s*\("), "population sd in a published interval (S6)"),
    (re.compile(r"1\.96\s*\*"), "normal quantile instead of Student-t (S6)"),
    (re.compile(r"def\s+ci95\s*\("), "hand-rolled interval estimator (S6/S15)"),
    (re.compile(r"per_round\"\]\[-1\]"), "single-endpoint-round read (S7)"),
]
#: hypothesis_tests.py IS the library; analysis_common is its analyzer-facing
#: front-end. Everything else must import from them.
EXEMPT = {"hypothesis_tests.py", "analysis_common.py"}


class TestNoHandRolledStatistics:
    def test_scripts_do_not_hand_roll_intervals_or_endpoint_reads(self):
        offenders = []
        for p in sorted((REPO / "scripts").glob("*.py")):
            if p.name in EXEMPT:
                continue
            text = p.read_text()
            for pat, why in BANNED:
                for m in pat.finditer(text):
                    line = text.count("\n", 0, m.start()) + 1
                    offenders.append(f"{p.name}:{line} {why}")
        assert not offenders, (
            "hand-rolled statistics found — import from scripts.analysis_common "
            "(writeup/11 §2 'reuse only; never hand-roll'):\n  " + "\n  ".join(offenders))

    def test_the_analyzers_actually_import_the_library(self):
        for name in ("analyze_wave3.py", "analyze_cp2.py", "horizon_trajectory.py",
                     "meeting_analysis.py", "analyze_diag_cb.py",
                     "extract_phase1b.py", "importance_concentration.py"):
            text = (REPO / "scripts" / name).read_text()
            assert "analysis_common" in text, f"{name} bypasses the shared estimators"


class TestSummaryColumnCompat:
    """NT-01 rename must not break exhibits reading a regenerated summary.csv.

    `extract_phase1b` writes the new `median_send_enqueue` spelling while the
    campaign CSVs on disk still carry `median_send`; the C4 and Pareto
    exhibits must read either.  (Verifier objection, fix workflow round 3.)
    """

    def test_reader_accepts_both_spellings(self):
        assert ac.summary_enqueue_s({"median_send_enqueue": "0.379"}) == 0.379
        assert ac.summary_enqueue_s({"median_send": "0.379"}) == 0.379
        # New spelling wins when a row somehow carries both.
        assert ac.summary_enqueue_s(
            {"median_send_enqueue": "0.5", "median_send": "0.379"}) == 0.5

    def test_missing_or_blank_is_none(self):
        assert ac.summary_enqueue_s({}) is None
        assert ac.summary_enqueue_s({"median_send": ""}) is None

    def test_c4_decomposition_reads_new_spelling(self, monkeypatch):
        """The exhibit itself, driven by a regenerated-style row set."""
        import scripts.meeting_analysis as ma
        rows = [{"exp": "exp4", "median_send_enqueue": "0.379"},
                {"exp": "exp4", "median_send_enqueue": "0.381"}]
        monkeypatch.setattr(ma, "summary_rows", lambda: rows)
        picked = ma.med([v for r in rows if r["exp"] == "exp4"
                         for v in [ac.summary_enqueue_s(r)] if v is not None])
        assert picked == pytest.approx(0.380)
