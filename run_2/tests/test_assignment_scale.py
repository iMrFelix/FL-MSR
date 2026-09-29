"""Scale-up tests for the >20-layer tail fallback (Phase-1 plan T2).

Pins the quantized-utility DP (complement knapsack, utility grid 1e-4 of
the eps*U budget), the greedy+DP combined picker, and the auto-selection
boundary at 20 tail candidates; carries the plan's property obligations
(validation c): planned coverage >= (1-eps) always, must_receive never
shed.  The DP-vs-exact criterion on the logged Stage-A manifests and the
L=65 runtime/LP-gap report live in ``scripts/validate_dp_vs_exact.py``;
this file keeps the same machinery honest on synthetic instances at unit
speed.
"""

import math
import time

import numpy as np
import pytest

from scripts.validate_dp_vs_exact import (
    build_eps_sweep,
    resnet20_layer_profile,
    resnet20_sizes,
    run_part_b,
    synthetic_scores,
)
from src.importance.assignment import (
    CoverageEFTStrategy,
    StochasticTailStrategy,
    _enumerate_max_shed,
    _fractional_shed_bytes,
    _greedy_density_shed,
    _quantized_dp_shed,
)

_THREE_CLASS_BW = {0: 6.0, 1: 3.0, 2: 1.0}

#: The gate doc §8 counterexample (same as test_assignment.py): the density
#: prefix sheds 9 bytes, the exact knapsack 10.
_COUNTER_SCORES = {"a": 10.0, "b": 6.0, "c": 6.0}
_COUNTER_SIZES = {"a": 10, "b": 9, "c": 9}
_COUNTER_EPS = 10.0 / 22.0  # budget = eps*U = 10


def _assign(strategy, scores, bandwidths=_THREE_CLASS_BW, sizes=None,
            epsilon=0.0, must_receive=None, ages=None):
    return strategy.assign(
        scores=scores,
        sizes=sizes if sizes is not None else {name: 1024 for name in scores},
        bandwidths=bandwidths,
        epsilon=epsilon,
        must_receive=must_receive if must_receive is not None else set(),
        ages=ages,
    )


def _arrays(scores, sizes):
    names = sorted(scores)
    utilities = np.array([scores[n] for n in names], dtype=np.float64)
    byte_sizes = np.array([float(sizes[n]) for n in names], dtype=np.float64)
    return names, utilities, byte_sizes


def _random_instance(seed, n_min=21, n_max=80, zero_frac=0.1):
    """A fuzzed instance large enough to hit the DP path naturally."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(n_min, n_max + 1))
    names = [f"l{i:03d}" for i in range(n)]
    scores = {
        name: 0.0 if rng.uniform() < zero_frac else float(rng.uniform(0.01, 10.0))
        for name in names
    }
    sizes = {name: int(rng.integers(1, 200_000)) for name in names}
    epsilon = float(rng.choice([0.0625, 0.2, 0.5, 0.9]))
    n_must = int(rng.integers(0, 6))
    must = set(rng.choice(names, size=n_must, replace=False).tolist())
    return scores, sizes, epsilon, must


def _shed_bytes(tail, sizes):
    return sum(sizes[name] for name in tail)


# ---------------------------------------------------------------------------
# Auto-selection boundary (plan T2: "auto-selected at L > 20")
# ---------------------------------------------------------------------------

class TestAutoSelection:

    def _uniform(self, n):
        scores = {f"l{i:03d}": 1.0 for i in range(n)}
        sizes = {name: 1000 for name in scores}
        return scores, sizes

    def test_at_20_candidates_stays_exact(self):
        scores, sizes = self._uniform(20)
        result = _assign(CoverageEFTStrategy(), scores, sizes=sizes,
                         epsilon=0.2)
        assert result.diagnostics["tail_method"] == "exact_enumeration"

    def test_above_20_candidates_selects_dp_fallback(self):
        scores, sizes = self._uniform(21)
        result = _assign(CoverageEFTStrategy(), scores, sizes=sizes,
                         epsilon=0.2)
        assert result.diagnostics["tail_method"] == "quantized_dp_fallback"
        assert result.diagnostics["dp_grid_units"] == 10_000
        assert result.diagnostics["fallback_winner"] in {"dp", "greedy"}

    def test_threshold_counts_tail_candidates_not_layers(self):
        """must_receive layers never enter the knapsack, so 22 layers with
        2 protected ones still take the exact path."""
        scores, sizes = self._uniform(22)
        result = _assign(
            CoverageEFTStrategy(), scores, sizes=sizes, epsilon=0.2,
            must_receive={"l000", "l001"},
        )
        assert result.diagnostics["tail_method"] == "exact_enumeration"

    def test_epsilon_zero_short_circuits_before_any_solver(self):
        scores, sizes = self._uniform(40)
        result = _assign(CoverageEFTStrategy(), scores, sizes=sizes,
                         epsilon=0.0)
        assert result.tail == set()
        assert result.diagnostics["tail_method"] == "no_tail"


# ---------------------------------------------------------------------------
# The quantized DP itself
# ---------------------------------------------------------------------------

class TestQuantizedDP:

    def test_recovers_exact_optimum_on_gate_counterexample(self):
        names, utilities, byte_sizes = _arrays(_COUNTER_SCORES, _COUNTER_SIZES)
        shed = _quantized_dp_shed(names, utilities, byte_sizes, budget=10.0)
        assert shed == {"a"}

    def test_greedy_component_alone_stays_suboptimal_here(self):
        """The LP-rounding counterexample stays on record: the greedy half
        of the fallback sheds 9 < 10 bytes by itself."""
        names, utilities, byte_sizes = _arrays(_COUNTER_SCORES, _COUNTER_SIZES)
        shed = _greedy_density_shed(names, utilities, byte_sizes, budget=10.0)
        assert shed == {"b"}
        assert _shed_bytes(shed, _COUNTER_SIZES) == 9

    def test_empty_inputs_and_zero_budget(self):
        names, utilities, byte_sizes = _arrays(_COUNTER_SCORES, _COUNTER_SIZES)
        assert _quantized_dp_shed([], np.array([]), np.array([]), 1.0) == set()
        assert _quantized_dp_shed(names, utilities, byte_sizes, 0.0) == set()
        assert _quantized_dp_shed(names, utilities, byte_sizes, -1.0) == set()

    def test_zero_utility_layers_shed_iff_they_carry_bytes(self):
        scores = {"freebytes": 0.0, "freenothing": 0.0, "pricey": 5.0}
        sizes = {"freebytes": 4000, "freenothing": 0, "pricey": 10}
        names, utilities, byte_sizes = _arrays(scores, sizes)
        shed = _quantized_dp_shed(names, utilities, byte_sizes, budget=1.0)
        assert "freebytes" in shed
        assert "freenothing" not in shed  # zero bytes buy nothing
        assert "pricey" not in shed  # exceeds the budget

    def test_item_exactly_at_budget_is_sheddable(self):
        scores = {"edge": 2.0, "other": 8.0}
        sizes = {"edge": 100, "other": 100}
        names, utilities, byte_sizes = _arrays(scores, sizes)
        shed = _quantized_dp_shed(names, utilities, byte_sizes, budget=2.0)
        assert shed == {"edge"}

    def test_deterministic(self):
        scores, sizes, epsilon, _ = _random_instance(7)
        names, utilities, byte_sizes = _arrays(scores, sizes)
        budget = epsilon * float(sum(scores.values()))
        first = _quantized_dp_shed(names, utilities, byte_sizes, budget)
        for _ in range(3):
            assert _quantized_dp_shed(
                names, utilities, byte_sizes, budget
            ) == first

    @pytest.mark.parametrize("seed", range(12))
    def test_never_beats_and_stays_within_1pct_of_exact(self, seed):
        """On enumerable instances the DP is (i) never byte-better than the
        true optimum — both are feasible integral sheds, the enumeration is
        optimal — and (ii) within the plan's 1%-of-total-bytes criterion."""
        scores, sizes, epsilon, _ = _random_instance(
            seed + 100, n_min=8, n_max=14,
        )
        names, utilities, byte_sizes = _arrays(scores, sizes)
        budget = epsilon * float(sum(scores.values()))
        exact = _enumerate_max_shed(names, utilities, byte_sizes, budget)
        dp = _quantized_dp_shed(names, utilities, byte_sizes, budget)
        bytes_exact = _shed_bytes(exact, sizes)
        bytes_dp = _shed_bytes(dp, sizes)
        assert bytes_dp <= bytes_exact
        total = sum(sizes.values())
        assert (bytes_exact - bytes_dp) / total < 0.01
        # Feasibility in *exact* utility arithmetic (ceil-quantization).
        assert sum(scores[n] for n in dp) <= budget * (1 + 1e-9) + 1e-12

    @pytest.mark.parametrize("seed", range(8))
    def test_feasible_at_scale(self, seed):
        """At DP-only sizes (no enumeration possible) the budget constraint
        must still hold in exact arithmetic."""
        scores, sizes, epsilon, _ = _random_instance(seed + 300)
        names, utilities, byte_sizes = _arrays(scores, sizes)
        budget = epsilon * float(sum(scores.values()))
        dp = _quantized_dp_shed(names, utilities, byte_sizes, budget)
        assert sum(scores[n] for n in dp) <= budget * (1 + 1e-9) + 1e-12


# ---------------------------------------------------------------------------
# Combined picker (greedy density + quantized DP)
# ---------------------------------------------------------------------------

class TestCombinedFallback:

    def test_greedy_reclaims_quantization_boundary_case(self):
        """21 equal-utility layers fit the budget exactly in real arithmetic
        but ceil-quantize to 21 x ceil(1e4/21) = 10 017 > 10 000 grid units,
        so the DP alone can shed only 20 of them: the greedy component must
        win and the combined tail must match the true optimum."""
        scores = {f"u{i:02d}": 1.0 for i in range(21)}
        scores["big"] = 79.0
        sizes = {name: 100 for name in scores}
        sizes["big"] = 10
        result = _assign(
            CoverageEFTStrategy(), scores, sizes=sizes,
            epsilon=0.21,  # budget = 0.21 * 100 = 21 = all unit layers
        )
        assert result.diagnostics["tail_method"] == "quantized_dp_fallback"
        assert result.diagnostics["fallback_winner"] == "greedy"
        assert result.tail == {f"u{i:02d}" for i in range(21)}
        assert result.diagnostics["greedy_shed_bytes"] == 2100
        assert result.diagnostics["dp_shed_bytes"] == 2000
        assert result.diagnostics["tail_bytes"] == 2100

    def test_dp_wins_where_greedy_is_blind(self):
        """Forcing the fallback on the gate counterexample: the DP side
        supplies the optimum the greedy misses."""
        result = _assign(
            CoverageEFTStrategy(exact_tail_limit=0), _COUNTER_SCORES,
            sizes=_COUNTER_SIZES, epsilon=_COUNTER_EPS,
        )
        assert result.diagnostics["tail_method"] == "quantized_dp_fallback"
        assert result.diagnostics["fallback_winner"] == "dp"
        assert result.tail == {"a"}
        assert result.diagnostics["tail_bytes"] == 10

    @pytest.mark.parametrize("seed", range(10))
    def test_combined_never_below_either_component(self, seed):
        scores, sizes, epsilon, _ = _random_instance(seed + 500)
        names, utilities, byte_sizes = _arrays(scores, sizes)
        total = float(sum(scores.values()))
        budget = epsilon * total + 1e-9 * max(1.0, total)
        greedy = _greedy_density_shed(names, utilities, byte_sizes, budget)
        dp = _quantized_dp_shed(names, utilities, byte_sizes, budget)
        result = _assign(CoverageEFTStrategy(), scores, sizes=sizes,
                         epsilon=epsilon)
        combined_bytes = result.diagnostics["tail_bytes"]
        assert combined_bytes >= _shed_bytes(greedy, sizes)
        assert combined_bytes >= _shed_bytes(dp, sizes)

    @pytest.mark.parametrize("seed", range(6))
    def test_forced_fallback_matches_exact_within_1pct_small_n(self, seed):
        """Strategy-level DP-vs-exact mirror of the validation script's
        criterion (a), on enumerable synthetic instances."""
        scores, sizes, epsilon, _ = _random_instance(
            seed + 700, n_min=6, n_max=16,
        )
        exact = _assign(CoverageEFTStrategy(), scores, sizes=sizes,
                        epsilon=epsilon)
        forced = _assign(CoverageEFTStrategy(exact_tail_limit=0), scores,
                         sizes=sizes, epsilon=epsilon)
        assert exact.diagnostics["tail_method"] == "exact_enumeration"
        assert forced.diagnostics["tail_method"] == "quantized_dp_fallback"
        delta = (
            exact.diagnostics["tail_bytes"] - forced.diagnostics["tail_bytes"]
        )
        assert delta >= 0
        assert delta / sum(sizes.values()) < 0.01


# ---------------------------------------------------------------------------
# Plan validation (c): properties at scale
# ---------------------------------------------------------------------------

class TestScaleProperties:

    @pytest.mark.parametrize("seed", range(25))
    def test_coverage_and_must_receive_properties(self, seed):
        """The two pre-registered properties (plan T2c) on fuzzed DP-path
        instances: planned coverage >= (1 - eps) and must_receive never
        shed; plus the partition/assignment contract."""
        scores, sizes, epsilon, must = _random_instance(seed)
        result = _assign(CoverageEFTStrategy(), scores, sizes=sizes,
                         epsilon=epsilon, must_receive=must)

        # Coverage: shed scheduling utility within the eps budget (+fp tol).
        total = sum(scores.values())
        shed = sum(scores[name] for name in result.tail)
        assert shed <= epsilon * total + 1e-6 * max(1.0, total)
        assert result.diagnostics["coverage_planned"] >= 1 - epsilon - 1e-6

        # must_receive: never shed, always head.
        assert must <= result.head
        assert not (must & result.tail)

        # Everything transmits; head/tail partition the assignment.
        assert set(result.assignment) == set(scores)
        assert result.head | result.tail == set(scores)
        assert set(result.assignment.values()) <= set(_THREE_CLASS_BW)

    @pytest.mark.parametrize("seed", range(5))
    def test_deterministic_across_instances_and_calls(self, seed):
        scores, sizes, epsilon, must = _random_instance(seed + 900)
        first = _assign(CoverageEFTStrategy(), scores, sizes=sizes,
                        epsilon=epsilon, must_receive=must)
        second = _assign(CoverageEFTStrategy(), scores, sizes=sizes,
                         epsilon=epsilon, must_receive=must)
        assert first.assignment == second.assignment
        assert first.tail == second.tail
        assert first.diagnostics["tail_bytes"] == second.diagnostics["tail_bytes"]

    def test_stochastic_boundary_rides_on_dp_fallback(self):
        """aging_mode='stochastic_tail' at L>20: the randomized boundary
        must sample on top of the DP base tail and keep coverage."""
        rng = np.random.default_rng(4)
        scores = {f"l{i:03d}": float(rng.uniform(0.5, 1.5)) for i in range(30)}
        sizes = {name: 1000 for name in scores}
        total = sum(scores.values())
        strategy = StochasticTailStrategy(seed=11)
        for _ in range(10):
            result = _assign(strategy, scores, sizes=sizes, epsilon=0.2,
                             must_receive={"l000"})
            assert result.diagnostics["tail_method"] == "stochastic_boundary"
            assert result.diagnostics["base_tail_method"] == "quantized_dp_fallback"
            shed = sum(scores[name] for name in result.tail)
            assert shed <= 0.2 * total + 1e-9 * total
            assert "l000" not in result.tail


# ---------------------------------------------------------------------------
# ResNet-20 profile (the regime the fallback exists for)
# ---------------------------------------------------------------------------

class TestResNet20Profile:

    def test_profile_shape(self):
        profile = resnet20_layer_profile()
        assert len(profile) == 65  # the plan's "~65 vars"
        assert sum(p for _, p in profile) == 272_474  # canonical ~0.27 M
        sizes = resnet20_sizes()
        assert max(sizes.values()) == 147_456  # the 3x3x64x64 kernels
        assert min(sizes.values()) == 40  # the dense bias

    @pytest.mark.parametrize("model", ["iid_lognormal", "anti_correlated"])
    @pytest.mark.parametrize("eps", [1.0 / 16.0, 0.2, 0.5])
    def test_l65_properties_under_both_utility_models(self, model, eps):
        sizes = resnet20_sizes()
        scores = synthetic_scores(model, np.random.default_rng(3))
        must = {"conv_in/kernel", "dense/kernel"}
        result = _assign(CoverageEFTStrategy(), scores, sizes=sizes,
                         epsilon=eps, must_receive=must)
        assert result.diagnostics["tail_method"] == "quantized_dp_fallback"
        total = sum(scores.values())
        shed = sum(scores[name] for name in result.tail)
        assert shed <= eps * total + 1e-6 * max(1.0, total)
        assert must <= result.head
        # The LP (fractional-knapsack) relaxation upper-bounds any integral
        # shed — the bound part (b) of the validation script reports against.
        candidates = sorted(n for n in scores if n not in must)
        lp_shed = _fractional_shed_bytes(
            candidates,
            np.array([scores[n] for n in candidates], dtype=np.float64),
            np.array([float(sizes[n]) for n in candidates], dtype=np.float64),
            eps * total + 1e-9 * max(1.0, total),
        )
        assert result.diagnostics["tail_bytes"] <= lp_shed + 1e-6

    def test_runtime_tripwire_at_l65(self):
        """Regression tripwire only (generous CI bound): the plan's 50 ms
        criterion is enforced by scripts/validate_dp_vs_exact.py part (b)
        on a quiet host; here a 200 ms median guards against an
        accidental complexity regression."""
        sizes = resnet20_sizes()
        scores = synthetic_scores("anti_correlated", np.random.default_rng(0))
        strategy = CoverageEFTStrategy()
        _assign(strategy, scores, sizes=sizes, epsilon=0.2)  # warmup
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            _assign(strategy, scores, sizes=sizes, epsilon=0.2)
            samples.append(time.perf_counter() - t0)
        assert sorted(samples)[len(samples) // 2] < 0.200

    def test_part_b_driver_smoke(self):
        """The validation script's synthetic driver end-to-end (tiny sweep,
        no runtime assertion — that belongs to the quiet-host run)."""
        summary = run_part_b(
            eps_sweep=[1.0 / 16.0, 0.2],
            n_per_model=1,
            seed=0,
            runtime_budget_ms=1e9,
        )
        assert summary["status"] == "ok"
        assert summary["auto_selected_dp"] is True
        assert summary["n_negative_gaps"] == 0
        assert summary["n_coverage_violations"] == 0
        assert summary["beta_gap_max"] >= 0.0

    def test_eps_sweep_contains_anchor_and_excludes_zero(self):
        sweep = build_eps_sweep()
        assert 0.0 not in sweep
        assert any(math.isclose(e, 1.0 / 16.0) for e in sweep)
        assert sweep == sorted(sweep)
