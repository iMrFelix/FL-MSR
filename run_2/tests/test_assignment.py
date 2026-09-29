"""Tests for the overnight assignment strategies and aging.

Pins the theory-repair behaviours from writeup/01-candidate-selection.md §8:
exact complement-knapsack tail (vs. the LP density prefix — the gate doc's
counterexample lives here), size-descending EFT head (vs. proportional
fill), Smith-order transmit queues, density-space additive aging with the
τ_max hard cap, stochastic boundary sampling with enforced coverage, and the
network-blind cyclic rotation.
"""

import math

import numpy as np
import pytest

from src.importance.aging import apply_aging
from src.importance.assignment import (
    AssignmentStrategy,
    ByteBalancedStrategy,
    CoverageEFTStrategy,
    CyclicStrategy,
    StochasticTailStrategy,
    make_strategy,
    registered_strategies,
)

_THREE_CLASS_BW = {0: 6.0, 1: 3.0, 2: 1.0}

#: The gate doc §8 counterexample: density order is a (1.0), then b, c
#: (0.667 each).  With budget 10 the density prefix sheds {b} or {c}
#: (9 bytes) while the exact knapsack sheds {a} (10 bytes).
_COUNTER_SCORES = {"a": 10.0, "b": 6.0, "c": 6.0}
_COUNTER_SIZES = {"a": 10, "b": 9, "c": 9}
_COUNTER_EPS = 10.0 / 22.0  # budget = ε·U_total = 10


def _assign(strategy: AssignmentStrategy, scores, bandwidths=_THREE_CLASS_BW,
            sizes=None, epsilon=0.0, must_receive=None, ages=None):
    """Call assign() with sane defaults so each test states only its deltas."""
    return strategy.assign(
        scores=scores,
        sizes=sizes if sizes is not None else {name: 1024 for name in scores},
        bandwidths=bandwidths,
        epsilon=epsilon,
        must_receive=must_receive if must_receive is not None else set(),
        ages=ages,
    )


def _kernel_instance():
    """A DeepCNN-pathology instance: one 147 KB low-importance kernel plus
    eight 4 KB high-importance biases (the E2/E6 anti-correlation)."""
    sizes = {f"b{i}": 4000 for i in range(8)}
    sizes["kernel"] = 147_000
    scores = {f"b{i}": 1.0 for i in range(8)}
    scores["kernel"] = 0.01
    return scores, sizes


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

class TestRegistryNames:

    def test_all_schema_literals_registered(self):
        # Must mirror src/config/schema.py:TrainingConfig.assignment_strategy.
        assert {"gap_based", "byte_balanced", "coverage_eft", "cyclic"} <= set(
            registered_strategies()
        )

    def test_make_strategy_builds_each(self):
        assert isinstance(make_strategy("byte_balanced"), ByteBalancedStrategy)
        assert isinstance(make_strategy("coverage_eft"), CoverageEFTStrategy)
        assert isinstance(make_strategy("cyclic", cyclic_k=2), CyclicStrategy)

    def test_stochastic_arm_via_coverage_eft_kwargs(self):
        """aging_mode='stochastic_tail' maps to coverage_eft kwargs, not a
        fifth registry name (interface-doc contract §1.6)."""
        strategy = make_strategy("coverage_eft", stochastic_tail=True, seed=5)
        result = _assign(
            strategy, _COUNTER_SCORES, sizes=_COUNTER_SIZES,
            epsilon=_COUNTER_EPS,
        )
        assert result.diagnostics["tail_method"] == "stochastic_boundary"


# ---------------------------------------------------------------------------
# Byte-balanced control
# ---------------------------------------------------------------------------

class TestByteBalanced:

    def test_bytes_proportional_to_bandwidth(self):
        scores = {f"l{i}": 10.0 - i for i in range(10)}
        sizes = {name: 1000 for name in scores}
        result = _assign(ByteBalancedStrategy(), scores, sizes=sizes)
        assert result.diagnostics["class_bytes"] == {
            "0": 6000, "1": 3000, "2": 1000,
        }

    def test_importance_ordered_within_and_across_classes(self):
        """Rank order must map to non-decreasing class index: the most
        important layers ride the fastest class."""
        scores = {f"l{i}": 10.0 - i for i in range(10)}
        sizes = {name: 1000 for name in scores}
        result = _assign(ByteBalancedStrategy(), scores, sizes=sizes)
        ranked = sorted(scores, key=lambda n: -scores[n])
        classes = [result.assignment[name] for name in ranked]
        assert classes == sorted(classes)
        assert classes[0] == 0

    def test_head_all_tail_empty_regardless_of_epsilon(self):
        """Assignment-only control: shedding stays disabled at any ε."""
        scores = {"a": 5.0, "b": 1.0}
        result = _assign(ByteBalancedStrategy(), scores, epsilon=0.5,
                         must_receive={"b"})
        assert result.head == {"a", "b"}
        assert result.tail == set()

    def test_predicted_is_fluid_makespan(self):
        scores = {"a": 2.0, "b": 1.0}
        sizes = {"a": 5000, "b": 5000}
        result = _assign(ByteBalancedStrategy(), scores, sizes=sizes)
        # 10 000 bytes over ΣB = 10 Mbps: 80 000 bits / 1e7 bit/s = 8 ms.
        assert result.predicted_t_eps == pytest.approx(0.008)

    def test_kernel_granularity_pathology_documented(self):
        """The control's known imperfection: prefix fill cannot split the
        147 KB kernel, so the whole update lands on class 0 here."""
        scores, sizes = _kernel_instance()
        result = _assign(ByteBalancedStrategy(), scores, sizes=sizes)
        assert set(result.assignment.values()) == {0}
        assert result.diagnostics["realized_makespan_s"] == pytest.approx(
            179_000 * 8 / 6e6,
        )

    def test_unshaped_class_takes_everything(self):
        result = _assign(
            ByteBalancedStrategy(), {"a": 2.0, "b": 1.0},
            bandwidths={0: float("inf"), 1: 3.0},
        )
        assert set(result.assignment.values()) == {0}
        assert result.predicted_t_eps == 0.0

    def test_empty_scores(self):
        result = _assign(ByteBalancedStrategy(), {})
        assert result.assignment == {}
        assert result.head == set() and result.tail == set()


# ---------------------------------------------------------------------------
# Coverage-EFT: exact knapsack tail (gate §8 repair 1)
# ---------------------------------------------------------------------------

class TestKnapsackExactness:

    def test_gate_counterexample_exact_beats_density_prefix(self):
        """u=(10,6,6), s=(10,9,9), budget 10: the density prefix sheds 9
        bytes ({b} or {c}); the exact complement knapsack sheds {a} = 10."""
        result = _assign(
            CoverageEFTStrategy(), _COUNTER_SCORES, sizes=_COUNTER_SIZES,
            epsilon=_COUNTER_EPS,
        )
        assert result.tail == {"a"}
        assert result.diagnostics["tail_method"] == "exact_enumeration"
        assert result.diagnostics["tail_bytes"] == 10
        # Coverage constraint honoured: shed utility ≤ ε·U_total.
        assert result.diagnostics["shed_utility"] <= 10.0 + 1e-6

    def test_dp_fallback_recovers_exact_optimum_here(self):
        """Forcing the >20-layer fallback path on the same instance: the
        quantized-utility DP (Phase-1 plan T2) recovers the exact optimum
        the old greedy-only fallback missed (10 > 9 shed bytes).  The
        greedy component's documented LP-rounding loss stays pinned in
        tests/test_assignment_scale.py."""
        result = _assign(
            CoverageEFTStrategy(exact_tail_limit=0), _COUNTER_SCORES,
            sizes=_COUNTER_SIZES, epsilon=_COUNTER_EPS,
        )
        assert result.diagnostics["tail_method"] == "quantized_dp_fallback"
        assert result.tail == {"a"}  # the exact knapsack answer
        assert result.diagnostics["tail_bytes"] == 10
        assert result.diagnostics["fallback_winner"] == "dp"

    def test_must_receive_excluded_from_tail(self):
        result = _assign(
            CoverageEFTStrategy(), _COUNTER_SCORES, sizes=_COUNTER_SIZES,
            epsilon=_COUNTER_EPS, must_receive={"a"},
        )
        # With "a" protected the best feasible shed is one of the u=6
        # layers; ties resolve deterministically to the lowest bitmask.
        assert result.tail == {"b"}
        assert "a" in result.head

    def test_epsilon_zero_sheds_nothing(self):
        result = _assign(
            CoverageEFTStrategy(), _COUNTER_SCORES, sizes=_COUNTER_SIZES,
            epsilon=0.0,
        )
        assert result.tail == set()
        assert result.head == set(_COUNTER_SCORES)
        assert result.diagnostics["tail_method"] == "no_tail"

    def test_zero_utility_layers_always_shed(self):
        """A zero-score layer is free shed bytes under any ε > 0 — the
        starvation source aging exists to counteract."""
        scores = {"a": 5.0, "zero": 0.0}
        sizes = {"a": 10, "zero": 5000}
        result = _assign(CoverageEFTStrategy(), scores, sizes=sizes,
                         epsilon=0.01)
        assert result.tail == {"zero"}

    def test_degenerate_all_zero_scores_refuse_to_shed(self):
        result = _assign(
            CoverageEFTStrategy(), {"a": 0.0, "b": 0.0},
            sizes={"a": 100, "b": 200}, epsilon=0.2,
        )
        assert result.tail == set()
        assert result.diagnostics["tail_method"] == "degenerate_scores_no_tail"

    def test_lp_fluid_bound_lower_bounds_prediction(self):
        result = _assign(
            CoverageEFTStrategy(), _COUNTER_SCORES, sizes=_COUNTER_SIZES,
            epsilon=_COUNTER_EPS,
        )
        assert (
            result.diagnostics["lp_fluid_bound_s"]
            <= result.predicted_t_eps + 1e-12
        )


# ---------------------------------------------------------------------------
# Coverage-EFT: head placement and transmit order (gate §8 repairs 2+3)
# ---------------------------------------------------------------------------

class TestCoverageEFTHead:

    def test_eft_beats_proportional_fill_on_kernel_instance(self):
        """The E2/E6 instance: proportional prefix fill dumps all 179 KB on
        class 0 (0.239 s); EFT pins the kernel to the fast class and spreads
        the biases (0.196 s — the kernel's own drain, a physical floor)."""
        scores, sizes = _kernel_instance()

        balanced = _assign(ByteBalancedStrategy(), scores, sizes=sizes)
        eft = _assign(CoverageEFTStrategy(), scores, sizes=sizes, epsilon=0.0)

        kernel_floor = 147_000 * 8 / 6e6
        assert eft.predicted_t_eps == pytest.approx(kernel_floor)
        assert eft.predicted_t_eps < balanced.diagnostics["realized_makespan_s"]
        assert eft.assignment["kernel"] == 0  # fastest class, size-descending

    def test_predicted_equals_max_head_load(self):
        # Single 8 Mbps class: prediction is just the head drain time.
        result = _assign(
            CoverageEFTStrategy(), {"a": 1.0, "b": 1.0},
            sizes={"a": 1000, "b": 2000}, bandwidths={0: 8.0}, epsilon=0.0,
        )
        assert result.predicted_t_eps == pytest.approx(3000 * 8 / 8e6)
        assert result.diagnostics["eft_head_loads_s"]["0"] == pytest.approx(
            result.predicted_t_eps,
        )

    def test_within_class_order_is_density_descending_heads_then_tails(self):
        scores = {"a": 8.0, "b": 9.0, "c": 1.0, "d": 0.0}
        sizes = {"a": 2, "b": 3, "c": 1, "d": 4}  # densities 4, 3, 1, 0
        result = _assign(
            CoverageEFTStrategy(), scores, sizes=sizes,
            bandwidths={0: 5.0}, epsilon=0.05,  # budget 0.9 -> tail {d}
        )
        assert result.tail == {"d"}
        # Insertion order of the assignment dict IS the transmit order.
        assert list(result.assignment) == ["a", "b", "c", "d"]
        assert result.diagnostics["class_order"] == {"0": ["a", "b", "c", "d"]}

    def test_tail_layers_are_still_assigned_to_classes(self):
        """The tail is best-effort, not dropped: every tail layer must hold
        a class so it rides behind the head on a real socket."""
        result = _assign(
            CoverageEFTStrategy(), _COUNTER_SCORES, sizes=_COUNTER_SIZES,
            epsilon=_COUNTER_EPS,
        )
        for name in result.tail:
            assert name in result.assignment

    def test_empty_scores(self):
        result = _assign(CoverageEFTStrategy(), {})
        assert result.assignment == {}
        assert result.predicted_t_eps is None

    @pytest.mark.parametrize("seed", range(8))
    def test_invariants_on_random_instances(self, seed):
        """Partition/contract invariants under fuzzed inputs."""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 15))
        names = [f"l{i:02d}" for i in range(n)]
        scores = {name: float(rng.uniform(0.0, 10.0)) for name in names}
        sizes = {name: int(rng.integers(1, 50_000)) for name in names}
        epsilon = float(rng.choice([0.0, 0.0625, 0.2, 0.5]))
        n_must = int(rng.integers(0, min(3, n) + 1))
        must = set(rng.choice(names, size=n_must, replace=False).tolist())

        result = _assign(
            CoverageEFTStrategy(), scores, sizes=sizes, epsilon=epsilon,
            must_receive=must,
        )

        # Everything transmits; partition enforced by the constructor.
        assert set(result.assignment) == set(names)
        assert must <= result.head
        assert set(result.assignment.values()) <= set(_THREE_CLASS_BW)
        # Coverage constraint: shed sched utility within budget (+fp tol).
        total = sum(scores.values())
        shed = sum(scores[name] for name in result.tail)
        assert shed <= epsilon * total + 1e-6 * max(1.0, total)
        # LP fluid bound never exceeds the discrete prediction.
        if result.predicted_t_eps is not None:
            assert (
                result.diagnostics["lp_fluid_bound_s"]
                <= result.predicted_t_eps + 1e-9
            )
        # Per-class transmit order: heads strictly before tails.
        for layer_list in result.diagnostics["class_order"].values():
            seen_tail = False
            for name in layer_list:
                if name in result.tail:
                    seen_tail = True
                else:
                    assert not seen_tail, "head layer queued after a tail layer"


# ---------------------------------------------------------------------------
# Stochastic tail (FedLUAR-style starvation control)
# ---------------------------------------------------------------------------

class TestStochasticTail:

    #: Six near-tied unit layers (density 0.01) + one clearly-head layer.
    #: Budget 2.5 admits exactly two unit layers per round.
    _SCORES = {**{f"u{i}": 1.0 for i in range(6)}, "big": 10.0}
    _SIZES = {**{f"u{i}": 100 for i in range(6)}, "big": 10}
    _EPS = 2.5 / 16.0

    def test_coverage_enforced_on_every_draw(self):
        strategy = StochasticTailStrategy(seed=123)
        total = sum(self._SCORES.values())
        for _ in range(40):
            result = _assign(strategy, self._SCORES, sizes=self._SIZES,
                             epsilon=self._EPS)
            shed = sum(self._SCORES[name] for name in result.tail)
            assert shed <= self._EPS * total + 1e-9 * total
            assert result.diagnostics["coverage_planned"] >= 1 - self._EPS - 1e-9
            assert len(result.tail) == 2  # budget admits exactly two
            assert "big" not in result.tail  # far from the boundary

    def test_membership_actually_varies(self):
        strategy = StochasticTailStrategy(seed=123)
        tails = {
            frozenset(
                _assign(strategy, self._SCORES, sizes=self._SIZES,
                        epsilon=self._EPS).tail
            )
            for _ in range(40)
        }
        assert len(tails) >= 2, "boundary sampling never varied the tail"

    def test_same_seed_reproduces_sequence(self):
        """Run-level reproducibility: the node seed pins the draw sequence."""
        first = StochasticTailStrategy(seed=42)
        second = StochasticTailStrategy(seed=42)
        for _ in range(6):
            tail_a = _assign(first, self._SCORES, sizes=self._SIZES,
                             epsilon=self._EPS).tail
            tail_b = _assign(second, self._SCORES, sizes=self._SIZES,
                             epsilon=self._EPS).tail
            assert tail_a == tail_b

    def test_far_from_boundary_layers_stay_shed(self):
        """Layers shed far below the boundary density are firm: sampling
        only touches the near-tied pool."""
        scores = {"head": 50.0, "cheap": 0.001, "t1": 1.0, "t2": 1.0}
        sizes = {"head": 10, "cheap": 10_000, "t1": 100, "t2": 100}
        # densities: head 5.0, cheap 1e-7, t1/t2 0.01.
        # budget = ε·U ≈ 1.6: tail* = {cheap, t1 or t2}.
        strategy = StochasticTailStrategy(seed=7)
        for _ in range(10):
            result = _assign(strategy, scores, sizes=sizes,
                             epsilon=1.6 / 52.001)
            assert "cheap" in result.tail
            assert "head" in result.head

    def test_must_receive_never_sampled_into_tail(self):
        strategy = StochasticTailStrategy(seed=11)
        for _ in range(20):
            result = _assign(strategy, self._SCORES, sizes=self._SIZES,
                             epsilon=self._EPS, must_receive={"u0"})
            assert "u0" not in result.tail

    def test_epsilon_zero_is_deterministic_no_tail(self):
        result = _assign(StochasticTailStrategy(seed=3), self._SCORES,
                         sizes=self._SIZES, epsilon=0.0)
        assert result.tail == set()
        assert result.diagnostics["tail_method"] == "no_tail"


# ---------------------------------------------------------------------------
# Cyclic control
# ---------------------------------------------------------------------------

class TestCyclic:

    _LAYERS = list("abcdefg")  # depth order

    def _scores(self, values=None):
        return {name: (values or {}).get(name, 1.0) for name in self._LAYERS}

    def test_rotation_sequence_and_determinism(self):
        expected = [
            {"a", "b", "c"}, {"d", "e", "f"}, {"g", "a", "b"},
            {"c", "d", "e"}, {"f", "g", "a"},
        ]
        first = CyclicStrategy(cyclic_k=3)
        second = CyclicStrategy(cyclic_k=3)
        for round_sel in expected:
            res_a = _assign(first, self._scores())
            res_b = _assign(second, self._scores())
            assert set(res_a.assignment) == round_sel
            assert set(res_b.assignment) == round_sel

    def test_unselected_layers_are_omitted_not_tailed(self):
        """FedPart-style omission semantics: the rest is not transmitted —
        absent from assignment, recorded in diagnostics."""
        result = _assign(CyclicStrategy(cyclic_k=3), self._scores())
        assert set(result.assignment) == {"a", "b", "c"}
        assert result.head == {"a", "b", "c"}
        assert result.tail == set()
        assert result.diagnostics["omitted_layers"] == ["d", "e", "f", "g"]

    def test_scores_ignored_network_blind(self):
        skewed = self._scores({"g": 1e9, "a": 0.0})
        result = _assign(CyclicStrategy(cyclic_k=2), skewed)
        assert set(result.assignment) == {"a", "b"}

    def test_depth_order_is_dict_insertion_order(self):
        scores = {"c3": 1.0, "a1": 1.0, "b2": 1.0}  # deliberately non-alpha
        result = _assign(CyclicStrategy(cyclic_k=1), scores)
        assert set(result.assignment) == {"c3"}

    def test_must_receive_force_included(self):
        result = _assign(CyclicStrategy(cyclic_k=3), self._scores(),
                         must_receive={"g"})
        assert set(result.assignment) == {"a", "b", "c", "g"}
        assert result.diagnostics["forced_must_receive"] == ["g"]
        assert "g" in result.head

    def test_k_larger_than_layer_count_sends_all(self):
        scores = {"x": 1.0, "y": 1.0}
        strategy = CyclicStrategy(cyclic_k=10)
        for _ in range(3):
            assert set(_assign(strategy, scores).assignment) == {"x", "y"}

    def test_byte_balanced_placement_of_selection(self):
        scores = {"a": 1.0, "b": 1.0, "c": 1.0}
        sizes = {"a": 6000, "b": 3000, "c": 1000}
        result = _assign(CyclicStrategy(cyclic_k=3), scores, sizes=sizes)
        assert result.assignment == {"a": 0, "b": 1, "c": 2}

    def test_predicted_is_fluid_makespan_of_selection(self):
        scores = self._scores()
        sizes = {name: 1000 for name in scores}
        result = _assign(CyclicStrategy(cyclic_k=2), scores, sizes=sizes)
        assert result.predicted_t_eps == pytest.approx(2000 * 8 / 10e6)

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError, match="cyclic_k"):
            CyclicStrategy(cyclic_k=0)

    def test_cyclic_k_is_required(self):
        with pytest.raises(TypeError):
            make_strategy("cyclic")


# ---------------------------------------------------------------------------
# Aging (density-space additive boost + τ_max hard cap)
# ---------------------------------------------------------------------------

class TestAging:

    def test_additive_boost_exact_arithmetic(self):
        # U=8, S=400 -> avg density 0.02; cold: 0 + 0.5·4·0.02·300 = 12.
        scores = {"hot": 8.0, "cold": 0.0}
        sizes = {"hot": 100, "cold": 300}
        effective, promoted = apply_aging(
            scores, {"cold": 4}, "additive_capped", 0.5, 0, sizes=sizes,
        )
        assert effective["cold"] == pytest.approx(12.0)
        assert effective["hot"] == pytest.approx(8.0)
        assert promoted == set()

    def test_boost_is_size_independent_in_density_space(self):
        """The point of the density form: equal age moves every layer's
        density by the same amount, regardless of byte size."""
        scores = {"small": 0.0, "large": 0.0, "ref": 10.0}
        sizes = {"small": 10, "large": 100_000, "ref": 100}
        effective, _ = apply_aging(
            scores, {"small": 3, "large": 3}, "additive_capped", 1.0, 0,
            sizes=sizes,
        )
        assert effective["small"] / sizes["small"] == pytest.approx(
            effective["large"] / sizes["large"],
        )

    def test_promotion_at_tau_max_boundary(self):
        scores = {"a": 1.0, "b": 1.0, "c": 1.0}
        sizes = {name: 10 for name in scores}
        _, promoted = apply_aging(
            scores, {"a": 4, "b": 5, "c": 6}, "additive_capped", 0.0, 5,
            sizes=sizes,
        )
        assert promoted == {"b", "c"}  # age ≥ τ_max promotes; 4 < 5 does not

    def test_tau_max_zero_disables_cap(self):
        _, promoted = apply_aging(
            {"a": 1.0}, {"a": 999}, "additive_capped", 0.0, 0, sizes={"a": 1},
        )
        assert promoted == set()

    def test_mode_none_is_identity(self):
        scores = {"a": 1.0, "b": 2.0}
        effective, promoted = apply_aging(scores, {"a": 50}, "none", 1.0, 1)
        assert effective == scores
        assert effective is not scores  # fresh dict, input not aliased
        assert promoted == set()

    def test_stochastic_tail_mode_leaves_scores_untouched(self):
        """Randomization lives in the strategy; the score path is identity,
        but a configured τ_max cap is still honoured (defence in depth)."""
        scores = {"a": 1.0, "b": 2.0}
        effective, promoted = apply_aging(
            scores, {"b": 7}, "stochastic_tail", 1.0, 5,
        )
        assert effective == scores
        assert promoted == {"b"}

    def test_additive_with_lambda_requires_sizes(self):
        with pytest.raises(ValueError, match="sizes"):
            apply_aging({"a": 1.0}, {"a": 1}, "additive_capped", 0.5, 0)

    def test_lambda_zero_is_pure_cap_arm(self):
        scores = {"a": 1.0, "b": 0.0}
        effective, promoted = apply_aging(
            scores, {"b": 9}, "additive_capped", 0.0, 5,
        )
        assert effective == scores
        assert promoted == {"b"}

    def test_degenerate_zero_scores_fall_back_to_unit_density(self):
        effective, _ = apply_aging(
            {"a": 0.0}, {"a": 2}, "additive_capped", 0.5, 0, sizes={"a": 100},
        )
        assert effective["a"] == pytest.approx(0.5 * 2 * 1.0 * 100)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="aging mode"):
            apply_aging({"a": 1.0}, {}, "multiplicative", 1.0, 0)

    def test_stale_age_keys_ignored(self):
        """Ages may reference layers absent this round (e.g. after a model
        change); they must neither boost nor promote ghosts."""
        effective, promoted = apply_aging(
            {"a": 1.0}, {"ghost": 99}, "additive_capped", 1.0, 5,
            sizes={"a": 10},
        )
        assert set(effective) == {"a"}
        assert promoted == set()

    def test_inputs_not_mutated(self):
        scores = {"a": 1.0}
        ages = {"a": 3}
        apply_aging(scores, ages, "additive_capped", 1.0, 2, sizes={"a": 10})
        assert scores == {"a": 1.0} and ages == {"a": 3}


class TestAgingStrategyIntegration:
    """End-to-end starvation arc: zero-utility layer sheds forever without
    aging, escapes via the density boost, and is hard-capped via τ_max
    promotion feeding must_receive."""

    _SCORES = {"a": 1.0, "z": 0.0}
    _SIZES = {"a": 100, "z": 100}
    _EPS = 0.4

    def test_starved_layer_escapes_tail_via_boost(self):
        strategy = CoverageEFTStrategy()
        # Without aging: z is free shed bytes -> always tail.
        before = _assign(strategy, dict(self._SCORES), sizes=self._SIZES,
                         epsilon=self._EPS)
        assert before.tail == {"z"}
        # Aged scores (z starved 10 rounds, λ=1): density boost outranks a.
        effective, _ = apply_aging(
            dict(self._SCORES), {"z": 10}, "additive_capped", 1.0, 0,
            sizes=self._SIZES,
        )
        after = _assign(strategy, effective, sizes=self._SIZES,
                        epsilon=self._EPS)
        assert "z" in after.head

    def test_tau_max_promotion_blocks_shedding_via_must_receive(self):
        effective, promoted = apply_aging(
            dict(self._SCORES), {"z": 3}, "additive_capped", 0.0, 3,
            sizes=self._SIZES,
        )
        assert promoted == {"z"}
        result = _assign(CoverageEFTStrategy(), effective, sizes=self._SIZES,
                         epsilon=self._EPS, must_receive=promoted)
        assert "z" in result.head
        assert "z" not in result.tail
