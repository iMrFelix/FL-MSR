"""Tests for the AssignmentStrategy interface, registry, and gap-based adapter.

The adapter must reproduce assign_traffic_classes() exactly: gap-based is the
control arm of the overnight matrix and any drift here silently invalidates
the baseline comparison (writeup/01-candidate-selection.md §7).
"""

import pytest

from src.importance.assignment import (
    AssignmentResult,
    AssignmentStrategy,
    GapBasedStrategy,
    make_strategy,
    register_strategy,
    registered_strategies,
)
from src.importance.traffic_mapping import assign_traffic_classes


_THREE_CLASS_BW = {0: 6.0, 1: 3.0, 2: 1.0}


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


# ---------------------------------------------------------------------------
# AssignmentResult contract
# ---------------------------------------------------------------------------

class TestAssignmentResult:

    def test_valid_partition_accepted(self):
        result = AssignmentResult(
            assignment={"a": 0, "b": 2},
            head={"a"},
            tail={"b"},
            predicted_t_eps=1.5,
            diagnostics={"note": "ok"},
        )
        assert result.assignment["b"] == 2
        assert result.predicted_t_eps == pytest.approx(1.5)

    def test_defaults(self):
        result = AssignmentResult(assignment={"a": 0}, head={"a"}, tail=set())
        assert result.predicted_t_eps is None
        assert result.diagnostics == {}

    def test_head_tail_overlap_rejected(self):
        with pytest.raises(ValueError, match="disjoint"):
            AssignmentResult(
                assignment={"a": 0, "b": 1},
                head={"a", "b"},
                tail={"b"},
            )

    def test_partition_must_cover_assignment(self):
        with pytest.raises(ValueError, match="head ∪ tail"):
            AssignmentResult(
                assignment={"a": 0, "b": 1},
                head={"a"},
                tail=set(),
            )

    def test_partition_must_not_exceed_assignment(self):
        with pytest.raises(ValueError, match="head ∪ tail"):
            AssignmentResult(
                assignment={"a": 0},
                head={"a", "ghost"},
                tail=set(),
            )

    def test_negative_class_rejected(self):
        with pytest.raises(ValueError, match=">= 0"):
            AssignmentResult(assignment={"a": -1}, head={"a"}, tail=set())

    def test_omitted_layers_are_legal(self):
        """Cyclic-style schedules assign a subset; the rest is simply unsent.

        The invariant is over assignment's keys, not over the model's full
        layer set, so a k-of-L assignment validates.
        """
        result = AssignmentResult(
            assignment={"a": 0},
            head={"a"},
            tail=set(),
        )
        assert "b" not in result.assignment


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_gap_based_registered_by_import(self):
        assert "gap_based" in registered_strategies()

    def test_make_strategy_returns_gap_based(self):
        strategy = make_strategy("gap_based")
        assert isinstance(strategy, GapBasedStrategy)

    def test_unknown_name_raises_with_listing(self):
        with pytest.raises(ValueError, match="gap_based"):
            make_strategy("definitely_not_registered")

    def test_duplicate_name_different_class_rejected(self):
        with pytest.raises(ValueError, match="already registered"):
            @register_strategy("gap_based")
            class Impostor(AssignmentStrategy):  # pragma: no cover - never used
                def assign(self, *, scores, sizes, bandwidths, epsilon,
                           must_receive, ages=None, trigger_scores=None):
                    raise NotImplementedError

    def test_reregistering_same_class_is_idempotent(self):
        """Module re-imports must not blow up the registry."""
        decorated = register_strategy("gap_based")(GapBasedStrategy)
        assert decorated is GapBasedStrategy
        assert registered_strategies().count("gap_based") == 1

    def test_abstract_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            AssignmentStrategy()


# ---------------------------------------------------------------------------
# Gap-based adapter: exact parity with the legacy function
# ---------------------------------------------------------------------------

class TestGapBasedParity:

    SCORE_SETS = [
        # The worked example from the design doc (three natural clusters).
        {
            "a": 1.00, "b": 0.90, "c": 0.30, "d": 0.28, "e": 0.27,
            "f": 0.04, "g": 0.03, "h": 0.02, "i": 0.01,
        },
        # Uniform scores -> equal-size-bucket fallback.
        {name: 1.0 for name in "abcdefghi"},
        # Fewer layers than classes -> one class each.
        {"high": 5.0, "low": 0.5},
        # Single layer.
        {"only": 3.14},
        # Tie-break case.
        {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0},
    ]

    @pytest.mark.parametrize("scores", SCORE_SETS)
    def test_assignment_matches_legacy_function(self, scores):
        expected_assignment, expected_gaps = assign_traffic_classes(
            scores, num_classes=len(_THREE_CLASS_BW),
        )
        result = _assign(GapBasedStrategy(), scores)
        assert result.assignment == expected_assignment
        assert result.diagnostics["gaps"] == expected_gaps

    @pytest.mark.parametrize("scores", SCORE_SETS)
    def test_head_is_everything_tail_is_empty(self, scores):
        result = _assign(GapBasedStrategy(), scores)
        assert result.head == set(scores)
        assert result.tail == set()

    def test_no_t_eps_prediction(self):
        """Gap-based has no fluid model; predicted_t_eps must stay None."""
        result = _assign(GapBasedStrategy(), {"a": 1.0, "b": 0.1})
        assert result.predicted_t_eps is None

    def test_num_classes_comes_from_bandwidths(self):
        scores = {"a": 9.0, "b": 5.0, "c": 0.1}
        result = _assign(GapBasedStrategy(), scores, bandwidths={0: 10.0})
        assert set(result.assignment.values()) == {0}

    def test_empty_bandwidths_degrades_to_single_class(self):
        result = _assign(GapBasedStrategy(), {"a": 1.0, "b": 0.2},
                         bandwidths={})
        assert set(result.assignment.values()) == {0}

    def test_ignores_epsilon_sizes_must_receive_and_ages(self):
        """The control arm must stay byte/deadline-blind by design (E2)."""
        scores = {"a": 1.00, "b": 0.90, "c": 0.04}
        baseline = _assign(GapBasedStrategy(), scores)
        perturbed = _assign(
            GapBasedStrategy(),
            scores,
            sizes={"a": 1, "b": 10**9, "c": 5},
            epsilon=0.5,
            must_receive={"c"},
            ages={"a": 0, "b": 7, "c": 3},
        )
        assert perturbed.assignment == baseline.assignment
        assert perturbed.head == baseline.head
        assert perturbed.tail == set()

    def test_empty_scores(self):
        result = _assign(GapBasedStrategy(), {})
        assert result.assignment == {}
        assert result.head == set()
        assert result.tail == set()
