"""Tests for assign_traffic_classes (gap-based splitting).

assign_traffic_classes returns (assignment: dict[str, int], gaps: list[float]).
"""

import pytest

from src.importance.traffic_mapping import assign_traffic_classes


# ---------------------------------------------------------------------------
# Gap-based splitting
# ---------------------------------------------------------------------------

def test_gap_based_clusters_user_example():
    """The worked example from the design doc: three natural clusters."""
    scores = {
        "a": 1.00,
        "b": 0.90,
        "c": 0.30,
        "d": 0.28,
        "e": 0.27,
        "f": 0.04,
        "g": 0.03,
        "h": 0.02,
        "i": 0.01,
    }
    result, gaps = assign_traffic_classes(scores, num_classes=3)

    assert result["a"] == 0
    assert result["b"] == 0
    assert result["c"] == 1
    assert result["d"] == 1
    assert result["e"] == 1
    assert result["f"] == 2
    assert result["g"] == 2
    assert result["h"] == 2
    assert result["i"] == 2
    # Two largest gaps are at indices 1 (0.60) and 4 (0.23)
    assert len(gaps) == 8
    assert abs(gaps[1] - 0.60) < 1e-9
    assert abs(gaps[4] - 0.23) < 1e-9


def test_gap_based_two_classes():
    """Single boundary at the largest gap."""
    scores = {"x": 5.0, "y": 4.5, "z": 0.1}
    result, gaps = assign_traffic_classes(scores, num_classes=2)
    # Gaps: 0.5, 4.4  → boundary after "y" (index 1)
    assert result["x"] == 0
    assert result["y"] == 0
    assert result["z"] == 1
    assert len(gaps) == 2


def test_gap_based_tie_break_prefers_earlier_boundary():
    """When two gaps are equal, the earlier (higher-importance) one wins."""
    scores = {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0}
    result, _ = assign_traffic_classes(scores, num_classes=2)
    # Largest gap is 1.0 appearing at indices 0, 1, 2 equally.
    # Tie-break: prefer earliest → boundary after index 0.
    assert result["a"] == 0
    assert result["b"] == 1
    assert result["c"] == 1
    assert result["d"] == 1


# ---------------------------------------------------------------------------
# Uniform fallback (equal-size buckets, alphabetical order)
# ---------------------------------------------------------------------------

def test_uniform_fallback_equal_scores():
    """All equal scores → equal-size buckets in alphabetical order."""
    scores = {name: 1.0 for name in ["a", "b", "c", "d", "e", "f", "g", "h", "i"]}
    result, gaps = assign_traffic_classes(scores, num_classes=3)
    # 9 layers, 3 classes → 3 each, alphabetical
    assert result == {
        "a": 0, "b": 0, "c": 0,
        "d": 1, "e": 1, "f": 1,
        "g": 2, "h": 2, "i": 2,
    }
    assert gaps == []


def test_uniform_fallback_remainder_distributed():
    """Remainder distributed to earlier (higher-priority) classes."""
    scores = {name: 0.0 for name in ["a", "b", "c", "d", "e"]}
    result, gaps = assign_traffic_classes(scores, num_classes=3)
    # 5 layers, 3 classes → 2+2+1
    assert result["a"] == 0
    assert result["b"] == 0
    assert result["c"] == 1
    assert result["d"] == 1
    assert result["e"] == 2
    assert gaps == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_class():
    """num_classes=1: everything in class 0, gaps is empty."""
    scores = {"x": 9.0, "y": 3.0, "z": 0.1}
    result, gaps = assign_traffic_classes(scores, num_classes=1)
    assert all(v == 0 for v in result.values())
    assert gaps == []


def test_zero_classes_treated_as_single():
    """num_classes<=1 edge case."""
    result, gaps = assign_traffic_classes({"a": 1.0, "b": 2.0}, num_classes=0)
    assert result == {"a": 0, "b": 0}
    assert gaps == []


def test_fewer_layers_than_classes():
    """Each layer gets its own class when n_layers <= num_classes."""
    scores = {"high": 5.0, "low": 0.5}
    result, gaps = assign_traffic_classes(scores, num_classes=4)
    # Sorted descending: high → 0, low → 1
    assert result["high"] == 0
    assert result["low"] == 1
    assert gaps == []


def test_single_layer():
    """One layer always goes to class 0 regardless of num_classes."""
    result, gaps = assign_traffic_classes({"only": 3.14}, num_classes=3)
    assert result == {"only": 0}
    assert gaps == []


def test_empty_scores():
    """Empty input returns empty mapping."""
    result, gaps = assign_traffic_classes({}, num_classes=3)
    assert result == {}
    assert gaps == []
