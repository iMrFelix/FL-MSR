"""Tests for the importance manifest builder and validator.

See docs/extensions/01-importance-manifest.md.
"""

import math

import pytest

from src.importance.manifest import (
    build_manifest,
    manifest_total,
    validate_manifest,
)
from src.proto_gen import federation_pb2


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def test_build_manifest_populates_round_and_node():
    m = build_manifest(round_num=7, source_node_id="node-2",
                       importance_scores={"a": 0.3})
    assert m.round == 7
    assert m.source_node_id == "node-2"


def test_build_manifest_preserves_layer_names_and_scores():
    scores = {"conv_0/kernel": 0.1, "conv_1/kernel": 0.6, "head/bias": 0.05}
    m = build_manifest(0, "node-1", scores)
    by_name = {e.layer_name: e.raw_score for e in m.entries}
    # float-cast may introduce tiny precision noise; allow small tolerance
    for name, expected in scores.items():
        assert math.isclose(by_name[name], expected, rel_tol=1e-6)


def test_build_manifest_default_must_receive_is_false():
    m = build_manifest(0, "node-1", {"a": 0.1, "b": 0.2})
    assert all(e.must_receive is False for e in m.entries)


def test_build_manifest_honours_must_receive_predicate():
    m = build_manifest(
        0, "node-1",
        {"small": 0.01, "big": 0.9},
        must_receive_predicate=lambda name, score: score >= 0.5,
    )
    by_name = {e.layer_name: e.must_receive for e in m.entries}
    assert by_name["big"] is True
    assert by_name["small"] is False


def test_manifest_serialises_and_round_trips():
    m = build_manifest(3, "node-0", {"l1": 0.4, "l2": 0.2})
    blob = m.SerializeToString()
    m2 = federation_pb2.RoundManifest()
    m2.ParseFromString(blob)
    assert m2.round == 3
    assert m2.source_node_id == "node-0"
    assert {e.layer_name for e in m2.entries} == {"l1", "l2"}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def test_validator_accepts_well_formed():
    m = build_manifest(0, "node-1", {"a": 0.0, "b": 1.0})
    validate_manifest(m)  # must not raise


def test_validator_rejects_empty_entries():
    m = federation_pb2.RoundManifest()
    m.round = 0
    m.source_node_id = "node-1"
    with pytest.raises(ValueError, match="non-empty"):
        validate_manifest(m)


def test_validator_rejects_negative_score():
    m = federation_pb2.RoundManifest()
    m.round = 0
    m.source_node_id = "node-1"
    e = m.entries.add()
    e.layer_name = "x"
    e.raw_score = -0.01
    with pytest.raises(ValueError, match="non-negative"):
        validate_manifest(m)


def test_validator_rejects_nan_score():
    m = federation_pb2.RoundManifest()
    m.round = 0
    m.source_node_id = "node-1"
    e = m.entries.add()
    e.layer_name = "x"
    e.raw_score = float("nan")
    with pytest.raises(ValueError, match="non-negative"):
        validate_manifest(m)


def test_validator_rejects_duplicate_layer_name():
    m = federation_pb2.RoundManifest()
    m.round = 0
    m.source_node_id = "node-1"
    for _ in range(2):
        e = m.entries.add()
        e.layer_name = "dup"
        e.raw_score = 0.1
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest(m)


def test_validator_rejects_empty_layer_name():
    m = federation_pb2.RoundManifest()
    m.round = 0
    m.source_node_id = "node-1"
    e = m.entries.add()
    e.layer_name = ""
    e.raw_score = 0.1
    with pytest.raises(ValueError, match="non-empty"):
        validate_manifest(m)


def test_validator_accepts_zero_score():
    """Zero is a legal score (e.g. a frozen layer with no gradient)."""
    m = build_manifest(0, "node-1", {"frozen": 0.0, "active": 0.5})
    validate_manifest(m)  # must not raise


# ---------------------------------------------------------------------------
# Total
# ---------------------------------------------------------------------------

def test_manifest_total_sums_scores():
    m = build_manifest(0, "node-1", {"a": 0.1, "b": 0.6, "c": 0.3})
    assert math.isclose(manifest_total(m), 1.0, rel_tol=1e-6)


def test_manifest_total_handles_zero_entries_for_robustness():
    """An empty manifest is invalid, but the total function should still be
    well-defined for callers that decide validation policy themselves."""
    m = federation_pb2.RoundManifest()
    assert manifest_total(m) == 0.0
