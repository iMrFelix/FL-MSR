"""Tests for the ε-deadline trigger in LayerBuffer.

See docs/extensions/02-epsilon-trigger.md.
"""

import numpy as np
import pytest

from src.importance.manifest import build_manifest
from src.training.update import BufferResult, LayerBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add(buf, name, idx, total, source="node-1", round_num=0, num_samples=100):
    """Convenience wrapper to add a one-element array layer to the buffer."""
    return buf.add_layer(
        source_node=source,
        round_num=round_num,
        layer_name=name,
        layer_index=idx,
        total_layers=total,
        array=np.array([float(idx)], dtype=np.float32),
        num_samples=num_samples,
    )


# ---------------------------------------------------------------------------
# Constructor / config validation
# ---------------------------------------------------------------------------

def test_constructor_rejects_negative_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        LayerBuffer(epsilon=-0.01)


def test_constructor_rejects_epsilon_ge_one():
    with pytest.raises(ValueError, match="epsilon"):
        LayerBuffer(epsilon=1.0)
    with pytest.raises(ValueError, match="epsilon"):
        LayerBuffer(epsilon=1.5)


def test_constructor_accepts_boundary_values():
    LayerBuffer(epsilon=0.0)
    LayerBuffer(epsilon=0.999)


# ---------------------------------------------------------------------------
# Backward-compatible "all layers in" trigger
# ---------------------------------------------------------------------------

def test_eps_zero_requires_all_layers():
    """With ε=0 and no manifest, behaviour matches the pre-extension buffer."""
    buf = LayerBuffer(epsilon=0.0)
    assert _add(buf, "l0", 0, 3) is None
    assert _add(buf, "l1", 1, 3) is None
    result = _add(buf, "l2", 2, 3)
    assert isinstance(result, BufferResult)
    assert result.is_partial is False
    assert result.missing_layers == frozenset()
    assert set(result.update.parameters.keys()) == {"l0", "l1", "l2"}


def test_all_layers_trigger_fires_even_without_manifest():
    """Even with ε>0 set, no manifest means we still wait for all layers."""
    buf = LayerBuffer(epsilon=0.5)
    assert _add(buf, "l0", 0, 2) is None
    result = _add(buf, "l1", 1, 2)
    assert isinstance(result, BufferResult)
    assert result.is_partial is False


# ---------------------------------------------------------------------------
# ε-coverage trigger
# ---------------------------------------------------------------------------

def test_eps_trigger_fires_when_coverage_met():
    """ε=0.2; tail of small-score layers can be dropped."""
    buf = LayerBuffer(epsilon=0.2)
    manifest = build_manifest(
        round_num=0, source_node_id="node-1",
        importance_scores={"l0": 0.5, "l1": 0.3, "l2": 0.1, "l3": 0.1},
    )
    buf.register_manifest(manifest)

    # total=1.0; target=0.8.  l0+l1=0.8 should fire.
    assert _add(buf, "l0", 0, 4) is None  # 0.5 < 0.8
    result = _add(buf, "l1", 1, 4)        # 0.8 ≥ 0.8
    assert isinstance(result, BufferResult)
    assert result.is_partial is True
    assert result.missing_layers == frozenset({"l2", "l3"})


def test_eps_trigger_protects_layer_above_threshold():
    """Layer with score > ε·total cannot be missing at fire time."""
    buf = LayerBuffer(epsilon=0.2)
    manifest = build_manifest(
        round_num=0, source_node_id="node-1",
        importance_scores={"big": 0.7, "med": 0.2, "tiny": 0.1},
    )
    buf.register_manifest(manifest)

    # target = 0.8·1.0 = 0.8.  Without "big", max receivable is 0.3 < 0.8.
    # So no partial fire is possible without "big".
    assert _add(buf, "med",  0, 3) is None
    assert _add(buf, "tiny", 1, 3) is None
    result = _add(buf, "big", 2, 3)
    # Now all three are in, so it's the all-in trigger (not partial).
    assert isinstance(result, BufferResult)
    assert result.is_partial is False


def test_must_receive_blocks_completion():
    """A must_receive layer must arrive even when coverage already met."""
    buf = LayerBuffer(epsilon=0.5)
    manifest = build_manifest(
        round_num=0, source_node_id="node-1",
        importance_scores={"a": 0.6, "must_b": 0.3, "c": 0.1},
        must_receive_predicate=lambda name, s: name == "must_b",
    )
    buf.register_manifest(manifest)

    # target = 0.5·1.0 = 0.5.  "a" alone (0.6) meets the score target but
    # must_b has not arrived; do NOT fire.
    assert _add(buf, "a", 0, 3) is None
    # Add the must_receive layer; coverage is 0.9 ≥ 0.5, must satisfied.
    result = _add(buf, "must_b", 1, 3)
    assert isinstance(result, BufferResult)
    assert result.is_partial is True
    assert result.missing_layers == frozenset({"c"})


# ---------------------------------------------------------------------------
# Manifest ordering edge cases
# ---------------------------------------------------------------------------

def test_layer_before_manifest_fires_on_manifest_registration_via_next_layer():
    """If a layer arrives before its manifest, coverage is checked only when
    the next layer arrives after the manifest is registered.

    (We do not retroactively re-check coverage on register_manifest; that is a
    deliberate simplicity choice — re-checking on register adds complexity for
    a vanishingly rare ordering, given the manifest goes on class 0.)
    """
    buf = LayerBuffer(epsilon=0.2)
    # Layer arrives first
    assert _add(buf, "l0", 0, 3) is None
    # Manifest arrives now
    manifest = build_manifest(
        round_num=0, source_node_id="node-1",
        importance_scores={"l0": 0.5, "l1": 0.3, "l2": 0.2},
    )
    buf.register_manifest(manifest)
    # The next layer should trigger coverage check
    result = _add(buf, "l1", 1, 3)
    assert isinstance(result, BufferResult)
    assert result.is_partial is True
    assert result.missing_layers == frozenset({"l2"})


def test_duplicate_manifest_overwrites_with_warning(caplog):
    """Two manifests for the same (source, round) — second overwrites."""
    import logging
    buf = LayerBuffer(epsilon=0.2)
    m1 = build_manifest(
        0, "node-1", {"l0": 1.0},
    )
    m2 = build_manifest(
        0, "node-1", {"l0": 1.0, "l1": 0.5},
    )
    with caplog.at_level(logging.WARNING):
        buf.register_manifest(m1)
        buf.register_manifest(m2)
    assert any("Duplicate manifest" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Late-arrival handling (Step-3 scope: silent drop)
# ---------------------------------------------------------------------------

def test_late_layer_after_fire_returns_none():
    """After the trigger fires, further layers for the same (source, round)
    are routed to the late-layer policy and the buffer returns None."""
    buf = LayerBuffer(epsilon=0.2)
    manifest = build_manifest(
        0, "node-1",
        {"l0": 0.6, "l1": 0.3, "tail": 0.1},
    )
    buf.register_manifest(manifest)
    _add(buf, "l0", 0, 3)
    fired = _add(buf, "l1", 1, 3)
    assert isinstance(fired, BufferResult)
    assert fired.is_partial is True

    # "tail" arrives late
    late = _add(buf, "tail", 2, 3)
    assert late is None


def test_late_layer_invokes_policy_exactly_once_per_arrival():
    """Late arrivals are handed to the configured policy once each."""
    from src.training.late_layer_policy import LateLayerPolicy

    calls: list[str] = []

    class RecordingPolicy(LateLayerPolicy):
        def on_late_arrival(self, source_node, round_num, layer_name,
                            array, num_samples):
            calls.append(layer_name)

    buf = LayerBuffer(epsilon=0.2, late_layer_policy=RecordingPolicy())
    manifest = build_manifest(
        0, "node-1",
        {"l0": 0.6, "l1": 0.3, "tail1": 0.05, "tail2": 0.05},
    )
    buf.register_manifest(manifest)
    _add(buf, "l0", 0, 4)
    fired = _add(buf, "l1", 1, 4)
    assert isinstance(fired, BufferResult)
    assert fired.is_partial is True
    # Two late layers; each should be reported exactly once.
    _add(buf, "tail1", 2, 4)
    _add(buf, "tail2", 3, 4)
    assert calls == ["tail1", "tail2"]


def test_late_layer_policy_exception_is_swallowed():
    """A buggy policy must not crash the receive path."""
    from src.training.late_layer_policy import LateLayerPolicy

    class ExplodingPolicy(LateLayerPolicy):
        def on_late_arrival(self, *args, **kwargs):
            raise RuntimeError("boom")

    buf = LayerBuffer(epsilon=0.2, late_layer_policy=ExplodingPolicy())
    manifest = build_manifest(
        0, "node-1",
        {"l0": 0.7, "l1": 0.2, "tail": 0.1},
    )
    buf.register_manifest(manifest)
    _add(buf, "l0", 0, 3)
    fired = _add(buf, "l1", 1, 3)
    assert isinstance(fired, BufferResult)
    # Late arrival; should not raise even though the policy does.
    result = _add(buf, "tail", 2, 3)
    assert result is None


# ---------------------------------------------------------------------------
# Garbage collection
# ---------------------------------------------------------------------------

def test_clear_stale_removes_manifests_and_fired_markers():
    buf = LayerBuffer(epsilon=0.2)
    manifest = build_manifest(
        round_num=0, source_node_id="node-1",
        importance_scores={"a": 1.0},
    )
    buf.register_manifest(manifest)
    # Fire the round
    _add(buf, "a", 0, 1)
    assert ("node-1", 0) in buf._fired
    assert ("node-1", 0) in buf._manifests

    # Advance well past staleness
    buf.clear_stale(current_round=10, max_staleness=2)
    assert ("node-1", 0) not in buf._fired
    assert ("node-1", 0) not in buf._manifests


def test_clear_stale_keeps_recent_manifests():
    buf = LayerBuffer(epsilon=0.2)
    manifest = build_manifest(
        round_num=5, source_node_id="node-1",
        importance_scores={"a": 1.0},
    )
    buf.register_manifest(manifest)
    buf.clear_stale(current_round=6, max_staleness=2)
    assert ("node-1", 5) in buf._manifests
