"""Tests for the late-layer policy interface and DropPolicy.

See docs/extensions/03-late-layer-policy.md.
"""

import numpy as np
import pytest

from src.training.late_layer_policy import (
    DropPolicy,
    LateLayerPolicy,
    make_policy,
)


def test_drop_policy_does_not_raise():
    p = DropPolicy()
    p.on_late_arrival(
        source_node="node-1", round_num=0,
        layer_name="conv_0/kernel",
        array=np.array([1.0], dtype=np.float32),
        num_samples=100,
    )  # must not raise


def test_drop_policy_is_late_layer_policy():
    assert isinstance(DropPolicy(), LateLayerPolicy)


def test_make_policy_returns_drop():
    p = make_policy("drop")
    assert isinstance(p, DropPolicy)


def test_make_policy_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown late-layer policy"):
        make_policy("teleport")


def test_make_policy_error_lists_registered_names():
    with pytest.raises(ValueError, match="drop"):
        make_policy("nonexistent")
