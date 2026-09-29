"""Tests for the v2 importance metrics (shipped-delta input, G2 score roles).

The metric functions are the trigger-accounting substrate of every overnight
arm (writeup/01-candidate-selection.md §7): delta_sq_norm is frozen as the
ε-trigger metric for ALL arms, the rest are sched-only.  Exact arithmetic is
pinned here so a silent change in a reduction would surface as a test diff,
not as an invalid cross-arm comparison.
"""

import numpy as np
import pytest

from src.importance.gradient_norm import GradientNormImportance
from src.importance.metrics_v2 import (
    DeltaSqNormImportance,
    RawNormImportance,
    RelativeImportance,
    SnrReweightImportance,
    UniformImportance,
    delta_sq_norm,
    make_metric_v2,
    raw_norm,
    registered_metrics_v2,
    relative,
    snr_reweight,
)


# ---------------------------------------------------------------------------
# Pure functions: exact values
# ---------------------------------------------------------------------------

class TestDeltaSqNorm:

    def test_exact_value(self):
        deltas = {"a": np.array([[3.0, 4.0]])}
        assert delta_sq_norm(deltas) == {"a": pytest.approx(25.0)}

    def test_multi_layer_and_shapes(self):
        deltas = {
            "conv/kernel": np.ones((2, 2, 3)),   # 12 ones -> 12.0
            "conv/bias": np.array([-2.0]),       # 4.0
        }
        scores = delta_sq_norm(deltas)
        assert scores["conv/kernel"] == pytest.approx(12.0)
        assert scores["conv/bias"] == pytest.approx(4.0)

    def test_zero_and_empty(self):
        scores = delta_sq_norm({
            "zero": np.zeros(5),
            "empty": np.array([]),
        })
        assert scores == {"zero": 0.0, "empty": 0.0}

    def test_float32_input_reduces_in_float64(self):
        # Sum of squares of many float32 values: a float32 accumulator
        # would drift; the implementation must upcast before reducing.
        arr = np.full(100_000, 0.1, dtype=np.float32)
        expected = float(
            np.asarray(arr, dtype=np.float64) @ np.asarray(arr, dtype=np.float64)
        )
        assert delta_sq_norm({"a": arr})["a"] == pytest.approx(expected, rel=1e-12)


class TestRelative:

    def test_exact_value(self):
        deltas = {"a": np.array([3.0, 4.0])}     # ||Δ|| = 5
        params = {"a": np.array([2.0])}          # ||θ|| = 2
        assert relative(deltas, params)["a"] == pytest.approx(2.5)

    def test_zero_weights_guard(self):
        # ||θ|| = 0 -> guard denominator 1e-12; the spec'd formula, kept
        # explicit so a "helpful" clamp would fail this test.
        deltas = {"a": np.array([1e-6])}
        params = {"a": np.zeros(3)}
        assert relative(deltas, params)["a"] == pytest.approx(1e-6 / 1e-12)

    def test_missing_param_key_raises(self):
        with pytest.raises(KeyError):
            relative({"a": np.ones(2)}, {})


class TestSnrReweight:

    def test_exact_value(self):
        # Δ = [2, 4]: mean|Δ| = 3, std = 1 -> SNR = 3 / (1 + 1e-12)
        deltas = {"a": np.array([2.0, 4.0])}
        base = {"a": 10.0}
        assert snr_reweight(deltas, base)["a"] == pytest.approx(30.0)

    def test_constant_delta_hits_guard_not_zero_division(self):
        # std = 0 -> ratio capped by the 1e-12 guard (documented behaviour:
        # sched-only, so an extreme value cannot corrupt the trigger).
        deltas = {"a": np.full(4, 2.0)}
        score = snr_reweight(deltas, {"a": 1.0})["a"]
        assert score == pytest.approx(2.0 / 1e-12)

    def test_empty_delta_scores_zero(self):
        assert snr_reweight({"a": np.array([])}, {"a": 7.0}) == {"a": 0.0}

    def test_base_scales_linearly(self):
        deltas = {"a": np.array([2.0, 4.0]), "b": np.array([2.0, 4.0])}
        scores = snr_reweight(deltas, {"a": 1.0, "b": 5.0})
        assert scores["b"] == pytest.approx(5.0 * scores["a"])


class TestRawNorm:

    def test_exact_value(self):
        assert raw_norm({"a": np.array([3.0, 4.0])})["a"] == pytest.approx(5.0)

    def test_degenerate_control_property(self):
        # E6: raw norm conflates magnitude with parameter count — a huge
        # near-zero kernel can outrank a small decisive bias.  The control
        # must KEEP this pathology.
        kernel = np.full(40_000, 0.01)   # ||.|| = 2.0
        bias = np.array([1.0])           # ||.|| = 1.0
        scores = raw_norm({"kernel": kernel, "bias": bias})
        assert scores["kernel"] > scores["bias"]


# ---------------------------------------------------------------------------
# ImportanceMetric adapters
# ---------------------------------------------------------------------------

class TestAdapters:

    _DELTAS = {
        "w": np.array([[1.0, -2.0], [2.0, 0.0]]),
        "b": np.array([0.5, 0.5]),
    }
    _PARAMS = {
        "w": np.array([[2.0, 0.0], [0.0, 0.0]]),
        "b": np.array([3.0, 4.0]),
    }

    def test_delta_sq_norm_adapter_matches_function(self):
        metric = DeltaSqNormImportance()
        scores = metric.compute_all(self._PARAMS, self._DELTAS, 0, {})
        assert scores == pytest.approx(delta_sq_norm(self._DELTAS))

    def test_relative_adapter_matches_function(self):
        metric = RelativeImportance()
        scores = metric.compute_all(self._PARAMS, self._DELTAS, 0, {})
        assert scores == pytest.approx(relative(self._DELTAS, self._PARAMS))

    def test_snr_adapter_uses_delta_sq_norm_base(self):
        metric = SnrReweightImportance()
        scores = metric.compute_all(self._PARAMS, self._DELTAS, 0, {})
        expected = snr_reweight(self._DELTAS, delta_sq_norm(self._DELTAS))
        assert scores == pytest.approx(expected)

    def test_raw_norm_adapter_matches_function(self):
        metric = RawNormImportance()
        scores = metric.compute_all(self._PARAMS, self._DELTAS, 0, {})
        assert scores == pytest.approx(raw_norm(self._DELTAS))

    def test_raw_norm_parity_with_legacy_gradient_norm(self):
        """The control arm must preserve GradientNormImportance bit-for-bit."""
        rng = np.random.default_rng(7)
        legacy = GradientNormImportance()
        control = RawNormImportance()
        for shape in [(3,), (4, 5), (2, 3, 4)]:
            arr = rng.normal(size=shape).astype(np.float32)
            weights = rng.normal(size=shape).astype(np.float32)
            assert control.compute("l", weights, arr, 0, {}) == pytest.approx(
                legacy.compute("l", weights, arr, 0, {}), rel=1e-6,
            )

    @pytest.mark.parametrize("metric_cls", [
        DeltaSqNormImportance, RelativeImportance,
        SnrReweightImportance, RawNormImportance,
    ])
    def test_missing_update_falls_back_to_one(self, metric_cls):
        """None update -> 1.0, the legacy uniform fallback (all metrics)."""
        score = metric_cls().compute("l", np.ones(3), None, 0, {})
        assert score == 1.0


class TestUniform:

    def test_score_is_one_regardless_of_update(self):
        """Importance-blind by construction: 1.0 with or without a delta."""
        m = UniformImportance()
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(64, 64, 3, 3)).astype(np.float32) * 100.0
        assert m.compute("l", np.ones(3), arr, 5, {}) == 1.0
        assert m.compute("l", np.ones(3), None, 5, {}) == 1.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:

    def test_registered_names_match_schema_literals(self):
        # Must mirror src/config/schema.py:TrainingConfig.importance_metric_v2.
        assert registered_metrics_v2() == [
            "delta_sq_norm", "raw_norm", "relative", "snr_reweight",
            "uniform",
        ]

    @pytest.mark.parametrize("name,cls", [
        ("delta_sq_norm", DeltaSqNormImportance),
        ("relative", RelativeImportance),
        ("snr_reweight", SnrReweightImportance),
        ("raw_norm", RawNormImportance),
        ("uniform", UniformImportance),
    ])
    def test_factory_builds_by_literal(self, name, cls):
        assert isinstance(make_metric_v2(name), cls)

    def test_unknown_name_raises_with_listing(self):
        with pytest.raises(ValueError, match="delta_sq_norm"):
            make_metric_v2("fisher")

    def test_factory_returns_fresh_instances(self):
        assert make_metric_v2("delta_sq_norm") is not make_metric_v2("delta_sq_norm")
