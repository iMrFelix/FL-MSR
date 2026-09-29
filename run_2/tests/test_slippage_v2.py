"""Tests for the gate-admitted slippage modes and canonical aggregation order.

Covers the FedAvg aggregator changes from writeup/01-candidate-selection.md
(§3 fix 5, §7 slippage set, ruling G3):

- Canonical accumulation order: permuted arrival order produces a
  bitwise-identical aggregate (restores the ε=0 bug-detector under fp
  non-associativity).
- 'renorm': weighted mean over arrived contributors only, weights
  renormalized to sum to 1 (no stale-fill term).
- 'recycle': missing contributor's value = current global + the layer's
  previous aggregated delta (FedLUAR-adapted), with stale-fill fallback
  on the first aggregation and delta rolling across rounds.
- Zero-arrived layers retain the current global exactly, in every mode.
- Mode-name aliases (schema literals vs short forms), telemetry dict,
  per-(layer, source) inclusion counters, late-layer policy registry.
"""

import logging

import numpy as np
import pytest

from src.algorithms.base import TrainingUpdate
from src.algorithms.fedavg import FedAvg
from src.training.late_layer_policy import (
    LateLayerPolicy,
    RecycleLastDeltaPolicy,
    RenormalizePolicy,
    make_policy,
)

KERNEL = "layer_0/kernel"
BIAS = "layer_0/bias"
KERNEL_SHAPE = (4, 3)
BIAS_SHAPE = (3,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params(kernel: float, bias: float | None = None) -> dict[str, np.ndarray]:
    """Constant-valued params; bias=None omits the bias layer (partial)."""
    p = {KERNEL: np.full(KERNEL_SHAPE, kernel, dtype=np.float32)}
    if bias is not None:
        p[BIAS] = np.full(BIAS_SHAPE, bias, dtype=np.float32)
    return p


def _update(
    source: str,
    round_num: int,
    parameters: dict[str, np.ndarray],
    num_samples: int,
) -> TrainingUpdate:
    return TrainingUpdate(
        source_node=source,
        round_num=round_num,
        parameters=parameters,
        num_samples=num_samples,
    )


def _aggregator(mode: str | None, workers: list[str]) -> FedAvg:
    config: dict = {"role": "aggregator"}
    if mode is not None:
        config["late_layer_policy"] = mode
    return FedAvg("node-0", workers, config)


# ---------------------------------------------------------------------------
# Mode selection (aliases, defaults, rejection)
# ---------------------------------------------------------------------------

class TestModeSelection:

    def test_default_mode_is_drop(self):
        algo = FedAvg("node-0", ["node-1"], {"role": "aggregator"})
        assert algo.slippage_mode == "drop"

    @pytest.mark.parametrize(
        ("config_value", "expected"),
        [
            ("drop", "drop"),
            ("renorm", "renorm"),
            ("renormalize", "renorm"),  # schema literal
            ("recycle", "recycle"),
            ("recycle_last_delta", "recycle"),  # schema literal
        ],
    )
    def test_aliases_normalize(self, config_value: str, expected: str):
        algo = _aggregator(config_value, ["node-1"])
        assert algo.slippage_mode == expected

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown late_layer_policy"):
            FedAvg(
                "node-0", ["node-1"],
                {"role": "aggregator", "late_layer_policy": "teleport"},
            )

    def test_worker_accepts_mode_config(self):
        """Mode parsing is role-independent; workers must not crash on the
        new schema literals (they share the training config dict)."""
        algo = FedAvg(
            "node-1", ["node-0"],
            {"role": "worker", "late_layer_policy": "recycle_last_delta"},
        )
        assert algo.slippage_mode == "recycle"

    @pytest.mark.asyncio
    async def test_worker_aggregate_unaffected_by_mode(self):
        """Workers adopt the global model regardless of slippage mode."""
        algo = FedAvg(
            "node-1", ["node-0"],
            {"role": "worker", "late_layer_policy": "renormalize"},
        )
        await algo.on_update_received(
            _update("node-0", 0, _params(10.0, 10.0), 100)
        )
        result = await algo.aggregate(_params(1.0, 1.0))
        for layer in result:
            np.testing.assert_allclose(result[layer], 10.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Late-layer policy registry (receiver-side construction by name)
# ---------------------------------------------------------------------------

class TestPolicyRegistry:

    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            ("recycle_last_delta", RecycleLastDeltaPolicy),
            ("recycle", RecycleLastDeltaPolicy),
            ("renormalize", RenormalizePolicy),
            ("renorm", RenormalizePolicy),
        ],
    )
    def test_make_policy_new_names(self, name: str, cls: type):
        policy = make_policy(name)
        assert isinstance(policy, cls)
        assert isinstance(policy, LateLayerPolicy)

    @pytest.mark.parametrize(
        "name", ["recycle_last_delta", "renormalize"],
    )
    def test_new_policies_do_not_raise_on_late_arrival(self, name: str):
        policy = make_policy(name)
        policy.on_late_arrival(
            source_node="node-1", round_num=0,
            layer_name=KERNEL,
            array=np.ones((2, 2), dtype=np.float32),
            num_samples=100,
        )  # must not raise


# ---------------------------------------------------------------------------
# Canonical aggregation order (G-cluster fix 5)
# ---------------------------------------------------------------------------

class TestCanonicalOrder:
    """Permuted arrival order must produce a bitwise-identical aggregate.

    fp addition is non-associative, so this only holds if the aggregator
    accumulates in a canonical (source-sorted) order rather than arrival
    order.  Random float32 values with non-dyadic weights make any
    order-dependence visible in the last bits.
    """

    def _random_updates(
        self, round_num: int, *, partial: bool = False,
    ) -> dict[str, TrainingUpdate]:
        rng = np.random.default_rng(1234)
        samples = {"node-1": 17, "node-2": 89, "node-3": 253}
        updates = {}
        for source, n_k in samples.items():
            params = {
                KERNEL: rng.standard_normal(KERNEL_SHAPE).astype(np.float32),
                BIAS: rng.standard_normal(BIAS_SHAPE).astype(np.float32),
            }
            if partial and source == "node-2":
                del params[BIAS]  # bias slipped past the trigger
            updates[source] = _update(source, round_num, params, n_k)
        return updates

    @pytest.mark.asyncio
    async def test_permuted_arrival_bitwise_identical(self):
        workers = ["node-1", "node-2", "node-3"]
        updates = self._random_updates(0)
        local = _params(0.5, 0.5)

        results = []
        for arrival_order in (
            ["node-1", "node-2", "node-3"],
            ["node-3", "node-1", "node-2"],
            ["node-2", "node-3", "node-1"],
        ):
            algo = _aggregator("drop", workers)
            for source in arrival_order:
                await algo.on_update_received(updates[source])
            results.append(await algo.aggregate(local))

        for other in results[1:]:
            assert set(other) == set(results[0])
            for layer in results[0]:
                assert np.array_equal(results[0][layer], other[layer]), (
                    f"layer {layer} differs across arrival orders"
                )

    @pytest.mark.asyncio
    async def test_permuted_arrival_bitwise_identical_partial_renorm(self):
        """Order invariance must also hold on the renorm slippage path."""
        workers = ["node-1", "node-2", "node-3"]
        updates = self._random_updates(0, partial=True)
        local = _params(0.5, 0.5)

        results = []
        for arrival_order in (
            ["node-2", "node-3", "node-1"],
            ["node-1", "node-3", "node-2"],
        ):
            algo = _aggregator("renorm", workers)
            for source in arrival_order:
                await algo.on_update_received(updates[source])
            results.append(await algo.aggregate(local))

        for layer in results[0]:
            assert np.array_equal(results[0][layer], results[1][layer]), (
                f"layer {layer} differs across arrival orders"
            )

    @pytest.mark.asyncio
    async def test_layer_name_order_is_sorted(self):
        """Output dict ordering is sorted by layer name, so downstream
        serialization order does not depend on set-iteration hash order."""
        algo = _aggregator("drop", ["node-1"])
        await algo.on_update_received(
            _update("node-1", 0, _params(1.0, 1.0), 100)
        )
        result = await algo.aggregate(_params(0.0, 0.0))
        assert list(result) == sorted(result)


# ---------------------------------------------------------------------------
# 'renorm': arrived-mass renormalization
# ---------------------------------------------------------------------------

class TestRenorm:

    @pytest.mark.asyncio
    async def test_weighted_mean_over_arrived_only(self):
        """Missing contributor's weight is renormalized away — the layer is
        the sample-weighted mean of the arrived contributors, with no
        stale-fill term from local_params."""
        algo = _aggregator("renorm", ["node-1", "node-2", "node-3"])
        await algo.on_update_received(
            _update("node-1", 0, _params(10.0, 10.0), 100)
        )
        # node-2's bias slipped past the trigger.
        await algo.on_update_received(
            _update("node-2", 0, _params(20.0), 300)
        )
        await algo.on_update_received(
            _update("node-3", 0, _params(30.0, 30.0), 600)
        )

        # local_params=999 must not leak into the renormalized layer.
        result = await algo.aggregate(_params(999.0, 999.0))

        # kernel (complete): (10*100 + 20*300 + 30*600) / 1000 = 25
        np.testing.assert_allclose(result[KERNEL], 25.0, rtol=1e-6)
        # bias (renorm over node-1, node-3): (10*100 + 30*600) / 700
        np.testing.assert_allclose(result[BIAS], 19000.0 / 700.0, rtol=1e-6)

        telemetry = algo.last_aggregation_telemetry
        assert telemetry[BIAS]["filled"] == "renorm"
        assert telemetry[BIAS]["arrived_sources"] == ["node-1", "node-3"]
        assert telemetry[KERNEL]["filled"] == "none"

    @pytest.mark.asyncio
    async def test_complete_layers_match_drop_mode_bitwise(self):
        """With no missing layers, renorm is the plain weighted average —
        identical to the control arm down to the last bit."""
        rng = np.random.default_rng(7)
        updates = [
            _update(
                f"node-{i}", 0,
                {
                    KERNEL: rng.standard_normal(KERNEL_SHAPE).astype(np.float32),
                    BIAS: rng.standard_normal(BIAS_SHAPE).astype(np.float32),
                },
                n_k,
            )
            for i, n_k in ((1, 130), (2, 270))
        ]
        local = _params(0.0, 0.0)

        results = {}
        for mode in ("renorm", "drop"):
            algo = _aggregator(mode, ["node-1", "node-2"])
            for u in updates:
                await algo.on_update_received(u)
            results[mode] = await algo.aggregate(local)

        for layer in results["drop"]:
            assert np.array_equal(results["drop"][layer], results["renorm"][layer])

    @pytest.mark.asyncio
    async def test_zero_arrived_mass_falls_back_to_uniform(self):
        """If every arrived contributor of a layer carries zero sample
        weight, w_k / W_arr is 0/0; the limit taken is uniform over the
        arrived set."""
        algo = _aggregator("renorm", ["node-1", "node-2"])
        # node-1 has 0 samples but contributed the bias.
        await algo.on_update_received(
            _update("node-1", 0, _params(10.0, 12.0), 0)
        )
        # node-2 has all the samples but its bias slipped.
        await algo.on_update_received(
            _update("node-2", 0, _params(20.0), 400)
        )

        result = await algo.aggregate(_params(999.0, 999.0))

        # kernel: (10*0 + 20*400) / 400 = 20
        np.testing.assert_allclose(result[KERNEL], 20.0, rtol=1e-6)
        # bias: only node-1 arrived, with zero weight → uniform → 12
        np.testing.assert_allclose(result[BIAS], 12.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# 'recycle': recycle-last-delta (FedLUAR-adapted)
# ---------------------------------------------------------------------------

class TestRecycle:

    @pytest.mark.asyncio
    async def test_recycle_value_is_global_plus_last_delta(self):
        """Hand-computed 2-round scenario.

        Round 0 (all layers arrive):
            θ1 = (10·100 + 20·300) / 400 = 17.5     (both layers)
            last_delta = θ1 − θ0 = 17.5 − 2.0 = 15.5
        Round 1 (node-2's bias slips):
            recycled bias for node-2 = global + last_delta = 17.5 + 15.5 = 33
            bias = 0.25·8 + 0.75·33 = 26.75
            kernel = 0.25·8 + 0.75·24 = 20.0
        """
        algo = _aggregator("recycle_last_delta", ["node-1", "node-2"])

        # --- Round 0: complete updates, initial global θ0 = 2.0 ---
        await algo.on_update_received(
            _update("node-1", 0, _params(10.0, 10.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 0, _params(20.0, 20.0), 300)
        )
        theta1 = await algo.aggregate(_params(2.0, 2.0))
        np.testing.assert_allclose(theta1[KERNEL], 17.5, atol=1e-6)
        np.testing.assert_allclose(theta1[BIAS], 17.5, atol=1e-6)
        algo.advance_round()

        # --- Round 1: node-2's bias slipped past the trigger ---
        await algo.on_update_received(
            _update("node-1", 1, _params(8.0, 8.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 1, _params(24.0), 300)
        )
        # The engine sets the model to θ1 between rounds, so the current
        # global passed in is exactly the previous aggregate.
        result = await algo.aggregate(theta1)

        np.testing.assert_allclose(result[KERNEL], 20.0, atol=1e-6)
        np.testing.assert_allclose(result[BIAS], 26.75, atol=1e-6)

        telemetry = algo.last_aggregation_telemetry
        assert telemetry[BIAS]["filled"] == "recycle"
        assert telemetry[BIAS]["arrived_sources"] == ["node-1"]
        assert telemetry[KERNEL]["filled"] == "none"

    @pytest.mark.asyncio
    async def test_first_round_falls_back_to_stale_fill(self):
        """No delta history exists at the first aggregation: a missing
        contributor is stale-filled from the current global, exactly like
        the drop control."""
        algo = _aggregator("recycle", ["node-1", "node-2"])
        await algo.on_update_received(
            _update("node-1", 0, _params(10.0, 10.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 0, _params(20.0), 300)
        )

        result = await algo.aggregate(_params(5.0, 5.0))

        # kernel: (10*100 + 20*300) / 400 = 17.5
        np.testing.assert_allclose(result[KERNEL], 17.5, atol=1e-6)
        # bias: stale-fill → (10*100 + 5*300) / 400 = 6.25
        np.testing.assert_allclose(result[BIAS], 6.25, atol=1e-6)
        assert algo.last_aggregation_telemetry[BIAS]["filled"] == "stale"

    @pytest.mark.asyncio
    async def test_last_delta_rolls_across_rounds(self):
        """The recycled delta is the motion between the LAST TWO
        aggregates, not the first ever computed.

        Continuing the 2-round scenario:
            θ2 = {kernel: 20.0, bias: 26.75}
            last_delta after round 1 = θ2 − θ1 = {kernel: 2.5, bias: 9.25}
        Round 2 (node-2's bias slips again):
            recycled bias = 26.75 + 9.25 = 36
            bias = 0.25·4 + 0.75·36 = 28.0
            kernel = 0.25·6 + 0.75·10 = 9.0
        """
        algo = _aggregator("recycle", ["node-1", "node-2"])

        await algo.on_update_received(
            _update("node-1", 0, _params(10.0, 10.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 0, _params(20.0, 20.0), 300)
        )
        theta1 = await algo.aggregate(_params(2.0, 2.0))
        algo.advance_round()

        await algo.on_update_received(
            _update("node-1", 1, _params(8.0, 8.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 1, _params(24.0), 300)
        )
        theta2 = await algo.aggregate(theta1)
        algo.advance_round()

        await algo.on_update_received(
            _update("node-1", 2, _params(6.0, 4.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 2, _params(10.0), 300)
        )
        result = await algo.aggregate(theta2)

        np.testing.assert_allclose(result[KERNEL], 9.0, atol=1e-6)
        np.testing.assert_allclose(result[BIAS], 28.0, atol=1e-6)
        assert algo.last_aggregation_telemetry[BIAS]["filled"] == "recycle"

    @pytest.mark.asyncio
    async def test_recycle_events_logged_per_layer(self, caplog):
        algo = _aggregator("recycle", ["node-1", "node-2"])
        await algo.on_update_received(
            _update("node-1", 0, _params(10.0, 10.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 0, _params(20.0, 20.0), 300)
        )
        theta1 = await algo.aggregate(_params(2.0, 2.0))
        algo.advance_round()

        await algo.on_update_received(
            _update("node-1", 1, _params(8.0, 8.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 1, _params(24.0), 300)
        )
        with caplog.at_level(logging.INFO, logger="src.algorithms.fedavg"):
            await algo.aggregate(theta1)

        recycle_logs = [
            r.message for r in caplog.records if "recycled last-delta" in r.message
        ]
        assert len(recycle_logs) == 1
        assert BIAS in recycle_logs[0]


# ---------------------------------------------------------------------------
# Degenerate shed: zero arrived contributors for a layer
# ---------------------------------------------------------------------------

class TestZeroArrivedLayer:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["drop", "renorm", "recycle"])
    async def test_global_retained_bitwise(self, mode: str, caplog):
        """A layer no worker contributed keeps the current global exactly
        (bitwise — no fp round-trip through the weighted sum), with a
        loud warning, in every slippage mode."""
        rng = np.random.default_rng(99)
        local = {
            KERNEL: rng.standard_normal(KERNEL_SHAPE).astype(np.float32),
            BIAS: rng.standard_normal(BIAS_SHAPE).astype(np.float32),
        }

        algo = _aggregator(mode, ["node-1", "node-2"])
        # Both workers' bias slipped past the trigger.
        await algo.on_update_received(_update("node-1", 0, _params(10.0), 100))
        await algo.on_update_received(_update("node-2", 0, _params(20.0), 300))

        with caplog.at_level(logging.WARNING, logger="src.algorithms.fedavg"):
            result = await algo.aggregate(local)

        assert np.array_equal(result[BIAS], local[BIAS])
        # Retained as a copy: mutating the result must not alias the
        # caller's array.
        assert result[BIAS] is not local[BIAS]
        # kernel still aggregates normally: (10*100 + 20*300)/400 = 17.5
        np.testing.assert_allclose(result[KERNEL], 17.5, atol=1e-6)

        telemetry = algo.last_aggregation_telemetry
        assert telemetry[BIAS]["arrived_sources"] == []
        assert telemetry[BIAS]["filled"] == "stale"
        assert any("ZERO arrived" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_recycle_does_not_compound_on_zero_arrived(self):
        """Zero-arrived retention overrides recycling: with no fresh
        evidence the delta must NOT be re-applied (ratchet guard), and the
        rolled delta for the retained layer collapses to ~0."""
        algo = _aggregator("recycle", ["node-1", "node-2"])

        await algo.on_update_received(
            _update("node-1", 0, _params(10.0, 10.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 0, _params(20.0, 20.0), 300)
        )
        theta1 = await algo.aggregate(_params(2.0, 2.0))  # bias 17.5, Δ 15.5
        algo.advance_round()

        # Round 1: bias missing from BOTH workers → retained, not 17.5+15.5.
        await algo.on_update_received(_update("node-1", 1, _params(8.0), 100))
        await algo.on_update_received(_update("node-2", 1, _params(24.0), 300))
        theta2 = await algo.aggregate(theta1)
        assert np.array_equal(theta2[BIAS], theta1[BIAS])
        algo.advance_round()

        # Round 2: node-2's bias missing again.  The rolled bias delta is
        # θ2 − θ1 = 0, so the recycled value equals the current global —
        # equivalent to stale-fill, no compounding of the old 15.5.
        await algo.on_update_received(
            _update("node-1", 2, _params(6.0, 9.5), 100)
        )
        await algo.on_update_received(_update("node-2", 2, _params(10.0), 300))
        result = await algo.aggregate(theta2)
        # bias = 0.25·9.5 + 0.75·(17.5 + 0) = 2.375 + 13.125 = 15.5
        np.testing.assert_allclose(result[BIAS], 15.5, atol=1e-6)


# ---------------------------------------------------------------------------
# Drop control: pre-gate stale-fill semantics preserved
# ---------------------------------------------------------------------------

class TestDropControl:

    @pytest.mark.asyncio
    async def test_stale_fill_unchanged(self):
        """The control arm keeps the original partial-update semantics:
        missing contribution = weight * local_params[layer]."""
        algo = _aggregator("drop", ["node-1", "node-2"])
        await algo.on_update_received(
            _update("node-1", 0, _params(10.0, 10.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 0, _params(20.0), 100)
        )

        result = await algo.aggregate(_params(5.0, 5.0))

        np.testing.assert_allclose(result[KERNEL], 15.0, atol=1e-6)
        # bias: 0.5*10 + 0.5*5 = 7.5
        np.testing.assert_allclose(result[BIAS], 7.5, atol=1e-6)

        telemetry = algo.last_aggregation_telemetry
        assert telemetry[BIAS]["filled"] == "stale"
        assert telemetry[BIAS]["arrived_sources"] == ["node-1"]


# ---------------------------------------------------------------------------
# Telemetry and inclusion counters
# ---------------------------------------------------------------------------

class TestTelemetryAndCounters:

    @pytest.mark.asyncio
    async def test_complete_round_telemetry_all_none(self):
        algo = _aggregator("drop", ["node-1", "node-2"])
        await algo.on_update_received(
            _update("node-1", 0, _params(1.0, 1.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 0, _params(2.0, 2.0), 100)
        )
        await algo.aggregate(_params(0.0, 0.0))

        telemetry = algo.last_aggregation_telemetry
        assert set(telemetry) == {KERNEL, BIAS}
        for layer in telemetry:
            assert telemetry[layer]["filled"] == "none"
            assert telemetry[layer]["arrived_sources"] == ["node-1", "node-2"]

    @pytest.mark.asyncio
    async def test_inclusion_counters_accumulate(self):
        """Per-(layer, source) counters track realized inclusion across
        rounds; absent sources appear with an explicit 0 count."""
        algo = _aggregator("drop", ["node-1", "node-2"])

        # Round 0: complete.
        await algo.on_update_received(
            _update("node-1", 0, _params(1.0, 1.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 0, _params(2.0, 2.0), 100)
        )
        await algo.aggregate(_params(0.0, 0.0))
        algo.advance_round()

        # Round 1: node-2's bias slipped.
        await algo.on_update_received(
            _update("node-1", 1, _params(1.0, 1.0), 100)
        )
        await algo.on_update_received(_update("node-2", 1, _params(2.0), 100))
        await algo.aggregate(_params(0.0, 0.0))

        assert algo.aggregation_count == 2
        assert algo.inclusion_counts[KERNEL] == {"node-1": 2, "node-2": 2}
        assert algo.inclusion_counts[BIAS] == {"node-1": 2, "node-2": 1}

    @pytest.mark.asyncio
    async def test_telemetry_refreshed_each_round(self):
        algo = _aggregator("drop", ["node-1", "node-2"])
        await algo.on_update_received(_update("node-1", 0, _params(1.0), 100))
        await algo.on_update_received(
            _update("node-2", 0, _params(2.0, 2.0), 100)
        )
        await algo.aggregate(_params(0.0, 0.0))
        assert algo.last_aggregation_telemetry[BIAS]["filled"] == "stale"
        algo.advance_round()

        await algo.on_update_received(
            _update("node-1", 1, _params(1.0, 1.0), 100)
        )
        await algo.on_update_received(
            _update("node-2", 1, _params(2.0, 2.0), 100)
        )
        await algo.aggregate(_params(0.0, 0.0))
        assert algo.last_aggregation_telemetry[BIAS]["filled"] == "none"
