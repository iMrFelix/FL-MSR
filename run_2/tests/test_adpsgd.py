"""Comprehensive tests for A-DPSGD (Asynchronous Decentralized Parallel SGD)."""

import asyncio

import numpy as np
import pytest

from src.algorithms.adpsgd import ADPSGD
from src.algorithms.base import TrainingUpdate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_params(val: float = 1.0) -> dict[str, np.ndarray]:
    """Create a small set of model parameters for testing."""
    return {
        "layer_0/kernel": np.full((4, 3), val, dtype=np.float32),
        "layer_0/bias": np.full((3,), val, dtype=np.float32),
    }


def _make_update(
    source: str, round_num: int, val: float = 2.0, num_samples: int = 100
) -> TrainingUpdate:
    return TrainingUpdate(
        source_node=source,
        round_num=round_num,
        parameters=_make_params(val),
        num_samples=num_samples,
    )


def _make_adpsgd(
    node_id: str = "node-0",
    neighbors: list[str] | None = None,
    staleness_threshold: int = 0,
) -> ADPSGD:
    if neighbors is None:
        neighbors = ["node-1", "node-2"]
    return ADPSGD(
        node_id=node_id,
        neighbors=neighbors,
        config={"staleness_threshold": staleness_threshold},
    )


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

class TestADPSGDProperties:

    def test_is_asynchronous(self):
        algo = _make_adpsgd()
        assert algo.is_synchronous is False

    def test_is_decentralized(self):
        algo = _make_adpsgd()
        assert algo.is_centralized is False

    def test_initial_round_is_zero(self):
        algo = _make_adpsgd()
        assert algo.get_round() == 0

    def test_advance_round(self):
        algo = _make_adpsgd()
        algo.advance_round()
        assert algo.get_round() == 1
        algo.advance_round()
        assert algo.get_round() == 2


# ---------------------------------------------------------------------------
# on_local_training_complete
# ---------------------------------------------------------------------------

class TestADPSGDSend:

    @pytest.mark.asyncio
    async def test_sends_to_all_neighbors(self):
        algo = _make_adpsgd(neighbors=["node-1", "node-2", "node-3"])
        params = _make_params()
        outgoing = await algo.on_local_training_complete(params, round_num=0, num_samples=100)
        dest_ids = [dest for dest, _ in outgoing]
        assert sorted(dest_ids) == ["node-1", "node-2", "node-3"]

    @pytest.mark.asyncio
    async def test_sends_correct_update_content(self):
        algo = _make_adpsgd()
        params = _make_params(val=3.14)
        outgoing = await algo.on_local_training_complete(params, round_num=5, num_samples=200)
        for _, update in outgoing:
            assert update.source_node == "node-0"
            assert update.round_num == 5
            assert update.num_samples == 200
            np.testing.assert_array_equal(
                update.parameters["layer_0/kernel"],
                np.full((4, 3), 3.14, dtype=np.float32),
            )

    @pytest.mark.asyncio
    async def test_single_neighbor(self):
        """Edge case: only one neighbor."""
        algo = _make_adpsgd(neighbors=["node-1"])
        params = _make_params()
        outgoing = await algo.on_local_training_complete(params, round_num=0, num_samples=50)
        assert len(outgoing) == 1
        assert outgoing[0][0] == "node-1"


# ---------------------------------------------------------------------------
# on_update_received
# ---------------------------------------------------------------------------

class TestADPSGDReceive:

    @pytest.mark.asyncio
    async def test_accepts_neighbor_update(self):
        algo = _make_adpsgd()
        update = _make_update("node-1", round_num=0)
        await algo.on_update_received(update)
        assert algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_rejects_non_neighbor(self):
        algo = _make_adpsgd(neighbors=["node-1"])
        update = _make_update("node-99", round_num=0)
        await algo.on_update_received(update)
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_accepts_any_round_without_staleness(self):
        """With staleness_threshold=0, all rounds should be accepted."""
        algo = _make_adpsgd(staleness_threshold=0)
        # Advance local round far ahead
        for _ in range(100):
            algo.advance_round()
        update = _make_update("node-1", round_num=0)
        await algo.on_update_received(update)
        assert algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_staleness_threshold_accepts_within_range(self):
        algo = _make_adpsgd(staleness_threshold=3)
        # Local iteration = 5
        for _ in range(5):
            algo.advance_round()
        # Update from iteration 3 (staleness = 2 <= 3) → accept
        update = _make_update("node-1", round_num=3)
        await algo.on_update_received(update)
        assert algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_staleness_threshold_rejects_too_old(self):
        algo = _make_adpsgd(staleness_threshold=3)
        for _ in range(10):
            algo.advance_round()
        # Update from iteration 0 (staleness = 10 > 3) → reject
        update = _make_update("node-1", round_num=0)
        await algo.on_update_received(update)
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_staleness_threshold_boundary_exact(self):
        """Staleness exactly at threshold should be accepted."""
        algo = _make_adpsgd(staleness_threshold=5)
        for _ in range(7):
            algo.advance_round()
        # staleness = |7 - 2| = 5 == threshold → accept
        update = _make_update("node-1", round_num=2)
        await algo.on_update_received(update)
        assert algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_staleness_threshold_boundary_one_over(self):
        """Staleness one beyond threshold should be rejected."""
        algo = _make_adpsgd(staleness_threshold=5)
        for _ in range(7):
            algo.advance_round()
        # staleness = |7 - 1| = 6 > 5 → reject
        update = _make_update("node-1", round_num=1)
        await algo.on_update_received(update)
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_staleness_future_update(self):
        """Update from the future (sender ahead of us). abs() handles this."""
        algo = _make_adpsgd(staleness_threshold=3)
        # Local = 0, sender = 5, staleness = 5 > 3 → reject
        update = _make_update("node-1", round_num=5)
        await algo.on_update_received(update)
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_staleness_future_within_range(self):
        """Future update within threshold should be accepted."""
        algo = _make_adpsgd(staleness_threshold=3)
        # Local = 0, sender = 2, staleness = 2 <= 3 → accept
        update = _make_update("node-1", round_num=2)
        await algo.on_update_received(update)
        assert algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_multiple_updates_buffer(self):
        """Multiple updates should all be buffered."""
        algo = _make_adpsgd(neighbors=["node-1", "node-2", "node-3"])
        await algo.on_update_received(_make_update("node-1", 0, val=1.0))
        await algo.on_update_received(_make_update("node-2", 0, val=2.0))
        await algo.on_update_received(_make_update("node-3", 0, val=3.0))
        assert len(algo._update_buffer) == 3

    @pytest.mark.asyncio
    async def test_same_neighbor_multiple_updates(self):
        """Same neighbor can send multiple updates in async mode."""
        algo = _make_adpsgd(neighbors=["node-1"])
        await algo.on_update_received(_make_update("node-1", 0, val=1.0))
        await algo.on_update_received(_make_update("node-1", 1, val=2.0))
        assert len(algo._update_buffer) == 2

    @pytest.mark.asyncio
    async def test_signals_aggregation_event(self):
        """Aggregation event should be set when an update arrives."""
        algo = _make_adpsgd()
        assert not algo._aggregation_event.is_set()
        await algo.on_update_received(_make_update("node-1", 0))
        assert algo._aggregation_event.is_set()


# ---------------------------------------------------------------------------
# ready_to_aggregate
# ---------------------------------------------------------------------------

class TestADPSGDReady:

    def test_not_ready_initially(self):
        algo = _make_adpsgd()
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_ready_after_one_update(self):
        algo = _make_adpsgd()
        await algo.on_update_received(_make_update("node-1", 0))
        assert algo.ready_to_aggregate()


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestADPSGDAggregation:

    @pytest.mark.asyncio
    async def test_single_update_5050_mix(self):
        """One neighbor update → 0.5 * local + 0.5 * neighbor."""
        algo = _make_adpsgd()
        local_params = _make_params(val=10.0)
        neighbor_update = _make_update("node-1", round_num=0, val=20.0)
        await algo.on_update_received(neighbor_update)

        result = await algo.aggregate(local_params)

        # Expected: 0.5 * 10 + 0.5 * 20 = 15
        for layer in result:
            np.testing.assert_allclose(result[layer], 15.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_multiple_updates_averaged_then_mixed(self):
        """Two neighbor updates → avg neighbors first, then 50/50 with local."""
        algo = _make_adpsgd(neighbors=["node-1", "node-2"])
        local_params = _make_params(val=10.0)
        await algo.on_update_received(_make_update("node-1", 0, val=20.0))
        await algo.on_update_received(_make_update("node-2", 0, val=30.0))

        result = await algo.aggregate(local_params)

        # neighbor_avg = (20 + 30) / 2 = 25
        # result = 0.5 * 10 + 0.5 * 25 = 17.5
        for layer in result:
            np.testing.assert_allclose(result[layer], 17.5, atol=1e-6)

    @pytest.mark.asyncio
    async def test_three_updates_from_same_neighbor(self):
        """Three updates from the same neighbor should all be averaged."""
        algo = _make_adpsgd(neighbors=["node-1"])
        local_params = _make_params(val=0.0)
        await algo.on_update_received(_make_update("node-1", 0, val=6.0))
        await algo.on_update_received(_make_update("node-1", 1, val=12.0))
        await algo.on_update_received(_make_update("node-1", 2, val=18.0))

        result = await algo.aggregate(local_params)

        # neighbor_avg = (6 + 12 + 18) / 3 = 12
        # result = 0.5 * 0 + 0.5 * 12 = 6
        for layer in result:
            np.testing.assert_allclose(result[layer], 6.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_aggregate_clears_buffer(self):
        """Buffer should be empty after aggregation."""
        algo = _make_adpsgd()
        await algo.on_update_received(_make_update("node-1", 0))
        local_params = _make_params()
        await algo.aggregate(local_params)
        assert len(algo._update_buffer) == 0
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_event_cleared_by_advance_round(self):
        """Aggregation event is cleared by advance_round(), not aggregate().

        aggregate() only clears the buffer; advance_round() owns the event
        lifecycle.  This avoids a race where an update arriving between
        aggregate() and advance_round() would have its event signal lost.
        """
        algo = _make_adpsgd()
        await algo.on_update_received(_make_update("node-1", 0))
        assert algo._aggregation_event.is_set()
        await algo.aggregate(_make_params())
        # Event is still set — aggregate does NOT clear it
        assert algo._aggregation_event.is_set()
        # advance_round clears it
        algo.advance_round()
        assert not algo._aggregation_event.is_set()

    @pytest.mark.asyncio
    async def test_aggregate_empty_buffer_returns_local(self):
        """Aggregating with no buffered updates returns local params unchanged."""
        algo = _make_adpsgd()
        local_params = _make_params(val=42.0)
        result = await algo.aggregate(local_params)
        for layer in result:
            np.testing.assert_array_equal(result[layer], local_params[layer])

    @pytest.mark.asyncio
    async def test_aggregate_preserves_layer_shapes(self):
        """Aggregation should not change parameter shapes."""
        algo = _make_adpsgd()
        local_params = _make_params(val=1.0)
        await algo.on_update_received(_make_update("node-1", 0, val=2.0))
        result = await algo.aggregate(local_params)
        for layer_name in local_params:
            assert result[layer_name].shape == local_params[layer_name].shape
            assert result[layer_name].dtype == local_params[layer_name].dtype

    @pytest.mark.asyncio
    async def test_aggregate_many_layers(self):
        """Test with a model that has many layers."""
        algo = _make_adpsgd(neighbors=["node-1"])
        local_params = {
            f"layer_{i}/kernel": np.full((10, 10), float(i), dtype=np.float32)
            for i in range(20)
        }
        neighbor_params = {
            f"layer_{i}/kernel": np.full((10, 10), float(i + 10), dtype=np.float32)
            for i in range(20)
        }
        update = TrainingUpdate(
            source_node="node-1", round_num=0,
            parameters=neighbor_params, num_samples=100,
        )
        await algo.on_update_received(update)
        result = await algo.aggregate(local_params)

        for i in range(20):
            name = f"layer_{i}/kernel"
            expected = 0.5 * float(i) + 0.5 * float(i + 10)
            np.testing.assert_allclose(result[name], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Staleness threshold = 1 (very strict)
# ---------------------------------------------------------------------------

class TestADPSGDStrictStaleness:

    @pytest.mark.asyncio
    async def test_staleness_1_accepts_same_round(self):
        algo = _make_adpsgd(staleness_threshold=1)
        for _ in range(5):
            algo.advance_round()
        await algo.on_update_received(_make_update("node-1", round_num=5))
        assert algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_staleness_1_accepts_one_behind(self):
        algo = _make_adpsgd(staleness_threshold=1)
        for _ in range(5):
            algo.advance_round()
        await algo.on_update_received(_make_update("node-1", round_num=4))
        assert algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_staleness_1_rejects_two_behind(self):
        algo = _make_adpsgd(staleness_threshold=1)
        for _ in range(5):
            algo.advance_round()
        await algo.on_update_received(_make_update("node-1", round_num=3))
        assert not algo.ready_to_aggregate()


# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------

class TestADPSGDDefaults:

    def test_default_staleness_is_zero(self):
        algo = ADPSGD(node_id="n", neighbors=["n2"], config={})
        assert algo._staleness_threshold == 0

    @pytest.mark.asyncio
    async def test_default_staleness_accepts_all(self):
        """With default config (no staleness_threshold key), all updates accepted."""
        algo = ADPSGD(node_id="n", neighbors=["n2"], config={})
        for _ in range(1000):
            algo.advance_round()
        await algo.on_update_received(_make_update("n2", round_num=0))
        assert algo.ready_to_aggregate()
