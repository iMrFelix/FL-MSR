"""Edge case and pathological tests for Stage 5 async algorithms.

Tests for bugs found during code audit:
- GossipSGD with empty neighbors (was IndexError)
- GossipSGD seed collision across nodes (was all same pattern)
- Defensive buffer copy in aggregate (race condition safety)
- NaN/Inf propagation through aggregation
- Buffer behavior after aggregate (clear + new update)
"""

import asyncio
import math

import numpy as np
import pytest

from src.algorithms.adpsgd import ADPSGD
from src.algorithms.base import TrainingUpdate
from src.algorithms.gossip_sgd import GossipSGD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(val: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "layer_0/kernel": np.full((4, 3), val, dtype=np.float32),
        "layer_0/bias": np.full((3,), val, dtype=np.float32),
    }


def _make_update(source: str, round_num: int, val: float = 2.0) -> TrainingUpdate:
    return TrainingUpdate(
        source_node=source,
        round_num=round_num,
        parameters=_make_params(val),
        num_samples=100,
    )


# ---------------------------------------------------------------------------
# GossipSGD: empty neighbors
# ---------------------------------------------------------------------------

class TestGossipEmptyNeighbors:

    @pytest.mark.asyncio
    async def test_empty_neighbors_returns_empty_list(self):
        """GossipSGD.on_local_training_complete should return [] with no neighbors."""
        algo = GossipSGD(
            node_id="node-0",
            neighbors=[],
            config={"seed": 42},
        )
        outgoing = await algo.on_local_training_complete(
            _make_params(), round_num=0, num_samples=100,
        )
        assert outgoing == []

    def test_empty_neighbors_ready_to_aggregate_false(self):
        """No neighbors means no updates, so never ready."""
        algo = GossipSGD(node_id="node-0", neighbors=[], config={})
        assert not algo.ready_to_aggregate()


class TestADPSGDEmptyNeighbors:

    @pytest.mark.asyncio
    async def test_empty_neighbors_returns_empty_list(self):
        algo = ADPSGD(node_id="node-0", neighbors=[], config={})
        outgoing = await algo.on_local_training_complete(
            _make_params(), round_num=0, num_samples=100,
        )
        assert outgoing == []


# ---------------------------------------------------------------------------
# GossipSGD: seed collision fix (different nodes get different patterns)
# ---------------------------------------------------------------------------

class TestGossipSeedPerNode:

    @pytest.mark.asyncio
    async def test_different_nodes_different_gossip_patterns(self):
        """Two nodes with same config seed but different IDs should gossip differently."""
        neighbors = ["node-1", "node-2", "node-3", "node-4"]
        algo_a = GossipSGD("node-0", neighbors, {"seed": 42})
        algo_b = GossipSGD("node-5", neighbors, {"seed": 42})

        selections_a = []
        selections_b = []
        for i in range(50):
            out_a = await algo_a.on_local_training_complete(
                _make_params(), round_num=i, num_samples=100,
            )
            out_b = await algo_b.on_local_training_complete(
                _make_params(), round_num=i, num_samples=100,
            )
            selections_a.append(out_a[0][0])
            selections_b.append(out_b[0][0])

        # Different node_ids should produce different selection sequences
        assert selections_a != selections_b, (
            "Nodes with different IDs but same seed should NOT have identical "
            "gossip patterns (seed collision bug)"
        )

    @pytest.mark.asyncio
    async def test_same_node_same_seed_still_reproducible(self):
        """Same node_id + same seed should still produce identical sequences."""
        neighbors = ["node-1", "node-2", "node-3"]
        algo_a = GossipSGD("node-0", neighbors, {"seed": 42})
        algo_b = GossipSGD("node-0", neighbors, {"seed": 42})

        for i in range(30):
            out_a = await algo_a.on_local_training_complete(
                _make_params(), round_num=i, num_samples=100,
            )
            out_b = await algo_b.on_local_training_complete(
                _make_params(), round_num=i, num_samples=100,
            )
            assert out_a[0][0] == out_b[0][0], f"Mismatch at iteration {i}"


# ---------------------------------------------------------------------------
# Defensive buffer copy (aggregate should work even if buffer mutated)
# ---------------------------------------------------------------------------

class TestBufferDefensiveCopy:

    @pytest.mark.asyncio
    async def test_aggregate_works_after_buffer_snapshot(self):
        """aggregate() should work correctly with the snapshot pattern."""
        algo = ADPSGD("node-0", ["node-1", "node-2"], {})
        await algo.on_update_received(_make_update("node-1", 0, val=10.0))
        await algo.on_update_received(_make_update("node-2", 0, val=20.0))

        local = _make_params(val=0.0)
        result = await algo.aggregate(local)

        # neighbor_avg = (10 + 20) / 2 = 15
        # result = 0.5 * 0 + 0.5 * 15 = 7.5
        for layer in result:
            np.testing.assert_allclose(result[layer], 7.5, atol=1e-6)

        # Buffer should be cleared
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_new_update_after_aggregate_goes_into_fresh_buffer(self):
        """An update arriving after aggregate should start a fresh buffer cycle."""
        algo = ADPSGD("node-0", ["node-1"], {})
        await algo.on_update_received(_make_update("node-1", 0, val=10.0))
        await algo.aggregate(_make_params(val=0.0))

        # Buffer is now clear
        assert not algo.ready_to_aggregate()

        # New update arrives
        await algo.on_update_received(_make_update("node-1", 1, val=30.0))
        assert algo.ready_to_aggregate()

        # Second aggregation uses only the new update
        result = await algo.aggregate(_make_params(val=0.0))
        for layer in result:
            np.testing.assert_allclose(result[layer], 15.0, atol=1e-6)  # 0.5*0 + 0.5*30


# ---------------------------------------------------------------------------
# NaN/Inf propagation
# ---------------------------------------------------------------------------

class TestNaNInfPropagation:
    """Document (not fix) the NaN/Inf behavior — these tests verify the
    CURRENT behavior.  A future improvement would add NaN filtering."""

    @pytest.mark.asyncio
    async def test_nan_update_propagates_to_aggregation(self):
        """NaN in a neighbor update currently propagates through aggregation."""
        algo = ADPSGD("node-0", ["node-1"], {})
        nan_params = {
            "layer_0/kernel": np.full((4, 3), float("nan"), dtype=np.float32),
            "layer_0/bias": np.full((3,), float("nan"), dtype=np.float32),
        }
        update = TrainingUpdate(
            source_node="node-1", round_num=0,
            parameters=nan_params, num_samples=100,
        )
        await algo.on_update_received(update)
        result = await algo.aggregate(_make_params(val=1.0))

        # 0.5 * 1.0 + 0.5 * NaN = NaN
        for layer in result:
            assert np.isnan(result[layer]).all(), (
                "NaN from neighbor should propagate through aggregation"
            )

    @pytest.mark.asyncio
    async def test_inf_update_propagates_to_aggregation(self):
        """Inf in a neighbor update currently propagates through aggregation."""
        algo = ADPSGD("node-0", ["node-1"], {})
        inf_params = {
            "layer_0/kernel": np.full((4, 3), float("inf"), dtype=np.float32),
            "layer_0/bias": np.full((3,), float("inf"), dtype=np.float32),
        }
        update = TrainingUpdate(
            source_node="node-1", round_num=0,
            parameters=inf_params, num_samples=100,
        )
        await algo.on_update_received(update)
        result = await algo.aggregate(_make_params(val=1.0))

        # 0.5 * 1.0 + 0.5 * inf = inf
        for layer in result:
            assert np.isinf(result[layer]).all()


# ---------------------------------------------------------------------------
# Aggregation with very large/small values
# ---------------------------------------------------------------------------

class TestNumericalEdgeCases:

    @pytest.mark.asyncio
    async def test_very_large_values_no_overflow(self):
        """Aggregation with large float32 values should not overflow."""
        algo = ADPSGD("node-0", ["node-1"], {})
        large_val = 1e30  # well within float32 range (~3.4e38)
        await algo.on_update_received(_make_update("node-1", 0, val=large_val))
        result = await algo.aggregate(_make_params(val=large_val))
        for layer in result:
            # 0.5 * 1e30 + 0.5 * 1e30 = 1e30
            np.testing.assert_allclose(result[layer], large_val, rtol=1e-5)

    @pytest.mark.asyncio
    async def test_very_small_values_precision(self):
        """Aggregation with very small values should maintain precision."""
        algo = ADPSGD("node-0", ["node-1"], {})
        small_val = 1e-30
        await algo.on_update_received(_make_update("node-1", 0, val=small_val))
        result = await algo.aggregate(_make_params(val=small_val))
        for layer in result:
            np.testing.assert_allclose(result[layer], small_val, rtol=1e-5)

    @pytest.mark.asyncio
    async def test_zero_local_params(self):
        algo = ADPSGD("node-0", ["node-1"], {})
        await algo.on_update_received(_make_update("node-1", 0, val=10.0))
        result = await algo.aggregate(_make_params(val=0.0))
        for layer in result:
            np.testing.assert_allclose(result[layer], 5.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_zero_neighbor_params(self):
        algo = ADPSGD("node-0", ["node-1"], {})
        await algo.on_update_received(_make_update("node-1", 0, val=0.0))
        result = await algo.aggregate(_make_params(val=10.0))
        for layer in result:
            np.testing.assert_allclose(result[layer], 5.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_negative_values(self):
        algo = ADPSGD("node-0", ["node-1"], {})
        await algo.on_update_received(_make_update("node-1", 0, val=-10.0))
        result = await algo.aggregate(_make_params(val=10.0))
        for layer in result:
            np.testing.assert_allclose(result[layer], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Rapid sequential aggregations
# ---------------------------------------------------------------------------

class TestRapidAggregation:

    @pytest.mark.asyncio
    async def test_multiple_aggregate_cycles(self):
        """Multiple aggregate cycles should work correctly."""
        algo = ADPSGD("node-0", ["node-1"], {})

        for cycle in range(5):
            val = float(cycle * 10)
            await algo.on_update_received(_make_update("node-1", cycle, val=val))
            result = await algo.aggregate(_make_params(val=0.0))
            for layer in result:
                np.testing.assert_allclose(
                    result[layer], val * 0.5, atol=1e-5,
                    err_msg=f"Failed at cycle {cycle}",
                )
            algo.advance_round()

    @pytest.mark.asyncio
    async def test_aggregate_then_immediately_receive_and_aggregate(self):
        """Back-to-back aggregate cycles with no gap."""
        algo = GossipSGD("node-0", ["node-1"], {"seed": 42})

        # Cycle 1
        await algo.on_update_received(_make_update("node-1", 0, val=4.0))
        r1 = await algo.aggregate(_make_params(val=0.0))
        for layer in r1:
            np.testing.assert_allclose(r1[layer], 2.0, atol=1e-6)

        # Cycle 2 (immediately after)
        await algo.on_update_received(_make_update("node-1", 1, val=8.0))
        r2 = await algo.aggregate(_make_params(val=0.0))
        for layer in r2:
            np.testing.assert_allclose(r2[layer], 4.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Staleness edge case: round_num = 0 when local is at 0
# ---------------------------------------------------------------------------

class TestStalenessRoundZero:

    @pytest.mark.asyncio
    async def test_both_at_round_zero_with_staleness(self):
        """Both local and sender at round 0 — should always be accepted."""
        algo = ADPSGD("node-0", ["node-1"], {"staleness_threshold": 1})
        await algo.on_update_received(_make_update("node-1", round_num=0))
        assert algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_staleness_with_very_high_rounds(self):
        """High round numbers should not cause overflow issues."""
        algo = ADPSGD("node-0", ["node-1"], {"staleness_threshold": 5})
        for _ in range(1_000_000):
            algo.advance_round()
        await algo.on_update_received(
            _make_update("node-1", round_num=999_998)
        )
        assert algo.ready_to_aggregate()  # staleness = 2 <= 5
