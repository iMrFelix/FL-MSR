"""Comprehensive tests for Gossip-SGD (Stochastic Gossip Decentralized SGD)."""

import asyncio
from collections import Counter

import numpy as np
import pytest

from src.algorithms.base import TrainingUpdate
from src.algorithms.gossip_sgd import GossipSGD


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_params(val: float = 1.0) -> dict[str, np.ndarray]:
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


def _make_gossip(
    node_id: str = "node-0",
    neighbors: list[str] | None = None,
    staleness_threshold: int = 0,
    seed: int = 42,
) -> GossipSGD:
    if neighbors is None:
        neighbors = ["node-1", "node-2", "node-3"]
    return GossipSGD(
        node_id=node_id,
        neighbors=neighbors,
        config={"staleness_threshold": staleness_threshold, "seed": seed},
    )


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

class TestGossipSGDProperties:

    def test_is_asynchronous(self):
        algo = _make_gossip()
        assert algo.is_synchronous is False

    def test_is_decentralized(self):
        algo = _make_gossip()
        assert algo.is_centralized is False

    def test_initial_round(self):
        algo = _make_gossip()
        assert algo.get_round() == 0


# ---------------------------------------------------------------------------
# on_local_training_complete — sends to exactly ONE neighbor
# ---------------------------------------------------------------------------

class TestGossipSGDSend:

    @pytest.mark.asyncio
    async def test_sends_to_exactly_one_neighbor(self):
        algo = _make_gossip(neighbors=["node-1", "node-2", "node-3"])
        outgoing = await algo.on_local_training_complete(
            _make_params(), round_num=0, num_samples=100,
        )
        assert len(outgoing) == 1

    @pytest.mark.asyncio
    async def test_sends_to_valid_neighbor(self):
        neighbors = ["node-1", "node-2", "node-3"]
        algo = _make_gossip(neighbors=neighbors)
        outgoing = await algo.on_local_training_complete(
            _make_params(), round_num=0, num_samples=100,
        )
        dest = outgoing[0][0]
        assert dest in neighbors

    @pytest.mark.asyncio
    async def test_sends_correct_update_content(self):
        algo = _make_gossip()
        params = _make_params(val=7.7)
        outgoing = await algo.on_local_training_complete(
            params, round_num=3, num_samples=500,
        )
        _, update = outgoing[0]
        assert update.source_node == "node-0"
        assert update.round_num == 3
        assert update.num_samples == 500
        np.testing.assert_allclose(
            update.parameters["layer_0/kernel"], 7.7, atol=1e-6,
        )

    @pytest.mark.asyncio
    async def test_single_neighbor_always_selected(self):
        """With only one neighbor, that neighbor is always chosen."""
        algo = _make_gossip(neighbors=["node-1"])
        for i in range(20):
            outgoing = await algo.on_local_training_complete(
                _make_params(), round_num=i, num_samples=50,
            )
            assert outgoing[0][0] == "node-1"


# ---------------------------------------------------------------------------
# Reproducibility — same seed produces same sequence
# ---------------------------------------------------------------------------

class TestGossipSGDReproducibility:

    @pytest.mark.asyncio
    async def test_same_seed_same_sequence(self):
        """Two instances with the same seed should pick the same neighbors."""
        neighbors = ["node-1", "node-2", "node-3", "node-4"]
        algo_a = _make_gossip(neighbors=neighbors, seed=123)
        algo_b = _make_gossip(neighbors=neighbors, seed=123)

        for i in range(50):
            out_a = await algo_a.on_local_training_complete(
                _make_params(), round_num=i, num_samples=100,
            )
            out_b = await algo_b.on_local_training_complete(
                _make_params(), round_num=i, num_samples=100,
            )
            assert out_a[0][0] == out_b[0][0], f"Mismatch at iteration {i}"

    @pytest.mark.asyncio
    async def test_different_seeds_different_sequence(self):
        """Different seeds should (almost certainly) produce different sequences."""
        neighbors = ["node-1", "node-2", "node-3", "node-4"]
        algo_a = _make_gossip(neighbors=neighbors, seed=42)
        algo_b = _make_gossip(neighbors=neighbors, seed=99)

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

        # With 4 neighbors and 50 selections, identical sequences are
        # astronomically unlikely with different seeds.
        assert selections_a != selections_b

    @pytest.mark.asyncio
    async def test_neighbor_selection_covers_all_neighbors(self):
        """Over many iterations, all neighbors should be selected at least once."""
        neighbors = ["node-1", "node-2", "node-3", "node-4"]
        algo = _make_gossip(neighbors=neighbors, seed=42)

        selected = set()
        for i in range(200):
            outgoing = await algo.on_local_training_complete(
                _make_params(), round_num=i, num_samples=100,
            )
            selected.add(outgoing[0][0])

        assert selected == set(neighbors), (
            f"Not all neighbors were selected after 200 iterations: "
            f"selected={selected}, expected={set(neighbors)}"
        )

    @pytest.mark.asyncio
    async def test_neighbor_selection_roughly_uniform(self):
        """Selection should be roughly uniform across neighbors."""
        neighbors = ["node-1", "node-2", "node-3", "node-4"]
        algo = _make_gossip(neighbors=neighbors, seed=42)

        counts: Counter = Counter()
        n_iters = 4000
        for i in range(n_iters):
            outgoing = await algo.on_local_training_complete(
                _make_params(), round_num=i, num_samples=100,
            )
            counts[outgoing[0][0]] += 1

        expected_per_neighbor = n_iters / len(neighbors)
        for neighbor, count in counts.items():
            # Allow 20% deviation from expected uniform distribution
            assert abs(count - expected_per_neighbor) < 0.2 * expected_per_neighbor, (
                f"{neighbor} selected {count} times, expected ~{expected_per_neighbor}"
            )


# ---------------------------------------------------------------------------
# on_update_received — staleness and buffering (same as ADPSGD)
# ---------------------------------------------------------------------------

class TestGossipSGDReceive:

    @pytest.mark.asyncio
    async def test_accepts_neighbor_update(self):
        algo = _make_gossip()
        await algo.on_update_received(_make_update("node-1", 0))
        assert algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_rejects_non_neighbor(self):
        algo = _make_gossip(neighbors=["node-1"])
        await algo.on_update_received(_make_update("node-99", 0))
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_staleness_filtering(self):
        algo = _make_gossip(staleness_threshold=2)
        for _ in range(10):
            algo.advance_round()
        # staleness = 10 - 0 = 10 > 2 → reject
        await algo.on_update_received(_make_update("node-1", round_num=0))
        assert not algo.ready_to_aggregate()
        # staleness = 10 - 9 = 1 <= 2 → accept
        await algo.on_update_received(_make_update("node-1", round_num=9))
        assert algo.ready_to_aggregate()


# ---------------------------------------------------------------------------
# aggregate — same 50/50 pairwise averaging as ADPSGD
# ---------------------------------------------------------------------------

class TestGossipSGDAggregation:

    @pytest.mark.asyncio
    async def test_single_update_5050_mix(self):
        algo = _make_gossip()
        local = _make_params(val=10.0)
        await algo.on_update_received(_make_update("node-1", 0, val=20.0))
        result = await algo.aggregate(local)
        for layer in result:
            np.testing.assert_allclose(result[layer], 15.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_empty_buffer_returns_local(self):
        algo = _make_gossip()
        local = _make_params(val=42.0)
        result = await algo.aggregate(local)
        for layer in result:
            np.testing.assert_array_equal(result[layer], local[layer])

    @pytest.mark.asyncio
    async def test_aggregate_clears_buffer(self):
        """aggregate() clears the buffer; advance_round() clears the event."""
        algo = _make_gossip()
        await algo.on_update_received(_make_update("node-1", 0))
        assert algo._aggregation_event.is_set()
        await algo.aggregate(_make_params())
        assert not algo.ready_to_aggregate()
        # Event is cleared by advance_round, not aggregate
        assert algo._aggregation_event.is_set()
        algo.advance_round()
        assert not algo._aggregation_event.is_set()

    @pytest.mark.asyncio
    async def test_multiple_updates_averaged(self):
        """Multiple gossip updates averaged then mixed 50/50."""
        algo = _make_gossip(neighbors=["node-1", "node-2"])
        local = _make_params(val=0.0)
        await algo.on_update_received(_make_update("node-1", 0, val=10.0))
        await algo.on_update_received(_make_update("node-2", 0, val=30.0))
        result = await algo.aggregate(local)
        # neighbor_avg = (10 + 30) / 2 = 20, result = 0.5 * 0 + 0.5 * 20 = 10
        for layer in result:
            np.testing.assert_allclose(result[layer], 10.0, atol=1e-6)


# ---------------------------------------------------------------------------
# RNG isolation — gossip neighbor selection shouldn't affect global RNG
# ---------------------------------------------------------------------------

class TestGossipSGDRNGIsolation:

    @pytest.mark.asyncio
    async def test_does_not_affect_global_random(self):
        """GossipSGD's RNG should be independent of the global random module."""
        import random
        random.seed(999)
        expected_global = [random.random() for _ in range(10)]

        # Now create a GossipSGD and use it
        random.seed(999)
        algo = _make_gossip(seed=12345)
        for i in range(100):
            await algo.on_local_training_complete(
                _make_params(), round_num=i, num_samples=100,
            )
        actual_global = [random.random() for _ in range(10)]

        assert expected_global == actual_global, (
            "GossipSGD's neighbor selection contaminated the global RNG"
        )


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

class TestGossipSGDDefaults:

    def test_default_staleness_is_zero(self):
        algo = GossipSGD(node_id="n", neighbors=["n2"], config={})
        assert algo._staleness_threshold == 0

    def test_default_seed_is_42(self):
        algo = GossipSGD(node_id="n", neighbors=["n2"], config={})
        # We can't directly check the seed, but we can verify reproducibility
        # by creating another instance with explicit seed=42
        algo2 = GossipSGD(node_id="n", neighbors=["n2"], config={"seed": 42})
        # Both should produce the same first random choice
        assert algo._rng.random() == algo2._rng.random()
