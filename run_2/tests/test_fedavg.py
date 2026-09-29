"""Tests for FedAvg centralized synchronous algorithm.

Covers both aggregator and worker roles:
- Properties (is_synchronous, is_centralized, role)
- Worker sends to neighbors (aggregator)
- Aggregator sends to neighbors (workers)
- Worker aggregate returns aggregator's model (no mixing)
- Aggregator aggregate does weighted average
- Ready-to-aggregate conditions for both roles
- Non-neighbor and wrong-round rejection
- Multiple rounds
- Edge cases (unequal num_samples, single worker, zero samples)
"""

import asyncio
import math

import numpy as np
import pytest

from src.algorithms.base import TrainingUpdate
from src.algorithms.fedavg import FedAvg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(val: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "layer_0/kernel": np.full((4, 3), val, dtype=np.float32),
        "layer_0/bias": np.full((3,), val, dtype=np.float32),
    }


def _make_update(
    source: str, round_num: int, val: float = 2.0, num_samples: int = 100,
) -> TrainingUpdate:
    return TrainingUpdate(
        source_node=source,
        round_num=round_num,
        parameters=_make_params(val),
        num_samples=num_samples,
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestFedAvgProperties:

    def test_worker_is_synchronous(self):
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        assert algo.is_synchronous is True

    def test_worker_is_centralized(self):
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        assert algo.is_centralized is True

    def test_aggregator_is_synchronous(self):
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        assert algo.is_synchronous is True

    def test_aggregator_is_centralized(self):
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        assert algo.is_centralized is True

    def test_worker_role(self):
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        assert algo.role == "worker"

    def test_aggregator_role(self):
        algo = FedAvg("node-0", ["node-1"], {"role": "aggregator"})
        assert algo.role == "aggregator"

    def test_default_role_is_worker(self):
        algo = FedAvg("node-1", ["node-0"], {})
        assert algo.role == "worker"


# ---------------------------------------------------------------------------
# Worker: send behavior
# ---------------------------------------------------------------------------

class TestWorkerSend:

    @pytest.mark.asyncio
    async def test_worker_sends_to_all_neighbors(self):
        """Worker should send to all neighbors (= the aggregator in star)."""
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        outgoing = await algo.on_local_training_complete(
            _make_params(), round_num=0, num_samples=100,
        )
        assert len(outgoing) == 1
        assert outgoing[0][0] == "node-0"

    @pytest.mark.asyncio
    async def test_worker_sends_correct_params(self):
        val = 5.0
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        outgoing = await algo.on_local_training_complete(
            _make_params(val), round_num=3, num_samples=200,
        )
        update = outgoing[0][1]
        assert update.source_node == "node-1"
        assert update.round_num == 3
        assert update.num_samples == 200
        np.testing.assert_array_equal(
            update.parameters["layer_0/kernel"],
            np.full((4, 3), val, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Aggregator: send behavior
# ---------------------------------------------------------------------------

class TestAggregatorSend:

    @pytest.mark.asyncio
    async def test_aggregator_sends_to_all_workers(self):
        """Aggregator broadcasts to all neighbors (= workers)."""
        algo = FedAvg("node-0", ["node-1", "node-2", "node-3"], {"role": "aggregator"})
        outgoing = await algo.on_local_training_complete(
            _make_params(val=10.0), round_num=0, num_samples=300,
        )
        assert len(outgoing) == 3
        dests = {dest for dest, _ in outgoing}
        assert dests == {"node-1", "node-2", "node-3"}


# ---------------------------------------------------------------------------
# Worker: receive and aggregate
# ---------------------------------------------------------------------------

class TestWorkerReceiveAggregate:

    @pytest.mark.asyncio
    async def test_worker_not_ready_initially(self):
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_worker_ready_after_aggregator_response(self):
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        await algo.on_update_received(_make_update("node-0", 0, val=5.0))
        assert algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_worker_aggregate_returns_aggregator_model(self):
        """Worker aggregate should return the aggregator's params, not a mix."""
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        await algo.on_update_received(_make_update("node-0", 0, val=10.0))

        # Worker's local params are val=1.0, but aggregate should ignore them
        result = await algo.aggregate(_make_params(val=1.0))
        for layer in result:
            np.testing.assert_allclose(result[layer], 10.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_worker_rejects_non_neighbor(self):
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        await algo.on_update_received(_make_update("node-99", 0))
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_worker_rejects_wrong_round(self):
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        await algo.on_update_received(_make_update("node-0", 1))  # round 1, expected 0
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_worker_buffer_cleared_after_aggregate(self):
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        await algo.on_update_received(_make_update("node-0", 0, val=5.0))
        await algo.aggregate(_make_params())
        assert not algo.ready_to_aggregate()


# ---------------------------------------------------------------------------
# Aggregator: receive and aggregate
# ---------------------------------------------------------------------------

class TestAggregatorReceiveAggregate:

    @pytest.mark.asyncio
    async def test_aggregator_not_ready_initially(self):
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_aggregator_not_ready_with_partial_updates(self):
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 0))
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_aggregator_ready_when_all_workers_reported(self):
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 0))
        await algo.on_update_received(_make_update("node-2", 0))
        assert algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_aggregator_uniform_average(self):
        """Equal samples → uniform average."""
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 0, val=10.0, num_samples=100))
        await algo.on_update_received(_make_update("node-2", 0, val=20.0, num_samples=100))

        result = await algo.aggregate(_make_params(val=0.0))
        # (10 * 100 + 20 * 100) / 200 = 15
        for layer in result:
            np.testing.assert_allclose(result[layer], 15.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_aggregator_weighted_average(self):
        """Unequal samples → weighted average."""
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 0, val=10.0, num_samples=300))
        await algo.on_update_received(_make_update("node-2", 0, val=20.0, num_samples=100))

        result = await algo.aggregate(_make_params(val=0.0))
        # (10 * 300 + 20 * 100) / 400 = 12.5
        for layer in result:
            np.testing.assert_allclose(result[layer], 12.5, atol=1e-6)

    @pytest.mark.asyncio
    async def test_aggregator_ignores_local_params(self):
        """Aggregator should NOT mix in its own local params."""
        algo = FedAvg("node-0", ["node-1"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 0, val=10.0, num_samples=100))

        # Passing local_params=999 should be completely ignored
        result = await algo.aggregate(_make_params(val=999.0))
        for layer in result:
            np.testing.assert_allclose(result[layer], 10.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_aggregator_rejects_non_neighbor(self):
        algo = FedAvg("node-0", ["node-1"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-99", 0))
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_aggregator_rejects_wrong_round(self):
        algo = FedAvg("node-0", ["node-1"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 5))  # round 5, expected 0
        assert not algo.ready_to_aggregate()

    @pytest.mark.asyncio
    async def test_aggregator_buffer_cleared_after_aggregate(self):
        algo = FedAvg("node-0", ["node-1"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 0))
        await algo.aggregate(_make_params())
        assert not algo.ready_to_aggregate()


# ---------------------------------------------------------------------------
# Partial-update aggregation (ε-deadline interaction; extension 02)
# ---------------------------------------------------------------------------

class TestAggregatorPartialUpdates:
    """Verify the aggregator handles ε-deadline partial updates correctly.

    Under ε-deadline (docs/extensions/02-epsilon-trigger.md) a worker may
    submit an update missing some layers.  The aggregator must:
    - average the layer over only the contributors, weighted by samples;
    - treat each missing contribution as if the worker agreed with
      local_params (the aggregator's current value);
    - keep local_params unchanged for layers no worker contributed.
    """

    def _partial_update(
        self, source: str, round_num: int, *,
        present_layers: dict[str, float], num_samples: int,
    ) -> TrainingUpdate:
        params = {
            name: np.full(shape, val, dtype=np.float32)
            for name, (shape, val) in present_layers.items()
        }
        return TrainingUpdate(
            source_node=source, round_num=round_num,
            parameters=params, num_samples=num_samples,
        )

    @pytest.mark.asyncio
    async def test_missing_layer_falls_back_to_local_params(self):
        """One of two workers omits a layer; the omitted contribution uses
        local_params, preserving total weight = 1."""
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        # Worker 1 sends both layers with value 10.0
        await algo.on_update_received(self._partial_update(
            "node-1", 0,
            present_layers={
                "layer_0/kernel": ((4, 3), 10.0),
                "layer_0/bias": ((3,), 10.0),
            },
            num_samples=100,
        ))
        # Worker 2 omits the bias entirely, sends only kernel with value 20.0
        await algo.on_update_received(self._partial_update(
            "node-2", 0,
            present_layers={"layer_0/kernel": ((4, 3), 20.0)},
            num_samples=100,
        ))

        local_params = _make_params(val=5.0)  # current aggregator value
        result = await algo.aggregate(local_params)

        # kernel: 0.5*10 + 0.5*20 = 15
        np.testing.assert_allclose(result["layer_0/kernel"], 15.0, atol=1e-6)
        # bias: worker-1 contributed 10 (weight 0.5), worker-2 missing →
        # treated as agreeing with local_params=5 (weight 0.5) → 7.5
        np.testing.assert_allclose(result["layer_0/bias"], 7.5, atol=1e-6)

    @pytest.mark.asyncio
    async def test_layer_no_worker_contributes_keeps_local_params(self):
        """If every worker omits a layer, the aggregator keeps its current
        value for that layer (no change)."""
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        # Both workers omit the bias
        await algo.on_update_received(self._partial_update(
            "node-1", 0,
            present_layers={"layer_0/kernel": ((4, 3), 10.0)},
            num_samples=100,
        ))
        await algo.on_update_received(self._partial_update(
            "node-2", 0,
            present_layers={"layer_0/kernel": ((4, 3), 20.0)},
            num_samples=100,
        ))

        local_params = _make_params(val=5.0)
        result = await algo.aggregate(local_params)

        # kernel averaged normally
        np.testing.assert_allclose(result["layer_0/kernel"], 15.0, atol=1e-6)
        # bias unchanged from local_params
        np.testing.assert_allclose(result["layer_0/bias"], 5.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_partial_update_with_weighted_samples(self):
        """Weighting still uses total samples; missing layers get the
        aggregator's current value at the worker's weight."""
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        await algo.on_update_received(self._partial_update(
            "node-1", 0,
            present_layers={
                "layer_0/kernel": ((4, 3), 10.0),
                "layer_0/bias": ((3,), 10.0),
            },
            num_samples=300,
        ))
        await algo.on_update_received(self._partial_update(
            "node-2", 0,
            present_layers={"layer_0/kernel": ((4, 3), 20.0)},
            num_samples=100,
        ))

        local_params = _make_params(val=5.0)
        result = await algo.aggregate(local_params)

        # kernel: (10*300 + 20*100)/400 = 12.5
        np.testing.assert_allclose(result["layer_0/kernel"], 12.5, atol=1e-6)
        # bias: (10*300 + 5*100)/400 = 8.75  (worker-2 missing → falls back to 5)
        np.testing.assert_allclose(result["layer_0/bias"], 8.75, atol=1e-6)

    @pytest.mark.asyncio
    async def test_all_complete_updates_unchanged_from_legacy(self):
        """With no missing layers, behaviour matches the pre-extension
        weighted average exactly."""
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 0, val=10.0, num_samples=100))
        await algo.on_update_received(_make_update("node-2", 0, val=20.0, num_samples=100))

        result = await algo.aggregate(_make_params(val=999.0))
        # local_params=999 must not leak in because every layer was contributed
        for layer in result:
            np.testing.assert_allclose(result[layer], 15.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Multi-round tests
# ---------------------------------------------------------------------------

class TestMultipleRounds:

    @pytest.mark.asyncio
    async def test_aggregator_multiple_rounds(self):
        """Aggregator should work across multiple rounds."""
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})

        for rnd in range(5):
            val1 = float(rnd * 10)
            val2 = float(rnd * 20)
            await algo.on_update_received(
                _make_update("node-1", rnd, val=val1, num_samples=100)
            )
            await algo.on_update_received(
                _make_update("node-2", rnd, val=val2, num_samples=100)
            )
            assert algo.ready_to_aggregate()

            result = await algo.aggregate(_make_params())
            expected = (val1 + val2) / 2  # equal samples → simple avg
            for layer in result:
                np.testing.assert_allclose(
                    result[layer], expected, atol=1e-5,
                    err_msg=f"Round {rnd}",
                )
            algo.advance_round()

    @pytest.mark.asyncio
    async def test_worker_multiple_rounds(self):
        """Worker should work across multiple rounds."""
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})

        for rnd in range(5):
            val = float(rnd * 10 + 1)
            await algo.on_update_received(
                _make_update("node-0", rnd, val=val)
            )
            result = await algo.aggregate(_make_params(val=0.0))
            for layer in result:
                np.testing.assert_allclose(result[layer], val, atol=1e-6)
            algo.advance_round()

    @pytest.mark.asyncio
    async def test_full_fedavg_round(self):
        """Simulate a complete FedAvg round: workers send → aggregator aggregates → workers receive."""
        aggregator = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        worker1 = FedAvg("node-1", ["node-0"], {"role": "worker"})
        worker2 = FedAvg("node-2", ["node-0"], {"role": "worker"})

        # Workers train and send
        w1_out = await worker1.on_local_training_complete(
            _make_params(val=10.0), round_num=0, num_samples=100,
        )
        w2_out = await worker2.on_local_training_complete(
            _make_params(val=20.0), round_num=0, num_samples=300,
        )

        # Aggregator receives worker updates
        await aggregator.on_update_received(w1_out[0][1])
        await aggregator.on_update_received(w2_out[0][1])
        assert aggregator.ready_to_aggregate()

        # Aggregator computes weighted average
        agg_result = await aggregator.aggregate(_make_params())
        # (10*100 + 20*300) / 400 = 17.5
        for layer in agg_result:
            np.testing.assert_allclose(agg_result[layer], 17.5, atol=1e-5)

        # Aggregator broadcasts back
        agg_out = await aggregator.on_local_training_complete(
            agg_result, round_num=0, num_samples=400,
        )
        assert len(agg_out) == 2

        # Workers receive and adopt global model
        for dest, update in agg_out:
            if dest == "node-1":
                await worker1.on_update_received(update)
            elif dest == "node-2":
                await worker2.on_update_received(update)

        w1_result = await worker1.aggregate(_make_params(val=10.0))
        w2_result = await worker2.aggregate(_make_params(val=20.0))

        # Both workers should have the same global model (17.5)
        for layer in w1_result:
            np.testing.assert_allclose(w1_result[layer], 17.5, atol=1e-5)
            np.testing.assert_allclose(w2_result[layer], 17.5, atol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_single_worker(self):
        """Aggregator with one worker should just return that worker's model."""
        algo = FedAvg("node-0", ["node-1"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 0, val=42.0))
        result = await algo.aggregate(_make_params())
        for layer in result:
            np.testing.assert_allclose(result[layer], 42.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_three_workers_weighted(self):
        """Three workers with different samples and values."""
        algo = FedAvg(
            "node-0",
            ["node-1", "node-2", "node-3"],
            {"role": "aggregator"},
        )
        # Worker 1: val=10, 100 samples
        # Worker 2: val=20, 200 samples
        # Worker 3: val=30, 300 samples
        await algo.on_update_received(_make_update("node-1", 0, val=10.0, num_samples=100))
        await algo.on_update_received(_make_update("node-2", 0, val=20.0, num_samples=200))
        await algo.on_update_received(_make_update("node-3", 0, val=30.0, num_samples=300))

        result = await algo.aggregate(_make_params())
        # (10*100 + 20*200 + 30*300) / 600 = (1000 + 4000 + 9000) / 600 = 23.333...
        expected = (10 * 100 + 20 * 200 + 30 * 300) / 600
        for layer in result:
            np.testing.assert_allclose(result[layer], expected, atol=1e-5)

    @pytest.mark.asyncio
    async def test_zero_samples_fallback(self):
        """If all workers report 0 samples, fall back to uniform average."""
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 0, val=10.0, num_samples=0))
        await algo.on_update_received(_make_update("node-2", 0, val=20.0, num_samples=0))
        result = await algo.aggregate(_make_params())
        for layer in result:
            np.testing.assert_allclose(result[layer], 15.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_aggregator_aggregate_raises_on_empty_buffer(self):
        """Calling aggregate on empty buffer should raise."""
        algo = FedAvg("node-0", ["node-1"], {"role": "aggregator"})
        with pytest.raises(RuntimeError, match="empty worker buffer"):
            await algo.aggregate(_make_params())

    @pytest.mark.asyncio
    async def test_worker_aggregate_raises_without_model(self):
        """Calling aggregate before receiving model should raise."""
        algo = FedAvg("node-1", ["node-0"], {"role": "worker"})
        with pytest.raises(RuntimeError, match="no aggregated model"):
            await algo.aggregate(_make_params())

    @pytest.mark.asyncio
    async def test_duplicate_worker_update_overwrites(self):
        """If a worker sends twice, last update wins."""
        algo = FedAvg("node-0", ["node-1"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 0, val=10.0))
        await algo.on_update_received(_make_update("node-1", 0, val=20.0))
        assert algo.ready_to_aggregate()
        result = await algo.aggregate(_make_params())
        for layer in result:
            np.testing.assert_allclose(result[layer], 20.0, atol=1e-6)

    @pytest.mark.asyncio
    async def test_negative_values(self):
        """Weighted average should work with negative values."""
        algo = FedAvg("node-0", ["node-1", "node-2"], {"role": "aggregator"})
        await algo.on_update_received(_make_update("node-1", 0, val=-10.0, num_samples=100))
        await algo.on_update_received(_make_update("node-2", 0, val=10.0, num_samples=100))
        result = await algo.aggregate(_make_params())
        for layer in result:
            np.testing.assert_allclose(result[layer], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Config propagation tests
# ---------------------------------------------------------------------------

class TestConfigPropagation:

    def test_role_propagated_to_training_config(self):
        """Verify generator propagates role into training sub-dict."""
        from src.config.schema import load_config
        from src.config.generator import generate_node_configs

        config = load_config("configs/examples/fedavg_star_4nodes.yaml")
        node_configs = generate_node_configs(config)

        # Node 0 should be aggregator
        assert node_configs["node-0"]["training"]["role"] == "aggregator"
        # Nodes 1-3 should be workers
        for i in range(1, 4):
            assert node_configs[f"node-{i}"]["training"]["role"] == "worker"

    def test_example_config_validates(self):
        """The FedAvg example config should pass schema validation."""
        from src.config.schema import load_config
        config = load_config("configs/examples/fedavg_star_4nodes.yaml")
        assert config.training.algorithm == "fedavg"
        assert config.federation.roles == {0: "aggregator"}
        assert config.federation.num_nodes == 4
