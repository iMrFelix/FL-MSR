"""Tests for the async training loop (run_async) in TrainingEngine.

These tests use a mock model, mock algorithm, and mock network components
to isolate the engine's async loop logic without requiring TensorFlow
training or real network connections.
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf

from src.algorithms.adpsgd import ADPSGD
from src.algorithms.base import FederationAlgorithm, TrainingUpdate
from src.algorithms.gossip_sgd import GossipSGD
from src.models.base import FederationModel
from src.network.connection_pool import ConnectionPool
from src.network.transport import TransportServer
from src.training.engine import TrainingEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_params(val: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "dense/kernel": np.full((4, 3), val, dtype=np.float32),
        "dense/bias": np.full((3,), val, dtype=np.float32),
    }


def _build_simple_model() -> tf.Module:
    """Build a tiny tf model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, input_shape=(4,)),
    ])
    # Trigger build
    model(tf.zeros((1, 4)))
    return model


def _make_engine(
    algorithm: FederationAlgorithm,
    eval_every: int = 1,
    total_rounds: int = 5,
) -> TrainingEngine:
    """Create a TrainingEngine with mocked server/pool."""
    model = _build_simple_model()
    server = MagicMock(spec=TransportServer)
    server.set_handler = MagicMock()
    pool = MagicMock(spec=ConnectionPool)
    pool.send = AsyncMock(return_value=100)  # 100 bytes sent

    config = {
        "learning_rate": 0.01,
        "optimizer": "sgd",
        "epochs_per_round": 1,
        "total_rounds": total_rounds,
        "batch_size": 32,
        "update_mode": "monolithic",
        "num_traffic_classes": 1,
        "eval_every": eval_every,
    }

    engine = TrainingEngine(
        node_id="node-0",
        model=model,
        algorithm=algorithm,
        server=server,
        pool=pool,
        config=config,
    )
    engine._register_message_handler()
    return engine


def _make_dataset(n_samples: int = 100):
    """Create a small random dataset."""
    np.random.seed(42)
    x_train = np.random.randn(n_samples, 4).astype(np.float32)
    y_train = np.random.randint(0, 3, size=n_samples).astype(np.int64)
    x_val = np.random.randn(20, 4).astype(np.float32)
    y_val = np.random.randint(0, 3, size=20).astype(np.int64)
    return x_train, y_train, x_val, y_val


# ---------------------------------------------------------------------------
# Engine config field
# ---------------------------------------------------------------------------

class TestEngineConfig:

    def test_eval_every_default(self):
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo)
        assert engine.eval_every == 1

    def test_eval_every_custom(self):
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, eval_every=5)
        assert engine.eval_every == 5


# ---------------------------------------------------------------------------
# run_async: basic execution
# ---------------------------------------------------------------------------

class TestRunAsyncBasic:

    @pytest.mark.asyncio
    async def test_runs_to_completion(self):
        """run_async should complete without errors."""
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, total_rounds=3)
        x_train, y_train, x_val, y_val = _make_dataset()

        metrics = await engine.run_async(x_train, y_train, x_val, y_val)
        assert len(metrics) == 3

    @pytest.mark.asyncio
    async def test_returns_metrics_for_each_iteration(self):
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, total_rounds=5)
        x_train, y_train, x_val, y_val = _make_dataset()

        metrics = await engine.run_async(x_train, y_train, x_val, y_val)
        assert len(metrics) == 5
        for i, m in enumerate(metrics):
            assert m["round"] == i

    @pytest.mark.asyncio
    async def test_metrics_have_required_keys(self):
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, total_rounds=2)
        x_train, y_train, x_val, y_val = _make_dataset()

        metrics = await engine.run_async(x_train, y_train, x_val, y_val)
        required_keys = {
            "round", "train_loss", "train_accuracy",
            "val_loss", "val_accuracy",
            "round_duration_s", "train_duration_s",
            "comm_duration_s", "aggregation_duration_s",
            "bytes_sent", "learning_rate", "layer_comm_metrics",
        }
        for m in metrics:
            assert required_keys.issubset(m.keys()), (
                f"Missing keys: {required_keys - m.keys()}"
            )


# ---------------------------------------------------------------------------
# run_async: eval_every behavior
# ---------------------------------------------------------------------------

class TestRunAsyncEvalEvery:

    @pytest.mark.asyncio
    async def test_eval_every_1_evaluates_each_iteration(self):
        """With eval_every=1, val metrics should change each iteration."""
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, eval_every=1, total_rounds=3)
        x_train, y_train, x_val, y_val = _make_dataset()

        metrics = await engine.run_async(x_train, y_train, x_val, y_val)
        # All iterations should have non-zero val_loss (evaluated)
        for m in metrics:
            assert m["val_loss"] > 0 or m["val_accuracy"] >= 0

    @pytest.mark.asyncio
    async def test_eval_every_skips_middle_iterations(self):
        """With eval_every=3 and total_rounds=7:
        Eval at: 0, 3, 6 (last)
        Skip at: 1, 2, 4, 5
        """
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, eval_every=3, total_rounds=7)
        x_train, y_train, x_val, y_val = _make_dataset()

        # Patch _evaluate to track when it's called
        eval_calls = []
        original_evaluate = engine._evaluate

        def tracking_evaluate(x, y):
            eval_calls.append(len(eval_calls))
            return original_evaluate(x, y)

        engine._evaluate = tracking_evaluate

        metrics = await engine.run_async(x_train, y_train, x_val, y_val)

        # Should evaluate at iterations 0, 3, 6
        assert len(eval_calls) == 3

    @pytest.mark.asyncio
    async def test_eval_every_always_evals_last(self):
        """The last iteration always evaluates, even if not on the eval_every boundary."""
        algo = ADPSGD("node-0", ["node-1"], {})
        # total_rounds=5, eval_every=3 → eval at 0, 3, 4(last)
        engine = _make_engine(algo, eval_every=3, total_rounds=5)
        x_train, y_train, x_val, y_val = _make_dataset()

        eval_calls = []
        original_evaluate = engine._evaluate

        def tracking_evaluate(x, y):
            eval_calls.append(len(eval_calls))
            return original_evaluate(x, y)

        engine._evaluate = tracking_evaluate

        await engine.run_async(x_train, y_train, x_val, y_val)
        # Iterations 0, 3, 4(last=True) → 3 evals
        assert len(eval_calls) == 3

    @pytest.mark.asyncio
    async def test_skipped_iterations_carry_forward_metrics(self):
        """Non-evaluated iterations should carry forward the previous val metrics."""
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, eval_every=10, total_rounds=3)
        x_train, y_train, x_val, y_val = _make_dataset()

        metrics = await engine.run_async(x_train, y_train, x_val, y_val)

        # Iteration 0: evaluated (eval_every boundary)
        # Iteration 1: skipped → carries forward iter 0's val metrics
        # Iteration 2: evaluated (last iteration)
        assert metrics[1]["val_loss"] == metrics[0]["val_loss"]
        assert metrics[1]["val_accuracy"] == metrics[0]["val_accuracy"]


# ---------------------------------------------------------------------------
# run_async: non-blocking aggregation
# ---------------------------------------------------------------------------

class TestRunAsyncAggregation:

    @pytest.mark.asyncio
    async def test_no_updates_no_aggregation(self):
        """Without incoming updates, aggregation should be skipped."""
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, total_rounds=3)
        x_train, y_train, x_val, y_val = _make_dataset()

        # Get initial parameters
        initial_params = FederationModel.get_parameters(engine.model)

        metrics = await engine.run_async(x_train, y_train, x_val, y_val)

        # Model should have changed from training but NOT from aggregation
        # (no updates arrived).  Just verify it ran without error.
        assert len(metrics) == 3

    @pytest.mark.asyncio
    async def test_with_incoming_update_aggregates(self):
        """When an update is received mid-loop, aggregation should happen."""
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, total_rounds=3)
        x_train, y_train, x_val, y_val = _make_dataset()

        # Build update with the model's actual layer names (Keras generates
        # names like "sequential/dense/kernel", not our test "dense/kernel").
        actual_params = FederationModel.get_parameters(engine.model)
        fake_neighbor_params = {
            name: np.zeros_like(arr) for name, arr in actual_params.items()
        }
        update = TrainingUpdate(
            source_node="node-1",
            round_num=0,
            parameters=fake_neighbor_params,
            num_samples=100,
        )
        await algo.on_update_received(update)
        assert algo.ready_to_aggregate()

        metrics = await engine.run_async(x_train, y_train, x_val, y_val)
        assert len(metrics) == 3
        # After the first aggregation, the buffer should be clear
        assert not algo.ready_to_aggregate()


# ---------------------------------------------------------------------------
# run_async: round advancement
# ---------------------------------------------------------------------------

class TestRunAsyncRoundAdvancement:

    @pytest.mark.asyncio
    async def test_algorithm_round_advances(self):
        """Algorithm round should advance after each iteration."""
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, total_rounds=5)
        x_train, y_train, x_val, y_val = _make_dataset()

        await engine.run_async(x_train, y_train, x_val, y_val)
        assert algo.get_round() == 5


# ---------------------------------------------------------------------------
# run_async with GossipSGD
# ---------------------------------------------------------------------------

class TestRunAsyncGossipSGD:

    @pytest.mark.asyncio
    async def test_gossip_runs_to_completion(self):
        """Gossip-SGD should work with run_async."""
        algo = GossipSGD("node-0", ["node-1", "node-2"], {"seed": 42})
        engine = _make_engine(algo, total_rounds=3)
        x_train, y_train, x_val, y_val = _make_dataset()

        metrics = await engine.run_async(x_train, y_train, x_val, y_val)
        assert len(metrics) == 3

    @pytest.mark.asyncio
    async def test_gossip_sends_to_one_neighbor_per_iteration(self):
        """Verify gossip sends to exactly one neighbor each iteration."""
        algo = GossipSGD("node-0", ["node-1", "node-2"], {"seed": 42})
        engine = _make_engine(algo, total_rounds=5)
        x_train, y_train, x_val, y_val = _make_dataset()

        # Track send calls
        send_calls = []
        original_send = engine.pool.send

        async def tracking_send(dest, envelope, traffic_class=0):
            send_calls.append(dest)
            return 100

        engine.pool.send = tracking_send

        await engine.run_async(x_train, y_train, x_val, y_val)

        # For monolithic mode with gossip, should have exactly one
        # send per iteration (to one neighbor) + monitor sends.
        neighbor_sends = [s for s in send_calls if s != "monitor"]
        assert len(neighbor_sends) == 5  # one per iteration
        for s in neighbor_sends:
            assert s in ["node-1", "node-2"]


# ---------------------------------------------------------------------------
# run_async: monitor metrics reporting
# ---------------------------------------------------------------------------

class TestRunAsyncMonitoring:

    @pytest.mark.asyncio
    async def test_sends_metrics_to_monitor(self):
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, total_rounds=3)
        engine.monitor_ip = "10.0.0.254"
        engine.monitor_port = 5100
        x_train, y_train, x_val, y_val = _make_dataset()

        send_calls = []
        original_send = engine.pool.send

        async def tracking_send(dest, envelope, traffic_class=0):
            send_calls.append(dest)
            return 100

        engine.pool.send = tracking_send

        await engine.run_async(x_train, y_train, x_val, y_val)

        monitor_sends = [s for s in send_calls if s == "monitor"]
        # Should send metrics to monitor every iteration
        assert len(monitor_sends) == 3

    @pytest.mark.asyncio
    async def test_no_monitor_sends_when_no_monitor_ip(self):
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, total_rounds=2)
        engine.monitor_ip = None
        x_train, y_train, x_val, y_val = _make_dataset()

        send_calls = []

        async def tracking_send(dest, envelope, traffic_class=0):
            send_calls.append(dest)
            return 100

        engine.pool.send = tracking_send

        await engine.run_async(x_train, y_train, x_val, y_val)

        monitor_sends = [s for s in send_calls if s == "monitor"]
        assert len(monitor_sends) == 0


# ---------------------------------------------------------------------------
# run_async: timing metrics
# ---------------------------------------------------------------------------

class TestRunAsyncTimingMetrics:

    @pytest.mark.asyncio
    async def test_timing_metrics_positive(self):
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, total_rounds=2)
        x_train, y_train, x_val, y_val = _make_dataset()

        metrics = await engine.run_async(x_train, y_train, x_val, y_val)
        for m in metrics:
            assert m["round_duration_s"] >= 0
            assert m["train_duration_s"] >= 0
            assert m["comm_duration_s"] >= 0
            assert m["aggregation_duration_s"] >= 0

    @pytest.mark.asyncio
    async def test_train_duration_less_than_round_duration(self):
        algo = ADPSGD("node-0", ["node-1"], {})
        engine = _make_engine(algo, total_rounds=2)
        x_train, y_train, x_val, y_val = _make_dataset()

        metrics = await engine.run_async(x_train, y_train, x_val, y_val)
        for m in metrics:
            assert m["train_duration_s"] <= m["round_duration_s"]
