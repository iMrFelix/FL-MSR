"""Integration-phase wiring tests (cross-module seams).

Each module landed with its own suite; these tests pin the seams the
integration phase wired up between them, per the module agents'
integration requests:

- the FedAvg slippage realization (``last_aggregation_telemetry`` /
  ``inclusion_counts`` / ``aggregation_count``) travels from the algorithm
  through ``run_aggregator`` into the report's reserved ``_aggregation``
  telemetry block (docs/extensions/04-overnight-interfaces.md §2);
- the collector hoists cancelled zombie-tail bytes out of the ``_sender``
  block into the per-entry ``cancelled_bytes`` scalar that
  ``scripts/posthoc_cost.py`` reads (pre-run fix 3, the RQ4 $-input);
- the analysis-side telemetry parser skips reserved underscore keys so
  bookkeeping blocks are never mistaken for source flows.
"""

from __future__ import annotations

import json

import numpy as np
import tensorflow as tf
from unittest.mock import AsyncMock, MagicMock

from src.algorithms.base import FederationAlgorithm, TrainingUpdate
from src.algorithms.fedavg import FedAvg
from src.monitoring.collector import MetricsCollector
from src.network.connection_pool import ConnectionPool
from src.network.transport import TransportServer
from src.proto_gen import federation_pb2
from src.training.engine import TrainingEngine
from scripts.overnight_common import uplink_observations


def _build_model() -> tf.Module:
    model = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(4,))])
    model(tf.zeros((1, 4)))
    return model


def _tiny_dataset():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(32, 4)).astype(np.float32)
    y_train = rng.integers(0, 3, size=32).astype(np.int64)
    x_val = rng.normal(size=(8, 4)).astype(np.float32)
    y_val = rng.integers(0, 3, size=8).astype(np.int64)
    return x_train, y_train, x_val, y_val


def _model_params(model: tf.Module) -> dict[str, np.ndarray]:
    return {
        var.path: var.numpy() for var in model.trainable_variables
    }


# ---------------------------------------------------------------------------
# FedAvg aggregation realization -> `_aggregation` telemetry block
# ---------------------------------------------------------------------------

class TestAggregationTelemetryBlock:

    async def test_run_aggregator_ships_aggregation_block(self):
        """A partial worker update surfaces as `_aggregation` in the report:
        per-layer arrived_sources/filled, cumulative inclusion_counts with
        explicit zeros, and the aggregation_count denominator."""
        config = {
            "learning_rate": 0.01, "optimizer": "sgd", "epochs_per_round": 1,
            "total_rounds": 1, "batch_size": 16,
            "update_mode": "per_layer", "num_traffic_classes": 3,
            "epsilon_deadline": 0.2, "watchdog_factor": 0.0,
            "importance_metric_v2": "delta_sq_norm",
            "assignment_strategy": "gap_based",
            "late_layer_policy": "drop", "seed": 7, "sync_timeout": 0.05,
            "role": "aggregator",
        }
        algorithm = FedAvg(
            node_id="node-0", neighbors=["node-1", "node-2"], config=config,
        )
        server = MagicMock(spec=TransportServer)
        pool = MagicMock(spec=ConnectionPool)
        sent: list[federation_pb2.Envelope] = []

        async def capture(dest, envelope, traffic_class=0):
            sent.append(envelope)
            return envelope.ByteSize()

        pool.send = capture
        model = _build_model()
        engine = TrainingEngine(
            node_id="node-0", model=model, algorithm=algorithm,
            server=server, pool=pool, config=config,
        )
        engine._register_message_handler()
        engine.monitor_ip = "10.0.0.254"
        engine.monitor_port = 5100

        # Pre-buffer the round-0 worker updates so the barrier releases
        # immediately: node-1 complete, node-2 missing one layer (slipped
        # past the trigger -> 'drop' stale-fills it at aggregation).
        params = _model_params(model)
        layer_names = sorted(params)
        missing_layer = layer_names[0]
        await algorithm.on_update_received(TrainingUpdate(
            source_node="node-1", round_num=0,
            parameters=dict(params), num_samples=10,
        ))
        partial = {k: v for k, v in params.items() if k != missing_layer}
        await algorithm.on_update_received(TrainingUpdate(
            source_node="node-2", round_num=0,
            parameters=partial, num_samples=10,
        ))

        history = await engine.run_aggregator(*_tiny_dataset())
        assert len(history) == 1

        reports = [
            e.metrics_report for e in sent
            if e.WhichOneof("payload") == "metrics_report"
        ]
        assert len(reports) == 1
        telemetry = json.loads(reports[0].uplink_telemetry_json)
        assert "_aggregation" in telemetry
        block = telemetry["_aggregation"]

        assert block["aggregation_count"] == 1
        layers = block["layers"]
        assert layers[missing_layer]["arrived_sources"] == ["node-1"]
        assert layers[missing_layer]["filled"] == "stale"
        complete = [n for n in layer_names if n != missing_layer]
        for name in complete:
            assert layers[name]["filled"] == "none"
            assert layers[name]["arrived_sources"] == ["node-1", "node-2"]
        counts = block["inclusion_counts"]
        assert counts[missing_layer] == {"node-1": 1, "node-2": 0}
        for name in complete:
            assert counts[name] == {"node-1": 1, "node-2": 1}

    async def test_snapshot_isolated_from_inplace_count_mutation(self):
        """inclusion_counts is mutated in place round over round; the
        captured block must be an as-of-capture copy, not a live view."""
        config = {
            "update_mode": "monolithic", "late_layer_policy": "drop",
            "role": "aggregator",
        }
        algorithm = FedAvg(
            node_id="node-0", neighbors=["node-1"], config=config,
        )
        engine = TrainingEngine(
            node_id="node-0", model=_build_model(), algorithm=algorithm,
            server=MagicMock(spec=TransportServer),
            pool=MagicMock(spec=ConnectionPool), config=config,
        )
        algorithm.last_aggregation_telemetry = {
            "dense/kernel": {"arrived_sources": ["node-1"], "filled": "none"},
        }
        algorithm.inclusion_counts = {"dense/kernel": {"node-1": 1}}
        algorithm.aggregation_count = 1

        engine._capture_aggregation_telemetry(0)
        algorithm.inclusion_counts["dense/kernel"]["node-1"] = 99

        block = engine._aggregation_info[0]
        assert block["inclusion_counts"]["dense/kernel"]["node-1"] == 1
        assert block["aggregation_count"] == 1

    async def test_algorithm_without_telemetry_produces_no_block(self):
        """Non-FedAvg algorithms (no telemetry attributes) must not crash
        the capture nor emit an `_aggregation` key."""

        class Bare(FederationAlgorithm):
            @property
            def is_synchronous(self):
                return True

            @property
            def is_centralized(self):
                return False

            async def on_local_training_complete(self, p, r, n):
                return []

            async def on_update_received(self, update):
                pass

            def ready_to_aggregate(self):
                return False

            async def aggregate(self, local_params):
                return local_params

        engine = TrainingEngine(
            node_id="node-0", model=_build_model(),
            algorithm=Bare("node-0", ["node-1"], {}),
            server=MagicMock(spec=TransportServer),
            pool=MagicMock(spec=ConnectionPool),
            config={"update_mode": "monolithic"},
        )
        engine._capture_aggregation_telemetry(0)
        assert engine._aggregation_info == {}
        assert engine._export_uplink_telemetry(0) == ""


# ---------------------------------------------------------------------------
# Collector: cancelled zombie-tail bytes hoisted for the cost scripts
# ---------------------------------------------------------------------------

def _report_envelope(telemetry: dict | None) -> federation_pb2.Envelope:
    report = federation_pb2.MetricsReport(node_id="node-1", round=2)
    if telemetry is not None:
        report.uplink_telemetry_json = json.dumps(telemetry)
    return federation_pb2.Envelope(
        source_node="node-1", dest_node="monitor", metrics_report=report,
    )


class TestCollectorCancelledBytesHoist:

    async def test_sender_block_hoisted_to_cancelled_bytes(self, tmp_path):
        """tail_cancelled_bytes + tail_cancelled prior_round_events sum into
        the documented per-entry scalar; tail_sent_late events do not (those
        bytes traversed the wire and are priced as sent)."""
        collector = MetricsCollector(
            host="127.0.0.1", port=0, logdir=str(tmp_path),
            num_nodes=2, total_rounds=3,
        )
        telemetry = {
            "_sender": {
                "predicted_t_eps": 0.4,
                "tail_cancelled_bytes": 1234,
                "tail_cancelled_layers": ["conv_0/kernel"],
                "prior_round_events": [
                    {"round": 1, "type": "tail_cancelled", "bytes": 100,
                     "layers": ["dense_1/kernel"]},
                    {"round": 1, "type": "tail_sent_late", "bytes": 999,
                     "layers": ["dense_1/bias"]},
                ],
            },
        }
        await collector._handle_message(_report_envelope(telemetry))
        stored = collector.get_all_metrics()[0]
        assert stored["cancelled_bytes"] == 1334
        # The full block remains available for per-round re-attribution.
        assert stored["uplink_telemetry"]["_sender"]["tail_cancelled_bytes"] \
            == 1234

    async def test_no_cancellations_leave_key_absent(self, tmp_path):
        collector = MetricsCollector(
            host="127.0.0.1", port=0, logdir=str(tmp_path),
            num_nodes=2, total_rounds=3,
        )
        await collector._handle_message(
            _report_envelope({"_sender": {"predicted_t_eps": 0.4}})
        )
        await collector._handle_message(_report_envelope(None))
        for stored in collector.get_all_metrics():
            assert "cancelled_bytes" not in stored

    async def test_malformed_sender_block_tolerated(self, tmp_path):
        """Defensive parsing: junk shapes must not drop the report."""
        collector = MetricsCollector(
            host="127.0.0.1", port=0, logdir=str(tmp_path),
            num_nodes=2, total_rounds=3,
        )
        telemetry = {
            "_sender": {
                "tail_cancelled_bytes": "not-a-number",
                "prior_round_events": [
                    "junk", {"type": "tail_cancelled", "bytes": None},
                ],
            },
        }
        await collector._handle_message(_report_envelope(telemetry))
        stored = collector.get_all_metrics()[0]
        assert "cancelled_bytes" not in stored
        assert "uplink_telemetry" in stored


# ---------------------------------------------------------------------------
# Analysis: reserved underscore keys are not source flows
# ---------------------------------------------------------------------------

class TestUplinkObservationsReservedKeys:

    def test_underscore_keys_skipped(self):
        # Deliberately in the pre-NT-03 spelling: reports on disk look like
        # this, and the parser must keep reading them.
        flow = {
            "manifest_arrival_rel": 0.01,
            "trigger_fire_rel": 0.5,
            "watchdog_fired": False,
            "t_eps_local": 0.49,
            "layer_arrivals_rel": {"dense/kernel": 0.3},
            "shed_layers": [],
            "kappa_realized": 0.0,
            "ordering_violation": False,
        }
        report = {
            "per_round": [
                {
                    "round": 0,
                    "nodes": {
                        "node-0": {
                            "uplink_telemetry": {
                                "node-1": flow,
                                "_sender": {"predicted_t_eps": 0.4},
                                "_aggregation": {"aggregation_count": 1},
                            },
                        },
                    },
                },
            ],
        }
        observations = uplink_observations("run-x", report, "node-0")
        assert [o.source for o in observations] == ["node-1"]
        assert observations[0].t_eps_local_receiver_s == 0.49
