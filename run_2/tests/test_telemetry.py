"""Tests for the gate-doc pre-run measurement-validity fixes (Agent C wiring).

Covers, per writeup/01-candidate-selection.md §3 and
docs/extensions/04-overnight-interfaces.md §5:

- receiver-side uplink telemetry (G1) end-to-end into the collector,
- the T_max watchdog (G4): forced fire + cancellation on natural fire,
- the manifest-after-payload race fix (fix 4) + ordering-violation flag,
- the degenerate-manifest guard (fix 6),
- the ε warm-up schedule (computed, not signalled),
- zombie-tail cancellation accounting (fix 3),
- per-class SO_SNDBUF pinning (G1 fix 2),
- the G5 concurrent-destination broadcast + manifest-less downlink,
- the metric-input swap (shipped delta vs last-gradient control) and
  manifest sched_score plumbing (G2).
"""

from __future__ import annotations

import asyncio
import json
import socket

import numpy as np
import pytest
import tensorflow as tf
from unittest.mock import AsyncMock, MagicMock

from src.algorithms.base import FederationAlgorithm, TrainingUpdate
from src.importance.assignment import AssignmentResult
from src.importance.manifest import build_manifest, validate_manifest
from src.monitoring.collector import MetricsCollector
from src.network.connection_pool import ConnectionPool
from src.network.transport import TransportServer
from src.proto_gen import federation_pb2
from src.training.engine import TrainingEngine, _SendPlan


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

class RecordingAlgorithm(FederationAlgorithm):
    """Minimal algorithm capturing every delivered update."""

    def __init__(
        self,
        node_id: str = "node-0",
        neighbors: list[str] | None = None,
        centralized: bool = False,
    ):
        super().__init__(node_id, neighbors or ["node-1"], {})
        self.received: list[TrainingUpdate] = []
        self._centralized = centralized

    @property
    def is_synchronous(self) -> bool:
        return True

    @property
    def is_centralized(self) -> bool:
        return self._centralized

    async def on_local_training_complete(self, model_params, round_num, num_samples):
        update = TrainingUpdate(self.node_id, round_num, model_params, num_samples)
        return [(neighbor, update) for neighbor in self.neighbors]

    async def on_update_received(self, update: TrainingUpdate) -> None:
        self.received.append(update)

    def ready_to_aggregate(self) -> bool:
        return False

    async def aggregate(self, local_params):
        return local_params


def _build_model() -> tf.Module:
    model = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(4,))])
    model(tf.zeros((1, 4)))
    return model


def make_engine(
    *,
    update_mode: str = "per_layer",
    epsilon: float = 0.5,
    warmup: int = 0,
    watchdog_factor: float = 0.0,
    metric_v2: str = "delta_sq_norm",
    strategy: str = "gap_based",
    num_classes: int = 3,
    centralized: bool = False,
    neighbors: list[str] | None = None,
) -> TrainingEngine:
    algorithm = RecordingAlgorithm(neighbors=neighbors, centralized=centralized)
    server = MagicMock(spec=TransportServer)
    pool = MagicMock(spec=ConnectionPool)
    pool.send = AsyncMock(return_value=100)
    config = {
        "learning_rate": 0.01,
        "optimizer": "sgd",
        "epochs_per_round": 1,
        "total_rounds": 3,
        "batch_size": 32,
        "update_mode": update_mode,
        "num_traffic_classes": num_classes,
        "epsilon_deadline": epsilon,
        "epsilon_warmup_rounds": warmup,
        "watchdog_factor": watchdog_factor,
        "importance_metric_v2": metric_v2,
        "assignment_strategy": strategy,
        "late_layer_policy": "drop",
        "seed": 7,
    }
    engine = TrainingEngine(
        node_id="node-0",
        model=_build_model(),
        algorithm=algorithm,
        server=server,
        pool=pool,
        config=config,
    )
    engine._register_message_handler()
    return engine


def _handler(engine: TrainingEngine):
    """Return the message handler the engine registered on its server."""
    return engine.server.set_handler.call_args[0][0]


def _manifest_envelope(
    source: str,
    round_num: int,
    scores: dict[str, float],
    must_receive: set[str] | None = None,
) -> federation_pb2.Envelope:
    must = must_receive or set()
    manifest = build_manifest(
        round_num=round_num,
        source_node_id=source,
        importance_scores=scores,
        must_receive_predicate=lambda name, _s: name in must,
    )
    return federation_pb2.Envelope(
        source_node=source, dest_node="node-0", round_manifest=manifest,
    )


def _layer_envelope(
    source: str,
    round_num: int,
    layer_name: str,
    layer_index: int,
    total_layers: int,
) -> federation_pb2.Envelope:
    array = np.full((4,), float(layer_index), dtype=np.float32)
    layer_update = federation_pb2.LayerUpdate(
        layer_name=layer_name,
        parameters=array.tobytes(),
        shape=[4],
        dtype="float32",
        layer_index=layer_index,
        total_layers=total_layers,
        num_samples=10,
        round=round_num,
    )
    return federation_pb2.Envelope(
        source_node=source, dest_node="node-0", layer_update=layer_update,
    )


def _cleanup(engine: TrainingEngine) -> None:
    """Cancel timers/tasks an aborted test may leave behind."""
    for task in list(engine._watchdog_tasks.values()):
        task.cancel()
    engine._watchdog_tasks.clear()
    for tasks in engine._tail_tasks.values():
        for task in tasks:
            task.cancel()
    engine._tail_tasks.clear()


# ---------------------------------------------------------------------------
# Manifest sched_score plumbing (G2)
# ---------------------------------------------------------------------------

class TestManifestSchedScores:

    def test_sched_scores_written_when_provided(self):
        manifest = build_manifest(
            round_num=1,
            source_node_id="node-1",
            importance_scores={"a": 1.0, "b": 2.0},
            sched_scores={"a": 0.25, "b": 4.0},
        )
        by_name = {e.layer_name: e for e in manifest.entries}
        assert by_name["a"].sched_score == pytest.approx(0.25)
        assert by_name["b"].sched_score == pytest.approx(4.0)
        validate_manifest(manifest)

    def test_sched_scores_default_to_zero(self):
        """0.0 on the wire means 'same as raw_score' (proto3 default)."""
        manifest = build_manifest(
            round_num=1,
            source_node_id="node-1",
            importance_scores={"a": 1.0},
        )
        assert manifest.entries[0].sched_score == 0.0

    def test_validate_rejects_non_finite_sched_score(self):
        manifest = build_manifest(
            round_num=1,
            source_node_id="node-1",
            importance_scores={"a": 1.0},
            sched_scores={"a": float("nan")},
        )
        with pytest.raises(ValueError, match="sched_score"):
            validate_manifest(manifest)


# ---------------------------------------------------------------------------
# ε warm-up schedule (interface doc §1.8)
# ---------------------------------------------------------------------------

class TestEpsilonWarmup:

    def test_effective_epsilon_schedule(self):
        engine = make_engine(epsilon=0.4, warmup=2)
        assert engine._effective_epsilon(0) == 0.0
        assert engine._effective_epsilon(1) == 0.0
        assert engine._effective_epsilon(2) == 0.4
        assert engine._effective_epsilon(10) == 0.4

    async def test_warmup_round_requires_all_layers(self):
        """During warm-up the coverage trigger is inert (ε forced to 0)."""
        engine = make_engine(epsilon=0.5, warmup=1)
        handler = _handler(engine)
        scores = {"a": 10.0, "b": 1.0, "c": 1.0}

        await handler(_manifest_envelope("node-1", 0, scores))
        # 'a' alone covers 10/12 > (1-0.5) of the mass — but round 0 is a
        # warm-up round, so nothing may fire before all layers arrive.
        await handler(_layer_envelope("node-1", 0, "a", 0, 3))
        assert engine.algorithm.received == []
        await handler(_layer_envelope("node-1", 0, "b", 1, 3))
        assert engine.algorithm.received == []
        await handler(_layer_envelope("node-1", 0, "c", 2, 3))
        assert len(engine.algorithm.received) == 1
        assert set(engine.algorithm.received[0].parameters) == {"a", "b", "c"}

    async def test_post_warmup_round_uses_configured_epsilon(self):
        engine = make_engine(epsilon=0.5, warmup=1)
        handler = _handler(engine)
        scores = {"a": 10.0, "b": 1.0, "c": 1.0}

        await handler(_manifest_envelope("node-1", 1, scores))
        await handler(_layer_envelope("node-1", 1, "a", 0, 3))
        assert len(engine.algorithm.received) == 1  # coverage trigger fired
        assert set(engine.algorithm.received[0].parameters) == {"a"}


# ---------------------------------------------------------------------------
# Manifest race fix (pre-run fix 4)
# ---------------------------------------------------------------------------

class TestManifestRace:

    async def test_coverage_reevaluated_on_late_manifest(self):
        """Payloads racing ahead of their manifest fire at registration."""
        engine = make_engine(epsilon=0.5)
        handler = _handler(engine)

        # Payloads first: 'a' would satisfy coverage, but no manifest yet.
        await handler(_layer_envelope("node-1", 0, "a", 0, 3))
        assert engine.algorithm.received == []

        # Manifest arrives late -> coverage re-check fires immediately.
        await handler(
            _manifest_envelope("node-1", 0, {"a": 10.0, "b": 1.0, "c": 1.0})
        )
        assert len(engine.algorithm.received) == 1
        update = engine.algorithm.received[0]
        assert set(update.parameters) == {"a"}

        telemetry = json.loads(engine._export_uplink_telemetry(0))
        flow = telemetry["node-1"]
        assert flow["ordering_violation"] is True
        assert flow["t_eps_local_receiver_s"] is not None
        assert sorted(flow["shed_layers"]) == ["b", "c"]

    async def test_no_ordering_violation_when_manifest_first(self):
        engine = make_engine(epsilon=0.5)
        handler = _handler(engine)
        await handler(_manifest_envelope("node-1", 0, {"a": 1.0, "b": 1.0}))
        await handler(_layer_envelope("node-1", 0, "a", 0, 2))
        await handler(_layer_envelope("node-1", 0, "b", 1, 2))
        telemetry = json.loads(engine._export_uplink_telemetry(0))
        assert telemetry["node-1"]["ordering_violation"] is False


# ---------------------------------------------------------------------------
# Degenerate-manifest guard (pre-run fix 6)
# ---------------------------------------------------------------------------

class TestDegenerateManifestGuard:

    async def test_zero_mass_manifest_falls_back_to_count_trigger(self, caplog):
        engine = make_engine(epsilon=0.5)
        handler = _handler(engine)
        scores = {"a": 0.0, "b": 0.0, "c": 0.0}

        with caplog.at_level("WARNING"):
            await handler(_manifest_envelope("node-1", 0, scores))
        assert any("DEGENERATE MANIFEST" in r.message for r in caplog.records)

        # A vacuous threshold ((1-ε)·0 == 0) must NOT fire on first arrival.
        await handler(_layer_envelope("node-1", 0, "a", 0, 3))
        await handler(_layer_envelope("node-1", 0, "b", 1, 3))
        assert engine.algorithm.received == []

        # Count-based completion still works.
        await handler(_layer_envelope("node-1", 0, "c", 2, 3))
        assert len(engine.algorithm.received) == 1
        assert not engine.algorithm.received[0].parameters.keys() - {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Watchdog (gate ruling G4)
# ---------------------------------------------------------------------------

def _incoming_edge(src_idx: int, bandwidth_mbps: float) -> dict:
    return {
        "src": src_idx,
        "dst": 0,
        "classes": {
            0: {"bandwidth_mbps": bandwidth_mbps, "latency_ms": 0.0,
                "drop_rate": 0.0},
        },
    }


class TestWatchdog:

    async def test_watchdog_force_fires_partial_flow(self):
        engine = make_engine(epsilon=0.2, watchdog_factor=3.0)
        engine.incoming_edges = [_incoming_edge(1, 10_000.0)]  # tiny T_max
        handler = _handler(engine)
        try:
            scores = {"a": 1.0, "b": 1.0, "c": 1.0}
            await handler(_manifest_envelope("node-1", 0, scores))
            assert ("node-1", 0) in engine._watchdog_tasks
            await handler(_layer_envelope("node-1", 0, "a", 0, 3))

            # Coverage 1/3 < 0.8 of mass: only the watchdog can complete.
            for _ in range(100):
                if engine.algorithm.received:
                    break
                await asyncio.sleep(0.01)
            assert len(engine.algorithm.received) == 1
            update = engine.algorithm.received[0]
            assert set(update.parameters) == {"a"}

            telemetry = json.loads(engine._export_uplink_telemetry(0))
            flow = telemetry["node-1"]
            assert flow["watchdog_fired"] is True
            assert sorted(flow["shed_layers"]) == ["b", "c"]
            assert flow["kappa_realized"] == pytest.approx(2.0 / 3.0)
        finally:
            _cleanup(engine)

    async def test_watchdog_cancelled_on_natural_completion(self):
        engine = make_engine(epsilon=0.2, watchdog_factor=1000.0)
        engine.incoming_edges = [_incoming_edge(1, 0.001)]  # huge T_max
        handler = _handler(engine)
        try:
            await handler(_manifest_envelope("node-1", 0, {"a": 1.0, "b": 1.0}))
            assert ("node-1", 0) in engine._watchdog_tasks
            await handler(_layer_envelope("node-1", 0, "a", 0, 2))
            await handler(_layer_envelope("node-1", 0, "b", 1, 2))
            assert len(engine.algorithm.received) == 1
            # Natural completion must cancel the timer; no double delivery.
            assert ("node-1", 0) not in engine._watchdog_tasks
            await asyncio.sleep(0.02)
            assert len(engine.algorithm.received) == 1
            telemetry = json.loads(engine._export_uplink_telemetry(0))
            assert telemetry["node-1"]["watchdog_fired"] is False
        finally:
            _cleanup(engine)

    async def test_watchdog_not_armed_without_bandwidth_info(self):
        engine = make_engine(epsilon=0.2, watchdog_factor=3.0)
        handler = _handler(engine)  # no incoming_edges set
        await handler(_manifest_envelope("node-1", 0, {"a": 1.0, "b": 1.0}))
        assert engine._watchdog_tasks == {}

    async def test_watchdog_disabled_by_config(self):
        engine = make_engine(epsilon=0.2, watchdog_factor=0.0)
        engine.incoming_edges = [_incoming_edge(1, 10_000.0)]
        handler = _handler(engine)
        await handler(_manifest_envelope("node-1", 0, {"a": 1.0, "b": 1.0}))
        assert engine._watchdog_tasks == {}


# ---------------------------------------------------------------------------
# Telemetry JSON: schema + collector round-trip (G1, end-to-end)
# ---------------------------------------------------------------------------

class TestTelemetryRoundTrip:

    async def test_export_schema_fields(self):
        engine = make_engine(epsilon=0.3)
        handler = _handler(engine)
        engine._round_starts[0] = 0.0  # exercise the offset path

        scores = {"a": 3.0, "b": 1.0}
        await handler(_manifest_envelope("node-1", 0, scores))
        await handler(_layer_envelope("node-1", 0, "a", 0, 2))
        # 'a' covers 3/4 > (1-0.3): partial fire, 'b' shed.
        assert len(engine.algorithm.received) == 1

        telemetry = json.loads(engine._export_uplink_telemetry(0))
        flow = telemetry["node-1"]
        expected_keys = {
            "manifest_arrival_rel_receiver_s", "trigger_fire_rel_receiver_s",
            "watchdog_fired", "t_eps_local_receiver_s",
            "layer_arrivals_rel_receiver_s", "manifest_wire_time_oneway_s",
            "layer_wire_times_oneway_s", "t_cover_oneway_s",
            "wire_clock_anomaly", "shed_layers", "kappa_realized",
            "kappa_slip", "shed_mass_fraction", "ordering_violation",
        }
        assert set(flow) == expected_keys
        assert flow["t_eps_local_receiver_s"] == pytest.approx(
            flow["trigger_fire_rel_receiver_s"]
            - flow["manifest_arrival_rel_receiver_s"],
            abs=1e-6,
        )
        assert flow["shed_layers"] == ["b"]
        assert flow["kappa_realized"] == pytest.approx(0.25)
        assert "a" in flow["layer_arrivals_rel_receiver_s"]
        # Every time-valued key carries a clock domain (audit NT-03).
        assert all(
            key.endswith(("_receiver_s", "_oneway_s", "_sender_s"))
            for key in flow
            if key.endswith("_s")
        )

    async def test_empty_when_nothing_to_report(self):
        engine = make_engine()
        assert engine._export_uplink_telemetry(0) == ""

    async def test_report_to_collector_round_trip(self, tmp_path):
        """Engine report -> MetricsReport -> collector -> report.json entry."""
        engine = make_engine(epsilon=0.3)
        handler = _handler(engine)
        engine.monitor_ip = "10.0.0.254"
        engine.monitor_port = 5100

        await handler(_manifest_envelope("node-1", 0, {"a": 3.0, "b": 1.0}))
        await handler(_layer_envelope("node-1", 0, "a", 0, 2))

        sent_envelopes: list[federation_pb2.Envelope] = []

        async def capture_send(dest, envelope, traffic_class=0):
            sent_envelopes.append(envelope)
            return envelope.ByteSize()

        engine.pool.send = capture_send
        metrics = {
            "round": 0, "train_loss": 0.5, "train_accuracy": 0.8,
            "val_loss": 0.6, "val_accuracy": 0.7, "round_duration_s": 1.0,
            "train_duration_s": 0.5, "comm_duration_s": 0.2,
            "send_enqueue_duration_sender_s": 0.1,
            "barrier_wait_duration_s": 0.1,
            "aggregation_duration_s": 0.05, "layer_comm_metrics": [],
        }
        await engine._send_metrics_to_monitor(metrics)
        assert len(sent_envelopes) == 1
        report = sent_envelopes[0].metrics_report
        assert report.uplink_telemetry_json  # non-empty sidecar

        collector = MetricsCollector(
            host="127.0.0.1", port=0, logdir=str(tmp_path),
            num_nodes=1, total_rounds=3,
        )
        await collector._handle_message(sent_envelopes[0])
        stored = collector.get_all_metrics()
        assert len(stored) == 1
        uplink = stored[0]["uplink_telemetry"]
        assert uplink["node-1"]["shed_layers"] == ["b"]
        assert uplink["node-1"]["kappa_realized"] == pytest.approx(0.25)
        assert uplink["node-1"]["watchdog_fired"] is False
        # round 0 is now reported: later tail activity must divert to events
        assert 0 in engine._reported_rounds

    async def test_collector_tolerates_absent_telemetry(self, tmp_path):
        report = federation_pb2.MetricsReport(node_id="node-2", round=1)
        envelope = federation_pb2.Envelope(
            source_node="node-2", dest_node="monitor", metrics_report=report,
        )
        collector = MetricsCollector(
            host="127.0.0.1", port=0, logdir=str(tmp_path),
            num_nodes=2, total_rounds=3,
        )
        await collector._handle_message(envelope)
        stored = collector.get_all_metrics()
        assert "uplink_telemetry" not in stored[0]


# ---------------------------------------------------------------------------
# Send path: omission semantics, downlink, concurrency (G5), zombie tails
# ---------------------------------------------------------------------------

def _params(*names: str) -> dict[str, np.ndarray]:
    return {
        name: np.full((8,), float(i), dtype=np.float32)
        for i, name in enumerate(names)
    }


def _plan(
    round_num: int,
    assignment: dict[str, int],
    head: set[str],
    trigger_scores: dict[str, float] | None,
) -> _SendPlan:
    return _SendPlan(
        round_num=round_num,
        assignment=AssignmentResult(
            assignment=assignment,
            head=head,
            tail=set(assignment) - head,
        ),
        trigger_scores=trigger_scores,
        sched_scores=None,
        must_receive=set(),
    )


class TestSendPath:

    async def test_omitted_layers_not_transmitted(self):
        """Manifest entries == transmitted layers; total_layers matches."""
        engine = make_engine()
        params = _params("a", "b", "c")
        sent: list[federation_pb2.Envelope] = []

        async def capture(dest, envelope, traffic_class=0):
            sent.append(envelope)
            return envelope.ByteSize()

        engine.pool.send = capture
        plan = _plan(0, {"a": 0, "b": 1}, {"a", "b"}, {"a": 2.0, "b": 1.0})
        update = TrainingUpdate("node-0", 0, params, 10)
        await engine._send_updates([("node-1", update)], plan)

        manifests = [e for e in sent if e.WhichOneof("payload") == "round_manifest"]
        layers = [e for e in sent if e.WhichOneof("payload") == "layer_update"]
        assert len(manifests) == 1
        assert {e.layer_name for e in manifests[0].round_manifest.entries} == {"a", "b"}
        assert {e.layer_update.layer_name for e in layers} == {"a", "b"}
        assert all(e.layer_update.total_layers == 2 for e in layers)

    async def test_downlink_plan_full_head_no_manifest(self):
        """G5: broadcast covers every layer, sends no manifest."""
        engine = make_engine()
        params = _params("a", "b", "c")
        plan = engine._downlink_plan(params, round_num=1)
        assert plan.trigger_scores is None
        assert plan.assignment.head == set(params)
        assert plan.assignment.tail == set()

        sent: list[federation_pb2.Envelope] = []

        async def capture(dest, envelope, traffic_class=0):
            sent.append(envelope)
            return envelope.ByteSize()

        engine.pool.send = capture
        update = TrainingUpdate("node-0", 1, params, 10)
        await engine._send_updates(
            [("node-1", update), ("node-2", update)], plan,
        )
        kinds = {e.WhichOneof("payload") for e in sent}
        assert kinds == {"layer_update"}  # manifest-less downlink
        per_dest = {}
        for e in sent:
            per_dest.setdefault(e.dest_node, set()).add(e.layer_update.layer_name)
        assert per_dest == {"node-1": {"a", "b", "c"}, "node-2": {"a", "b", "c"}}

    async def test_monolithic_broadcast_concurrent_destinations(self):
        """G5: destinations overlap instead of serializing."""
        engine = make_engine(update_mode="monolithic", num_classes=1)
        active = 0
        peak = 0

        async def slow_send(dest, envelope, traffic_class=0):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return 50

        engine.pool.send = slow_send
        params = _params("a", "b")
        update = TrainingUpdate("node-0", 0, params, 10)
        destinations = [(f"node-{i}", update) for i in (1, 2, 3)]
        total, metrics = await engine._send_updates(destinations, None)
        assert total == 150
        assert metrics == []
        assert peak >= 2  # overlapping sends, not a serial loop

    async def test_zombie_tail_cancellation_counts_unsent_bytes(self):
        """Fix 3: closing the round skips unsent tail envelopes only."""
        engine = make_engine(centralized=True)
        params = _params("a", "b", "c", "d")
        tail_gate = asyncio.Event()
        sent_layers: list[str] = []

        async def gated_send(dest, envelope, traffic_class=0):
            if envelope.WhichOneof("payload") == "layer_update":
                name = envelope.layer_update.layer_name
                sent_layers.append(name)
                if name == "b":
                    # 'b' is mid-send when the round closes: it cannot be
                    # recalled and must complete.
                    await tail_gate.wait()
            return envelope.ByteSize()

        engine.pool.send = gated_send
        plan = _plan(
            0, {"a": 0, "b": 0, "c": 0, "d": 0}, {"a"},
            {"a": 4.0, "b": 1.0, "c": 1.0, "d": 1.0},
        )
        update = TrainingUpdate("node-0", 0, params, 10)
        # Returns after the head ('a'); the tail (b, c, d) continues in the
        # background on the class-0 queue.
        await engine._send_updates([("node-1", update)], plan)
        assert "a" in sent_layers

        # Wait until 'b' is in-flight, then close the round (broadcast).
        for _ in range(100):
            if "b" in sent_layers:
                break
            await asyncio.sleep(0.005)
        engine._maybe_signal_rounds_closed(0)
        tail_gate.set()
        await engine._retire_tail_rounds_before(1)

        assert sent_layers == ["a", "b"]  # c, d cancelled before sending
        ledger = engine._tail_ledgers[0]
        assert sorted(ledger["cancelled_layers"]) == ["c", "d"]
        assert ledger["cancelled_bytes"] > 0
        # 'b' completed during the round: accounted as a tail send.
        tail_bytes, tail_metrics = engine._collect_tail_round_metrics(0)
        assert [m["layer_name"] for m in tail_metrics] == ["b"]
        assert tail_bytes > 0
        # And the cancellation surfaces in the _sender telemetry block.
        telemetry = json.loads(engine._export_uplink_telemetry(0))
        sender = telemetry["_sender"]
        assert sender["tail_cancelled_bytes"] == ledger["cancelled_bytes"]
        assert sorted(sender["tail_cancelled_layers"]) == ["c", "d"]

    async def test_aggregator_ignores_round_close_signal(self):
        engine = make_engine(centralized=True)
        engine._is_aggregator = True
        engine._tail_cancel_events[0] = asyncio.Event()
        engine._maybe_signal_rounds_closed(5)
        assert not engine._tail_cancel_events[0].is_set()

    async def test_decentralized_ignores_round_close_signal(self):
        engine = make_engine(centralized=False)
        engine._tail_cancel_events[0] = asyncio.Event()
        engine._maybe_signal_rounds_closed(5)
        assert not engine._tail_cancel_events[0].is_set()


# ---------------------------------------------------------------------------
# Metric-input swap (gate headline fix 3) + G2 frozen accounting
# ---------------------------------------------------------------------------

class TestMetricInputSwap:

    def test_delta_sq_norm_trigger_scores_from_shipped_delta(self):
        engine = make_engine(metric_v2="delta_sq_norm")
        params = _params("a", "b")
        deltas = {
            "a": np.array([3.0, 4.0], dtype=np.float32),   # ‖Δ‖² = 25
            "b": np.array([1.0, 0.0], dtype=np.float32),   # ‖Δ‖² = 1
        }
        engine.last_gradients = {  # must be ignored by the delta path
            "a": np.array([100.0], dtype=np.float32),
            "b": np.array([100.0], dtype=np.float32),
        }
        plan = engine._prepare_layer_dissemination(params, deltas, 0)
        assert plan.trigger_scores["a"] == pytest.approx(25.0)
        assert plan.trigger_scores["b"] == pytest.approx(1.0)
        # Single-metric arm: sched == trigger -> wire keeps the 0.0
        # "same as raw_score" convention.
        assert plan.sched_scores is None

    def test_raw_norm_control_keeps_last_gradient_path(self):
        """The control arm reproduces its native E2 configuration."""
        engine = make_engine(metric_v2="raw_norm")
        params = _params("a", "b")
        deltas = {  # must be ignored by the control path
            "a": np.array([9.0, 9.0], dtype=np.float32),
            "b": np.array([9.0, 9.0], dtype=np.float32),
        }
        engine.last_gradients = {
            "a": np.array([3.0, 4.0], dtype=np.float32),    # ‖g‖ = 5
            "b": np.array([0.5, 0.0], dtype=np.float32),    # ‖g‖ = 0.5
        }
        plan = engine._prepare_layer_dissemination(params, deltas, 0)
        assert plan.trigger_scores["a"] == pytest.approx(5.0)
        assert plan.trigger_scores["b"] == pytest.approx(0.5)
        assert plan.sched_scores is None

    def test_ages_increment_for_tail_and_reset_for_head(self):
        # LEGACY basis only (audit TRIG-1/ML-01): head placement is a sender
        # scheduling decision with no delivery guarantee, so the default
        # basis is receiver-acknowledged inclusion — covered in
        # tests/test_audit_t2_mechanisms.py.
        engine = make_engine(metric_v2="delta_sq_norm")
        engine.aging_age_basis = "head_placement"
        engine._layer_ages = {"a": 4, "b": 2, "c": 1}
        params = _params("a", "b", "c")
        deltas = {n: np.ones(2, dtype=np.float32) for n in params}

        # Force a known head/tail split via a stub strategy.
        class StubStrategy:
            def assign(self, **kwargs):
                self.kwargs = kwargs
                return AssignmentResult(
                    assignment={"a": 0, "b": 1},  # 'c' omitted this round
                    head={"a"},
                    tail={"b"},
                )

        stub = StubStrategy()
        engine._assignment_strategy = stub
        engine._prepare_layer_dissemination(params, deltas, 0)
        assert engine._layer_ages == {"a": 0, "b": 3, "c": 2}
        # Strategy receives the pre-update ages.
        assert stub.kwargs["ages"] == {"a": 4, "b": 2, "c": 1}


# ---------------------------------------------------------------------------
# End-to-end run loops (wiring smoke: per-layer worker + aggregator G5)
# ---------------------------------------------------------------------------

class ImmediateBarrierAlgorithm(RecordingAlgorithm):
    """Sync algorithm whose barrier releases immediately (no real peers)."""

    async def wait_for_aggregation(self, timeout=None):
        return True


def _tiny_dataset():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(32, 4)).astype(np.float32)
    y_train = rng.integers(0, 3, size=32).astype(np.int64)
    x_val = rng.normal(size=(8, 4)).astype(np.float32)
    y_val = rng.integers(0, 3, size=8).astype(np.int64)
    return x_train, y_train, x_val, y_val


class TestRunLoopsEndToEnd:

    async def test_run_sync_per_layer_full_wiring(self):
        """run_sync in per-layer mode: delta metrics -> plan -> manifest ->
        per-class sends -> _sender telemetry in the monitor report."""
        algorithm = ImmediateBarrierAlgorithm(neighbors=["node-1"])
        server = MagicMock(spec=TransportServer)
        pool = MagicMock(spec=ConnectionPool)
        sent: list[federation_pb2.Envelope] = []

        async def capture(dest, envelope, traffic_class=0):
            sent.append(envelope)
            return envelope.ByteSize()

        pool.send = capture
        config = {
            "learning_rate": 0.01, "optimizer": "sgd", "epochs_per_round": 1,
            "total_rounds": 2, "batch_size": 16,
            "update_mode": "per_layer", "num_traffic_classes": 3,
            "epsilon_deadline": 0.2, "epsilon_warmup_rounds": 0,
            "watchdog_factor": 3.0,
            "importance_metric_v2": "delta_sq_norm",
            "assignment_strategy": "gap_based",
            "late_layer_policy": "drop", "seed": 7, "sync_timeout": 0.05,
        }
        engine = TrainingEngine(
            node_id="node-0", model=_build_model(), algorithm=algorithm,
            server=server, pool=pool, config=config,
        )
        engine._register_message_handler()
        engine.monitor_ip = "10.0.0.254"
        engine.monitor_port = 5100

        history = await engine.run_sync(*_tiny_dataset())
        assert len(history) == 2
        assert all(h["layer_comm_metrics"] for h in history)

        manifests = [
            e for e in sent
            if e.WhichOneof("payload") == "round_manifest"
            and e.dest_node == "node-1"
        ]
        assert len(manifests) == 2  # one per round per destination
        # G2: trigger scores on the wire are the delta-sq-norm accounting;
        # single-metric arm keeps sched_score at the 0.0 default.
        for env in manifests:
            for entry in env.round_manifest.entries:
                assert entry.raw_score >= 0.0
                assert entry.sched_score == 0.0

        reports = [
            e.metrics_report for e in sent
            if e.WhichOneof("payload") == "metrics_report"
        ]
        assert len(reports) == 2
        telemetry = json.loads(reports[0].uplink_telemetry_json)
        # This worker received nothing — only the sender block and the
        # clock-domain legend may appear.
        assert set(telemetry) == {"_sender", "_clock_domains"}
        assert "predicted_t_eps_model_s" in telemetry["_sender"]
        assert telemetry["_sender"]["effective_epsilon"] == pytest.approx(0.2)
        assert telemetry["_clock_domains"]["oneway_negative_count"] == 0

    async def test_run_aggregator_per_layer_broadcast(self):
        """run_aggregator (G5): manifest-less, concurrent, byte-balanced
        downlink broadcast in per-layer mode."""
        algorithm = ImmediateBarrierAlgorithm(
            neighbors=["node-1", "node-2"], centralized=True,
        )
        server = MagicMock(spec=TransportServer)
        pool = MagicMock(spec=ConnectionPool)
        sent: list[federation_pb2.Envelope] = []

        async def capture(dest, envelope, traffic_class=0):
            sent.append(envelope)
            return envelope.ByteSize()

        pool.send = capture
        config = {
            "learning_rate": 0.01, "optimizer": "sgd", "epochs_per_round": 1,
            "total_rounds": 1, "batch_size": 16,
            "update_mode": "per_layer", "num_traffic_classes": 3,
            "epsilon_deadline": 0.2, "watchdog_factor": 3.0,
            "importance_metric_v2": "delta_sq_norm",
            "assignment_strategy": "gap_based",
            "late_layer_policy": "drop", "seed": 7, "sync_timeout": 0.05,
        }
        engine = TrainingEngine(
            node_id="node-0", model=_build_model(), algorithm=algorithm,
            server=server, pool=pool, config=config,
        )
        engine._register_message_handler()
        engine.outgoing_edges = [
            {
                "src": 0, "dst": dst,
                "classes": {
                    0: {"bandwidth_mbps": 6.0, "latency_ms": 0, "drop_rate": 0},
                    1: {"bandwidth_mbps": 3.0, "latency_ms": 0, "drop_rate": 0},
                    2: {"bandwidth_mbps": 1.0, "latency_ms": 0, "drop_rate": 0},
                },
            }
            for dst in (1, 2)
        ]

        history = await engine.run_aggregator(*_tiny_dataset())
        assert len(history) == 1
        assert engine._is_aggregator is True

        manifests = [
            e for e in sent if e.WhichOneof("payload") == "round_manifest"
        ]
        assert manifests == []  # downlink is manifest-less by design
        layers = [e for e in sent if e.WhichOneof("payload") == "layer_update"]
        per_dest: dict[str, set[str]] = {}
        for env in layers:
            per_dest.setdefault(env.dest_node, set()).add(
                env.layer_update.layer_name
            )
        model_layers = {
            var.path for var in engine.model.trainable_variables
        }
        # Workers need the full model: every layer reaches every worker.
        assert per_dest == {"node-1": model_layers, "node-2": model_layers}
        assert all(
            env.layer_update.total_layers == len(model_layers)
            for env in layers
        )


# ---------------------------------------------------------------------------
# SO_SNDBUF pinning (G1 fix 2)
# ---------------------------------------------------------------------------

class TestSndbufPinning:

    async def test_sndbuf_pinned_for_shaped_class(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(4)
        port = server_sock.getsockname()[1]
        pool = ConnectionPool("node-0", {0: 0})
        try:
            await pool.connect(
                "peer", "127.0.0.1", port,
                num_classes=1, max_retries=1,
                class_bandwidths_mbps={0: 1.0},  # 1 Mbps -> 31250 B
            )
            sock = pool._connections[("peer", 0)]
            effective = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
            requested = int(1.0 * 1e6 / 8 * 0.25)
            # Linux doubles the requested value; macOS returns it verbatim.
            assert requested <= effective <= 4 * requested
        finally:
            await pool.close_all()
            server_sock.close()

    def test_pin_is_noop_for_unshaped_class(self):
        """inf / None / non-positive bandwidths leave autotuning alone."""
        pool = ConnectionPool("node-0", {0: 0})
        for bandwidth in (None, float("inf"), 0.0, -1.0):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                before = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                pool._pin_sndbuf(sock, "peer", 0, bandwidth)
                after = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                assert after == before, f"bandwidth={bandwidth!r}"
            finally:
                sock.close()
