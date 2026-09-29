"""Tests for the T1 timing-instrumentation fixes (writeup/19 audit).

The audit's core finding is that the only "send time" the framework produced
was the return of an asyncio socket write — an event about the socket write
queue having room, not about bytes reaching the wire — and that the baseline
the system is compared against had no receiver-side clock at all.  These
tests pin the four repairs:

- **NT-01** — a shared-host-clock one-way wire time per layer: the sender
  stamps ``Envelope.t_send_start_sender_s`` immediately before the first byte
  goes to the socket, the receiver subtracts it from its own arrival stamp,
  and the result is exported per layer.
- **NT-01/BYTE-08 rename** — the buffer-accept quantity is
  ``send_enqueue_duration_sender_s`` everywhere; the old time-like name
  survives only as the frozen protobuf field.
- **NT-05** — receiver-side first/last-byte stamps per flow, so the
  monolithic ModelUpdate and the downlink broadcast (both manifest-less)
  finally have a wire-comparable completion time.
- **NT-03** — every exported time key carries its clock domain as a suffix.
- **NT-08** — the per-class SO_SNDBUF request vs grant is in the telemetry,
  so an asymmetric buffer discount is visible in the data.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time

import numpy as np
import pytest
import tensorflow as tf
from unittest.mock import AsyncMock, MagicMock

from src.algorithms.base import FederationAlgorithm, TrainingUpdate
from src.importance.manifest import build_manifest
from src.monitoring.collector import MetricsCollector
from src.network.connection_pool import ConnectionPool, _KERNEL_DOUBLES_SNDBUF
from src.network.framing import FrameReader, encode_message
from src.network.transport import TransportServer
from src.proto_gen import federation_pb2
from src.training.engine import TrainingEngine


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

class _NullAlgorithm(FederationAlgorithm):
    """Algorithm that only records what was delivered."""

    def __init__(self, node_id: str = "node-0"):
        super().__init__(node_id, ["node-1"], {})
        self.received: list[TrainingUpdate] = []

    @property
    def is_synchronous(self) -> bool:
        return True

    @property
    def is_centralized(self) -> bool:
        return False

    async def on_local_training_complete(self, model_params, round_num, num_samples):
        return []

    async def on_update_received(self, update: TrainingUpdate) -> None:
        self.received.append(update)

    def ready_to_aggregate(self) -> bool:
        return False

    async def aggregate(self, local_params):
        return local_params


def _make_engine(*, update_mode: str = "per_layer", epsilon: float = 0.5):
    model = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(4,))])
    model(tf.zeros((1, 4)))
    pool = MagicMock(spec=ConnectionPool)
    pool.send = AsyncMock(return_value=100)
    pool.sndbuf_telemetry = MagicMock(return_value=[])
    engine = TrainingEngine(
        node_id="node-0",
        model=model,
        algorithm=_NullAlgorithm(),
        server=MagicMock(spec=TransportServer),
        pool=pool,
        config={
            "learning_rate": 0.01, "optimizer": "sgd", "epochs_per_round": 1,
            "total_rounds": 3, "batch_size": 32, "update_mode": update_mode,
            "num_traffic_classes": 1, "epsilon_deadline": epsilon,
            "epsilon_warmup_rounds": 0, "watchdog_factor": 0.0,
            "importance_metric_v2": "delta_sq_norm",
            "assignment_strategy": "gap_based", "late_layer_policy": "drop",
            "seed": 7,
        },
    )
    engine._register_message_handler()
    return engine


def _handler(engine: TrainingEngine):
    return engine.server.set_handler.call_args[0][0]


def _observer(engine: TrainingEngine):
    return engine.server.set_arrival_observer.call_args[0][0]


def _layer_envelope(
    source: str, round_num: int, layer: str, index: int, total: int,
    t_send_start: float | None = None,
) -> federation_pb2.Envelope:
    array = np.full((4,), float(index), dtype=np.float32)
    envelope = federation_pb2.Envelope(
        source_node=source,
        dest_node="node-0",
        layer_update=federation_pb2.LayerUpdate(
            layer_name=layer, parameters=array.tobytes(), shape=[4],
            dtype="float32", layer_index=index, total_layers=total,
            num_samples=10, round=round_num,
        ),
    )
    if t_send_start is not None:
        envelope.t_send_start_sender_s = t_send_start
    return envelope


def _manifest_envelope(
    source: str, round_num: int, scores: dict[str, float],
    t_send_start: float | None = None,
) -> federation_pb2.Envelope:
    envelope = federation_pb2.Envelope(
        source_node=source,
        dest_node="node-0",
        round_manifest=build_manifest(
            round_num=round_num, source_node_id=source,
            importance_scores=scores,
        ),
    )
    if t_send_start is not None:
        envelope.t_send_start_sender_s = t_send_start
    return envelope


def _model_envelope(
    source: str, round_num: int, t_send_start: float | None = None,
) -> federation_pb2.Envelope:
    array = np.arange(4, dtype=np.float32)
    envelope = federation_pb2.Envelope(
        source_node=source,
        dest_node="node-0",
        model_update=federation_pb2.ModelUpdate(
            parameters=array.tobytes(), num_samples=10, round=round_num,
            layer_meta=[federation_pb2.LayerMeta(
                name="a", shape=[4], dtype="float32", offset=0, size=16,
            )],
        ),
    )
    if t_send_start is not None:
        envelope.t_send_start_sender_s = t_send_start
    return envelope


async def _deliver(engine: TrainingEngine, envelope, first: float, last: float):
    """Feed an envelope through observer + handler, as the transport does."""
    _observer(engine)(envelope, first, last)
    await _handler(engine)(envelope)


# ---------------------------------------------------------------------------
# NT-01: sender stamp -> one-way wire time
# ---------------------------------------------------------------------------

class TestSendStartStamp:

    async def test_pool_stamps_immediately_before_the_socket(self):
        """The stamp lands between the caller's build and the socket write."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(4)
        port = server_sock.getsockname()[1]
        pool = ConnectionPool("node-1", {0: 0})
        try:
            await pool.connect("peer", "127.0.0.1", port, num_classes=1,
                               max_retries=1)
            envelope = _layer_envelope("node-1", 0, "a", 0, 1)
            assert envelope.t_send_start_sender_s == 0.0  # proto3 "absent"
            before = time.monotonic()
            await pool.send("peer", envelope, traffic_class=0)
            after = time.monotonic()
            assert before <= envelope.t_send_start_sender_s <= after
        finally:
            await pool.close_all()
            server_sock.close()

    async def test_per_layer_wire_time_exported(self):
        engine = _make_engine(epsilon=0.0)
        base = time.monotonic()
        await _deliver(
            engine, _manifest_envelope("node-1", 0, {"a": 1.0, "b": 1.0},
                                       t_send_start=base),
            base + 0.01, base + 0.02,
        )
        await _deliver(
            engine, _layer_envelope("node-1", 0, "a", 0, 2,
                                    t_send_start=base + 0.10),
            base + 0.20, base + 0.30,
        )
        await _deliver(
            engine, _layer_envelope("node-1", 0, "b", 1, 2,
                                    t_send_start=base + 0.40),
            base + 0.50, base + 0.55,
        )
        engine._round_starts[0] = base
        flow = json.loads(engine._export_uplink_telemetry(0))["node-1"]

        wire = flow["layer_wire_times_oneway_s"]
        # Layer 'a': handed to the socket at +0.10, last byte read at +0.30.
        assert wire["a"] == pytest.approx(0.20, abs=1e-6)
        assert wire["b"] == pytest.approx(0.15, abs=1e-6)
        assert flow["manifest_wire_time_oneway_s"] == pytest.approx(0.02, abs=1e-6)
        assert flow["wire_clock_anomaly"] is False
        # Coverage on the wire clock: manifest send-start -> trigger fire.
        assert flow["t_cover_oneway_s"] == pytest.approx(0.55, abs=1e-6)
        # ...strictly longer than the receiver-local t_eps, which starts only
        # once the manifest has already crossed the wire.
        assert flow["t_cover_oneway_s"] > flow["t_eps_local_receiver_s"]

    async def test_unstamped_sender_yields_null_wire_times(self):
        """A pre-fix sender must produce nulls, never a stamp read as epoch."""
        engine = _make_engine(epsilon=0.0)
        now = time.monotonic()
        await _deliver(engine, _manifest_envelope("node-1", 0, {"a": 1.0}),
                       now, now)
        await _deliver(engine, _layer_envelope("node-1", 0, "a", 0, 1),
                       now, now)
        flow = json.loads(engine._export_uplink_telemetry(0))["node-1"]
        assert flow["manifest_wire_time_oneway_s"] is None
        assert flow["layer_wire_times_oneway_s"] == {}
        assert flow["t_cover_oneway_s"] is None
        assert flow["wire_clock_anomaly"] is False

    async def test_negative_one_way_is_counted_not_hidden(self):
        """Clocks that are not shared must be visible, not clipped."""
        engine = _make_engine(epsilon=0.0)
        base = time.monotonic()
        # Sender stamp in the receiver's future: impossible on one clock.
        await _deliver(
            engine, _manifest_envelope("node-1", 0, {"a": 1.0},
                                       t_send_start=base + 5.0),
            base, base,
        )
        await _deliver(
            engine, _layer_envelope("node-1", 0, "a", 0, 1,
                                    t_send_start=base + 5.0),
            base, base,
        )
        telemetry = json.loads(engine._export_uplink_telemetry(0))
        assert telemetry["node-1"]["wire_clock_anomaly"] is True
        assert telemetry["node-1"]["layer_wire_times_oneway_s"]["a"] < 0
        assert telemetry["_clock_domains"]["oneway_negative_count"] > 0

    async def test_round_trip_one_way_time_over_a_real_socket(self):
        """End-to-end on one host: pool.send -> transport -> observer."""
        observed: list[tuple[str, float, float, float]] = []

        def observer(envelope, first_byte, last_byte):
            observed.append((
                envelope.WhichOneof("payload"), first_byte, last_byte,
                envelope.t_send_start_sender_s,
            ))

        server = TransportServer("127.0.0.1", 0)
        server.set_handler(AsyncMock())
        server.set_arrival_observer(observer)
        await server.start()
        port = server._server_sock.getsockname()[1]
        pool = ConnectionPool("node-1", {0: 0})
        try:
            await pool.connect("peer", "127.0.0.1", port, num_classes=1,
                               max_retries=1)
            await pool.send("peer", _layer_envelope("node-1", 0, "a", 0, 1))
            for _ in range(200):
                if observed:
                    break
                await asyncio.sleep(0.005)
            assert observed, "envelope never arrived"
            payload, first_byte, last_byte, send_start = observed[0]
            assert payload == "layer_update"
            assert send_start > 0.0
            # Same host kernel clock: the one-way time is small and positive.
            wire_time = last_byte - send_start
            assert 0.0 < wire_time < 5.0
            assert first_byte <= last_byte
        finally:
            await pool.close_all()
            await server.stop()

    async def test_engine_exports_wire_times_from_a_real_transfer(self):
        """Whole chain: pool stamp -> TCP -> transport -> engine telemetry."""
        engine = _make_engine(epsilon=0.0)
        server = TransportServer("127.0.0.1", 0)
        server.set_handler(_handler(engine))
        server.set_arrival_observer(_observer(engine))
        await server.start()
        port = server._server_sock.getsockname()[1]
        pool = ConnectionPool("node-1", {0: 0})
        try:
            await pool.connect("peer", "127.0.0.1", port, num_classes=1,
                               max_retries=1)
            engine._round_starts[0] = time.monotonic()
            await pool.send("peer", _manifest_envelope("node-1", 0, {"a": 1.0}))
            await pool.send("peer", _layer_envelope("node-1", 0, "a", 0, 1))
            for _ in range(200):
                if engine.algorithm.received:
                    break
                await asyncio.sleep(0.005)
            assert engine.algorithm.received, "round never completed"

            flow = json.loads(engine._export_uplink_telemetry(0))["node-1"]
            assert flow["wire_clock_anomaly"] is False
            assert 0.0 < flow["layer_wire_times_oneway_s"]["a"] < 5.0
            assert 0.0 < flow["manifest_wire_time_oneway_s"] < 5.0
            # Coverage measured on the wire necessarily covers the manifest's
            # own crossing, so it exceeds the receiver-local interval.
            assert (
                flow["t_cover_oneway_s"]
                > flow["t_eps_local_receiver_s"]
            )
        finally:
            await pool.close_all()
            await server.stop()


# ---------------------------------------------------------------------------
# NT-05: receiver-side stamps for manifest-less flows
# ---------------------------------------------------------------------------

class TestReceiverFlows:

    async def test_monolithic_flow_gets_a_completion_time(self):
        engine = _make_engine(update_mode="monolithic")
        base = time.monotonic()
        engine._round_starts[0] = base
        await _deliver(
            engine, _model_envelope("node-1", 0, t_send_start=base + 0.10),
            base + 0.30, base + 0.90,
        )
        telemetry = json.loads(engine._export_uplink_telemetry(0))
        # No manifest, so no source-keyed flow — but a receiver clock exists.
        assert [k for k in telemetry if not k.startswith("_")] == []
        entry = telemetry["_receiver_flows"]["node-1|model_update"]
        assert entry["completion_wire_oneway_s"] == pytest.approx(0.80, abs=1e-6)
        assert entry["first_byte_wire_oneway_s"] == pytest.approx(0.20, abs=1e-6)
        assert entry["first_byte_rel_receiver_s"] == pytest.approx(0.30, abs=1e-6)
        assert entry["messages"] == 1
        assert entry["bytes"] > 0
        assert entry["wire_clock_anomaly"] is False

    async def test_downlink_broadcast_flow_is_measured(self):
        """The manifest-less broadcast is the largest transfer in the round."""
        engine = _make_engine(epsilon=0.0)
        base = time.monotonic()
        engine._round_starts[0] = base
        for index, (layer, sent, arrived) in enumerate([
            ("a", 0.10, 0.40), ("b", 0.45, 0.80),
        ]):
            await _deliver(
                engine,
                _layer_envelope("node-0", 0, layer, index, 2,
                                t_send_start=base + sent),
                base + arrived - 0.05, base + arrived,
            )
        telemetry = json.loads(engine._export_uplink_telemetry(0))
        entry = telemetry["_receiver_flows"]["node-0|layer_update"]
        # Flow spans the first byte handed over to the last byte read.
        assert entry["completion_wire_oneway_s"] == pytest.approx(0.70, abs=1e-6)
        assert entry["messages"] == 2

    async def test_flow_stamps_are_reaped_with_round_state(self):
        engine = _make_engine(update_mode="monolithic")
        now = time.monotonic()
        await _deliver(engine, _model_envelope("node-1", 0, t_send_start=now),
                       now, now)
        assert engine._flow_stamps
        engine._gc_round_state(current_round=5)
        assert engine._flow_stamps == {}


class TestTransportArrivalStamps:

    def test_first_byte_stamp_spans_partial_reads(self):
        """A frame split across reads keeps the stamp of its opening read."""
        server = TransportServer("127.0.0.1", 0)
        observed: list[tuple[float, float]] = []
        server.set_arrival_observer(
            lambda env, first, last: observed.append((first, last))
        )
        data = encode_message(_layer_envelope("node-1", 0, "a", 0, 1))
        reader = FrameReader()

        # Emulate _handle_connection's stamping over two reads.
        assert reader.feed(data[:10]) == []
        assert reader.has_partial_frame() is True
        envelopes = reader.feed(data[10:])
        assert len(envelopes) == 1
        assert reader.has_partial_frame() is False
        server._notify_arrival(envelopes[0], 1.0, 2.0)
        assert observed == [(1.0, 2.0)]

    def test_observer_exception_never_breaks_the_receive_path(self):
        server = TransportServer("127.0.0.1", 0)

        def boom(envelope, first, last):
            raise RuntimeError("telemetry bug")

        server.set_arrival_observer(boom)
        server._notify_arrival(_layer_envelope("node-1", 0, "a", 0, 1), 1.0, 2.0)


# ---------------------------------------------------------------------------
# NT-01 rename: the old time-like name must not survive
# ---------------------------------------------------------------------------

class TestEnqueueRename:

    async def test_per_layer_metrics_use_the_enqueue_name(self):
        engine = _make_engine(epsilon=0.0)
        head_done: asyncio.Future = asyncio.get_running_loop().create_future()
        await engine._send_class_queue(
            "node-1", 0, [_layer_envelope("node-0", 0, "a", 0, 1)], [],
            head_done, asyncio.Event(), 0,
        )
        _, metrics = head_done.result()
        assert "send_enqueue_duration_sender_s" in metrics[0]
        assert "send_duration_s" not in metrics[0]

    async def test_report_and_collector_carry_the_enqueue_name(self, tmp_path):
        engine = _make_engine(epsilon=0.0)
        engine.monitor_ip = "10.0.0.254"
        sent: list[federation_pb2.Envelope] = []

        async def capture(dest, envelope, traffic_class=0):
            sent.append(envelope)
            return envelope.ByteSize()

        engine.pool.send = capture
        await engine._send_metrics_to_monitor({
            "round": 0, "train_loss": 0.5, "train_accuracy": 0.8,
            "val_loss": 0.6, "val_accuracy": 0.7, "round_duration_s": 1.0,
            "train_duration_s": 0.5, "comm_duration_s": 0.2,
            "send_enqueue_duration_sender_s": 0.125,
            "barrier_wait_duration_s": 0.1, "aggregation_duration_s": 0.05,
            "layer_comm_metrics": [{
                "layer_name": "a",
                "send_enqueue_duration_sender_s": 0.02,
                "importance": 1.0, "traffic_class": 0, "bytes_sent": 64,
            }],
        })
        # Frozen proto field name carries the value (wire compatibility)...
        report = sent[0].metrics_report
        assert report.send_duration_s == pytest.approx(0.125)

        collector = MetricsCollector(
            host="127.0.0.1", port=0, logdir=str(tmp_path),
            num_nodes=1, total_rounds=3,
        )
        await collector._handle_message(sent[0])
        stored = collector.get_all_metrics()[0]
        # ...but nothing the collector stores is named as a wire time.
        assert stored["send_enqueue_duration_sender_s"] == pytest.approx(0.125)
        assert "send_duration_s" not in stored
        layer = stored["layer_comm_metrics"][0]
        assert layer["send_enqueue_duration_sender_s"] == pytest.approx(0.02)
        assert "send_duration_s" not in layer


# ---------------------------------------------------------------------------
# NT-03: clock-domain tagging
# ---------------------------------------------------------------------------

class TestClockDomainTags:

    async def test_every_exported_time_key_names_its_domain(self):
        engine = _make_engine(epsilon=0.0)
        base = time.monotonic()
        engine._round_starts[0] = base
        await _deliver(
            engine, _manifest_envelope("node-1", 0, {"a": 1.0},
                                       t_send_start=base),
            base + 0.01, base + 0.02,
        )
        await _deliver(
            engine, _layer_envelope("node-1", 0, "a", 0, 1,
                                    t_send_start=base + 0.05),
            base + 0.10, base + 0.20,
        )
        await _deliver(engine, _model_envelope("node-2", 0, t_send_start=base),
                       base + 0.1, base + 0.4)
        telemetry = json.loads(engine._export_uplink_telemetry(0))

        allowed = ("_receiver_s", "_oneway_s", "_sender_s", "_model_s")
        blocks = [telemetry["node-1"], telemetry["_receiver_flows"]["node-2|model_update"]]
        for block in blocks:
            untagged = [
                key for key in block
                if key.endswith("_s") and not key.endswith(allowed)
            ]
            assert untagged == [], untagged

        legend = telemetry["_clock_domains"]["suffixes"]
        assert set(legend) == {"_sender_s", "_receiver_s", "_oneway_s", "_model_s"}

    async def test_collector_exposes_domain_tagged_scalars(self, tmp_path):
        """The receiver-side and one-way KPIs stay on separate TB series."""
        telemetry = {
            "node-1": {
                "t_eps_local_receiver_s": 0.4,
                "t_cover_oneway_s": 0.6,
                "watchdog_fired": False,
            },
        }
        report = federation_pb2.MetricsReport(
            node_id="node-0", round=0,
            uplink_telemetry_json=json.dumps(telemetry),
        )
        envelope = federation_pb2.Envelope(
            source_node="node-0", dest_node="monitor", metrics_report=report,
        )
        collector = MetricsCollector(
            host="127.0.0.1", port=0, logdir=str(tmp_path),
            num_nodes=1, total_rounds=1,
        )
        writer = MagicMock()
        collector._writers["node-0"] = writer
        await collector._handle_message(envelope)

        scalars = writer.write_round_metrics.call_args[0][1]
        assert scalars["uplink/t_eps_mean_receiver_s"] == pytest.approx(0.4)
        assert scalars["uplink/t_cover_mean_oneway_s"] == pytest.approx(0.6)
        # The receiver-side and one-way KPIs are distinct series, so nothing
        # downstream can average them into one "time" (audit NT-03).
        assert "uplink/t_eps_mean_s" not in scalars
        stored = collector.get_all_metrics()[0]
        flow = stored["uplink_telemetry"]["node-1"]
        assert flow["t_eps_local_receiver_s"] == 0.4
        assert flow["t_cover_oneway_s"] == 0.6


# ---------------------------------------------------------------------------
# NT-08 / BYTE-08: SO_SNDBUF request vs grant in the data
# ---------------------------------------------------------------------------

class TestSndbufTelemetry:

    async def test_pinned_class_records_request_and_grant(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(4)
        port = server_sock.getsockname()[1]
        pool = ConnectionPool("node-0", {0: 0})
        try:
            await pool.connect(
                "peer", "127.0.0.1", port, num_classes=1, max_retries=1,
                class_bandwidths_mbps={0: 1.0},  # 1 Mbps -> 31250 B requested
            )
            records = pool.sndbuf_telemetry()
            assert len(records) == 1
            record = records[0]
            assert record["pinned"] is True
            assert record["requested_bytes"] == 31250
            assert record["expected_bytes"] == (
                62500 if _KERNEL_DOUBLES_SNDBUF else 31250
            )
            assert record["effective_bytes"] > 0
            # Buffered seconds of line rate: the quantity that must match
            # across classes for the enqueue clock to carry one bias.
            assert record["granted_line_s"] == pytest.approx(
                record["effective_bytes"] * 8.0 / 1e6
            )
            assert record["clamped"] is False
        finally:
            await pool.close_all()
            server_sock.close()

    def test_clamped_grant_is_flagged(self):
        """A kernel that grants less than asked must not do so silently."""
        pool = ConnectionPool("node-0", {0: 0})
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # 10 Gbps -> a 312.5 MB request no kernel will honour.
            pool._pin_sndbuf(sock, "peer", 0, 10_000.0)
        finally:
            sock.close()
        record = pool.sndbuf_telemetry()[0]
        assert record["clamped"] is True
        assert record["effective_bytes"] < record["expected_bytes"]

    def test_unshaped_class_is_still_recorded(self):
        pool = ConnectionPool("node-0", {0: 0})
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            pool._pin_sndbuf(sock, "peer", 1, None)
        finally:
            sock.close()
        record = pool.sndbuf_telemetry()[0]
        assert record["pinned"] is False
        assert record["requested_bytes"] is None
        assert record["effective_bytes"] > 0
        assert record["clamped"] is False

    async def test_sndbuf_reaches_the_sender_telemetry_block(self):
        engine = _make_engine(epsilon=0.0)
        engine.pool.sndbuf_telemetry = MagicMock(return_value=[
            {"neighbor": "node-1", "traffic_class": 0, "bandwidth_mbps": 6.0,
             "pinned": True, "requested_bytes": 187500,
             "expected_bytes": 375000, "effective_bytes": 375000,
             "granted_line_s": 0.5, "clamped": False},
        ])
        engine._assignment_info[0] = {
            "predicted_t_eps": 0.2, "epsilon": 0.0, "diagnostics": {},
        }
        block = json.loads(engine._export_uplink_telemetry(0))["_sender"]
        assert block["socket_sndbuf"][0]["granted_line_s"] == 0.5
        assert block["predicted_t_eps_model_s"] == 0.2

    async def test_broken_pool_telemetry_never_fails_a_round(self):
        engine = _make_engine(epsilon=0.0)
        engine.pool.sndbuf_telemetry = MagicMock(side_effect=RuntimeError("nope"))
        engine._assignment_info[0] = {
            "predicted_t_eps": 0.2, "epsilon": 0.0, "diagnostics": {},
        }
        block = json.loads(engine._export_uplink_telemetry(0))["_sender"]
        assert "socket_sndbuf" not in block


# ---------------------------------------------------------------------------
# Blast radius: the new reserved blocks must not confuse downstream readers
# ---------------------------------------------------------------------------

class TestDownstreamCompatibility:

    def test_role_detection_survives_the_new_blocks(self):
        """`socket_sndbuf` gives every node a `_sender` block, including the
        aggregator — role inference must still key off `_aggregation`."""
        from scripts import analysis_common as ac

        sndbuf = [{"neighbor": "node-1", "traffic_class": 0,
                   "granted_line_s": 0.5, "clamped": False}]
        nodes = {
            "node-0": {"uplink_telemetry": {
                "node-1": {"t_eps_local_receiver_s": 0.4},
                "_aggregation": {"aggregation_count": 1},
                "_receiver_flows": {"node-1|layer_update": {"messages": 2}},
                "_sender": {"socket_sndbuf": sndbuf},
                "_clock_domains": {"oneway_negative_count": 0},
            }},
            "node-1": {"uplink_telemetry": {
                "_sender": {"socket_sndbuf": sndbuf},
                "_receiver_flows": {"node-0|layer_update": {"messages": 14}},
                "_clock_domains": {"oneway_negative_count": 0},
            }},
        }
        assert ac.split_roles(nodes) == ("node-0", {"node-1"})

    def test_receiver_flow_keys_are_not_mistaken_for_senders(self):
        """`_receiver_flows` keys look like node ids but are nested."""
        from scripts.overnight_common import uplink_observations

        report = {"per_round": [{"round": 0, "nodes": {"node-0": {
            "uplink_telemetry": {
                "node-1": {
                    "t_eps_local_receiver_s": 0.4,
                    "manifest_arrival_rel_receiver_s": 0.01,
                    "layer_arrivals_rel_receiver_s": {"a": 0.3},
                    "t_cover_oneway_s": 0.6,
                    "layer_wire_times_oneway_s": {"a": 0.25},
                },
                "_receiver_flows": {"node-2|model_update": {"messages": 1}},
                "_clock_domains": {"oneway_negative_count": 0},
            },
        }}}]}
        observations = uplink_observations("run-x", report, "node-0")
        assert [o.source for o in observations] == ["node-1"]
        assert observations[0].t_cover_oneway_s == 0.6
        assert observations[0].layer_wire_times_oneway_s == {"a": 0.25}
