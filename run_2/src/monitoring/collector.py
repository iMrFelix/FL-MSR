"""Central metrics collector — runs in the monitor container.

Receives MetricsReport protobuf messages from all nodes, writes them to
TensorBoard, and stores them for the final JSON report.

In per-layer mode (Stage 3+), the MetricsReport includes per-layer
communication metrics (``LayerCommMetric`` repeated field) with bytes
sent, enqueue duration, importance score, and traffic class for each layer.
The collector aggregates these by traffic class to produce distribution
scalars (e.g. ``traffic/class_0_bytes``, ``traffic/class_0_fraction``)
that are written to TensorBoard alongside the standard training metrics.

Clock domains (audit NT-01/NT-03): the protobuf field ``send_duration_s``
keeps its name and number for wire compatibility, but it holds a
KERNEL-ACCEPT duration in the sender's clock domain, so everything this
module emits names it ``send_enqueue_duration_sender_s``.  Receiver-side and
one-way quantities carry ``_receiver_s`` / ``_oneway_s`` suffixes; two keys
with different suffixes must never be placed on one axis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.monitoring.tensorboard_writer import TensorBoardWriter
from src.network.transport import TransportServer
from src.proto_gen import federation_pb2

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects metrics from all federation nodes."""

    def __init__(
        self,
        host: str,
        port: int,
        logdir: str,
        num_nodes: int,
        total_rounds: int,
    ):
        self._host = host
        self._port = port
        self._logdir = logdir
        self._num_nodes = num_nodes
        self._total_rounds = total_rounds
        self._server = TransportServer(host, port)
        self._writers: dict[str, TensorBoardWriter] = {}
        self._all_metrics: list[dict[str, Any]] = []
        self._rounds_completed: dict[str, int] = {}  # node_id -> max round received
        self._done_event = asyncio.Event()

    async def start(self) -> None:
        """Start the collector server."""
        self._server.set_handler(self._handle_message)
        await self._server.start()
        logger.info(f"Metrics collector listening on {self._host}:{self._port}")

    async def _handle_message(self, envelope: federation_pb2.Envelope) -> None:
        """Process incoming metrics reports."""
        payload_type = envelope.WhichOneof("payload")
        if payload_type != "metrics_report":
            return

        report = envelope.metrics_report
        node_id = report.node_id

        # Get or create TensorBoard writer for this node
        if node_id not in self._writers:
            self._writers[node_id] = TensorBoardWriter(self._logdir, node_id)

        # Write to TensorBoard
        tb_metrics = {
            "train/loss": report.train_loss,
            "train/accuracy": report.train_accuracy,
            "val/loss": report.val_loss,
            "val/accuracy": report.val_accuracy,
            "timing/round_duration_s": report.round_duration_s,
            "timing/train_duration_s": report.train_duration_s,
            "timing/comm_duration_s": report.comm_duration_s,
            "timing/send_enqueue_duration_sender_s": report.send_duration_s,
            "timing/barrier_wait_duration_s": report.barrier_wait_duration_s,
            "timing/aggregation_duration_s": report.aggregation_duration_s,
            "hyperparams/learning_rate": report.learning_rate,
        }

        # Per-class traffic distribution (per-layer mode only).
        # Aggregate bytes sent and enqueue duration by traffic class to show
        # how traffic is distributed across classes.  These TensorBoard
        # scalars are useful for verifying that tc/netem rules are working
        # (class 0 = high priority, class 1 = lower priority).
        if report.layer_comm_metrics:
            class_bytes: dict[int, int] = {}
            class_enqueue_time: dict[int, float] = {}
            total_layer_bytes = 0
            for lm in report.layer_comm_metrics:
                tc = lm.traffic_class
                class_bytes[tc] = class_bytes.get(tc, 0) + lm.bytes_sent
                class_enqueue_time[tc] = (
                    class_enqueue_time.get(tc, 0.0) + lm.send_duration_s
                )
                total_layer_bytes += lm.bytes_sent

            for tc, byte_count in sorted(class_bytes.items()):
                tb_metrics[f"traffic/class_{tc}_bytes"] = float(byte_count)
                tb_metrics[
                    f"traffic/class_{tc}_send_enqueue_duration_sender_s"
                ] = class_enqueue_time[tc]
                if total_layer_bytes > 0:
                    tb_metrics[f"traffic/class_{tc}_fraction"] = (
                        float(byte_count) / float(total_layer_bytes)
                    )

        # Receiver-side uplink telemetry (gate ruling G1).  The JSON sidecar
        # is parsed once here so report.json carries structured data, and a
        # handful of TensorBoard scalars summarize the primary KPI for live
        # supervision.  Schema-light by design: parse defensively, never let
        # a malformed sidecar drop the whole report.
        uplink_telemetry: dict[str, Any] | None = None
        if report.uplink_telemetry_json:
            try:
                uplink_telemetry = json.loads(report.uplink_telemetry_json)
            except json.JSONDecodeError as exc:
                logger.warning(
                    f"Malformed uplink_telemetry_json from {node_id} "
                    f"round {report.round}: {exc}"
                )
        if isinstance(uplink_telemetry, dict):
            # Keys starting with "_" are reserved bookkeeping blocks
            # (e.g. "_sender", "_receiver_flows"), not source-flow entries.
            def _flow_times(key: str) -> list[float]:
                return [
                    flow[key]
                    for source, flow in uplink_telemetry.items()
                    if not source.startswith("_")
                    and isinstance(flow, dict)
                    and isinstance(flow.get(key), (int, float))
                ]

            t_eps_values = _flow_times("t_eps_local_receiver_s")
            if t_eps_values:
                tb_metrics["uplink/t_eps_mean_receiver_s"] = (
                    sum(t_eps_values) / len(t_eps_values)
                )
                tb_metrics["uplink/t_eps_max_receiver_s"] = max(t_eps_values)
            # Coverage completion on the wire clock (audit NT-01/NT-05): the
            # one that may be compared against a monolithic flow's completion
            # stamp, since both span the same sender->receiver pair.
            t_cover_values = _flow_times("t_cover_oneway_s")
            if t_cover_values:
                tb_metrics["uplink/t_cover_mean_oneway_s"] = (
                    sum(t_cover_values) / len(t_cover_values)
                )
                tb_metrics["uplink/t_cover_max_oneway_s"] = max(t_cover_values)
            watchdog_count = sum(
                1
                for source, flow in uplink_telemetry.items()
                if not source.startswith("_")
                and isinstance(flow, dict)
                and flow.get("watchdog_fired")
            )
            if watchdog_count:
                tb_metrics["uplink/watchdog_fired_count"] = float(
                    watchdog_count
                )

        self._writers[node_id].write_round_metrics(report.round, tb_metrics)

        # Store for final report
        stored_entry: dict[str, Any] = {
            "node_id": node_id,
            "round": report.round,
            "train_loss": report.train_loss,
            "train_accuracy": report.train_accuracy,
            "val_loss": report.val_loss,
            "val_accuracy": report.val_accuracy,
            "round_duration_s": report.round_duration_s,
            "train_duration_s": report.train_duration_s,
            "comm_duration_s": report.comm_duration_s,
            "send_enqueue_duration_sender_s": report.send_duration_s,
            "barrier_wait_duration_s": report.barrier_wait_duration_s,
            "aggregation_duration_s": report.aggregation_duration_s,
            "learning_rate": report.learning_rate,
        }

        # Divergence marker (audit ML-04).  Stored only when set, so the key
        # doubles as the gate: `entry.get("diverged")` is falsy for every
        # healthy round and for pre-fix reports alike.  A diverged round's
        # val_accuracy is NOT data — NaN logits collapse it to chance level
        # instead of to a missing value — so analyses must exclude it from
        # endpoint reads and from byte statistics.
        if report.diverged:
            stored_entry["diverged"] = True
            stored_entry["diverged_reason"] = report.diverged_reason
            logger.error(
                f"DIVERGED: {node_id} round {report.round} "
                f"({report.diverged_reason}) — round excluded from valid data"
            )

        # Include per-layer metrics in stored data for the final JSON
        # report.  This gives full visibility into which layers were sent
        # on which traffic class with what timing per round.
        if report.layer_comm_metrics:
            stored_entry["layer_comm_metrics"] = [
                {
                    "layer_name": lm.layer_name,
                    "send_enqueue_duration_sender_s": lm.send_duration_s,
                    "importance": lm.importance,
                    "traffic_class": lm.traffic_class,
                    "bytes_sent": lm.bytes_sent,
                }
                for lm in report.layer_comm_metrics
            ]

        # Store the parsed uplink telemetry (per-source t_eps_local_receiver_s
        # and one-way wire times, shed sets, kappa_realized, watchdog/ordering
        # flags + the "_sender" / "_aggregation" / "_receiver_flows" /
        # "_clock_domains" bookkeeping blocks) into the per-round entry — the
        # analysis scripts' primary Tier-1 input.  Absent/empty/malformed
        # sidecars simply leave the key out.
        if isinstance(uplink_telemetry, dict) and uplink_telemetry:
            stored_entry["uplink_telemetry"] = uplink_telemetry
            # Hoist cancelled zombie-tail bytes (pre-run fix 3, the RQ4
            # $-input) out of the "_sender" block into the documented
            # per-entry scalar so the post-hoc cost scripts find them under
            # their published key.  prior_round_events carry cancellations
            # whose own round's report had already shipped; for run-level
            # cost totals the round attribution is immaterial, so they sum
            # into this entry's scalar (per-round re-attribution can still
            # be done from the events themselves).
            sender_block = uplink_telemetry.get("_sender")
            if isinstance(sender_block, dict):
                cancelled = sender_block.get("tail_cancelled_bytes", 0)
                cancelled = int(cancelled) if isinstance(
                    cancelled, (int, float)
                ) else 0
                for event in sender_block.get("prior_round_events") or []:
                    if (
                        isinstance(event, dict)
                        and event.get("type") == "tail_cancelled"
                        and isinstance(event.get("bytes"), (int, float))
                    ):
                        cancelled += int(event["bytes"])
                if cancelled > 0:
                    stored_entry["cancelled_bytes"] = cancelled

        self._all_metrics.append(stored_entry)

        # Track completion
        self._rounds_completed[node_id] = max(
            self._rounds_completed.get(node_id, -1), report.round
        )

        logger.debug(
            f"Received metrics from {node_id} round {report.round}: "
            f"val_acc={report.val_accuracy:.4f}"
        )

        # Check if all nodes have completed all rounds
        if len(self._rounds_completed) >= self._num_nodes:
            if all(
                r >= self._total_rounds - 1
                for r in self._rounds_completed.values()
            ):
                logger.info("All nodes completed training")
                self._done_event.set()

    async def wait_for_completion(self, timeout: float = 3600.0) -> bool:
        """Wait for all nodes to finish. Returns True if done, False if timed out."""
        try:
            await asyncio.wait_for(self._done_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for all nodes to complete")
            return False

    def get_all_metrics(self) -> list[dict[str, Any]]:
        """Return all collected metrics."""
        return self._all_metrics

    async def stop(self) -> None:
        """Stop the collector and close writers."""
        for writer in self._writers.values():
            writer.close()
        await self._server.stop()
