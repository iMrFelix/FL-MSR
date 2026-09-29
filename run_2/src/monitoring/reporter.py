"""Per-node metric reporter.

Sends MetricsReport protobuf messages from a node to the central monitor.

Note: In Stages 1-2, the reporter functionality is integrated directly
into TrainingEngine._send_metrics_to_monitor().  This standalone module
exists for future flexibility — e.g. a background metrics sender that
doesn't block the training loop, or a reporter that batches multiple
rounds before sending.
"""

from __future__ import annotations

import logging
import time

from src.network.connection_pool import ConnectionPool
from src.proto_gen import federation_pb2

logger = logging.getLogger(__name__)


async def send_metrics_report(
    pool: ConnectionPool,
    node_id: str,
    round_num: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
    round_duration_s: float,
    train_duration_s: float,
    comm_duration_s: float,
    aggregation_duration_s: float,
    learning_rate: float,
) -> None:
    """Send a metrics report to the monitor via the connection pool.

    The monitor connection is always on traffic class 0 since metrics
    don't need QoS differentiation — they're small messages that don't
    compete with model updates for bandwidth.
    """
    report = federation_pb2.MetricsReport(
        node_id=node_id,
        round=round_num,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        round_duration_s=round_duration_s,
        train_duration_s=train_duration_s,
        comm_duration_s=comm_duration_s,
        aggregation_duration_s=aggregation_duration_s,
        learning_rate=learning_rate,
    )
    envelope = federation_pb2.Envelope(
        source_node=node_id,
        dest_node="monitor",
        timestamp_ns=time.monotonic_ns(),
        metrics_report=report,
    )
    try:
        await pool.send("monitor", envelope, traffic_class=0)
    except (ConnectionError, OSError) as e:
        logger.warning(f"[{node_id}] Failed to send metrics: {e}")
