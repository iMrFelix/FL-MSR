"""TensorBoard writer for federated learning metrics.

Writes per-node metrics to a TensorBoard logdir with one run per node.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import tensorflow as tf

logger = logging.getLogger(__name__)


class TensorBoardWriter:
    """Writes metrics to TensorBoard summary files."""

    def __init__(self, logdir: str, node_id: str):
        self._logdir = Path(logdir) / node_id
        self._logdir.mkdir(parents=True, exist_ok=True)
        self._writer = tf.summary.create_file_writer(str(self._logdir))
        logger.info(f"TensorBoard writer initialized: {self._logdir}")

    def write_round_metrics(self, round_num: int, metrics: dict[str, float]) -> None:
        """Write metrics for a single round."""
        with self._writer.as_default():
            for key, value in metrics.items():
                tf.summary.scalar(key, value, step=round_num)
            self._writer.flush()

    def close(self) -> None:
        """Flush and close the writer."""
        self._writer.flush()
        self._writer.close()
