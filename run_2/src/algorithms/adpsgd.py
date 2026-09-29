"""A-DPSGD (Asynchronous Decentralized Parallel Stochastic Gradient Descent).

Asynchronous, decentralized algorithm where each node:
1. Trains locally for E epochs
2. Sends its model parameters to all neighbors immediately (no barrier)
3. Aggregates whenever one or more neighbor updates have arrived
4. Mixes local and received parameters via pairwise averaging

Unlike synchronous D-PSGD, nodes do NOT wait for all neighbors before
aggregating.  Any received update (subject to an optional staleness
threshold) triggers aggregation.  This lets faster nodes proceed without
being blocked by slower ones.

The aggregation rule is:
    new_params = 0.5 * local_params + 0.5 * mean(buffered_updates)

When multiple updates are buffered, they are averaged first before
mixing with the local model.

Reference: Lian et al., "Asynchronous Decentralized Parallel
Stochastic Gradient Descent", ICML 2018.
"""

from __future__ import annotations

import logging

import numpy as np

from src.algorithms.base import FederationAlgorithm, TrainingUpdate

logger = logging.getLogger(__name__)


class ADPSGD(FederationAlgorithm):
    """Asynchronous Decentralized Parallel SGD — async, decentralized.

    Parameters
    ----------
    node_id : str
        Unique identifier for this node.
    neighbors : list[str]
        IDs of neighboring nodes in the topology.
    config : dict
        Training configuration dict.  Reads:
        - ``staleness_threshold`` (int, default 0): Maximum acceptable
          iteration gap between the sender's iteration and ours.
          0 means no staleness filtering (accept all updates).
    """

    def __init__(self, node_id: str, neighbors: list[str], config: dict):
        super().__init__(node_id, neighbors, config)
        # Buffer for received updates: list of TrainingUpdate objects.
        # Unlike sync DPSGD which keys by source (one per neighbor per
        # round), async mode can accumulate multiple updates between
        # aggregation steps.
        self._update_buffer: list[TrainingUpdate] = []

        # Staleness threshold: discard updates whose iteration is more
        # than this many steps behind ours.  0 = accept everything.
        self._staleness_threshold: int = config.get("staleness_threshold", 0)

    @property
    def is_synchronous(self) -> bool:
        return False

    @property
    def is_centralized(self) -> bool:
        return False

    async def on_local_training_complete(
        self,
        model_params: dict[str, np.ndarray],
        round_num: int,
        num_samples: int,
    ) -> list[tuple[str, TrainingUpdate]]:
        """Send model parameters to all neighbors (same as sync DPSGD).

        In async mode there's no barrier — we send immediately and continue
        to the next local training iteration.
        """
        update = TrainingUpdate(
            source_node=self.node_id,
            round_num=round_num,
            parameters=model_params,
            num_samples=num_samples,
        )
        return [(neighbor, update) for neighbor in self.neighbors]

    async def on_update_received(self, update: TrainingUpdate) -> None:
        """Buffer a received update, applying staleness filtering.

        An update is accepted if:
        - It comes from a known neighbor.
        - It passes the staleness check (if threshold > 0): the sender's
          iteration must be within ``staleness_threshold`` of our own.

        Any accepted update immediately signals that aggregation is ready.
        """
        if update.source_node not in self.neighbors:
            logger.warning(
                f"[{self.node_id}] Received update from non-neighbor "
                f"{update.source_node}, ignoring"
            )
            return

        # Staleness check: reject updates that are too old.
        if self._staleness_threshold > 0:
            staleness = abs(self._round - update.round_num)
            if staleness > self._staleness_threshold:
                logger.debug(
                    f"[{self.node_id}] Discarding stale update from "
                    f"{update.source_node} (their iter={update.round_num}, "
                    f"ours={self._round}, threshold={self._staleness_threshold})"
                )
                return

        self._update_buffer.append(update)
        logger.debug(
            f"[{self.node_id}] Buffered async update from {update.source_node} "
            f"(their iter={update.round_num}, ours={self._round}). "
            f"Buffer size: {len(self._update_buffer)}"
        )

        # Signal that we have at least one update ready for aggregation.
        if self.ready_to_aggregate():
            self._aggregation_event.set()

    def ready_to_aggregate(self) -> bool:
        """Ready as soon as any update is in the buffer."""
        return len(self._update_buffer) > 0

    async def aggregate(
        self, local_params: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Pairwise average: 50% local + 50% mean of buffered updates.

        If multiple updates have accumulated since the last aggregation,
        they are averaged first (uniform weight), then blended with the
        local parameters at a 50/50 ratio.

        This is the standard A-DPSGD mixing rule: each node gradually
        converges toward the average of its neighborhood while still
        retaining half of its own model.
        """
        if not self._update_buffer:
            # No updates to aggregate — return local params unchanged.
            return local_params

        # Snapshot and clear the buffer atomically.  Although the asyncio
        # event loop is single-threaded, on_update_received can run between
        # any two await points.  Taking a snapshot ensures aggregate() is
        # safe even if an await is added in the future.
        updates = list(self._update_buffer)
        self._update_buffer.clear()
        num_updates = len(updates)

        # Average the buffered updates (uniform weight across senders).
        neighbor_avg: dict[str, np.ndarray] = {}
        for layer_name in local_params:
            stacked = np.stack(
                [u.parameters[layer_name] for u in updates],
                axis=0,
            )
            neighbor_avg[layer_name] = np.mean(stacked, axis=0)

        # 50/50 mix of local model and neighbor average.
        aggregated: dict[str, np.ndarray] = {}
        for layer_name in local_params:
            aggregated[layer_name] = (
                0.5 * local_params[layer_name] + 0.5 * neighbor_avg[layer_name]
            )

        logger.info(
            f"[{self.node_id}] Async aggregation: mixed local with "
            f"{num_updates} buffered update(s) (iter {self._round})"
        )
        return aggregated
