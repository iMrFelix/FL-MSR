"""Gossip-SGD (Stochastic Gossip Decentralized SGD).

Asynchronous, decentralized gossip protocol where each node:
1. Trains locally for E epochs
2. Randomly selects ONE neighbor and sends its model parameters
3. Aggregates whenever one or more updates have arrived from peers
4. Mixes local and received parameters via pairwise averaging

The key difference from A-DPSGD is the communication pattern: instead
of broadcasting to all neighbors (which scales linearly with degree),
Gossip-SGD sends to exactly one randomly chosen neighbor per iteration.
This reduces communication volume at the cost of slower information
diffusion through the topology.

The aggregation rule is identical to A-DPSGD:
    new_params = 0.5 * local_params + 0.5 * mean(buffered_updates)

Neighbor selection uses a dedicated ``random.Random`` instance seeded
from the config so that the gossip pattern is reproducible.

Reference: Blot et al., "Gossip-based Distributed Stochastic Gradient
Descent", ICML Workshop 2016; Daily et al., "GossipGraD: Scalable Deep
Learning using Gossip Communication based Asynchronous Gradient
Descent", 2018.
"""

from __future__ import annotations

import logging
import random

import numpy as np

from src.algorithms.base import FederationAlgorithm, TrainingUpdate

logger = logging.getLogger(__name__)


class GossipSGD(FederationAlgorithm):
    """Gossip-SGD — async, decentralized, stochastic neighbor selection.

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
        - ``seed`` (int, default 42): Seed for the RNG that picks
          which neighbor to gossip with each iteration.
    """

    def __init__(self, node_id: str, neighbors: list[str], config: dict):
        super().__init__(node_id, neighbors, config)
        # Buffer for received updates — list, not keyed by source,
        # because in async mode the same neighbor could send multiple
        # updates between our aggregation steps.
        self._update_buffer: list[TrainingUpdate] = []

        # Staleness threshold: 0 = accept everything.
        self._staleness_threshold: int = config.get("staleness_threshold", 0)

        # Dedicated RNG for reproducible neighbor selection.  Using a
        # separate instance avoids interference with the global RNG
        # (which is seeded for model init reproducibility).  We mix
        # the node_id into the seed so that different nodes get
        # independent gossip patterns — otherwise all nodes would
        # select the same neighbor index each round, creating
        # degenerate communication patterns.
        base_seed = config.get("seed", 42)
        gossip_seed = hash((base_seed, node_id))
        self._rng = random.Random(gossip_seed)

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
        """Send model parameters to ONE randomly chosen neighbor.

        The neighbor is selected uniformly at random from the topology
        neighbors using the seeded RNG.  This stochastic communication
        pattern is the defining characteristic of gossip protocols.

        Returns an empty list if there are no neighbors (isolated node).
        """
        if not self.neighbors:
            return []
        target = self._rng.choice(self.neighbors)
        update = TrainingUpdate(
            source_node=self.node_id,
            round_num=round_num,
            parameters=model_params,
            num_samples=num_samples,
        )
        logger.debug(
            f"[{self.node_id}] Gossip: sending to {target} (iter {round_num})"
        )
        return [(target, update)]

    async def on_update_received(self, update: TrainingUpdate) -> None:
        """Buffer a received update, applying staleness filtering.

        Same acceptance logic as A-DPSGD: non-neighbor updates are
        rejected, and updates exceeding the staleness threshold are
        discarded.  Any accepted update signals aggregation readiness.
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
                    f"[{self.node_id}] Discarding stale gossip update from "
                    f"{update.source_node} (their iter={update.round_num}, "
                    f"ours={self._round}, threshold={self._staleness_threshold})"
                )
                return

        self._update_buffer.append(update)
        logger.debug(
            f"[{self.node_id}] Buffered gossip update from {update.source_node} "
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

        Identical to A-DPSGD aggregation.  When multiple updates have
        accumulated (e.g. two neighbors happened to gossip to us in quick
        succession), they are averaged uniformly first, then blended
        50/50 with the local model.
        """
        if not self._update_buffer:
            # No updates to aggregate — return local params unchanged.
            return local_params

        # Snapshot and clear the buffer atomically (see ADPSGD.aggregate
        # for rationale on the defensive copy).
        updates = list(self._update_buffer)
        self._update_buffer.clear()
        num_updates = len(updates)

        # Average the buffered updates (uniform weight).
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
            f"[{self.node_id}] Gossip aggregation: mixed local with "
            f"{num_updates} buffered update(s) (iter {self._round})"
        )
        return aggregated
