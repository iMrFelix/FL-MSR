"""D-PSGD (Decentralized Parallel Stochastic Gradient Descent).

Synchronous, decentralized algorithm where each node:
1. Trains locally for E epochs
2. Sends its model parameters to all neighbors
3. Waits for all neighbor updates
4. Averages own + all neighbor parameters (equal weight)
5. Sets model to the averaged parameters

Reference: Lian et al., "Can Decentralized Algorithms Outperform Centralized Algorithms?", NeurIPS 2017.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from src.algorithms.base import FederationAlgorithm, TrainingUpdate

logger = logging.getLogger(__name__)


class DPSGD(FederationAlgorithm):
    """Decentralized Parallel SGD — synchronous, decentralized."""

    def __init__(self, node_id: str, neighbors: list[str], config: dict):
        super().__init__(node_id, neighbors, config)
        # Buffer for received updates: source_node -> TrainingUpdate
        self._update_buffer: dict[str, TrainingUpdate] = {}
        self._timeout: float = config.get("sync_timeout", 120.0)

    @property
    def is_synchronous(self) -> bool:
        return True

    @property
    def is_centralized(self) -> bool:
        return False

    async def on_local_training_complete(
        self,
        model_params: dict[str, np.ndarray],
        round_num: int,
        num_samples: int,
    ) -> list[tuple[str, TrainingUpdate]]:
        """Send model parameters to all neighbors."""
        update = TrainingUpdate(
            source_node=self.node_id,
            round_num=round_num,
            parameters=model_params,
            num_samples=num_samples,
        )
        return [(neighbor, update) for neighbor in self.neighbors]

    async def on_update_received(self, update: TrainingUpdate) -> None:
        """Store received update. Signal when all neighbors have been received."""
        if update.source_node not in self.neighbors:
            logger.warning(
                f"[{self.node_id}] Received update from non-neighbor "
                f"{update.source_node}, ignoring"
            )
            return

        if update.round_num != self._round:
            logger.warning(
                f"[{self.node_id}] Received update for round {update.round_num} "
                f"but current round is {self._round}, ignoring"
            )
            return

        self._update_buffer[update.source_node] = update
        logger.debug(
            f"[{self.node_id}] Received update from {update.source_node} "
            f"(round {update.round_num}). "
            f"Buffer: {len(self._update_buffer)}/{len(self.neighbors)}"
        )

        if self.ready_to_aggregate():
            self._aggregation_event.set()

    def ready_to_aggregate(self) -> bool:
        """Ready when all neighbors have sent their updates for this round."""
        return all(n in self._update_buffer for n in self.neighbors)

    async def aggregate(
        self, local_params: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Equal-weight average of own parameters + all neighbor parameters.

        D-PSGD consensus step: x_i^{t+1} = (1/|N_i|+1) * (x_i^t + sum_{j in N_i} x_j^t)
        where N_i is the set of neighbors of node i.
        """
        all_params = [local_params] + [
            update.parameters for update in self._update_buffer.values()
        ]
        num_models = len(all_params)

        aggregated: dict[str, np.ndarray] = {}
        for layer_name in local_params:
            stacked = np.stack([p[layer_name] for p in all_params], axis=0)
            aggregated[layer_name] = np.mean(stacked, axis=0)

        # Clear buffer for next round
        self._update_buffer.clear()

        logger.info(
            f"[{self.node_id}] Aggregated {num_models} models (round {self._round})"
        )
        return aggregated
