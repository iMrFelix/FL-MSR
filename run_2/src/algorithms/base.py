"""Base class for all federation algorithms.

The training engine calls these methods at the appropriate points.
The algorithm decides WHEN to send, WHAT to send, and HOW to aggregate.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrainingUpdate:
    """A training update from a single node."""

    source_node: str
    round_num: int
    parameters: dict[str, np.ndarray]  # layer_name -> weights
    num_samples: int


class FederationAlgorithm(ABC):
    """Base class for all federation algorithms.

    Concrete implementations must define:
    - is_synchronous / is_centralized properties
    - on_local_training_complete: what to send after training
    - on_update_received: how to store incoming updates
    - ready_to_aggregate: when aggregation conditions are met
    - aggregate: how to combine updates
    """

    def __init__(self, node_id: str, neighbors: list[str], config: dict):
        self.node_id = node_id
        self.neighbors = neighbors
        self.config = config
        self._round: int = 0
        self._aggregation_event = asyncio.Event()

    @property
    @abstractmethod
    def is_synchronous(self) -> bool:
        """Whether this algorithm operates in synchronous rounds."""
        ...

    @property
    @abstractmethod
    def is_centralized(self) -> bool:
        """Whether this algorithm uses a central aggregator."""
        ...

    @abstractmethod
    async def on_local_training_complete(
        self,
        model_params: dict[str, np.ndarray],
        round_num: int,
        num_samples: int,
    ) -> list[tuple[str, TrainingUpdate]]:
        """Called after local training finishes.

        Returns a list of (destination_node, update) pairs to be sent.
        For DPSGD: returns [(neighbor, update) for each neighbor].
        For FedAvg workers: returns [(aggregator, update)].
        For FedAvg aggregator: returns [] (waits for workers).
        """
        ...

    @abstractmethod
    async def on_update_received(self, update: TrainingUpdate) -> None:
        """Called when an update arrives from another node.
        The algorithm stores it internally until aggregation conditions are met.
        """
        ...

    @abstractmethod
    def ready_to_aggregate(self) -> bool:
        """Returns True when enough updates have been received to aggregate.

        For sync DPSGD: True when all neighbor updates for the current round arrived.
        For async algorithms: may always return True, or use staleness thresholds.
        """
        ...

    @abstractmethod
    async def aggregate(
        self, local_params: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Perform aggregation using stored updates + local parameters.

        Returns the new model parameters.
        Clears internal update buffer after aggregation.
        """
        ...

    def get_round(self) -> int:
        """Current round number."""
        return self._round

    def advance_round(self) -> None:
        """Advance to the next round and reset aggregation event."""
        self._round += 1
        self._aggregation_event.clear()

    async def wait_for_aggregation(self, timeout: float | None = None) -> bool:
        """Wait until aggregation conditions are met.

        Returns True if ready, False if timed out.
        """
        try:
            await asyncio.wait_for(self._aggregation_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
