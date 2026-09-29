from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class FederationDataset(ABC):
    """Base class for federated learning datasets.

    Partitioning happens on the HOST before containers launch.
    Inside containers, nodes load their pre-built .npz file.
    """

    @classmethod
    @abstractmethod
    def prepare_partitions(
        cls,
        total_nodes: int,
        output_dir: str,
        partition_strategy: str = "iid",
        seed: int = 42,
        **kwargs,
    ) -> list[Path]:
        """Download dataset, partition, save per-node .npz files. Runs on HOST."""
        ...

    @abstractmethod
    def load_from_file(self, path: str) -> None:
        """Load a pre-partitioned .npz file. Sets internal train/val/test arrays."""
        ...

    @abstractmethod
    def get_input_shape(self) -> tuple:
        """Input shape for model, e.g. (784,) for MNIST."""
        ...

    @abstractmethod
    def get_num_classes(self) -> int:
        """Number of output classes."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Dataset identifier."""
        ...

    @abstractmethod
    def get_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_train, y_train)."""
        ...

    @abstractmethod
    def get_val_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_val, y_val)."""
        ...

    @abstractmethod
    def get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_test, y_test) — shared global test set."""
        ...
