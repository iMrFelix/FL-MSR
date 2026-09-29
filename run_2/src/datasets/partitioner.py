from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class Partitioner(ABC):
    @abstractmethod
    def partition(
        self, x: np.ndarray, y: np.ndarray, num_nodes: int, seed: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split (x, y) into num_nodes partitions. Returns list of (x_i, y_i)."""
        ...


class IIDPartitioner(Partitioner):
    """Shuffle and split evenly."""
    def partition(self, x, y, num_nodes, seed):
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(x))
        splits = np.array_split(indices, num_nodes)
        return [(x[s], y[s]) for s in splits]


class DirichletPartitioner(Partitioner):
    """Non-IID via Dirichlet distribution over labels.
    Lower alpha = more heterogeneous. alpha=1000 ≈ IID.
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def partition(self, x, y, num_nodes, seed):
        rng = np.random.RandomState(seed)
        num_classes = len(np.unique(y))
        # For each class, sample a distribution over nodes
        node_indices: list[list[int]] = [[] for _ in range(num_nodes)]
        for c in range(num_classes):
            class_indices = np.where(y == c)[0]
            rng.shuffle(class_indices)
            # Sample proportions from Dirichlet
            proportions = rng.dirichlet([self.alpha] * num_nodes)
            # Split class indices according to proportions
            splits = (proportions * len(class_indices)).astype(int)
            # Fix rounding: give remainder to random nodes
            remainder = len(class_indices) - splits.sum()
            for i in range(remainder):
                splits[i % num_nodes] += 1
            start = 0
            for node_id, count in enumerate(splits):
                node_indices[node_id].extend(class_indices[start:start + count])
                start += count
        return [(x[np.array(idx)], y[np.array(idx)]) for idx in node_indices]


class PathologicalPartitioner(Partitioner):
    """Each node gets data from only `classes_per_node` classes.
    Classic pathological non-IID from McMahan et al. 2017.
    """
    def __init__(self, classes_per_node: int = 2):
        self.classes_per_node = classes_per_node

    def partition(self, x, y, num_nodes, seed):
        rng = np.random.RandomState(seed)
        num_classes = len(np.unique(y))
        # Sort by label
        sorted_indices = np.argsort(y)
        # Split into shards (num_nodes * classes_per_node shards total)
        num_shards = num_nodes * self.classes_per_node
        shard_size = len(x) // num_shards
        shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
        # Assign classes_per_node shards to each node
        shard_ids = list(range(num_shards))
        rng.shuffle(shard_ids)
        node_indices = []
        for i in range(num_nodes):
            node_shards = shard_ids[i * self.classes_per_node:(i + 1) * self.classes_per_node]
            idx = np.concatenate([shards[s] for s in node_shards])
            node_indices.append(idx)
        return [(x[idx], y[idx]) for idx in node_indices]


def create_partitioner(strategy: str, **kwargs) -> Partitioner:
    if strategy == "iid":
        return IIDPartitioner()
    elif strategy == "dirichlet":
        return DirichletPartitioner(alpha=kwargs.get("alpha", 0.5))
    elif strategy == "pathological":
        return PathologicalPartitioner(classes_per_node=kwargs.get("classes_per_node", 2))
    else:
        raise ValueError(f"Unknown partition strategy: {strategy}")
