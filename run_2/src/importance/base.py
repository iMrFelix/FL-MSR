"""Base class for importance metrics.

The importance score determines which traffic class a layer's
per-layer update message will be sent on.

Higher importance -> higher priority traffic class -> better network treatment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ImportanceMetric(ABC):
    """Computes importance scores for layers in a training update."""

    @abstractmethod
    def compute(
        self,
        layer_name: str,
        layer_weights: np.ndarray,
        layer_gradients: np.ndarray | None,
        round_num: int,
        context: dict,
    ) -> float:
        """Compute importance score for a single layer.

        Args:
            layer_name: Name of the layer.
            layer_weights: Current weight values.
            layer_gradients: Gradients (if available).
            round_num: Current training round.
            context: Arbitrary context dict for stateful metrics.

        Returns:
            Float importance score.
        """
        ...

    def compute_all(
        self,
        model_params: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray] | None,
        round_num: int,
        context: dict,
    ) -> dict[str, float]:
        """Compute importance for all layers.

        Default: calls compute() per layer. Override for cross-layer metrics.
        """
        result = {}
        for name, weights in model_params.items():
            grad = gradients.get(name) if gradients else None
            result[name] = self.compute(name, weights, grad, round_num, context)
        return result
