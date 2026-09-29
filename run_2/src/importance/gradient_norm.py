"""Gradient norm importance metric.

Assigns importance based on the L2 norm of the gradients for each layer.
Layers with larger gradient norms are considered more important.
"""

from __future__ import annotations

import numpy as np

from src.importance.base import ImportanceMetric


class GradientNormImportance(ImportanceMetric):
    """Importance = L2 norm of the layer's gradients."""

    def compute(
        self,
        layer_name: str,
        layer_weights: np.ndarray,
        layer_gradients: np.ndarray | None,
        round_num: int,
        context: dict,
    ) -> float:
        if layer_gradients is None:
            return 1.0  # Fallback: equal importance
        return float(np.linalg.norm(layer_gradients))
