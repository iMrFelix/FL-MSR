"""Uniform importance metric (baseline).

All layers get equal importance — effectively no prioritization.
"""

from __future__ import annotations

import numpy as np

from src.importance.base import ImportanceMetric


class UniformImportance(ImportanceMetric):
    """All layers have equal importance (score = 1.0)."""

    def compute(
        self,
        layer_name: str,
        layer_weights: np.ndarray,
        layer_gradients: np.ndarray | None,
        round_num: int,
        context: dict,
    ) -> float:
        return 1.0
