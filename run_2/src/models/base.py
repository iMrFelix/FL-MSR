from __future__ import annotations
from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class FederationModel(ABC):
    """Base class for model definitions.

    Models subclass tf.Module (NOT tf.keras.Model).
    No compile()/fit() — the training engine owns the optimizer and loss.
    """

    @abstractmethod
    def build(self, input_shape: tuple, num_classes: int) -> tf.Module:
        """Build and return a tf.Module (model architecture only)."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Model identifier."""
        ...

    @staticmethod
    def get_parameters(model: tf.Module) -> dict[str, np.ndarray]:
        """Extract parameters as {var_path: numpy_array}.

        Uses var.path (e.g. 'dense_0/kernel') instead of var.name
        because Keras 3 var.name is just 'kernel'/'bias' — not unique
        across layers.
        """
        return {var.path: var.numpy() for var in model.trainable_variables}

    @staticmethod
    def set_parameters(model: tf.Module, params: dict[str, np.ndarray]) -> None:
        """Set parameters from dict keyed by var.path."""
        for var in model.trainable_variables:
            if var.path in params:
                var.assign(params[var.path])

    @staticmethod
    def get_parameter_count(model: tf.Module) -> int:
        return sum(np.prod(var.shape) for var in model.trainable_variables)
