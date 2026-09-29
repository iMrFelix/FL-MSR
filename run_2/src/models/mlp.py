from __future__ import annotations
import tensorflow as tf
from .base import FederationModel


class MLP(tf.Module):
    """Simple MLP for MNIST. Subclasses tf.Module directly."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10, name: str = "mlp"):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(200, activation="relu", name="dense_0")
        self.dense2 = tf.keras.layers.Dense(200, activation="relu", name="dense_1")
        self.head = tf.keras.layers.Dense(num_classes, name="head")
        # Build layers by calling with dummy input
        dummy = tf.zeros((1, input_dim))
        self(dummy, training=False)

    @property
    def trainable_variables(self):
        """Collect trainable variables from all Keras sub-layers.

        Keras 3 layers are no longer tf.Module subclasses, so
        tf.Module.trainable_variables won't discover them automatically.
        """
        variables = []
        for layer in [self.dense1, self.dense2, self.head]:
            variables.extend(layer.trainable_variables)
        return variables

    def __call__(self, x, training=False):
        x = tf.reshape(x, [-1, tf.shape(x)[-1]])  # Flatten if needed
        x = self.dense1(x)
        x = self.dense2(x)
        return self.head(x)   # logits, no softmax


class MLPModelDef(FederationModel):
    """FederationModel wrapper for MLP."""

    def build(self, input_shape: tuple, num_classes: int) -> tf.Module:
        input_dim = 1
        for d in input_shape:
            input_dim *= d
        return MLP(input_dim=input_dim, num_classes=num_classes)

    def get_name(self) -> str:
        return "mlp"
