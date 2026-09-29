"""Simple CNN for CIFAR-10 federated learning experiments.

This module provides a small convolutional network suitable for CIFAR-10
classification.  The architecture is deliberately compact (~591K params)
so that training is feasible on CPU-only Docker containers while being
large enough that per-round computation and communication are non-trivial.

Architecture
------------
::

    Conv2D(32, 3x3, ReLU, same) -> Conv2D(32, 3x3, ReLU, same) -> MaxPool(2x2)
    Conv2D(64, 3x3, ReLU, same) -> Conv2D(64, 3x3, ReLU, same) -> MaxPool(2x2)
    Flatten -> Dense(128, ReLU) -> Dense(num_classes)   [logits]

The model returns raw logits (no softmax).  The training engine owns the
loss function (SparseCategoricalCrossentropy) and applies it externally.
"""

from __future__ import annotations
import tensorflow as tf
from .base import FederationModel


class SimpleCNN(tf.Module):
    """Small CNN for CIFAR-10.  Subclasses tf.Module directly."""

    def __init__(
        self,
        input_shape: tuple = (32, 32, 3),
        num_classes: int = 10,
        name: str = "simple_cnn",
    ):
        super().__init__(name=name)
        self._input_shape = input_shape

        # --- Convolutional block 1 ---
        self.conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same", name="conv_0",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same", name="conv_1",
        )
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2), name="pool_0")

        # --- Convolutional block 2 ---
        self.conv3 = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="conv_2",
        )
        self.conv4 = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="conv_3",
        )
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2), name="pool_1")

        # --- Classifier head ---
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.dense1 = tf.keras.layers.Dense(128, activation="relu", name="dense_0")
        self.head = tf.keras.layers.Dense(num_classes, name="head")

        # Build all layers by running a dummy forward pass
        dummy = tf.zeros((1, *input_shape))
        self(dummy, training=False)

    @property
    def trainable_variables(self):
        """Collect trainable variables from all Keras sub-layers.

        Keras 3 layers are no longer tf.Module subclasses, so
        tf.Module.trainable_variables won't discover them automatically.
        Pooling and Flatten layers have no trainable weights.
        """
        variables = []
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4,
                      self.dense1, self.head]:
            variables.extend(layer.trainable_variables)
        return variables

    def __call__(self, x, training=False):
        # Ensure input has spatial dimensions (batch, H, W, C)
        x = tf.reshape(x, [-1, *self._input_shape])
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        # Head
        x = self.flatten(x)
        x = self.dense1(x)
        return self.head(x)   # logits, no softmax


class CNNModelDef(FederationModel):
    """FederationModel wrapper for SimpleCNN."""

    def build(self, input_shape: tuple, num_classes: int) -> tf.Module:
        return SimpleCNN(input_shape=input_shape, num_classes=num_classes)

    def get_name(self) -> str:
        return "cnn"
