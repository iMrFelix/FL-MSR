"""Distributed-byte FEMNIST CNN — a per-layer-routing-friendly redesign (v3).

Why this model exists
---------------------
The LEAF/FedLUAR reference ``femnist_cnn`` (Conv→Conv→Flatten→Dense(2048)→head)
trains well on FEMNIST but concentrates **97.3% of its bytes in one
``dense_0/kernel``**, which is also high-importance — so the
importance-coverage mechanism degenerates (Phase-2 P0, 2026-06-18: no shed,
per-layer slower than monolithic).

Iteration history (all 2026-06-18, see writeup/08-phase2-execution-log.md):
- v1 (GAP + 5-layer dense tower): distributed bytes (sender β_plan=0.93) but
  did **not train** (val-acc flat ~0.04) — a deep MLP on GAP'd features.
- v2 (constant-width conv + GAP): fast (4.2 s/round) but also barely trained
  (~0.07 even at lr=0.1) — a single Dense(62) on GAP'd features is too weak a
  classifier for 62 classes.
- Root cause: **a capable classifier head is needed to learn 62-way FEMNIST**,
  and the original gets it from its big Flatten→Dense. GAP throws that away.

**v3 keeps the classifier capacity but splits it** so no single layer
dominates: Conv extractor → Flatten → Dense(1024) → Dense(512) → head. Two
moderate dense kernels (~49% / ~44%) of differing importance replace one 97%
kernel — the lower-importance one is a byte-heavy *sheddable* tail (β up to
~0.44), while the dense capacity recovers the trainability GAP lost. Three
poolings shrink the flatten input (3×3×64 = 576) so the dense layers stay
moderate and CPU compute stays low (moltres has no GPU).

Note: this model is run with lr≈0.1 (calibrated up to compensate for the
schema dropping FedLUAR's momentum=0.9 — DEF-7; momentum ≈ a 10× effective-lr
multiplier, so lr-only SGD at 0.01 under-trains).

Architecture (5 weight-bearing layers → 10 manifest variables)
--------------------------------------------------------------
::

    Conv2D(32, 3x3, relu, same) -> MaxPool   28 -> 14
    Conv2D(64, 3x3, relu, same) -> MaxPool   14 ->  7
    Conv2D(64, 3x3, relu, same) -> MaxPool    7 ->  3
    Flatten (3*3*64 = 576)
    Dense(1024, relu)   (dense_0)
    Dense(512,  relu)   (dense_1)
    Dense(62)           (head; logits, no softmax)

Parameter distribution
----------------------
::

    conv_0/kernel    3*3*1*32   =      288   (0.0%)
    conv_1/kernel    3*3*32*64  =   18,432   (1.5%)
    conv_2/kernel    3*3*64*64  =   36,864   (3.1%)
    dense_0/kernel   576*1024   =  589,824   (49.0%)  ┐ two moderate dense
    dense_1/kernel   1024*512   =  524,288   (43.6%)  ┘ kernels, not one 97%
    head/kernel      512*62     =   31,744   (2.6%)
    total (incl. biases)        = 1,203,198 params = 4.59 MiB float32

Subclasses tf.Module — identical contract to ``femnist_cnn`` / ``deep_cnn``.
"""

from __future__ import annotations
import tensorflow as tf
from .base import FederationModel

#: Exact trainable-parameter count (see module docstring derivation).
EXPECTED_PARAMETER_COUNT = 1_203_198


class FEMNISTCNNDistributed(tf.Module):
    """Distributed-byte FEMNIST CNN (conv + split dense head). Subclasses tf.Module."""

    def __init__(
        self,
        input_shape: tuple = (28, 28, 1),
        num_classes: int = 62,
        name: str = "femnist_cnn_distributed",
    ):
        super().__init__(name=name)
        self._input_shape = input_shape

        self.conv_0 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_0")
        self.pool_0 = tf.keras.layers.MaxPool2D((2, 2), name="pool_0")   # 28 -> 14
        self.conv_1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_1")
        self.pool_1 = tf.keras.layers.MaxPool2D((2, 2), name="pool_1")   # 14 -> 7
        self.conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_2")
        self.pool_2 = tf.keras.layers.MaxPool2D((2, 2), name="pool_2")   # 7 -> 3
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.dense_0 = tf.keras.layers.Dense(1024, activation="relu", name="dense_0")
        self.dense_1 = tf.keras.layers.Dense(512, activation="relu", name="dense_1")
        self.head = tf.keras.layers.Dense(num_classes, name="head")

        dummy = tf.zeros((1, *input_shape))
        self(dummy, training=False)

    @property
    def trainable_variables(self):
        """Weight-bearing sub-layers in order; pooling/flatten have no weights."""
        variables = []
        for layer in [self.conv_0, self.conv_1, self.conv_2,
                      self.dense_0, self.dense_1, self.head]:
            variables.extend(layer.trainable_variables)
        return variables

    def __call__(self, x, training=False):
        x = tf.reshape(x, [-1, *self._input_shape])
        x = self.pool_0(self.conv_0(x))
        x = self.pool_1(self.conv_1(x))
        x = self.pool_2(self.conv_2(x))
        x = self.flatten(x)
        x = self.dense_0(x)
        x = self.dense_1(x)
        return self.head(x)   # logits, no softmax


class FEMNISTCNNDistributedModelDef(FederationModel):
    """FederationModel wrapper for FEMNISTCNNDistributed."""

    def build(self, input_shape: tuple, num_classes: int) -> tf.Module:
        return FEMNISTCNNDistributed(input_shape=input_shape, num_classes=num_classes)

    def get_name(self) -> str:
        return "femnist_cnn_distributed"
