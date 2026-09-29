"""Constant-width deep CNN for CIFAR-10 layer-importance experiments.

This model replaces the original SimpleCNN's Flatten→Dense(128) bottleneck
with a GlobalAveragePooling layer, distributing parameters across six
convolutional layers of identical size instead of concentrating ~89% in a
single fully-connected layer.

Why the original SimpleCNN is unsuitable for per-layer routing experiments
--------------------------------------------------------------------------
SimpleCNN applies two MaxPool(2×2) layers to a 32×32 input, reducing spatial
dimensions to 8×8.  Flattening yields 8×8×64 = 4,096 features.  A subsequent
Dense(128) therefore has 4,096×128 = 524,288 parameters — 88.7% of the entire
model.  Any per-layer routing experiment collapses to a single binary question:
"did that one layer land in the fast traffic class?"

Fix: GlobalAveragePooling collapses 8×8×64 → 64 features by averaging each
channel spatially.  Dense(10) then has only 640 parameters, leaving the
parameter budget distributed across the convolutional layers.

Architecture
------------
::

    Block 1: Conv2D(64, 3×3, relu, same) → Conv2D(64, 3×3, relu, same) → MaxPool(2×2)
    Block 2: Conv2D(64, 3×3, relu, same) → Conv2D(64, 3×3, relu, same) → MaxPool(2×2)
    Block 3: Conv2D(64, 3×3, relu, same) → Conv2D(64, 3×3, relu, same) → MaxPool(2×2)
    GlobalAveragePooling → Dense(10)   [logits, no softmax]

Parameter distribution (~188K params, ~731 KB as float32):

    conv_0/kernel  3×3×3×64   =   1,728  (0.9%)   ← input layer, small
    conv_1/kernel  3×3×64×64  =  36,864  (19.6%)  ┐
    conv_2/kernel  3×3×64×64  =  36,864  (19.6%)  │ five equal-sized kernels,
    conv_3/kernel  3×3×64×64  =  36,864  (19.6%)  │ each ~147 KB — no single
    conv_4/kernel  3×3×64×64  =  36,864  (19.6%)  │ layer dominates
    conv_5/kernel  3×3×64×64  =  36,864  (19.6%)  ┘
    head/kernel    64×10      =     640  (0.3%)

Traffic class assignment under 3-class routing (5+5+4 split, 14 variables)
---------------------------------------------------------------------------
Gradient-norm importance routes deeper (later) layers — which carry higher
gradient norms — to class 0 (fast).  Uniform importance uses alphabetical
tie-breaking, placing the late layers conv_5/kernel and head in class 2 (slow).

    gradient_norm → class 0: conv_3,4,5 kernels + biases   (~443 KB)
                    class 1: conv_1,2 kernels + biases       (~296 KB)
                    class 2: conv_0, head + biases           ( ~10 KB)

    uniform       → class 0: conv_0, conv_1 kernels + biases (~152 KB)
                    class 1: conv_2,3,4 kernels + biases     (~433 KB)
                    class 2: conv_5/kernel + head + biases   (~150 KB)  ← slow

With 3 classes at 6+3+1 Mbps (sum = 10 Mbps, same as a monolithic 10 Mbps run):
    monolithic      : 731 KB / 10 Mbps = 0.59 s/direction
    gradient_norm   : max(443/6, 296/3, 10/1) Mbit = 0.79 s/direction (+35%)
    uniform         : max(152/6, 433/3, 150/1) Mbit = 1.20 s/direction (+103%)
"""

from __future__ import annotations
import tensorflow as tf
from .base import FederationModel


class DeepCNN(tf.Module):
    """Constant-width CNN for CIFAR-10.  Subclasses tf.Module directly."""

    def __init__(
        self,
        input_shape: tuple = (32, 32, 3),
        num_classes: int = 10,
        name: str = "deep_cnn",
    ):
        super().__init__(name=name)
        self._input_shape = input_shape

        # --- Convolutional block 1 ---
        self.conv_0 = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="conv_0",
        )
        self.conv_1 = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="conv_1",
        )
        self.pool_0 = tf.keras.layers.MaxPool2D((2, 2), name="pool_0")

        # --- Convolutional block 2 ---
        self.conv_2 = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="conv_2",
        )
        self.conv_3 = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="conv_3",
        )
        self.pool_1 = tf.keras.layers.MaxPool2D((2, 2), name="pool_1")

        # --- Convolutional block 3 ---
        self.conv_4 = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="conv_4",
        )
        self.conv_5 = tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same", name="conv_5",
        )
        self.pool_2 = tf.keras.layers.MaxPool2D((2, 2), name="pool_2")

        # --- Classifier head ---
        # GlobalAveragePooling collapses (H, W, 64) → 64 features.
        # Dense(10) has only 64×10 = 640 params — no large FC layer.
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")
        self.head = tf.keras.layers.Dense(num_classes, name="head")

        # Build all layers via a dummy forward pass
        dummy = tf.zeros((1, *input_shape))
        self(dummy, training=False)

    @property
    def trainable_variables(self):
        """Collect trainable variables from all Keras sub-layers.

        GAP and pooling layers have no trainable weights and are omitted.
        """
        variables = []
        for layer in [
            self.conv_0, self.conv_1, self.conv_2,
            self.conv_3, self.conv_4, self.conv_5,
            self.head,
        ]:
            variables.extend(layer.trainable_variables)
        return variables

    def __call__(self, x, training=False):
        x = tf.reshape(x, [-1, *self._input_shape])
        # Block 1
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.pool_0(x)
        # Block 2
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.pool_1(x)
        # Block 3
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.pool_2(x)
        # Head
        x = self.gap(x)
        return self.head(x)   # logits, no softmax


class DeepCNNModelDef(FederationModel):
    """FederationModel wrapper for DeepCNN."""

    def build(self, input_shape: tuple, num_classes: int) -> tf.Module:
        return DeepCNN(input_shape=input_shape, num_classes=num_classes)

    def get_name(self) -> str:
        return "deep_cnn"
