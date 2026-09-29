"""FEMNIST 4-layer CNN — the FedLUAR / LEAF reference model.

This is the CNN used for the FEMNIST benchmark in the FedLUAR paper
(PRIOR_WORK/FedLUAR_Layer_Recycling_NeurIPS_2025.pdf, "FEMNIST (CNN)"),
which in turn is the LEAF reference model for FEMNIST (LEAF paper,
Appendix B: "a model with two convolutional layers followed by pooling,
and a final dense layer with 2048 units").

Architecture (4 weight-bearing layers)
---------------------------------------
::

    Conv2D(32, 5x5, relu, same) -> MaxPool(2x2)
    Conv2D(64, 5x5, relu, same) -> MaxPool(2x2)
    Flatten -> Dense(2048, relu) -> Dense(62)   [logits, no softmax]

Verification against FedLUAR (how the exact shape was pinned)
--------------------------------------------------------------
FedLUAR never prints the layer sizes, but two of its tables fix them:

- Figure 3 / Table 16 ("the δ = 2 out of 4 layers in CNN"): the FEMNIST
  CNN has exactly 4 layers.
- Table 1 reports the FedAvg server memory footprint for FEMNIST (CNN)
  as 806.11 MB with a = 32 active clients (a · d).  With 28x28x1 inputs
  and the layer stack above:

      conv_0   5*5*1*32  + 32   =      832 params
      conv_1   5*5*32*64 + 64   =   51,264
      dense_0  3136*2048 + 2048 = 6,424,576   (7*7*64 = 3,136 features)
      head     2048*62   + 62   =   127,038
      total                     = 6,603,710 params
                                = 26,414,840 B float32 = 25.19 MiB
      x 32 clients              = 806.1 MiB  ✓ (Table 1: 806.11 MB)

  A 3x3-kernel variant gives 802.0 MiB and a 1024-unit dense gives
  ~417 MiB — both inconsistent with Table 1; 5x5 + 2048 is the unique
  fit, and it matches the published LEAF reference implementation.

Parameter distribution (per-layer routing perspective)
-------------------------------------------------------
``dense_0/kernel`` holds 97.3% of the parameters (24.5 MiB of 25.2 MiB)
— FEMNIST is therefore a *single-dominant-layer* regime, the opposite
of DeepCNN's constant-width profile.  FedLUAR observes exactly this
("the layer with the largest number of parameters tends to be recycled
most frequently", §4.3 + Figure 3).  For ε-deadline scheduling this
makes the knee geometry extreme: shedding the dense kernel removes
~97% of bytes at whatever utility mass it carries that round.

The model subclasses tf.Module (not tf.keras.Model); the training
engine owns optimizer and loss.  Variable paths are 8 manifest entries:
conv_0/{kernel,bias}, conv_1/{kernel,bias}, dense_0/{kernel,bias},
head/{kernel,bias}.
"""

from __future__ import annotations
import tensorflow as tf
from .base import FederationModel

#: Exact trainable-parameter count of the reference architecture; see
#: the module docstring for the derivation against FedLUAR Table 1.
EXPECTED_PARAMETER_COUNT = 6_603_710


class FEMNISTCNN(tf.Module):
    """LEAF/FedLUAR reference CNN for FEMNIST.  Subclasses tf.Module."""

    def __init__(
        self,
        input_shape: tuple = (28, 28, 1),
        num_classes: int = 62,
        name: str = "femnist_cnn",
    ):
        super().__init__(name=name)
        self._input_shape = input_shape

        self.conv_0 = tf.keras.layers.Conv2D(
            32, (5, 5), activation="relu", padding="same", name="conv_0",
        )
        self.pool_0 = tf.keras.layers.MaxPool2D((2, 2), name="pool_0")
        self.conv_1 = tf.keras.layers.Conv2D(
            64, (5, 5), activation="relu", padding="same", name="conv_1",
        )
        self.pool_1 = tf.keras.layers.MaxPool2D((2, 2), name="pool_1")
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.dense_0 = tf.keras.layers.Dense(
            2048, activation="relu", name="dense_0",
        )
        self.head = tf.keras.layers.Dense(num_classes, name="head")

        # Build all layers via a dummy forward pass
        dummy = tf.zeros((1, *input_shape))
        self(dummy, training=False)

    @property
    def trainable_variables(self):
        """Collect trainable variables from all Keras sub-layers.

        Keras 3 layers are not tf.Module subclasses, so tf.Module's
        automatic discovery cannot find them.  Pooling and Flatten
        layers have no trainable weights and are omitted.
        """
        variables = []
        for layer in [self.conv_0, self.conv_1, self.dense_0, self.head]:
            variables.extend(layer.trainable_variables)
        return variables

    def __call__(self, x, training=False):
        x = tf.reshape(x, [-1, *self._input_shape])
        x = self.conv_0(x)
        x = self.pool_0(x)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.flatten(x)
        x = self.dense_0(x)
        return self.head(x)   # logits, no softmax


class FEMNISTCNNModelDef(FederationModel):
    """FederationModel wrapper for FEMNISTCNN."""

    def build(self, input_shape: tuple, num_classes: int) -> tf.Module:
        return FEMNISTCNN(input_shape=input_shape, num_classes=num_classes)

    def get_name(self) -> str:
        return "femnist_cnn"
