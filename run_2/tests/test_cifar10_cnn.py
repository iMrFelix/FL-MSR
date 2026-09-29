"""Tests for CIFAR-10 dataset and CNN model.

Covers:
  - CIFAR-10 dataset: loading, shapes, normalization, partitioning
  - CNN model: build, forward pass, parameter count, trainable variables,
    gradient computation, parameter serialization round-trip
  - Config: example YAML passes schema validation
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.datasets.cifar10 import CIFAR10Dataset
from src.models.base import FederationModel
from src.models.cnn import CNNModelDef, SimpleCNN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_cifar10_npz(path: Path, n_train: int = 100, n_val: int = 20, n_test: int = 50):
    """Create a fake CIFAR-10 .npz file with the correct structure."""
    rng = np.random.RandomState(0)
    np.savez(
        path,
        x_train=rng.rand(n_train, 32, 32, 3).astype(np.float32),
        y_train=rng.randint(0, 10, size=n_train).astype(np.int64),
        x_val=rng.rand(n_val, 32, 32, 3).astype(np.float32),
        y_val=rng.randint(0, 10, size=n_val).astype(np.int64),
        x_test=rng.rand(n_test, 32, 32, 3).astype(np.float32),
        y_test=rng.randint(0, 10, size=n_test).astype(np.int64),
    )


# ===========================================================================
# Dataset tests
# ===========================================================================

class TestCIFAR10Dataset:
    """Tests for CIFAR10Dataset."""

    def test_input_shape(self):
        ds = CIFAR10Dataset()
        assert ds.get_input_shape() == (32, 32, 3)

    def test_num_classes(self):
        ds = CIFAR10Dataset()
        assert ds.get_num_classes() == 10

    def test_name(self):
        ds = CIFAR10Dataset()
        assert ds.get_name() == "cifar10"

    def test_load_from_file(self):
        """Loading a well-formed .npz populates all arrays with correct shapes."""
        ds = CIFAR10Dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "node-0.npz"
            _make_fake_cifar10_npz(path, n_train=80, n_val=20, n_test=50)
            ds.load_from_file(str(path))

        x_train, y_train = ds.get_train_data()
        x_val, y_val = ds.get_val_data()
        x_test, y_test = ds.get_test_data()

        # Shapes
        assert x_train.shape == (80, 32, 32, 3)
        assert y_train.shape == (80,)
        assert x_val.shape == (20, 32, 32, 3)
        assert y_val.shape == (20,)
        assert x_test.shape == (50, 32, 32, 3)
        assert y_test.shape == (50,)

        # Dtypes
        assert x_train.dtype == np.float32
        assert y_train.dtype == np.int64

    def test_load_from_file_missing_keys(self):
        """Missing keys in .npz raise ValueError with helpful message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.npz"
            np.savez(path, x_train=np.zeros((1, 32, 32, 3)))
            ds = CIFAR10Dataset()
            with pytest.raises(ValueError, match="missing keys"):
                ds.load_from_file(str(path))

    def test_values_normalized(self):
        """Pixel values should be in [0, 1]."""
        ds = CIFAR10Dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "node-0.npz"
            _make_fake_cifar10_npz(path)
            ds.load_from_file(str(path))

        x_train, _ = ds.get_train_data()
        assert x_train.min() >= 0.0
        assert x_train.max() <= 1.0

    def test_prepare_partitions_creates_files(self):
        """prepare_partitions creates one .npz per node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = CIFAR10Dataset.prepare_partitions(
                total_nodes=4,
                output_dir=tmpdir,
                partition_strategy="iid",
                seed=42,
            )
            assert len(paths) == 4
            for p in paths:
                assert Path(p).exists()
                assert str(p).endswith(".npz")

    def test_prepare_partitions_loadable(self):
        """Each partition file can be loaded and has correct spatial shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            CIFAR10Dataset.prepare_partitions(
                total_nodes=2,
                output_dir=tmpdir,
                partition_strategy="iid",
                seed=42,
            )
            ds = CIFAR10Dataset()
            ds.load_from_file(str(Path(tmpdir) / "node-0.npz"))
            x_train, _ = ds.get_train_data()
            # Must be 4-D: (N, 32, 32, 3) — NOT flattened
            assert len(x_train.shape) == 4
            assert x_train.shape[1:] == (32, 32, 3)

    def test_prepare_partitions_roughly_equal(self):
        """IID partitions should be roughly equal in size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            CIFAR10Dataset.prepare_partitions(
                total_nodes=4,
                output_dir=tmpdir,
                partition_strategy="iid",
                seed=42,
            )
            sizes = []
            for i in range(4):
                ds = CIFAR10Dataset()
                ds.load_from_file(str(Path(tmpdir) / f"node-{i}.npz"))
                x, _ = ds.get_train_data()
                sizes.append(len(x))

            # All nodes should have roughly 50000 * 0.9 / 4 ≈ 11250 train samples
            for s in sizes:
                assert 10000 < s < 13000


# ===========================================================================
# Model tests
# ===========================================================================

class TestSimpleCNN:
    """Tests for the SimpleCNN tf.Module."""

    def test_build(self):
        """SimpleCNN builds without error and is a tf.Module."""
        model = SimpleCNN(input_shape=(32, 32, 3), num_classes=10)
        assert isinstance(model, tf.Module)

    def test_output_shape(self):
        """Forward pass produces logits with shape (batch, num_classes)."""
        model = SimpleCNN(input_shape=(32, 32, 3), num_classes=10)
        x = tf.random.normal((4, 32, 32, 3))
        logits = model(x, training=False)
        assert logits.shape == (4, 10)

    def test_logits_not_softmax(self):
        """Output should be raw logits, not probabilities (sums != 1)."""
        model = SimpleCNN(input_shape=(32, 32, 3), num_classes=10)
        x = tf.random.normal((4, 32, 32, 3))
        logits = model(x, training=False)
        row_sums = tf.reduce_sum(logits, axis=1).numpy()
        # Logits should NOT sum to 1 (softmax would make them sum to 1)
        assert not np.allclose(row_sums, 1.0, atol=0.1)

    def test_parameter_count(self):
        """Model should have roughly 591K trainable parameters."""
        model = SimpleCNN(input_shape=(32, 32, 3), num_classes=10)
        count = FederationModel.get_parameter_count(model)
        # Expected: conv1(896) + conv2(9248) + conv3(18496) + conv4(36928)
        #         + dense1(524416) + head(1290) = 591274
        assert 500_000 < count < 700_000

    def test_trainable_variables_count(self):
        """All 12 trainable variables should be discovered.

        4 conv layers x (kernel + bias) + 2 dense layers x (kernel + bias) = 12.
        """
        model = SimpleCNN(input_shape=(32, 32, 3), num_classes=10)
        assert len(model.trainable_variables) == 12

    def test_gradient_tape_works(self):
        """GradientTape produces non-None gradients for all variables.

        This validates compatibility with the training engine's _train_epoch
        which uses tf.GradientTape for manual training.
        """
        model = SimpleCNN(input_shape=(32, 32, 3), num_classes=10)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        x = tf.random.normal((4, 32, 32, 3))
        y = tf.constant([0, 3, 5, 9])

        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) == 12
        for g in grads:
            assert g is not None


class TestCNNModelDef:
    """Tests for the CNNModelDef FederationModel wrapper."""

    def test_build(self):
        """CNNModelDef.build() returns a working tf.Module."""
        model = CNNModelDef().build(input_shape=(32, 32, 3), num_classes=10)
        assert isinstance(model, tf.Module)
        logits = model(tf.zeros((1, 32, 32, 3)))
        assert logits.shape == (1, 10)

    def test_get_name(self):
        assert CNNModelDef().get_name() == "cnn"

    def test_get_set_parameters_roundtrip(self):
        """get_parameters / set_parameters round-trip preserves values."""
        model = CNNModelDef().build(input_shape=(32, 32, 3), num_classes=10)

        # Get original params
        params = FederationModel.get_parameters(model)
        assert len(params) == 12

        # Modify model (set all to zeros)
        zeros = {k: np.zeros_like(v) for k, v in params.items()}
        FederationModel.set_parameters(model, zeros)

        # Verify zeros
        current = FederationModel.get_parameters(model)
        for k, v in current.items():
            assert np.allclose(v, 0.0)

        # Restore original
        FederationModel.set_parameters(model, params)
        restored = FederationModel.get_parameters(model)
        for k in params:
            assert np.allclose(restored[k], params[k])


# ===========================================================================
# Config test
# ===========================================================================

class TestConfig:
    """Tests for the CIFAR-10 + CNN example config."""

    def test_example_config_validates(self):
        """The example YAML passes schema validation."""
        from src.config.schema import load_config
        config = load_config("configs/examples/fedavg_cifar10_cnn.yaml")
        assert config.training.dataset.name == "cifar10"
        assert config.training.model == "cnn"
        assert config.training.algorithm == "fedavg"
        assert config.training.total_rounds == 50
        assert config.training.epochs_per_round == 1
