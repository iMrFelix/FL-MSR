"""Tests for Stage 5 config schema additions (staleness_threshold, eval_every)."""

import yaml
import pytest
from pydantic import ValidationError

from src.config.schema import ExperimentConfig, TrainingConfig, load_config
from src.config.generator import generate_node_configs


# ---------------------------------------------------------------------------
# Minimal valid config dict for testing
# ---------------------------------------------------------------------------

def _base_config(**training_overrides) -> dict:
    """Build a minimal valid experiment config dict with optional training overrides."""
    training = {
        "dataset": {
            "name": "mnist",
            "partition": {"strategy": "iid", "seed": 42},
        },
        "model": "mlp",
        "algorithm": "adpsgd",
        "epochs_per_round": 1,
        "total_rounds": 10,
        "batch_size": 64,
        "learning_rate": 0.01,
        "optimizer": "sgd",
    }
    training.update(training_overrides)
    return {
        "federation": {
            "num_nodes": 2,
            "topology": {
                "edges": [
                    {"src": 0, "dst": 1, "classes": {0: {"latency_ms": 5, "drop_rate": 0.0}}},
                    {"src": 1, "dst": 0, "classes": {0: {"latency_ms": 5, "drop_rate": 0.0}}},
                ],
            },
        },
        "traffic_classes": {
            "num_classes": 1,
            "dscp_mapping": {0: 0},
        },
        "training": training,
    }


# ---------------------------------------------------------------------------
# Schema validation: new fields
# ---------------------------------------------------------------------------

class TestSchemaNewFields:

    def test_defaults_applied_when_omitted(self):
        """staleness_threshold and eval_every should default when not specified."""
        cfg = ExperimentConfig.model_validate(_base_config())
        assert cfg.training.staleness_threshold == 0
        assert cfg.training.eval_every == 1

    def test_staleness_threshold_accepted(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(staleness_threshold=5)
        )
        assert cfg.training.staleness_threshold == 5

    def test_eval_every_accepted(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(eval_every=10)
        )
        assert cfg.training.eval_every == 10

    def test_both_fields_together(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(staleness_threshold=3, eval_every=5)
        )
        assert cfg.training.staleness_threshold == 3
        assert cfg.training.eval_every == 5

    def test_staleness_threshold_zero(self):
        """Zero is valid (means no filtering)."""
        cfg = ExperimentConfig.model_validate(
            _base_config(staleness_threshold=0)
        )
        assert cfg.training.staleness_threshold == 0

    def test_eval_every_one(self):
        """One is valid (means every iteration)."""
        cfg = ExperimentConfig.model_validate(
            _base_config(eval_every=1)
        )
        assert cfg.training.eval_every == 1


# ---------------------------------------------------------------------------
# Schema validation: invalid values
# ---------------------------------------------------------------------------

class TestSchemaValidation:

    def test_staleness_threshold_negative_rejected(self):
        with pytest.raises(ValidationError, match="staleness_threshold"):
            ExperimentConfig.model_validate(
                _base_config(staleness_threshold=-1)
            )

    def test_eval_every_zero_rejected(self):
        with pytest.raises(ValidationError, match="eval_every"):
            ExperimentConfig.model_validate(
                _base_config(eval_every=0)
            )

    def test_eval_every_negative_rejected(self):
        with pytest.raises(ValidationError, match="eval_every"):
            ExperimentConfig.model_validate(
                _base_config(eval_every=-5)
            )

    def test_staleness_threshold_float_rejected(self):
        """staleness_threshold must be an integer, not a float."""
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(
                _base_config(staleness_threshold=2.5)
            )

    def test_eval_every_float_rejected(self):
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate(
                _base_config(eval_every=2.5)
            )


# ---------------------------------------------------------------------------
# Schema validation: algorithm names
# ---------------------------------------------------------------------------

class TestSchemaAlgorithmNames:

    @pytest.mark.parametrize("algo", ["dpsgd", "adpsgd", "gossip_sgd"])
    def test_valid_algorithms_accepted(self, algo):
        """All registered algorithm names should pass schema validation."""
        cfg = ExperimentConfig.model_validate(_base_config(algorithm=algo))
        assert cfg.training.algorithm == algo


# ---------------------------------------------------------------------------
# Backward compatibility: existing configs still valid
# ---------------------------------------------------------------------------

class TestSchemaBackwardCompat:

    def test_dpsgd_config_still_valid(self):
        """A D-PSGD config without the new fields should still validate."""
        cfg = ExperimentConfig.model_validate(
            _base_config(algorithm="dpsgd")
        )
        assert cfg.training.algorithm == "dpsgd"
        assert cfg.training.staleness_threshold == 0
        assert cfg.training.eval_every == 1


# ---------------------------------------------------------------------------
# Generator: new fields propagated to node configs
# ---------------------------------------------------------------------------

class TestGeneratorPropagation:

    def test_staleness_threshold_propagated(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(staleness_threshold=7)
        )
        node_configs = generate_node_configs(cfg)
        for name, node_cfg in node_configs.items():
            assert node_cfg["training"]["staleness_threshold"] == 7

    def test_eval_every_propagated(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(eval_every=3)
        )
        node_configs = generate_node_configs(cfg)
        for name, node_cfg in node_configs.items():
            assert node_cfg["training"]["eval_every"] == 3

    def test_seed_propagated_into_training(self):
        """The partition seed should be propagated into the training sub-dict."""
        cfg = ExperimentConfig.model_validate(_base_config())
        node_configs = generate_node_configs(cfg)
        for name, node_cfg in node_configs.items():
            assert "seed" in node_cfg["training"]
            assert node_cfg["training"]["seed"] == 42

    def test_seed_matches_top_level(self):
        """Training sub-dict seed should match the top-level node seed."""
        cfg = ExperimentConfig.model_validate(_base_config())
        node_configs = generate_node_configs(cfg)
        for name, node_cfg in node_configs.items():
            assert node_cfg["seed"] == node_cfg["training"]["seed"]

    def test_algorithm_propagated(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(algorithm="gossip_sgd")
        )
        node_configs = generate_node_configs(cfg)
        for name, node_cfg in node_configs.items():
            assert node_cfg["training"]["algorithm"] == "gossip_sgd"


# ---------------------------------------------------------------------------
# Example config file validation
# ---------------------------------------------------------------------------

class TestExampleConfigs:

    def test_adpsgd_example_config_validates(self):
        """The example A-DPSGD config should pass schema validation."""
        cfg = load_config("configs/examples/adpsgd_ring_4nodes.yaml")
        assert cfg.training.algorithm == "adpsgd"
        assert cfg.training.staleness_threshold == 5
        assert cfg.training.eval_every == 5
        assert cfg.federation.num_nodes == 4

    def test_dpsgd_quick_example_still_validates(self):
        """Existing example configs should still validate with new schema."""
        cfg = load_config("configs/examples/dpsgd_ring_4nodes_quick.yaml")
        assert cfg.training.algorithm == "dpsgd"
        # New fields should get defaults
        assert cfg.training.staleness_threshold == 0
        assert cfg.training.eval_every == 1

    def test_dpsgd_perlayer_example_still_validates(self):
        cfg = load_config("configs/examples/dpsgd_ring_4nodes_perlayer.yaml")
        assert cfg.training.algorithm == "dpsgd"

    def test_dpsgd_ring_example_still_validates(self):
        cfg = load_config("configs/examples/dpsgd_ring_4nodes.yaml")
        assert cfg.training.algorithm == "dpsgd"


# ---------------------------------------------------------------------------
# Large/extreme values
# ---------------------------------------------------------------------------

class TestSchemaExtremeValues:

    def test_very_large_staleness_threshold(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(staleness_threshold=999999)
        )
        assert cfg.training.staleness_threshold == 999999

    def test_very_large_eval_every(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(eval_every=100000)
        )
        assert cfg.training.eval_every == 100000
