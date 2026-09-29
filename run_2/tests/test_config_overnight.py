"""Tests for the overnight-evaluation config additions.

Covers the new TrainingConfig knobs (writeup/01-candidate-selection.md §7),
their defaults (which must preserve pre-existing behaviour), validation, and
the generator passthrough into per-node configs (selective, so every field
must be propagated explicitly).
"""

import pytest
from pydantic import ValidationError

from src.config.generator import generate_node_configs
from src.config.schema import ExperimentConfig, load_config


# ---------------------------------------------------------------------------
# Minimal valid config dict (mirrors tests/test_config_stage5.py)
# ---------------------------------------------------------------------------

def _base_config(**training_overrides) -> dict:
    training = {
        "dataset": {
            "name": "mnist",
            "partition": {"strategy": "iid", "seed": 42},
        },
        "model": "mlp",
        "algorithm": "fedavg",
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
# Defaults preserve current behaviour
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_all_new_fields_default_safely(self):
        cfg = ExperimentConfig.model_validate(_base_config())
        t = cfg.training
        assert t.importance_metric_v2 == "raw_norm"
        assert t.assignment_strategy == "gap_based"
        assert t.aging_mode == "none"
        assert t.aging_lambda == 0.0
        assert t.aging_tau_max == 0
        assert t.cyclic_k == 0
        assert t.epsilon_warmup_rounds == 0
        assert t.watchdog_factor == 3.0

    def test_late_layer_policy_still_defaults_to_drop(self):
        cfg = ExperimentConfig.model_validate(_base_config())
        assert cfg.training.late_layer_policy == "drop"


# ---------------------------------------------------------------------------
# Accepted values
# ---------------------------------------------------------------------------

class TestAcceptedValues:

    @pytest.mark.parametrize(
        "metric", ["raw_norm", "delta_sq_norm", "relative", "snr_reweight"]
    )
    def test_importance_metric_v2_literals(self, metric):
        cfg = ExperimentConfig.model_validate(
            _base_config(importance_metric_v2=metric)
        )
        assert cfg.training.importance_metric_v2 == metric

    @pytest.mark.parametrize(
        "strategy", ["gap_based", "byte_balanced", "coverage_eft"]
    )
    def test_assignment_strategy_literals(self, strategy):
        cfg = ExperimentConfig.model_validate(
            _base_config(assignment_strategy=strategy)
        )
        assert cfg.training.assignment_strategy == strategy

    def test_cyclic_with_k_accepted(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(assignment_strategy="cyclic", cyclic_k=3)
        )
        assert cfg.training.assignment_strategy == "cyclic"
        assert cfg.training.cyclic_k == 3

    @pytest.mark.parametrize(
        "mode", ["none", "additive_capped", "stochastic_tail"]
    )
    def test_aging_mode_literals(self, mode):
        cfg = ExperimentConfig.model_validate(_base_config(aging_mode=mode))
        assert cfg.training.aging_mode == mode

    def test_aging_parameters(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(
                aging_mode="additive_capped",
                aging_lambda=0.5,
                aging_tau_max=4,
            )
        )
        assert cfg.training.aging_lambda == pytest.approx(0.5)
        assert cfg.training.aging_tau_max == 4

    def test_epsilon_warmup_rounds(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(epsilon_warmup_rounds=3, epsilon_deadline=0.2)
        )
        assert cfg.training.epsilon_warmup_rounds == 3

    @pytest.mark.parametrize("factor", [0.0, -1.0, 5.5])
    def test_watchdog_factor_any_float_including_disable(self, factor):
        """<= 0 means 'disabled', so negatives and zero must validate."""
        cfg = ExperimentConfig.model_validate(
            _base_config(watchdog_factor=factor)
        )
        assert cfg.training.watchdog_factor == pytest.approx(factor)

    @pytest.mark.parametrize(
        "policy", ["drop", "recycle_last_delta", "renormalize"]
    )
    def test_late_layer_policy_gate_g3_set(self, policy):
        cfg = ExperimentConfig.model_validate(
            _base_config(late_layer_policy=policy)
        )
        assert cfg.training.late_layer_policy == policy


# ---------------------------------------------------------------------------
# Rejected values
# ---------------------------------------------------------------------------

class TestRejectedValues:

    def test_unknown_importance_metric_v2_rejected(self):
        with pytest.raises(ValidationError, match="importance_metric_v2"):
            ExperimentConfig.model_validate(
                _base_config(importance_metric_v2="fisher")
            )

    def test_unknown_assignment_strategy_rejected(self):
        with pytest.raises(ValidationError, match="assignment_strategy"):
            ExperimentConfig.model_validate(
                _base_config(assignment_strategy="density_greedy")
            )

    def test_unknown_aging_mode_rejected(self):
        with pytest.raises(ValidationError, match="aging_mode"):
            ExperimentConfig.model_validate(
                _base_config(aging_mode="multiplicative")
            )

    def test_unknown_late_layer_policy_rejected(self):
        with pytest.raises(ValidationError, match="late_layer_policy"):
            ExperimentConfig.model_validate(
                _base_config(late_layer_policy="async_apply")
            )

    def test_negative_aging_lambda_rejected(self):
        with pytest.raises(ValidationError, match="aging_lambda"):
            ExperimentConfig.model_validate(_base_config(aging_lambda=-0.1))

    def test_negative_aging_tau_max_rejected(self):
        with pytest.raises(ValidationError, match="aging_tau_max"):
            ExperimentConfig.model_validate(_base_config(aging_tau_max=-1))

    def test_negative_cyclic_k_rejected(self):
        with pytest.raises(ValidationError, match="cyclic_k"):
            ExperimentConfig.model_validate(_base_config(cyclic_k=-2))

    def test_negative_epsilon_warmup_rounds_rejected(self):
        with pytest.raises(ValidationError, match="epsilon_warmup_rounds"):
            ExperimentConfig.model_validate(
                _base_config(epsilon_warmup_rounds=-1)
            )

    def test_cyclic_without_k_rejected(self):
        """cyclic with cyclic_k=0 would transmit nothing -> load-time error."""
        with pytest.raises(ValidationError, match="cyclic_k >= 1"):
            ExperimentConfig.model_validate(
                _base_config(assignment_strategy="cyclic")
            )

    def test_cyclic_k_alone_is_fine(self):
        """cyclic_k set while another strategy is active is merely unused."""
        cfg = ExperimentConfig.model_validate(_base_config(cyclic_k=5))
        assert cfg.training.assignment_strategy == "gap_based"


# ---------------------------------------------------------------------------
# Generator passthrough
# ---------------------------------------------------------------------------

class TestGeneratorPropagation:

    def test_all_new_fields_propagated_with_values(self):
        cfg = ExperimentConfig.model_validate(
            _base_config(
                importance_metric_v2="delta_sq_norm",
                assignment_strategy="coverage_eft",
                aging_mode="additive_capped",
                aging_lambda=0.25,
                aging_tau_max=3,
                cyclic_k=2,
                epsilon_warmup_rounds=1,
                watchdog_factor=2.0,
                late_layer_policy="recycle_last_delta",
            )
        )
        node_configs = generate_node_configs(cfg)
        assert node_configs  # at least one node
        for node_cfg in node_configs.values():
            t = node_cfg["training"]
            assert t["importance_metric_v2"] == "delta_sq_norm"
            assert t["assignment_strategy"] == "coverage_eft"
            assert t["aging_mode"] == "additive_capped"
            assert t["aging_lambda"] == pytest.approx(0.25)
            assert t["aging_tau_max"] == 3
            assert t["cyclic_k"] == 2
            assert t["epsilon_warmup_rounds"] == 1
            assert t["watchdog_factor"] == pytest.approx(2.0)
            assert t["late_layer_policy"] == "recycle_last_delta"

    def test_defaults_propagated_when_omitted(self):
        """Nodes must see explicit defaults, never rely on engine fallbacks."""
        cfg = ExperimentConfig.model_validate(_base_config())
        node_configs = generate_node_configs(cfg)
        for node_cfg in node_configs.values():
            t = node_cfg["training"]
            assert t["importance_metric_v2"] == "raw_norm"
            assert t["assignment_strategy"] == "gap_based"
            assert t["aging_mode"] == "none"
            assert t["aging_lambda"] == 0.0
            assert t["aging_tau_max"] == 0
            assert t["cyclic_k"] == 0
            assert t["epsilon_warmup_rounds"] == 0
            assert t["watchdog_factor"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_existing_example_configs_still_validate(self):
        for path in [
            "configs/examples/dpsgd_ring_4nodes_quick.yaml",
            "configs/examples/dpsgd_ring_4nodes_perlayer.yaml",
            "configs/examples/adpsgd_ring_4nodes.yaml",
        ]:
            cfg = load_config(path)
            assert cfg.training.importance_metric_v2 == "raw_norm"
            assert cfg.training.assignment_strategy == "gap_based"
            assert cfg.training.aging_mode == "none"
            assert cfg.training.watchdog_factor == pytest.approx(3.0)
