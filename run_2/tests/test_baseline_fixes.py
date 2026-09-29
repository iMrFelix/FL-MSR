"""Tests for the competent-baseline fixes (writeup/16 §5, 2026-08-05).

Covers:
- schema: clipnorm / lr_schedule / server_momentum / workers_only validation
- generator: selective passthrough of the new knobs into node configs
- FedAvgM: server-momentum aggregation math (velocity accumulation,
  beta=0 bit-identity)
- LR schedule: engine _scheduled_lr arithmetic via a minimal stub
- cifar10 workers_only partitioning (synthetic data, load_data patched)
"""

import numpy as np
import pytest

from src.algorithms.base import TrainingUpdate
from src.algorithms.fedavg import FedAvg
from src.training.engine import TrainingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(val: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "layer_0/kernel": np.full((4, 3), val, dtype=np.float32),
        "layer_0/bias": np.full((3,), val, dtype=np.float32),
    }


def _make_update(
    source: str, round_num: int, val: float = 2.0, num_samples: int = 100,
) -> TrainingUpdate:
    return TrainingUpdate(
        source_node=source,
        round_num=round_num,
        parameters=_make_params(val),
        num_samples=num_samples,
    )


def _aggregator(server_momentum: float = 0.0) -> FedAvg:
    return FedAvg(
        "node-0",
        ["node-1", "node-2"],
        {"role": "aggregator", "server_momentum": server_momentum},
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchema:

    def _training(self, **patch):
        from src.config.schema import TrainingConfig
        base = dict(
            dataset={
                "name": "cifar10",
                "partition": {
                    "strategy": "dirichlet", "alpha": 0.1, "seed": 41,
                },
            },
            model="deep_cnn", algorithm="fedavg", epochs_per_round=1,
            total_rounds=20, batch_size=64, learning_rate=0.1,
            optimizer="sgd",
        )
        base.update(patch)
        return TrainingConfig(**base)

    def test_defaults_preserve_legacy(self):
        tr = self._training()
        assert tr.clipnorm is None
        assert tr.lr_schedule == "constant"
        assert tr.server_momentum == 0.0
        assert tr.dataset.partition.workers_only is False

    def test_clipnorm_positive_accepted(self):
        assert self._training(clipnorm=1.0).clipnorm == 1.0

    def test_clipnorm_zero_rejected(self):
        with pytest.raises(Exception):
            self._training(clipnorm=0.0)

    def test_step_schedule_requires_decay_every(self):
        with pytest.raises(Exception, match="lr_decay_every"):
            self._training(lr_schedule="step")
        tr = self._training(lr_schedule="step", lr_decay_every=8)
        assert tr.lr_decay_every == 8

    def test_server_momentum_bounds(self):
        assert self._training(server_momentum=0.9).server_momentum == 0.9
        with pytest.raises(Exception):
            self._training(server_momentum=1.0)

    def test_workers_only_flag(self):
        tr = self._training(
            dataset={
                "name": "cifar10",
                "partition": {
                    "strategy": "dirichlet", "alpha": 0.1, "seed": 41,
                    "workers_only": True,
                },
            },
        )
        assert tr.dataset.partition.workers_only is True


# ---------------------------------------------------------------------------
# Generator passthrough
# ---------------------------------------------------------------------------

class TestGeneratorPassthrough:

    def test_new_knobs_reach_node_configs(self, tmp_path):
        import yaml
        from src.config.schema import load_config
        from src.config.generator import generate_node_configs
        from pathlib import Path

        base = Path(
            "configs/experiments/phase1b/exp1_grid/coverage_eft_eps03/"
            "seed41.yaml"
        )
        cfg = yaml.safe_load(base.read_text())
        cfg["training"].update(
            clipnorm=1.0, lr_schedule="cosine", server_momentum=0.9,
        )
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        node_cfgs = generate_node_configs(load_config(str(p)))
        for name, ncfg in node_cfgs.items():
            tr = ncfg["training"]
            assert tr["clipnorm"] == 1.0, name
            assert tr["lr_schedule"] == "cosine", name
            assert tr["server_momentum"] == 0.9, name
            assert tr["lr_decay_factor"] == 0.1, name
            assert tr["lr_decay_every"] == 0, name

    def test_audit_knobs_reach_node_configs_with_fixed_defaults(self, tmp_path):
        """The T2 audit knobs default to the CORRECTED semantics.

        A missing passthrough here would silently run the pre-fix mechanism
        inside the containers (interface-doc rule 1.7).
        """
        from src.config.schema import load_config
        from src.config.generator import generate_node_configs
        from pathlib import Path

        base = Path(
            "configs/experiments/phase1b/exp1_grid/coverage_eft_eps03/"
            "seed41.yaml"
        )
        node_cfgs = generate_node_configs(load_config(str(base)))
        for name, ncfg in node_cfgs.items():
            tr = ncfg["training"]
            assert tr["aging_age_basis"] == "inclusion", name
            assert tr["epsilon_budget_metric"] == "trigger", name
            assert tr["lr_eta_min"] is None, name  # engine default: 1% of lr


# ---------------------------------------------------------------------------
# FedAvgM server momentum
# ---------------------------------------------------------------------------

class TestServerMomentum:

    def _aggregate_once(self, algo, round_num, val, local):
        async def _run():
            u1 = _make_update("node-1", round_num, val=val)
            u2 = _make_update("node-2", round_num, val=val)
            algo._round = round_num
            await algo.on_update_received(u1)
            await algo.on_update_received(u2)
            return await algo.aggregate(local)
        import asyncio
        # asyncio.run, not get_event_loop(): the latter depends on another
        # test having left a loop installed in this thread.
        return asyncio.run(_run())

    def test_beta_zero_is_plain_fedavg(self):
        algo = _aggregator(server_momentum=0.0)
        result = self._aggregate_once(algo, 0, val=2.0, local=_make_params(1.0))
        np.testing.assert_allclose(result["layer_0/kernel"], 2.0)
        assert algo._server_velocity == {}

    def test_first_round_matches_plain_fedavg(self):
        # v_0 = delta -> theta_1 identical to plain FedAvg on round 0.
        algo = _aggregator(server_momentum=0.9)
        result = self._aggregate_once(algo, 0, val=2.0, local=_make_params(1.0))
        np.testing.assert_allclose(result["layer_0/kernel"], 2.0)

    def test_velocity_accumulates(self):
        # Round 0: delta=1, v=1, theta=2.  Round 1: workers again at
        # theta+1=3 -> delta=1, v=0.9*1+1=1.9, theta=2+1.9=3.9.
        algo = _aggregator(server_momentum=0.9)
        r0 = self._aggregate_once(algo, 0, val=2.0, local=_make_params(1.0))
        np.testing.assert_allclose(r0["layer_0/kernel"], 2.0)
        r1 = self._aggregate_once(
            algo, 1, val=3.0, local={k: v.copy() for k, v in r0.items()},
        )
        np.testing.assert_allclose(r1["layer_0/kernel"], 3.9, rtol=1e-6)

    def test_invalid_beta_fails_fast(self):
        with pytest.raises(ValueError, match="server_momentum"):
            _aggregator(server_momentum=1.5)


# ---------------------------------------------------------------------------
# LR schedule arithmetic (engine methods on a minimal stub)
# ---------------------------------------------------------------------------

class _LrStub:
    """Bare object carrying just what _scheduled_lr reads."""

    _scheduled_lr = TrainingEngine._scheduled_lr
    _eta_min = TrainingEngine._eta_min

    def __init__(self, schedule, base_lr=0.1, total_rounds=20, **cfg):
        self.config = {"lr_schedule": schedule, **cfg}
        self._base_lr = base_lr
        self.total_rounds = total_rounds


class TestLrSchedule:

    def test_constant_is_base(self):
        stub = _LrStub("constant")
        assert stub._scheduled_lr(0) == 0.1
        assert stub._scheduled_lr(19) == 0.1

    def test_cosine_endpoints(self):
        stub = _LrStub("cosine", base_lr=0.1, total_rounds=20)
        assert stub._scheduled_lr(0) == pytest.approx(0.1)
        assert stub._scheduled_lr(10) < 0.06  # past halfway, decayed

    def test_cosine_never_reaches_zero(self):
        # Audit ML-03: cos(pi) is EXACTLY -1.0, so the pre-fix schedule
        # trained the final round at lr = 0 — zero delta, degenerate
        # manifest, ε mechanism off, full model transmitted, in the round
        # the endpoint metrics are read from.
        for total in (20, 150):
            stub = _LrStub("cosine", base_lr=0.1, total_rounds=total)
            final = stub._scheduled_lr(total - 1)
            assert final > 0.0
            assert final == pytest.approx(0.001)  # default 1% floor
        # Explicit floor is honoured, and 0.0 reproduces the pre-fix run.
        floored = _LrStub(
            "cosine", base_lr=0.1, total_rounds=20, lr_eta_min=0.02,
        )
        assert floored._scheduled_lr(19) == pytest.approx(0.02)
        assert floored._scheduled_lr(0) == pytest.approx(0.1)
        legacy = _LrStub(
            "cosine", base_lr=0.1, total_rounds=20, lr_eta_min=0.0,
        )
        assert legacy._scheduled_lr(19) == pytest.approx(0.0, abs=1e-12)

    def test_step_decay(self):
        stub = _LrStub(
            "step", base_lr=0.1, lr_decay_every=8, lr_decay_factor=0.5,
        )
        assert stub._scheduled_lr(7) == pytest.approx(0.1)
        assert stub._scheduled_lr(8) == pytest.approx(0.05)
        assert stub._scheduled_lr(16) == pytest.approx(0.025)

    def test_step_is_floored(self):
        # Same floor applies to 'step': 0.5^k underflows the treatment long
        # before it underflows float64.
        stub = _LrStub(
            "step", base_lr=0.1, lr_decay_every=1, lr_decay_factor=0.1,
        )
        assert stub._scheduled_lr(20) == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# cifar10 workers_only partitioning
# ---------------------------------------------------------------------------

class TestWorkersOnlyPartition:

    def test_no_stranded_shard(self, tmp_path, monkeypatch):
        import tensorflow as tf
        from src.datasets.cifar10 import CIFAR10Dataset

        rng = np.random.RandomState(0)
        n = 2000
        x = rng.rand(n, 32, 32, 3).astype(np.float32)
        y = rng.randint(0, 10, size=(n, 1)).astype(np.int64)
        xt = rng.rand(100, 32, 32, 3).astype(np.float32)
        yt = rng.randint(0, 10, size=(100, 1)).astype(np.int64)
        monkeypatch.setattr(
            tf.keras.datasets.cifar10, "load_data",
            lambda: ((x, y), (xt, yt)),
        )

        paths = CIFAR10Dataset.prepare_partitions(
            total_nodes=4, output_dir=str(tmp_path),
            partition_strategy="dirichlet", seed=41, alpha=0.1,
            workers_only=True,
        )
        assert len(paths) == 4
        sizes = {}
        for p in paths:
            d = np.load(p)
            sizes[p.name] = len(d["x_train"]) + len(d["x_val"])
        # Workers 1-3 jointly hold ALL n samples; node-0 has only the
        # 256-sample copied sliver.
        assert sizes["node-0.npz"] == 256
        assert sum(v for k, v in sizes.items() if k != "node-0.npz") == n

    def test_default_unchanged(self, tmp_path, monkeypatch):
        import tensorflow as tf
        from src.datasets.cifar10 import CIFAR10Dataset

        rng = np.random.RandomState(0)
        n = 1000
        x = rng.rand(n, 32, 32, 3).astype(np.float32)
        y = rng.randint(0, 10, size=(n, 1)).astype(np.int64)
        monkeypatch.setattr(
            tf.keras.datasets.cifar10, "load_data",
            lambda: ((x, y), (x[:50], y[:50])),
        )
        paths = CIFAR10Dataset.prepare_partitions(
            total_nodes=4, output_dir=str(tmp_path),
            partition_strategy="dirichlet", seed=41, alpha=0.1,
        )
        total = sum(
            len(np.load(p)["x_train"]) + len(np.load(p)["x_val"])
            for p in paths
        )
        assert total == n  # all 4 nodes partition the pool, as before
