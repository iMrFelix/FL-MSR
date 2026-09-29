"""Tests for the T2 (mechanism-semantics) audit fixes.

Each class pins one finding of writeup/19-audit-findings-raw.json, in the
form "the defect, stated as a test that fails on the pre-fix code":

- TRIG-1/ML-01: the τ_max staleness counter resets on RECEIVER-acknowledged
  inclusion, not on the sender's own head placement, so the cap bounds
  realized staleness.  Old basis behind ``aging_age_basis``.
- TRIG-2/BYTE-04: coverage SLIPPAGE ships as the NEW key ``kappa_slip`` and
  deliberately skipped mass as ``shed_mass_fraction``; ``kappa_realized``
  keeps its historical total-shed-mass meaning so the two generations of
  report pool without a definition clash.
- ML-02: FedAvgM velocity accumulates fresh-arrival motion only, so
  momentum and the recycle fill cannot compound (gain β, never β+f).
- ML-03: the cosine/step schedules are floored at ``lr_eta_min`` > 0, so no
  round trains at lr = 0 (arithmetic lives in test_baseline_fixes.py; here
  we pin the engine-level consequence that the treatment stays on).
- ML-04: non-finite loss / weights / importance set a `diverged` marker on
  the round instead of silently reporting chance-level accuracy — and the
  run continues.
- TRIG-5/ML-07: the ε shed budget is metered in frozen trigger mass, so
  arms whose sched metric differs (the 'uniform' ordering control) are
  coverage-matched instead of byte-greedy.
"""

from __future__ import annotations

import asyncio
import json
import math

import numpy as np
import pytest
import tensorflow as tf
from unittest.mock import AsyncMock, MagicMock

from scripts import analysis_common as ac
from src.algorithms.base import FederationAlgorithm, TrainingUpdate
from src.algorithms.fedavg import FedAvg
from src.importance.assignment import CoverageEFTStrategy
from src.importance.manifest import build_manifest
from src.monitoring.collector import MetricsCollector
from src.network.connection_pool import ConnectionPool
from src.network.transport import TransportServer
from src.proto_gen import federation_pb2
from src.training.engine import TrainingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubAlgorithm(FederationAlgorithm):
    """Minimal synchronous algorithm; no aggregation behaviour needed."""

    def __init__(self, neighbors: list[str] | None = None):
        super().__init__("node-0", neighbors or ["node-1"], {})
        self.received: list[TrainingUpdate] = []

    @property
    def is_synchronous(self) -> bool:
        return True

    @property
    def is_centralized(self) -> bool:
        return True

    async def on_local_training_complete(self, model_params, round_num, n):
        update = TrainingUpdate("node-0", round_num, model_params, n)
        return [(neighbor, update) for neighbor in self.neighbors]

    async def on_update_received(self, update: TrainingUpdate) -> None:
        self.received.append(update)

    def ready_to_aggregate(self) -> bool:
        return False

    async def aggregate(self, local_params):
        return local_params


def _model() -> tf.Module:
    model = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(4,))])
    model(tf.zeros((1, 4)))
    return model


def _engine(**overrides) -> TrainingEngine:
    config = {
        "learning_rate": 0.01,
        "optimizer": "sgd",
        "epochs_per_round": 1,
        "total_rounds": 20,
        "batch_size": 32,
        "update_mode": "per_layer",
        "num_traffic_classes": 3,
        "epsilon_deadline": 0.3,
        "importance_metric_v2": "delta_sq_norm",
        "assignment_strategy": "coverage_eft",
        "late_layer_policy": "drop",
        "seed": 7,
    }
    config.update(overrides)
    pool = MagicMock(spec=ConnectionPool)
    pool.send = AsyncMock(return_value=100)
    engine = TrainingEngine(
        node_id="node-0",
        model=_model(),
        algorithm=_StubAlgorithm(),
        server=MagicMock(spec=TransportServer),
        pool=pool,
        config=config,
    )
    return engine


def _params(*names: str) -> dict[str, np.ndarray]:
    return {name: np.ones(4, dtype=np.float32) for name in names}


def _ack(round_num: int, included: list[str]) -> federation_pb2.SkipAdvice:
    return federation_pb2.SkipAdvice(
        round=round_num, inclusion_ack=True, included_layers=included,
    )


# ---------------------------------------------------------------------------
# TRIG-1 / ML-01 — staleness keyed to realized inclusion
# ---------------------------------------------------------------------------

class TestInclusionAgedStaleness:

    def test_age_resets_only_for_acknowledged_layers(self):
        engine = _engine(aging_mode="additive_capped", aging_tau_max=3)
        engine._layer_ages = {"a": 1, "b": 4, "c": 0}
        engine._on_skip_advice(_ack(0, ["a"]))

        ages = engine._advance_layer_ages(_params("a", "b", "c"), 1)

        assert ages == {"a": 0, "b": 5, "c": 1}
        assert engine._age_basis_realized == "inclusion"

    def test_head_placement_alone_never_resets_the_counter(self):
        """The defect itself: shed at the receiver, reset at the sender.

        A layer head-placed every round but never included must age until
        the cap promotes it; under the pre-fix basis its age is pinned at 0
        and the promotion never arms (84/84 spurious resets in
        w3/recycle_aging_eps03 were exactly this).
        """
        params = _params("a", "b")
        engine = _engine(aging_mode="additive_capped", aging_tau_max=3)
        # Round 0's aggregation included 'b' only, three rounds running.
        for round_num in range(1, 4):
            engine._on_skip_advice(_ack(round_num - 1, ["b"]))
            ages = engine._advance_layer_ages(params, round_num)
        assert ages["a"] == 3  # hit the cap
        assert ages["b"] == 0

        legacy = _engine(
            aging_mode="additive_capped",
            aging_tau_max=3,
            aging_age_basis="head_placement",
        )
        for round_num in range(1, 4):
            legacy._on_skip_advice(_ack(round_num - 1, ["b"]))
            legacy_ages = legacy._advance_layer_ages(params, round_num)
            # Sender-side head placement of 'a' every round.
            legacy._layer_ages = {
                name: 0 if name == "a" else legacy_ages[name] + 1
                for name in params
            }
        assert legacy._layer_ages["a"] == 0  # the pre-fix blind spot

    def test_tau_max_promotion_arms_from_realized_absence(self):
        engine = _engine(aging_mode="additive_capped", aging_tau_max=2)
        params = _params("a", "b")
        deltas = {
            "a": np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "b": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        for round_num in range(3):
            if round_num:
                engine._on_skip_advice(_ack(round_num - 1, ["b"]))
            plan = engine._prepare_layer_dissemination(
                params, deltas, round_num,
            )
        # 'a' has been absent from the aggregate for two rounds: it must be
        # promoted to must_receive, which blocks round completion until it
        # actually arrives.
        assert "a" in plan.must_receive

    def test_missing_ack_ages_every_layer(self):
        engine = _engine(aging_mode="additive_capped", aging_tau_max=3)
        engine._on_skip_advice(_ack(0, ["a", "b"]))
        engine._advance_layer_ages(_params("a", "b"), 1)
        # Round 1's ack never arrives: never reset on an unverified proxy.
        ages = engine._advance_layer_ages(_params("a", "b"), 2)
        assert ages == {"a": 1, "b": 1}
        assert engine._age_basis_realized == "inclusion_ack_missing"

    def test_no_ack_channel_falls_back_loudly(self, caplog):
        """Decentralized arms have no aggregator: keep the old basis, warn."""
        engine = _engine(aging_mode="additive_capped", aging_tau_max=3)
        engine._layer_ages = {"a": 2}
        with caplog.at_level("WARNING"):
            ages = engine._advance_layer_ages(_params("a"), 1)
        assert ages == {"a": 2}  # untouched; caller applies the old rule
        assert engine._age_basis_realized == "head_placement_fallback"
        assert "TRIG-1" in caplog.text

    def test_round_zero_starts_fresh(self):
        engine = _engine(aging_mode="additive_capped", aging_tau_max=3)
        assert engine._advance_layer_ages(_params("a"), 0) == {"a": 0}

    def test_ages_and_basis_reach_the_report(self):
        engine = _engine(aging_mode="additive_capped", aging_tau_max=3)
        params = _params("a", "b")
        deltas = {n: np.ones(4, dtype=np.float32) for n in params}
        engine._on_skip_advice(_ack(0, ["a"]))
        engine._prepare_layer_dissemination(params, deltas, 1)
        block = engine._build_sender_block(1)
        assert block["age_basis"] == "inclusion"
        assert block["layer_ages"] == {"a": 0, "b": 1}


class TestInclusionAckChannel:

    def test_realized_inclusions_from_aggregation_telemetry(self):
        engine = _engine()
        engine.algorithm.last_aggregation_telemetry = {
            "a": {"arrived_sources": ["node-1", "node-2"], "filled": "none"},
            "b": {"arrived_sources": ["node-2"], "filled": "recycle"},
        }
        engine.algorithm.inclusion_counts = {
            "a": {"node-1": 1, "node-2": 1},
            "b": {"node-1": 0, "node-2": 1},
        }
        engine.algorithm.aggregation_count = 1
        engine._capture_aggregation_telemetry(4)

        assert engine._realized_inclusions(4) == {
            "node-1": ["a"],
            "node-2": ["a", "b"],
        }

    def test_zero_arrival_source_still_gets_an_ack(self):
        engine = _engine()
        engine.algorithm.last_aggregation_telemetry = {
            "a": {"arrived_sources": ["node-2"], "filled": "recycle"},
        }
        engine.algorithm.inclusion_counts = {"a": {"node-1": 0, "node-2": 1}}
        engine._capture_aggregation_telemetry(2)
        # "none of yours made it" is the statement the counter needs most.
        assert engine._realized_inclusions(2)["node-1"] == []

    def test_acks_are_sent_even_with_skip_feedback_off(self):
        engine = _engine(aging_mode="additive_capped", aging_tau_max=3)
        assert engine.skip_feedback_mode == "off"
        engine.algorithm.last_aggregation_telemetry = {
            "a": {"arrived_sources": ["node-1"], "filled": "none"},
        }
        engine.algorithm.inclusion_counts = {"a": {"node-1": 1}}
        engine._capture_aggregation_telemetry(0)

        asyncio.run(engine._send_skip_advice(0))

        envelope = engine.pool.send.call_args[0][1]
        assert envelope.skip_advice.inclusion_ack is True
        assert list(envelope.skip_advice.included_layers) == ["a"]
        assert list(envelope.skip_advice.layer_names) == []

    def test_no_ack_traffic_for_arms_that_do_not_age(self):
        """Byte accounting must not move for the non-aging arms."""
        engine = _engine()  # aging_mode='none', skip_feedback='off'
        engine.algorithm.last_aggregation_telemetry = {
            "a": {"arrived_sources": ["node-1"], "filled": "none"},
        }
        engine.algorithm.inclusion_counts = {"a": {"node-1": 1}}
        engine._capture_aggregation_telemetry(0)

        assert asyncio.run(engine._send_skip_advice(0)) == 0
        engine.pool.send.assert_not_called()

    def test_ack_travels_aggregator_to_sender_ages(self):
        """Join the two halves: aggregator telemetry -> wire -> ages."""
        aggregator = _engine(aging_mode="additive_capped", aging_tau_max=3)
        aggregator.algorithm.last_aggregation_telemetry = {
            "a": {"arrived_sources": ["node-1"], "filled": "none"},
            "b": {"arrived_sources": [], "filled": "stale"},
        }
        aggregator.algorithm.inclusion_counts = {
            "a": {"node-1": 1}, "b": {"node-1": 0},
        }
        aggregator._capture_aggregation_telemetry(0)
        asyncio.run(aggregator._send_skip_advice(0))
        envelope = aggregator.pool.send.call_args[0][1]

        worker = _engine(aging_mode="additive_capped", aging_tau_max=3)
        worker._on_skip_advice(envelope.skip_advice)
        ages = worker._advance_layer_ages(_params("a", "b"), 1)

        assert ages == {"a": 0, "b": 1}

    def test_ack_is_consumed_with_skip_feedback_off(self):
        engine = _engine(aging_mode="additive_capped", aging_tau_max=3)
        assert engine.skip_feedback_mode == "off"
        engine._on_skip_advice(_ack(3, ["a"]))
        assert engine._inclusion_acks[3] == frozenset({"a"})
        assert engine._skip_advice == {}  # advice half still ignored


# ---------------------------------------------------------------------------
# TRIG-2 / BYTE-04 — κ split
# ---------------------------------------------------------------------------

class TestKappaSplit:

    def _flow(self, engine, scores, skipped, arrived):
        manifest = build_manifest(
            round_num=0,
            source_node_id="node-1",
            importance_scores=scores,
            skipped_layers=skipped,
        )
        engine._round_manifests[0] = {"node-1": manifest}
        record = engine._uplink_record("node-1", 0)
        record["manifest_arrival"] = 1.0
        record["trigger_fire"] = 2.0
        record["layer_arrivals"] = {name: 1.5 for name in arrived}
        record["shed_layers"] = sorted(set(scores) - set(arrived))
        engine._round_starts[0] = 0.0
        return json.loads(engine._export_uplink_telemetry(0))["node-1"]

    def test_deliberate_omission_is_not_slippage(self):
        engine = _engine(skip_feedback="shed")
        flow = self._flow(
            engine,
            scores={"a": 6.0, "b": 3.0, "c": 1.0},
            skipped=["b"],
            arrived=["a"],
        )
        # 'b' was advised away (recycle-covered by design), 'c' genuinely
        # slipped.  One symbol for both is what printed κ = 0.998 next to
        # the invariant κ ≤ ε.
        assert flow["kappa_slip"] == pytest.approx(0.1)
        assert flow["shed_mass_fraction"] == pytest.approx(0.3)
        assert flow["skip_omitted_layers"] == ["b"]
        assert sorted(flow["shed_layers"]) == ["b", "c"]
        # The two are disjoint and sum to the total, which is exactly what
        # `kappa_realized` has always meant.
        assert flow["kappa_realized"] == pytest.approx(0.4)
        assert (
            flow["kappa_slip"] + flow["shed_mass_fraction"]
            == pytest.approx(flow["kappa_realized"])
        )

    def test_kappa_realized_keeps_its_historical_meaning(self):
        """The split must be a NEW symbol, not a redefinition (TRIG-2).

        Every campaign on disk stores total shed mass under `kappa_realized`.
        Had the fix redefined that key in place, a pooled or before/after
        re-analysis would silently mix two quantities under one name with no
        discriminator in the file; instead `kappa_slip` appears beside it and
        its presence IS the discriminator.
        """
        engine = _engine(skip_feedback="shed")
        flow = self._flow(
            engine,
            scores={"a": 6.0, "b": 3.0, "c": 1.0},
            skipped=["b", "c"],
            arrived=["a"],
        )
        # Wholly-deliberate shed set: the pre-fix column would have read 0.4
        # here, and it still does — while the ε-comparable number is 0.
        assert flow["kappa_realized"] == pytest.approx(0.4)
        assert flow["kappa_slip"] == pytest.approx(0.0)
        assert flow["shed_mass_fraction"] == pytest.approx(0.4)

    def test_trigger_path_arms_are_unchanged(self):
        engine = _engine()
        flow = self._flow(
            engine,
            scores={"a": 7.0, "b": 3.0},
            skipped=[],
            arrived=["a"],
        )
        # No deliberate omission ⇒ the two κ keys coincide, so the trigger-path
        # arms' published κ column means the same thing before and after.
        assert flow["kappa_realized"] == pytest.approx(0.3)
        assert flow["kappa_slip"] == pytest.approx(0.3)
        assert flow["shed_mass_fraction"] == pytest.approx(0.0)
        assert "skip_omitted_layers" not in flow

    def test_analysis_reader_dispatches_on_the_new_key(self):
        """The engine's own output, read back through the one κ reader.

        Guards the end-to-end contract the audit objection is about: a
        post-audit flow must resolve to slippage 0.1 (not the 0.4 total), and
        the pre-audit shape of the SAME flow — `kappa_realized` alone — must
        not be mistaken for slippage.
        """
        engine = _engine(skip_feedback="shed")
        flow = self._flow(
            engine,
            scores={"a": 6.0, "b": 3.0, "c": 1.0},
            skipped=["b"],
            arrived=["a"],
        )
        split = ac.kappa_split(flow)
        assert split.source == "engine_split" and split.resolved
        assert split.slip == pytest.approx(0.1)
        assert split.recycled == pytest.approx(0.3)
        assert split.conflated == pytest.approx(0.4)

        legacy = {k: v for k, v in flow.items() if k != "kappa_slip"}
        legacy.pop("shed_mass_fraction")
        stale = ac.kappa_split(legacy)
        # Mixed shed set, no split in the file: unresolved, NOT 0.4-as-slip.
        assert not stale.resolved and stale.slip is None
        assert stale.conflated == pytest.approx(0.4)


class TestKappaConsumers:
    """No consumer may read `kappa_realized` and call the result slippage.

    The two masses are only useful if the readers downstream of the engine
    keep them apart; these pin the two flatteners the audit named.
    """

    def _report(self, flow):
        return {"per_round": [{"round": 0, "nodes": {
            "node-0": {"uplink_telemetry": {"node-1": flow}},
        }}]}

    def test_snac_kappa_bar_reports_total_and_slippage(self):
        from scripts import snac_extractor as sx

        post = {"kappa_realized": 0.4, "kappa_slip": 0.1,
                "shed_mass_fraction": 0.3, "shed_layers": ["b", "c"],
                "skip_omitted_layers": ["b"]}
        total, slip = sx.kappa_bar(self._report(post), "node-0")
        assert total == pytest.approx(0.4) and slip == pytest.approx(0.1)

        # Pre-audit, mixed shed set: the total is still readable, the
        # slippage is not — and must come back None rather than as the total.
        pre = {"kappa_realized": 0.4, "shed_layers": ["b", "c"],
               "skip_omitted_layers": ["b"]}
        total, slip = sx.kappa_bar(self._report(pre), "node-0")
        assert total == pytest.approx(0.4) and slip is None

    def test_uplink_observation_carries_both_kappas(self):
        from scripts import overnight_common as oc

        flow = {"kappa_realized": 0.4, "kappa_slip": 0.1,
                "shed_mass_fraction": 0.3, "shed_layers": ["b", "c"]}
        obs = oc.uplink_observations("run", self._report(flow), "node-0")
        assert len(obs) == 1
        assert obs[0].kappa_realized == pytest.approx(0.4)
        assert obs[0].kappa_slip == pytest.approx(0.1)

        pre = {"kappa_realized": 0.4, "shed_layers": ["b", "c"]}
        assert oc.uplink_observations("run", self._report(pre), "node-0")[
            0].kappa_slip is None


# ---------------------------------------------------------------------------
# ML-02 — FedAvgM must not accelerate slippage fills
# ---------------------------------------------------------------------------

LAYER = "layer_0/kernel"


def _worker_update(source: str, round_num: int, value: float | None):
    """Update carrying LAYER at ``value``; None means the layer is missing."""
    params = (
        {} if value is None
        else {LAYER: np.full((2,), value, dtype=np.float32)}
    )
    return TrainingUpdate(source, round_num, params, 100)


def _fedavg_aggregator(**config) -> FedAvg:
    cfg = {"role": "aggregator", **config}
    return FedAvg("node-0", ["node-1", "node-2"], cfg)


def _aggregate(algo: FedAvg, round_num: int, values, local):
    async def _run():
        algo._round = round_num
        for source, value in values.items():
            await algo.on_update_received(
                _worker_update(source, round_num, value)
            )
        return await algo.aggregate(local)

    return asyncio.run(_run())


class TestMomentumFillFeedback:

    def _theta(self, value: float) -> dict[str, np.ndarray]:
        return {LAYER: np.full((2,), value, dtype=np.float32)}

    def test_starved_layer_drift_is_bounded_not_compounded(self):
        """The regression the finding asks for, at the worst β we ship.

        A layer whose only fresh contributor stops moving is fed by the
        recycle fill alone.  Pre-fix the velocity absorbed that fill and
        re-fed it, giving a per-round gain of β + f = 1.4 — the shed
        147 kB kernel grew 4505x in 20 rounds.  With the fresh/fill split
        the fill decays geometrically (f per round) and the velocity decays
        at β, so the TOTAL drift over any number of starved rounds is
        bounded by Δ_prev·(1/(1−f) + β/(1−β)) — and the per-round motion is
        strictly decreasing instead of exploding.
        """
        algo = _fedavg_aggregator(
            server_momentum=0.9, late_layer_policy="recycle_last_delta",
        )
        theta = self._theta(0.0)
        # Round 0: both workers move the layer by +1 -> Δ_prev = 1.
        theta = _aggregate(algo, 0, {"node-1": 1.0, "node-2": 1.0}, theta)
        assert float(theta[LAYER][0]) == pytest.approx(1.0)
        start = float(theta[LAYER][0])

        # Rounds 1..20: node-2's layer never arrives and node-1 re-sends the
        # global unchanged, so there is no fresh evidence at all.
        steps = []
        for round_num in range(1, 21):
            base = float(theta[LAYER][0])
            theta = _aggregate(
                algo, round_num, {"node-1": base, "node-2": None}, theta,
            )
            steps.append(float(theta[LAYER][0]) - base)

        drift = float(theta[LAYER][0]) - start
        bound = 1.0 / (1.0 - 0.5) + 0.9 / (1.0 - 0.9)  # f = 0.5, β = 0.9
        assert drift < bound
        assert steps == sorted(steps, reverse=True)  # decaying, not growing
        # By round 20 the fill term (f^19) is gone and what remains is pure
        # velocity decay β^20 — no fabricated mass is being re-amplified.
        assert steps[-1] == pytest.approx(0.9 ** 20, abs=1e-3)

    def test_zero_arrived_layer_is_not_moved_by_velocity(self):
        """A layer the log reports as retained bitwise must not move."""
        algo = _fedavg_aggregator(server_momentum=0.9)
        theta = self._theta(0.0)
        theta = _aggregate(algo, 0, {"node-1": 1.0, "node-2": 1.0}, theta)
        for round_num in (1, 2, 3):
            before = theta[LAYER].copy()
            theta = _aggregate(
                algo, round_num, {"node-1": None, "node-2": None}, theta,
            )
            np.testing.assert_array_equal(theta[LAYER], before)

    def test_fresh_layer_keeps_plain_fedavgm_behaviour(self):
        """Complete layers are untouched by the fix: v = βv + Δ."""
        algo = _fedavg_aggregator(server_momentum=0.9)
        theta = self._theta(1.0)
        r0 = _aggregate(algo, 0, {"node-1": 2.0, "node-2": 2.0}, theta)
        np.testing.assert_allclose(r0[LAYER], 2.0)
        r1 = _aggregate(
            algo, 1, {"node-1": 3.0, "node-2": 3.0},
            {k: v.copy() for k, v in r0.items()},
        )
        np.testing.assert_allclose(r1[LAYER], 3.9, rtol=1e-6)

    def test_beta_zero_is_bit_identical_under_partial_arrival(self):
        """No knob needed: at β = 0 the fix cannot change anything."""
        theta = self._theta(0.0)
        algo = _fedavg_aggregator(
            server_momentum=0.0, late_layer_policy="recycle_last_delta",
        )
        theta = _aggregate(algo, 0, {"node-1": 1.0, "node-2": 1.0}, theta)
        result = _aggregate(algo, 1, {"node-1": 4.0, "node-2": None}, theta)
        # 0.5*4 + 0.5*(theta + Δ_prev) = 2 + 0.5*2 = 3, exactly.
        assert float(result[LAYER][0]) == pytest.approx(3.0)

    def test_recycle_state_is_rolled_pre_momentum(self):
        algo = _fedavg_aggregator(
            server_momentum=0.9, late_layer_policy="recycle_last_delta",
        )
        theta = self._theta(0.0)
        _aggregate(algo, 0, {"node-1": 1.0, "node-2": 1.0}, theta)
        # Δ_prev is the CLIENT motion (1.0), not the momentum-accelerated
        # step the global actually took.
        np.testing.assert_allclose(algo._last_delta[LAYER], 1.0)


# ---------------------------------------------------------------------------
# ML-03 — the treatment must stay on in the final round
# ---------------------------------------------------------------------------

class TestLrFloorKeepsTreatmentOn:

    def test_final_round_lr_is_positive_for_live_horizons(self):
        for total in (20, 50, 150):
            engine = _engine(lr_schedule="cosine", total_rounds=total)
            assert engine._scheduled_lr(total - 1) > 0.0

    def test_legacy_zero_floor_is_still_reachable(self):
        engine = _engine(
            lr_schedule="cosine", total_rounds=20, lr_eta_min=0.0,
        )
        assert engine._scheduled_lr(19) == 0.0

    def test_floor_never_exceeds_the_base_lr(self):
        engine = _engine(
            lr_schedule="cosine", total_rounds=20, lr_eta_min=10.0,
        )
        assert engine._scheduled_lr(19) == pytest.approx(0.01)
        assert engine._scheduled_lr(0) == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# ML-04 — divergence marker
# ---------------------------------------------------------------------------

class TestDivergenceGuard:

    def test_non_finite_train_loss_flags_the_round(self, caplog):
        engine = _engine()
        with caplog.at_level("ERROR"):
            engine._check_round_health(3, float("nan"), 1.0, None)
        assert engine._diverged_rounds[3] == "train_loss=nan"
        assert "DIVERGENCE" in caplog.text

    def test_non_finite_val_loss_flags_the_round(self):
        engine = _engine()
        engine._check_round_health(1, 0.5, float("nan"), None)
        assert engine._diverged_rounds[1] == "val_loss=nan"

    def test_non_finite_weights_flag_the_round(self):
        engine = _engine()
        params = {
            "a": np.ones(3, dtype=np.float32),
            "b": np.array([1.0, np.inf, 2.0], dtype=np.float32),
        }
        engine._check_round_health(2, 0.5, 0.5, params)
        assert engine._diverged_rounds[2] == "non_finite_weights:b"

    def test_healthy_round_is_not_flagged(self):
        engine = _engine()
        engine._check_round_health(0, 0.5, 0.4, _params("a"))
        assert engine._diverged_rounds == {}

    def test_non_finite_importance_is_flagged_not_silently_floored(self):
        """``max(nan, 0.0)`` returns nan — the sanitizer that never was."""
        assert math.isnan(max(float("nan"), 0.0))  # the CPython behaviour
        engine = _engine()
        clean = engine._finite_scores(
            {"a": float("nan"), "b": -1.0, "c": 2.0}, 5,
        )
        assert clean == {"a": 0.0, "b": 0.0, "c": 2.0}
        assert engine._diverged_rounds[5] == "non_finite_importance:a"

    def test_first_reason_wins(self):
        engine = _engine()
        engine._flag_divergence(1, "train_loss=nan")
        engine._flag_divergence(1, "val_loss=nan")
        assert engine._diverged_rounds[1] == "train_loss=nan"

    def _collect(self, tmp_path, report) -> dict:
        collector = MetricsCollector(
            host="127.0.0.1", port=0, logdir=str(tmp_path),
            num_nodes=1, total_rounds=3,
        )
        envelope = federation_pb2.Envelope(
            source_node="node-1", dest_node="monitor", metrics_report=report,
        )
        asyncio.run(collector._handle_message(envelope))
        return collector.get_all_metrics()[-1]

    def test_marker_survives_the_report_round_trip(self, tmp_path):
        entry = self._collect(
            tmp_path,
            federation_pb2.MetricsReport(
                node_id="node-1",
                round=7,
                val_accuracy=0.1,
                val_loss=float("nan"),
                diverged=True,
                diverged_reason="val_loss=nan",
            ),
        )
        assert entry["diverged"] is True
        assert entry["diverged_reason"] == "val_loss=nan"

    def test_healthy_report_carries_no_marker(self, tmp_path):
        entry = self._collect(
            tmp_path,
            federation_pb2.MetricsReport(
                node_id="node-1", round=1, val_accuracy=0.5,
            ),
        )
        assert "diverged" not in entry


# ---------------------------------------------------------------------------
# TRIG-5 / ML-07 — ε metered in frozen trigger mass
# ---------------------------------------------------------------------------

# The CP2 shape: four fat conv kernels plus ten small tensors.  With uniform
# sched scores the budget becomes "ε·L layers" and the byte-maximizing tail
# is deterministically those four kernels — 79% of the bytes and, in trigger
# units, far more than ε of the coverage the receiver meters.
_KERNELS = {f"conv_{i}/kernel": 147_500 for i in range(1, 5)}
_SMALL = {f"small_{i}": 2_000 for i in range(10)}
_SIZES = {**_KERNELS, **_SMALL}
_TRIGGER = {
    **{name: 5.0 for name in _KERNELS},
    **{name: 1.0 for name in _SMALL},
}
_BANDWIDTHS = {0: 6.0, 1: 3.0, 2: 1.0}
EPSILON = 0.3


def _coverage(tail: set[str]) -> float:
    total = sum(_TRIGGER.values())
    return 1.0 - sum(_TRIGGER[name] for name in tail) / total


class TestTriggerMassBudget:

    def _assign(self, sched, **kwargs):
        strategy = CoverageEFTStrategy(**kwargs)
        return strategy.assign(
            scores=sched,
            sizes=_SIZES,
            bandwidths=_BANDWIDTHS,
            epsilon=EPSILON,
            must_receive=set(),
            trigger_scores=_TRIGGER,
        )

    def test_uniform_control_is_coverage_matched(self):
        uniform = {name: 1.0 for name in _SIZES}
        result = self._assign(uniform)
        assert _coverage(result.tail) >= 1.0 - EPSILON - 1e-9
        assert result.diagnostics["coverage_planned"] == pytest.approx(
            _coverage(result.tail)
        )

    def test_pre_fix_metering_reproduces_the_infeasible_head(self):
        uniform = {name: 1.0 for name in _SIZES}
        result = self._assign(uniform, budget_metric="sched")
        # ε·14 = 4.2 "layers" -> the four largest kernels, every round.
        assert result.tail == set(_KERNELS)
        assert _coverage(result.tail) < 1.0 - EPSILON

    def test_single_metric_arms_are_untouched(self):
        strategy = CoverageEFTStrategy()
        common = dict(
            sizes=_SIZES,
            bandwidths=_BANDWIDTHS,
            epsilon=EPSILON,
            must_receive=set(),
        )
        with_trigger = strategy.assign(
            scores=dict(_TRIGGER), trigger_scores=dict(_TRIGGER), **common
        )
        without = CoverageEFTStrategy().assign(scores=dict(_TRIGGER), **common)
        assert with_trigger.assignment == without.assignment
        assert with_trigger.tail == without.tail

    def test_sched_scores_still_shape_the_shed_set(self):
        """The aging boost acts through the budget — it must survive.

        Metering ONLY in trigger units would make the sched metric (and
        therefore the whole additive-aging mechanism) inert.
        """
        starved = "small_3"
        boosted = dict.fromkeys(_SIZES, 1.0)
        boosted[starved] = 1_000.0  # what apply_aging does to a starved layer
        result = self._assign(boosted)
        assert starved not in result.tail

    def test_coverage_diagnostics_are_reported_in_both_units(self):
        uniform = {name: 1.0 for name in _SIZES}
        result = self._assign(uniform)
        diag = result.diagnostics
        assert diag["budget_metric"] == "trigger"
        assert diag["coverage_planned"] == pytest.approx(_coverage(result.tail))
        assert diag["coverage_planned_sched"] != pytest.approx(
            diag["coverage_planned"]
        )

    def test_fallback_path_honours_both_budgets(self):
        """> 20 candidates: greedy + DP + the repair pass."""
        sizes = {f"L{i}": (150_000 if i < 8 else 2_000) for i in range(30)}
        trigger = {name: (5.0 if size > 100_000 else 1.0)
                   for name, size in sizes.items()}
        uniform = dict.fromkeys(sizes, 1.0)
        strategy = CoverageEFTStrategy(exact_tail_limit=5)
        result = strategy.assign(
            scores=uniform,
            sizes=sizes,
            bandwidths=_BANDWIDTHS,
            epsilon=EPSILON,
            must_receive=set(),
            trigger_scores=trigger,
        )
        shed_trigger = sum(trigger[name] for name in result.tail)
        shed_sched = float(len(result.tail))
        assert shed_trigger <= EPSILON * sum(trigger.values()) + 1e-6
        assert shed_sched <= EPSILON * len(uniform) + 1e-6

    def test_unknown_budget_metric_fails_fast(self):
        with pytest.raises(ValueError, match="budget_metric"):
            CoverageEFTStrategy(budget_metric="bytes")
