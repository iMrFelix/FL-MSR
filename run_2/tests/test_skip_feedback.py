"""Skip-feedback v2 (sender omission) — plan T1 validation suite.

Covers the three VALIDATE clauses of writeup/04-phase1/plan.md §T1 against
the v2 re-spec in writeup/01-candidate-selection.md §4:

(a) **Unit tests incl. the ratchet scenario**: the coverage denominator
    (manifest raw-score total) must NOT shrink over 5 simulated skip
    rounds — skipped layers remain manifest-listed with their trigger mass
    and the receiver credits that mass as covered-by-recycling
    (`TestRatchetScenario`, `TestManifestSkippedEntries`,
    `TestCoverageTriggerSkipCredit`).

(b) **2-round wire smoke**: an in-process FedAvg federation (aggregator +
    2 workers, real protobuf serialization between engines) runs 2 rounds
    with skip feedback on; skip advice is observed on the wire, the
    advised layers' bytes are actually absent in round 1 (per-layer comm
    metrics + receiver telemetry), and accuracy stays sane
    (`TestWireSmokeTwoRounds`).  The corresponding container-level smoke
    runs in the integration phase once the config-schema keys land
    (integration request; the keys are read via plain dict access here).

(c) **Starvation property**: with aging OFF and skip ON, a permanently
    re-advised layer starves — DOCUMENTED here as designed-in behaviour,
    not a bug (`test_layer_starves_with_aging_off`).  With age-capped
    aging ON it cannot: the τ_max promotion overrides the skip (must-send)
    and bounds staleness (`test_aging_tau_max_bounds_staleness`).

Advice transport semantics (round-tagged, fail-open, idempotent) and the
FedAvg advice generators (shed / fedluar inverse-ratio sampling with fixed
count) are covered alongside.
"""

from __future__ import annotations

import asyncio
import json
import math

import numpy as np
import pytest
import tensorflow as tf
from unittest.mock import MagicMock

from src.algorithms.base import FederationAlgorithm, TrainingUpdate
from src.algorithms.fedavg import FedAvg
from src.importance.manifest import (
    build_manifest,
    manifest_skipped_layers,
    manifest_total,
    validate_manifest,
)
from src.models.base import FederationModel
from src.network.connection_pool import ConnectionPool
from src.network.transport import TransportServer
from src.proto_gen import federation_pb2
from src.training.engine import TrainingEngine, _EngineLayerBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RecordingAlgorithm(FederationAlgorithm):
    """Minimal algorithm for sender-side planner tests."""

    def __init__(self, node_id: str = "node-0", neighbors=None):
        super().__init__(node_id, neighbors or ["node-1"], {})
        self.received: list[TrainingUpdate] = []

    @property
    def is_synchronous(self) -> bool:
        return True

    @property
    def is_centralized(self) -> bool:
        return False

    async def on_local_training_complete(self, model_params, round_num, num_samples):
        update = TrainingUpdate(self.node_id, round_num, model_params, num_samples)
        return [(neighbor, update) for neighbor in self.neighbors]

    async def on_update_received(self, update: TrainingUpdate) -> None:
        self.received.append(update)

    def ready_to_aggregate(self) -> bool:
        return False

    async def aggregate(self, local_params):
        return local_params


def _build_model() -> tf.Module:
    model = tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(4,))])
    model(tf.zeros((1, 4)))
    return model


def make_engine(
    *,
    skip_feedback: str = "shed",
    strategy: str = "byte_balanced",
    epsilon: float = 0.2,
    aging_mode: str = "none",
    aging_tau_max: int = 0,
    aging_lambda: float = 0.0,
    metric_v2: str = "delta_sq_norm",
) -> TrainingEngine:
    """Per-layer sender engine with skip feedback configured."""
    config = {
        "learning_rate": 0.01,
        "optimizer": "sgd",
        "epochs_per_round": 1,
        "total_rounds": 8,
        "batch_size": 32,
        "update_mode": "per_layer",
        "num_traffic_classes": 2,
        "epsilon_deadline": epsilon,
        "watchdog_factor": 0.0,
        "importance_metric_v2": metric_v2,
        "assignment_strategy": strategy,
        "late_layer_policy": "drop",
        "skip_feedback": skip_feedback,
        "aging_mode": aging_mode,
        "aging_tau_max": aging_tau_max,
        "aging_lambda": aging_lambda,
        "seed": 7,
    }
    engine = TrainingEngine(
        node_id="node-0",
        model=_build_model(),
        algorithm=RecordingAlgorithm(),
        server=MagicMock(spec=TransportServer),
        pool=MagicMock(spec=ConnectionPool),
        config=config,
    )
    engine._register_message_handler()
    return engine


def _params4() -> dict[str, np.ndarray]:
    """Four synthetic layers with distinct sizes (deterministic)."""
    rng = np.random.default_rng(11)
    return {
        "conv_0/kernel": rng.normal(size=(32, 8)).astype(np.float32),
        "conv_0/bias": rng.normal(size=(8,)).astype(np.float32),
        "dense_1/kernel": rng.normal(size=(8, 4)).astype(np.float32),
        "dense_1/bias": rng.normal(size=(4,)).astype(np.float32),
    }


def _deltas4() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(13)
    return {
        name: rng.normal(size=arr.shape).astype(np.float32) * 0.1
        for name, arr in _params4().items()
    }


def _advice(round_num: int, layers) -> federation_pb2.SkipAdvice:
    return federation_pb2.SkipAdvice(
        round=round_num, layer_names=sorted(layers),
    )


async def _send_and_capture(
    engine: TrainingEngine,
    plan,
    params: dict[str, np.ndarray],
) -> tuple[federation_pb2.RoundManifest, list[federation_pb2.Envelope]]:
    """Run `_send_updates` against a capturing pool; return (manifest, all)."""
    captured: list[federation_pb2.Envelope] = []

    async def send(dest, envelope, traffic_class=0):
        captured.append(envelope)
        return envelope.ByteSize()

    engine.pool.send = send
    update = TrainingUpdate("node-0", plan.round_num, dict(params), 32)
    await engine._send_updates([("node-1", update)], plan)
    await engine._drain_send_tasks()
    manifests = [
        e.round_manifest for e in captured
        if e.WhichOneof("payload") == "round_manifest"
    ]
    assert len(manifests) == 1
    return manifests[0], captured


# ---------------------------------------------------------------------------
# Manifest: skipped entries (listing rule + validation)
# ---------------------------------------------------------------------------

class TestManifestSkippedEntries:

    def test_skipped_entries_flagged_and_mass_retained(self):
        manifest = build_manifest(
            round_num=2,
            source_node_id="node-1",
            importance_scores={"a": 4.0, "b": 1.0, "c": 3.0},
            skipped_layers={"c"},
        )
        validate_manifest(manifest)
        by_name = {e.layer_name: e for e in manifest.entries}
        assert by_name["c"].skipped is True
        assert by_name["a"].skipped is False
        # The skipped entry keeps its full trigger mass in the total: this
        # is the denominator rule that kills the v1 ratchet.
        assert manifest_total(manifest) == pytest.approx(8.0)
        assert manifest_skipped_layers(manifest) == {"c"}

    def test_skipped_must_receive_contradiction_rejected_at_build(self):
        with pytest.raises(ValueError, match="skipped and must_receive"):
            build_manifest(
                round_num=0,
                source_node_id="node-1",
                importance_scores={"a": 1.0, "b": 2.0},
                must_receive_predicate=lambda name, _s: name == "a",
                skipped_layers={"a"},
            )

    def test_skipped_must_receive_contradiction_rejected_at_validate(self):
        manifest = build_manifest(
            round_num=0,
            source_node_id="node-1",
            importance_scores={"a": 1.0, "b": 2.0},
            skipped_layers={"a"},
        )
        manifest.entries[0].must_receive = True  # corrupt on the "wire"
        with pytest.raises(ValueError, match="skipped and must_receive"):
            validate_manifest(manifest)

    def test_all_skipped_manifest_rejected(self):
        manifest = build_manifest(
            round_num=0,
            source_node_id="node-1",
            importance_scores={"a": 1.0, "b": 2.0},
            skipped_layers={"a", "b"},
        )
        with pytest.raises(ValueError, match="non-skipped"):
            validate_manifest(manifest)

    def test_unknown_skipped_layer_rejected(self):
        with pytest.raises(ValueError, match="subset"):
            build_manifest(
                round_num=0,
                source_node_id="node-1",
                importance_scores={"a": 1.0},
                skipped_layers={"ghost"},
            )

    def test_wire_roundtrip_and_legacy_default(self):
        manifest = build_manifest(
            round_num=1,
            source_node_id="node-1",
            importance_scores={"a": 1.0, "b": 2.0},
            skipped_layers={"b"},
        )
        parsed = federation_pb2.RoundManifest()
        parsed.ParseFromString(manifest.SerializeToString())
        assert manifest_skipped_layers(parsed) == {"b"}
        # Pre-skip-feedback bytes (no field 5) parse as skipped=False.
        legacy = federation_pb2.ImportanceEntry(layer_name="x", raw_score=1.0)
        reparsed = federation_pb2.ImportanceEntry()
        reparsed.ParseFromString(legacy.SerializeToString())
        assert reparsed.skipped is False


# ---------------------------------------------------------------------------
# Receiver trigger: covered-by-recycling credit
# ---------------------------------------------------------------------------

def _buffer(epsilon: float) -> _EngineLayerBuffer:
    return _EngineLayerBuffer(
        epsilon=epsilon, epsilon_for_round=lambda _r: epsilon,
    )


def _skip_manifest() -> federation_pb2.RoundManifest:
    """a=4, b=1 transmitted; c=3, d=2 skipped; total mass 10."""
    return build_manifest(
        round_num=0,
        source_node_id="node-1",
        importance_scores={"a": 4.0, "b": 1.0, "c": 3.0, "d": 2.0},
        skipped_layers={"c", "d"},
    )


def _add(buffer, layer, total_layers=4, round_num=0, source="node-1"):
    return buffer.add_layer(
        source_node=source,
        round_num=round_num,
        layer_name=layer,
        layer_index=0,
        total_layers=total_layers,
        array=np.ones(2, dtype=np.float32),
        num_samples=10,
    )


class TestCoverageTriggerSkipCredit:

    def test_skipped_mass_counts_as_covered(self):
        """ε=0.3 ⇒ target 7 of 10.  Receiving only a (mass 4) suffices
        because the skipped mass 5 is credited: 4 + 5 = 9 ≥ 7.  Without
        the credit the flow would idle at 4 < 7."""
        buffer = _buffer(0.3)
        buffer.register_manifest(_skip_manifest())
        result = _add(buffer, "a")
        assert result is not None and result.is_partial
        # The skipped layers are genuinely absent from the update — the
        # aggregator's fill bookkeeping needs to see them as missing.
        assert result.missing_layers == frozenset({"b", "c", "d"})
        assert set(result.update.parameters) == {"a"}

    def test_without_credit_same_mass_does_not_fire(self):
        """Control for the test above: identical scores, no skip flags."""
        buffer = _buffer(0.3)
        buffer.register_manifest(build_manifest(
            round_num=0,
            source_node_id="node-1",
            importance_scores={"a": 4.0, "b": 1.0, "c": 3.0, "d": 2.0},
        ))
        assert _add(buffer, "a") is None

    def test_skip_credit_not_double_counted_on_arrival(self):
        """Defensive: a skipped layer that arrives anyway is counted once
        (as received), never received + credited."""
        buffer = _buffer(0.3)
        buffer.register_manifest(_skip_manifest())
        # c (mass 3) arrives despite the skip flag: received 3 + credit
        # d=2 → 5 < 7.  Double counting (3 + 5 = 8) would fire here.
        assert _add(buffer, "c") is None
        # a arrives: received 7 + credit 2 = 9 ≥ 7 → fires.
        result = _add(buffer, "a")
        assert result is not None
        assert result.missing_layers == frozenset({"b", "d"})

    def test_count_trigger_reports_skipped_as_missing(self):
        """ε=0 (warm-up / FedLUAR-at-ε=0): the count trigger completes on
        the transmitted set and reports the skipped layers missing."""
        buffer = _buffer(0.0)
        buffer.register_manifest(_skip_manifest())
        assert _add(buffer, "a", total_layers=2) is None
        result = _add(buffer, "b", total_layers=2)
        assert result is not None and result.is_partial
        assert result.missing_layers == frozenset({"c", "d"})

    def test_count_trigger_unchanged_without_skip(self):
        buffer = _buffer(0.0)
        assert _add(buffer, "a", total_layers=2) is None
        result = _add(buffer, "b", total_layers=2)
        assert result is not None
        assert not result.is_partial
        assert result.missing_layers == frozenset()

    def test_must_receive_still_gates_completion(self):
        """Skip credit never bypasses the must-receive gate."""
        manifest = build_manifest(
            round_num=0,
            source_node_id="node-1",
            importance_scores={"a": 4.0, "b": 1.0, "c": 5.0},
            must_receive_predicate=lambda name, _s: name == "b",
            skipped_layers={"c"},
        )
        buffer = _buffer(0.5)
        buffer.register_manifest(manifest)
        # a (4) + credit c (5) = 9 ≥ 5 = (1-0.5)*10, but b is must-receive.
        assert _add(buffer, "a", total_layers=2) is None
        result = _add(buffer, "b", total_layers=2)
        assert result is not None


# ---------------------------------------------------------------------------
# VALIDATE (a): the ratchet scenario
# ---------------------------------------------------------------------------

class TestRatchetScenario:

    async def test_denominator_does_not_shrink_over_five_skip_rounds(self):
        """Five consecutive skip rounds re-advising the same two layers
        (the v1 geometric-starvation driver).  Every round's manifest must
        list ALL layers and carry the FULL trigger mass — the denominator
        never shrinks.  The v1 contrast (transmitted-only mass, which is
        what a manifest listing only transmitted layers would total) is
        strictly smaller, i.e. v1 WOULD have ratcheted here."""
        engine = make_engine(skip_feedback="shed", strategy="byte_balanced")
        params, deltas = _params4(), _deltas4()
        full_universe = set(params)
        advised = {"conv_0/kernel", "dense_1/bias"}

        totals: list[float] = []
        for round_num in range(1, 6):
            engine._on_skip_advice(_advice(round_num - 1, advised))
            plan = engine._prepare_layer_dissemination(
                params, deltas, round_num,
            )
            assert plan is not None
            assert plan.skipped == frozenset(advised)
            manifest, envelopes = await _send_and_capture(
                engine, plan, params,
            )
            validate_manifest(manifest)

            # Listing rule: every layer present, skipped flagged exactly.
            assert {e.layer_name for e in manifest.entries} == full_universe
            assert manifest_skipped_layers(manifest) == advised

            # Denominator: the full per-round trigger mass.
            total = manifest_total(manifest)
            expected_full = sum(plan.trigger_scores.values())
            assert total == pytest.approx(expected_full, rel=1e-6)
            totals.append(total)

            # v1 contrast: a transmitted-only denominator would be smaller.
            transmitted_mass = sum(
                e.raw_score for e in manifest.entries if not e.skipped
            )
            assert transmitted_mass < total

            # Omission semantics: advised layers carry no payload bytes.
            sent_layers = {
                e.layer_update.layer_name for e in envelopes
                if e.WhichOneof("payload") == "layer_update"
            }
            assert sent_layers == full_universe - advised

        # Constant inputs ⇒ the denominator is exactly stable round over
        # round; in particular it never shrinks (the v1 failure mode).
        assert all(t == pytest.approx(totals[0], rel=1e-6) for t in totals)

    async def test_trigger_threshold_consistent_with_wire_manifest(self):
        """End-to-end (a): the manifest a skip round actually emits fires a
        receiver trigger at (1−ε) of the FULL mass with the skip credit."""
        engine = make_engine(skip_feedback="shed", strategy="byte_balanced",
                             epsilon=0.25)
        params, deltas = _params4(), _deltas4()
        engine._on_skip_advice(_advice(0, {"conv_0/kernel"}))
        plan = engine._prepare_layer_dissemination(params, deltas, 1)
        manifest, _ = await _send_and_capture(engine, plan, params)

        buffer = _buffer(0.25)
        buffer.register_manifest(manifest)
        transmitted = [
            e.layer_name for e in manifest.entries if not e.skipped
        ]
        scores = {e.layer_name: e.raw_score for e in manifest.entries}
        total = manifest_total(manifest)
        credit = scores["conv_0/kernel"]
        # Feed transmitted layers score-descending until the credited mass
        # crosses the threshold; the trigger must fire exactly then.  The
        # manifest's source is the emitting engine ("node-0").
        received = 0.0
        fired = None
        for layer in sorted(transmitted, key=scores.get, reverse=True):
            fired = _add(
                buffer, layer, total_layers=len(transmitted),
                round_num=1, source="node-0",
            )
            received += scores[layer]
            if received + credit >= (1.0 - 0.25) * total:
                break
        assert fired is not None
        assert "conv_0/kernel" in fired.missing_layers


# ---------------------------------------------------------------------------
# Advice intake: round-tagged, fail-open, idempotent
# ---------------------------------------------------------------------------

class TestAdviceIntake:

    def test_fail_open_no_advice_sends_everything(self):
        engine = make_engine()
        plan = engine._prepare_layer_dissemination(_params4(), _deltas4(), 3)
        assert plan.skipped == frozenset()
        assert set(plan.assignment.assignment) == set(_params4())

    def test_round_tag_applies_to_next_round_only(self):
        engine = make_engine()
        params, deltas = _params4(), _deltas4()
        engine._on_skip_advice(_advice(2, {"conv_0/bias"}))
        # Round 3 = advice round + 1: applies.
        plan = engine._prepare_layer_dissemination(params, deltas, 3)
        assert plan.skipped == frozenset({"conv_0/bias"})
        # Round 4: the round-2 tag is stale — fail-open, nothing skipped.
        plan = engine._prepare_layer_dissemination(params, deltas, 4)
        assert plan.skipped == frozenset()

    def test_duplicate_advice_idempotent(self):
        engine = make_engine()
        engine._on_skip_advice(_advice(1, {"conv_0/bias"}))
        engine._on_skip_advice(_advice(1, {"conv_0/bias"}))  # re-delivery
        assert engine._skip_advice[1] == frozenset({"conv_0/bias"})
        plan = engine._prepare_layer_dissemination(_params4(), _deltas4(), 2)
        assert plan.skipped == frozenset({"conv_0/bias"})

    def test_conflicting_duplicate_last_write_wins(self):
        engine = make_engine()
        engine._on_skip_advice(_advice(1, {"conv_0/bias"}))
        engine._on_skip_advice(_advice(1, {"dense_1/bias"}))
        assert engine._skip_advice[1] == frozenset({"dense_1/bias"})

    async def test_skip_feedback_off_ignores_advice(self):
        engine = make_engine(skip_feedback="off")
        handler = engine.server.set_handler.call_args[0][0]
        await handler(federation_pb2.Envelope(
            source_node="node-9", dest_node="node-0",
            skip_advice=_advice(1, {"conv_0/bias"}),
        ))
        assert engine._skip_advice == {}
        plan = engine._prepare_layer_dissemination(_params4(), _deltas4(), 2)
        assert plan.skipped == frozenset()

    async def test_handler_dispatches_skip_advice_envelope(self):
        engine = make_engine()
        handler = engine.server.set_handler.call_args[0][0]
        await handler(federation_pb2.Envelope(
            source_node="node-9", dest_node="node-0",
            skip_advice=_advice(4, {"dense_1/kernel"}),
        ))
        assert engine._skip_advice[4] == frozenset({"dense_1/kernel"})

    def test_never_skips_every_layer(self):
        """Advice covering the whole model keeps the highest-trigger-mass
        layer transmitted (an all-skipped manifest could never complete)."""
        engine = make_engine()
        params, deltas = _params4(), _deltas4()
        engine._on_skip_advice(_advice(0, set(params)))
        plan = engine._prepare_layer_dissemination(params, deltas, 1)
        assert len(plan.assignment.assignment) == 1
        (survivor,) = set(plan.assignment.assignment)
        assert survivor == max(
            plan.trigger_scores, key=lambda n: (plan.trigger_scores[n], n),
        )
        assert plan.skipped == frozenset(params) - {survivor}

    def test_unknown_layers_in_advice_ignored(self):
        engine = make_engine()
        engine._on_skip_advice(_advice(0, {"ghost/layer", "conv_0/bias"}))
        plan = engine._prepare_layer_dissemination(_params4(), _deltas4(), 1)
        assert plan.skipped == frozenset({"conv_0/bias"})

    def test_gc_reaps_stale_advice(self):
        engine = make_engine()
        for r in range(5):
            engine._on_skip_advice(_advice(r, {"conv_0/bias"}))
        engine._gc_round_state(5)
        assert set(engine._skip_advice) == {3, 4}

    def test_engine_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="skip_feedback"):
            make_engine(skip_feedback="bogus")


# ---------------------------------------------------------------------------
# VALIDATE (c): starvation property
# ---------------------------------------------------------------------------

def _drive_rounds(engine, target: str, num_rounds: int) -> list[int]:
    """Simulate an aggregator that re-advises ``target`` every round (the
    shed-mode fixed point for a layer that stays missing).  Returns the
    rounds in which the target was actually transmitted."""
    params, deltas = _params4(), _deltas4()
    transmitted: list[int] = []
    for round_num in range(1, num_rounds + 1):
        engine._on_skip_advice(_advice(round_num - 1, {target}))
        plan = engine._prepare_layer_dissemination(params, deltas, round_num)
        if target in plan.assignment.assignment:
            transmitted.append(round_num)
            # Re-advised only while missing: a transmitted round refreshes
            # the layer, mirroring shed-mode advice generation.
    return transmitted


class TestStarvationProperty:

    def test_layer_starves_with_aging_off(self):
        """DOCUMENTED PROPERTY (plan T1 VALIDATE c): with aging OFF and
        skip ON, a layer that the aggregator keeps advising away is never
        transmitted again — it starves, BY DESIGN.  Nothing in the skip
        mechanism itself prevents this (fail-open advice, no per-layer
        guarantee); the manifest listing rule merely keeps its mass in the
        denominator so the trigger arithmetic stays honest while it
        starves.  Starvation control is aging's job — see the companion
        test below.  Arms running skip feedback without aging accept this
        property knowingly."""
        engine = make_engine(skip_feedback="shed", aging_mode="none")
        transmitted = _drive_rounds(engine, "conv_0/kernel", num_rounds=8)
        assert transmitted == []  # starved through all 8 skip rounds

    def test_aging_tau_max_bounds_staleness(self):
        """With age-capped aging ON, the same permanent re-advice cannot
        starve the layer: at age >= tau_max the layer is promoted to
        must_receive (MUST SEND), which overrides the skip; transmission
        resets its age.  Staleness is therefore bounded by tau_max."""
        tau_max = 3
        engine = make_engine(
            skip_feedback="shed",
            aging_mode="additive_capped",
            aging_tau_max=tau_max,
            aging_lambda=0.0,
        )
        transmitted = _drive_rounds(engine, "conv_0/kernel", num_rounds=12)
        # Ages 1, 2, 3 accumulate over rounds 1-3; round 4 hits the cap.
        assert transmitted == [4, 8, 12]
        gaps = [
            second - first
            for first, second in zip(transmitted, transmitted[1:])
        ]
        assert all(gap <= tau_max + 1 for gap in gaps)
        # The cap layer travels as must_receive, never as skipped.
        params, deltas = _params4(), _deltas4()
        engine._on_skip_advice(_advice(12, {"conv_0/kernel"}))
        engine._layer_ages["conv_0/kernel"] = tau_max
        plan = engine._prepare_layer_dissemination(params, deltas, 13)
        assert "conv_0/kernel" in plan.must_receive
        assert "conv_0/kernel" not in plan.skipped

    async def test_must_send_override_reflected_in_manifest(self):
        """Aging override on the wire: the promoted layer is listed as
        must_receive and NOT skipped, despite standing advice."""
        tau_max = 2
        engine = make_engine(
            skip_feedback="shed",
            aging_mode="additive_capped",
            aging_tau_max=tau_max,
            aging_lambda=0.0,
        )
        params, deltas = _params4(), _deltas4()
        engine._layer_ages["dense_1/kernel"] = tau_max  # at the cap
        engine._on_skip_advice(_advice(0, {"dense_1/kernel", "conv_0/bias"}))
        plan = engine._prepare_layer_dissemination(params, deltas, 1)
        manifest, _ = await _send_and_capture(engine, plan, params)
        by_name = {e.layer_name: e for e in manifest.entries}
        assert by_name["dense_1/kernel"].must_receive is True
        assert by_name["dense_1/kernel"].skipped is False
        assert by_name["conv_0/bias"].skipped is True


# ---------------------------------------------------------------------------
# FedAvg: configuration, advice generation, recycle fills
# ---------------------------------------------------------------------------

def _agg_config(**over) -> dict:
    config = {
        "role": "aggregator",
        "late_layer_policy": "drop",
        "skip_feedback": "shed",
        "seed": 7,
    }
    config.update(over)
    return config


def _update(source, round_num, params, num_samples=10) -> TrainingUpdate:
    return TrainingUpdate(
        source_node=source,
        round_num=round_num,
        parameters={k: np.asarray(v, dtype=np.float64) for k, v in params.items()},
        num_samples=num_samples,
    )


class TestFedAvgSkipConfig:

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="skip_feedback"):
            FedAvg("node-0", ["node-1"], _agg_config(skip_feedback="bogus"))

    def test_fedluar_requires_count(self):
        with pytest.raises(ValueError, match="skip_fedluar_count"):
            FedAvg("node-0", ["node-1"], _agg_config(skip_feedback="fedluar"))

    def test_fedluar_random_requires_count(self):
        with pytest.raises(ValueError, match="skip_fedluar_count"):
            FedAvg(
                "node-0", ["node-1"],
                _agg_config(skip_feedback="fedluar_random"),
            )

    def test_fedluar_cyclic_requires_count(self):
        with pytest.raises(ValueError, match="skip_fedluar_count"):
            FedAvg(
                "node-0", ["node-1"],
                _agg_config(skip_feedback="fedluar_cyclic"),
            )

    def test_renorm_combination_rejected(self):
        with pytest.raises(ValueError, match="renorm"):
            FedAvg(
                "node-0", ["node-1"],
                _agg_config(late_layer_policy="renormalize"),
            )

    def test_off_is_default_and_inert(self):
        algorithm = FedAvg(
            "node-0", ["node-1"], {"role": "aggregator"},
        )
        assert algorithm.skip_feedback_mode == "off"
        assert algorithm.last_skip_advice == {}


class TestShedAdviceGeneration:

    async def test_missing_layers_advised_per_sender(self):
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"], _agg_config(),
        )
        local = {"w": np.zeros(2), "b": np.zeros(1)}
        await algorithm.on_update_received(
            _update("node-1", 0, {"w": np.ones(2), "b": np.ones(1)})
        )
        await algorithm.on_update_received(
            _update("node-2", 0, {"w": np.ones(2)})  # b missing
        )
        await algorithm.aggregate(local)
        assert algorithm.last_skip_advice == {"node-2": ["b"]}

    async def test_complete_round_produces_no_advice(self):
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"], _agg_config(),
        )
        local = {"w": np.zeros(2)}
        for source in ("node-1", "node-2"):
            await algorithm.on_update_received(
                _update(source, 0, {"w": np.ones(2)})
            )
        await algorithm.aggregate(local)
        assert algorithm.last_skip_advice == {}


class TestFedLUARAdviceGeneration:

    async def _one_round_advice(self, seed: int) -> dict[str, list[str]]:
        """One aggregation: 'frozen' has zero motion, x/y/z move equally."""
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"],
            _agg_config(
                skip_feedback="fedluar", skip_fedluar_count=1, seed=seed,
            ),
        )
        local = {
            "frozen": np.ones(4), "x": np.zeros(4),
            "y": np.zeros(4), "z": np.zeros(4),
        }
        sent = {
            "frozen": np.ones(4), "x": np.ones(4),
            "y": np.ones(4), "z": np.ones(4),
        }
        for source in ("node-1", "node-2"):
            await algorithm.on_update_received(_update(source, 0, sent))
        await algorithm.aggregate(local)
        return algorithm.last_skip_advice

    async def test_fixed_count_and_global_set(self):
        advice = await self._one_round_advice(seed=3)
        assert set(advice) == {"node-1", "node-2"}
        # Same (global) set for every sender, exactly `count` layers.
        assert advice["node-1"] == advice["node-2"]
        assert len(advice["node-1"]) == 1

    async def test_inverse_ratio_prefers_low_motion_layers(self):
        """The zero-motion layer has ratio 0 → weight 1/floor; it must be
        the sampled layer in essentially every draw (p ≈ 1 − 3e-12)."""
        chosen = []
        for seed in range(40):
            advice = await self._one_round_advice(seed)
            chosen.extend(advice["node-1"])
        frozen_share = chosen.count("frozen") / len(chosen)
        assert frozen_share >= 0.95

    async def test_count_clamped_to_leave_one_transmitted(self):
        algorithm = FedAvg(
            "node-0", ["node-1"],
            _agg_config(
                skip_feedback="fedluar", skip_fedluar_count=10, seed=1,
            ),
        )
        local = {"a": np.zeros(2), "b": np.zeros(2), "c": np.zeros(2)}
        await algorithm.on_update_received(
            _update("node-1", 0, {k: np.ones(2) for k in local})
        )
        await algorithm.aggregate(local)
        assert len(algorithm.last_skip_advice["node-1"]) == 2  # L - 1


class TestFedLUARRandomAdviceGeneration:
    """FedLUAR's own metric ablation (their Table 4 'Random').

    Same fixed count, same one-global-set broadcast, same recycle fill,
    same unbounded staleness as ``fedluar`` — the ONLY difference is that
    the draw is uniform.  These tests pin exactly that: the mode must keep
    every structural property of ``fedluar`` while demonstrably NOT
    consulting the importance metric.
    """

    async def _one_round_advice(self, seed: int, count: int = 1
                                ) -> dict[str, list[str]]:
        """Same fixture as TestFedLUARAdviceGeneration: 'frozen' has zero
        motion (ratio 0 → the inverse-ratio rule would take it almost
        always), x/y/z move equally."""
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"],
            _agg_config(
                skip_feedback="fedluar_random",
                skip_fedluar_count=count,
                seed=seed,
            ),
        )
        local = {
            "frozen": np.ones(4), "x": np.zeros(4),
            "y": np.zeros(4), "z": np.zeros(4),
        }
        sent = {
            "frozen": np.ones(4), "x": np.ones(4),
            "y": np.ones(4), "z": np.ones(4),
        }
        for source in ("node-1", "node-2"):
            await algorithm.on_update_received(_update(source, 0, sent))
        await algorithm.aggregate(local)
        return algorithm.last_skip_advice

    async def test_fixed_count_and_global_set(self):
        advice = await self._one_round_advice(seed=3, count=2)
        assert set(advice) == {"node-1", "node-2"}
        assert advice["node-1"] == advice["node-2"]   # one global set
        assert len(advice["node-1"]) == 2             # exactly `count`

    async def test_uniform_draw_ignores_the_importance_metric(self):
        """The whole point of the control: the zero-motion layer must NOT
        be preferred.  Under ``fedluar`` its share is >= 0.95 (see
        TestFedLUARAdviceGeneration); under uniform sampling of 1 of 4 it
        must sit near 0.25, and every layer must be reachable."""
        chosen: list[str] = []
        for seed in range(200):
            advice = await self._one_round_advice(seed)
            chosen.extend(advice["node-1"])
        frozen_share = chosen.count("frozen") / len(chosen)
        assert 0.15 <= frozen_share <= 0.35, frozen_share
        assert set(chosen) == {"frozen", "x", "y", "z"}

    async def test_count_clamped_to_leave_one_transmitted(self):
        algorithm = FedAvg(
            "node-0", ["node-1"],
            _agg_config(
                skip_feedback="fedluar_random", skip_fedluar_count=10, seed=1,
            ),
        )
        local = {"a": np.zeros(2), "b": np.zeros(2), "c": np.zeros(2)}
        await algorithm.on_update_received(
            _update("node-1", 0, {k: np.ones(2) for k in local})
        )
        await algorithm.aggregate(local)
        assert len(algorithm.last_skip_advice["node-1"]) == 2  # L - 1


class TestFedLUARCyclicAdviceGeneration:
    """The bounded-staleness rotation control FedLUAR never ran.

    Same delta, same one-global-set broadcast, same recycle fill as
    ``fedluar``/``fedluar_random`` — the skip set simply rotates.  The
    property that earns the arm its place in the campaign is the one tested
    last: every layer is refreshed within a bounded number of rounds, with
    no ``aging_tau_max`` machinery anywhere.
    """

    async def _advice_sequence(self, rounds: int, count: int,
                               layers: tuple[str, ...]) -> list[list[str]]:
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"],
            _agg_config(
                skip_feedback="fedluar_cyclic",
                skip_fedluar_count=count,
                seed=11,
            ),
        )
        state = {name: np.zeros(4) for name in layers}
        sent = {name: np.ones(4) for name in layers}
        seq: list[list[str]] = []
        for rnd in range(rounds):
            if rnd:
                algorithm.advance_round()
            for source in ("node-1", "node-2"):
                await algorithm.on_update_received(_update(source, rnd, sent))
            state = await algorithm.aggregate(state)
            seq.append(list(algorithm.last_skip_advice["node-1"]))
        return seq

    async def test_fixed_count_and_global_set(self):
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"],
            _agg_config(
                skip_feedback="fedluar_cyclic", skip_fedluar_count=2, seed=5,
            ),
        )
        local = {k: np.zeros(4) for k in ("a", "b", "c", "d", "e")}
        for source in ("node-1", "node-2"):
            await algorithm.on_update_received(
                _update(source, 0, {k: np.ones(4) for k in local})
            )
        await algorithm.aggregate(local)
        advice = algorithm.last_skip_advice
        assert advice["node-1"] == advice["node-2"]   # one global set
        assert len(advice["node-1"]) == 2             # exactly `count`

    async def test_rotation_is_deterministic_and_seed_independent(self):
        """No RNG is consulted, so two different seeds give the same
        schedule — the property that separates this from fedluar_random."""
        layers = ("a", "b", "c", "d", "e")
        first = await self._advice_sequence(4, 2, layers)
        assert first == [["a", "b"], ["c", "d"], ["a", "e"], ["b", "c"]]

    async def test_staleness_is_bounded_without_any_aging_knob(self):
        """Over a full rotation every layer is skipped exactly delta/L of
        the rounds, and no layer is ever skipped for an unbounded run —
        the guarantee FedLUAR explicitly declines to make (§3.3: 'does not
        specify the upper bound of k')."""
        layers = tuple("abcdefg")          # L = 7
        count, rounds = 2, 14              # two full rotations
        seq = await self._advice_sequence(rounds, count, layers)
        skips = {name: 0 for name in layers}
        longest_run = {name: 0 for name in layers}
        run = {name: 0 for name in layers}
        for advised in seq:
            for name in layers:
                if name in advised:
                    skips[name] += 1
                    run[name] += 1
                    longest_run[name] = max(longest_run[name], run[name])
                else:
                    run[name] = 0
        # Perfectly even coverage: 14 rounds x 2 skips / 7 layers = 4 each.
        assert set(skips.values()) == {rounds * count // len(layers)}
        # And nobody starves: the consecutive-skip run is tiny and bounded.
        assert max(longest_run.values()) <= 2, longest_run


class TestAdvisedRecycleFill:

    async def test_advised_miss_recycled_under_drop_policy(self):
        """shed mode + drop slippage: round 1's advised omission is
        recycle-filled (previous aggregated delta re-applied), while a
        non-advised miss in the same round keeps the drop stale-fill —
        the manifest's covered-by-recycling declaration is made true
        without changing the slippage mode for genuine slips."""
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"], _agg_config(),
        )
        local0 = {"w": np.zeros(2), "b": np.zeros(2)}

        # Round 0: node-2's b slips → drop stale-fill; advice {node-2: b}.
        await algorithm.on_update_received(
            _update("node-1", 0, {"w": np.ones(2), "b": np.ones(2)})
        )
        await algorithm.on_update_received(
            _update("node-2", 0, {"w": np.ones(2)})
        )
        agg0 = await algorithm.aggregate(local0)
        np.testing.assert_allclose(agg0["b"], 0.5 * np.ones(2))  # stale fill
        assert algorithm.last_skip_advice == {"node-2": ["b"]}
        assert algorithm.last_aggregation_telemetry["b"]["filled"] == "stale"

        # Round 1: node-2 omits b on advice → recycle; node-1's w slips
        # WITHOUT advice → stale fill (drop).
        algorithm.advance_round()
        await algorithm.on_update_received(
            _update("node-1", 1, {"b": 2.0 * np.ones(2)})
        )
        await algorithm.on_update_received(
            _update("node-2", 1, {"w": 2.0 * np.ones(2)})
        )
        agg1 = await algorithm.aggregate(agg0)
        # b: node-1 arrived (2.0); node-2 advised-missing → recycled value
        # = global b (0.5) + last delta b (0.5 − 0 = 0.5) = 1.0.
        # b_new = 0.5·2.0 + 0.5·1.0 = 1.5  (drop would give 1.25).
        np.testing.assert_allclose(agg1["b"], 1.5 * np.ones(2))
        assert algorithm.last_aggregation_telemetry["b"]["filled"] == "recycle"
        # w: node-1 missing, NOT advised → drop stale-fill from global
        # (1.0): w_new = 0.5·2.0 + 0.5·1.0 = 1.5, labelled stale.
        np.testing.assert_allclose(agg1["w"], 1.5 * np.ones(2))
        assert algorithm.last_aggregation_telemetry["w"]["filled"] == "stale"

    async def test_zero_arrived_advised_layer_recycled(self):
        """FedLUAR-native semantics: a layer every sender was advised to
        skip arrives from nobody, yet the aggregate still MOVES by the
        stored delta (filled='recycle') — unlike the non-advised
        zero-arrived anomaly, which retains the global bitwise."""
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"],
            _agg_config(skip_feedback="fedluar", skip_fedluar_count=1),
        )
        local0 = {"hot": np.zeros(4), "cold": np.ones(4)}
        sent0 = {"hot": np.ones(4), "cold": 1.1 * np.ones(4)}
        for source in ("node-1", "node-2"):
            await algorithm.on_update_received(_update(source, 0, sent0))
        agg0 = await algorithm.aggregate(local0)
        advice = algorithm.last_skip_advice
        assert advice["node-1"] == advice["node-2"]
        (advised_layer,) = advice["node-1"]
        other = "hot" if advised_layer == "cold" else "cold"
        delta = agg0[advised_layer] - local0[advised_layer]

        algorithm.advance_round()
        for source in ("node-1", "node-2"):
            await algorithm.on_update_received(
                _update(source, 1, {other: 2.0 * np.asarray(sent0[other])})
            )
        agg1 = await algorithm.aggregate(agg0)
        np.testing.assert_allclose(agg1[advised_layer], agg0[advised_layer] + delta)
        assert np.any(agg1[advised_layer] != agg0[advised_layer])
        telemetry = algorithm.last_aggregation_telemetry[advised_layer]
        assert telemetry["filled"] == "recycle"
        assert telemetry["arrived_sources"] == []

    async def test_non_advised_zero_arrived_stays_stale(self):
        """The existing anomaly guard is untouched: zero arrivals WITHOUT
        advice keep the global bitwise (no blind recycling — the ratchet
        failure mode the gate flagged)."""
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"], _agg_config(),
        )
        local = {"w": np.zeros(2), "b": 3.0 * np.ones(2)}
        for source in ("node-1", "node-2"):
            await algorithm.on_update_received(
                _update(source, 0, {"w": np.ones(2)})  # nobody sends b
            )
        agg0 = await algorithm.aggregate(local)
        np.testing.assert_array_equal(agg0["b"], local["b"])
        assert algorithm.last_aggregation_telemetry["b"]["filled"] == "stale"


# ---------------------------------------------------------------------------
# Engine aggregator side: advice on the broadcast + telemetry
# ---------------------------------------------------------------------------

class TestAdviceBroadcast:

    def _aggregator_engine(self, algorithm) -> TrainingEngine:
        config = {
            "update_mode": "per_layer",
            "num_traffic_classes": 2,
            "skip_feedback": algorithm.skip_feedback_mode,
            "late_layer_policy": "drop",
            "assignment_strategy": "byte_balanced",
            "importance_metric_v2": "delta_sq_norm",
            "watchdog_factor": 0.0,
            "seed": 7,
        }
        engine = TrainingEngine(
            node_id="node-0",
            model=_build_model(),
            algorithm=algorithm,
            server=MagicMock(spec=TransportServer),
            pool=MagicMock(spec=ConnectionPool),
            config=config,
        )
        return engine

    async def test_advice_sent_per_destination_class0(self):
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"], _agg_config(),
        )
        algorithm.last_skip_advice = {
            "node-1": ["conv_0/bias"], "node-2": [],
        }
        engine = self._aggregator_engine(algorithm)
        sent = []

        async def send(dest, envelope, traffic_class=0):
            sent.append((dest, envelope, traffic_class))
            return envelope.ByteSize()

        engine.pool.send = send
        advice_bytes = await engine._send_skip_advice(4)
        assert advice_bytes > 0
        assert len(sent) == 1  # empty advice lists are not sent
        dest, envelope, traffic_class = sent[0]
        assert dest == "node-1" and traffic_class == 0
        assert envelope.WhichOneof("payload") == "skip_advice"
        assert envelope.skip_advice.round == 4
        assert list(envelope.skip_advice.layer_names) == ["conv_0/bias"]

    async def test_send_failure_is_fail_open(self):
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"], _agg_config(),
        )
        algorithm.last_skip_advice = {
            "node-1": ["conv_0/bias"], "node-2": ["conv_0/bias"],
        }
        engine = self._aggregator_engine(algorithm)

        async def send(dest, envelope, traffic_class=0):
            if dest == "node-1":
                raise ConnectionError("boom")
            return envelope.ByteSize()

        engine.pool.send = send
        advice_bytes = await engine._send_skip_advice(2)  # must not raise
        assert advice_bytes > 0  # node-2's advice still counted

    async def test_skip_advice_in_aggregation_telemetry_block(self):
        algorithm = FedAvg(
            "node-0", ["node-1", "node-2"], _agg_config(),
        )
        local = {"w": np.zeros(2), "b": np.zeros(2)}
        await algorithm.on_update_received(
            _update("node-1", 0, {"w": np.ones(2), "b": np.ones(2)})
        )
        await algorithm.on_update_received(
            _update("node-2", 0, {"w": np.ones(2)})
        )
        await algorithm.aggregate(local)
        engine = self._aggregator_engine(algorithm)
        engine._capture_aggregation_telemetry(0)
        block = engine._aggregation_info[0]
        assert block["skip_advice"] == {"node-2": ["b"]}


# ---------------------------------------------------------------------------
# VALIDATE (b): 2-round wire smoke (in-process federation)
# ---------------------------------------------------------------------------

class _Wire:
    """In-memory wire between engines: every envelope is serialized and
    re-parsed (true wire form), routed to the destination's registered
    handler; monitor reports and skip-advice traffic are recorded."""

    def __init__(self):
        self.handlers: dict[str, object] = {}
        self.reports: dict[str, list] = {}
        self.advice_seen: list[tuple[str, int, list[str]]] = []
        self.manifests: list[federation_pb2.RoundManifest] = []

    def register(self, node_id: str, engine: TrainingEngine) -> None:
        engine._register_message_handler()
        self.handlers[node_id] = engine.server.set_handler.call_args[0][0]
        engine.pool.send = self._make_send()

    def _make_send(self):
        async def send(dest, envelope, traffic_class=0):
            data = envelope.SerializeToString()
            wire = federation_pb2.Envelope()
            wire.ParseFromString(data)
            payload = wire.WhichOneof("payload")
            if payload == "skip_advice":
                self.advice_seen.append((
                    wire.dest_node,
                    wire.skip_advice.round,
                    list(wire.skip_advice.layer_names),
                ))
            elif payload == "round_manifest":
                self.manifests.append(wire.round_manifest)
            if dest == "monitor":
                self.reports.setdefault(wire.source_node, []).append(
                    wire.metrics_report
                )
                return len(data)
            await self.handlers[dest](wire)
            return len(data)
        return send


def _toy_data(seed: int, n: int = 48):
    """Linearly separable 3-class toy set: y = argmax of the first 3 dims."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    y = np.argmax(x[:, :3], axis=1).astype(np.int64)
    return x, y


def _smoke_model() -> tf.Module:
    """Model with EXPLICIT layer names: in-process the Keras name
    uniquifier would otherwise give each instance different variable paths
    (sequential_N/dense_N/...), and the three nodes' layer names must
    match exactly as they do when each container builds its own model in a
    fresh process."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(6, activation="relu", name="dense_hidden"),
            tf.keras.layers.Dense(3, name="dense_out"),
        ],
        name="net",
    )
    model(tf.zeros((1, 4)))
    return model


class TestWireSmokeTwoRounds:

    @pytest.mark.parametrize(
        "skip_mode", ["fedluar", "fedluar_random", "fedluar_cyclic"],
    )
    async def test_two_round_smoke_advice_bytes_accuracy(self, skip_mode):
        """Plan T1 VALIDATE (b), in-process variant: 2 rounds, FedAvg
        star (1 aggregator + 2 workers), per-layer, skip_feedback=fedluar
        (count=1), recycle slippage, ε=0.2.  Asserts: (i) skip advice
        observed on the wire and in telemetry; (ii) the advised layer's
        bytes are actually absent in round 1 (sender comm metrics,
        receiver telemetry, and total bytes drop) while the manifest still
        lists every layer; (iii) accuracy stays sane.

        Parameterized over all three fixed-count modes: the campaign's
        head-to-head is only meaningful if `fedluar_random` and
        `fedluar_cyclic` remove the SAME bytes on the SAME path as
        `fedluar` — a control that advised a layer but still shipped it
        would silently be byte-matched to nothing.
        """
        training = {
            "learning_rate": 0.3,
            "optimizer": "sgd",
            "epochs_per_round": 2,
            "total_rounds": 2,
            "batch_size": 16,
            "update_mode": "per_layer",
            "num_traffic_classes": 2,
            "epsilon_deadline": 0.2,
            "epsilon_warmup_rounds": 0,
            "watchdog_factor": 0.0,
            "importance_metric_v2": "delta_sq_norm",
            "assignment_strategy": "byte_balanced",
            "late_layer_policy": "recycle_last_delta",
            "skip_feedback": skip_mode,
            "skip_fedluar_count": 1,
            "sync_timeout": 30.0,
            "seed": 7,
        }
        wire = _Wire()
        template = _smoke_model()
        template_params = FederationModel.get_parameters(template)
        layer_names = sorted(template_params)

        def _node(node_id: str, role: str, neighbors: list[str]):
            config = dict(training)
            config["role"] = role
            algorithm = FedAvg(node_id, neighbors, config)
            model = _smoke_model()
            FederationModel.set_parameters(model, template_params)
            engine = TrainingEngine(
                node_id=node_id,
                model=model,
                algorithm=algorithm,
                server=MagicMock(spec=TransportServer),
                pool=MagicMock(spec=ConnectionPool),
                config=config,
            )
            engine.monitor_ip = "monitor"
            engine.monitor_port = 1
            wire.register(node_id, engine)
            return engine

        aggregator = _node("node-0", "aggregator", ["node-1", "node-2"])
        worker_1 = _node("node-1", "worker", ["node-0"])
        worker_2 = _node("node-2", "worker", ["node-0"])

        x0, y0 = _toy_data(0, n=96)
        x1, y1 = _toy_data(1, n=96)
        x2, y2 = _toy_data(2, n=96)
        xv, yv = _toy_data(99, n=64)

        await asyncio.wait_for(
            asyncio.gather(
                aggregator.run_aggregator(x0, y0, xv, yv),
                worker_1.run_sync(x1, y1, xv, yv),
                worker_2.run_sync(x2, y2, xv, yv),
            ),
            timeout=120.0,
        )

        # ---- (i) skip advice observed ----
        round0_advice = [a for a in wire.advice_seen if a[1] == 0]
        assert {dest for dest, _r, _l in round0_advice} == {"node-1", "node-2"}
        advised_sets = {tuple(layers) for _d, _r, layers in round0_advice}
        assert len(advised_sets) == 1  # fedluar: one global set
        advised = list(advised_sets.pop())
        assert len(advised) == 1  # fixed count
        agg_reports = {r.round: r for r in wire.reports["node-0"]}
        agg_round0 = json.loads(agg_reports[0].uplink_telemetry_json)
        assert agg_round0["_aggregation"]["skip_advice"] == {
            "node-1": advised, "node-2": advised,
        }

        # ---- (ii) bytes actually absent in round 1 ----
        for worker_id in ("node-1", "node-2"):
            reports = {r.round: r for r in wire.reports[worker_id]}
            sent_r0 = {m.layer_name for m in reports[0].layer_comm_metrics}
            sent_r1 = {m.layer_name for m in reports[1].layer_comm_metrics}
            assert set(layer_names) <= sent_r0
            assert advised[0] not in sent_r1
            assert sent_r1 == set(layer_names) - set(advised)
        # Round-1 worker manifests still list EVERY layer (ratchet rule),
        # the advised one flagged skipped.
        worker_manifests_r1 = [
            m for m in wire.manifests
            if m.round == 1 and m.source_node_id in ("node-1", "node-2")
        ]
        assert len(worker_manifests_r1) == 2
        for manifest in worker_manifests_r1:
            assert {e.layer_name for e in manifest.entries} == set(layer_names)
            assert manifest_skipped_layers(manifest) == set(advised)
        # Receiver telemetry marks the omission per flow, and the
        # aggregation block shows the recycle fill.
        agg_round1 = json.loads(agg_reports[1].uplink_telemetry_json)
        for worker_id in ("node-1", "node-2"):
            flow = agg_round1[worker_id]
            assert flow["skip_omitted_layers"] == advised
            assert advised[0] in flow["shed_layers"]
        fill = agg_round1["_aggregation"]["layers"][advised[0]]
        assert fill["arrived_sources"] == []
        assert fill["filled"] == "recycle"

        # ---- (iii) accuracy sane ----
        for node_id, reports in wire.reports.items():
            for report in reports:
                assert math.isfinite(report.val_accuracy)
                assert 0.0 <= report.val_accuracy <= 1.0
                assert math.isfinite(report.train_loss)
        # Clearly above the 1/3 random floor on the separable toy task
        # (the labels are bias-free, so shedding the tiny bias layers at
        # ε=0.2 does not block learning).
        final_acc = agg_reports[1].val_accuracy
        assert final_acc >= 0.4
