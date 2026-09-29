"""Training engine: the main loop running inside each node container.

This is the central orchestrator for each federated learning node.  It
supports three execution modes:

**Synchronous** (``run_sync``):
    Every round runs the full pipeline:
    1. Local training → 2. Importance scores → 3. Send updates →
    4. Wait for all neighbor updates (blocking barrier) → 5. Aggregate →
    6. Evaluate → 7. Report metrics.
    Used by D-PSGD (decentralized) and FedAvg workers (centralized).

**Asynchronous** (``run_async``):
    Each iteration runs a similar pipeline but step 4 is non-blocking:
    1. Local training → 2. Importance scores → 3. Send updates →
    4. Check for buffered updates (non-blocking) → 5. Aggregate if any →
    6. Evaluate (periodic, every ``eval_every`` iterations) → 7. Report.
    Used by A-DPSGD and Gossip-SGD.

**Aggregator** (``run_aggregator``):
    Server-side loop for centralized algorithms (e.g. FedAvg):
    1. Wait for all workers → 2. Aggregate (weighted average) →
    3. Send global model back to workers → 4. Evaluate → 5. Report.
    The aggregator does NOT train locally.

The engine does NOT own the transport server (that lives in node.py and is
started before the engine so incoming messages are never dropped).  It does
own the optimizer, loss function, importance metric, and layer buffer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import tensorflow as tf

from src.algorithms.base import FederationAlgorithm, TrainingUpdate
from src.importance.assignment import (
    AssignmentResult,
    AssignmentStrategy,
    make_strategy,
)
from src.importance.base import ImportanceMetric
from src.importance.manifest import (
    build_manifest,
    manifest_skipped_layers,
    manifest_total,
    validate_manifest,
)
from src.training.late_layer_policy import make_policy
from src.models.base import FederationModel
from src.network.framing import HEADER_SIZE as FRAME_HEADER_SIZE
from src.network.connection_pool import ConnectionPool
from src.network.transport import TransportServer
from src.proto_gen import federation_pb2
from src.training.update import (
    BufferResult,
    LayerBuffer,
    deserialize_layer_update,
    deserialize_model_update,
    serialize_layer_updates,
    serialize_model_update,
    serialize_round_manifest,
)

# Populate the assignment-strategy registry (byte_balanced, coverage_eft,
# cyclic) as an import side effect.  The module is import-pure; a checkout
# that only carries the gap_based control adapter still works — strategies
# that were requested by config but are unavailable fail loudly inside
# make_strategy instead.
try:  # pragma: no cover - exercised implicitly via the registry
    import src.importance.strategies  # noqa: F401
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


# Estimated per-layer wire overhead on top of the raw array bytes: protobuf
# field tags, layer name, shape/dtype metadata, Envelope wrapper, and the
# 4-byte length-prefix frame.  Used for the assignment `sizes` input and the
# watchdog fluid bound; the exact value is uncritical (vs ~50 KB mean layer
# payloads) but it keeps tiny bias layers from being scored as 4-byte sends.
# Recorded here so analysis knows what the sizes meant.
_LAYER_ENVELOPE_OVERHEAD_BYTES = 96

# Reserved key inside `MetricsReport.uplink_telemetry_json` for sender-side
# bookkeeping (predicted t_eps, assignment diagnostics, cancelled zombie-tail
# bytes).  Node ids never start with "_", so consumers iterating "sender
# observed this round" entries must skip keys with this prefix (the schema in
# docs/extensions/04-overnight-interfaces.md §2 tells them to parse
# defensively).
_TELEMETRY_SENDER_KEY = "_sender"

# Reserved key for the aggregator's per-round aggregation realization
# (FedAvg slippage telemetry): which sources' values actually entered each
# layer's aggregate and which fill action covered the missing ones, plus
# cumulative per-(layer, source) inclusion counters.  Joined offline with
# the same report's per-source manifests/shed sets, this is the realized-κ
# input (gate doc §3 fix 1 / §5 item 4).  Same underscore-skip rule as
# `_sender`.
_TELEMETRY_AGGREGATION_KEY = "_aggregation"

# Reserved key for transport-level per-flow arrival stamps (audit NT-05).
# Receiver-side instrumentation used to hang off the importance manifest,
# which made it structurally unavailable to exactly the two flows the system
# is compared against — the monolithic ModelUpdate and the aggregator's
# 2.25 MB downlink broadcast, neither of which carries a manifest.  Every
# inbound envelope is stamped at the transport instead, so those flows now
# have a receiver clock and a wire-comparable completion time.  Same
# underscore-skip rule as `_sender`.
_TELEMETRY_FLOWS_KEY = "_receiver_flows"

# Reserved key for the clock-domain legend (audit NT-03).  The enabling
# condition for putting a sender-side buffer-accept time and a receiver-side
# coverage time on one axis was that both were bare floats named `*_s`; every
# exported time now carries its domain in the key name, and this block states
# the convention in-band plus the evidence for the shared-clock premise the
# `_oneway_s` values rest on.
_TELEMETRY_CLOCK_DOMAINS_KEY = "_clock_domains"

_CLOCK_DOMAIN_SUFFIXES = {
    "_sender_s": (
        "interval measured entirely on the sending node's CLOCK_MONOTONIC"
    ),
    "_receiver_s": (
        "interval or offset measured entirely on the receiving node's "
        "CLOCK_MONOTONIC (offsets are relative to the receiver's round start)"
    ),
    "_oneway_s": (
        "receiver arrival stamp minus Envelope.t_send_start_sender_s: a true "
        "one-way wire time, valid only because all containers share the host "
        "kernel clock"
    ),
    "_model_s": (
        "predicted by the assignment model, measured on no clock at all; "
        "never comparable with a measured time without saying so"
    ),
}


def _json_safe(obj: Any) -> Any:
    """`json.dumps` fallback: numpy scalars -> float, anything else -> str.

    Strategy diagnostics are JSON-serializable by contract, but a stray
    np.float64 must degrade the value, never crash the round's report.
    """
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


# ---------------------------------------------------------------------------
# Importance metrics (v2 path)
# ---------------------------------------------------------------------------

class _DeltaSqNormMetric(ImportanceMetric):
    """`‖Δ_ℓ‖₂²` of the shipped update delta — the frozen trigger metric.

    Engine-local fallback so that G2 trigger accounting works even when the
    full metrics-v2 module is not present; the formula is unambiguous, so
    there is no drift risk versus the pluggable implementation.
    """

    def compute(
        self,
        layer_name: str,
        layer_weights: np.ndarray,
        layer_gradients: np.ndarray | None,
        round_num: int,
        context: dict,
    ) -> float:
        if layer_gradients is None:
            return 1.0  # fallback: equal importance (mirrors raw-norm metric)
        flat = np.asarray(layer_gradients, dtype=np.float64).ravel()
        return float(np.dot(flat, flat))


def _build_metric_v2(name: str) -> ImportanceMetric:
    """Instantiate a v2 importance metric by config literal.

    Prefers the pluggable factory (``src.importance.metrics_v2``,
    interface doc §3.1) so all four admitted literals resolve; falls back to
    engine-local implementations for the two literals the engine itself
    depends on (`delta_sq_norm` trigger accounting, `raw_norm` control) so
    the engine stays functional standalone.
    """
    try:
        from src.importance.metrics_v2 import make_metric_v2
        return make_metric_v2(name)
    except ImportError:
        pass
    if name == "delta_sq_norm":
        return _DeltaSqNormMetric()
    if name == "raw_norm":
        from src.importance.gradient_norm import GradientNormImportance
        return GradientNormImportance()
    raise ValueError(
        f"Importance metric v2 {name!r} requires src/importance/metrics_v2.py "
        f"(not importable in this checkout)"
    )


# ---------------------------------------------------------------------------
# Receive-side buffer with per-round ε, manifest-race re-check, force-fire
# ---------------------------------------------------------------------------

class _EngineLayerBuffer(LayerBuffer):
    """`LayerBuffer` extended with the gate-doc receive-path semantics.

    Additions over the base class (which is kept untouched for existing
    callers/tests):

    - **Per-round effective ε** (interface doc §1.8): the coverage trigger
      consults ``epsilon_for_round(round)`` instead of the fixed constructor
      value, implementing the ε warm-up schedule without any protocol
      signalling (sender and receiver derive it from config + round number).
    - **Manifest-race fix** (pre-run fix 4): ``register_manifest`` re-runs
      the coverage check so payloads that arrived *before* their manifest
      can complete the round at registration time.
    - **Degenerate-manifest guard** (pre-run fix 6): a manifest whose total
      raw score is <= 0 makes the coverage threshold vacuous, so the
      ε-trigger is disabled for that flow and completion falls back to the
      count-based trigger (all transmitted layers); flagged loudly once.
    - **`force_fire`** (watchdog G4): completes a flow with whatever has
      arrived, used by the engine's T_max watchdog timer.
    - **Covered-by-recycling credit** (skip-feedback v2, plan T1): manifest
      entries flagged ``skipped`` will never arrive, but their ``raw_score``
      stays in the denominator AND counts toward the coverage target — the
      aggregator recycle-fills them, so their mass is covered without wire
      bytes.  The count trigger likewise treats them as expected-absent:
      when every *transmitted* layer arrived, the flow completes with the
      skipped set reported as missing (it is genuinely absent from the
      reassembled update; aggregation fills it).
    """

    def __init__(
        self,
        epsilon: float = 0.0,
        late_layer_policy=None,
        epsilon_for_round: Callable[[int], float] | None = None,
    ):
        super().__init__(epsilon=epsilon, late_layer_policy=late_layer_policy)
        self._epsilon_for_round: Callable[[int], float] = (
            epsilon_for_round if epsilon_for_round is not None
            else (lambda _round: self.epsilon)
        )
        self._degenerate_warned: set[tuple[str, int]] = set()

    def add_layer(
        self,
        source_node: str,
        round_num: int,
        layer_name: str,
        layer_index: int,
        total_layers: int,
        array: np.ndarray,
        num_samples: int,
    ) -> BufferResult | None:
        key = (source_node, round_num)

        if key in self._fired:
            # Late arrival — delegate to the base class, which routes it to
            # the late-layer policy.
            return super().add_layer(
                source_node=source_node,
                round_num=round_num,
                layer_name=layer_name,
                layer_index=layer_index,
                total_layers=total_layers,
                array=array,
                num_samples=num_samples,
            )

        self._buffers.setdefault(key, {})[layer_name] = array
        if key not in self._metadata:
            self._metadata[key] = (total_layers, num_samples)

        # Trigger 1: "all transmitted layers in" — count-based.  With
        # omission-aware senders (cyclic schedules, skip-feedback omission)
        # total_layers counts only the layers actually transmitted this
        # round, so this also covers the ε=0 / warm-up case exactly.  When
        # the manifest lists skipped entries, those are genuinely absent
        # from the reassembled update — report them as missing so the
        # telemetry shed set and the aggregator's fill bookkeeping see them
        # (the same accounting the coverage trigger produces).
        if len(self._buffers[key]) >= total_layers:
            skipped_missing = self._skipped_not_received(key)
            return self._fire(
                key,
                is_partial=bool(skipped_missing),
                missing=skipped_missing,
            )

        # Trigger 2: ε-coverage — manifest-driven, per-round ε.
        return self.check_coverage(key)

    def _skipped_not_received(self, key: tuple[str, int]) -> frozenset[str]:
        """Manifest entries flagged ``skipped`` that have not arrived.

        Defensive on both ends: no manifest means no skip knowledge (empty
        set, the legacy count-trigger semantics), and a skipped layer that
        arrived anyway (advice raced a must-send override) is not reported
        missing.
        """
        manifest = self._manifests.get(key)
        if manifest is None:
            return frozenset()
        skipped = manifest_skipped_layers(manifest)
        if not skipped:
            return frozenset()
        return frozenset(skipped - set(self._buffers.get(key, ())))

    def check_coverage(self, key: tuple[str, int]) -> BufferResult | None:
        """Run the ε-coverage trigger check for ``key`` against buffered layers.

        Shared by ``add_layer`` (payload arrivals) and ``register_manifest``
        (manifest-after-payload race).  Returns a `BufferResult` when the
        trigger fires, else None.
        """
        if key in self._fired or key not in self._manifests:
            return None
        received = self._buffers.get(key)
        if not received:
            return None
        epsilon = float(self._epsilon_for_round(key[1]))
        if epsilon <= 0.0:
            return None  # warm-up / ε=0: only the count trigger completes
        manifest = self._manifests[key]
        scores = {e.layer_name: e.raw_score for e in manifest.entries}
        total_score = sum(scores.values())
        if total_score <= 0.0:
            # Degenerate manifest: (1-ε)·total == 0 would fire on the first
            # arrival regardless of content.  Fall back to the count-based
            # trigger (all transmitted layers) instead.
            if key not in self._degenerate_warned:
                self._degenerate_warned.add(key)
                logger.warning(
                    "DEGENERATE MANIFEST from %s round %d: total raw_score "
                    "%.4g <= 0 — ε-coverage trigger disabled for this flow; "
                    "falling back to count-based completion (all layers)",
                    key[0], key[1], total_score,
                )
            return None
        must_receive = {
            e.layer_name for e in manifest.entries if e.must_receive
        }
        target = (1.0 - epsilon) * total_score
        received_names = received.keys()
        received_score = sum(scores.get(n, 0.0) for n in received_names)
        # Covered-by-recycling credit (skip-feedback v2): entries flagged
        # `skipped` never travel, but the aggregator recycle-fills them, so
        # their trigger mass counts as covered.  Crucially their raw_score
        # ALSO stays inside total_score above — the denominator covers the
        # full per-round mass, killing the v1 shrinking-denominator ratchet
        # (each skip round sheds ε of the full mass, not of a remnant).
        # Defensive: a skipped layer that arrived anyway is counted once,
        # as received.
        skip_credit = sum(
            e.raw_score
            for e in manifest.entries
            if e.skipped and e.layer_name not in received_names
        )
        if (
            must_receive.issubset(received_names)
            and received_score + skip_credit >= target
        ):
            missing = frozenset(scores.keys()) - frozenset(received_names)
            return self._fire(key, is_partial=True, missing=missing)
        return None

    def register_manifest(
        self, manifest: federation_pb2.RoundManifest
    ) -> BufferResult | None:
        """Register a manifest and re-evaluate coverage (pre-run fix 4).

        Payloads may have raced ahead of their manifest (separate sockets);
        without this re-check an already-sufficient buffer would only
        complete on the *next* payload arrival — or never, if the sender
        shed the rest of the round.
        """
        super().register_manifest(manifest)
        return self.check_coverage((manifest.source_node_id, manifest.round))

    def force_fire(self, key: tuple[str, int]) -> BufferResult | None:
        """Complete ``key`` with whatever arrived (watchdog G4).

        Returns None when the flow already completed naturally (the timer
        lost the race).  Safe with zero arrivals: the resulting update then
        carries no parameters and num_samples=0, which aggregation treats
        as an all-missing partial contribution.
        """
        if key in self._fired:
            return None
        manifest = self._manifests.get(key)
        received = self._buffers.setdefault(key, {})
        if key not in self._metadata:
            announced = len(manifest.entries) if manifest is not None else 0
            self._metadata[key] = (announced, 0)
        if manifest is not None:
            missing = frozenset(
                e.layer_name for e in manifest.entries
            ) - frozenset(received)
        else:
            missing = frozenset()
        return self._fire(key, is_partial=True, missing=missing)


@dataclass
class _SendPlan:
    """Everything `_send_updates` needs for one per-layer dissemination round.

    Built once per round (worker uplink via `_prepare_layer_dissemination`,
    aggregator downlink via `_downlink_plan`) and applied to every
    destination, honouring the assign-once-per-round contract.

    Attributes:
        round_num: The round the plan belongs to.
        assignment: The strategy's `AssignmentResult`; iteration order of
            ``assignment.assignment`` is the send order (within a class the
            strategy's order, e.g. utility-density for coverage scheduling).
        trigger_scores: Frozen ε-trigger accounting scores (manifest
            ``raw_score``).  None means "send no manifest" — the downlink
            case, where workers must receive the full model and coverage
            semantics do not apply.
        sched_scores: Scheduling scores for the manifest ``sched_score``
            field, or None when they equal the trigger scores (the wire
            then keeps the 0.0 = "same as raw_score" convention).
        must_receive: Layers flagged must-receive in the manifest (staleness
            cap from aging); always a subset of the assigned layers.
        skipped: Layers omitted this round on aggregator skip advice
            (skip-feedback v2).  Disjoint from the assignment; still
            manifest-listed with their trigger mass, flagged ``skipped``
            (covered-by-recycling accounting at the receiver).  Empty for
            non-skip arms and for the downlink.
        log_scores: Per-layer scores written to ``layer_comm_metrics``
            ``importance`` for telemetry only — decoupled from
            ``trigger_scores`` so the downlink can log its real
            ``delta_sq_norm`` motion without triggering a manifest (G5).
            None falls back to ``trigger_scores`` (the uplink case, where
            the manifest scores and the logged scores coincide).
    """

    round_num: int
    assignment: AssignmentResult
    trigger_scores: dict[str, float] | None
    sched_scores: dict[str, float] | None
    must_receive: set[str]
    skipped: frozenset[str] = frozenset()
    log_scores: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Training engine
# ---------------------------------------------------------------------------

class TrainingEngine:
    """Main training engine for a federated learning node.

    Created by ``node.py`` after the model, algorithm, and network
    components are ready.  The engine's message handler must be registered
    on the transport server *before* neighbor connections are established
    (see node.py) so that no incoming updates are dropped.
    """

    def __init__(
        self,
        node_id: str,
        model: tf.Module,
        algorithm: FederationAlgorithm,
        server: TransportServer,
        pool: ConnectionPool,
        config: dict,
    ):
        self.node_id = node_id
        self.model = model
        self.algorithm = algorithm
        self.server = server
        self.pool = pool
        self.config = config

        # ---- Optimizer and loss (engine owns these, not the model) ----
        lr = config.get("learning_rate", 0.01)
        opt_name = config.get("optimizer", "sgd")
        # Baseline fixes (writeup/16 §5): clipnorm=None and
        # lr_schedule='constant' reproduce the legacy optimizer exactly.
        self._base_lr: float = float(lr)
        opt_kwargs: dict[str, Any] = {}
        if config.get("clipnorm") is not None:
            opt_kwargs["clipnorm"] = float(config["clipnorm"])
        if opt_name == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=config.get("momentum", 0.0),
                **opt_kwargs,
            )
        elif opt_name == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr, **opt_kwargs
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {opt_name!r}. "
                f"Supported: 'sgd', 'adam'."
            )

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # ---- Config values ----
        self.epochs_per_round = config.get("epochs_per_round", 1)
        self.total_rounds = config.get("total_rounds", 50)
        self.batch_size = config.get("batch_size", 64)
        self.update_mode = config.get("update_mode", "monolithic")
        self.num_traffic_classes = config.get("num_traffic_classes", 1)

        # ---- Async-specific config ----
        # eval_every: how often to run validation in async mode.  In sync
        # mode evaluation happens every round regardless of this value.
        # Default 1 means evaluate every iteration (same as sync).
        self.eval_every: int = config.get("eval_every", 1)

        # ---- Importance metrics (per-layer mode only) ----
        # Gate ruling G2 splits scoring in two: trigger accounting is FROZEN
        # on delta-sq-norm of the shipped multi-epoch delta for every arm,
        # while the configured `importance_metric_v2` only steers scheduling
        # (manifest sched_score + assignment strategy).  Single exception:
        # the raw_norm control arm reproduces the legacy E2 configuration
        # exactly — raw L2 norm of the last minibatch gradient as BOTH
        # trigger and sched score (its native config, by design).
        #
        # The legacy `importance_metric` knob is superseded by this path and
        # ignored (raw_norm over last-batch gradients IS the old
        # gradient_norm behaviour).
        self.importance_metric_v2_name: str = config.get(
            "importance_metric_v2", "raw_norm"
        )
        self.assignment_strategy_name: str = config.get(
            "assignment_strategy", "gap_based"
        )
        self._trigger_metric: ImportanceMetric | None = None
        self._sched_metric: ImportanceMetric | None = None
        self._assignment_strategy: AssignmentStrategy | None = None
        if self.update_mode == "per_layer":
            if config.get("importance_metric"):
                logger.info(
                    f"[{node_id}] Legacy 'importance_metric' "
                    f"({config['importance_metric']!r}) is superseded by "
                    f"'importance_metric_v2' "
                    f"({self.importance_metric_v2_name!r}) and ignored"
                )
            if self.importance_metric_v2_name == "raw_norm":
                self._trigger_metric = _build_metric_v2("raw_norm")
                self._sched_metric = None  # sched == trigger in the control
            else:
                self._trigger_metric = _build_metric_v2("delta_sq_norm")
                self._sched_metric = (
                    None
                    if self.importance_metric_v2_name == "delta_sq_norm"
                    else _build_metric_v2(self.importance_metric_v2_name)
                )
            strategy_kwargs: dict[str, Any] = {}
            if self.assignment_strategy_name == "cyclic":
                strategy_kwargs["cyclic_k"] = int(config.get("cyclic_k", 0))
            elif self.assignment_strategy_name == "coverage_eft":
                # The FedLUAR-style randomized tail boundary lives inside
                # the coverage strategy (constructor knob), not in the
                # aging score transform; seeded for reproducibility.
                strategy_kwargs["seed"] = int(config.get("seed", 42))
                if config.get("aging_mode") == "stochastic_tail":
                    strategy_kwargs["stochastic_tail"] = True
                # ε budget units (audit TRIG-5): 'trigger' also meters the
                # shed against the frozen trigger mass the receiver
                # enforces, so arms with different sched metrics are
                # coverage-matched.  'sched' reproduces pre-fix campaigns.
                strategy_kwargs["budget_metric"] = str(
                    config.get("epsilon_budget_metric", "trigger")
                )
            self._assignment_strategy = make_strategy(
                self.assignment_strategy_name, **strategy_kwargs
            )

        # ---- Skip-feedback v2 (plan T1) ----
        # Aggregator side: after each aggregation, per-sender skip advice
        # from the algorithm (`last_skip_advice`) is piggybacked on the
        # broadcast (round-tagged SkipAdvice envelopes, class 0, sent
        # before the model payload).  Sender side: advice tagged round r
        # applies to the round r+1 dissemination plan ONLY — stale tags are
        # ignored (fail-open), re-delivery overwrites the same slot
        # (idempotent).  Layers omitted on advice remain manifest-listed,
        # flagged `skipped`, with their full trigger mass.
        self.skip_feedback_mode: str = (
            config.get("skip_feedback", "off") or "off"
        )
        if self.skip_feedback_mode not in (
            "off", "shed", "fedluar", "fedluar_random", "fedluar_cyclic",
        ):
            raise ValueError(
                f"Unknown skip_feedback {self.skip_feedback_mode!r}. "
                "Valid values: off, shed, fedluar, fedluar_random, "
                "fedluar_cyclic"
            )
        # advice round (the round whose aggregation produced it) -> layers.
        self._skip_advice: dict[int, frozenset[str]] = {}

        # ---- Aging (starvation control, sched scores only — G2) ----
        self.aging_mode: str = config.get("aging_mode", "none") or "none"
        self.aging_lambda = float(config.get("aging_lambda", 0.0))
        self.aging_tau_max = int(config.get("aging_tau_max", 0))
        # Which event resets an age (audit TRIG-1/ML-01).  'inclusion' (the
        # default) resets only on the aggregator's acknowledgement that the
        # layer's value ENTERED THE AGGREGATE, so aging_tau_max bounds
        # realized staleness; 'head_placement' is the pre-fix proxy, where a
        # layer could be head-placed (age reset) every round and shed at the
        # receiver every round — measured absence runs of 12-20 at τ_max=3.
        self.aging_age_basis: str = (
            config.get("aging_age_basis", "inclusion") or "inclusion"
        )
        if self.aging_age_basis not in ("inclusion", "head_placement"):
            raise ValueError(
                f"Unknown aging_age_basis {self.aging_age_basis!r}. "
                f"Valid values: inclusion, head_placement"
            )
        # rounds since the layer's contribution last entered the aggregate
        # ('inclusion' basis) or was last head-placed ('head_placement').
        self._layer_ages: dict[str, int] = {}
        # Realized-inclusion acks from the aggregator, keyed by the round
        # whose aggregation produced them: {round: frozenset(layers)}.
        self._inclusion_acks: dict[int, frozenset[str]] = {}
        # Whether this node has ever received an ack; a node with no ack
        # channel at all (decentralized algorithms) keeps the pre-fix basis
        # rather than ageing every layer to the cap, and says so loudly.
        self._inclusion_ack_seen = False
        self._age_basis_warned = False
        # The basis actually used for the most recent round's ages, exported
        # in the `_sender` telemetry block so the invariant is checkable.
        self._age_basis_realized = self.aging_age_basis
        self._aging_rng = np.random.default_rng(int(config.get("seed", 42)))
        self._apply_aging_fn: Callable | None = None
        self._aging_unavailable_warned = False

        # ---- Layer buffer for per-layer reassembly ----
        # When receiving per-layer updates from neighbors, individual
        # LayerUpdate messages are accumulated here until all layers for a
        # (source, round) pair have arrived, at which point a complete
        # TrainingUpdate is handed to the algorithm.
        # ε-deadline coverage tolerance; 0.0 preserves legacy "wait for
        # every layer" behaviour.  Only consulted in per-layer mode.
        # See docs/extensions/02-epsilon-trigger.md.
        self.epsilon_deadline = float(config.get("epsilon_deadline", 0.0))
        # Round-indexed ε schedule (warm-up): rounds < epsilon_warmup_rounds
        # run at ε=0.  Computed, never signalled — sender and receiver derive
        # the same value from config + round number (interface doc §1.8).
        self.epsilon_warmup_rounds = int(config.get("epsilon_warmup_rounds", 0))
        # Late-layer policy: what to do with layers that arrive after the
        # ε-trigger fired.  See docs/extensions/03-late-layer-policy.md.
        self.late_layer_policy = make_policy(
            config.get("late_layer_policy", "drop")
        )
        self._layer_buffer = _EngineLayerBuffer(
            epsilon=self.epsilon_deadline,
            late_layer_policy=self.late_layer_policy,
            epsilon_for_round=self._effective_epsilon,
        )

        # ---- Receiver-side watchdog (gate ruling G4) ----
        # A (source, round) flow is force-completed T_max = watchdog_factor
        # x fluid-bound seconds after its manifest arrives; <= 0 disables.
        self.watchdog_factor = float(config.get("watchdog_factor", 3.0))
        self._watchdog_tasks: dict[tuple[str, int], asyncio.Task] = {}

        # ---- Edge bandwidth info (set by node.py after construction) ----
        # outgoing_edges feeds the assignment strategies' `bandwidths` input
        # and the SNDBUF-aware send path; incoming_edges feeds the watchdog
        # fluid bound.  Both default to empty, in which case bandwidths are
        # treated as unshaped (inf) and the watchdog stays disarmed.
        self.outgoing_edges: list[dict] = []
        self.incoming_edges: list[dict] = []

        # ---- Receiver-side uplink telemetry (gate ruling G1) ----
        # Per (source, round): absolute receiver-local monotonic stamps for
        # manifest arrival, per-layer arrivals, and trigger fire, plus the
        # realized shed set.  Exported per round as offsets relative to the
        # receiver's round-start instant (schema: extension doc §2) inside
        # MetricsReport.uplink_telemetry_json.
        self._uplink_flows: dict[tuple[str, int], dict[str, Any]] = {}
        self._round_starts: dict[int, float] = {}

        # ---- Transport-level flow stamps (audit NT-05) ----
        # (source, round, payload_type) -> first/last byte receiver stamps and
        # the earliest sender send-start stamp seen on the flow.  Fed by the
        # arrival observer for EVERY inbound envelope, so manifest-less flows
        # (monolithic uplink, downlink broadcast) get a receiver clock and a
        # one-way completion time instead of nothing at all.
        self._flow_stamps: dict[tuple[str, int, str], dict[str, Any]] = {}
        # Stamps for the envelope currently being dispatched, as
        # (envelope, first_byte_s, last_byte_s).  The observer runs
        # synchronously immediately before the handler with no await in
        # between, so the handler reads the true last-byte instant instead of
        # a post-parse time.monotonic(); the identity check keeps direct
        # handler calls (tests, non-transport dispatch) correct.
        self._arrival_stamps: tuple[Any, float, float] | None = None

        # ---- Zombie-tail bookkeeping (pre-run fix 3) ----
        # Tail (deadline-exempt) envelopes are sent by background per-class
        # tasks so the round can progress; when the round closes (broadcast
        # received / next round opens) the unsent remainder is cancelled
        # between envelopes and the saved bytes are logged.
        self._tail_tasks: dict[int, list[asyncio.Task]] = {}
        self._tail_cancel_events: dict[int, asyncio.Event] = {}
        self._tail_ledgers: dict[int, dict[str, Any]] = {}
        self._sender_events: list[dict[str, Any]] = []
        self._reported_rounds: set[int] = set()

        # ---- Sender-side per-round assignment info (for the report) ----
        self._assignment_info: dict[int, dict[str, Any]] = {}

        # ---- Aggregation realization (aggregator role only) ----
        # Snapshot of the algorithm's slippage telemetry taken right after
        # each aggregate() call (the attributes are refreshed in place by
        # the next aggregation, so they must be captured per round).
        self._aggregation_info: dict[int, dict[str, Any]] = {}

        # True only inside run_aggregator; the broadcast-arrival zombie
        # cancellation hook must not trigger on the aggregator itself.
        self._is_aggregator = False
        # Receiver-side per-layer byte sizes (watchdog fluid bound), cached
        # because trainable-variable shapes never change within a run.
        self._receiver_sizes_cache: dict[str, int] | None = None
        self._downlink_strategy: AssignmentStrategy | None = None
        self._downlink_fallback_warned = False

        # ---- Round manifests (one per (round, source_node) in per-layer mode) ----
        # Populated by the message handler when a RoundManifest envelope is
        # received before its layer payloads.  Consumed by the ε-trigger
        # (extension 02) to compute the absolute completion threshold
        # (1−ε)·Σ raw_score for a given (round, source) pair.
        # Structure: {round_num: {source_node_id: RoundManifest}}
        self._round_manifests: dict[int, dict[str, federation_pb2.RoundManifest]] = {}

        # ---- Last-batch gradients (captured during training) ----
        # Only the gradients from the final mini-batch of the last epoch
        # are stored.  Since the metric-input swap (gate headline fix 3)
        # the scheduling/trigger metrics consume the shipped multi-epoch
        # delta instead; this proxy remains the input of the raw_norm
        # control arm only (its native E2 configuration).
        self.last_gradients: dict[str, np.ndarray] | None = None

        # ---- Metrics history ----
        self.metrics_history: list[dict[str, Any]] = []

        # ---- Divergence markers (audit ML-04) ----
        # {round: first failed check}.  Divergence produces IN-RANGE values,
        # not missing ones — NaN logits make tf.argmax return class 0, so a
        # collapsed CIFAR-10 run reports val_accuracy = 0.100 (chance) and
        # every downstream mean accepts it.  The run is never aborted
        # (collapse studies need the trajectory); the marker travels in the
        # MetricsReport so analyses can gate on it.
        self._diverged_rounds: dict[int, str] = {}

        # ---- Monitor connection info (set by node.py) ----
        self.monitor_ip: str | None = None
        self.monitor_port: int | None = None

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _register_message_handler(self) -> None:
        """Register a callback on the transport server to dispatch incoming
        protobuf envelopes to the correct handler.

        This must be called *before* any TCP connections are accepted from
        neighbors, otherwise messages that arrive during the connection
        phase would be silently dropped (the server ignores messages when
        no handler is set).

        Monolithic updates are deserialized in one shot.  Per-layer updates
        go through the LayerBuffer, which reassembles individual
        LayerUpdate messages into a complete TrainingUpdate once all layers
        for a given (source_node, round) pair have been received.
        """

        async def handler(envelope: federation_pb2.Envelope) -> None:
            try:
                await dispatch(envelope)
            finally:
                # Release the arrival-stamp slot: it holds a reference to the
                # envelope (up to a whole serialized model) purely so
                # `_arrival_stamp` can identify it during this dispatch.
                self._arrival_stamps = None

        async def dispatch(envelope: federation_pb2.Envelope) -> None:
            payload_type = envelope.WhichOneof("payload")
            if payload_type == "model_update":
                update = deserialize_model_update(envelope)
                await self.algorithm.on_update_received(update)
                # Centralized worker: the aggregator's broadcast for round r
                # implies round r is closed — any still-queued tail of
                # rounds <= r is zombie traffic (pre-run fix 3).
                self._maybe_signal_rounds_closed(update.round_num)
            elif payload_type == "round_manifest":
                arrival, wire_time = self._arrival_stamp(envelope)
                result = self._on_round_manifest(
                    envelope.round_manifest, arrival, wire_time,
                )
                if result is not None:
                    # Manifest-after-payload race (pre-run fix 4): the
                    # already-buffered payloads satisfied the coverage
                    # trigger the moment the manifest registered.
                    logger.info(
                        f"[{self.node_id}] ε-trigger fired at manifest "
                        f"registration (payloads raced ahead) for "
                        f"source={result.update.source_node} "
                        f"round={result.update.round_num}; "
                        f"missing={sorted(result.missing_layers)}"
                    )
                    self._finalize_uplink_flow(result, time.monotonic())
                    await self.algorithm.on_update_received(result.update)
            elif payload_type == "layer_update":
                now, wire_time = self._arrival_stamp(envelope)
                (
                    source,
                    round_num,
                    layer_idx,
                    total_layers,
                    array,
                    num_samples,
                ) = deserialize_layer_update(envelope)
                self._observe_layer_arrival(
                    source, round_num, envelope.layer_update.layer_name, now,
                    wire_time,
                )
                result = self._layer_buffer.add_layer(
                    source_node=source,
                    round_num=round_num,
                    layer_name=envelope.layer_update.layer_name,
                    layer_index=layer_idx,
                    total_layers=total_layers,
                    array=array,
                    num_samples=num_samples,
                )
                if result is not None:
                    # Either all transmitted layers arrived
                    # (is_partial=False) or the ε-trigger fired before the
                    # tail arrived (is_partial=True).  Forward what we have;
                    # late layers go to the late-layer policy (extension 03).
                    if result.is_partial:
                        logger.info(
                            f"[{self.node_id}] ε-trigger fired for "
                            f"source={result.update.source_node} "
                            f"round={result.update.round_num}; "
                            f"missing={sorted(result.missing_layers)}"
                        )
                    self._finalize_uplink_flow(result, now)
                    await self.algorithm.on_update_received(result.update)
                    self._maybe_signal_rounds_closed(result.update.round_num)
            elif payload_type == "skip_advice":
                self._on_skip_advice(envelope.skip_advice)
            elif payload_type == "control_signal":
                action = envelope.control_signal.action
                if action == federation_pb2.ControlSignal.SHUTDOWN:
                    logger.info(f"[{self.node_id}] Received shutdown signal")
            else:
                logger.warning(
                    f"[{self.node_id}] Unknown payload type: {payload_type}"
                )

        self.server.set_handler(handler)
        self.server.set_arrival_observer(self._observe_envelope_arrival)

    # ------------------------------------------------------------------
    # Transport-level arrival stamps (audit NT-05)
    # ------------------------------------------------------------------

    @staticmethod
    def _envelope_round(envelope: federation_pb2.Envelope) -> int | None:
        """Round number carried by an envelope, or None if it carries none."""
        payload_type = envelope.WhichOneof("payload")
        if payload_type == "model_update":
            return int(envelope.model_update.round)
        if payload_type == "layer_update":
            return int(envelope.layer_update.round)
        if payload_type == "round_manifest":
            return int(envelope.round_manifest.round)
        return None

    def _observe_envelope_arrival(
        self,
        envelope: federation_pb2.Envelope,
        first_byte_s: float,
        last_byte_s: float,
    ) -> None:
        """Record transport-level arrival stamps for one inbound envelope.

        Registered on the transport server, so it sees *every* flow rather
        than only manifested per-layer uplinks: the monolithic ModelUpdate
        and the aggregator's downlink broadcast finally get a receiver clock,
        which is what turns the headline monolithic-vs-per-layer ratio from
        measured-over-modelled into measured-over-measured.

        Per (source, round, payload type) the ledger keeps the first and last
        byte instants and the earliest sender send-start stamp on the flow,
        from which the export derives one-way wire times.
        """
        self._arrival_stamps = (envelope, first_byte_s, last_byte_s)
        round_num = self._envelope_round(envelope)
        if round_num is None:
            return  # control traffic (skip advice, barriers): no flow to time
        payload_type = envelope.WhichOneof("payload")
        key = (envelope.source_node, round_num, payload_type)
        # Framed size: the length prefix is on the wire too.
        wire_bytes = envelope.ByteSize() + FRAME_HEADER_SIZE
        send_start = envelope.t_send_start_sender_s
        entry = self._flow_stamps.get(key)
        if entry is None:
            entry = {
                "first_byte": first_byte_s,
                "last_byte": last_byte_s,
                "send_start": send_start if send_start > 0 else None,
                "messages": 0,
                "bytes": 0,
            }
            self._flow_stamps[key] = entry
        else:
            entry["first_byte"] = min(entry["first_byte"], first_byte_s)
            entry["last_byte"] = max(entry["last_byte"], last_byte_s)
            if send_start > 0:
                entry["send_start"] = (
                    send_start if entry["send_start"] is None
                    else min(entry["send_start"], send_start)
                )
        entry["messages"] += 1
        entry["bytes"] += wire_bytes

    def _arrival_stamp(
        self, envelope: federation_pb2.Envelope
    ) -> tuple[float, float | None]:
        """Receiver last-byte instant and one-way wire time for *envelope*.

        Returns ``(arrival_receiver_s, wire_time_oneway_s)``.  The wire time
        is ``None`` when the sender did not stamp the envelope (proto3 default
        0.0 means "absent", never "epoch") — the honest answer for a
        pre-fix sender or an envelope that never went through
        ``ConnectionPool.send``.  A negative result is passed through so the
        export can count it: negative one-way times are the proof that the two
        ends are NOT on one clock, and must surface as an anomaly rather than
        be silently clipped into plausibility.

        Falls back to ``time.monotonic()`` when the slot does not hold this
        envelope: direct handler dispatch (tests, non-transport callers), or —
        should the "no await between observer and handler" invariant ever
        break — a concurrent connection having overwritten the slot.  The
        identity check turns that into a slightly late stamp instead of a
        stamp attributed to the wrong flow.
        """
        stamps = self._arrival_stamps
        if stamps is not None and stamps[0] is envelope:
            arrival = stamps[2]
        else:
            arrival = time.monotonic()
        send_start = envelope.t_send_start_sender_s
        wire_time = arrival - send_start if send_start > 0 else None
        return arrival, wire_time

    def _on_round_manifest(
        self,
        manifest: federation_pb2.RoundManifest,
        arrival: float | None = None,
        wire_time: float | None = None,
    ) -> BufferResult | None:
        """Buffer a `RoundManifest` for later use by the ε-trigger.

        The manifest is validated immediately; malformed manifests are logged
        and dropped (the round will then degrade to "wait for all layers"
        behaviour, which is safe).  See
        ``docs/extensions/01-importance-manifest.md``.

        Side effects (gate-doc pre-run fixes):
        - records the manifest arrival in the uplink telemetry (G1) and
          flags an ordering violation when payloads preceded it (fix 4);
        - warns loudly about degenerate manifests, which fall back to the
          count-based trigger (fix 6);
        - arms the T_max watchdog for the flow (G4);
        - re-evaluates coverage against already-buffered payloads and
          returns the resulting `BufferResult` when that completes the
          round (fix 4) — the caller must forward it to the algorithm.

        ``arrival`` / ``wire_time`` come from the transport arrival stamps
        (audit NT-05); both default to None for direct callers, in which case
        the arrival is taken here and no one-way time is recorded.
        """
        now = arrival if arrival is not None else time.monotonic()
        try:
            validate_manifest(manifest)
        except ValueError as exc:
            logger.warning(
                f"[{self.node_id}] Dropping malformed manifest from "
                f"{manifest.source_node_id!r} round={manifest.round}: {exc}"
            )
            return None

        record = self._uplink_record(manifest.source_node_id, manifest.round)
        if record["manifest_arrival"] is None:
            record["manifest_arrival"] = now
            # Algebraically the sender's own t_send_start_sender_s; kept as a
            # receiver-side subtraction so the flow record holds exactly the
            # two quantities the export needs.
            record["manifest_send_start"] = (
                now - wire_time if wire_time is not None else None
            )
            record["manifest_wire_time"] = wire_time
        if record["layer_arrivals"]:
            # Payload(s) beat the manifest: counted per pre-run fix 4.  The
            # invariant "manifest first on class 0" should make this rare;
            # nonzero rates point at class-0 queueing or socket issues.
            record["ordering_violation"] = True

        total = manifest_total(manifest)
        if total <= 0.0:
            logger.warning(
                f"[{self.node_id}] DEGENERATE MANIFEST from "
                f"{manifest.source_node_id} round={manifest.round}: total "
                f"raw_score {total:.4g} <= 0 — completion falls back to the "
                f"count-based trigger (all transmitted layers)"
            )

        per_source = self._round_manifests.setdefault(manifest.round, {})
        per_source[manifest.source_node_id] = manifest
        # Register with the layer buffer; this re-runs the coverage check so
        # payloads that raced ahead of the manifest can complete the round
        # immediately (pre-run fix 4).
        result = self._layer_buffer.register_manifest(manifest)
        if result is None:
            # Flow still open: bound its worst case with the watchdog (G4).
            self._arm_watchdog(manifest)
        logger.info(
            f"[{self.node_id}] Manifest from {manifest.source_node_id} "
            f"round={manifest.round}: "
            f"{len(manifest.entries)} entries, "
            f"total={total:.4g}"
        )
        return result

    # ------------------------------------------------------------------
    # Skip-feedback v2: advice intake (sender side)
    # ------------------------------------------------------------------

    def _on_skip_advice(self, advice: federation_pb2.SkipAdvice) -> None:
        """Store round-tagged skip advice + inclusion ack from the broadcast.

        Fail-open in both directions: with ``skip_feedback='off'`` the
        advice is dropped on the floor (this node never omits), and a node
        that never receives advice simply sends everything.  Idempotent:
        re-delivery overwrites the same ``round -> layers`` slot, so
        applying the advice twice yields the same omission set; a
        *differing* duplicate (should not happen — one aggregation per
        round) is logged and the last write wins.

        The realized-inclusion ack (audit TRIG-1/ML-01) rides the same
        envelope and is processed FIRST and unconditionally: it is the
        staleness mechanism's arming input, and it is meaningful in every
        arm, including the ones that never omit anything.
        """
        if advice.inclusion_ack:
            self._inclusion_acks[advice.round] = frozenset(
                advice.included_layers
            )
            self._inclusion_ack_seen = True
            logger.debug(
                f"[{self.node_id}] Inclusion ack for round {advice.round}: "
                f"{len(advice.included_layers)} layer(s) entered the aggregate"
            )
        if self.skip_feedback_mode == "off":
            logger.debug(
                f"[{self.node_id}] Ignoring skip advice for round "
                f"{advice.round} (skip_feedback=off)"
            )
            return
        layers = frozenset(advice.layer_names)
        previous = self._skip_advice.get(advice.round)
        if previous is not None and previous != layers:
            logger.warning(
                f"[{self.node_id}] Conflicting duplicate skip advice for "
                f"round {advice.round}: {sorted(previous)} -> "
                f"{sorted(layers)}; last write wins"
            )
        self._skip_advice[advice.round] = layers
        logger.info(
            f"[{self.node_id}] Skip advice for round {advice.round} "
            f"(applies to round {advice.round + 1}): "
            f"{len(layers)} layer(s) advised"
        )

    # ------------------------------------------------------------------
    # ε schedule
    # ------------------------------------------------------------------

    def _effective_epsilon(self, round_num: int) -> float:
        """Effective ε for ``round_num`` under the warm-up schedule.

        ``0.0`` for rounds below ``epsilon_warmup_rounds`` (full coverage
        during the critical learning period), ``epsilon_deadline``
        afterwards.  Sender (assignment) and receiver (trigger) both use
        this formula so the schedule needs no protocol signalling
        (interface doc §1.8).
        """
        if round_num < self.epsilon_warmup_rounds:
            return 0.0
        return self.epsilon_deadline

    # ------------------------------------------------------------------
    # Receiver-side uplink telemetry (gate ruling G1)
    # ------------------------------------------------------------------

    def _uplink_record(self, source: str, round_num: int) -> dict[str, Any]:
        """Get or create the telemetry record for a (source, round) flow."""
        key = (source, round_num)
        record = self._uplink_flows.get(key)
        if record is None:
            record = {
                "manifest_arrival": None,   # absolute monotonic stamps;
                "layer_arrivals": {},       # converted to round-relative
                "trigger_fire": None,       # offsets only at export time
                "watchdog_fired": False,
                "ordering_violation": False,
                "shed_layers": None,        # set at completion
                # One-way wire times (audit NT-01): receiver arrival minus the
                # sender's t_send_start_sender_s on the shared host clock.
                # None entries mean the sender did not stamp the envelope.
                "manifest_send_start": None,
                "manifest_wire_time": None,
                "layer_wire_times": {},
            }
            self._uplink_flows[key] = record
        return record

    def _observe_layer_arrival(
        self,
        source: str,
        round_num: int,
        layer_name: str,
        now: float,
        wire_time: float | None = None,
    ) -> None:
        """Record a payload arrival (first arrival wins on duplicates).

        ``wire_time`` is the layer's one-way wire time (audit NT-01): the
        interval from the instant the sender handed the layer's first byte to
        its class socket to the instant this receiver read its last byte.  It
        replaces the buffer-accept ``send_duration`` as the per-layer time
        that actually describes the wire.
        """
        record = self._uplink_record(source, round_num)
        if layer_name in record["layer_arrivals"]:
            return  # duplicate delivery: the first arrival is the arrival
        record["layer_arrivals"][layer_name] = now
        if wire_time is not None:
            record["layer_wire_times"][layer_name] = wire_time

    def _finalize_uplink_flow(
        self,
        result: BufferResult,
        now: float,
        watchdog_fired: bool = False,
    ) -> None:
        """Stamp flow completion and cancel the flow's watchdog."""
        key = (result.update.source_node, result.update.round_num)
        self._cancel_watchdog(key)
        record = self._uplink_record(*key)
        record["trigger_fire"] = now
        record["watchdog_fired"] = watchdog_fired
        record["shed_layers"] = sorted(result.missing_layers)

    def _export_uplink_telemetry(self, round_num: int) -> str:
        """Serialize round ``round_num``'s receiver telemetry to JSON.

        Schema: docs/extensions/04-overnight-interfaces.md §2.  Every
        time-valued key carries its clock domain as a suffix (audit NT-03):

        ``_receiver_s``
            receiver-local monotonic seconds relative to the receiver's
            round-start instant (G1).  Arrivals that physically preceded the
            local round start come out negative, which is fine — the primary
            KPI ``t_eps_local_receiver_s`` is a pure receiver-local interval
            and unaffected by the reference choice.
        ``_oneway_s``
            receiver arrival minus the sender's ``t_send_start_sender_s``: a
            true one-way wire time, available because every container shares
            the host kernel's CLOCK_MONOTONIC.  ``null`` when the sender did
            not stamp; negative values are counted as clock anomalies rather
            than reported, since they would disprove the shared-clock premise.

        Per-flow coverage comes out as THREE numbers (audit TRIG-2/BYTE-04)
        satisfying ``kappa_realized == kappa_slip + shed_mass_fraction``:

        ``kappa_slip``
            coverage SLIPPAGE — trigger mass that was scheduled to arrive
            and did not.  This is the quantity the ε bound speaks about, and
            the only one comparable against ε.  NEW key: its presence is what
            tells a reader the flow carries the split at all.
        ``shed_mass_fraction``
            DELIBERATELY omitted mass — layers the sender was advised to
            skip, which the aggregator recycle-fills by design.  Not a
            failure to cover, and not comparable against ε.
        ``kappa_realized``
            their sum — total shed mass, i.e. exactly what this key has
            always meant, kept unchanged so that reports written before and
            after the audit can be pooled without a definition clash.

        Manifested (uplink) flows are exported under their source-node key.
        Manifest-less flows — the monolithic ModelUpdate and the aggregator's
        downlink broadcast — are exported under the reserved
        ``_receiver_flows`` block (audit NT-05) so that they have a receiver
        clock of their own without being mistaken for ε-coverage flows by
        consumers that iterate source keys.  The other reserved keys carry
        sender-side per-round bookkeeping (``_sender``: predicted t_ε,
        assignment diagnostics, cancelled zombie-tail bytes, SO_SNDBUF
        grants), the aggregation realization (``_aggregation``), and the
        clock-domain legend (``_clock_domains``).  Consumers iterate source
        entries defensively and skip underscore keys.
        """
        round_start = self._round_starts.get(round_num)
        flows: dict[str, Any] = {}
        anomalies = 0
        for (source, rnd), record in self._uplink_flows.items():
            if rnd != round_num or record["manifest_arrival"] is None:
                continue
            base = (
                round_start if round_start is not None
                else record["manifest_arrival"]
            )
            manifest = self._round_manifests.get(rnd, {}).get(source)
            scores: dict[str, float] = (
                {e.layer_name: e.raw_score for e in manifest.entries}
                if manifest is not None else {}
            )
            total_score = sum(scores.values())
            if record["trigger_fire"] is not None:
                shed = record["shed_layers"] or []
                trigger_rel = record["trigger_fire"] - base
                t_eps = record["trigger_fire"] - record["manifest_arrival"]
            else:
                # Round never completed for this source (e.g. barrier
                # timeout with the watchdog disabled): report the live gap.
                shed = sorted(
                    set(scores) - set(record["layer_arrivals"])
                )
                trigger_rel = None
                t_eps = None
            # κ SPLIT (audit TRIG-2/BYTE-04).  The shed set mixes two things
            # with opposite meanings: mass that was SUPPOSED to arrive and
            # did not (coverage slippage — the quantity the ε bound is about,
            # and the one that must stay ≤ ε), and mass the sender was
            # advised to omit, which the aggregator recycle-fills by design.
            # Reporting both under one symbol is what put κ up to 0.998 in
            # the frontier table next to the invariant "κ ≤ ε".  Rotation
            # omission (cyclic) is a third thing and is NOT visible here at
            # all: those layers are absent from the manifest, so they leave
            # the denominator too — a coverage-denominator effect, not a shed.
            #
            # The split is published as a NEW symbol, not by redefining the
            # old one: `kappa_realized` keeps the meaning every report on disk
            # already has — total shed mass, FedLUAR's recycled-mass κ, which
            # ratified decision D-T1.2 fixed so the fedluar-alias baseline
            # stays comparable (κ_gross) — and the bounded quantity that
            # decision calls κ_net ships beside it as `kappa_slip`.  Redefining
            # the key in place would have made a pooled or before/after
            # re-analysis silently mix two definitions under one name with
            # nothing in the file to tell them apart; the presence of
            # `kappa_slip` is itself the discriminator readers dispatch on
            # (scripts/analysis_common.kappa_split).  Invariant, exactly:
            #   kappa_realized == kappa_slip + shed_mass_fraction.
            skipped_layers = (
                manifest_skipped_layers(manifest) if manifest is not None
                else set()
            )
            deliberate = [name for name in shed if name in skipped_layers]
            kappa_slip = (
                sum(
                    scores.get(name, 0.0)
                    for name in shed if name not in skipped_layers
                ) / total_score
                if total_score > 0 else None
            )
            shed_mass_fraction = (
                sum(scores.get(name, 0.0) for name in deliberate) / total_score
                if total_score > 0 else None
            )
            kappa = (
                kappa_slip + shed_mass_fraction
                if kappa_slip is not None and shed_mass_fraction is not None
                else None
            )
            # Coverage-completion latency on the WIRE clock: from the instant
            # the sender started putting the manifest on the socket to the
            # instant the ε-trigger fired here.  Unlike the receiver-local
            # t_eps (which starts at manifest arrival) this is directly
            # comparable with a monolithic flow's completion time in
            # `_receiver_flows`: both span the same sender->receiver pair.
            send_start = record["manifest_send_start"]
            t_cover = (
                record["trigger_fire"] - send_start
                if send_start is not None and record["trigger_fire"] is not None
                else None
            )
            wire_times = dict(record["layer_wire_times"])
            flow_anomalies = sum(
                1 for value in
                [record["manifest_wire_time"], t_cover, *wire_times.values()]
                if value is not None and value < 0.0
            )
            anomalies += flow_anomalies
            flows[source] = {
                "manifest_arrival_rel_receiver_s": (
                    record["manifest_arrival"] - base
                ),
                "trigger_fire_rel_receiver_s": trigger_rel,
                "watchdog_fired": record["watchdog_fired"],
                "t_eps_local_receiver_s": t_eps,
                "layer_arrivals_rel_receiver_s": {
                    name: stamp - base
                    for name, stamp in record["layer_arrivals"].items()
                },
                "manifest_wire_time_oneway_s": record["manifest_wire_time"],
                "layer_wire_times_oneway_s": wire_times,
                "t_cover_oneway_s": t_cover,
                "wire_clock_anomaly": flow_anomalies > 0,
                "shed_layers": list(shed),
                "kappa_realized": kappa,
                "kappa_slip": kappa_slip,
                "shed_mass_fraction": shed_mass_fraction,
                "ordering_violation": record["ordering_violation"],
            }
            # Skip-feedback v2: layers the sender omitted on advice, read
            # straight off the manifest flags.  They still appear inside
            # shed_layers (they are genuinely absent at completion) and inside
            # `kappa_realized`, but their mass is broken out as
            # `shed_mass_fraction` and excluded from `kappa_slip`.
            if skipped_layers:
                flows[source]["skip_omitted_layers"] = sorted(skipped_layers)
        receiver_flows, flow_anomalies = self._build_receiver_flows_block(
            round_num, round_start,
        )
        anomalies += flow_anomalies
        if receiver_flows:
            flows[_TELEMETRY_FLOWS_KEY] = receiver_flows
        sender_block = self._build_sender_block(round_num)
        if sender_block:
            flows[_TELEMETRY_SENDER_KEY] = sender_block
        aggregation_block = self._aggregation_info.get(round_num)
        if aggregation_block:
            flows[_TELEMETRY_AGGREGATION_KEY] = aggregation_block
        if flows:
            flows[_TELEMETRY_CLOCK_DOMAINS_KEY] = {
                "suffixes": _CLOCK_DOMAIN_SUFFIXES,
                "oneway_negative_count": anomalies,
            }
        return json.dumps(flows, default=_json_safe) if flows else ""

    def _build_receiver_flows_block(
        self, round_num: int, round_start: float | None
    ) -> tuple[dict[str, Any], int]:
        """Transport-level arrival stamps per flow (audit NT-05).

        Keyed ``"<source>|<payload_type>"`` — a source can deliver both a
        manifest and its layers in one round, and the monolithic and
        per-layer payload types must never be pooled by accident.

        ``completion_wire_oneway_s`` is the quantity monolithic mode never
        had: the interval from the sender handing the flow's first byte to a
        socket to this receiver reading its last byte.  For a monolithic
        ModelUpdate that is the whole model's wire time, directly comparable
        with a per-layer arm's ``t_cover_oneway_s``.

        Returns the block and the number of negative one-way values found
        (nonzero ⇒ the two ends are not on one clock).
        """
        block: dict[str, Any] = {}
        anomalies = 0
        for (source, rnd, payload_type), entry in self._flow_stamps.items():
            if rnd != round_num:
                continue
            base = round_start if round_start is not None else entry["first_byte"]
            send_start = entry["send_start"]
            first_wire = (
                entry["first_byte"] - send_start if send_start is not None
                else None
            )
            completion_wire = (
                entry["last_byte"] - send_start if send_start is not None
                else None
            )
            negative = sum(
                1 for value in (first_wire, completion_wire)
                if value is not None and value < 0.0
            )
            anomalies += negative
            block[f"{source}|{payload_type}"] = {
                "first_byte_rel_receiver_s": entry["first_byte"] - base,
                "last_byte_rel_receiver_s": entry["last_byte"] - base,
                "first_byte_wire_oneway_s": first_wire,
                "completion_wire_oneway_s": completion_wire,
                "wire_clock_anomaly": negative > 0,
                "messages": entry["messages"],
                "bytes": entry["bytes"],
            }
        return block, anomalies

    def _capture_aggregation_telemetry(self, round_num: int) -> None:
        """Snapshot the algorithm's aggregation realization for the report.

        FedAvg refreshes ``last_aggregation_telemetry`` in place on every
        aggregation, so the per-round copy must be taken immediately after
        the ``aggregate()`` call.  Algorithms without slippage telemetry
        (DPSGD/ADPSGD/Gossip) simply produce no ``_aggregation`` block —
        hence the getattr guards.  ``inclusion_counts`` is cumulative;
        per-(layer, source) inclusion *rates* are count / aggregation_count.
        """
        telemetry = getattr(self.algorithm, "last_aggregation_telemetry", None)
        if not telemetry:
            return
        block: dict[str, Any] = {"layers": telemetry}
        counts = getattr(self.algorithm, "inclusion_counts", None)
        if counts:
            # inclusion_counts is mutated in place across aggregations —
            # copy so the block stays a faithful as-of-this-round snapshot.
            block["inclusion_counts"] = {
                layer: dict(per_source) for layer, per_source in counts.items()
            }
        block["aggregation_count"] = int(
            getattr(self.algorithm, "aggregation_count", 0)
        )
        # Skip-feedback v2: the advice this aggregation produced for the
        # NEXT round, as piggybacked on the broadcast — {sender: [layers]}.
        # Copied (the algorithm refreshes its dict per aggregation) so the
        # block stays an as-of-this-round snapshot.
        advice = getattr(self.algorithm, "last_skip_advice", None)
        if advice:
            block["skip_advice"] = {
                source: sorted(layers)
                for source, layers in advice.items()
                if layers
            }
        self._aggregation_info[round_num] = block

    def _realized_inclusions(self, round_num: int) -> dict[str, list[str]]:
        """Per-source layers that ENTERED this round's aggregate (TRIG-1).

        Read straight off the snapshot ``_capture_aggregation_telemetry``
        already takes: ``layers[ℓ].arrived_sources`` is exactly "whose fresh
        value went into layer ℓ's average".  Sources that contributed
        nothing still get an (empty) entry — "none of yours made it" is the
        statement the staleness counter needs most, and the universe comes
        from the cumulative ``inclusion_counts``, which lists every source
        seen in the round's updates including the zero-arrival ones.

        Empty dict when the algorithm publishes no inclusion telemetry
        (decentralized arms), which the sender side reads as "no ack
        channel" rather than as "nothing was included".
        """
        block = self._aggregation_info.get(round_num)
        if not block:
            return {}
        layers = block.get("layers") or {}
        sources: set[str] = set()
        for per_source in (block.get("inclusion_counts") or {}).values():
            sources.update(per_source)
        included: dict[str, list[str]] = {source: [] for source in sources}
        for layer_name, entry in layers.items():
            for source in entry.get("arrived_sources") or ():
                included.setdefault(source, []).append(layer_name)
        return {source: sorted(names) for source, names in included.items()}

    def _build_sender_block(self, round_num: int) -> dict[str, Any]:
        """Sender-side bookkeeping for the ``_sender`` telemetry key.

        ``predicted_t_eps_model_s`` and everything inside
        ``assignment_diagnostics`` are strategy predictions (`_model_s`
        domain), not measurements — they may be compared against a measured
        t_ε only as prediction-vs-realization, never plotted on one axis as
        if they were the same quantity (audit NT-03).
        """
        block: dict[str, Any] = {}
        info = self._assignment_info.get(round_num)
        if info is not None:
            block["predicted_t_eps_model_s"] = info["predicted_t_eps"]
            block["effective_epsilon"] = info["epsilon"]
            block["assignment_diagnostics"] = info["diagnostics"]
            if info.get("skip_omitted"):
                # Layers this sender omitted on aggregator advice this
                # round (skip-feedback v2) — the sender-side mirror of the
                # receiver's per-flow `skip_omitted_layers`.
                block["skip_omitted_layers"] = info["skip_omitted"]
            if "layer_ages" in info:
                # Realized staleness this round's plan was built from, plus
                # the event basis that produced it (audit TRIG-1/ML-01:
                # 'inclusion' = acknowledged aggregate entry, the basis
                # under which τ_max is a real bound; 'head_placement*' =
                # the pre-fix proxy, which bounds nothing).
                block["layer_ages"] = info["layer_ages"]
                block["age_basis"] = info["age_basis"]
        ledger = self._tail_ledgers.get(round_num)
        if ledger is not None and ledger["cancelled_bytes"] > 0:
            block["tail_cancelled_bytes"] = ledger["cancelled_bytes"]
            block["tail_cancelled_layers"] = sorted(
                set(ledger["cancelled_layers"])
            )
        if self._sender_events:
            # Tail activity observed after its round's report had already
            # shipped (late completions, late cancellations) — attributed
            # to its own round so $-accounting can re-aggregate.
            block["prior_round_events"] = self._sender_events
            self._sender_events = []
        sndbuf = self._sndbuf_telemetry()
        if sndbuf:
            # Per-class SO_SNDBUF request vs grant (audit NT-08/BYTE-08).  The
            # buffer sets how much of a class's payload the enqueue clock
            # never sees, so an asymmetric grant is an arm-dependent bias on
            # every sender-side time; carried in the data, not only in logs.
            block["socket_sndbuf"] = sndbuf
        return block

    def _sndbuf_telemetry(self) -> list[dict[str, Any]]:
        """Pool SO_SNDBUF records, tolerating pools that don't expose them."""
        getter = getattr(self.pool, "sndbuf_telemetry", None)
        if not callable(getter):
            return []
        try:
            records = getter()
        except Exception:  # noqa: BLE001 - telemetry must not fail a round
            logger.exception(f"[{self.node_id}] SO_SNDBUF telemetry failed")
            return []
        return records if isinstance(records, list) else []

    # ------------------------------------------------------------------
    # Watchdog (gate ruling G4)
    # ------------------------------------------------------------------

    def _receiver_layer_sizes(self) -> dict[str, int]:
        """Estimated per-layer wire bytes, derived from the local model.

        The manifest does not carry sizes; the receiver infers them from
        its own (architecturally identical) model.  Cached — trainable
        variable shapes are fixed for the lifetime of a run.
        """
        if self._receiver_sizes_cache is None:
            # Keras 3 variables expose dtype as a plain string ('float32');
            # tf.Variable exposes a tf.DType (whose .name is the string).
            self._receiver_sizes_cache = {
                var.path: int(np.prod(var.shape))
                * np.dtype(getattr(var.dtype, "name", var.dtype)).itemsize
                + _LAYER_ENVELOPE_OVERHEAD_BYTES
                for var in self.model.trainable_variables
            }
        return self._receiver_sizes_cache

    def _incoming_bandwidth_mbps_total(self, source: str) -> float | None:
        """Sum of finite shaped class bandwidths on the edge from ``source``.

        Unshaped classes (bandwidth None) are excluded: data may drain
        faster through them, which only makes the resulting T_max more
        conservative (it fires later than 3x the true fluid bound — still
        bounded, never spuriously early).  Returns None when no finite
        class exists, in which case the watchdog cannot be armed.
        """
        for edge in self.incoming_edges:
            if f"node-{edge.get('src')}" != source:
                continue
            total = 0.0
            for params in (edge.get("classes") or {}).values():
                bandwidth = params.get("bandwidth_mbps")
                if bandwidth is not None and math.isfinite(bandwidth):
                    total += float(bandwidth)
            return total if total > 0 else None
        return None

    def _arm_watchdog(self, manifest: federation_pb2.RoundManifest) -> None:
        """Arm the T_max timer for a freshly-registered manifest.

        T_max = watchdog_factor x fluid bound, where the fluid bound is the
        manifest's total estimated bytes drained at the sum of the incoming
        edge's shaped bandwidths.  Skipped when disabled by config or when
        no finite bandwidth information is available (then only the
        engine-level sync timeout bounds the round — logged at debug).
        """
        if self.watchdog_factor <= 0:
            return
        key = (manifest.source_node_id, manifest.round)
        if key in self._watchdog_tasks:
            return
        bandwidth_total = self._incoming_bandwidth_mbps_total(
            manifest.source_node_id
        )
        if bandwidth_total is None:
            logger.debug(
                f"[{self.node_id}] Watchdog not armed for {key}: no finite "
                f"incoming bandwidth info"
            )
            return
        sizes = self._receiver_layer_sizes()
        known = list(sizes.values())
        fallback = int(np.mean(known)) if known else 0
        # Skipped entries (skip-feedback v2) never travel — the expected
        # wire bytes are the transmitted entries only, so the fluid bound
        # (and hence T_max) is metered on those.  Including the skipped
        # bytes would only loosen the watchdog for exactly the flows that
        # shed the most.
        total_bytes = sum(
            sizes.get(entry.layer_name, fallback)
            for entry in manifest.entries
            if not entry.skipped
        )
        fluid_bound_s = total_bytes * 8.0 / (bandwidth_total * 1e6)
        if not math.isfinite(fluid_bound_s) or fluid_bound_s <= 0:
            return
        t_max = self.watchdog_factor * fluid_bound_s
        self._watchdog_tasks[key] = asyncio.create_task(
            self._watchdog_expire(key, t_max)
        )

    async def _watchdog_expire(
        self, key: tuple[str, int], t_max: float
    ) -> None:
        """Force-complete ``key`` after T_max (unless cancelled first)."""
        try:
            await asyncio.sleep(t_max)
        except asyncio.CancelledError:
            return
        self._watchdog_tasks.pop(key, None)
        result = self._layer_buffer.force_fire(key)
        if result is None:
            return  # natural completion won the race
        now = time.monotonic()
        logger.warning(
            f"[{self.node_id}] WATCHDOG fired for source={key[0]} "
            f"round={key[1]} after T_max={t_max:.3f}s; forcing completion "
            f"with missing={sorted(result.missing_layers)}"
        )
        self._finalize_uplink_flow(result, now, watchdog_fired=True)
        await self.algorithm.on_update_received(result.update)

    def _cancel_watchdog(self, key: tuple[str, int]) -> None:
        task = self._watchdog_tasks.pop(key, None)
        if task is not None and not task.done():
            task.cancel()

    # ------------------------------------------------------------------
    # Per-round dissemination planning (sender side)
    # ------------------------------------------------------------------

    def _layer_payload_sizes(
        self, params: dict[str, np.ndarray]
    ) -> dict[str, int]:
        """Serialized payload bytes per layer: array nbytes + envelope
        overhead (`_LAYER_ENVELOPE_OVERHEAD_BYTES`; recorded in diagnostics
        via this constant's docstring contract)."""
        return {
            name: int(arr.nbytes) + _LAYER_ENVELOPE_OVERHEAD_BYTES
            for name, arr in params.items()
        }

    def _outgoing_class_bandwidths(self) -> dict[int, float]:
        """Per traffic class: shaped Mbps for this node's egress.

        When several outgoing edges exist (decentralized topologies) the
        per-class minimum is used — the conservative shared-plan choice,
        since `assign()` runs once per round for all destinations.
        Unshaped/unknown classes are encoded as inf per the strategy
        contract (§1.5).
        """
        bandwidths = {
            cls: float("inf") for cls in range(self.num_traffic_classes)
        }
        for edge in self.outgoing_edges:
            for cls_key, params in (edge.get("classes") or {}).items():
                cls = int(cls_key)
                if cls not in bandwidths:
                    continue
                bandwidth = params.get("bandwidth_mbps")
                value = (
                    float(bandwidth)
                    if bandwidth is not None and math.isfinite(bandwidth)
                    else float("inf")
                )
                bandwidths[cls] = min(bandwidths[cls], value)
        return bandwidths

    def _resolve_apply_aging(self) -> Callable | None:
        """Lazily import the aging transform; warn once when unavailable."""
        if self._apply_aging_fn is not None:
            return self._apply_aging_fn
        try:
            from src.importance.aging import apply_aging
        except ImportError:
            if not self._aging_unavailable_warned:
                self._aging_unavailable_warned = True
                logger.warning(
                    f"[{self.node_id}] aging_mode={self.aging_mode!r} "
                    f"requested but src/importance/aging.py is not "
                    f"importable — aging disabled for this run"
                )
            return None
        self._apply_aging_fn = apply_aging
        return apply_aging

    def _advance_layer_ages(
        self, params: dict[str, np.ndarray], round_num: int
    ) -> dict[str, int]:
        """Roll the per-layer staleness counters into ``round_num``.

        Audit TRIG-1/ML-01.  ``age_ℓ`` is meant to be "rounds since layer ℓ's
        contribution last entered the aggregate", and only the aggregator
        knows that.  Under the default ``inclusion`` basis the counters are
        therefore advanced HERE, from the previous round's acknowledgement
        (``SkipAdvice.included_layers``, piggybacked on the broadcast ahead
        of the model payload, so it is always in hand before this round is
        planned): acknowledged layers reset to 0, every other layer ages by
        one.  ``aging_tau_max`` then bounds realized staleness, which is the
        precondition the convergence sketch rests on.

        Two fallbacks, both loud and both recorded in ``_age_basis_realized``
        (exported in the report's ``_sender`` block):

        - **never any ack** (a decentralized algorithm with no aggregator
          broadcast, or an aggregator whose algorithm has no inclusion
          telemetry): keep the pre-fix head-placement basis, which the
          caller applies after the assignment.  Ageing every layer instead
          would silently promote everything to must_receive at τ_max and
          turn the arm into full transmission.
        - **acks seen before, this one missing** (message loss): age every
          layer.  Never reset on an unverified proxy — that is the defect.

        Under the ``head_placement`` basis this is a pure read of the
        current counters; the caller owns the (legacy) update.
        """
        ages = {name: self._layer_ages.get(name, 0) for name in params}
        if self.aging_mode == "none" or self.aging_age_basis == "head_placement":
            # No starvation control (nothing consumes the counters), or the
            # legacy basis: keep the pre-fix update, applied by the caller.
            self._age_basis_realized = "head_placement"
            return ages
        if round_num <= 0:
            # Nothing has been aggregated yet: everything is fresh.
            self._age_basis_realized = "inclusion"
            self._layer_ages = dict(ages)
            return ages
        acked = self._inclusion_acks.get(round_num - 1)
        if acked is None and not self._inclusion_ack_seen:
            self._age_basis_realized = "head_placement_fallback"
            if not self._age_basis_warned:
                self._age_basis_warned = True
                logger.warning(
                    f"[{self.node_id}] aging_age_basis='inclusion' but no "
                    f"realized-inclusion ack has ever arrived (round "
                    f"{round_num}); falling back to the pre-fix "
                    f"head-placement basis — aging_tau_max is then a "
                    f"scheduling period, NOT a staleness bound (audit "
                    f"TRIG-1/ML-01)"
                )
            return ages
        self._age_basis_realized = (
            "inclusion" if acked is not None else "inclusion_ack_missing"
        )
        if acked is None:
            logger.warning(
                f"[{self.node_id}] No inclusion ack for round "
                f"{round_num - 1}; ageing every layer (never reset a "
                f"staleness counter on an unverified proxy)"
            )
        included = acked or frozenset()
        advanced = {
            name: 0 if name in included else age + 1
            for name, age in ages.items()
        }
        self._layer_ages = dict(advanced)
        return advanced

    def _prepare_layer_dissemination(
        self,
        params: dict[str, np.ndarray],
        deltas: dict[str, np.ndarray] | None,
        round_num: int,
    ) -> _SendPlan | None:
        """Build the per-round uplink send plan (per-layer mode).

        Implements the gate-doc sender pipeline: shipped-delta metric input
        (headline fix 3), frozen trigger accounting + separate sched scores
        (G2), aging on sched scores only, ε warm-up, and pluggable
        assignment.  The per-layer age counters are advanced by
        :meth:`_advance_layer_ages` (receiver-acknowledged inclusion by
        default; head placement only under the legacy basis, applied after
        the assignment below).
        """
        if self.update_mode != "per_layer" or self._assignment_strategy is None:
            return None

        # Metric input — the shipped multi-epoch delta for every arm except
        # the raw_norm control, which keeps its native E2 configuration
        # (raw L2 norm of the last minibatch gradient).
        if self.importance_metric_v2_name == "raw_norm":
            metric_input = self.last_gradients
        else:
            metric_input = deltas
        trigger_scores = self._finite_scores(
            self._trigger_metric.compute_all(
                model_params=params,
                gradients=metric_input,
                round_num=round_num,
                context={},
            ),
            round_num,
        )
        sched_scores: dict[str, float] | None = None
        if self._sched_metric is not None:
            sched_scores = self._finite_scores(
                self._sched_metric.compute_all(
                    model_params=params,
                    gradients=metric_input,
                    round_num=round_num,
                    context={},
                ),
                round_num,
            )

        # Aging acts on sched scores only — never on trigger accounting (G2).
        sizes = self._layer_payload_sizes(params)
        ages = self._advance_layer_ages(params, round_num)
        must_receive: set[str] = set()
        effective_sched = dict(sched_scores or trigger_scores)
        if self.aging_mode != "none":
            apply_aging = self._resolve_apply_aging()
            if apply_aging is not None:
                effective_sched, extra_must = apply_aging(
                    scores=effective_sched,
                    ages=ages,
                    mode=self.aging_mode,
                    lam=self.aging_lambda,
                    tau_max=self.aging_tau_max,
                    sizes=sizes,  # required by the density-space boost
                    rng=self._aging_rng,
                )
                effective_sched = dict(effective_sched)
                must_receive |= set(extra_must)
        # The manifest writes sched_score only when it actually differs
        # from the frozen trigger score (otherwise the wire keeps the 0.0
        # "same as raw_score" convention — G2 / proto3 default).
        sched_differs = effective_sched != trigger_scores

        # ---- Skip-feedback v2: apply round-tagged aggregator advice ----
        # Advice produced by round r-1's aggregation (and piggybacked on
        # that broadcast) applies to THIS round r only; older tags are
        # never consulted (fail-open — no advice, no omission).  Aging
        # overrides skip: a layer promoted to must_receive (age >= tau_max)
        # MUST be sent, so it is exempted from the skip set.  The skipped
        # layers are removed from the strategy's universe (no head budget
        # is wasted on them) but stay manifest-listed with their trigger
        # mass, flagged `skipped` (covered-by-recycling at the receiver).
        skip_set: set[str] = set()
        if self.skip_feedback_mode != "off" and round_num > 0:
            advice = self._skip_advice.get(round_num - 1)
            if advice:
                skip_set = (set(advice) & set(params)) - must_receive
                if skip_set and len(skip_set) == len(params):
                    # Never skip everything: an all-skipped manifest could
                    # only complete via watchdog.  Retain the layer with the
                    # largest trigger mass — if something must travel, send
                    # the most coverage-valuable one.
                    keep = max(
                        skip_set,
                        key=lambda n: (trigger_scores.get(n, 0.0), n),
                    )
                    skip_set.discard(keep)
                    logger.warning(
                        f"[{self.node_id}] Skip advice for round "
                        f"{round_num} covered every layer; keeping "
                        f"{keep!r} transmitted (highest trigger mass)"
                    )

        effective_eps = self._effective_epsilon(round_num)
        if skip_set:
            send_scores = {
                name: score
                for name, score in effective_sched.items()
                if name not in skip_set
            }
            send_sizes = {
                name: size
                for name, size in sizes.items()
                if name not in skip_set
            }
            send_ages = {
                name: age for name, age in ages.items() if name not in skip_set
            }
            send_trigger = {
                name: score
                for name, score in trigger_scores.items()
                if name not in skip_set
            }
        else:
            send_scores, send_sizes, send_ages = effective_sched, sizes, ages
            send_trigger = trigger_scores
        # Note on the ε budget: the strategy meters its tail budget on the
        # transmitted universe (ε·Σ_send), while the receiver's trigger
        # credits the skipped mass K and meters (1−ε)·(Σ_send + K).  The
        # planned head mass H >= (1−ε)·Σ_send always satisfies the trigger:
        # H + K >= (1−ε)·Σ_send + K >= (1−ε)·(Σ_send + K).  The strategy is
        # therefore conservative (sheds slightly less than the trigger
        # would allow), never under-covering.  The frozen trigger scores go
        # in alongside the sched ones so the strategy can meter the shed in
        # the units the receiver enforces (audit TRIG-5) — they order and
        # place nothing (G2).
        result = self._assignment_strategy.assign(
            scores=send_scores,
            sizes=send_sizes,
            bandwidths=self._outgoing_class_bandwidths(),
            epsilon=effective_eps,
            must_receive=must_receive,
            ages=send_ages,
            trigger_scores=send_trigger,
        )
        if not result.assignment:
            # An empty assignment would announce nothing and send nothing,
            # deadlocking every receiver on this flow.  Strategies must not
            # do this (cyclic_k >= 1 is schema-enforced); recover loudly.
            # The fallback spans the transmitted universe only — skipped
            # layers stay omitted (they are manifest-listed separately).
            logger.error(
                f"[{self.node_id}] Assignment strategy "
                f"{self.assignment_strategy_name!r} returned an empty "
                f"assignment for round {round_num}; falling back to "
                f"all-layers-on-class-0"
            )
            result = AssignmentResult(
                assignment={name: 0 for name in send_scores},
                head=set(send_scores),
                tail=set(),
            )

        # Age bookkeeping, LEGACY BASIS ONLY (audit TRIG-1/ML-01): head
        # placement is a sender scheduling decision and carries no arrival
        # guarantee, so resetting on it bounds nothing.  Under the default
        # 'inclusion' basis the counters were already advanced from the
        # aggregator's ack in _advance_layer_ages and must not be touched
        # here; this branch exists to reproduce pre-fix campaigns.
        if self._age_basis_realized.startswith("head_placement"):
            for name in params:
                self._layer_ages[name] = (
                    0 if name in result.head else ages[name] + 1
                )

        unsendable = must_receive - set(result.assignment)
        if unsendable:
            logger.warning(
                f"[{self.node_id}] must_receive layers omitted from the "
                f"assignment (strategy contract violation, flags dropped): "
                f"{sorted(unsendable)}"
            )

        self._assignment_info[round_num] = {
            "predicted_t_eps": result.predicted_t_eps,
            "epsilon": effective_eps,
            "diagnostics": result.diagnostics,
        }
        if self.aging_mode != "none":
            # Realized staleness bookkeeping (audit ML-01 fix c): the ages
            # THIS round's plan was built from, and which event basis
            # produced them, so the τ_max invariant is checkable offline
            # instead of being inferred from head placements.
            self._assignment_info[round_num]["layer_ages"] = dict(ages)
            self._assignment_info[round_num]["age_basis"] = (
                self._age_basis_realized
            )
        if skip_set:
            self._assignment_info[round_num]["skip_omitted"] = sorted(skip_set)
        logger.info(
            f"[{self.node_id}] Round {round_num} assignment "
            f"[{self.assignment_strategy_name}/"
            f"{self.importance_metric_v2_name}]: eps={effective_eps:.4g} "
            f"head={len(result.head)} tail={len(result.tail)} "
            f"omitted={len(params) - len(result.assignment)} "
            f"skip_advised={len(skip_set)} "
            f"must_receive={len(must_receive)} "
            f"predicted_t_eps={result.predicted_t_eps}"
        )
        if round_num == 0:
            for name in sorted(result.assignment):
                logger.info(
                    f"[{self.node_id}] Layer {name}: "
                    f"raw={trigger_scores.get(name, 0.0):.4g} "
                    f"sched={effective_sched.get(name, 0.0):.4g} "
                    f"-> class {result.assignment[name]} "
                    f"({'head' if name in result.head else 'tail'})"
                )

        return _SendPlan(
            round_num=round_num,
            assignment=result,
            trigger_scores=trigger_scores,
            sched_scores=effective_sched if sched_differs else None,
            must_receive=must_receive & set(result.assignment),
            skipped=frozenset(skip_set - set(result.assignment)),
        )

    def _downlink_plan(
        self, params: dict[str, np.ndarray], round_num: int
    ) -> _SendPlan | None:
        """Build the aggregator's per-layer broadcast plan (gate ruling G5).

        The downlink class map is byte-balanced and importance-blind:
        workers replace their model wholesale, so they need *every* layer —
        coverage semantics (and therefore manifests, importance ordering,
        and tails) do not apply.  Byte-balance is the ε=0/makespan optimum,
        which is exactly the downlink objective.  No manifest is sent
        (``trigger_scores=None``): workers complete via the count-based
        all-layers trigger.

        For telemetry the plan carries ``log_scores``: the real per-layer
        ``delta_sq_norm`` of the aggregator's realized motion (the same
        metric the uplink logs), so ``layer_comm_metrics.importance`` on
        aggregator rows reflects the round's aggregated delta instead of the
        old ambiguous 1.0 sentinel.  This is logging-only — it does not gate
        the manifest (``trigger_scores`` stays None, preserving G5).
        """
        if self.update_mode != "per_layer":
            return None
        bandwidths = self._outgoing_class_bandwidths()
        assignment: AssignmentResult | None = None
        if any(math.isfinite(b) for b in bandwidths.values()):
            if self._downlink_strategy is None:
                try:
                    self._downlink_strategy = make_strategy("byte_balanced")
                except ValueError:
                    pass
            if self._downlink_strategy is not None:
                try:
                    assignment = self._downlink_strategy.assign(
                        scores={name: 1.0 for name in params},
                        sizes=self._layer_payload_sizes(params),
                        bandwidths=bandwidths,
                        epsilon=0.0,
                        must_receive=set(),
                        ages=None,
                    )
                except Exception:
                    logger.exception(
                        f"[{self.node_id}] byte_balanced downlink assignment "
                        f"failed; broadcasting on class 0"
                    )
                    assignment = None
        if assignment is None:
            if not self._downlink_fallback_warned:
                self._downlink_fallback_warned = True
                logger.warning(
                    f"[{self.node_id}] byte_balanced strategy unavailable "
                    f"for the downlink broadcast — falling back to the "
                    f"legacy single-class map (class 0 only)"
                )
            assignment = AssignmentResult(
                assignment={name: 0 for name in params},
                head=set(params),
                tail=set(),
            )
        return _SendPlan(
            round_num=round_num,
            assignment=assignment,
            trigger_scores=None,
            sched_scores=None,
            must_receive=set(),
            log_scores=self._downlink_log_scores(params),
        )

    def _downlink_log_scores(
        self, params: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Real per-layer ``delta_sq_norm`` for the aggregator broadcast rows.

        Scores the aggregator's realized per-layer motion with the same
        trigger metric the uplink uses (``self._trigger_metric``), so the
        logged ``importance`` on downlink rows is the round's aggregated
        ``‖Δ_ℓ‖₂²`` rather than the ambiguous 1.0 sentinel.  Telemetry only.

        The realized motion lives on the recycle-tracking algorithm
        (``_last_delta``); when it is unavailable — a non-recycle arm, or the
        cold-start round before any motion is recorded — every layer logs
        ``nan`` (an unambiguous "no real score" marker, unlike 1.0 which is
        a legitimate uplink score).
        """
        last_delta = getattr(self.algorithm, "_last_delta", None)
        if not last_delta:
            return {name: float("nan") for name in params}
        scores = self._trigger_metric.compute_all(
            model_params=params,
            gradients=last_delta,
            round_num=0,
            context={},
        )
        return {
            name: (
                max(0.0, float(scores[name]))
                if name in scores and name in last_delta
                else float("nan")
            )
            for name in params
        }

    # ------------------------------------------------------------------
    # Zombie-tail lifecycle (pre-run fix 3)
    # ------------------------------------------------------------------

    def _tail_ledger(self, round_num: int) -> dict[str, Any]:
        ledger = self._tail_ledgers.get(round_num)
        if ledger is None:
            ledger = {
                "metrics": [],          # completed tail sends, not yet reported
                "cancelled_bytes": 0,
                "cancelled_layers": [],
            }
            self._tail_ledgers[round_num] = ledger
        return ledger

    def _maybe_signal_rounds_closed(self, up_to_round: int) -> None:
        """Worker-side broadcast hook: rounds <= ``up_to_round`` are closed.

        Only meaningful for centralized workers, where any inbound update
        is the aggregator's broadcast: aggregation for that round is done,
        so still-queued tail envelopes of it are pure zombie bytes.  The
        cancel events make the per-class tail coroutines exit at the next
        envelope boundary; envelopes already inside ``sock_sendall`` (or
        accepted into the pinned SNDBUF) cannot be recalled — that residue
        is bounded by SNDBUF pinning (G1).
        """
        if self._is_aggregator or not self.algorithm.is_centralized:
            return
        for round_num, event in self._tail_cancel_events.items():
            if round_num <= up_to_round:
                event.set()

    def _record_tail_cancellation(
        self, round_num: int, envelopes: list[federation_pb2.Envelope]
    ) -> None:
        """Account for tail envelopes that were cancelled before sending."""
        if not envelopes:
            return
        cancelled_bytes = sum(env.ByteSize() + 4 for env in envelopes)
        layers = [env.layer_update.layer_name for env in envelopes]
        if round_num in self._reported_rounds:
            self._sender_events.append({
                "round": round_num,
                "type": "tail_cancelled",
                "bytes": cancelled_bytes,
                "layers": layers,
            })
        else:
            ledger = self._tail_ledger(round_num)
            ledger["cancelled_bytes"] += cancelled_bytes
            ledger["cancelled_layers"].extend(layers)
        logger.info(
            f"[{self.node_id}] Cancelled {len(envelopes)} unsent zombie-tail "
            f"envelope(s) of round {round_num} ({cancelled_bytes} bytes saved)"
        )

    async def _retire_tail_rounds_before(self, round_num: int) -> None:
        """Cancel-and-await tail tasks of rounds before ``round_num``.

        Called when a new send round opens: the prior round's tail must
        release the per-class sockets before new envelopes are queued, both
        to avoid interleaved writes on a shared socket and to stop zombie
        bytes from delaying the new round's head.  The await is bounded by
        at most one in-flight envelope per (destination, class) — see
        `_maybe_signal_rounds_closed` on why mid-send envelopes cannot be
        recalled.
        """
        stale_tasks: list[asyncio.Task] = []
        for old_round in [r for r in self._tail_tasks if r < round_num]:
            event = self._tail_cancel_events.get(old_round)
            if event is not None:
                event.set()
            stale_tasks.extend(self._tail_tasks.pop(old_round))
            self._tail_cancel_events.pop(old_round, None)
        if stale_tasks:
            await asyncio.gather(*stale_tasks, return_exceptions=True)

    async def _drain_send_tasks(self) -> None:
        """Let all outstanding tail tasks finish (end of training).

        The final round's tail is still useful to peers completing their
        last aggregation, so it drains instead of being cancelled; the
        shutdown grace period in node.py covers the socket lifetime.
        """
        tasks = [t for ts in self._tail_tasks.values() for t in ts]
        self._tail_tasks.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in self._watchdog_tasks.values():
            if not task.done():
                task.cancel()
        self._watchdog_tasks.clear()

    def _collect_tail_round_metrics(
        self, round_num: int
    ) -> tuple[int, list[dict]]:
        """Drain tail sends of ``round_num`` completed since dispatch.

        Tail envelopes typically finish during the barrier wait /
        aggregation / evaluation phases, so most are accounted in their own
        round's report here; stragglers completing after the report are
        attributed via `_sender` prior_round_events instead.
        """
        ledger = self._tail_ledgers.get(round_num)
        if ledger is None:
            return 0, []
        metrics = ledger["metrics"]
        ledger["metrics"] = []
        return sum(m["bytes_sent"] for m in metrics), metrics

    def _gc_round_state(self, current_round: int, max_staleness: int = 2) -> None:
        """Reap engine-side per-round state older than ``max_staleness``.

        Mirrors `LayerBuffer.clear_stale`.  Also cancels watchdogs of
        reaped flows (they can no longer deliver a usable update) and
        prunes the manifest cache, which previously grew unboundedly.
        """
        threshold = current_round - max_staleness
        for key in [k for k in self._uplink_flows if k[1] < threshold]:
            del self._uplink_flows[key]
        for key in [k for k in self._flow_stamps if k[1] < threshold]:
            del self._flow_stamps[key]
        for key in [k for k in self._watchdog_tasks if k[1] < threshold]:
            self._cancel_watchdog(key)
        for rnd in [r for r in self._round_manifests if r < threshold]:
            del self._round_manifests[rnd]
        for rnd in [r for r in self._round_starts if r < threshold]:
            del self._round_starts[rnd]
        for rnd in [r for r in self._assignment_info if r < threshold]:
            del self._assignment_info[rnd]
        for rnd in [r for r in self._aggregation_info if r < threshold]:
            del self._aggregation_info[rnd]
        for rnd in [r for r in self._tail_ledgers if r < threshold]:
            del self._tail_ledgers[rnd]
        # Skip advice tagged for rounds already in the past can never apply
        # again (round-tag check); reap it with the same staleness window.
        # Same for inclusion acks: only round r-1's ack ages round r.
        for rnd in [r for r in self._skip_advice if r < threshold]:
            del self._skip_advice[rnd]
        for rnd in [r for r in self._inclusion_acks if r < threshold]:
            del self._inclusion_acks[rnd]
        self._reported_rounds = {
            r for r in self._reported_rounds if r >= threshold
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _flag_divergence(self, round_num: int, reason: str) -> None:
        """Mark ``round_num`` diverged (audit ML-04).  Never aborts the run.

        First reason wins — the later checks of the same round are almost
        always consequences of the first (NaN weights make every subsequent
        loss NaN).  Logged at ERROR because it invalidates the round's
        accuracy, byte and coverage numbers alike; the training loop
        deliberately continues, because the collapse studies are about the
        trajectory AFTER the blow-up.
        """
        if round_num in self._diverged_rounds:
            return
        self._diverged_rounds[round_num] = reason
        logger.error(
            f"[{self.node_id}] DIVERGENCE in round {round_num}: {reason} — "
            f"this round's metrics are NOT data (accuracy collapses to the "
            f"chance level rather than to a missing value); the run "
            f"continues and the round is flagged `diverged` in its report"
        )

    def _first_non_finite_layer(
        self, params: dict[str, np.ndarray]
    ) -> str | None:
        """Name of the first layer holding a non-finite weight, if any."""
        for name in sorted(params):
            if not np.all(np.isfinite(params[name])):
                return name
        return None

    def _check_round_health(
        self,
        round_num: int,
        train_loss: float,
        val_loss: float,
        params: dict[str, np.ndarray] | None,
    ) -> None:
        """Run the per-round finiteness gates (audit ML-04).

        Cheap: two scalar checks plus one ``np.isfinite`` sweep over the
        model (~180 K floats for DeepCNN, microseconds).  The weight sweep
        is what catches a collapse that has not yet reached the reported
        losses.
        """
        if not math.isfinite(train_loss):
            self._flag_divergence(round_num, f"train_loss={train_loss}")
        if not math.isfinite(val_loss):
            self._flag_divergence(round_num, f"val_loss={val_loss}")
        if params:
            bad = self._first_non_finite_layer(params)
            if bad is not None:
                self._flag_divergence(
                    round_num, f"non_finite_weights:{bad}"
                )

    def _finite_scores(
        self, scores: dict[str, float], round_num: int
    ) -> dict[str, float]:
        """Clamp importance scores to non-negative, flagging non-finite ones.

        ``max(0.0, float(score))`` looks like a sanitizer but is not:
        CPython's ``max`` returns its first argument when the comparison is
        False, so ``max(nan, 0.0)`` is nan — the exact case that must never
        pass silently (audit ML-04).  A non-finite score means the local
        model has already diverged, so it is flagged here and then zeroed:
        zeroing keeps the manifest well-formed (the receiver would drop a
        NaN-scored manifest and fall back to the count trigger, hiding the
        collapse), while the marker records that the round is not data.
        """
        clean: dict[str, float] = {}
        for name, value in scores.items():
            score = float(value)
            if not math.isfinite(score):
                self._flag_divergence(
                    round_num, f"non_finite_importance:{name}"
                )
                score = 0.0
            clean[name] = max(0.0, score)
        return clean

    def _train_epoch(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[float, float, dict[str, np.ndarray]]:
        """Run one epoch of GradientTape training.

        This is a synchronous function that runs inside a thread pool via
        ``loop.run_in_executor``.  It builds a shuffled tf.data pipeline,
        iterates over mini-batches, and applies gradients.  The gradients
        of the *last* mini-batch in the epoch are captured and returned —
        these are used by the importance metric to rank layers.

        Returns:
            (average_loss, accuracy, last_batch_gradients)
        """
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(len(x_train)).batch(self.batch_size)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        last_grads: dict[str, np.ndarray] = {}

        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                logits = self.model(x_batch, training=True)
                loss = self.loss_fn(y_batch, logits)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            batch_size = len(y_batch)
            total_loss += float(loss) * batch_size
            predictions = tf.argmax(logits, axis=1)
            total_correct += int(tf.reduce_sum(
                tf.cast(
                    predictions == tf.cast(y_batch, predictions.dtype),
                    tf.int32,
                )
            ))
            total_samples += batch_size

            # Capture the last mini-batch gradients.  For the importance
            # metric this is a cheap proxy — an alternative would be to
            # accumulate or exponentially average over the whole epoch,
            # but last-batch is what the literature typically uses.
            last_grads = {
                var.path: grad.numpy()
                for var, grad in zip(self.model.trainable_variables, gradients)
                if grad is not None
            }

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return avg_loss, accuracy, last_grads

    def _evaluate(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, float]:
        """Evaluate the model on a dataset.  Returns (loss, accuracy).

        Like ``_train_epoch``, this is a synchronous function intended to
        run inside a thread pool.
        """
        logits = self.model(x, training=False)
        loss = float(self.loss_fn(y, logits))
        predictions = tf.argmax(logits, axis=1)
        accuracy = float(tf.reduce_mean(
            tf.cast(
                predictions == tf.cast(y, predictions.dtype), tf.float32
            )
        ))
        return loss, accuracy

    # ------------------------------------------------------------------
    # Communication
    # ------------------------------------------------------------------

    async def _send_class_queue(
        self,
        dest_id: str,
        traffic_class: int,
        head_envelopes: list[federation_pb2.Envelope],
        tail_envelopes: list[federation_pb2.Envelope],
        head_done: asyncio.Future,
        cancel_event: asyncio.Event,
        round_num: int,
    ) -> None:
        """Drain one (destination, traffic class) send queue: head then tail.

        Exactly one coroutine per (destination, class, round) writes the
        class socket, so per-socket byte order is FIFO and frames never
        interleave.  The head portion resolves ``head_done`` (awaited by
        `_send_updates` — heads govern t_ε); the tail portion continues in
        the background and checks ``cancel_event`` between envelopes
        (cooperative cancellation: an envelope inside ``sock_sendall``
        cannot be recalled without corrupting the stream framing, so
        cancellation only ever skips *unsent* envelopes — pre-run fix 3).
        """
        head_bytes = 0
        head_metrics: list[dict] = []
        try:
            for envelope in head_envelopes:
                send_start = time.monotonic()
                sent = await self.pool.send(
                    dest_id, envelope, traffic_class=traffic_class,
                )
                # ENQUEUE time, not wire time (audit NT-01): sock_sendall
                # returns when the kernel accepted the bytes, so a layer that
                # fits inside SO_SNDBUF times a memcpy.  Kept because socket
                # back-pressure is still the sender-side cost signal, and
                # because the class queues overlap in wall-clock so the sum is
                # not an end-to-end time either.  The layer's wire time is the
                # receiver's layer_wire_times_oneway_s; the coverage KPI is
                # t_eps_local_receiver_s (G1).
                head_bytes += sent
                head_metrics.append({
                    "layer_name": envelope.layer_update.layer_name,
                    "send_enqueue_duration_sender_s": (
                        time.monotonic() - send_start
                    ),
                    "importance": envelope.layer_update.importance,
                    "traffic_class": traffic_class,
                    "bytes_sent": sent,
                })
            if not head_done.done():
                head_done.set_result((head_bytes, head_metrics))
        except Exception as exc:  # noqa: BLE001 - surfaced via the future
            if not head_done.done():
                head_done.set_exception(exc)
            else:  # pragma: no cover - defensive
                logger.warning(
                    f"[{self.node_id}] Head send to {dest_id} class "
                    f"{traffic_class} failed after completion: {exc}"
                )
            return

        if not tail_envelopes:
            return
        ledger = self._tail_ledger(round_num)
        for index, envelope in enumerate(tail_envelopes):
            if cancel_event.is_set():
                self._record_tail_cancellation(
                    round_num, tail_envelopes[index:]
                )
                return
            layer_name = envelope.layer_update.layer_name
            try:
                send_start = time.monotonic()
                sent = await self.pool.send(
                    dest_id, envelope, traffic_class=traffic_class,
                )
            except asyncio.CancelledError:
                # Hard task cancellation (shutdown path) — anything from
                # this envelope on was not (fully) sent.
                self._record_tail_cancellation(
                    round_num, tail_envelopes[index:]
                )
                raise
            except (ConnectionError, OSError) as exc:
                logger.warning(
                    f"[{self.node_id}] Tail send to {dest_id} class "
                    f"{traffic_class} failed at {layer_name!r}: {exc}"
                )
                self._record_tail_cancellation(
                    round_num, tail_envelopes[index:]
                )
                return
            entry = {
                "layer_name": layer_name,
                "send_enqueue_duration_sender_s": time.monotonic() - send_start,
                "importance": envelope.layer_update.importance,
                "traffic_class": traffic_class,
                "bytes_sent": sent,
            }
            if round_num in self._reported_rounds:
                # This round's MetricsReport already shipped; attribute the
                # straggler via the _sender block of the next report so the
                # $-accounting still sees every byte.
                self._sender_events.append({
                    "round": round_num,
                    "type": "tail_sent_late",
                    "bytes": sent,
                    "layers": [layer_name],
                })
            else:
                ledger["metrics"].append(entry)

    async def _send_updates(
        self,
        destinations: list[tuple[str, TrainingUpdate]],
        plan: _SendPlan | None,
    ) -> tuple[int, list[dict]]:
        """Serialize and send training updates to all destinations.

        Destinations are served **concurrently** (gate ruling G5: the
        aggregator's broadcast was serialized over workers, adding a
        constant per-round penalty; the same gather also benefits
        multi-neighbor decentralized sends).

        In **monolithic mode** each destination receives a single
        ModelUpdate protobuf over traffic class 0; ``plan`` is ignored.

        In **per-layer mode** ``plan`` drives everything: the layers it
        assigns are the layers transmitted (omission semantics — absent
        layers are simply not sent this round), head layers are awaited
        (they govern t_ε), tail layers continue on background per-class
        tasks subject to zombie cancellation, and — when the plan carries
        trigger scores — each destination first receives a `RoundManifest`
        on class 0 so the ε-trigger threshold is known before any payload
        arrives.  Downlink plans carry no trigger scores: no manifest is
        sent and receivers complete via the count trigger.

        Returns:
            Tuple of ``(bytes_sent, layer_comm_metrics)`` covering the
            manifest and head sends; completed tail sends are accounted
            separately via `_collect_tail_round_metrics` (they finish
            during the barrier/aggregation window or later).  In monolithic
            mode the metric list is empty (no per-layer breakdown).
        """
        if not destinations:
            return 0, []
        total_bytes = 0
        layer_metrics: list[dict] = []

        if self.update_mode == "per_layer":
            if plan is None:
                # Defensive fallback for direct callers: full model on
                # class 0, no manifest (legacy pre-manifest behaviour).
                template = destinations[0][1]
                plan = _SendPlan(
                    round_num=template.round_num,
                    assignment=AssignmentResult(
                        assignment={n: 0 for n in template.parameters},
                        head=set(template.parameters),
                        tail=set(),
                    ),
                    trigger_scores=None,
                    sched_scores=None,
                    must_receive=set(),
                )
            round_num = plan.round_num
            assignment = plan.assignment

            # A new send round implicitly closes every previous one: cancel
            # and await their tail tasks so (a) zombie bytes stop competing
            # with this round's head and (b) no two coroutines ever write
            # the same class socket (pre-run fix 3).
            await self._retire_tail_rounds_before(round_num)
            cancel_event = self._tail_cancel_events.setdefault(
                round_num, asyncio.Event()
            )

            manifest = None
            if plan.trigger_scores is not None:
                # Entries cover the transmitted layers in assignment order
                # (within a class that is the strategy's send order), PLUS
                # the skip-omitted layers appended at the end, flagged
                # `skipped` with their full trigger mass (skip-feedback v2:
                # the denominator must never shrink — the receiver credits
                # this mass as covered-by-recycling).
                ordered_scores = {
                    name: plan.trigger_scores.get(name, 0.0)
                    for name in assignment.assignment
                }
                for name in sorted(plan.skipped):
                    ordered_scores[name] = plan.trigger_scores.get(name, 0.0)
                manifest = build_manifest(
                    round_num=round_num,
                    source_node_id=self.node_id,
                    importance_scores=ordered_scores,
                    must_receive_predicate=(
                        lambda name, _score: name in plan.must_receive
                    ),
                    sched_scores=(
                        {
                            name: plan.sched_scores.get(name, 0.0)
                            for name in ordered_scores
                        }
                        if plan.sched_scores is not None else None
                    ),
                    skipped_layers=plan.skipped,
                )

            loop = asyncio.get_running_loop()

            async def _dispatch(
                dest_id: str, update: TrainingUpdate
            ) -> tuple[int, list[dict]]:
                dest_bytes = 0
                dest_metrics: list[dict] = []
                # Omission semantics: transmit exactly the assigned layers,
                # in assignment order.  total_layers in each LayerUpdate
                # then counts transmitted layers, keeping the receiver's
                # count trigger consistent with partial schedules (cyclic).
                filtered = TrainingUpdate(
                    source_node=update.source_node,
                    round_num=update.round_num,
                    parameters={
                        name: update.parameters[name]
                        for name in assignment.assignment
                        if name in update.parameters
                    },
                    num_samples=update.num_samples,
                )
                if manifest is not None:
                    manifest_env = serialize_round_manifest(
                        manifest, self.node_id, dest_id,
                    )
                    # Awaited before any payload task starts: the manifest
                    # must precede every payload at the receiver (class 0).
                    dest_bytes += await self.pool.send(
                        dest_id, manifest_env, traffic_class=0,
                    )
                layer_envelopes = serialize_layer_updates(
                    filtered,
                    self.node_id,
                    dest_id,
                    # Logged importance prefers the telemetry-only
                    # ``log_scores`` (the downlink's real delta_sq_norm
                    # motion) and falls back to the manifest trigger scores
                    # on the uplink where the two coincide.
                    importance_scores=(
                        plan.log_scores
                        if plan.log_scores is not None
                        else plan.trigger_scores
                    ),
                    traffic_classes=assignment.assignment,
                )
                # One queue per traffic class: each class has its own
                # tc-shaped socket whose kernel queue drains independently,
                # so classes are sent concurrently (max-over-classes, not
                # the sum — see docs/experiments/02-send-perf.md, P0c).
                by_class: dict[int, list] = {}
                for envelope, traffic_class in layer_envelopes:
                    by_class.setdefault(traffic_class, []).append(envelope)
                head_futures: list[asyncio.Future] = []
                for traffic_class, envelopes in sorted(by_class.items()):
                    head_envs = [
                        e for e in envelopes
                        if e.layer_update.layer_name in assignment.head
                    ]
                    tail_envs = [
                        e for e in envelopes
                        if e.layer_update.layer_name not in assignment.head
                    ]
                    head_done: asyncio.Future = loop.create_future()
                    task = asyncio.create_task(self._send_class_queue(
                        dest_id, traffic_class, head_envs, tail_envs,
                        head_done, cancel_event, round_num,
                    ))
                    self._tail_tasks.setdefault(round_num, []).append(task)
                    head_futures.append(head_done)
                for head_bytes, head_metrics in await asyncio.gather(
                    *head_futures
                ):
                    dest_bytes += head_bytes
                    dest_metrics.extend(head_metrics)
                return dest_bytes, dest_metrics

            results = await asyncio.gather(*[
                _dispatch(dest_id, update)
                for dest_id, update in destinations
            ])
            for dest_bytes, dest_metrics in results:
                total_bytes += dest_bytes
                layer_metrics.extend(dest_metrics)
        else:
            # Monolithic: single envelope per destination, always class 0,
            # destinations concurrent (the G5 broadcast gather — the only
            # monolithic-mode change).
            async def _dispatch_monolithic(
                dest_id: str, update: TrainingUpdate
            ) -> int:
                envelope = serialize_model_update(
                    update, self.node_id, dest_id,
                )
                return await self.pool.send(
                    dest_id, envelope, traffic_class=0,
                )

            sent_counts = await asyncio.gather(*[
                _dispatch_monolithic(dest_id, update)
                for dest_id, update in destinations
            ])
            total_bytes += sum(sent_counts)

        return total_bytes, layer_metrics

    async def _send_skip_advice(self, round_num: int) -> int:
        """Piggyback skip advice + inclusion acks on the broadcast (plan T1).

        Called by ``run_aggregator`` inside the send window, immediately
        BEFORE the model broadcast: each envelope rides class 0 of its
        destination's connection, so per-socket FIFO guarantees it arrives
        before the model payload — the worker therefore has it strictly
        before planning the next round.

        Two payloads share the message.  **Skip advice** (plan T1) is
        round-tagged with ``round_num``; receivers apply it to
        ``round_num + 1`` only.  **Realized-inclusion acks** (audit
        TRIG-1/ML-01) tell each worker which of ITS layers actually entered
        this round's aggregate; that — not the sender's own head placement —
        is what resets the staleness counters, so an ack is sent to every
        worker that took part, including one whose layers all missed.

        Fail-open for the advice half (a lost message only costs bytes),
        fail-safe for the ack half (a worker that misses an ack ages every
        layer rather than resetting on an unverified proxy).  Returns the
        bytes actually sent (counted into the round's ``bytes_sent``).
        """
        if self.update_mode != "per_layer":
            return 0
        advice_by_dest: dict[str, list[str]] = {}
        if self.skip_feedback_mode != "off":
            advice_by_dest = {
                destination: sorted(layers)
                for destination, layers in (
                    getattr(self.algorithm, "last_skip_advice", None) or {}
                ).items()
                if layers
            }
        # Acks cost one small class-0 envelope per worker per round, so they
        # are sent only for the arms that consume them — otherwise every
        # non-aging arm's downlink byte count would move for nothing.  The
        # aggregator reads its OWN aging config: node configs are generated
        # from one experiment YAML, so it mirrors the workers'.  A worker
        # that ages but gets no acks falls back loudly (see
        # _advance_layer_ages), it does not silently mis-age.
        acks_by_dest = (
            self._realized_inclusions(round_num)
            if self.aging_mode != "none"
            and self.aging_age_basis == "inclusion"
            else {}
        )
        targets = sorted(set(advice_by_dest) | set(acks_by_dest))
        if not targets:
            return 0

        async def _one(destination: str) -> int:
            envelope = federation_pb2.Envelope(
                source_node=self.node_id,
                dest_node=destination,
                timestamp_ns=time.monotonic_ns(),
                skip_advice=federation_pb2.SkipAdvice(
                    round=round_num,
                    layer_names=advice_by_dest.get(destination, []),
                    inclusion_ack=destination in acks_by_dest,
                    included_layers=acks_by_dest.get(destination, []),
                ),
            )
            return await self.pool.send(
                destination, envelope, traffic_class=0,
            )

        results = await asyncio.gather(
            *[_one(destination) for destination in targets],
            return_exceptions=True,
        )
        total_bytes = 0
        for destination, outcome in zip(targets, results):
            if isinstance(outcome, BaseException):
                logger.warning(
                    f"[{self.node_id}] Skip advice / inclusion ack to "
                    f"{destination} for round {round_num} not delivered "
                    f"(the worker will transmit everything and age every "
                    f"layer): {outcome}"
                )
            else:
                total_bytes += outcome
        logger.info(
            f"[{self.node_id}] Skip advice round {round_num}: "
            f"{len(advice_by_dest)} worker(s) advised, "
            f"{len(acks_by_dest)} inclusion ack(s), {total_bytes} bytes"
        )
        return total_bytes

    async def _send_metrics_to_monitor(self, metrics: dict[str, Any]) -> None:
        """Send a MetricsReport protobuf to the monitor container.

        The monitor aggregates these into TensorBoard logs and a final JSON
        report.  If the monitor is unreachable the error is logged but does
        not interrupt training.
        """
        if not self.monitor_ip:
            return

        report = federation_pb2.MetricsReport(
            node_id=self.node_id,
            round=metrics["round"],
            train_loss=metrics["train_loss"],
            train_accuracy=metrics["train_accuracy"],
            val_loss=metrics["val_loss"],
            val_accuracy=metrics["val_accuracy"],
            round_duration_s=metrics["round_duration_s"],
            train_duration_s=metrics["train_duration_s"],
            comm_duration_s=metrics["comm_duration_s"],
            # Proto field name/number frozen for wire compatibility; the value
            # is the kernel-accept (enqueue) duration and is exported under
            # its true name everywhere else (audit NT-01).
            send_duration_s=metrics.get(
                "send_enqueue_duration_sender_s", 0.0
            ),
            barrier_wait_duration_s=metrics.get("barrier_wait_duration_s", 0.0),
            aggregation_duration_s=metrics["aggregation_duration_s"],
            learning_rate=float(self.optimizer.learning_rate),
            # Divergence marker (audit ML-04): empty reason == healthy round.
            diverged=bool(metrics.get("diverged_reason")),
            diverged_reason=metrics.get("diverged_reason", ""),
        )

        # Populate per-layer communication metrics (per-layer mode only).
        # These carry the bytes-sent, enqueue duration, importance score, and
        # traffic class for each layer — the monitor aggregates them into
        # per-class TensorBoard scalars for traffic distribution analysis.
        for lm in metrics.get("layer_comm_metrics", []):
            report.layer_comm_metrics.append(
                federation_pb2.LayerCommMetric(
                    layer_name=lm["layer_name"],
                    send_duration_s=lm["send_enqueue_duration_sender_s"],
                    importance=lm["importance"],
                    traffic_class=lm["traffic_class"],
                    bytes_sent=lm["bytes_sent"],
                )
            )

        # Receiver-side uplink telemetry sidecar (G1): the primary Tier-1
        # KPI t_eps_local_receiver_s travels here.  Empty string when this node
        # received no manifested per-layer traffic this round.
        report.uplink_telemetry_json = self._export_uplink_telemetry(
            metrics["round"]
        )
        # From this point on, tail activity for this round must be
        # attributed via the next report's _sender block.
        self._reported_rounds.add(metrics["round"])

        envelope = federation_pb2.Envelope(
            source_node=self.node_id,
            dest_node="monitor",
            timestamp_ns=time.monotonic_ns(),
            metrics_report=report,
        )
        try:
            await self.pool.send("monitor", envelope, traffic_class=0)
        except (ConnectionError, OSError) as e:
            logger.warning(
                f"[{self.node_id}] Failed to send metrics to monitor: {e}"
            )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def _eta_min(self) -> float:
        """Learning-rate floor for the non-constant schedules (audit ML-03).

        ``lr_eta_min`` if configured, else 1 % of the base LR.  Explicit 0.0
        reproduces the pre-fix schedules, which reach EXACTLY zero on the
        final round.  Never above the base LR (a floor that exceeds the
        ceiling would silently turn cosine into a constant).
        """
        configured = self.config.get("lr_eta_min")
        eta_min = (
            0.01 * self._base_lr if configured is None else float(configured)
        )
        return max(0.0, min(eta_min, self._base_lr))

    def _scheduled_lr(self, round_num: int) -> float:
        """Round-indexed learning rate (writeup/16 §5.2).

        'constant' returns base_lr unchanged (legacy, bit-exact).  'cosine'
        anneals base_lr -> eta_min over total_rounds; 'step' multiplies by
        lr_decay_factor every lr_decay_every rounds, floored at eta_min.

        The floor is the audit ML-03 fix and it is not cosmetic.  With
        ``horizon = total_rounds - 1`` the last round has t/horizon = 1 and
        ``cos(pi)`` is exactly -1.0, so the pre-fix schedule trained the
        final round at lr = 0 — with no weight decay and no client momentum
        the shipped delta is then identically zero, which (a) makes every
        delta-sq-norm trigger score 0, so the manifest is degenerate and the
        ε mechanism is OFF, (b) refuses to shed, so the full model is
        transmitted in the round steady-state byte statistics read, and
        (c) leaves endpoint accuracy a re-evaluation of the previous round's
        model (or, under server momentum, a pure θ+βv extrapolation).  All
        three land exactly in the round the campaigns read their headline
        numbers from.
        """
        schedule = self.config.get("lr_schedule", "constant")
        if schedule == "cosine":
            eta_min = self._eta_min()
            horizon = max(self.total_rounds - 1, 1)
            t = min(round_num, horizon)
            cosine = 0.5 * (1.0 + math.cos(math.pi * t / horizon))
            return eta_min + (self._base_lr - eta_min) * cosine
        if schedule == "step":
            every = int(self.config.get("lr_decay_every", 0))
            if every >= 1:
                factor = float(self.config.get("lr_decay_factor", 0.1))
                return max(
                    self._eta_min(),
                    self._base_lr * factor ** (round_num // every),
                )
        return self._base_lr

    def _apply_lr_schedule(self, round_num: int) -> None:
        """Assign this round's scheduled LR to the optimizer (idempotent).

        Under 'constant' the assign never fires, keeping legacy runs
        bit-identical.
        """
        new_lr = self._scheduled_lr(round_num)
        if new_lr != float(self.optimizer.learning_rate):
            self.optimizer.learning_rate.assign(new_lr)
            logger.info(
                f"[{self.node_id}] LR schedule "
                f"[{self.config.get('lr_schedule', 'constant')}]: "
                f"round {round_num} lr={new_lr:.6g}"
            )

    async def run_sync(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Run the synchronous training loop (D-PSGD, FedAvg, etc.).

        Each round proceeds as:
          train → compute importance → send → wait → aggregate → evaluate → report

        The loop is fully async: training and evaluation run in the default
        thread-pool executor, while sending/receiving/aggregating happen on
        the event loop.

        IMPORTANT: ``_register_message_handler()`` must be called *before*
        neighbor connections are established.  The caller (``node.py``) is
        responsible for doing this after constructing the engine but before
        calling ``connect()`` on the pool.  We do NOT register here because
        by the time ``run_sync`` is called, connections are already open and
        messages may have already arrived.
        """
        loop = asyncio.get_running_loop()
        num_samples = len(x_train)

        logger.info(
            f"[{self.node_id}] Starting synchronous training: "
            f"{self.total_rounds} rounds, {self.epochs_per_round} epochs/round, "
            f"{num_samples} samples, mode={self.update_mode}"
        )

        for round_num in range(self.total_rounds):
            round_start = time.monotonic()
            # Receiver-local round-start instant: the reference for this
            # round's uplink telemetry offsets (G1).
            self._round_starts[round_num] = round_start
            self._apply_lr_schedule(round_num)

            # Snapshot the shipped base BEFORE training: the importance
            # input is the shipped multi-epoch delta θ_after − θ_round_start
            # (gate headline fix 3), not the last minibatch gradient.
            # ~731 KB copy for DeepCNN — negligible.
            round_start_params: dict[str, np.ndarray] | None = None
            if self.update_mode == "per_layer":
                round_start_params = FederationModel.get_parameters(self.model)

            # --- 1. Local training (non-blocking) ---
            train_start = time.monotonic()
            train_loss = 0.0
            train_acc = 0.0
            for _ in range(self.epochs_per_round):
                epoch_loss, epoch_acc, grads = await loop.run_in_executor(
                    None, self._train_epoch, x_train, y_train,
                )
                train_loss = epoch_loss
                train_acc = epoch_acc
                self.last_gradients = grads
            train_duration = time.monotonic() - train_start

            # --- 2. Plan the dissemination (per-layer mode only) ---
            # Trigger scores (frozen delta-sq-norm accounting, G2), sched
            # scores (configured metric_v2 + aging), warm-up ε, and the
            # pluggable assignment — all derived from the shipped delta.
            params = FederationModel.get_parameters(self.model)
            plan: _SendPlan | None = None
            if self.update_mode == "per_layer":
                deltas = {
                    name: params[name] - round_start_params[name]
                    for name in params
                    if name in round_start_params
                }
                plan = self._prepare_layer_dissemination(
                    params, deltas, round_num,
                )

            # --- 3. Send updates ---
            # send_enqueue_duration measures ONLY the time in _send_updates
            # (handing manifest + head to the kernel; the deadline-exempt tail
            # continues in the background); barrier_wait below measures the
            # idle time blocked on the slowest peer.  They were previously
            # conflated into one comm_duration, which hid that sending is a
            # tiny fraction of the total (docs/experiments/02-send-perf.md).
            #
            # It is an ENQUEUE time in the SENDER clock domain, never wire
            # time (audit NT-01): sock_sendall returns at kernel accept, so
            # any payload smaller than SO_SNDBUF is timed as a memcpy.  Wire
            # truth is receiver-side — t_cover_oneway_s per flow and the
            # `_receiver_flows` completion stamps.
            send_start = time.monotonic()
            outgoing = await self.algorithm.on_local_training_complete(
                params, round_num, num_samples,
            )
            bytes_sent, layer_comm_metrics = await self._send_updates(
                outgoing, plan,
            )
            send_enqueue_duration = time.monotonic() - send_start

            # --- 4. Wait for all neighbor updates ---
            wait_start = time.monotonic()
            sync_timeout = self.config.get("sync_timeout", 120.0)
            ready = await self.algorithm.wait_for_aggregation(timeout=sync_timeout)
            if not ready:
                logger.error(
                    f"[{self.node_id}] Timeout waiting for neighbors "
                    f"in round {round_num}"
                )
            barrier_wait_duration = time.monotonic() - wait_start
            comm_duration = send_enqueue_duration + barrier_wait_duration

            # --- 5. Aggregate ---
            agg_start = time.monotonic()
            new_params = await self.algorithm.aggregate(params)
            FederationModel.set_parameters(self.model, new_params)
            agg_duration = time.monotonic() - agg_start

            # --- 6. Evaluate ---
            val_loss, val_acc = await loop.run_in_executor(
                None, self._evaluate, x_val, y_val,
            )

            round_duration = time.monotonic() - round_start

            # --- 7. Record and report metrics ---
            # Fold in tail sends that completed during the barrier /
            # aggregation / evaluation window so per-class byte accounting
            # stays complete; later stragglers ride the next report's
            # _sender block.
            tail_bytes, tail_metrics = self._collect_tail_round_metrics(
                round_num
            )
            # Divergence gate (audit ML-04): run on the post-aggregation
            # weights, so a collapse is caught in the round it happens
            # rather than inferred from a 0.100 accuracy three rounds later.
            self._check_round_health(
                round_num, train_loss, val_loss, new_params,
            )
            metrics = {
                "round": round_num,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "round_duration_s": round_duration,
                "train_duration_s": train_duration,
                "comm_duration_s": comm_duration,
                "send_enqueue_duration_sender_s": send_enqueue_duration,
                "barrier_wait_duration_s": barrier_wait_duration,
                "aggregation_duration_s": agg_duration,
                "bytes_sent": bytes_sent + tail_bytes,
                "learning_rate": float(self.optimizer.learning_rate),
                # Per-layer communication metrics — populated in per-layer
                # mode only.  The monitor aggregates these by traffic class
                # for TensorBoard visualization.
                "layer_comm_metrics": layer_comm_metrics + tail_metrics,
                "diverged_reason": self._diverged_rounds.get(round_num, ""),
            }
            self.metrics_history.append(metrics)

            logger.info(
                f"[{self.node_id}] Round {round_num}/{self.total_rounds}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"time={round_duration:.2f}s"
            )

            await self._send_metrics_to_monitor(metrics)

            # --- 8. Advance round ---
            self.algorithm.advance_round()
            self._layer_buffer.clear_stale(round_num)
            self._gc_round_state(round_num)

        # Let the final round's tail drain (still useful to peers finishing
        # their last aggregation) and reap receiver-side timers.
        await self._drain_send_tasks()
        logger.info(
            f"[{self.node_id}] Training complete after {self.total_rounds} rounds"
        )
        return self.metrics_history

    async def run_async(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Run the asynchronous training loop (A-DPSGD, Gossip-SGD, etc.).

        Structurally similar to ``run_sync`` but with two key differences:

        1. **No blocking barrier** — instead of calling
           ``wait_for_aggregation`` (which blocks until all neighbors have
           responded), we call ``ready_to_aggregate`` (a non-blocking
           check).  If updates have arrived, we aggregate immediately;
           otherwise we continue to the next iteration without waiting.

        2. **Periodic evaluation** — to reduce overhead in async mode
           (where iterations are fast), validation is only run every
           ``eval_every`` iterations instead of every single one.  The
           last iteration always evaluates regardless of this setting.

        The ``round`` field in metrics reports is set to the local iteration
        number.  Since each node advances at its own pace, the monitor sees
        different ``round`` values per node at any given time.

        IMPORTANT: same handler-registration requirement as ``run_sync``.
        """
        loop = asyncio.get_running_loop()
        num_samples = len(x_train)

        logger.info(
            f"[{self.node_id}] Starting asynchronous training: "
            f"{self.total_rounds} iterations, "
            f"{self.epochs_per_round} epochs/iteration, "
            f"{num_samples} samples, mode={self.update_mode}, "
            f"eval_every={self.eval_every}"
        )

        for iteration in range(self.total_rounds):
            iter_start = time.monotonic()
            # Receiver-local reference instant for this iteration's uplink
            # telemetry offsets (G1).  Iterations are the async "rounds".
            self._round_starts[iteration] = iter_start
            self._apply_lr_schedule(iteration)

            # Shipped-delta base snapshot — see run_sync for rationale.
            round_start_params: dict[str, np.ndarray] | None = None
            if self.update_mode == "per_layer":
                round_start_params = FederationModel.get_parameters(self.model)

            # --- 1. Local training (non-blocking) ---
            train_start = time.monotonic()
            train_loss = 0.0
            train_acc = 0.0
            for _ in range(self.epochs_per_round):
                epoch_loss, epoch_acc, grads = await loop.run_in_executor(
                    None, self._train_epoch, x_train, y_train,
                )
                train_loss = epoch_loss
                train_acc = epoch_acc
                self.last_gradients = grads
            train_duration = time.monotonic() - train_start

            # --- 2. Plan the dissemination (per-layer mode only) ---
            params = FederationModel.get_parameters(self.model)
            plan: _SendPlan | None = None
            if self.update_mode == "per_layer":
                deltas = {
                    name: params[name] - round_start_params[name]
                    for name in params
                    if name in round_start_params
                }
                plan = self._prepare_layer_dissemination(
                    params, deltas, iteration,
                )

            # --- 3. Send updates ---
            # Async mode does not block on a barrier (step 4 is non-blocking),
            # so the whole comm window is the enqueue window and barrier_wait
            # is 0.  Sender clock domain, kernel-accept semantics — see
            # run_sync for why this is not a wire time (audit NT-01).
            send_start = time.monotonic()
            outgoing = await self.algorithm.on_local_training_complete(
                params, iteration, num_samples,
            )
            bytes_sent, layer_comm_metrics = await self._send_updates(
                outgoing, plan,
            )
            send_enqueue_duration = time.monotonic() - send_start
            barrier_wait_duration = 0.0
            comm_duration = send_enqueue_duration

            # --- 4. Non-blocking aggregation check ---
            # In async mode we do NOT wait for neighbors.  If any updates
            # have been buffered by on_update_received (running on the
            # event loop while we were training), aggregate now.  Otherwise
            # skip aggregation and continue with the next iteration — the
            # model stays unchanged until updates arrive.
            agg_start = time.monotonic()
            if self.algorithm.ready_to_aggregate():
                new_params = await self.algorithm.aggregate(params)
                FederationModel.set_parameters(self.model, new_params)
            agg_duration = time.monotonic() - agg_start

            # --- 5. Evaluate (periodically) ---
            is_last = iteration == self.total_rounds - 1
            should_eval = (iteration % self.eval_every == 0) or is_last

            if should_eval:
                val_loss, val_acc = await loop.run_in_executor(
                    None, self._evaluate, x_val, y_val,
                )
            else:
                # Carry forward last known validation metrics (or 0 if
                # we haven't evaluated yet).
                val_loss = (
                    self.metrics_history[-1]["val_loss"]
                    if self.metrics_history else 0.0
                )
                val_acc = (
                    self.metrics_history[-1]["val_accuracy"]
                    if self.metrics_history else 0.0
                )

            iter_duration = time.monotonic() - iter_start

            # --- 6. Record and report metrics ---
            # "round" in the metrics dict is set to the local iteration
            # number.  The monitor treats this as the round counter for
            # completion detection (all(r >= total_rounds - 1)).
            tail_bytes, tail_metrics = self._collect_tail_round_metrics(
                iteration
            )
            # Divergence gate (audit ML-04), on the model as it stands after
            # whatever aggregation this iteration did.
            self._check_round_health(
                iteration, train_loss, val_loss,
                FederationModel.get_parameters(self.model),
            )
            metrics = {
                "round": iteration,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "round_duration_s": iter_duration,
                "train_duration_s": train_duration,
                "comm_duration_s": comm_duration,
                "send_enqueue_duration_sender_s": send_enqueue_duration,
                "barrier_wait_duration_s": barrier_wait_duration,
                "aggregation_duration_s": agg_duration,
                "bytes_sent": bytes_sent + tail_bytes,
                "learning_rate": float(self.optimizer.learning_rate),
                "layer_comm_metrics": layer_comm_metrics + tail_metrics,
                "diverged_reason": self._diverged_rounds.get(iteration, ""),
            }
            self.metrics_history.append(metrics)

            logger.info(
                f"[{self.node_id}] Iter {iteration}/{self.total_rounds}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"time={iter_duration:.2f}s"
            )

            await self._send_metrics_to_monitor(metrics)

            # --- 7. Advance iteration ---
            self.algorithm.advance_round()
            # Use the staleness threshold from config for the layer buffer
            # cleanup too, so partially-received per-layer updates are kept
            # as long as the algorithm would accept the complete update.
            staleness = self.config.get("staleness_threshold", 2)
            max_staleness = max(staleness, 2)  # at least 2 to avoid premature cleanup
            self._layer_buffer.clear_stale(iteration, max_staleness=max_staleness)
            self._gc_round_state(iteration, max_staleness=max_staleness)

        await self._drain_send_tasks()
        logger.info(
            f"[{self.node_id}] Async training complete after "
            f"{self.total_rounds} iterations"
        )
        return self.metrics_history

    async def run_aggregator(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Run the aggregator loop for centralized algorithms (e.g. FedAvg).

        The aggregator does NOT train locally.  Each round:

          1. Wait for all workers to send their model updates.
          2. Aggregate them (e.g. weighted average for FedAvg).
          3. Set the local model to the aggregated parameters (for eval).
          4. Send the aggregated model back to all workers.
          5. Evaluate on the local validation set.
          6. Report metrics to the monitor.
          7. Advance round.

        The ``x_train`` / ``y_train`` arguments are accepted for API
        symmetry with ``run_sync`` but are not used — the aggregator does
        not perform local gradient descent.

        IMPORTANT: same handler-registration requirement as ``run_sync``.
        """
        loop = asyncio.get_running_loop()
        # Receiving the (own) broadcast must never trigger zombie-tail
        # cancellation on the aggregator; uplink completion does that here.
        self._is_aggregator = True

        logger.info(
            f"[{self.node_id}] Starting aggregator loop: "
            f"{self.total_rounds} rounds, "
            f"waiting for {len(self.algorithm.neighbors)} workers"
        )

        for round_num in range(self.total_rounds):
            round_start = time.monotonic()
            # Receiver-local reference instant for this round's uplink
            # telemetry offsets (G1) — the aggregator is the primary
            # t_eps_local_receiver_s observer.
            self._round_starts[round_num] = round_start
            self._apply_lr_schedule(round_num)

            # --- 1. Wait for all workers ---
            # For the aggregator, barrier_wait is the time spent here waiting
            # for the slowest worker's update; send_enqueue_duration (below)
            # covers the broadcast back to workers.  Symmetric with the worker
            # split — and, like the worker's, a kernel-accept time, not the
            # broadcast's wire time (audit NT-01).  The downlink's wire time
            # is measured at the workers, in their `_receiver_flows` block.
            wait_start = time.monotonic()
            sync_timeout = self.config.get("sync_timeout", 120.0)
            ready = await self.algorithm.wait_for_aggregation(
                timeout=sync_timeout,
            )
            if not ready:
                logger.error(
                    f"[{self.node_id}] Timeout waiting for workers "
                    f"in round {round_num} — retrying"
                )
                # Not all workers responded in time.  Retry the wait
                # rather than crashing — transient slowness (e.g. CPU
                # contention in Docker) should not abort the experiment.
                continue
            barrier_wait_duration = time.monotonic() - wait_start

            # --- 2. Aggregate (weighted average) ---
            agg_start = time.monotonic()
            # Pass current model params for interface compliance, but the
            # FedAvg aggregator ignores them (it uses only worker updates).
            current_params = FederationModel.get_parameters(self.model)
            aggregated = await self.algorithm.aggregate(current_params)
            agg_duration = time.monotonic() - agg_start
            # Snapshot the slippage realization (which sources entered each
            # layer, which fill action covered the missing ones) for the
            # report's `_aggregation` telemetry block — must happen before
            # the next aggregate() refreshes the attributes.
            self._capture_aggregation_telemetry(round_num)

            # --- 3. Set model to aggregated params (for evaluation) ---
            FederationModel.set_parameters(self.model, aggregated)

            # --- 4. Send aggregated model back to all workers ---
            # Reuse on_local_training_complete — the aggregator "completes"
            # by producing the global average.  Total samples is the sum
            # across all workers (already computed during aggregation).
            # Workers are served concurrently and, in per-layer mode, the
            # class map is byte-balanced and importance-blind (gate ruling
            # G5; rationale in _downlink_plan).
            send_start = time.monotonic()
            total_samples = len(x_train)  # aggregator's own partition size
            outgoing = await self.algorithm.on_local_training_complete(
                aggregated, round_num, total_samples,
            )
            # Skip-feedback v2: per-sender advice rides class 0 ahead of
            # the model payload (per-socket FIFO ⇒ advice precedes the
            # broadcast at every worker).  No-op for non-skip arms.
            advice_bytes = await self._send_skip_advice(round_num)
            downlink_plan = self._downlink_plan(aggregated, round_num)
            bytes_sent, layer_comm_metrics = await self._send_updates(
                outgoing, downlink_plan,
            )
            bytes_sent += advice_bytes
            send_enqueue_duration = time.monotonic() - send_start
            comm_duration = send_enqueue_duration + barrier_wait_duration

            # --- 5. Evaluate ---
            val_loss, val_acc = await loop.run_in_executor(
                None, self._evaluate, x_val, y_val,
            )

            round_duration = time.monotonic() - round_start

            # --- 6. Record and report metrics ---
            # Divergence gate (audit ML-04): the aggregator does not train,
            # so its 0.0 train_loss is not a signal — the aggregate's
            # weights and the val loss are.  A NaN entering here means a
            # worker shipped one and the whole federation is downstream of
            # it, which is exactly what must not be reported as a 0.100
            # accuracy datapoint.
            self._check_round_health(round_num, 0.0, val_loss, aggregated)
            # train_loss / train_accuracy are 0 because the aggregator
            # does not train locally.  The monitor uses these fields for
            # TensorBoard logging — 0 values clearly mark this node as
            # the aggregator in the plots.
            metrics = {
                "round": round_num,
                "train_loss": 0.0,
                "train_accuracy": 0.0,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "round_duration_s": round_duration,
                "train_duration_s": 0.0,
                "comm_duration_s": comm_duration,
                "send_enqueue_duration_sender_s": send_enqueue_duration,
                "barrier_wait_duration_s": barrier_wait_duration,
                "aggregation_duration_s": agg_duration,
                "bytes_sent": bytes_sent,
                "learning_rate": float(self.optimizer.learning_rate),
                "layer_comm_metrics": layer_comm_metrics,
                "diverged_reason": self._diverged_rounds.get(round_num, ""),
            }
            self.metrics_history.append(metrics)

            logger.info(
                f"[{self.node_id}] Round {round_num}/{self.total_rounds}: "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"agg_time={agg_duration:.3f}s, "
                f"round_time={round_duration:.2f}s"
            )

            await self._send_metrics_to_monitor(metrics)

            # --- 7. Advance round ---
            self.algorithm.advance_round()
            self._layer_buffer.clear_stale(round_num)
            self._gc_round_state(round_num)

        await self._drain_send_tasks()
        logger.info(
            f"[{self.node_id}] Aggregator complete after "
            f"{self.total_rounds} rounds"
        )
        return self.metrics_history
