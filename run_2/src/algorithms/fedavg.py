"""FedAvg (Federated Averaging) — centralized, synchronous.

Classic centralized federated learning algorithm where a designated
aggregator node coordinates training across multiple worker nodes.

**Worker loop** per round:
  1. Train locally for ``epochs_per_round``.
  2. Send the updated model parameters to the aggregator.
  3. Wait for the aggregator to return the globally averaged model.
  4. Replace the local model with the received global model.

**Aggregator loop** per round:
  1. Wait for all workers to submit their model updates.
  2. Compute a sample-weighted average of the worker parameters:
     ``new_params = Σ (n_k / N) * params_k``
     where ``n_k`` is the number of training samples on worker *k* and
     ``N`` is the total across all workers.
  3. Broadcast the averaged model back to all workers.
  4. (Optionally) evaluate on a local validation set.

The algorithm is implemented as a single class with role-based behavior.
The ``role`` field in the config dict (set to ``"aggregator"`` for the
aggregation server, defaulting to ``"worker"`` for everyone else)
determines which code path runs.  This avoids needing two separate
algorithm classes in the registry.

In a star topology the aggregator has N-1 neighbors (all workers) and
each worker has exactly one neighbor (the aggregator).  The algorithm
does not enforce star topology — it works with any connected graph where
one node is designated aggregator — but star is the standard setup.

**Slippage modes.**  Under the ε-deadline (per-layer mode) a worker's
update may be missing layers that slipped past the trigger.  How the
aggregator compensates for a missing (layer, source) contribution is
selected by the ``late_layer_policy`` config key (gate ruling G3,
``writeup/01-candidate-selection.md`` §7): ``drop`` stale-fills from the
current global (control), ``renormalize``/``renorm`` averages over the
arrived contributors with renormalized weights, and
``recycle_last_delta``/``recycle`` re-applies the layer's previous
aggregated delta on the missing sender's behalf (FedLUAR-adapted).
See :meth:`FedAvg._aggregate_as_aggregator`.

**Skip-feedback v2** (``writeup/04-phase1/plan.md`` T1; v2 re-spec in
``writeup/01-candidate-selection.md`` §4).  When ``skip_feedback`` is not
``"off"`` the aggregator additionally produces per-sender *skip advice*
after every aggregation — "do not resend these layers next round" — which
the engine piggybacks on the model broadcast (round-tagged, fail-open,
idempotent; see ``proto/federation.proto:SkipAdvice``).  Four advice modes:

- ``"shed"``: advise each sender the layers its update was missing this
  round (they were filled at aggregation, so resending next round buys
  little) — the v1 idea, made ratchet-safe by the manifest listing rule.
- ``"fedluar"``: the FedLUAR-native baseline.  A FIXED COUNT
  (``skip_fedluar_count``) of layers is sampled with probability inversely
  proportional to the layer's aggregated update-to-weight ratio
  ``‖Δ_ℓ‖/‖θ_ℓ‖`` (FedLUAR's selection rule), and the SAME set is advised
  to every sender (server-side global selection, as in the paper).
  Combined with recycle filling this reproduces FedLUAR's
  recycling-instead-of-transmitting natively — the strongest baseline.
- ``"fedluar_random"``: FedLUAR's own metric ablation (their Table 4
  "Random" row).  Same fixed count, same one-global-set broadcast, same
  recycle fill, same unbounded staleness — but the layers are drawn
  UNIFORMLY without replacement, consulting no importance signal.  It is
  the control that isolates the ``‖Δ_ℓ‖/‖θ_ℓ‖`` metric, and (being
  byte-blind) its expected uplink is exactly ``(L-delta)/L`` of the model,
  matching ``assignment_strategy="cyclic"`` at ``cyclic_k = L-delta`` in
  expectation while differing from it in staleness (unbounded vs the
  rotation's hard ``L/k``-round refresh).
- ``"fedluar_cyclic"``: the control FedLUAR never ran.  Same fixed count,
  same global set, same recycle fill, same bytes — but the skip set
  ROTATES round-robin, so staleness is bounded for free and no ``tau_max``
  machinery is needed.  Unlike ``assignment_strategy="cyclic"`` (which
  omits layers from the manifest entirely, leaving the aggregator to
  FREEZE them), this rotates through the recycle path, so
  ``fedluar_cyclic`` vs ``cyclic`` isolates the FILL rule at matched
  bytes — our analogue of FedLUAR's Table 5 — while ``fedluar_cyclic`` vs
  ``fedluar`` / ``fedluar_random`` isolates the SELECTION rule.

Contributions missing *because they were advised away* are always
recycle-filled (the layer's previous aggregated delta is re-applied on the
sender's behalf) regardless of ``late_layer_policy`` — the sender's
manifest declares that mass covered-by-recycling, and this is what makes
the declaration true.  ``renormalize`` is rejected in combination with
skip feedback: renormalizing re-weights away exactly the mass the manifest
declares covered.  Advised layers that arrive ANYWAY (advice lost, or
aging's τ_max forcing a send) are aggregated normally — fail-open.

Reference: McMahan et al., "Communication-Efficient Learning of Deep
Networks from Decentralized Data", AISTATS 2017.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.algorithms.base import FederationAlgorithm, TrainingUpdate

logger = logging.getLogger(__name__)


# Normalized slippage-mode names, keyed by every accepted spelling.  The
# config schema (src/config/schema.py) uses the long literals; the short
# forms are convenient internal/test aliases.  Keeping both avoids a
# silent mismatch between schema-validated configs and hand-built dicts.
_SLIPPAGE_MODE_ALIASES: dict[str, str] = {
    "drop": "drop",
    "renorm": "renorm",
    "renormalize": "renorm",
    "recycle": "recycle",
    "recycle_last_delta": "recycle",
}

#: skip_feedback config literals (skip-feedback v2, plan T1).  Must match
#: the schema literal set; the engine validates the same values.
_SKIP_FEEDBACK_MODES = (
    "off", "shed", "fedluar", "fedluar_random", "fedluar_cyclic",
)

#: The three fixed-count, one-global-set advice modes.  They are identical in
#: delta, in the global broadcast, in the recycle fill and in the bytes they
#: remove; they differ ONLY in how the delta skipped layers are chosen:
#: inverse-ratio sampling (FedLUAR's method), uniform i.i.d. resampling
#: (FedLUAR's own Table 4 "Random" ablation) and deterministic round-robin
#: rotation (the bounded-staleness control FedLUAR never ran).  All three
#: require ``skip_fedluar_count >= 1``.
_FEDLUAR_MODES = ("fedluar", "fedluar_random", "fedluar_cyclic")

#: Ratio floor for the FedLUAR inverse-ratio sampling weights: a layer whose
#: aggregated motion is (numerically) zero gets weight 1/floor — strongly
#: preferred for recycling, which is exactly FedLUAR's intent (no motion ⇒
#: recycling is free), without producing infinities.
_FEDLUAR_RATIO_FLOOR = 1e-12


class FedAvg(FederationAlgorithm):
    """Federated Averaging — centralized, synchronous, role-based.

    Parameters
    ----------
    node_id : str
        Unique identifier for this node.
    neighbors : list[str]
        IDs of neighboring nodes.  For workers this is typically just the
        aggregator; for the aggregator it is the list of all workers.
    config : dict
        Training configuration dict.  Reads:
        - ``role`` (str, default ``"worker"``): ``"aggregator"`` for the
          aggregation server, ``"worker"`` for training participants.
        - ``sync_timeout`` (float, default 120.0): Timeout in seconds for
          synchronous waits (workers waiting for the aggregator, aggregator
          waiting for all workers).
        - ``late_layer_policy`` (str, default ``"drop"``): slippage mode
          for layers missing from partial updates.  One of ``"drop"``,
          ``"renormalize"`` (alias ``"renorm"``) or
          ``"recycle_last_delta"`` (alias ``"recycle"``).  Only the
          aggregator acts on it; see ``_aggregate_as_aggregator``.
        - ``skip_feedback`` (str, default ``"off"``): skip-advice mode
          (``"off"``, ``"shed"``, ``"fedluar"``, ``"fedluar_random"`` —
          module docstring).  ``"renormalize"`` slippage is rejected in
          combination with skip feedback (semantic contradiction; module
          docstring).
        - ``skip_fedluar_count`` (int, default 0): FIXED number of layers
          advised per round in ``"fedluar"`` / ``"fedluar_random"`` mode;
          must be >= 1 there (clamped to L-1 at advice time so at least one
          layer always transmits).  Ignored by the other modes.
        - ``seed`` (int, default 42): seeds the fedluar sampling RNG (both
          the inverse-ratio and the uniform draw).
    """

    def __init__(self, node_id: str, neighbors: list[str], config: dict):
        super().__init__(node_id, neighbors, config)

        self._role: str = config.get("role", "worker")
        self._timeout: float = config.get("sync_timeout", 120.0)

        # ---- FedAvgM server momentum (Hsu et al. 2019; writeup/16 §5) ----
        # beta=0.0 skips the transform entirely, keeping plain FedAvg
        # bit-identical.  Velocity state lives on the aggregator only, and
        # accumulates FRESH-arrival motion only (audit ML-02; see the
        # decomposition at the momentum step in _aggregate_as_aggregator).
        self._server_momentum: float = float(
            config.get("server_momentum", 0.0) or 0.0
        )
        if not 0.0 <= self._server_momentum < 1.0:
            raise ValueError(
                f"server_momentum must be in [0, 1) "
                f"(got {self._server_momentum})"
            )
        self._server_velocity: dict[str, np.ndarray] = {}

        # Slippage mode (gate ruling G3).  Normalized here so the rest of
        # the class only ever compares against 'drop'/'renorm'/'recycle'.
        # Fail fast on typos: a misconfigured arm must die at construction
        # time, not silently run the control policy overnight.
        raw_mode = config.get("late_layer_policy", "drop")
        try:
            self._slippage_mode: str = _SLIPPAGE_MODE_ALIASES[raw_mode]
        except KeyError:
            valid = ", ".join(sorted(_SLIPPAGE_MODE_ALIASES))
            raise ValueError(
                f"Unknown late_layer_policy {raw_mode!r} for FedAvg. "
                f"Valid values: {valid}"
            ) from None

        # ---- Skip-feedback v2 (plan T1) ----
        # Same fail-fast discipline as the slippage mode: a misconfigured
        # arm must die at construction, not silently run as a control.
        self._skip_mode: str = config.get("skip_feedback", "off") or "off"
        if self._skip_mode not in _SKIP_FEEDBACK_MODES:
            valid = ", ".join(_SKIP_FEEDBACK_MODES)
            raise ValueError(
                f"Unknown skip_feedback {self._skip_mode!r} for FedAvg. "
                f"Valid values: {valid}"
            )
        self._skip_fedluar_count: int = int(config.get("skip_fedluar_count", 0))
        if self._skip_mode in _FEDLUAR_MODES and self._skip_fedluar_count < 1:
            raise ValueError(
                f"skip_feedback={self._skip_mode!r} requires "
                f"skip_fedluar_count >= 1 (got {self._skip_fedluar_count})"
            )
        if self._skip_mode != "off" and self._slippage_mode == "renorm":
            raise ValueError(
                "skip_feedback cannot be combined with "
                "late_layer_policy='renormalize': skipped layers are "
                "declared covered-by-recycling in the manifest, but "
                "renormalization re-weights their mass away instead of "
                "recycling it (use 'drop' or 'recycle_last_delta')"
            )
        self._skip_rng = np.random.default_rng(int(config.get("seed", 42)))
        #: Rotation counter for ``fedluar_cyclic`` — advanced once per
        #: advice computation (i.e. once per aggregation), never elsewhere,
        #: so the rotation phase is a pure function of the advice index.
        self._skip_cycle_index: int = 0

        # last_skip_advice: the advice produced by the most recent
        # aggregation — {destination worker id: [layer names]} — polled by
        # the engine and piggybacked on the broadcast (round-tagged there).
        # Always a fresh dict per aggregation; empty when skip_feedback is
        # off or nothing is advised.  Workers never populate it.
        self.last_skip_advice: dict[str, list[str]] = {}

        # ---- Aggregation telemetry (engine-pollable) ----
        # Refreshed by every _aggregate_as_aggregator call; workers never
        # populate them.  Defined unconditionally so the engine can read
        # them without role checks.
        #
        # last_aggregation_telemetry: per-layer realization of the most
        # recent aggregation —
        #   {layer: {"arrived_sources": [sorted source ids that contributed
        #            the layer], "filled": "none"|"stale"|"recycle"|"renorm"}}
        # "filled" describes the fill action taken for MISSING contributors
        # ("none" = layer complete).  An empty arrived_sources list with
        # "stale" marks the degenerate zero-arrived case (global retained).
        # Together with the round's manifests this is the realized-κ input:
        # which (layer, source) mass actually entered the aggregate.
        self.last_aggregation_telemetry: dict[str, dict[str, Any]] = {}
        # inclusion_counts: cumulative per-(layer, source) inclusion
        # counters, {layer: {source: #aggregations where the source
        # contributed the layer}}.  Sources that never contributed a layer
        # appear with an explicit 0.  Rate = count / aggregation_count.
        self.inclusion_counts: dict[str, dict[str, int]] = {}
        self.aggregation_count: int = 0

        if self._role == "aggregator":
            # Aggregator buffers worker updates keyed by source node.
            # One update per worker per round (last-write wins if a
            # duplicate arrives, though that shouldn't happen in the
            # synchronous protocol).
            self._worker_buffer: dict[str, TrainingUpdate] = {}
            # Recycle-last-delta state (maintained when _slippage_mode ==
            # 'recycle' OR skip feedback is on — advised omissions are
            # recycle-filled regardless of the slippage mode): _last_delta
            # holds the previous aggregation's CLIENT motion per layer, i.e.
            # its output measured against the global the contributors
            # started from, re-applied on behalf of missing contributors.
            # Client motion, not the server's step: a stand-in for a missing
            # WORKER contribution must not carry the server optimizer's
            # acceleration, or momentum and fill compound (audit ML-02).
            # None until the first aggregation has rolled it, so a missing
            # contributor in round 0 is stale-filled instead.
            self._track_recycle: bool = (
                self._slippage_mode == "recycle" or self._skip_mode != "off"
            )
            self._last_delta: dict[str, np.ndarray] | None = None
            # Advice issued after round r's aggregation, consumed by round
            # r+1's aggregation to recognize advised omissions:
            # {source: set(layer names)}.  Replaced wholesale each round
            # (round-tagging on the wire is the engine's job).
            self._advice_outstanding: dict[str, set[str]] = {}
        else:
            # Worker stores the single aggregated model received from
            # the aggregator.  None means "not yet received this round".
            self._aggregated_model: TrainingUpdate | None = None

    @property
    def is_synchronous(self) -> bool:
        return True

    @property
    def is_centralized(self) -> bool:
        return True

    @property
    def role(self) -> str:
        """The role of this node: ``"aggregator"`` or ``"worker"``."""
        return self._role

    @property
    def slippage_mode(self) -> str:
        """Normalized slippage mode: ``"drop"``, ``"renorm"`` or ``"recycle"``."""
        return self._slippage_mode

    @property
    def skip_feedback_mode(self) -> str:
        """Skip-advice mode: ``"off"``, ``"shed"`` or ``"fedluar"``."""
        return self._skip_mode

    async def on_local_training_complete(
        self,
        model_params: dict[str, np.ndarray],
        round_num: int,
        num_samples: int,
    ) -> list[tuple[str, TrainingUpdate]]:
        """Send model parameters to neighbors.

        For **workers**: sends the locally-trained model to all neighbors
        (which in a star topology is just the aggregator).

        For the **aggregator**: sends the globally-averaged model to all
        workers.  This is called by ``run_aggregator()`` *after*
        aggregation — the ``model_params`` argument is the freshly
        computed weighted average.
        """
        update = TrainingUpdate(
            source_node=self.node_id,
            round_num=round_num,
            parameters=model_params,
            num_samples=num_samples,
        )
        destinations = [(neighbor, update) for neighbor in self.neighbors]

        if self._role == "aggregator":
            logger.debug(
                f"[{self.node_id}] FedAvg aggregator: broadcasting "
                f"averaged model to {len(destinations)} workers "
                f"(round {round_num})"
            )
        else:
            logger.debug(
                f"[{self.node_id}] FedAvg worker: sending model to "
                f"{len(destinations)} neighbor(s) (round {round_num})"
            )

        return destinations

    async def on_update_received(self, update: TrainingUpdate) -> None:
        """Handle an incoming update.

        For **workers**: stores the aggregated model from the aggregator.
        Rejects updates from non-neighbors and wrong rounds.

        For the **aggregator**: buffers a worker's model update.  Rejects
        non-neighbor senders and wrong-round updates.  Signals readiness
        when all workers have reported.
        """
        if update.source_node not in self.neighbors:
            logger.warning(
                f"[{self.node_id}] Received update from non-neighbor "
                f"{update.source_node}, ignoring"
            )
            return

        if update.round_num != self._round:
            logger.warning(
                f"[{self.node_id}] Received update for round "
                f"{update.round_num} but current round is {self._round}, "
                f"ignoring"
            )
            return

        if self._role == "aggregator":
            self._worker_buffer[update.source_node] = update
            logger.debug(
                f"[{self.node_id}] FedAvg aggregator: buffered update "
                f"from {update.source_node} (round {update.round_num}). "
                f"Buffer: {len(self._worker_buffer)}/{len(self.neighbors)}"
            )
        else:
            self._aggregated_model = update
            logger.debug(
                f"[{self.node_id}] FedAvg worker: received aggregated "
                f"model from {update.source_node} (round {update.round_num})"
            )

        if self.ready_to_aggregate():
            self._aggregation_event.set()

    def ready_to_aggregate(self) -> bool:
        """Check whether this node is ready to aggregate.

        For the **aggregator**: ready when all workers (= all neighbors)
        have submitted their updates for the current round.

        For **workers**: ready when the aggregated model has been received
        from the aggregator.
        """
        if self._role == "aggregator":
            return all(n in self._worker_buffer for n in self.neighbors)
        else:
            return self._aggregated_model is not None

    async def aggregate(
        self, local_params: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Perform role-specific aggregation.

        For the **aggregator**: computes a sample-weighted average of all
        buffered worker models.  The ``local_params`` argument is ignored
        because the aggregator does not train locally — its model is set
        to the weighted average of the workers.

        For **workers**: returns the aggregated model received from the
        aggregator.  The ``local_params`` argument is ignored — standard
        FedAvg replaces the worker model entirely with the global average.

        In both cases the internal buffer is cleared after aggregation.
        """
        if self._role == "aggregator":
            return self._aggregate_as_aggregator(local_params)
        else:
            return self._aggregate_as_worker()

    def _aggregate_as_aggregator(
        self, local_params: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Weighted average of all worker updates, slippage-mode aware.

        Each worker's contribution is weighted by its number of training
        samples:  ``new_params[layer] = Σ (n_k / N) * params_k[layer]``

        **Canonical accumulation order.**  Updates are accumulated sorted
        by ``source_node`` (and layers processed sorted by name), never in
        network-arrival order.  Floating-point addition is not
        associative, so arrival-order accumulation would make two runs
        that received identical values differ in the last bits — silently
        breaking the ε=0 bug-detector ("accuracy divergence at ε=0 means
        the implementation is broken").  With the canonical order the
        detector's claim holds bitwise, not just in exact arithmetic
        (writeup/01-candidate-selection.md §3, fix 5).

        **Partial-update handling (slippage).**  Under ε-deadline
        (extension 02) a worker may submit an update missing some layers
        (those that slipped past the trigger).  The compensation for a
        missing (layer, source) contribution is the configured slippage
        mode (gate ruling G3):

        - ``drop`` (control): the missing contribution is stale-filled
          with the aggregator's current value,
          ``weight * local_params[layer]``, preserving total weight = 1 —
          the pre-gate behaviour, kept as the control arm.
        - ``renorm``: the layer is averaged over the *arrived*
          contributors only, with weights renormalized to sum to 1
          (``w_k / W_arr``).  No stale-fill term, so the aggregate moves
          at full step length instead of being pulled back toward the
          previous global (isolates the verified drop bias).
        - ``recycle``: the missing contributor is assumed to have moved
          with the crowd — its contribution is
          ``local_params[layer] + last_delta[layer]``, re-applying the
          layer's previous *aggregated* motion on the missing sender's
          behalf (FedLUAR's recycling semantics, Eq. 3–5, adapted to
          parameter-space FedAvg).  On the first aggregation there is no
          delta history yet, so it falls back to stale-fill.

        A layer that *no* worker contributed keeps its current aggregator
        value exactly, in every mode, with a loud log — UNLESS every
        missing contributor was advised to skip it (skip-feedback v2), in
        which case the layer is recycle-filled: ``θ + last_delta`` with
        full weight, FedLUAR's reuse of the stored aggregated update.  The
        distinction matters: an *unexpected* zero-arrived layer is a
        slippage anomaly where blind recycling would re-apply the same
        delta on zero fresh evidence (the ratchet failure mode the gate
        flagged, writeup/01-candidate-selection.md §4), while an *advised*
        zero-arrived layer is the mechanism working as designed — the
        manifest declared the mass covered-by-recycling, and the advice
        rotation (fedluar sampling / shed re-evaluation) plus the aging cap
        bound how long any layer stays in that state.

        **Skip-feedback fills.**  A missing (layer, source) contribution
        that was advised away by the previous round's skip advice is
        recycle-filled regardless of the slippage mode (the manifest
        declared it covered-by-recycling; ``drop`` would quietly turn the
        declaration into a lie).  Non-advised missing contributions follow
        the configured slippage mode unchanged.

        **Server optimizer vs. fills (audit ML-02).**  When
        ``server_momentum > 0`` the velocity accumulates only the FRESH
        component of each layer's motion; imputed mass (recycled Δ_prev, or
        a bitwise-retained zero-arrival layer) passes through unaccelerated
        and the recycle state is rolled from the pre-momentum aggregate.
        Without that split, momentum and the fill feed each other with a
        per-round gain of ``beta + f`` and the shed layers blow up
        exponentially.  Everything here is a no-op at ``beta = 0``.

        Side effects: refreshes ``last_aggregation_telemetry``,
        ``inclusion_counts``, ``aggregation_count`` and (skip-feedback
        arms) ``last_skip_advice`` so the engine can report realized
        per-(layer, source) inclusion and realized-κ inputs and piggyback
        the next round's advice on the broadcast.
        """
        if not self._worker_buffer:
            raise RuntimeError(
                f"[{self.node_id}] aggregate() called on aggregator "
                f"with empty worker buffer"
            )

        # Canonical order: sorted by source id.  A dict preserves
        # insertion (= arrival) order, which would tie the fp rounding
        # pattern to network timing.
        updates = [self._worker_buffer[src] for src in sorted(self._worker_buffer)]
        total_samples = sum(u.num_samples for u in updates)

        if total_samples == 0:
            # Fallback to uniform average if all num_samples are 0.
            total_samples = len(updates)
            sample_counts = [1] * len(updates)
        else:
            sample_counts = [u.num_samples for u in updates]

        # Union of layer names across local_params and every worker.  Using
        # the union (rather than updates[0]'s keys) is correct under
        # partial updates: different workers may have different subsets,
        # and local_params provides the canonical full set.
        all_layers: set[str] = set(local_params.keys())
        for u in updates:
            all_layers.update(u.parameters.keys())

        # Skip advice issued after the PREVIOUS aggregation: missing
        # contributions matching it are the advised omissions this round.
        advice_consumed: dict[str, set[str]] = (
            self._advice_outstanding if self._skip_mode != "off" else {}
        )

        aggregated: dict[str, np.ndarray] = {}
        telemetry: dict[str, dict[str, Any]] = {}
        zero_arrived_layers: list[str] = []
        recycle_events: dict[str, int] = {}  # layer -> #recycled contributors
        partial_layer_count = 0
        # Slippage-fill bookkeeping for the server optimizer (audit ML-02).
        # fill_deltas[ℓ] is the FABRICATED part of layer ℓ's motion — the
        # recycled Δ_prev re-applied on behalf of contributors that did not
        # arrive, weighted by their sample mass.  no_fresh_layers are the
        # layers with zero arrived contributors, i.e. no fresh evidence at
        # all.  Both cross the optimizer boundary as data: the momentum step
        # below cannot otherwise distinguish motion supported by gradients
        # from motion invented to cover missing ones, and accelerates both.
        fill_deltas: dict[str, np.ndarray] = {}
        no_fresh_layers: set[str] = set()

        # Layers are processed in sorted order.  Per-layer values do not
        # depend on it (layers are independent sums), but it makes the
        # output dict ordering — and therefore downstream broadcast
        # serialization order — stable across runs (set iteration order
        # varies with PYTHONHASHSEED).
        for layer_name in sorted(all_layers):
            # Canonical shape: prefer local_params, else any worker's value.
            canonical = local_params.get(layer_name)
            if canonical is None:
                for u in updates:
                    if layer_name in u.parameters:
                        canonical = u.parameters[layer_name]
                        break
            if canonical is None:
                continue  # unreachable; defensive

            fallback = local_params.get(layer_name)
            arrived = [
                (u, n_k)
                for u, n_k in zip(updates, sample_counts)
                if layer_name in u.parameters
            ]
            arrived_sources = [u.source_node for u, _ in arrived]

            # Inclusion bookkeeping.  Absent sources get an explicit 0 so
            # the engine can compute inclusion rates without separately
            # knowing the worker set.
            layer_counts = self.inclusion_counts.setdefault(layer_name, {})
            for u in updates:
                layer_counts.setdefault(u.source_node, 0)
            for src in arrived_sources:
                layer_counts[src] += 1

            if len(arrived) == len(updates):
                # Complete layer: plain weighted average (legacy path).
                weighted_sum = np.zeros_like(canonical)
                for u, n_k in arrived:
                    weighted_sum += (n_k / total_samples) * u.parameters[layer_name]
                aggregated[layer_name] = weighted_sum
                telemetry[layer_name] = {
                    "arrived_sources": arrived_sources,
                    "filled": "none",
                }
                continue

            partial_layer_count += 1

            # Per-source advised flags for this layer (skip-feedback v2).
            advised_missing = {
                u.source_node
                for u in updates
                if layer_name not in u.parameters
                and layer_name in advice_consumed.get(u.source_node, ())
            }

            # The recycled stand-in value: the layer's previous aggregated
            # motion re-applied to the current global.  Available from the
            # second aggregation onward whenever recycle state is tracked
            # (recycle slippage or skip feedback); the same per-layer value
            # serves every missing contributor, so hoist it.
            recycled_value: np.ndarray | None = None
            recycled_delta: np.ndarray | None = None
            if (
                self._track_recycle
                and self._last_delta is not None
                and fallback is not None
            ):
                last_delta = self._last_delta.get(layer_name)
                if last_delta is not None:
                    recycled_delta = last_delta
                    recycled_value = fallback + last_delta

            if not arrived:
                # Zero fresh evidence either way: the server optimizer must
                # leave this layer alone (audit ML-02 — momentum used to
                # move a layer the log reports as retained bitwise).
                no_fresh_layers.add(layer_name)
                if (
                    advised_missing == {u.source_node for u in updates}
                    and recycled_value is not None
                ):
                    # Every contributor was advised to skip this layer
                    # (fedluar mode advises the same set to all senders, so
                    # this is the EXPECTED case there): recycle-fill at full
                    # weight — Σ w_k · (θ + Δ_prev) = θ + Δ_prev.  This is
                    # FedLUAR's reuse of the stored aggregated update; the
                    # manifest declared exactly this mass covered.
                    aggregated[layer_name] = recycled_value
                    recycle_events[layer_name] = len(updates)
                    telemetry[layer_name] = {
                        "arrived_sources": [],
                        "filled": "recycle",
                    }
                    continue
                # Degenerate shed: nobody contributed this layer and not
                # everyone was advised away (or no delta history yet).
                # Retain the current global bitwise in every mode (see
                # docstring).  ``fallback`` cannot be None here: a layer
                # absent from every update can only have entered all_layers
                # via local_params.
                aggregated[layer_name] = fallback.copy()
                zero_arrived_layers.append(layer_name)
                telemetry[layer_name] = {
                    "arrived_sources": [],
                    "filled": "stale",
                }
                continue

            if self._slippage_mode == "renorm":
                # Arrived-mass renormalization: weights w_k / W_arr over
                # arrived contributors only; no stale-fill term.
                arrived_samples = sum(n_k for _, n_k in arrived)
                weighted_sum = np.zeros_like(canonical)
                if arrived_samples > 0:
                    for u, n_k in arrived:
                        weighted_sum += (
                            n_k / arrived_samples
                        ) * u.parameters[layer_name]
                else:
                    # The arrived contributors all carry zero sample
                    # weight, making w_k / W_arr a 0/0.  Uniform over the
                    # arrived set is the natural limit (mirrors the
                    # all-zero total_samples fallback above).
                    for u, _ in arrived:
                        weighted_sum += u.parameters[layer_name] / len(arrived)
                aggregated[layer_name] = weighted_sum
                telemetry[layer_name] = {
                    "arrived_sources": arrived_sources,
                    "filled": "renorm",
                }
                continue

            # 'drop' and 'recycle': per-contributor fill, preserving
            # Σ weight = 1.  Recycle applies to a missing contribution when
            # the slippage mode is 'recycle' (every miss) OR the specific
            # (layer, source) was advised away by skip feedback — advised
            # omissions are recycle-covered even under 'drop' (module
            # docstring).
            weighted_sum = np.zeros_like(canonical)
            recycled_count = 0
            recycled_weight = 0.0
            for u, n_k in zip(updates, sample_counts):
                weight = n_k / total_samples
                if layer_name in u.parameters:
                    weighted_sum += weight * u.parameters[layer_name]
                elif recycled_value is not None and (
                    self._slippage_mode == "recycle"
                    or u.source_node in advised_missing
                ):
                    weighted_sum += weight * recycled_value
                    recycled_count += 1
                    recycled_weight += weight
                elif fallback is not None:
                    weighted_sum += weight * fallback
                else:
                    # No fallback (layer not in local_params either).  Skip;
                    # the weight is effectively re-distributed to zero.
                    pass
            aggregated[layer_name] = weighted_sum
            if recycled_count > 0:
                recycle_events[layer_name] = recycled_count
                # The fabricated share of this layer's motion: stale fills
                # contribute weight*θ, i.e. no motion at all, so only the
                # recycled mass shows up here (audit ML-02).
                if recycled_delta is not None:
                    fill_deltas[layer_name] = recycled_weight * recycled_delta
            telemetry[layer_name] = {
                "arrived_sources": arrived_sources,
                "filled": "recycle" if recycled_count > 0 else "stale",
            }

        # ---- FedAvgM server momentum (Hsu et al. 2019) ----
        # v_t = beta*v_{t-1} + Δ_fresh,t ;  theta_t = theta_{t-1} + Δ_fill,t
        # + v_t, where the round's aggregate motion is decomposed as
        #
        #     Δ_t = theta_agg − theta_{t−1} = Δ_fresh,t + Δ_fill,t
        #     Δ_fill,t = f_t · Δ_prev      (recycled mass; 0 for stale fills)
        #     Δ_fresh,t = Σ_arrived w_k (theta_k − theta_{t−1})
        #
        # AUDIT ML-02 — why the decomposition, not the plain Δ_t.  Applying
        # momentum to the slippage-FILLED aggregate and then rolling the
        # recycle state from the POST-momentum result closes a loop:
        # velocity re-enters next round's fill and the fill re-enters
        # velocity, giving v_t = (beta + f_t)·v_{t−1} + M_t — unstable for
        # f_t >= 1 − beta, i.e. for a single missing contributor of three at
        # beta = 0.9.  Replayed against the real per-round arrival sets it
        # inflated a shed 147 kB kernel 737x in 20 rounds while an
        # always-complete layer inflated 6x: a ~122x distortion of the
        # layer-wise update geometry, growing exponentially with the
        # horizon.  Two changes break both halves of the loop:
        #
        # (1) only the FRESH component enters the velocity, so the per-round
        #     gain is beta and nothing else; the fabricated mass passes
        #     straight through, applied exactly once.  A layer with NO
        #     arrived contributor is skipped entirely — its velocity is
        #     neither updated nor applied, so a layer the log reports as
        #     "kept their current global value" is no longer moved by
        #     beta·v (bounded by beta/(1−beta) = 9x the last delta before).
        # (2) the recycle state is rolled from the PRE-momentum aggregate
        #     (below), so the stand-in for a missing CLIENT contribution is
        #     client-side motion, not the server's accelerated step.
        #
        # Both are no-ops at beta = 0 (this block does not run and
        # pre_momentum is the aggregate itself), so every plain-FedAvg
        # campaign stays bit-identical and needs no compatibility knob.
        # Layers with no previous global value (defensively) pass through.
        pre_momentum = aggregated
        if self._server_momentum > 0.0:
            pre_momentum = dict(aggregated)
            for layer_name in sorted(aggregated):
                base = local_params.get(layer_name)
                if base is None or layer_name in no_fresh_layers:
                    continue
                fill = fill_deltas.get(layer_name)
                delta = aggregated[layer_name] - base
                fresh = delta if fill is None else delta - fill
                velocity = self._server_velocity.get(layer_name)
                velocity = (
                    fresh
                    if velocity is None
                    else self._server_momentum * velocity + fresh
                )
                self._server_velocity[layer_name] = velocity
                aggregated[layer_name] = (
                    base + velocity if fill is None
                    else base + fill + velocity
                )

        # Roll the recycle state: this round's aggregated CLIENT motion —
        # the pre-momentum aggregate measured against the global the
        # contributors trained from — becomes next round's recyclable
        # delta.  Note the fixed point this creates for advised
        # zero-arrived layers: a recycle-filled layer moved by exactly
        # Δ_prev, so its next delta is again Δ_prev — FedLUAR's
        # stored-update reuse.  The delta refreshes the moment fresh
        # contributions arrive (advice rotation / aging cap).  With
        # beta = 0, and with the aggregator's model always set to the
        # previous aggregate between rounds, this is the pre-audit
        # θ_new − θ_prev exactly.
        if self._track_recycle:
            self._last_delta = {
                name: pre_momentum[name] - local_params[name]
                for name in pre_momentum
                if name in local_params
            }

        num_workers = len(updates)
        self._worker_buffer.clear()
        self.last_aggregation_telemetry = telemetry
        self.aggregation_count += 1

        # ---- Skip advice for the NEXT round (skip-feedback v2) ----
        # Computed AFTER the recycle roll so fedluar ratios use this
        # round's realized motion.  The engine piggybacks the result on the
        # broadcast (round-tagged); we remember it to recognize advised
        # omissions at the next aggregation.
        if self._skip_mode != "off":
            self.last_skip_advice = self._compute_skip_advice(
                aggregated, telemetry,
            )
            self._advice_outstanding = {
                src: set(layers)
                for src, layers in self.last_skip_advice.items()
            }
            if self.last_skip_advice:
                logger.info(
                    f"[{self.node_id}] FedAvg skip advice "
                    f"({self._skip_mode}) for next round: "
                    f"{ {s: len(l) for s, l in self.last_skip_advice.items()} }"
                )
        else:
            self.last_skip_advice = {}

        if zero_arrived_layers:
            logger.warning(
                f"[{self.node_id}] FedAvg aggregation round {self._round}: "
                f"{len(zero_arrived_layers)} layer(s) had ZERO arrived "
                f"contributors and kept their current global value: "
                f"{zero_arrived_layers}"
            )
        if recycle_events:
            logger.info(
                f"[{self.node_id}] FedAvg aggregation round {self._round}: "
                f"recycled last-delta for missing contributors "
                f"(layer -> count): {recycle_events}"
            )
        if partial_layer_count > 0:
            logger.info(
                f"[{self.node_id}] FedAvg aggregation: weighted average of "
                f"{num_workers} workers ({total_samples} total samples, "
                f"round {self._round}); {partial_layer_count} layer(s) "
                f"had at least one missing contributor "
                f"(slippage mode: {self._slippage_mode})"
            )
        else:
            logger.info(
                f"[{self.node_id}] FedAvg aggregation: weighted average of "
                f"{num_workers} workers ({total_samples} total samples, "
                f"round {self._round})"
            )
        return aggregated

    def _compute_skip_advice(
        self,
        aggregated: dict[str, np.ndarray],
        telemetry: dict[str, dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Per-sender skip advice for the next round (skip-feedback v2).

        ``shed`` mode: each sender is advised the layers its update was
        missing this round — they were filled at aggregation, so the
        freshest information the aggregator can piggyback is "your slipped
        layers are recycle-covered; spend next round's bytes elsewhere".
        Per-sender by construction (different senders shed differently).

        ``fedluar_random`` mode: the metric ablation (FedLUAR Table 4
        "Random").  Identical to ``fedluar`` in count, in the one-global-set
        broadcast, in the recycle fill and in leaving staleness unbounded —
        the layers are simply drawn UNIFORMLY without replacement instead of
        inverse-ratio.  Because the draw ignores byte size, its expected
        uplink share is exactly ``(L - delta) / L`` of the model, which is
        what makes it byte-matched in expectation to the cyclic rotation
        control at ``cyclic_k = L - delta``.  Any accuracy gap between
        ``fedluar`` and ``fedluar_random`` at the same delta is attributable
        to the ``‖Δ_ℓ‖/‖θ_ℓ‖`` metric and to nothing else.

        ``fedluar`` mode: FedLUAR's server-side selection, advised
        identically to every sender.  Exactly ``skip_fedluar_count`` layers
        (clamped to L−1 so at least one layer always transmits) are sampled
        WITHOUT replacement with probability proportional to the inverse of
        the layer's update-to-weight ratio ``‖Δ_ℓ‖₂ / ‖θ_ℓ‖₂`` computed
        from this round's realized aggregated motion (``_last_delta``) —
        low-ratio (slow-moving) layers are the cheapest to recycle.  All
        ratios at the floor (e.g. round 0 of a cold start where no motion
        is recorded yet) degrade to uniform sampling.

        Starvation note (plan T1 VALIDATE c): advice alone can starve a
        layer — ``shed`` re-advises whatever stays missing, and fedluar
        sampling has no per-layer guarantee.  The bound comes from aging:
        senders suppress the skip for any layer at ``age >= tau_max``
        (must-send override), which refreshes the recycle delta.  With
        aging off, starvation is possible BY DESIGN and documented in
        tests/test_skip_feedback.py.
        """
        if self._role != "aggregator" or self._skip_mode == "off":
            return {}
        workers = sorted(self.neighbors)

        if self._skip_mode == "shed":
            advice: dict[str, list[str]] = {}
            for src in workers:
                missing = sorted(
                    layer
                    for layer, info in telemetry.items()
                    if src not in info["arrived_sources"]
                )
                if missing:
                    advice[src] = missing
            return advice

        # fedluar / fedluar_random: fixed count, one global set, resampled
        # every round.  The ONLY difference is the sampling distribution.
        names = sorted(aggregated)
        k = min(self._skip_fedluar_count, len(names) - 1)
        if k < 1:
            return {}
        if self._skip_mode == "fedluar_cyclic":
            # The control FedLUAR never ran: a deterministic round-robin
            # skip set.  Same delta, same global set, same recycle fill and
            # the same expected bytes as fedluar_random — but staleness is
            # BOUNDED for free: over ceil(L / gcd(L, delta)) advice rounds
            # every layer is skipped exactly delta / L of the time and its
            # consecutive-skip run is bounded, with no tau_max machinery.
            # FedLUAR's Table 4 tests only i.i.d. uniform "Random", so this
            # row does not exist in the paper.
            index = self._skip_cycle_index
            self._skip_cycle_index += 1
            start = (index * k) % len(names)
            layers = sorted(
                names[(start + i) % len(names)] for i in range(k)
            )
            return {src: list(layers) for src in workers}
        if self._skip_mode == "fedluar_random":
            # FedLUAR Table 4 "Random": i.i.d. uniform resampling of delta
            # layers each round.  No importance signal is consulted at all —
            # this is the metric-ablation control, byte-matched IN
            # EXPECTATION to fedluar at the same delta and matched exactly in
            # count.  Staleness stays unbounded (same family as fedluar),
            # which is what separates it from the cyclic rotation control.
            probabilities = None
        else:
            last_delta = self._last_delta or {}
            ratios = np.empty(len(names), dtype=np.float64)
            for i, name in enumerate(names):
                delta = last_delta.get(name)
                delta_norm = (
                    float(np.linalg.norm(np.asarray(delta, dtype=np.float64)))
                    if delta is not None else 0.0
                )
                theta_norm = float(
                    np.linalg.norm(
                        np.asarray(aggregated[name], dtype=np.float64)
                    )
                )
                ratios[i] = delta_norm / max(theta_norm, _FEDLUAR_RATIO_FLOOR)
            weights = 1.0 / np.maximum(ratios, _FEDLUAR_RATIO_FLOOR)
            if not np.all(np.isfinite(weights)) or weights.sum() <= 0.0:
                weights = np.ones(len(names), dtype=np.float64)
            probabilities = weights / weights.sum()
        chosen = self._skip_rng.choice(
            len(names), size=k, replace=False, p=probabilities,
        )
        layers = sorted(names[int(i)] for i in chosen)
        return {src: list(layers) for src in workers}

    def _aggregate_as_worker(self) -> dict[str, np.ndarray]:
        """Return the aggregated model received from the aggregator.

        The worker simply adopts the global model — no local mixing.
        """
        if self._aggregated_model is None:
            raise RuntimeError(
                f"[{self.node_id}] aggregate() called on worker but "
                f"no aggregated model received"
            )

        result = self._aggregated_model.parameters
        self._aggregated_model = None

        logger.info(
            f"[{self.node_id}] FedAvg worker: adopted aggregated model "
            f"(round {self._round})"
        )
        return result
