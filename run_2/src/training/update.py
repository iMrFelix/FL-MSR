"""Training update serialization: numpy arrays <-> protobuf messages.

Handles both monolithic (ModelUpdate) and per-layer (LayerUpdate) modes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from src.algorithms.base import TrainingUpdate
from src.proto_gen import federation_pb2
from src.training.late_layer_policy import LateLayerPolicy

logger = logging.getLogger(__name__)


def serialize_model_update(
    update: TrainingUpdate,
    source_node: str,
    dest_node: str,
) -> federation_pb2.Envelope:
    """Serialize a TrainingUpdate into a monolithic ModelUpdate envelope."""
    # Concatenate all layer parameters into a single bytes buffer
    layer_metas = []
    buffers = []
    offset = 0

    for var_name, arr in update.parameters.items():
        arr_bytes = arr.tobytes()
        meta = federation_pb2.LayerMeta(
            name=var_name,
            shape=list(arr.shape),
            dtype=str(arr.dtype),
            offset=offset,
            size=len(arr_bytes),
        )
        layer_metas.append(meta)
        buffers.append(arr_bytes)
        offset += len(arr_bytes)

    all_bytes = b"".join(buffers)

    model_update = federation_pb2.ModelUpdate(
        parameters=all_bytes,
        num_samples=update.num_samples,
        round=update.round_num,
        layer_meta=layer_metas,
    )

    envelope = federation_pb2.Envelope(
        source_node=source_node,
        dest_node=dest_node,
        timestamp_ns=time.monotonic_ns(),
        model_update=model_update,
    )
    return envelope


def deserialize_model_update(envelope: federation_pb2.Envelope) -> TrainingUpdate:
    """Deserialize a ModelUpdate envelope back into a TrainingUpdate."""
    msg = envelope.model_update
    parameters: dict[str, np.ndarray] = {}

    for meta in msg.layer_meta:
        arr_bytes = msg.parameters[meta.offset : meta.offset + meta.size]
        arr = np.frombuffer(arr_bytes, dtype=np.dtype(meta.dtype)).copy()
        arr = arr.reshape(list(meta.shape))
        parameters[meta.name] = arr

    return TrainingUpdate(
        source_node=envelope.source_node,
        round_num=msg.round,
        parameters=parameters,
        num_samples=msg.num_samples,
    )


def serialize_round_manifest(
    manifest: federation_pb2.RoundManifest,
    source_node: str,
    dest_node: str,
) -> federation_pb2.Envelope:
    """Wrap a `RoundManifest` in an `Envelope` addressed to a single destination.

    The manifest itself is built upstream (in :mod:`src.importance.manifest`);
    this helper only handles the framing.  See
    ``docs/extensions/01-importance-manifest.md``.
    """
    return federation_pb2.Envelope(
        source_node=source_node,
        dest_node=dest_node,
        timestamp_ns=time.monotonic_ns(),
        round_manifest=manifest,
    )


def serialize_layer_updates(
    update: TrainingUpdate,
    source_node: str,
    dest_node: str,
    importance_scores: dict[str, float] | None = None,
    traffic_classes: dict[str, int] | None = None,
) -> list[tuple[federation_pb2.Envelope, int]]:
    """Serialize a TrainingUpdate into per-layer LayerUpdate envelopes.

    Returns list of (envelope, traffic_class) tuples.
    """
    layer_names = list(update.parameters.keys())
    total_layers = len(layer_names)
    results = []

    for idx, var_name in enumerate(layer_names):
        arr = update.parameters[var_name]
        importance = (importance_scores or {}).get(var_name, 1.0)
        tc = (traffic_classes or {}).get(var_name, 0)

        layer_update = federation_pb2.LayerUpdate(
            layer_name=var_name,
            parameters=arr.tobytes(),
            shape=list(arr.shape),
            dtype=str(arr.dtype),
            layer_index=idx,
            total_layers=total_layers,
            num_samples=update.num_samples,
            round=update.round_num,
            importance=importance,
            traffic_class=tc,
        )

        envelope = federation_pb2.Envelope(
            source_node=source_node,
            dest_node=dest_node,
            timestamp_ns=time.monotonic_ns(),
            layer_update=layer_update,
        )
        results.append((envelope, tc))

    return results


def deserialize_layer_update(
    envelope: federation_pb2.Envelope,
) -> tuple[str, int, int, int, np.ndarray, int]:
    """Deserialize a single LayerUpdate.

    Returns (source_node, round_num, layer_index, total_layers, array, num_samples).
    """
    msg = envelope.layer_update
    arr = np.frombuffer(msg.parameters, dtype=np.dtype(msg.dtype)).copy()
    arr = arr.reshape(list(msg.shape))
    return (
        envelope.source_node,
        msg.round,
        msg.layer_index,
        msg.total_layers,
        arr,
        msg.num_samples,
    )


@dataclass(frozen=True)
class BufferResult:
    """Outcome of a `LayerBuffer.add_layer` call that completed a round.

    Attributes:
        update: The reassembled `TrainingUpdate`.
        is_partial: True iff the ε-trigger fired before *all* layers arrived;
            i.e. some layers in the manifest are absent.
        missing_layers: Names of manifest entries that did **not** arrive
            before the trigger fired.  Empty when ``is_partial`` is False.
    """

    update: TrainingUpdate
    is_partial: bool
    missing_layers: frozenset[str] = field(default_factory=frozenset)


class LayerBuffer:
    """Reassembles per-layer updates into complete `TrainingUpdate`s.

    Buffers layers from a specific source node for a specific round.  A
    round can be *completed* in two ways:

    1. **"All layers in"** — every layer announced by the LayerUpdate
       ``total_layers`` count has arrived.  This is the legacy trigger and
       fires regardless of whether a `RoundManifest` was registered.
    2. **"ε-coverage"** — a `RoundManifest` has been registered for the
       ``(source, round)`` pair, the receiver has accumulated
       ``received_score ≥ (1 − ε) · total``, and every layer flagged
       ``must_receive`` in the manifest has arrived.  This trigger fires
       only when ``epsilon > 0``.

    See ``docs/extensions/02-epsilon-trigger.md`` for the full contract,
    including the ε-guarantee proof.

    After either trigger fires for a given ``(source, round)``, subsequent
    `add_layer` calls for the *same* pair return ``None`` and the layer is
    silently dropped.  (The late-layer policy in extension 03 will replace
    this drop with a configurable handler.)
    """

    def __init__(
        self,
        epsilon: float = 0.0,
        late_layer_policy: LateLayerPolicy | None = None,
    ):
        if not (0.0 <= epsilon < 1.0):
            raise ValueError(
                f"epsilon must be in [0, 1), got {epsilon!r}"
            )
        self.epsilon = epsilon
        # Policy invoked when a layer arrives for an already-fired
        # (source, round) pair.  None preserves the legacy silent-drop
        # behaviour for callers that don't care about late layers.  See
        # docs/extensions/03-late-layer-policy.md.
        self.late_layer_policy = late_layer_policy

        # (source_node, round_num) -> {layer_name: np.ndarray}
        self._buffers: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        # (source_node, round_num) -> (total_layers, num_samples)
        self._metadata: dict[tuple[str, int], tuple[int, int]] = {}
        # (source_node, round_num) -> RoundManifest
        self._manifests: dict[
            tuple[str, int], federation_pb2.RoundManifest
        ] = {}
        # (source_node, round_num) pairs that have already fired a result.
        self._fired: set[tuple[str, int]] = set()

    # ------------------------------------------------------------------
    # Manifest registration
    # ------------------------------------------------------------------

    def register_manifest(
        self, manifest: federation_pb2.RoundManifest
    ) -> None:
        """Register a `RoundManifest` for its ``(source, round)`` pair.

        Idempotent: a second manifest for the same pair overwrites the
        first and emits a warning (this should not happen in a well-behaved
        system).
        """
        key = (manifest.source_node_id, manifest.round)
        if key in self._manifests:
            logger.warning(
                f"Duplicate manifest for source={manifest.source_node_id!r} "
                f"round={manifest.round}; overwriting previous entry"
            )
        self._manifests[key] = manifest

    # ------------------------------------------------------------------
    # Layer arrival
    # ------------------------------------------------------------------

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
        """Add a layer; possibly return a `BufferResult` that completes the round.

        Return value:
            - ``None``: the round is not yet complete; keep waiting.
            - `BufferResult` with ``is_partial=False``: all layers in.
            - `BufferResult` with ``is_partial=True``: ε-trigger fired
              before all layers in; ``missing_layers`` lists the absentees.
        """
        key = (source_node, round_num)

        if key in self._fired:
            # This round already completed.  Hand the late arrival to the
            # policy (if any); the buffer itself does not retain it.  See
            # docs/extensions/03-late-layer-policy.md.
            if self.late_layer_policy is not None:
                try:
                    self.late_layer_policy.on_late_arrival(
                        source_node=source_node,
                        round_num=round_num,
                        layer_name=layer_name,
                        array=array,
                        num_samples=num_samples,
                    )
                except Exception as exc:  # noqa: BLE001 - keep robust on receive path
                    logger.warning(
                        "Late-layer policy raised on %s (round %d): %s",
                        layer_name, round_num, exc,
                    )
            return None

        # Accumulate
        self._buffers.setdefault(key, {})[layer_name] = array
        if key not in self._metadata:
            self._metadata[key] = (total_layers, num_samples)

        # Trigger 1: "all layers in" — count-based, manifest-agnostic.
        if len(self._buffers[key]) >= total_layers:
            return self._fire(key, is_partial=False, missing=frozenset())

        # Trigger 2: ε-coverage — manifest-driven.
        if self.epsilon > 0.0 and key in self._manifests:
            manifest = self._manifests[key]
            scores = {e.layer_name: e.raw_score for e in manifest.entries}
            must_receive = {e.layer_name for e in manifest.entries
                            if e.must_receive}
            total_score = sum(scores.values())
            target = (1.0 - self.epsilon) * total_score
            received = self._buffers[key].keys()
            received_score = sum(scores.get(name, 0.0) for name in received)
            must_done = must_receive.issubset(received)
            if must_done and received_score >= target:
                missing = frozenset(scores.keys()) - frozenset(received)
                return self._fire(key, is_partial=True, missing=missing)

        return None

    def _fire(
        self,
        key: tuple[str, int],
        *,
        is_partial: bool,
        missing: frozenset[str],
    ) -> BufferResult:
        """Pop the buffer for ``key`` and return a `BufferResult`."""
        source_node, round_num = key
        params = self._buffers.pop(key)
        total, samples = self._metadata.pop(key)
        # Keep the manifest around for diagnostics; clear_stale will reap it.
        self._fired.add(key)
        return BufferResult(
            update=TrainingUpdate(
                source_node=source_node,
                round_num=round_num,
                parameters=params,
                num_samples=samples,
            ),
            is_partial=is_partial,
            missing_layers=missing,
        )

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    def clear_stale(self, current_round: int, max_staleness: int = 2) -> None:
        """Remove buffers, manifests, and fired markers for stale rounds."""
        def is_stale(key: tuple[str, int]) -> bool:
            return key[1] < current_round - max_staleness

        for key in [k for k in self._buffers if is_stale(k)]:
            del self._buffers[key]
            self._metadata.pop(key, None)
        for key in [k for k in self._manifests if is_stale(k)]:
            del self._manifests[key]
        self._fired = {k for k in self._fired if not is_stale(k)}
