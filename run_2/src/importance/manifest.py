"""Build and validate `RoundManifest` protobuf messages.

The manifest is the ImpRoute substrate: at the start of every outbound update
in a round, a node sends a `RoundManifest` listing every layer it will
transmit, its raw importance score, and an optional ``must_receive`` flag.

Skip-feedback v2 (writeup/04-phase1/plan.md T1, spec in
writeup/01-candidate-selection.md §4) extends the listing invariant: layers
the sender deliberately omits on aggregator skip advice are STILL listed,
flagged ``skipped``, with their full trigger-mass ``raw_score``.  The
receiver counts that mass as covered-by-recycling, so the ε denominator
never shrinks round-over-round (the v1 coverage-denominator ratchet).

See ``docs/extensions/01-importance-manifest.md`` for the conceptual contract
and ``docs/decisions/0003-manifest-without-normalisation.md`` for the
rationale behind sending raw (un-normalised) scores.
"""

from __future__ import annotations

import math
from typing import Callable, Collection, Mapping

from src.proto_gen import federation_pb2


MustReceivePredicate = Callable[[str, float], bool]
"""Predicate (layer_name, raw_score) -> bool deciding whether a layer is
critical enough to block round completion regardless of ε-coverage."""


def _never(layer_name: str, raw_score: float) -> bool:
    """Default ``must_receive`` predicate: never block."""
    return False


def build_manifest(
    round_num: int,
    source_node_id: str,
    importance_scores: Mapping[str, float],
    must_receive_predicate: MustReceivePredicate | None = None,
    sched_scores: Mapping[str, float] | None = None,
    skipped_layers: Collection[str] | None = None,
) -> federation_pb2.RoundManifest:
    """Construct a `RoundManifest` from a raw-score dict.

    The order of entries follows iteration order of ``importance_scores``.
    Callers that need deterministic ordering should pass an ordered mapping.

    Args:
        round_num: The training round this manifest belongs to.
        source_node_id: The sending node's identifier (e.g. "node-1").
        importance_scores: Mapping from layer name to raw importance score.
            Under gate ruling G2 this is the FROZEN ε-trigger accounting
            metric (delta-sq-norm for all arms except the raw-norm control);
            scores must be non-negative; this is enforced by
            :func:`validate_manifest`.  Under skip-feedback v2 this mapping
            covers the transmitted layers AND the skip-omitted ones (the
            denominator must not shrink — v1 ratchet).
        must_receive_predicate: Optional predicate deciding ``must_receive``.
            Defaults to never.
        sched_scores: Optional mapping from layer name to the *scheduling*
            score (gate ruling G2).  When provided, each entry's
            ``sched_score`` field is populated; layers missing from the
            mapping keep the proto3 default 0.0, which receivers read as
            "same as raw_score".  Pass *None* for single-metric arms where
            the scheduling score equals the trigger score — the field then
            stays 0.0 on the wire, keeping old bytes valid.
        skipped_layers: Layers listed but deliberately NOT transmitted this
            round (skip-feedback v2 sender omission).  Each such entry is
            flagged ``skipped``; its ``raw_score`` stays in the manifest
            total and the receiver counts it as covered-by-recycling.
            Must be a subset of ``importance_scores`` keys; a layer that is
            both skipped and must-receive is a contract violation (aging's
            τ_max promotion overrides skip upstream) and raises here.

    Returns:
        A populated `RoundManifest`. Not validated; call
        :func:`validate_manifest` if validation is required.

    Raises:
        ValueError: if a skipped layer is selected as must-receive by the
            predicate, or if ``skipped_layers`` is not a subset of
            ``importance_scores``.
    """
    if must_receive_predicate is None:
        must_receive_predicate = _never
    skipped = frozenset(skipped_layers or ())
    unknown = skipped - set(importance_scores)
    if unknown:
        raise ValueError(
            f"skipped_layers must be a subset of importance_scores; "
            f"unknown: {sorted(unknown)}"
        )

    manifest = federation_pb2.RoundManifest()
    manifest.round = round_num
    manifest.source_node_id = source_node_id
    for name, score in importance_scores.items():
        entry = manifest.entries.add()
        entry.layer_name = name
        entry.raw_score = float(score)
        entry.must_receive = bool(must_receive_predicate(name, float(score)))
        if name in skipped:
            if entry.must_receive:
                raise ValueError(
                    f"layer {name!r} is both skipped and must_receive — "
                    f"a skipped layer never arrives, so it can never "
                    f"satisfy a must_receive gate (aging overrides skip "
                    f"upstream: age >= tau_max means MUST SEND)"
                )
            entry.skipped = True
        if sched_scores is not None and name in sched_scores:
            entry.sched_score = float(sched_scores[name])
    return manifest


def validate_manifest(manifest: federation_pb2.RoundManifest) -> None:
    """Raise :class:`ValueError` if the manifest is malformed.

    Checks:
    - ``entries`` is non-empty.
    - Every ``raw_score`` is finite and non-negative.
    - No duplicate ``layer_name`` within ``entries``.
    - No entry is both ``skipped`` and ``must_receive`` (a skipped layer
      never arrives, so it could never satisfy the must-receive gate —
      the flow would deadlock until the watchdog).
    - At least one entry is NOT ``skipped`` (something must actually be on
      the wire, or the receiver could only ever complete via watchdog).

    Notes:
    - Does **not** check that scores sum to anything in particular. Raw
      scores are intentional (ADR 0003).
    - ``round`` and ``source_node_id`` are not validated here; they are
      checked at higher protocol layers where context is available.
    """
    if len(manifest.entries) == 0:
        raise ValueError("RoundManifest.entries must be non-empty")

    seen: set[str] = set()
    num_unskipped = 0
    for entry in manifest.entries:
        if not entry.layer_name:
            raise ValueError("ImportanceEntry.layer_name must be non-empty")
        if entry.layer_name in seen:
            raise ValueError(
                f"duplicate layer_name in manifest: {entry.layer_name!r}"
            )
        seen.add(entry.layer_name)
        # NaN is not >= 0; the comparison filters both negative and NaN scores.
        if not (entry.raw_score >= 0):
            raise ValueError(
                f"ImportanceEntry.raw_score must be non-negative; got "
                f"{entry.raw_score!r} for layer {entry.layer_name!r}"
            )
        # sched_score is advisory (never read by the trigger, G2) so its
        # sign is not constrained here, but NaN/inf always indicate a bug
        # in the scheduling metric and would poison downstream analysis.
        if not math.isfinite(entry.sched_score):
            raise ValueError(
                f"ImportanceEntry.sched_score must be finite; got "
                f"{entry.sched_score!r} for layer {entry.layer_name!r}"
            )
        if entry.skipped:
            if entry.must_receive:
                raise ValueError(
                    f"ImportanceEntry for layer {entry.layer_name!r} is both "
                    f"skipped and must_receive — contradictory: the layer "
                    f"will never arrive but blocks completion until it does"
                )
        else:
            num_unskipped += 1
    if num_unskipped == 0:
        raise ValueError(
            "RoundManifest must contain at least one non-skipped entry; an "
            "all-skipped manifest announces a flow that can never complete "
            "(senders must retain at least one transmitted layer)"
        )


def manifest_total(manifest: federation_pb2.RoundManifest) -> float:
    """Return the sum of all raw scores in the manifest.

    Used by the ε-trigger (extension 02) to compute the absolute completion
    threshold ``(1 − ε) · total``.  Includes ``skipped`` entries by design:
    the denominator covers the full per-round utility mass, transmitted or
    not (skip-feedback v2 — this is exactly what kills the v1 ratchet).
    """
    return float(sum(entry.raw_score for entry in manifest.entries))


def manifest_skipped_layers(
    manifest: federation_pb2.RoundManifest,
) -> set[str]:
    """Names of entries flagged ``skipped`` (skip-feedback v2 omissions).

    The receiver's ε-trigger credits these layers' ``raw_score`` as
    covered-by-recycling: they will never arrive on the wire, but the
    aggregator re-applies their previous aggregated delta, so their mass is
    accounted for without shrinking the denominator.
    """
    return {entry.layer_name for entry in manifest.entries if entry.skipped}
