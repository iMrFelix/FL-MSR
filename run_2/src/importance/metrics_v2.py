"""Importance metrics v2: scores computed on the shipped multi-epoch delta.

Step-2 gate headline 3 (``writeup/01-candidate-selection.md`` §1.3, §7):
importance must be computed on the **shipped delta**
``Δ_ℓ = θ_after_training − θ_shipped_base`` — the object the network
actually transmits and the aggregator actually consumes — not on the last
minibatch gradient, which is a single-batch noise sample of it.  The engine
therefore passes the per-layer delta dict through the existing
``gradients`` parameter of :meth:`ImportanceMetric.compute_all`
(``docs/extensions/04-overnight-interfaces.md`` §3.1).

Score-role contract (gate ruling G2 — binding for every arm):

- **Trigger accounting is frozen on** :func:`delta_sq_norm`.  The receiver's
  ε-trigger reads ``ImportanceEntry.raw_score`` only, and ``raw_score`` is
  the delta-sq-norm for ALL arms, so "ε of importance mass" means the same
  thing in every run.
- Every other metric here is **sched-only**: it feeds
  ``ImportanceEntry.sched_score`` and the assignment strategy, influencing
  *which class / order / tail* a layer gets — never *when the round counts
  as complete*.

The module exposes two layers:

1. Pure functions (:func:`delta_sq_norm`, :func:`relative`,
   :func:`snr_reweight`, :func:`raw_norm`) mapping array dicts to score
   dicts — directly testable, no engine coupling.
2. Thin :class:`~src.importance.base.ImportanceMetric` subclasses plus a
   factory (:func:`make_metric_v2`) keyed by the ``importance_metric_v2``
   config literal, so the engine builds the frozen trigger metric and the
   configured sched metric without special-casing.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from src.importance.base import ImportanceMetric

#: Divide-by-zero guard used by the ratio metrics.  Kept module-level so
#: tests and documentation reference one authoritative value.
_RATIO_GUARD = 1e-12

#: Score used when no update array is available for a layer.  Matches the
#: legacy ``GradientNormImportance`` fallback so the control arm's behaviour
#: is preserved bit-for-bit when gradients are missing.
_MISSING_UPDATE_SCORE = 1.0


# ---------------------------------------------------------------------------
# Pure per-layer reductions
# ---------------------------------------------------------------------------

def _as_f64(arr: np.ndarray) -> np.ndarray:
    """Flatten and upcast to float64 before reducing.

    Model arrays are float32; accumulating squares in float32 loses
    precision on large layers (the 147 KB kernels), which would make
    trigger accounting subtly seed-dependent.  One upcast per layer per
    round is negligible next to training.
    """
    return np.asarray(arr, dtype=np.float64).ravel()


def _sq_norm(arr: np.ndarray) -> float:
    flat = _as_f64(arr)
    if flat.size == 0:
        return 0.0
    return float(flat @ flat)


def _l2_norm(arr: np.ndarray) -> float:
    flat = _as_f64(arr)
    if flat.size == 0:
        return 0.0
    return float(np.linalg.norm(flat))


def _snr(arr: np.ndarray) -> float:
    """mean(|x|) / (std(x) + guard) — the FedTLU-style signal-to-noise ratio.

    ``std`` is the population standard deviation (numpy default, ddof=0).
    A constant (zero-variance) delta has std 0, so the guard caps the ratio
    at mean/1e-12 rather than dividing by zero; that is the specified
    formula, and since the metric is sched-only an extreme value can only
    mis-schedule a layer, never corrupt round completion (G2).
    """
    flat = _as_f64(arr)
    if flat.size == 0:
        return 0.0
    return float(np.mean(np.abs(flat)) / (np.std(flat) + _RATIO_GUARD))


# ---------------------------------------------------------------------------
# Metric functions (dict -> dict)
# ---------------------------------------------------------------------------

def delta_sq_norm(deltas: Mapping[str, np.ndarray]) -> dict[str, float]:
    """``u_ℓ = ‖Δ_ℓ‖₂²`` — squared L2 norm of the shipped delta.

    Objective it derives from: the first-order loss regret of withholding
    layer ℓ from the aggregate.  With Δ ≈ −η·ḡ (the update is a scaled
    descent direction), omitting Δ_ℓ forgoes ≈ ⟨∇L, Δ_ℓ⟩ ≈ (1/η)·‖Δ_ℓ‖₂²
    of first-order loss decrease, so squared delta norm is the additive
    per-layer regret unit the ε-trigger should meter.

    Role (G2): **the frozen trigger-accounting metric for ALL arms**, and
    the default scheduling metric (``importance_metric_v2='delta_sq_norm'``).
    """
    return {name: _sq_norm(d) for name, d in deltas.items()}


def relative(
    deltas: Mapping[str, np.ndarray],
    params: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """``u_ℓ = ‖Δ_ℓ‖₂ / (‖θ_ℓ‖₂ + 1e-12)`` — update-to-weight ratio.

    FedLUAR Eq. 1.  Objective it derives from: scale-invariant layer drift —
    a layer matters this round in proportion to how much it moved *relative
    to its own magnitude*, which de-confounds raw norm from per-layer weight
    scale (batch-norm/bias layers live on very different scales than
    kernels).

    Role (G2): **sched-only**.  Feeds ``sched_score``/assignment; trigger
    accounting stays on :func:`delta_sq_norm`.

    Args:
        deltas: Layer name -> shipped delta array.
        params: Layer name -> current parameter array θ_ℓ; must cover every
            key of ``deltas``.
    """
    return {
        name: _l2_norm(d) / (_l2_norm(params[name]) + _RATIO_GUARD)
        for name, d in deltas.items()
    }


def snr_reweight(
    deltas: Mapping[str, np.ndarray],
    base: Mapping[str, float],
) -> dict[str, float]:
    """``u_ℓ = base_ℓ · mean(|Δ_ℓ|) / (std(Δ_ℓ) + 1e-12)`` — SNR reweighting.

    FedTLU-derived.  Objective it derives from: prefer *coherent* updates —
    a delta whose entries shift together (high mean-to-std ratio) is a
    consistent signal worth landing early, while a high-variance delta of
    the same energy is closer to gradient noise.  The base utility
    (normally :func:`delta_sq_norm` of the same deltas, see
    :class:`SnrReweightImportance`) is amplified or damped by that
    coherence factor.

    Role (G2): **sched-only**; never enters trigger accounting.

    Args:
        deltas: Layer name -> shipped delta array.
        base: Layer name -> base utility ``base_ℓ``; must cover every key
            of ``deltas``.
    """
    return {name: base[name] * _snr(d) for name, d in deltas.items()}


def raw_norm(gradients: Mapping[str, np.ndarray]) -> dict[str, float]:
    """``u_ℓ = ‖g_ℓ‖₂`` — plain L2 norm of the update array (control).

    The legacy metric, kept **as the degenerate control** (design doc E6):
    it conflates per-parameter signal with parameter count, so tiny
    high-norm biases outrank 147 KB kernels.  The reduction is exactly the
    historical ``GradientNormImportance`` semantic; only the input changed —
    it historically reduced the last-minibatch gradient, and post-gate the
    engine feeds the shipped delta through the same parameter (interface
    doc §3.1: ``raw_norm`` = ‖Δ‖₂ control).

    Role (G2): **sched-only** control arm; trigger accounting stays on
    :func:`delta_sq_norm`.
    """
    return {name: _l2_norm(g) for name, g in gradients.items()}


# ---------------------------------------------------------------------------
# ImportanceMetric adapters + factory
# ---------------------------------------------------------------------------
#
# The engine computes two score dicts per round through one code path
# (interface doc §3.1): trigger = make_metric_v2("delta_sq_norm"),
# sched = make_metric_v2(config["importance_metric_v2"]).  All adapters read
# the shipped delta from the ``layer_gradients``/``gradients`` parameter and
# fall back to _MISSING_UPDATE_SCORE when it is absent (legacy behaviour:
# uniform scores rather than a crash when a mode supplies no update arrays).

class DeltaSqNormImportance(ImportanceMetric):
    """Adapter for :func:`delta_sq_norm` (trigger + default sched)."""

    def compute(
        self,
        layer_name: str,
        layer_weights: np.ndarray,
        layer_gradients: np.ndarray | None,
        round_num: int,
        context: dict,
    ) -> float:
        if layer_gradients is None:
            return _MISSING_UPDATE_SCORE
        return _sq_norm(layer_gradients)


class RelativeImportance(ImportanceMetric):
    """Adapter for :func:`relative` (sched-only); θ_ℓ from ``layer_weights``."""

    def compute(
        self,
        layer_name: str,
        layer_weights: np.ndarray,
        layer_gradients: np.ndarray | None,
        round_num: int,
        context: dict,
    ) -> float:
        if layer_gradients is None:
            return _MISSING_UPDATE_SCORE
        return _l2_norm(layer_gradients) / (_l2_norm(layer_weights) + _RATIO_GUARD)


class SnrReweightImportance(ImportanceMetric):
    """Adapter for :func:`snr_reweight` (sched-only).

    The base utility is the layer's own delta-sq-norm, making the score
    ``‖Δ_ℓ‖₂² · SNR(Δ_ℓ)`` — the frozen regret unit reweighted by update
    coherence.  Both factors depend only on layer ℓ, so per-layer
    ``compute`` is exact (no cross-layer state).
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
            return _MISSING_UPDATE_SCORE
        return _sq_norm(layer_gradients) * _snr(layer_gradients)


class RawNormImportance(ImportanceMetric):
    """Adapter for :func:`raw_norm` (control).

    Byte-identical behaviour to the legacy ``GradientNormImportance``
    (including the missing-update fallback of 1.0); kept as a distinct
    class so the v2 factory covers every ``importance_metric_v2`` literal.
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
            return _MISSING_UPDATE_SCORE
        return _l2_norm(layer_gradients)


class UniformImportance(ImportanceMetric):
    """Importance-blind control (sched-only): every layer scores 1.0.

    Under coverage-EFT the density ordering degenerates to 1/bytes —
    smallest layers first, i.e. the naive "shed the big cheap layers"
    byte-greedy heuristic.  Trigger accounting stays frozen on
    delta-sq-norm (gate ruling G2), so this arm isolates the value of
    importance-aware *ordering* at matched coverage semantics: the
    CP2 Claim-A attribution control (writeup/15 §make-or-break item 1).
    """

    def compute(
        self,
        layer_name: str,
        layer_weights: np.ndarray,
        layer_gradients: np.ndarray | None,
        round_num: int,
        context: dict,
    ) -> float:
        return 1.0


_METRIC_V2_REGISTRY: dict[str, type[ImportanceMetric]] = {
    "delta_sq_norm": DeltaSqNormImportance,
    "relative": RelativeImportance,
    "snr_reweight": SnrReweightImportance,
    "raw_norm": RawNormImportance,
    "uniform": UniformImportance,
}


def registered_metrics_v2() -> list[str]:
    """Return the sorted ``importance_metric_v2`` config literals."""
    return sorted(_METRIC_V2_REGISTRY)


def make_metric_v2(name: str) -> ImportanceMetric:
    """Construct a v2 importance metric by ``importance_metric_v2`` literal.

    Args:
        name: One of ``delta_sq_norm``, ``relative``, ``snr_reweight``,
            ``raw_norm`` (must match ``src/config/schema.py``).

    Raises:
        ValueError: if ``name`` is not a registered v2 metric.
    """
    try:
        cls = _METRIC_V2_REGISTRY[name]
    except KeyError:
        registered = ", ".join(registered_metrics_v2())
        raise ValueError(
            f"Unknown importance metric v2 {name!r}. Registered: {registered}"
        ) from None
    return cls()
