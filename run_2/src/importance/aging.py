"""Starvation control for tail layers (gate doc §7: age-capped additive aging).

The ε-tail must not starve: a layer that keeps being shed accrues staleness
and its first-order regret compounds (design doc §2.3).  This module is the
pure scoring half of the mechanism; the **engine owns the age counters** and
calls :func:`apply_aging` once per round before invoking the assignment
strategy (interface doc §3.3 / §5.1 step 3).

``age_ℓ`` = rounds since layer ℓ's contribution last ENTERED THE AGGREGATE,
counted from the aggregator's per-(source, layer) inclusion acknowledgement
(audit TRIG-1/ML-01, engine ``aging_age_basis='inclusion'``).  It used to be
counted from the sender's own head placement, which is a scheduling decision
carrying no delivery guarantee: a layer could be head-placed — and its age
reset — every round while the receiver's ε-trigger closed the round before
it landed, every round.  Measured on campaigns/w3, that produced absence
runs of 12 (and 20 over 50 rounds) at ``tau_max = 3``, with 84/84 of the
spurious resets on the big conv kernels the cap exists to protect.  The cap
bounds nothing unless the counter is keyed to the receiver's event.

Why the boost is *density-space*, not score-space
-------------------------------------------------
The interface doc sketched ``ũ = u + λ·age``.  That additive score boost is
**not starvation-safe under density ordering**: the coverage scheduler ranks
by ``u/s``, so a score boost of ``λ·age`` raises a layer's density by
``λ·age/s`` — for a 147 KB kernel that is ~10⁴–10⁵× weaker than for a small
bias at equal age, i.e. exactly the byte-heavy layers the tail prefers to
shed would effectively never age out of it.  We therefore boost in density
space (the gate-doc form):

    density'_ℓ = u_ℓ/s_ℓ + λ · age_ℓ · (U_total/S_total)

equivalently, in score space (what this function returns, so score-ordered
strategies see a consistent object):

    ũ_ℓ = u_ℓ + λ · age_ℓ · (U_total/S_total) · s_ℓ

Every starved layer's density now rises at the same size-independent rate of
``λ`` average-density units per round, so any layer crosses any shed
boundary in bounded time.  The ``U_total/S_total`` factor (the round's mean
utility density) makes λ dimensionless and transferable across metrics whose
absolute scales differ by orders of magnitude.

The multiplicative form ``u·(1+λ·age)`` is **excluded** — triple-confirmed
in step 2 as not starvation-safe (``u = 0`` stays 0 forever).

The hard cap (the part the convergence sketch actually needs): layers with
``age ≥ τ_max > 0`` are returned as ``must_receive`` promotions; the engine
merges them into the manifest flags and the strategy's ``must_receive``
argument, forcing the layer into the head and blocking round completion
until it arrives.  Bounded staleness ``τ_max`` is what makes the
local-SGD-style bound go through (design doc §2.3) — and it is a real bound
only because the arming condition above now fires on realized inclusion and
the enforcement half blocks until the promoted layer actually arrives.  Both
halves are receiver-side facts; either one measured on a proxy voids the
guarantee.

G2: aging transforms **sched scores only**.  Trigger accounting
(``raw_score`` = delta-sq-norm) is frozen and never aged.
"""

from __future__ import annotations

import logging
import math
from typing import Mapping

import numpy as np

logger = logging.getLogger(__name__)

#: aging_mode literals accepted by :func:`apply_aging`
#: (must match ``src/config/schema.py:TrainingConfig.aging_mode``).
AGING_MODES = ("none", "additive_capped", "stochastic_tail")


def _tau_promotions(
    scores: Mapping[str, float],
    ages: Mapping[str, int],
    tau_max: int,
) -> set[str]:
    """Layers whose staleness has hit the hard cap (``τ_max = 0`` disables)."""
    if tau_max <= 0:
        return set()
    return {
        name for name, age in ages.items()
        if age >= tau_max and name in scores
    }


def apply_aging(
    scores: dict[str, float],
    ages: dict[str, int],
    mode: str,
    lam: float,
    tau_max: int,
    *,
    sizes: Mapping[str, int] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, float], set[str]]:
    """Apply starvation control to one round's scheduling scores.

    Args:
        scores: Layer name -> sched score for this round (G2: never the
            trigger scores).  Not mutated.
        ages: Layer name -> rounds since the layer's contribution last
            entered the aggregate (engine-maintained from the aggregator's
            inclusion acks — audit TRIG-1; missing layers are age 0).
        mode: One of :data:`AGING_MODES`.
            ``none``: aging disabled entirely — scores returned unchanged,
            no promotions (``tau_max`` is ignored).
            ``additive_capped``: density-space additive boost (module
            docstring) + τ_max promotion.
            ``stochastic_tail``: scores returned unchanged — the FedLUAR
            -style randomization is the *competing* starvation solution and
            lives inside the coverage strategy's tail sampling
            (``CoverageEFTStrategy(stochastic_tail=True)``), not in the
            scores; τ_max promotion is still honoured when configured
            (defence in depth; the run matrix keeps it 0 for this arm).
        lam: Additive density boost per round of staleness
            (``aging_lambda``); 0 disables the boost.
        tau_max: Hard staleness cap (``aging_tau_max``); a layer with
            ``age ≥ tau_max > 0`` is promoted into ``must_receive``.
            0 disables the cap.
        sizes: Layer name -> payload bytes; **required** for
            ``additive_capped`` with ``lam > 0`` (the density-space boost
            needs S_total and per-layer sizes), must cover every key of
            ``scores``.  Ignored by the other modes.  Keyword-only (as is
            ``rng``) so a caller following the interface doc's original
            five-positional sketch cannot silently bind an rng here.
        rng: Unused here; accepted so the engine can pass its seeded
            generator uniformly (the stochastic arm consumes its rng in the
            strategy constructor instead).

    Returns:
        ``(effective_scores, must_receive_additions)``.  The score dict is
        a fresh dict; the set contains the τ_max promotions the engine must
        union into the round's ``must_receive``.

    Raises:
        ValueError: on an unknown ``mode``, or when ``additive_capped``
            with ``lam > 0`` is missing ``sizes``.
    """
    if mode not in AGING_MODES:
        registered = ", ".join(AGING_MODES)
        raise ValueError(
            f"Unknown aging mode {mode!r}. Registered: {registered}"
        )

    if mode == "none":
        return dict(scores), set()

    promotions = _tau_promotions(scores, ages, tau_max)
    if promotions:
        logger.info(
            "Aging cap: promoting %d layer(s) to must_receive at "
            "tau_max=%d: %s", len(promotions), tau_max, sorted(promotions),
        )

    if mode == "stochastic_tail" or lam <= 0.0:
        # No score transformation: stochastic_tail randomizes in the
        # strategy; additive_capped with lam=0 is the pure-cap arm.
        return dict(scores), promotions

    # mode == "additive_capped" with lam > 0: density-space boost.
    if sizes is None:
        raise ValueError(
            "apply_aging(mode='additive_capped') with aging_lambda > 0 "
            "requires sizes (the density-space boost needs per-layer bytes)"
        )

    u_total = float(sum(scores.values()))
    s_total = float(sum(sizes[name] for name in scores))
    avg_density = u_total / s_total if s_total > 0 else 0.0
    if not math.isfinite(avg_density) or avg_density <= 0.0:
        # Degenerate round (all-zero scores or zero/absent bytes): there is
        # no meaningful density scale, so fall back to unit density rather
        # than silently disabling the boost.  The all-zero-score case also
        # trips the engine's degenerate-manifest guard (pre-run fix 6).
        logger.warning(
            "Aging boost: degenerate average density (U_total=%g, "
            "S_total=%g); falling back to 1.0", u_total, s_total,
        )
        avg_density = 1.0

    effective = {
        name: score + lam * ages.get(name, 0) * avg_density * sizes[name]
        for name, score in scores.items()
    }
    return effective, promotions
