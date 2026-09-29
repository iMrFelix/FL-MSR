"""Pre-registered confirmatory statistics for the Phase-2 campaign.

This is the **paired-test driver** named in the analysis-software list
(`writeup/04-phase1/protocol.md` §5.6), implementing exactly the
procedures frozen in §5.3 and the verdict ladder of §3.  It is committed
before Stage P3 launches (precondition P-2) so that no confirmatory test
is authored after seeing hypothesis data.

Design constraints honoured here:

- **numpy-only** core (the campaign `.venv` ships numpy, not scipy); a
  defensive `scipy` import is used *only* to cross-check the t-test CDF
  when scipy happens to be present, never as a hard dependency.  Every
  p-value the verdict logic consumes is computed from a self-contained
  numpy/stdlib implementation.
- All confirmatory inputs are **arrays of per-seed paired differences**
  `d_s = metric(treatment, s) - metric(control, s)` over the *common*
  seed set (P-4: no mixed seed compositions; the caller is responsible
  for handing in paired, equal-length arrays).
- n is small and fixed (5 confirmatory seeds), so the permutation and
  Spearman nulls are **exact**: all 2^n sign flips (32 for n=5) and all
  n! orderings (120 for n=5) are enumerated directly, never sampled.

Statistical procedures (§5.3), one public function each:

================  ===========================================================
§5.3 procedure    function
================  ===========================================================
paired one-sided  ``paired_t_one_sided``         (df = n-1)
sign-flip perm     ``sign_flip_permutation``      (exact, 2^n patterns)
sign test          ``sign_test``                  (binomial sign-consistency)
non-inferiority    ``noninferiority_ucb``         (one-sided 95 % t-UCB)
neutrality band    ``neutrality_band``            (H-P7a; no NHST)
Spearman + perm    ``spearman_exact``             (exact, n! orderings)
Holm               ``holm``                       (within a family)
verdict            ``directional_verdict`` /      (§3 ladder)
                   ``noninferiority_verdict``
================  ===========================================================

Library API: pure functions over numpy arrays / python lists, each
returning a small frozen dataclass so callers (and the campaign report)
get named fields rather than positional tuples.

CLI: given a table of per-(arm, seed) metric values (CSV long-format or
JSON), runs the H-P1..H-P7 directional / non-inferiority contrasts of
§3–§4.5 and prints a verdict table.  See ``--help`` and the module-level
``HYPOTHESES`` registry for the wired contrasts.

Self-test::

    python -m scripts.hypothesis_tests --selftest

reproduces the DeepCNN-C1 pattern (paired diffs
-2.7,-2.8,-2.0,-2.7,-1.8): an all-negative contrast whose sign test
returns p = 1/32, with the t-test, exact sign-flip test and sign test
agreeing in direction, and exercises the UCB and Spearman paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from itertools import permutations, product
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# ``scipy`` is optional and defensively imported: when present it is used
# only to *cross-check* the Student-t survival function (a belt-and-braces
# numerical agreement assertion in the self-test), never to compute a
# p-value the verdict logic depends on.  The campaign environment has no
# scipy; the module must import and run identically without it.
try:  # pragma: no cover - environment dependent
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover - the expected campaign case
    _scipy_stats = None

HAVE_SCIPY = _scipy_stats is not None

#: Pre-registered confirmatory seed set (§4.2). The common seed set size.
N_SEEDS = 5
#: Significance level for every confirmatory test (§3).
ALPHA = 0.05
#: Non-inferiority margin in percentage points (§3, §5.3; H-P2 / H-P6ii).
NONINF_MARGIN_PP = 1.0


# ===========================================================================
# Student-t distribution (numpy/stdlib only)
# ===========================================================================
#
# We need the upper-tail probability  P(T_df > t)  for the one-sided t-test
# and the upper 95 % critical value  t_{0.95, df}  for the non-inferiority
# UCB.  Both come from the regularised incomplete beta function, which we
# implement with the standard Lentz continued fraction (Numerical Recipes
# §6.4).  This keeps the module scipy-free while matching scipy to ~1e-12.


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz's algorithm)."""
    tiny = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1.0e-14:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """Upper-tail survival function P(T_df > t) for Student's t.

    Pure numpy/stdlib (incomplete beta); agrees with
    ``scipy.stats.t.sf`` to ~1e-12.  Handles the degenerate ``df<=0``
    case (returns NaN) and is symmetric about 0.
    """
    if not math.isfinite(t):
        return float("nan") if math.isnan(t) else (0.0 if t > 0 else 1.0)
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    half = 0.5 * _betai(0.5 * df, 0.5, x)
    return half if t > 0 else 1.0 - half


def student_t_ppf(p: float, df: float) -> float:
    """Inverse CDF (quantile) of Student's t at probability ``p``.

    Bisection on ``student_t_sf`` — only ever called for a fixed handful
    of critical values (the 95 % UCB), so the O(60-iteration) bisection
    is irrelevant to runtime and avoids importing a special-function
    library.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1); got {p}")
    if df <= 0:
        return float("nan")
    target_sf = 1.0 - p
    lo, hi = -1.0e6, 1.0e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_sf(mid, df) > target_sf:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1.0e-12:
            break
    return 0.5 * (lo + hi)


# ===========================================================================
# Small helpers
# ===========================================================================


def _as_diffs(diffs: Sequence[float] | np.ndarray) -> np.ndarray:
    """Coerce to a 1-D float array, dropping nothing (NaN ⇒ error)."""
    arr = np.asarray(diffs, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("empty difference array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "non-finite paired difference(s); the caller must drop or "
            "impute censored seeds before testing (§5.3 censoring rules)"
        )
    return arr


def paired_diffs(
    treatment: Sequence[float], control: Sequence[float]
) -> np.ndarray:
    """Seed-paired differences ``treatment[s] - control[s]``.

    Both sequences must be the *same length* and aligned by seed (P-4
    seed-composition rule: the caller hands in the common seed set in a
    fixed order).
    """
    t = np.asarray(treatment, dtype=float).ravel()
    c = np.asarray(control, dtype=float).ravel()
    if t.shape != c.shape:
        raise ValueError(
            f"unequal/length-mismatched arms: {t.shape} vs {c.shape}; "
            "confirmatory contrasts require the common seed set (P-4)"
        )
    return t - c


def dominant_seed(diffs: Sequence[float] | np.ndarray) -> int:
    """Index of a single dominating seed, or -1 if none dominates.

    §5.3: report Wilcoxon "if a single seed dominates (|d_s| > 3× median
    |d|)".  Returns the offending index so the caller can flag the
    contrast; -1 when no seed exceeds the 3× band.
    """
    arr = _as_diffs(diffs)
    mag = np.abs(arr)
    med = float(np.median(mag))
    if med <= 0.0:
        return -1
    over = np.flatnonzero(mag > 3.0 * med)
    return int(over[0]) if over.size else -1


# ===========================================================================
# 1. Paired one-sided t-test  (§5.3 directional)
# ===========================================================================


@dataclass(frozen=True)
class TTestResult:
    """One-sided paired t-test across seeds (df = n-1)."""

    mean: float
    sd: float
    se: float
    t: float
    df: int
    p: float
    n: int
    direction: str  # "less" or "greater"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean, "sd": self.sd, "se": self.se, "t": self.t,
            "df": self.df, "p": self.p, "n": self.n,
            "direction": self.direction,
        }


def paired_t_one_sided(
    diffs: Sequence[float] | np.ndarray,
    direction: str = "greater",
) -> TTestResult:
    """One-sided paired t-test on per-seed differences (df = n-1).

    This is the one-sample t on the paired-difference vector — identical
    to a paired t-test (§5.3 "one-sided paired t").

    ``direction='greater'`` tests H1: mean(d) > 0 (the treatment exceeds
    the control, e.g. ΔACC(recycle-drop) > 0 for H-P3).
    ``direction='less'`` tests H1: mean(d) < 0 (e.g. ΔACC(4) < ΔACC(7)
    for H-P4 when ``diffs`` is the cost-difference ACC(4)-ACC(7), framed
    so "our cost is smaller" ⇒ negative).

    Uses the sample SD with ddof=1 (Bessel) so df = n-1, the
    pre-registered df=4 at n=5.
    """
    if direction not in ("greater", "less"):
        raise ValueError("direction must be 'greater' or 'less'")
    arr = _as_diffs(diffs)
    n = arr.size
    if n < 2:
        raise ValueError("need >= 2 seeds for a paired t-test")
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    df = n - 1
    if sd == 0.0:
        # Degenerate: zero within-seed spread.  A nonzero mean in the
        # hypothesised direction is infinitely significant (p→0); a mean
        # of exactly 0, or one against the direction, is p=1.
        if mean == 0.0:
            t, p = 0.0, 1.0
        elif (direction == "greater" and mean > 0) or (
            direction == "less" and mean < 0
        ):
            t = math.inf if mean > 0 else -math.inf
            p = 0.0
        else:
            t = math.inf if mean > 0 else -math.inf
            p = 1.0
        return TTestResult(mean, sd, 0.0, t, df, p, n, direction)
    se = sd / math.sqrt(n)
    t = mean / se
    # Upper tail for 'greater', lower tail for 'less'.
    p = student_t_sf(t, df) if direction == "greater" else student_t_sf(-t, df)
    return TTestResult(mean, sd, se, float(t), df, float(p), n, direction)


# ===========================================================================
# 2. Exact sign-flip permutation test  (§5.3 robustness)
# ===========================================================================


@dataclass(frozen=True)
class PermutationResult:
    """Exact sign-flip permutation test over all 2^n sign patterns."""

    observed: float        # observed mean of the diffs
    p: float               # one-sided permutation p
    n: int
    n_patterns: int        # 2^n
    n_extreme: int         # # of patterns at least as extreme
    min_p: float           # 1 / 2^n, the floor
    direction: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed, "p": self.p, "n": self.n,
            "n_patterns": self.n_patterns, "n_extreme": self.n_extreme,
            "min_p": self.min_p, "direction": self.direction,
        }


def sign_flip_permutation(
    diffs: Sequence[float] | np.ndarray,
    direction: str = "greater",
) -> PermutationResult:
    """EXACT sign-flip permutation test (all 2^n patterns enumerated).

    Under the paired-design exchangeability null, each per-seed
    difference is equally likely to carry either sign, so the exact
    randomisation distribution of the mean is obtained by enumerating all
    2^n {+1,-1} sign assignments — 32 for the n=5 confirmatory set.  The
    one-sided p is the fraction of sign patterns whose mean is at least
    as extreme (≥ for 'greater', ≤ for 'less') as the observed mean.

    The smallest attainable p is 1/2^n (= 1/32 ≈ 0.031 at n=5): the
    single all-same-sign pattern.  This is the pre-registered robustness
    statistic; "ROBUSTLY SUPPORTED" requires exact p ≤ 1/32 (§3).

    Test statistic: the *sum* (equivalently the mean) of the signed
    differences.  We compare against the observed mean with a tiny
    tolerance so that the observed pattern itself always counts as "at
    least as extreme" (avoids fp-equality misses).
    """
    if direction not in ("greater", "less"):
        raise ValueError("direction must be 'greater' or 'less'")
    arr = _as_diffs(diffs)
    n = arr.size
    if n > 24:
        # 2^24 ≈ 1.7e7 is the sane ceiling for full enumeration; the
        # confirmatory design is n=5, so this only guards misuse.
        raise ValueError(
            f"exact enumeration of 2^{n} patterns is infeasible; "
            "n is meant to be the 5-seed confirmatory set"
        )
    observed = float(np.mean(arr))
    n_patterns = 1 << n
    # Vectorised: build the 2^n x n sign matrix of ±1 and average |arr|·sign.
    bits = ((np.arange(n_patterns)[:, None] >> np.arange(n)) & 1).astype(float)
    signs = 1.0 - 2.0 * bits  # 0->+1, 1->-1
    means = (signs * arr[None, :]).mean(axis=1)
    tol = 1.0e-12 * max(1.0, abs(observed))
    if direction == "greater":
        n_extreme = int(np.count_nonzero(means >= observed - tol))
    else:
        n_extreme = int(np.count_nonzero(means <= observed + tol))
    p = n_extreme / n_patterns
    return PermutationResult(
        observed=observed, p=float(p), n=n, n_patterns=n_patterns,
        n_extreme=n_extreme, min_p=1.0 / n_patterns, direction=direction,
    )


# ===========================================================================
# 3. Sign test (binomial sign-consistency)  -- the load-bearing statistic
# ===========================================================================


def _binom_sf_ge(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact via stdlib comb."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(
        sum(math.comb(n, j) * p**j * (1.0 - p) ** (n - j) for j in range(k, n + 1))
    )


@dataclass(frozen=True)
class SignTestResult:
    """Exact binomial sign test for sign-consistency across seeds."""

    n_pos: int
    n_neg: int
    n_zero: int
    n_eff: int             # n_pos + n_neg (zeros dropped)
    n_concordant: int      # count in the hypothesised direction
    p: float               # one-sided exact binomial
    direction: str         # "greater" => count positives; "less" => negatives
    all_same_sign: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_pos": self.n_pos, "n_neg": self.n_neg, "n_zero": self.n_zero,
            "n_eff": self.n_eff, "n_concordant": self.n_concordant,
            "p": self.p, "direction": self.direction,
            "all_same_sign": self.all_same_sign,
        }


def sign_test(
    diffs: Sequence[float] | np.ndarray,
    direction: str = "greater",
) -> SignTestResult:
    """One-sided exact binomial sign test on the per-seed difference signs.

    The pre-registered *sign-consistency* statistic and, at n=5, the
    load-bearing one: 5/5 differences sharing the hypothesised sign gives
    p = (1/2)^5 = 1/32 ≈ 0.031 — the same floor the exact sign-flip
    permutation test attains, by the same all-same-sign event.

    Zeros are dropped (the conservative two-sided-style convention) and
    reduce the effective n.  ``direction='greater'`` counts positive
    differences as concordant; ``'less'`` counts negatives.  The p-value
    is P(Binomial(n_eff, 1/2) >= n_concordant).
    """
    if direction not in ("greater", "less"):
        raise ValueError("direction must be 'greater' or 'less'")
    arr = _as_diffs(diffs)
    n_pos = int(np.count_nonzero(arr > 0))
    n_neg = int(np.count_nonzero(arr < 0))
    n_zero = int(np.count_nonzero(arr == 0))
    n_eff = n_pos + n_neg
    n_concordant = n_pos if direction == "greater" else n_neg
    if n_eff == 0:
        p = 1.0
    else:
        p = _binom_sf_ge(n_concordant, n_eff, 0.5)
    all_same = n_eff > 0 and (n_concordant == n_eff)
    return SignTestResult(
        n_pos=n_pos, n_neg=n_neg, n_zero=n_zero, n_eff=n_eff,
        n_concordant=n_concordant, p=float(p), direction=direction,
        all_same_sign=bool(all_same),
    )


# ===========================================================================
# 4. Non-inferiority upper confidence bound  (§5.3; H-P2, H-P6ii)
# ===========================================================================


@dataclass(frozen=True)
class NonInferiorityResult:
    """One-sided 95 % t-UCB against a fixed margin (lower-is-better cost)."""

    mean: float
    sd: float
    se: float
    ucb: float             # one-sided upper 95 % confidence bound
    margin: float
    conf: float
    df: int
    n: int
    non_inferior: bool     # ucb < margin

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean, "sd": self.sd, "se": self.se, "ucb": self.ucb,
            "margin": self.margin, "conf": self.conf, "df": self.df,
            "n": self.n, "non_inferior": self.non_inferior,
        }


def noninferiority_ucb(
    diffs: Sequence[float] | np.ndarray,
    margin: float = NONINF_MARGIN_PP,
    conf: float = 0.95,
) -> NonInferiorityResult:
    """One-sided upper 95 % confidence bound for NON-INFERIORITY.

    For H-P2 / H-P6ii the difference is framed as a *cost* where smaller
    is better — e.g. ΔACC = ACC(control) - ACC(treatment) in pp, so a
    positive value means the treatment lost accuracy.  Non-inferiority
    holds when we can rule out a loss as large as the margin, i.e. when
    the one-sided upper confidence bound

        UCB = mean(d) + t_{conf, n-1} · se

    is strictly below ``margin`` (1.0 pp, §3).  ``non_inferior`` is the
    pre-registered accept condition (UCB < margin).

    The bound uses the Student-t critical value (small-sample correct at
    n=5), not a normal z, per §5.3 ("one-sided 95 % t-UCB").
    """
    arr = _as_diffs(diffs)
    n = arr.size
    if n < 2:
        raise ValueError("need >= 2 seeds for a t-UCB")
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    df = n - 1
    se = sd / math.sqrt(n)
    t_crit = student_t_ppf(conf, df)
    ucb = mean + t_crit * se
    return NonInferiorityResult(
        mean=mean, sd=sd, se=se, ucb=float(ucb), margin=float(margin),
        conf=conf, df=df, n=n, non_inferior=bool(ucb < margin),
    )


# ===========================================================================
# 5. Neutrality band  (§5.3; H-P7a -- no NHST)
# ===========================================================================


@dataclass(frozen=True)
class NeutralityResult:
    """Pre-registered neutrality band check (H-P7a): |Δ| < band."""

    delta: float
    band: float
    noise_floor: float
    rel_value: float
    within_band: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "delta": self.delta, "band": self.band,
            "noise_floor": self.noise_floor, "rel_value": self.rel_value,
            "within_band": self.within_band,
        }


def neutrality_band(
    delta_median: float,
    noise_floor: float,
    reference_value: float,
    noise_mult: float = 2.0,
    rel_frac: float = 0.05,
) -> NeutralityResult:
    """H-P7a neutrality band: ``|Δ median t_ε| < max(2·noise, 5 % of t_ε)``.

    Not an NHST (§5.3 "pre-registered band, no NHST"): a supportive check
    that the age cap costs nothing in wire time.  ``reference_value`` is
    the arm's own median t_ε used for the 5 % relative leg.  Returns
    ``within_band=True`` when the cap is wire-neutral.
    """
    band = max(noise_mult * abs(noise_floor), rel_frac * abs(reference_value))
    return NeutralityResult(
        delta=float(delta_median), band=float(band),
        noise_floor=float(noise_floor),
        rel_value=float(rel_frac * abs(reference_value)),
        within_band=bool(abs(delta_median) < band),
    )


# ===========================================================================
# 6. Spearman rank correlation with exact permutation p  (§5.3; H-P5ii)
# ===========================================================================


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (scipy-free 'average' method)."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0  # 1-based average rank over the tie block
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ = Pearson correlation of the average ranks."""
    rx = _rankdata_average(x)
    ry = _rankdata_average(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


@dataclass(frozen=True)
class SpearmanResult:
    """Spearman ρ with exact permutation p over all n! orderings."""

    rho: float
    p: float
    n: int
    n_perms: int           # n!
    n_extreme: int
    min_p: float           # 1 / n!
    direction: str         # "greater" (positive assoc) or "less"

    def as_dict(self) -> dict[str, Any]:
        return {
            "rho": self.rho, "p": self.p, "n": self.n,
            "n_perms": self.n_perms, "n_extreme": self.n_extreme,
            "min_p": self.min_p, "direction": self.direction,
        }


def spearman_exact(
    x: Sequence[float],
    y: Sequence[float],
    direction: str = "greater",
) -> SpearmanResult:
    """Spearman rank correlation with EXACT permutation p (n! orderings).

    H-P5ii dose-response: correlate the per-seed accuracy gap d_s with the
    measured client-importance divergence D_imp(s).  With n=5 the exact
    null enumerates all 5! = 120 pairings of y against the fixed x; the
    one-sided p is the fraction of pairings whose ρ is at least as extreme
    as observed (≥ for 'greater', a positive dose-response).

    Minimum attainable p is 1/n! = 1/120 ≈ 0.0083.  This is supporting
    evidence, not a gate (§3, H-P5).
    """
    if direction not in ("greater", "less"):
        raise ValueError("direction must be 'greater' or 'less'")
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.shape != ya.shape:
        raise ValueError(f"x, y length mismatch: {xa.shape} vs {ya.shape}")
    n = xa.size
    if n < 3:
        raise ValueError("need >= 3 points for a rank correlation")
    if n > 9:
        # 9! = 362880; beyond that exact enumeration is wasteful and the
        # design never needs it (n=5 confirmatory).
        raise ValueError(
            f"exact n! enumeration infeasible for n={n}; design is n=5"
        )
    rho = _spearman_rho(xa, ya)
    rx = _rankdata_average(xa)
    ry = _rankdata_average(ya)
    # Permute y's ranks against fixed x's ranks; ρ is monotone in the rank
    # dot product, so enumerate that.
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = math.sqrt(float(np.dot(rx_c, rx_c)) * float(np.dot(ry_c, ry_c)))
    n_perms = math.factorial(n)
    tol = 1.0e-12 * max(1.0, abs(rho))
    n_extreme = 0
    if denom == 0.0:
        # No variance in ranks ⇒ ρ undefined; treat as null (p=1).
        return SpearmanResult(
            rho=float("nan"), p=1.0, n=n, n_perms=n_perms, n_extreme=n_perms,
            min_p=1.0 / n_perms, direction=direction,
        )
    for perm in permutations(range(n)):
        r = float(np.dot(rx_c, ry_c[list(perm)]) / denom)
        if direction == "greater":
            if r >= rho - tol:
                n_extreme += 1
        else:
            if r <= rho + tol:
                n_extreme += 1
    p = n_extreme / n_perms
    return SpearmanResult(
        rho=float(rho), p=float(p), n=n, n_perms=n_perms,
        n_extreme=n_extreme, min_p=1.0 / n_perms, direction=direction,
    )


# ===========================================================================
# 7. Holm correction within a hypothesis family  (§5.3 multiplicity)
# ===========================================================================


@dataclass(frozen=True)
class HolmResult:
    """Holm-Bonferroni step-down adjustment within one family."""

    labels: tuple[str, ...]
    raw_p: tuple[float, ...]
    adj_p: tuple[float, ...]      # adjusted p, aligned to ``labels``
    reject: tuple[bool, ...]      # adj_p <= alpha
    alpha: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "raw_p": list(self.raw_p),
            "adj_p": list(self.adj_p),
            "reject": list(self.reject),
            "alpha": self.alpha,
        }


def holm(
    pvalues: Sequence[float],
    labels: Sequence[str] | None = None,
    alpha: float = ALPHA,
) -> HolmResult:
    """Holm step-down adjustment for a family of one-sided p-values.

    §5.3 multiplicity: Holm *within each hypothesis family* (e.g. H-P4's
    two contrasts; a hypothesis tested at both ε points).  No cross-family
    correction.  Returns monotone-enforced adjusted p-values aligned to
    the input order, and the reject flags at ``alpha``.

    Adjusted p for the k-th smallest raw p (0-based) is
    ``(m - k) · p`` carried forward as a running maximum, capped at 1.
    """
    p = [float(x) for x in pvalues]
    m = len(p)
    if m == 0:
        return HolmResult((), (), (), (), alpha)
    if labels is None:
        labels = tuple(f"contrast_{i}" for i in range(m))
    else:
        labels = tuple(labels)
        if len(labels) != m:
            raise ValueError("labels length must match pvalues length")
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running = 0.0
    for k, idx in enumerate(order):
        val = (m - k) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    reject = tuple(adj[i] <= alpha for i in range(m))
    return HolmResult(
        labels=labels, raw_p=tuple(p), adj_p=tuple(adj),
        reject=reject, alpha=alpha,
    )


# ===========================================================================
# Verdict mapper  (§3 ladder)
# ===========================================================================

ROBUSTLY_SUPPORTED = "ROBUSTLY SUPPORTED"
SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT SUPPORTED"
NOT_EVALUABLE = "NOT EVALUABLE"


@dataclass(frozen=True)
class Verdict:
    """A single hypothesis-contrast resolution (§3 / §5.4)."""

    label: str
    verdict: str
    direction: str
    mean_effect: float
    min_effect: float
    t_p: float
    perm_p: float
    sign_p: float
    perm_min_p: float
    effect_ok: bool
    concordant: bool       # t-test and exact tests agree in direction
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "verdict": self.verdict,
            "direction": self.direction, "mean_effect": self.mean_effect,
            "min_effect": self.min_effect, "t_p": self.t_p,
            "perm_p": self.perm_p, "sign_p": self.sign_p,
            "perm_min_p": self.perm_min_p, "effect_ok": self.effect_ok,
            "concordant": self.concordant, "notes": list(self.notes),
        }


def directional_verdict(
    diffs: Sequence[float] | np.ndarray,
    direction: str,
    min_effect: float,
    label: str = "",
    alpha: float = ALPHA,
    p_for_verdict: float | None = None,
) -> Verdict:
    """Map a directional contrast to the §3 verdict ladder.

    Rules (§3, §5.3):

    - **SUPPORTED**  iff the (Holm-adjusted, if supplied via
      ``p_for_verdict``) one-sided p < α AND the *signed* mean effect
      meets the pre-registered minimum in the hypothesised direction.
    - **ROBUSTLY SUPPORTED** additionally requires the exact sign-flip
      permutation p ≤ 1/2^n (= 1/32 at n=5).
    - **NOT SUPPORTED** otherwise.
    - Discordance between the t-test and the exact tests (different
      direction of the effect) downgrades to NOT SUPPORTED and is flagged
      "directional only" (§5.3).

    ``p_for_verdict`` lets a caller substitute the Holm-adjusted p of the
    family while still reporting the raw t/perm/sign p's; when None the
    raw t-test p is used.

    The "minimum effect" is checked on the *signed* mean in the
    hypothesised direction: for ``direction='greater'`` the mean must be
    ≥ +min_effect; for ``'less'`` it must be ≤ -min_effect.  ``min_effect``
    is given as a positive magnitude.
    """
    t_res = paired_t_one_sided(diffs, direction=direction)
    perm_res = sign_flip_permutation(diffs, direction=direction)
    sign_res = sign_test(diffs, direction=direction)
    mean = t_res.mean

    notes: list[str] = []

    # Effect-size gate on the signed mean.
    if direction == "greater":
        effect_ok = mean >= min_effect
        # t/exact concordance: all should point the same way (mean sign vs
        # hypothesised). The exact tests' "extremeness" is in-direction by
        # construction; concordance is really "does the observed mean sit
        # on the hypothesised side".
        concordant = mean > 0
    else:
        effect_ok = mean <= -min_effect
        concordant = mean < 0

    p_test = t_res.p if p_for_verdict is None else float(p_for_verdict)

    # A t-test that is significant in the *wrong* tail cannot happen here
    # (one-sided p is large when the mean is against the direction), but
    # guard the discordance case for completeness / Wilcoxon-trigger note.
    dom = dominant_seed(diffs)
    if dom >= 0:
        notes.append(
            f"single seed dominates (idx {dom}, |d|>3x median |d|); "
            "report Wilcoxon per §5.3"
        )

    robust = perm_res.p <= perm_res.min_p + 1e-15

    if not concordant:
        verdict = NOT_SUPPORTED
        notes.append("mean effect is against the hypothesised direction")
    elif p_test < alpha and effect_ok:
        verdict = ROBUSTLY_SUPPORTED if robust else SUPPORTED
    else:
        verdict = NOT_SUPPORTED
        if p_test >= alpha:
            notes.append(f"p={p_test:.4f} >= alpha={alpha}")
        if not effect_ok:
            notes.append(
                f"mean effect {mean:+.3f} below minimum {min_effect:.3f} pp"
            )

    # Discordance flag (§5.3): t vs exact disagree on significance.
    t_sig = t_res.p < alpha
    perm_sig = perm_res.p < alpha
    if t_sig != perm_sig and verdict != NOT_EVALUABLE:
        notes.append(
            "t-test and exact sign-flip disagree on significance "
            "=> 'directional only'"
        )

    return Verdict(
        label=label, verdict=verdict, direction=direction,
        mean_effect=float(mean), min_effect=float(min_effect),
        t_p=float(t_res.p), perm_p=float(perm_res.p), sign_p=float(sign_res.p),
        perm_min_p=float(perm_res.min_p), effect_ok=bool(effect_ok),
        concordant=bool(concordant), notes=tuple(notes),
    )


def noninferiority_verdict(
    diffs: Sequence[float] | np.ndarray,
    margin: float = NONINF_MARGIN_PP,
    label: str = "",
    conf: float = 0.95,
) -> Verdict:
    """Map a non-inferiority contrast (H-P2, H-P6ii) to the §3 ladder.

    ``diffs`` is the per-seed accuracy *cost* (ACC_control - ACC_treatment,
    pp; positive ⇒ the treatment lost accuracy).  SUPPORTED iff the
    one-sided 95 % upper confidence bound is below the margin (§5.3); the
    ROBUST tier does not apply to non-inferiority (no permutation tier is
    pre-registered for it), so this returns at most SUPPORTED.
    """
    res = noninferiority_ucb(diffs, margin=margin, conf=conf)
    verdict = SUPPORTED if res.non_inferior else NOT_SUPPORTED
    notes = [
        f"UCB={res.ucb:+.3f} pp {'<' if res.non_inferior else '>='} "
        f"margin={margin:.2f} pp"
    ]
    return Verdict(
        label=label, verdict=verdict, direction="noninferiority",
        mean_effect=float(res.mean), min_effect=float(margin),
        t_p=float("nan"), perm_p=float("nan"), sign_p=float("nan"),
        perm_min_p=float("nan"), effect_ok=bool(res.non_inferior),
        concordant=True, notes=tuple(notes),
    )


# ===========================================================================
# Hypothesis registry  (§3 + §4.5 run matrix wiring)
# ===========================================================================
#
# Each entry describes one confirmatory contrast: which two arms it pairs,
# which metric, the contrast "sense" (how to turn the two arms into a
# per-seed difference and which tail to test), the pre-registered minimum
# effect, and the family it belongs to for Holm grouping.  Arm numbers are
# the §4.5 P3 matrix rows.


@dataclass(frozen=True)
class Contrast:
    """A pre-registered confirmatory contrast (one row of §3/§4.5)."""

    label: str
    family: str
    kind: str              # "directional" | "noninferiority"
    treatment_arm: str
    control_arm: str
    metric: str
    # How to build per-seed diffs and which tail:
    #   directional: diff = sense * (treatment_metric - control_metric),
    #     tested 'greater' (sense flips the sign so the hypothesis is
    #     always "diff > 0 in the claimed direction").
    #   noninferiority: cost = control_metric - treatment_metric.
    sense: float           # +1 or -1 (directional only)
    direction: str         # 'greater' or 'less' (the claim's natural tail)
    min_effect: float      # pre-registered minimum (pp), magnitude
    description: str = ""


#: The wired confirmatory contrasts.  H-P1 (wire-time / tracking) and the
#: H-P7a neutrality / H-P7c staleness legs are telemetry-threshold checks,
#: not seed-paired metric contrasts, so they are documented here but not
#: run by this driver (they belong to the SNAC/μ̄ + integrity-gate scripts,
#: §5.6); this driver owns the seed-paired NHST/UCB/correlation legs.
HYPOTHESES: tuple[Contrast, ...] = (
    # H-P2: ε_mid accuracy non-inferior to monolithic control (arm 3 vs 1).
    Contrast(
        label="H-P2",
        family="H-P2",
        kind="noninferiority",
        treatment_arm="3",
        control_arm="1",
        metric="acc",
        sense=+1.0,
        direction="less",          # cost; UCB < margin
        min_effect=NONINF_MARGIN_PP,
        description="ε_mid ACC non-inferior to monolithic (margin 1.0 pp)",
    ),
    # H-P3: recycle beats drop at ε_high (arm 4 vs 5), ΔACC>0, min 0.5 pp.
    Contrast(
        label="H-P3",
        family="H-P3",
        kind="directional",
        treatment_arm="4",
        control_arm="5",
        metric="acc",
        sense=+1.0,
        direction="greater",
        min_effect=0.5,
        description="recycle_last_delta > drop at ε_high (min 0.5 pp)",
    ),
    # H-P4a (primary): coverage-EFT cost < cyclic cost (arm 4 vs 7), min 1.0.
    #   ΔACC(4) < ΔACC(7) where ΔACC = ACC_mono - ACC_arm; equivalently
    #   ACC(4) > ACC(7) at matched μ̄.  We test ACC(4)-ACC(7) > 0.
    Contrast(
        label="H-P4a",
        family="H-P4",
        kind="directional",
        treatment_arm="4",
        control_arm="7",
        metric="acc",
        sense=+1.0,
        direction="greater",
        min_effect=1.0,
        description="coverage-EFT < cyclic accuracy cost at matched μ̄ "
                    "(min 1.0 pp)",
    ),
    # H-P4b (secondary): coverage-EFT cost < byte-balanced (arm 4 vs 6), 0.5.
    Contrast(
        label="H-P4b",
        family="H-P4",
        kind="directional",
        treatment_arm="4",
        control_arm="6",
        metric="acc",
        sense=+1.0,
        direction="greater",
        min_effect=0.5,
        description="coverage-EFT < byte-balanced accuracy cost at matched "
                    "μ̄ (min 0.5 pp)",
    ),
    # H-P5i: per-sender selection beats global (arm 9 vs 10), ACC, min 1.0.
    Contrast(
        label="H-P5",
        family="H-P5",
        kind="directional",
        treatment_arm="9",
        control_arm="10",
        metric="acc",
        sense=+1.0,
        direction="greater",
        min_effect=1.0,
        description="per-sender skip (9) > FedLUAR-mode global (10), "
                    "matched bytes (min 1.0 pp)",
    ),
    # H-P6i: skip-v2 reduces uplink bytes (arm 9 vs 3).  The pre-registered
    #   statistic is the one-sided paired test on the *per-seed byte ratio*
    #   (§3 H-P6 criteria i), with the saving fraction (1 - bytes_9/bytes_3)
    #   required to be >= μ̄(arm 3)/2.  Saving and the μ̄/2 minimum are both
    #   byte *fractions*, so kind='byte_ratio_saving' builds the
    #   commensurable per-seed quantity rather than a raw byte difference.
    #   The min effect is data-dependent (μ̄(arm 3)/2 from the μ̄ extraction,
    #   §4.4); supply it via CLI --hp6-min, else a 0.0 floor (significance
    #   that *any* bytes were removed).
    Contrast(
        label="H-P6i",
        family="H-P6",
        kind="byte_ratio_saving",
        treatment_arm="9",
        control_arm="3",
        metric="bytes_per_round",
        sense=+1.0,                # saving = 1 - bytes_9/bytes_3, claim > 0
        direction="greater",
        min_effect=0.0,            # overridden by --hp6-min (μ̄/2) at runtime
        description="skip-v2 reduces uplink bytes vs off (per-seed byte "
                    "ratio; saving >= μ̄/2)",
    ),
    # H-P6ii: ACC(9) non-inferior to ACC(3) (margin 1.0 pp).
    Contrast(
        label="H-P6ii",
        family="H-P6",
        kind="noninferiority",
        treatment_arm="9",
        control_arm="3",
        metric="acc",
        sense=+1.0,
        direction="less",
        min_effect=NONINF_MARGIN_PP,
        description="skip-v2 ACC non-inferior to skip-off (margin 1.0 pp)",
    ),
    # H-P7b: aging improves ACC vs none at ε_high (arm 4 vs 8), min 0.25 pp.
    Contrast(
        label="H-P7b",
        family="H-P7",
        kind="directional",
        treatment_arm="4",
        control_arm="8",
        metric="acc",
        sense=+1.0,
        direction="greater",
        min_effect=0.25,
        description="aging (4) > no-aging (8) at ε_high (min 0.25 pp)",
    ),
)


# ===========================================================================
# CLI input parsing: per-(arm, seed) metric table
# ===========================================================================
#
# Accepted inputs (auto-detected by extension / content):
#
#  * CSV long-format with header columns: arm, seed, metric, value
#       arm,seed,metric,value
#       4,42,acc,71.3
#       5,42,acc,69.9
#       ...
#  * CSV wide-format with header: arm, seed, <metric1>, <metric2>, ...
#       arm,seed,acc,bytes_per_round
#       4,42,71.3,1.2e7
#  * JSON: either
#       {"arm": {"seed": {"metric": value}}}            (nested)
#       or a list of {"arm":..,"seed":..,"metric":..,"value":..} records
#       or a list of {"arm":..,"seed":..,"acc":..,...} wide records.
#
# Arms are keyed by string (the §4.5 row numbers "1".."10"); seeds by int.


MetricTable = dict[str, dict[int, dict[str, float]]]


def _ingest_record(
    table: MetricTable, arm: Any, seed: Any, metric: str, value: Any
) -> None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return
    arm_s = str(arm).strip()
    try:
        seed_i = int(seed)
    except (TypeError, ValueError):
        raise ValueError(f"non-integer seed {seed!r}")
    table.setdefault(arm_s, {}).setdefault(seed_i, {})[str(metric).strip()] = (
        float(value)
    )


def load_metric_table(path: str | Path) -> MetricTable:
    """Load a per-(arm, seed) metric table from CSV or JSON (auto-detect)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    is_json = suffix == ".json" or text.lstrip()[:1] in "{["
    table: MetricTable = {}
    if is_json:
        data = json.loads(text)
        _ingest_json(table, data)
    else:
        _ingest_csv(table, text)
    if not table:
        raise ValueError(f"no usable (arm, seed, metric) records in {path}")
    return table


def _ingest_json(table: MetricTable, data: Any) -> None:
    if isinstance(data, dict):
        # nested arm -> seed -> metric -> value
        for arm, seeds in data.items():
            if not isinstance(seeds, dict):
                continue
            for seed, metrics in seeds.items():
                if not isinstance(metrics, dict):
                    continue
                for metric, value in metrics.items():
                    _ingest_record(table, arm, seed, metric, value)
    elif isinstance(data, list):
        for rec in data:
            if not isinstance(rec, dict):
                continue
            if "metric" in rec and "value" in rec:
                _ingest_record(
                    table, rec.get("arm"), rec.get("seed"),
                    rec["metric"], rec["value"],
                )
            else:
                arm, seed = rec.get("arm"), rec.get("seed")
                for key, value in rec.items():
                    if key in ("arm", "seed"):
                        continue
                    _ingest_record(table, arm, seed, key, value)
    else:
        raise ValueError("unsupported JSON shape for metric table")


def _ingest_csv(table: MetricTable, text: str) -> None:
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise ValueError("CSV has no header row")
    cols = {c.strip().lower(): c for c in reader.fieldnames}
    if "arm" not in cols or "seed" not in cols:
        raise ValueError("CSV must have 'arm' and 'seed' columns")
    long_format = "metric" in cols and "value" in cols
    metric_cols = [
        c for c in reader.fieldnames
        if c.strip().lower() not in ("arm", "seed", "metric", "value")
    ]
    for row in reader:
        arm, seed = row[cols["arm"]], row[cols["seed"]]
        if long_format:
            _ingest_record(
                table, arm, seed, row[cols["metric"]], row[cols["value"]]
            )
        else:
            for mc in metric_cols:
                _ingest_record(table, arm, seed, mc, row[mc])


def paired_arm_diffs(
    table: MetricTable,
    treatment_arm: str,
    control_arm: str,
    metric: str,
) -> tuple[np.ndarray, list[int]]:
    """Per-seed paired differences ``treatment - control`` on the common seeds.

    Enforces the P-4 seed-composition rule: only seeds present in *both*
    arms for ``metric`` are used, returned in sorted seed order alongside
    the seed list so callers can report the realised pairing.
    """
    t_seeds = table.get(treatment_arm, {})
    c_seeds = table.get(control_arm, {})
    common = sorted(
        s for s in t_seeds
        if s in c_seeds
        and metric in t_seeds[s]
        and metric in c_seeds[s]
    )
    if not common:
        raise KeyError(
            f"no common seeds with metric {metric!r} for arms "
            f"{treatment_arm} (treatment) and {control_arm} (control)"
        )
    t = np.array([t_seeds[s][metric] for s in common], dtype=float)
    c = np.array([c_seeds[s][metric] for s in common], dtype=float)
    return t - c, common


def paired_ratio_saving(
    table: MetricTable,
    treatment_arm: str,
    control_arm: str,
    metric: str,
) -> tuple[np.ndarray, list[int]]:
    """Per-seed fractional saving ``1 - treatment/control`` (H-P6i).

    The pre-registered H-P6i statistic is the one-sided paired test on the
    per-seed *byte ratio* (§3); the saving fraction ``1 - bytes_9/bytes_3``
    is dimensionless and directly comparable to the μ̄/2 minimum-effect
    threshold (a byte fraction).  Control values of 0 are skipped
    (undefined ratio) and reported by the shorter seed list.
    """
    t_seeds = table.get(treatment_arm, {})
    c_seeds = table.get(control_arm, {})
    common = sorted(
        s for s in t_seeds
        if s in c_seeds
        and metric in t_seeds[s]
        and metric in c_seeds[s]
        and c_seeds[s][metric] != 0.0
    )
    if not common:
        raise KeyError(
            f"no common seeds with nonzero control metric {metric!r} for "
            f"arms {treatment_arm} (treatment) and {control_arm} (control)"
        )
    saving = np.array(
        [1.0 - t_seeds[s][metric] / c_seeds[s][metric] for s in common],
        dtype=float,
    )
    return saving, common


# ===========================================================================
# CLI driver: run the wired hypotheses and print a verdict table
# ===========================================================================


def run_hypotheses(
    table: MetricTable,
    hypotheses: Sequence[Contrast] = HYPOTHESES,
    hp6_min: float | None = None,
) -> dict[str, Any]:
    """Run every wired contrast that has data; Holm within each family.

    Returns a structured dict: per-contrast results plus the
    family-level Holm adjustment and final verdicts.  Contrasts whose
    arms/seeds are missing resolve to NOT EVALUABLE with the reason cited
    (§5.4).
    """
    # 1. Compute each contrast's raw statistic where data exists.
    per_contrast: dict[str, dict[str, Any]] = {}
    for c in hypotheses:
        entry: dict[str, Any] = {
            "label": c.label, "family": c.family, "kind": c.kind,
            "treatment_arm": c.treatment_arm, "control_arm": c.control_arm,
            "metric": c.metric, "description": c.description,
            "min_effect": c.min_effect,
        }
        try:
            if c.kind == "byte_ratio_saving":
                paired, seeds = paired_ratio_saving(
                    table, c.treatment_arm, c.control_arm, c.metric
                )
            else:
                paired, seeds = paired_arm_diffs(
                    table, c.treatment_arm, c.control_arm, c.metric
                )
        except KeyError as exc:
            entry["verdict"] = NOT_EVALUABLE
            entry["reason"] = str(exc)
            per_contrast[c.label] = entry
            continue

        entry["seeds"] = seeds
        entry["n"] = len(seeds)
        if len(seeds) < N_SEEDS:
            entry["partial_seeds"] = True  # below the 5-seed confirmatory set

        if c.kind == "noninferiority":
            # cost = control - treatment = -(treatment - control) = -paired
            cost = -paired
            entry["cost_mean"] = float(np.mean(cost))
            ni = noninferiority_ucb(cost, margin=c.min_effect)
            entry["noninferiority"] = ni.as_dict()
            entry["_diffs"] = cost
        else:
            # directional (incl. byte_ratio_saving): orient so the claim is
            # "signed > 0".  For byte_ratio_saving ``paired`` is already the
            # per-seed saving fraction 1 - bytes_t/bytes_c (sense=+1).
            signed = c.sense * paired
            min_eff = c.min_effect
            if c.label == "H-P6i" and hp6_min is not None:
                min_eff = float(hp6_min)
                entry["min_effect"] = min_eff
            entry["mean_effect"] = float(np.mean(signed))
            entry["t"] = paired_t_one_sided(signed, "greater").as_dict()
            entry["perm"] = sign_flip_permutation(signed, "greater").as_dict()
            entry["sign"] = sign_test(signed, "greater").as_dict()
            entry["_diffs"] = signed
            entry["_min_effect"] = min_eff
        per_contrast[c.label] = entry

    # 2. Holm within each family over the *evaluable* contrasts' raw p's.
    families: dict[str, list[str]] = {}
    for c in hypotheses:
        families.setdefault(c.family, []).append(c.label)

    holm_by_family: dict[str, dict[str, Any]] = {}
    for fam, labels in families.items():
        evaluable = [
            lab for lab in labels
            if per_contrast[lab].get("verdict") != NOT_EVALUABLE
        ]
        # The family's one-sided p per contrast (t-test p for directional,
        # 1 - non-inferiority indicator is not a p; for non-inferiority we
        # Holm-adjust nothing — UCB decisions are not p-values.  We Holm
        # only the directional members; non-inferiority members keep their
        # own decision).  Per §5.3 Holm applies to the family's *tests*.
        dir_labels = [
            lab for lab in evaluable
            if per_contrast[lab]["kind"] in ("directional", "byte_ratio_saving")
        ]
        if len(dir_labels) >= 2:
            raw_p = [per_contrast[lab]["t"]["p"] for lab in dir_labels]
            hres = holm(raw_p, dir_labels)
            holm_by_family[fam] = hres.as_dict()
            for lab, ap in zip(dir_labels, hres.adj_p):
                per_contrast[lab]["holm_adj_p"] = float(ap)
        else:
            holm_by_family[fam] = {"note": "single directional contrast; "
                                   "no within-family Holm needed"}

    # 3. Final verdicts (using Holm-adjusted p where a family had >= 2).
    verdicts: dict[str, Any] = {}
    for c in hypotheses:
        entry = per_contrast[c.label]
        if entry.get("verdict") == NOT_EVALUABLE:
            verdicts[c.label] = {
                "verdict": NOT_EVALUABLE, "reason": entry.get("reason", ""),
            }
            continue
        diffs = entry["_diffs"]
        if c.kind == "noninferiority":
            v = noninferiority_verdict(diffs, margin=c.min_effect, label=c.label)
        else:
            p_adj = entry.get("holm_adj_p")
            v = directional_verdict(
                diffs, "greater", entry["_min_effect"], label=c.label,
                p_for_verdict=p_adj,
            )
        notes = list(v.notes)
        if entry.get("partial_seeds"):
            notes.append(
                f"PARTIAL: {entry['n']} seeds < {N_SEEDS} confirmatory "
                "(P-4 common-seed rule not fully met)"
            )
        vd = v.as_dict()
        vd["notes"] = notes
        verdicts[c.label] = vd

    # Strip private arrays before returning.
    for entry in per_contrast.values():
        entry.pop("_diffs", None)
        entry.pop("_min_effect", None)

    return {
        "contrasts": per_contrast,
        "holm_by_family": holm_by_family,
        "verdicts": verdicts,
    }


_VERDICT_RANK = {
    ROBUSTLY_SUPPORTED: 0,
    SUPPORTED: 1,
    NOT_SUPPORTED: 2,
    NOT_EVALUABLE: 3,
}


def format_verdict_table(result: dict[str, Any]) -> str:
    """Human-readable verdict table for the CLI."""
    rows: list[tuple[str, ...]] = []
    header = ("hypothesis", "verdict", "mean_eff", "min_eff", "t_p",
              "perm_p", "sign_p", "UCB/notes")
    for label, entry in result["contrasts"].items():
        vd = result["verdicts"].get(label, {})
        verdict = vd.get("verdict", "?")
        if verdict == NOT_EVALUABLE:
            rows.append((label, verdict, "-", "-", "-", "-", "-",
                         (vd.get("reason", "") or "")[:48]))
            continue
        if entry["kind"] == "noninferiority":
            ni = entry["noninferiority"]
            rows.append((
                label, verdict,
                f"{entry.get('cost_mean', float('nan')):+.3f}",
                f"{ni['margin']:.2f}",
                "-", "-", "-",
                f"UCB={ni['ucb']:+.3f}",
            ))
        else:
            t = entry["t"]
            perm = entry["perm"]
            sgn = entry["sign"]
            note = ""
            if entry.get("holm_adj_p") is not None:
                note = f"holm_p={entry['holm_adj_p']:.4f}"
            rows.append((
                label, verdict,
                f"{entry['mean_effect']:+.3f}",
                f"{entry.get('min_effect', float('nan')):.2f}",
                f"{t['p']:.4f}", f"{perm['p']:.4f}", f"{sgn['p']:.4f}",
                note,
            ))

    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "  "

    def fmt(row: Sequence[str]) -> str:
        return sep.join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    lines = [fmt(header), sep.join("-" * w for w in widths)]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)


# ===========================================================================
# Self-test
# ===========================================================================


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def selftest(verbose: bool = True) -> dict[str, Any]:
    """Synthetic-data proof reproducing the DeepCNN-C1 pattern.

    Paired diffs -2.7,-2.8,-2.0,-2.7,-1.8 (all negative, the C1 recycle-
    vs-drop style signal).  Verifies:

      * all five diffs share a sign (5/5 negative);
      * the sign test returns p = 1/32 (the load-bearing floor);
      * the exact sign-flip permutation test also returns 1/32;
      * the one-sided paired t-test is significant in the *same* (less)
        direction, so t / perm / sign agree in sign;
      * the non-inferiority UCB path runs and is well-ordered;
      * the Spearman exact-permutation path runs and is monotone.

    Returns a structured proof dict (also the basis for the
    StructuredOutput evidence).
    """
    out: dict[str, Any] = {"checks": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        out["checks"].append({"name": name, "pass": bool(ok), "detail": detail})
        if verbose:
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {name}" + (f" :: {detail}" if detail else ""))

    # The DeepCNN-C1 pattern: recycle - drop style negative deltas, OR
    # equivalently ACC(treatment) - ACC(control) all negative.  The claim's
    # natural direction is 'less' (treatment below control).
    diffs = [-2.7, -2.8, -2.0, -2.7, -1.8]
    out["diffs"] = diffs

    # --- sign test (load-bearing) ---
    st = sign_test(diffs, direction="less")
    check(
        "sign_test all-negative => 5/5 concordant",
        st.all_same_sign and st.n_concordant == 5 and st.n_neg == 5,
        f"n_neg={st.n_neg} n_concordant={st.n_concordant}",
    )
    check(
        "sign_test p == 1/32",
        _approx(st.p, 1.0 / 32.0),
        f"p={st.p:.6f} (1/32={1/32:.6f})",
    )

    # --- exact sign-flip permutation ---
    pm = sign_flip_permutation(diffs, direction="less")
    check(
        "permutation enumerates 2^5 = 32 patterns",
        pm.n_patterns == 32,
        f"n_patterns={pm.n_patterns}",
    )
    check(
        "permutation p == 1/32 (min_p floor attained)",
        _approx(pm.p, 1.0 / 32.0) and _approx(pm.min_p, 1.0 / 32.0),
        f"p={pm.p:.6f} min_p={pm.min_p:.6f} n_extreme={pm.n_extreme}",
    )

    # --- paired one-sided t-test ---
    tt = paired_t_one_sided(diffs, direction="less")
    check(
        "t-test significant in 'less' tail (mean<0, p<0.05)",
        tt.mean < 0 and tt.p < ALPHA and tt.df == 4,
        f"mean={tt.mean:.4f} t={tt.t:.4f} df={tt.df} p={tt.p:.6f}",
    )

    # --- agreement in sign across the three tests ---
    agree = (tt.mean < 0) and (st.n_neg == 5) and (pm.observed < 0)
    check(
        "t / sign / permutation agree in SIGN (all negative)",
        agree,
        f"t.mean={tt.mean:.3f} perm.obs={pm.observed:.3f} sign.n_neg={st.n_neg}",
    )

    # --- verdict mapping for this directional contrast (min_effect 0.5) ---
    v = directional_verdict(diffs, "less", min_effect=0.5, label="C1-synthetic")
    check(
        "directional verdict == ROBUSTLY SUPPORTED",
        v.verdict == ROBUSTLY_SUPPORTED,
        f"verdict={v.verdict} (perm_p={v.perm_p:.4f} <= 1/32, "
        f"mean={v.mean_effect:.3f} <= -0.5)",
    )

    # --- non-inferiority UCB path runs and is well-ordered ---
    # Frame a tiny accuracy cost (control - treatment) that should be
    # non-inferior at the 1.0 pp margin: cost ~ {0.1,-0.2,0.0,0.3,-0.1}.
    cost = [0.1, -0.2, 0.0, 0.3, -0.1]
    ni = noninferiority_ucb(cost, margin=NONINF_MARGIN_PP)
    check(
        "non-inferiority UCB runs, ordered mean<=UCB, decision computed",
        math.isfinite(ni.ucb) and ni.ucb >= ni.mean and ni.df == 4,
        f"mean={ni.mean:.4f} ucb={ni.ucb:.4f} margin={ni.margin} "
        f"non_inferior={ni.non_inferior}",
    )
    check(
        "non-inferiority verdict for small cost == SUPPORTED",
        noninferiority_verdict(cost).verdict == SUPPORTED,
        f"ucb={ni.ucb:.4f} < {NONINF_MARGIN_PP}",
    )

    # --- Spearman exact permutation path runs (monotone dose-response) ---
    # Pair a strictly increasing |gap| with a strictly increasing
    # divergence (no ties) => perfect rank concordance rho=1 and the
    # minimum attainable exact p = 1/120.  (Ties would average ranks and
    # legitimately pull rho below 1, so the clean dose-response uses
    # distinct values.)
    gap = [1.8, 2.0, 2.4, 2.7, 2.8]          # strictly increasing, no ties
    dimp = [0.10, 0.12, 0.15, 0.17, 0.20]    # strictly increasing
    sp = spearman_exact(gap, dimp, direction="greater")
    check(
        "spearman exact: rho==1 on strictly-monotone data, p==1/120",
        _approx(sp.rho, 1.0) and _approx(sp.p, 1.0 / 120.0)
        and sp.n_perms == 120,
        f"rho={sp.rho:.4f} p={sp.p:.6f} (1/120={1/120:.6f}) "
        f"n_perms={sp.n_perms}",
    )
    # Also exercise the realistic *tied* path so the C1 |gap| pattern
    # (which has a 2.7 tie) is covered: rho<1, p still well-formed.
    sp_tied = spearman_exact(
        [1.8, 2.0, 2.7, 2.7, 2.8], dimp, direction="greater"
    )
    check(
        "spearman exact handles ties (rho<1, valid p, 120 perms)",
        sp_tied.rho < 1.0 and 0.0 < sp_tied.p <= 1.0
        and sp_tied.n_perms == 120,
        f"rho={sp_tied.rho:.4f} p={sp_tied.p:.6f} n_perms={sp_tied.n_perms}",
    )

    # --- Holm sanity: two p's, monotone, capped at 1 ---
    h = holm([0.02, 0.04], ["a", "b"])
    check(
        "holm two-contrast adjustment monotone & capped",
        h.adj_p[0] <= h.adj_p[1] and all(p <= 1.0 for p in h.adj_p),
        f"adj_p={tuple(round(x, 4) for x in h.adj_p)}",
    )

    # --- scipy cross-check (only if scipy present; else informational) ---
    if HAVE_SCIPY:
        sci_sf = float(_scipy_stats.t.sf(abs(tt.t), tt.df))
        check(
            "student_t_sf agrees with scipy (~1e-10)",
            _approx(student_t_sf(abs(tt.t), tt.df), sci_sf, 1e-8),
            f"ours={student_t_sf(abs(tt.t), tt.df):.10f} scipy={sci_sf:.10f}",
        )
    else:
        out["checks"].append({
            "name": "scipy cross-check",
            "pass": True,
            "detail": "scipy absent (expected); numpy-only path used",
        })
        if verbose:
            print("[SKIP] scipy cross-check :: scipy absent (numpy-only path)")

    out["all_passed"] = all(c["pass"] for c in out["checks"])
    out["sign_test"] = st.as_dict()
    out["permutation"] = pm.as_dict()
    out["t_test"] = tt.as_dict()
    out["noninferiority"] = ni.as_dict()
    out["spearman"] = sp.as_dict()
    out["verdict"] = v.as_dict()
    out["have_scipy"] = HAVE_SCIPY
    if verbose:
        print()
        print("SELF-TEST:", "ALL PASSED" if out["all_passed"] else "FAILURES")
    return out


# ===========================================================================
# CLI
# ===========================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-registered confirmatory statistics for Phase-2 "
        "(protocol.md §3, §5.3): run the H-P1..H-P7 seed-paired "
        "contrasts and print a verdict table.",
    )
    parser.add_argument(
        "--metrics", metavar="PATH",
        help="Per-(arm, seed) metric table (CSV long/wide or JSON). "
        "Columns/keys: arm, seed, metric, value (or wide metric columns).",
    )
    parser.add_argument(
        "--hp6-min", type=float, default=None,
        help="H-P6i minimum byte-saving effect = μ̄(arm 3)/2 (pp of byte "
        "fraction), supplied from the μ̄ extraction (§4.4). Without it "
        "H-P6i uses a 0.0 floor (significance only).",
    )
    parser.add_argument(
        "--json-out", metavar="PATH", default=None,
        help="Write the full structured result (every statistic) to JSON.",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run the synthetic-data self-test (DeepCNN-C1 pattern) and exit.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the per-check self-test log (still returns status).",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        result = selftest(verbose=not args.quiet)
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(f"wrote self-test proof to {args.json_out}")
        return 0 if result["all_passed"] else 1

    if not args.metrics:
        parser.error("one of --metrics or --selftest is required")

    try:
        table = load_metric_table(args.metrics)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot load --metrics {args.metrics}: {exc}",
              file=sys.stderr)
        return 2

    arms = sorted(table.keys(), key=lambda a: (len(a), a))
    print(f"Loaded {len(arms)} arms: {arms}")
    for arm in arms:
        seeds = sorted(table[arm].keys())
        print(f"  arm {arm}: seeds {seeds}")

    result = run_hypotheses(table, hp6_min=args.hp6_min)

    print()
    print(format_verdict_table(result))
    print()

    # One-line summary per §5.4 (every resolution stated).
    summary = {v.get("verdict", "?"): 0 for v in result["verdicts"].values()}
    for v in result["verdicts"].values():
        summary[v.get("verdict", "?")] = summary.get(v.get("verdict", "?"), 0) + 1
    print("Resolutions:", ", ".join(f"{k}={n}" for k, n in summary.items()))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        print(f"wrote structured result to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
