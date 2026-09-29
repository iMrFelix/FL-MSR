"""Pluggable layer -> traffic-class assignment strategies.

The step-3 evaluation compares several assignment strategies (gap-based
control, byte-balanced, coverage-EFT, cyclic; see
``writeup/01-candidate-selection.md`` §7) under one engine.  The engine must
swap strategies by config name (``training.assignment_strategy``) with zero
code changes, so this module defines:

- :class:`AssignmentResult` — the result contract every strategy returns,
  carrying not just the class map but the head/tail split and the fluid-model
  prediction the analysis scripts compare against measured ``t_ε``.
- :class:`AssignmentStrategy` — the abstract interface.
- a name -> class registry (:func:`register_strategy`,
  :func:`make_strategy`), mirroring ``late_layer_policy.make_policy``.
- :class:`GapBasedStrategy` — an adapter preserving the existing
  :func:`~src.importance.traffic_mapping.assign_traffic_classes` behaviour
  bit-for-bit.  Gap-based is the measured-4.3×-loss naive baseline (design
  doc E2); it stays in the matrix as the control arm, so the adapter must
  not "fix" it.
- :class:`ByteBalancedStrategy` — the makespan (ε = 0) optimum control:
  bytes ∝ B_c, importance-ordered within class, no tail.
- :class:`CoverageEFTStrategy` — the treatment: exact complement-knapsack
  tail + size-descending EFT head + Smith-order transmit queues, with the
  three theory repairs of gate §8 baked in.  Above 20 tail candidates the
  exact enumeration auto-switches to the greedy + quantized-utility-DP
  fallback (Phase-1 plan T2; utility grid 1e-4 of the ε·U budget).
- :class:`StochasticTailStrategy` — coverage-EFT with randomized boundary
  membership (FedLUAR-style starvation control, the competitor to additive
  aging).
- :class:`CyclicStrategy` — network-blind FedPart-style rotation
  (dissemination-pattern attribution control).

Transmit-order convention: strategies encode the intended within-class
transmit order as the **insertion order of ``assignment``** (heads first, in
within-class priority order, then tails) and mirror it per class in
``diagnostics["class_order"]``.  The engine's per-class send queues are FIFO,
so sending in assignment-iteration order realizes the strategy's order on
the wire.

Call contract (important for stateful strategies such as cyclic):
``assign()`` is invoked **exactly once per round per node**, after local
training and before any send.  The returned class map applies to all
destinations of that round.  Strategies that rotate per round (cyclic) may
therefore keep an internal call counter; everything else should be derived
from the arguments alone.

Score semantics (gate ruling G2): ``scores`` are **scheduling** scores.  The
frozen trigger-accounting score (delta-sq-norm) travels separately in
``ImportanceEntry.raw_score``.  It does not order or place anything — but it
DOES constrain the shed set: ``assign()`` takes the trigger scores as a
second argument and meters the ε budget in trigger mass as well (audit
TRIG-5/ML-07), because ε is a statement about the coverage the RECEIVER
enforces.  Metering it in sched units alone made ε mean something different
in every arm whose sched metric differs from the trigger metric.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from src.importance.traffic_mapping import assign_traffic_classes

logger = logging.getLogger(__name__)

#: tc/netem "mbit" convention: 1 Mbps = 1e6 bit/s = 125 000 bytes/s.
_MBPS_TO_BYTES_PER_S = 1e6 / 8.0


@dataclass(frozen=True)
class AssignmentResult:
    """Outcome of one per-round assignment decision.

    Attributes:
        assignment: Layer name -> traffic class index (0 = highest priority).
            Layers *omitted* from the map are not transmitted this round
            (cyclic-style transmission schedules); the manifest builder must
            list only assigned layers.
        head: Layers the strategy expects to arrive before the ε-trigger
            fires (the (1−ε) coverage mass).  For ε = 0 or strategies without
            a head/tail notion this is simply every assigned layer.
        tail: Deadline-exempt layers riding best-effort behind the head.
            ``head`` and ``tail`` partition ``assignment``'s keys.
        predicted_t_eps: The strategy's own fluid-model prediction of t_ε in
            seconds (head bytes over class bandwidths), or None when the
            strategy makes no prediction (control arms).  Logged so analysis
            can compare predicted vs. receiver-measured t_ε per round.
        diagnostics: Free-form, strategy-specific extras (e.g. gap list,
            per-class byte loads, knapsack stats).  Keep values
            JSON-serializable: they end up in logs and run reports.
    """

    assignment: dict[str, int]
    head: set[str]
    tail: set[str]
    predicted_t_eps: float | None = None
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Enforced here rather than at each call site so a buggy strategy
        # fails loudly at construction, not as a silent mis-route later.
        overlap = self.head & self.tail
        if overlap:
            raise ValueError(
                f"head and tail must be disjoint; both contain {sorted(overlap)}"
            )
        assigned = set(self.assignment)
        if (self.head | self.tail) != assigned:
            raise ValueError(
                "head ∪ tail must equal the assigned layer set; "
                f"head ∪ tail = {sorted(self.head | self.tail)}, "
                f"assignment keys = {sorted(assigned)}"
            )
        negative = {n: c for n, c in self.assignment.items() if c < 0}
        if negative:
            raise ValueError(
                f"traffic class indices must be >= 0; got {negative}"
            )


class AssignmentStrategy(ABC):
    """Decides which traffic class each layer travels on this round."""

    @abstractmethod
    def assign(
        self,
        *,
        scores: dict[str, float],
        sizes: dict[str, int],
        bandwidths: dict[int, float],
        epsilon: float,
        must_receive: set[str],
        ages: dict[str, int] | None = None,
        trigger_scores: dict[str, float] | None = None,
    ) -> AssignmentResult:
        """Compute the per-layer traffic-class assignment for one round.

        Args:
            scores: Layer name -> **scheduling** score (G2; possibly aged).
                Non-negative; not normalised.
            sizes: Layer name -> serialized payload size in bytes.  Covers
                at least every key of ``scores``.
            bandwidths: Traffic class index -> shaped bandwidth in Mbps for
                classes ``0 .. C−1``.  Callers encode unshaped classes as
                ``float("inf")``.
            epsilon: The ε-deadline tolerance in [0, 1); 0 means makespan
                (no tail exists).
            must_receive: Layers that block round completion regardless of
                ε.  Strategies must place these in ``head``.
            ages: Layer name -> rounds since the layer last arrived at the
                aggregator (staleness), or None when aging is disabled.
                Aging-aware strategies fold this into effective scores;
                others ignore it.
            trigger_scores: Layer name -> the FROZEN ε-trigger accounting
                score the receiver meters coverage in (manifest
                ``raw_score``), or None when the caller has none.  Never
                orders or places anything (G2); coverage-metering strategies
                use it to keep the shed set inside ε of the mass the
                receiver actually counts (audit TRIG-5), so that ε means the
                same thing in every arm.  Strategies without a coverage
                notion ignore it.

        Returns:
            An :class:`AssignmentResult`; see its docstring for the
            head/tail/omission contract.
        """
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRATEGY_REGISTRY: dict[str, type[AssignmentStrategy]] = {}


def register_strategy(name: str):
    """Class decorator registering an `AssignmentStrategy` under ``name``.

    Names must match the ``training.assignment_strategy`` config literals
    ('gap_based', 'byte_balanced', 'coverage_eft', 'cyclic').  Re-registering
    the *same* class under the same name is a no-op (tolerates module
    re-imports); registering a different class under a taken name raises, so
    two strategies can never silently shadow each other.
    """

    def decorator(cls: type[AssignmentStrategy]) -> type[AssignmentStrategy]:
        existing = _STRATEGY_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Assignment strategy name {name!r} already registered "
                f"to {existing.__name__}"
            )
        _STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


def registered_strategies() -> list[str]:
    """Return the sorted names of all registered strategies."""
    return sorted(_STRATEGY_REGISTRY)


def make_strategy(name: str, **kwargs) -> AssignmentStrategy:
    """Construct an `AssignmentStrategy` by registered name.

    Args:
        name: A registered strategy name (see :func:`registered_strategies`).
        **kwargs: Forwarded to the strategy constructor (e.g. ``cyclic_k``
            for the cyclic strategy, aging parameters for aged variants).

    Raises:
        ValueError: if ``name`` is not registered.
    """
    try:
        cls = _STRATEGY_REGISTRY[name]
    except KeyError:
        registered = ", ".join(registered_strategies())
        raise ValueError(
            f"Unknown assignment strategy {name!r}. Registered: {registered}"
        ) from None
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Control arm: gap-based adapter
# ---------------------------------------------------------------------------

@register_strategy("gap_based")
class GapBasedStrategy(AssignmentStrategy):
    """Adapter over the legacy gap-based mapper — the naive control arm.

    Delegates verbatim to
    :func:`~src.importance.traffic_mapping.assign_traffic_classes` so the
    control arm measures exactly the historical behaviour (design doc E2),
    including its byte-blindness.  Ignores ``sizes``, ``epsilon``,
    ``must_receive`` and ``ages`` by design; the number of classes is taken
    from ``len(bandwidths)``.

    head = every assigned layer, tail = empty: gap-based has no deadline
    awareness, so nothing is declared shed-eligible up front.
    """

    def assign(
        self,
        *,
        scores: dict[str, float],
        sizes: dict[str, int],
        bandwidths: dict[int, float],
        epsilon: float,
        must_receive: set[str],
        ages: dict[str, int] | None = None,
        trigger_scores: dict[str, float] | None = None,
    ) -> AssignmentResult:
        assignment, gaps = assign_traffic_classes(dict(scores), len(bandwidths))
        return AssignmentResult(
            assignment=assignment,
            head=set(assignment),
            tail=set(),
            predicted_t_eps=None,
            diagnostics={"gaps": gaps},
        )


# ---------------------------------------------------------------------------
# Shared fluid-model helpers
# ---------------------------------------------------------------------------

def _drain_seconds(num_bytes: float, bandwidth_mbps: float) -> float | None:
    """Seconds to drain ``num_bytes`` through one class, or None if it can't.

    Unshaped classes (``float('inf')``, interface-doc contract §1.5) drain
    instantly; non-positive bandwidths cannot transmit at all.
    """
    if bandwidth_mbps <= 0:
        return None
    if math.isinf(bandwidth_mbps):
        return 0.0
    return num_bytes / (bandwidth_mbps * _MBPS_TO_BYTES_PER_S)


def _fluid_seconds(
    total_bytes: float, bandwidths: dict[int, float],
) -> float | None:
    """Fluid (perfect-split) drain time of ``total_bytes`` over all classes.

    ``bytes / Σ_c B_c`` — the lower bound any whole-layer assignment is
    compared against.  Returns 0.0 when any class is unshaped (infinite
    aggregate capacity) and None when no class has positive bandwidth.
    """
    total_bw = sum(b for b in bandwidths.values() if b > 0)
    if total_bw <= 0:
        return None
    if math.isinf(total_bw):
        return 0.0
    return total_bytes / (total_bw * _MBPS_TO_BYTES_PER_S)


def _proportional_weights(bandwidths: dict[int, float]) -> dict[int, float]:
    """Per-class byte-share weights for a ``bytes ∝ B_c`` split.

    If any class is unshaped the fluid optimum routes *everything* through
    unshaped pipes: infinite classes share equally, finite classes get a
    zero share.  Non-positive bandwidths always get zero.
    """
    inf_classes = [c for c, b in bandwidths.items() if b > 0 and math.isinf(b)]
    if inf_classes:
        return {c: (1.0 if c in inf_classes else 0.0) for c in bandwidths}
    return {c: (b if b > 0 else 0.0) for c, b in bandwidths.items()}


def _byte_balanced_fill(
    ordered_layers: list[str],
    sizes: dict[str, int],
    bandwidths: dict[int, float],
) -> dict[str, int]:
    """Assign layers (kept in the given transmit order) so bytes_c ∝ B_c.

    Prefix partition: walking the ordered list with a running byte offset,
    each layer goes to the first class whose cumulative byte target the
    layer's start offset has not yet reached.  Class loads therefore match
    their targets up to one layer of overshoot — the whole-layer granularity
    limit that the EFT head placement (and later sub-layer striping) exists
    to beat.

    Degenerate inputs mirror the gap-based adapter: no classes, or no class
    with positive weight, degrades to "everything on the lowest class".
    """
    if not ordered_layers:
        return {}
    classes = sorted(bandwidths)
    if not classes:
        return {name: 0 for name in ordered_layers}

    weights = _proportional_weights(bandwidths)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return {name: classes[0] for name in ordered_layers}

    total_bytes = float(sum(sizes[name] for name in ordered_layers))
    cumulative_targets: list[float] = []
    acc = 0.0
    for c in classes:
        acc += total_bytes * (weights[c] / total_weight)
        cumulative_targets.append(acc)

    assignment: dict[str, int] = {}
    offset = 0.0
    cls_pos = 0
    for name in ordered_layers:
        while (
            cls_pos < len(classes) - 1
            and offset >= cumulative_targets[cls_pos]
        ):
            cls_pos += 1
        assignment[name] = classes[cls_pos]
        offset += sizes[name]
    return assignment


def _density(utility: float, num_bytes: float) -> float:
    """Utility density u/s with the zero-byte convention.

    Zero-byte layers transmit for free: positive utility at zero bytes is
    infinitely dense (always send first), zero utility at zero bytes is
    neutral.
    """
    if num_bytes > 0:
        return utility / num_bytes
    return math.inf if utility > 0 else 0.0


def _eft_place(
    ordered_layers: list[str],
    sizes: dict[str, int],
    bandwidths: dict[int, float],
    classes: list[int],
    loads: dict[int, float],
) -> dict[str, int]:
    """Earliest-finish-time greedy over uniform machines, mutating ``loads``.

    For each layer (callers pass size-descending order, i.e. LPT) pick the
    class minimizing ``load_c + s/B_c``; ties go to the lowest class index
    for determinism.  Classes without positive bandwidth are skipped; if no
    class can transmit, fall back to the lowest class index so the result is
    still a total assignment (the wire will fail loudly elsewhere).
    """
    assignment: dict[str, int] = {}
    for name in ordered_layers:
        best_class: int | None = None
        best_finish = math.inf
        for c in classes:
            secs = _drain_seconds(sizes[name], bandwidths.get(c, 0.0))
            if secs is None:
                continue
            finish = loads[c] + secs
            if finish < best_finish:
                best_class, best_finish = c, finish
        if best_class is None:
            best_class, best_finish = classes[0], loads[classes[0]]
        assignment[name] = best_class
        loads[best_class] = best_finish
    return assignment


# ---------------------------------------------------------------------------
# Complement-knapsack tail selection (gate §8 theory repair 1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ShedBudget:
    """One ε shed constraint: per-candidate utilities and its absolute cap.

    The tail must satisfy EVERY constraint (audit TRIG-5/ML-07).  The
    primary one is metered in scheduling utility — the mechanism's own
    notion of importance, which is what the aging boost acts through — and
    the secondary one in the FROZEN trigger mass the receiver's ε-trigger
    actually enforces.  When the two metrics coincide (every delta-sq-norm
    arm) the constraints are identical and selection is unchanged.

    ``utilities`` is indexed like the candidate list it was built for.
    """

    label: str
    utilities: np.ndarray
    budget: float


def _fits(index: int, budgets: Sequence[_ShedBudget], used: np.ndarray) -> bool:
    """Whether candidate ``index`` still fits every budget given ``used``."""
    return all(
        used[b] + budgets[b].utilities[index] <= budgets[b].budget
        for b in range(len(budgets))
    )


def _enumerate_max_shed(
    names: list[str],
    utilities: np.ndarray,
    byte_sizes: np.ndarray,
    budget: float,
    *,
    extra_budgets: Sequence[_ShedBudget] = (),
) -> set[str]:
    """Exact complement knapsack by exhaustive subset enumeration.

    maximize Σ_{ℓ∈T} s_ℓ  subject to  Σ_{ℓ∈T} u_ℓ ≤ budget, and to every
    additional constraint in ``extra_budgets`` (audit TRIG-5: the ε budget
    must ALSO hold in the receiver's frozen trigger units, or arms whose
    sched metric differs from the trigger metric are not coverage-matched).

    Subset sums are built by iterative doubling (bit ``i`` of a subset index
    selects ``names[i]``), so feasibility and shed-byte mass are evaluated
    vectorized over all 2^n subsets at once: ~16 K subsets at DeepCNN's
    L = 14, ~1 M at the n ≤ 20 cap — milliseconds either way.  Each extra
    constraint adds one more subset-sum vector of the same size.

    Tie-breaking (fully deterministic): max shed bytes, then min shed
    utility (largest coverage margin), then fewest layers, then lowest
    subset index.
    """
    n = len(names)
    utility_sums = np.zeros(1, dtype=np.float64)
    byte_sums = np.zeros(1, dtype=np.float64)
    extra_sums = [np.zeros(1, dtype=np.float64) for _ in extra_budgets]
    for i in range(n):
        utility_sums = np.concatenate([utility_sums, utility_sums + utilities[i]])
        byte_sums = np.concatenate([byte_sums, byte_sums + byte_sizes[i]])
        for k, extra in enumerate(extra_budgets):
            extra_sums[k] = np.concatenate(
                [extra_sums[k], extra_sums[k] + extra.utilities[i]]
            )

    admissible = utility_sums <= budget  # index 0 (∅) always is
    for k, extra in enumerate(extra_budgets):
        admissible &= extra_sums[k] <= extra.budget
    feasible = np.flatnonzero(admissible)
    best_bytes = float(byte_sums[feasible].max())
    candidates = feasible[byte_sums[feasible] >= best_bytes - 1e-9]

    def _preference(idx: np.integer) -> tuple[float, int, int]:
        i = int(idx)
        return (float(utility_sums[i]), i.bit_count(), i)

    best = int(min(candidates, key=_preference))
    return {names[i] for i in range(n) if (best >> i) & 1}


def _greedy_density_shed(
    names: list[str],
    utilities: np.ndarray,
    byte_sizes: np.ndarray,
    budget: float,
    *,
    extra_budgets: Sequence[_ShedBudget] = (),
) -> set[str]:
    """Density-ascending greedy shed — one half of the > 20-layer fallback.

    Walks layers cheapest-utility-per-byte first, shedding any that still
    fits every budget.  Known suboptimal in isolation (gate §8: the density
    prefix is only the LP relaxation; counterexample on record and pinned in
    the tests), so above the exact-enumeration limit it runs as the
    companion of :func:`_quantized_dp_shed` (Phase-1 plan T2): the greedy
    meters *unquantized* utilities, so it can reclaim a budget-boundary shed
    that the DP's conservative ceil-quantization just excluded.  The
    combined picker in ``CoverageEFTStrategy._base_tail`` keeps whichever
    feasible shed carries more bytes.
    Deterministic: density ties break by name.
    """
    order = sorted(
        range(len(names)),
        key=lambda i: (_density(utilities[i], byte_sizes[i]), names[i]),
    )
    all_budgets = [_ShedBudget("primary", utilities, budget), *extra_budgets]
    shed: set[str] = set()
    used = np.zeros(len(all_budgets), dtype=np.float64)
    for i in order:
        if _fits(i, all_budgets, used):
            shed.add(names[i])
            for b, constraint in enumerate(all_budgets):
                used[b] += float(constraint.utilities[i])
    return shed


def _repair_shed(
    shed: set[str],
    names: list[str],
    byte_sizes: np.ndarray,
    budgets: Sequence[_ShedBudget],
) -> set[str]:
    """Drop tail members until ``shed`` fits every budget (audit TRIG-5).

    The quantized DP optimizes ONE utility axis; a second ε constraint (the
    frozen trigger mass) is enforced afterwards by removing the least
    valuable members — fewest shed bytes per unit of freed constrained
    utility — until every constraint holds.  Conservative, never
    over-shedding, and only reachable on the > 20-candidate fallback path
    (the exact enumerator handles all constraints jointly).  Deterministic:
    ratio ties break by name.
    """
    if not shed or not budgets:
        return shed
    index = {name: i for i, name in enumerate(names)}
    kept = set(shed)
    for constraint in budgets:
        used = sum(float(constraint.utilities[index[n]]) for n in kept)
        while used > constraint.budget and kept:
            # _density(a, b) is a/b with the zero-denominator convention, so
            # this ranks members by bytes gained per unit of constrained
            # utility spent — drop the cheapest such member first.
            worst = min(
                kept,
                key=lambda n: (
                    _density(
                        byte_sizes[index[n]],
                        constraint.utilities[index[n]],
                    ),
                    n,
                ),
            )
            used -= float(constraint.utilities[index[worst]])
            kept.discard(worst)
    return kept


def _quantized_dp_shed(
    names: list[str],
    utilities: np.ndarray,
    byte_sizes: np.ndarray,
    budget: float,
    grid_units: int = 10_000,
) -> set[str]:
    """Quantized-utility complement-knapsack DP — the > 20-layer workhorse.

    maximize Σ_{ℓ∈T} s_ℓ  subject to  Σ_{ℓ∈T} u_ℓ ≤ budget,

    solved exactly on a quantized utility axis (Phase-1 plan T2): the grid
    resolution is ``budget / grid_units`` — at the default 10 000 units,
    1e-4 of the ε·U budget.  Utilities are **ceil-quantized**, so a set the
    DP deems feasible satisfies the *exact* constraint too
    (``Σ u ≤ Σ ceil(u/q)·q ≤ grid_units·q = budget``): quantization can only
    make the DP conservative, never breach coverage.  The byte loss vs the
    unquantized optimum is the bytes of whatever fits in the excluded
    boundary sliver, at most ``n`` grid units of utility (≤ n·1e-4 of the
    budget) — the validation criterion (plan T2a: shed-byte delta < 1 %)
    is checked against the logged manifests by
    ``scripts/validate_dp_vs_exact.py``.

    Standard 0/1 knapsack over ``O(n · grid_units)`` states, vectorized over
    the budget axis per item (~0.65 M states at ResNet-20's n = 65 —
    single-digit milliseconds against the plan's 50 ms ceiling).  Shed-set
    reconstruction backtracks a per-item take table; ties prefer *not*
    shedding (matching the exact enumerator's min-utility/fewest-layers
    preference), and the smallest budget index attaining the byte optimum is
    used (min quantized shed utility).  Fully deterministic.

    Byte sizes are integral by the strategy contract; they are rounded to
    the nearest int so the DP compares exact integer byte totals (no fp
    ties).  Zero-utility layers quantize to weight 0 and are taken iff they
    carry bytes — free shed, same as the exact path.
    """
    n = len(names)
    if n == 0 or budget <= 0.0 or grid_units < 1:
        return set()

    resolution = budget / grid_units
    weights = np.ceil(
        np.maximum(np.asarray(utilities, dtype=np.float64), 0.0) / resolution
    ).astype(np.int64)
    values = np.rint(np.asarray(byte_sizes, dtype=np.float64)).astype(np.int64)

    dp = np.zeros(grid_units + 1, dtype=np.int64)
    take = np.zeros((n, grid_units + 1), dtype=bool)
    for i in range(n):
        w_i = int(weights[i])
        v_i = int(values[i])
        if w_i > grid_units:
            continue  # exceeds the whole budget on its own — never sheddable
        if w_i == 0:
            if v_i > 0:
                take[i, :] = True
                dp += v_i
            continue
        candidate = dp[: grid_units + 1 - w_i] + v_i
        improved = candidate > dp[w_i:]
        if improved.any():
            take[i, w_i:] = improved
            np.copyto(dp[w_i:], candidate, where=improved)

    # dp is monotone non-decreasing in the budget index, so dp[-1] is the
    # optimum and argmax of (dp == optimum) is the smallest index attaining
    # it — the minimum quantized shed utility among byte-optimal solutions.
    w_cur = int(np.argmax(dp == dp[-1]))
    shed: set[str] = set()
    for i in range(n - 1, -1, -1):
        if take[i, w_cur]:
            shed.add(names[i])
            w_cur -= int(weights[i])
    return shed


def _fractional_shed_bytes(
    names: list[str],
    utilities: np.ndarray,
    byte_sizes: np.ndarray,
    budget: float,
    *,
    extra_budgets: Sequence[_ShedBudget] = (),
) -> float:
    """Max shed bytes of the LP relaxation (fractional knapsack).

    Upper-bounds every integral shed, so the head fluid bound derived from
    it lower-bounds every achievable t_ε — logged as the integrality-gap
    reference (gate §7: offline OPT-gap diagnostics).

    With several ε constraints (audit TRIG-5) the multi-constraint LP
    optimum is at most each single-constraint LP optimum, so the minimum
    over the per-constraint relaxations is still a valid upper bound — and
    the one that stays a *tight* bound whenever one constraint dominates.
    """
    if extra_budgets:
        return min(
            _fractional_shed_bytes(names, utilities, byte_sizes, budget),
            *(
                _fractional_shed_bytes(
                    names, extra.utilities, byte_sizes, extra.budget,
                )
                for extra in extra_budgets
            ),
        )
    order = sorted(
        range(len(names)),
        key=lambda i: (_density(utilities[i], byte_sizes[i]), names[i]),
    )
    shed_bytes = 0.0
    used = 0.0
    for i in order:
        if used + utilities[i] <= budget:
            shed_bytes += float(byte_sizes[i])
            used += float(utilities[i])
            continue
        remaining = budget - used
        if remaining > 0 and utilities[i] > 0:
            shed_bytes += float(byte_sizes[i]) * (remaining / float(utilities[i]))
        break
    return shed_bytes


# ---------------------------------------------------------------------------
# Byte-balanced control arm
# ---------------------------------------------------------------------------

@register_strategy("byte_balanced")
class ByteBalancedStrategy(AssignmentStrategy):
    """Makespan control: bytes_c ∝ B_c, importance-ordered within class.

    Objective it solves: minimize ε = 0 makespan in the fluid model — at
    full coverage only byte balance matters and importance is irrelevant
    (design doc §2.2 limit regime).  Layers are ranked score-descending
    (alphabetical tie-break, matching the legacy mapper's determinism) and
    prefix-partitioned so class byte loads are proportional to bandwidth;
    the most important layers therefore ride the fastest class and each
    class transmits in importance order.

    This is the **assignment-only control**: head = every assigned layer,
    tail = ∅ regardless of ε (no shedding), so comparing it against
    coverage-EFT isolates what the ε-tail itself buys.  ``epsilon``,
    ``must_receive`` and ``ages`` are ignored by design (``must_receive ⊆
    head`` holds trivially).

    ``predicted_t_eps`` is the fluid makespan ``S/ΣB_c``; the realized
    per-class drain (which differs by up to one layer of overshoot) is in
    ``diagnostics`` for the integrality log.
    """

    def assign(
        self,
        *,
        scores: dict[str, float],
        sizes: dict[str, int],
        bandwidths: dict[int, float],
        epsilon: float,
        must_receive: set[str],
        ages: dict[str, int] | None = None,
        trigger_scores: dict[str, float] | None = None,
    ) -> AssignmentResult:
        order = sorted(scores, key=lambda name: (-scores[name], name))
        assignment = _byte_balanced_fill(order, sizes, bandwidths)

        total_bytes = sum(sizes[name] for name in order)
        class_bytes: dict[int, int] = {}
        for name, cls in assignment.items():
            class_bytes[cls] = class_bytes.get(cls, 0) + sizes[name]

        drains = [
            _drain_seconds(num_bytes, bandwidths.get(cls, 0.0))
            for cls, num_bytes in class_bytes.items()
        ]
        realized = max((d for d in drains if d is not None), default=None)

        return AssignmentResult(
            assignment=assignment,
            head=set(order),
            tail=set(),
            predicted_t_eps=_fluid_seconds(total_bytes, bandwidths),
            diagnostics={
                "class_bytes": {str(c): int(b) for c, b in sorted(class_bytes.items())},
                "total_bytes": int(total_bytes),
                "realized_makespan_s": realized,
                "class_order": _class_order(assignment),
            },
        )


def _class_order(assignment: dict[str, int]) -> dict[str, list[str]]:
    """Per-class transmit order (insertion order of ``assignment``)."""
    order: dict[str, list[str]] = {}
    for name, cls in assignment.items():
        order.setdefault(str(cls), []).append(name)
    return order


# ---------------------------------------------------------------------------
# Coverage-EFT: the treatment arm (design §2.2 + gate §8 theory repairs)
# ---------------------------------------------------------------------------

@register_strategy("coverage_eft")
class CoverageEFTStrategy(AssignmentStrategy):
    """min-t_ε scheduler: exact knapsack tail + EFT head + Smith-order queues.

    Objective it solves: minimize ``t_ε = min{t : U(t) ≥ 1−ε}`` for one
    round's update over C independent shaped pipes, decomposed per the gate
    §8 theory repairs:

    (a) **Tail = exact complement knapsack** (not the density prefix, which
        is only the LP relaxation — counterexample on record): maximize shed
        bytes subject to shed scheduling-utility ≤ ε·U_total, over layers
        outside ``must_receive``.  Solved by exhaustive subset enumeration up
        to ``EXACT_TAIL_LIMIT`` candidate layers (2^14 at DeepCNN — trivial);
        above that, the scale-up fallback (Phase-1 plan T2) auto-selects:
        greedy density + quantized-utility DP (complement knapsack on a
        utility grid of 1e-4 · ε·U, ``DP_GRID_UNITS`` units; exact at
        quantization resolution), keeping whichever feasible shed carries
        more bytes.  Both components respect the exact utility budget, so
        the coverage guarantee is identical to the enumeration path's.
    (b) **Head placement: size-descending earliest-finish-time greedy**
        across classes (``load_c += s/B_c``, pick the class minimizing
        finish).  Proportional fill is only the *fluid* optimum; LPT/EFT on
        uniform machines is within 1.38× of the optimal head makespan at
        C = 3 (Gonzalez–Ibarra–Sahni 1977; gate §8 repair 2).
    (c) **Within-class transmit order: utility density u/s descending**
        (Smith's rule).  Exactly optimal for the coverage-curve area
        ``∫(1−U(t))dt``, NOT for t_ε itself (gate §8 repair 3) — retained
        because the area objective is the right secondary criterion once
        (a)+(b) fix the t_ε-relevant structure.  Tail layers ride behind
        every head layer of their class, EFT-placed on top of the head
        loads, density-ordered among themselves.

    ``predicted_t_eps`` is ``max_c`` head-load finish under the EFT
    placement.  ``diagnostics`` carries realized β, head/tail bytes, the
    head fluid bound, and the LP fluid bound for the integrality-gap log.

    ε budget units (audit TRIG-5/ML-07 — this used to be the "G2 caveat").
    The knapsack is metered in **scheduling** utility, which is what the
    mechanism's own notion of importance (including the aging boost) acts
    through — but ε is a statement about the coverage the RECEIVER enforces,
    and it meters frozen ``raw_score`` mass.  Metering the budget in sched
    units alone made ε mean something different in every arm whose sched
    metric differs: the ``uniform`` control's budget became "ε·L layers", so
    the byte-greedy tail shed the four largest kernels every round and its
    planned head fell below (1−ε) of trigger mass in 88 % of them — the
    round then could not close on the head at all.  The tail therefore
    satisfies BOTH budgets: ``Σ_T u_sched ≤ ε·U_sched`` and
    ``Σ_T u_trigger ≤ ε·U_trigger``, the latter from the ``trigger_scores``
    argument.  When the two metrics coincide (every delta-sq-norm arm) the
    constraints are identical and the selection is bit-identical to before.

    Constructor knobs: ``stochastic_tail``/``tie_window``/``seed`` switch on
    the randomized boundary of :class:`StochasticTailStrategy` (used by the
    engine for ``aging_mode='stochastic_tail'``); ``budget_metric='sched'``
    restores the pre-fix single-budget metering for reproducing pre-fix
    campaigns; ``exact_tail_limit`` exists for tests.
    """

    #: Exhaustive-enumeration cap: 2^20 vectorized subset sums ≈ 8 MB and
    #: milliseconds; beyond this the greedy + quantized-DP fallback is
    #: auto-selected (plan T2: "auto-selected at L > 20").
    EXACT_TAIL_LIMIT = 20

    #: Utility-grid units of the DP fallback: resolution = budget/units =
    #: 1e-4 of the ε·U budget (plan T2 literal).  O(n · units) states.
    DP_GRID_UNITS = 10_000

    #: Relative feasibility tolerance on the utility budget, guarding
    #: boundary-exact subsets against floating-point exclusion (the bias is
    #: acceptance-side and bounded by 1e-9 of total utility).
    BUDGET_RTOL = 1e-9

    def __init__(
        self,
        *,
        stochastic_tail: bool = False,
        tie_window: float = 0.05,
        seed: int | None = None,
        exact_tail_limit: int | None = None,
        budget_metric: str = "trigger",
    ) -> None:
        if budget_metric not in ("trigger", "sched"):
            raise ValueError(
                f"Unknown budget_metric {budget_metric!r}. "
                f"Valid values: trigger, sched"
            )
        self._stochastic = bool(stochastic_tail)
        self._tie_window = float(tie_window)
        self._rng = np.random.default_rng(seed)
        self._exact_tail_limit = (
            self.EXACT_TAIL_LIMIT if exact_tail_limit is None
            else int(exact_tail_limit)
        )
        self._budget_metric = str(budget_metric)

    # -- tail selection ----------------------------------------------------

    def _base_tail(
        self,
        candidates: list[str],
        utilities: np.ndarray,
        byte_sizes: np.ndarray,
        budget: float,
        extra_budgets: Sequence[_ShedBudget] = (),
    ) -> tuple[set[str], str, dict]:
        """Deterministic tail selection: ``(tail, method, diagnostics)``.

        At or below ``exact_tail_limit`` candidates: exhaustive enumeration,
        which enforces ``extra_budgets`` (the frozen trigger-mass ε budget,
        audit TRIG-5) jointly with the primary one.

        Above it (plan T2 auto-selection): the quantized-utility DP and the
        greedy-density shed both run; the byte-heavier feasible shed wins
        (byte tie → smaller shed utility, full tie → DP).  The greedy can
        win only at budget-boundary cases its exact utility arithmetic
        admits but ceil-quantization excluded, so the combined result is
        never worse than either component.  The DP optimizes one utility
        axis at a time, so it runs once per constraint and each run is
        repaired against the others (:func:`_repair_shed`) before the
        byte-comparison — conservative, never over-shedding.
        """
        if len(candidates) <= self._exact_tail_limit:
            tail = _enumerate_max_shed(
                candidates, utilities, byte_sizes, budget,
                extra_budgets=extra_budgets,
            )
            return tail, "exact_enumeration", {}
        logger.debug(
            "coverage_eft: %d tail candidates exceed the exact-enumeration "
            "limit (%d); auto-selecting the greedy + quantized-utility DP "
            "fallback (utility grid %d units, plan T2)",
            len(candidates), self._exact_tail_limit, self.DP_GRID_UNITS,
        )
        index = {name: i for i, name in enumerate(candidates)}

        def _bytes_of(tail: set[str]) -> float:
            return float(sum(byte_sizes[index[name]] for name in tail))

        all_budgets = [_ShedBudget("primary", utilities, budget), *extra_budgets]
        dp_tail: set[str] = set()
        for axis, constraint in enumerate(all_budgets):
            others = [b for i, b in enumerate(all_budgets) if i != axis]
            axis_tail = _repair_shed(
                _quantized_dp_shed(
                    candidates, constraint.utilities, byte_sizes,
                    constraint.budget, grid_units=self.DP_GRID_UNITS,
                ),
                candidates, byte_sizes, others,
            )
            if _bytes_of(axis_tail) > _bytes_of(dp_tail):
                dp_tail = axis_tail
        greedy_tail = _greedy_density_shed(
            candidates, utilities, byte_sizes, budget,
            extra_budgets=extra_budgets,
        )

        def _shed_key(tail: set[str]) -> tuple[float, float]:
            # Lexicographic preference: more bytes, then less utility.
            return (
                -float(sum(byte_sizes[index[name]] for name in tail)),
                float(sum(utilities[index[name]] for name in tail)),
            )

        if _shed_key(greedy_tail) < _shed_key(dp_tail):
            tail, winner = greedy_tail, "greedy"
        else:
            tail, winner = dp_tail, "dp"
        info = {
            "dp_shed_bytes": int(sum(byte_sizes[index[n]] for n in dp_tail)),
            "greedy_shed_bytes": int(
                sum(byte_sizes[index[n]] for n in greedy_tail)
            ),
            "fallback_winner": winner,
            "dp_grid_units": self.DP_GRID_UNITS,
        }
        return tail, "quantized_dp_fallback", info

    def _stochastic_boundary_tail(
        self,
        candidates: list[str],
        utilities: np.ndarray,
        byte_sizes: np.ndarray,
        budget: float,
        extra_budgets: Sequence[_ShedBudget] = (),
    ) -> tuple[set[str], str, dict]:
        """Randomize tail membership among near-tied densities.

        The deterministic tail's head/tail boundary sits at some density;
        layers whose density is within ``tie_window`` (relative) of the
        boundary are near-indistinguishable in utility-per-byte, so which of
        them sheds is an arbitrary choice the deterministic solver would
        repeat every round — the starvation mechanism this arm exists to
        break (FedLUAR-style randomization, the competitor to additive
        aging).  Membership among the near-tied pool is therefore sampled,
        while layers shed far from the boundary stay shed (``firm`` set).

        Every utility budget is re-enforced during sampling — the scheduling
        one and, since audit TRIG-5, the frozen trigger-mass one — so the
        planned head always covers ≥ (1−ε) in both metrics; randomization
        trades shed *bytes*, never coverage.
        """
        base_tail, base_method, base_info = self._base_tail(
            candidates, utilities, byte_sizes, budget, extra_budgets,
        )
        diag: dict = {"base_tail_method": base_method, **base_info}
        if not base_tail:
            return base_tail, "stochastic_boundary", diag

        density = {
            name: _density(float(utilities[i]), float(byte_sizes[i]))
            for i, name in enumerate(candidates)
        }
        boundary = max(density[name] for name in base_tail)
        if math.isinf(boundary):
            pool = {n for n in candidates if math.isinf(density[n])}
        elif boundary <= 0.0:
            pool = {n for n in candidates if density[n] <= 0.0}
        else:
            low = boundary * (1.0 - self._tie_window)
            high = boundary * (1.0 + self._tie_window)
            pool = {n for n in candidates if low <= density[n] <= high}

        index = {name: i for i, name in enumerate(candidates)}
        all_budgets = [_ShedBudget("primary", utilities, budget), *extra_budgets]
        firm = base_tail - pool
        tail = set(firm)
        used = np.array(
            [
                sum(float(b.utilities[index[name]]) for name in firm)
                for b in all_budgets
            ],
            dtype=np.float64,
        )
        for name in self._rng.permutation(sorted(pool)):
            name = str(name)
            if _fits(index[name], all_budgets, used):
                tail.add(name)
                for b, constraint in enumerate(all_budgets):
                    used[b] += float(constraint.utilities[index[name]])

        diag.update({
            "boundary_density": float(boundary),
            "tie_pool": sorted(pool),
            "firm_tail": sorted(firm),
        })
        return tail, "stochastic_boundary", diag

    # -- main entry ----------------------------------------------------------

    def assign(
        self,
        *,
        scores: dict[str, float],
        sizes: dict[str, int],
        bandwidths: dict[int, float],
        epsilon: float,
        must_receive: set[str],
        ages: dict[str, int] | None = None,
        trigger_scores: dict[str, float] | None = None,
    ) -> AssignmentResult:
        layers = list(scores)
        if not layers:
            return AssignmentResult(
                assignment={}, head=set(), tail=set(),
                predicted_t_eps=None, diagnostics={"tail_method": "empty"},
            )

        must_eff = set(must_receive) & set(layers)
        utility_total = float(sum(scores.values()))
        budget = epsilon * utility_total
        budget_tol = budget + self.BUDGET_RTOL * max(1.0, abs(utility_total))
        # Frozen trigger mass: the units the RECEIVER meters ε in (audit
        # TRIG-5).  Only a live second constraint when the sched metric
        # actually differs; identical scores leave selection untouched, and a
        # trigger dict that does not cover every layer is ignored (the
        # engine always passes a complete one; direct callers may not).
        trigger_total = 0.0
        trigger_active = False
        if (
            self._budget_metric == "trigger"
            and trigger_scores is not None
            and all(name in trigger_scores for name in layers)
            and any(trigger_scores[name] != scores[name] for name in layers)
        ):
            trigger_total = float(sum(trigger_scores[name] for name in layers))
            trigger_active = trigger_total > 0.0

        # --- (a) tail selection --------------------------------------------
        candidates = sorted(name for name in layers if name not in must_eff)
        extra_budgets: list[_ShedBudget] = []
        if trigger_active and candidates:
            trigger_budget = epsilon * trigger_total
            extra_budgets.append(_ShedBudget(
                "trigger",
                np.array(
                    [trigger_scores[name] for name in candidates],
                    dtype=np.float64,
                ),
                trigger_budget + self.BUDGET_RTOL * max(1.0, trigger_total),
            ))
        tail_diag: dict = {}
        lp_shed_bytes = 0.0
        if epsilon <= 0.0 or not candidates:
            tail: set[str] = set()
            tail_method = "no_tail"
        elif utility_total <= 0.0:
            # Degenerate all-zero scores: any subset is "free" to shed, so
            # the byte-maximizing tail would shed every candidate.  Treat as
            # ε = 0 instead and let the engine's degenerate-manifest guard
            # (pre-run fix 6) own the loud failure.
            tail = set()
            tail_method = "degenerate_scores_no_tail"
            logger.warning(
                "coverage_eft: total scheduling utility is 0; refusing to "
                "shed (degenerate manifest, see pre-run fix 6)"
            )
        else:
            utilities = np.array(
                [scores[name] for name in candidates], dtype=np.float64,
            )
            byte_sizes = np.array(
                [float(sizes[name]) for name in candidates], dtype=np.float64,
            )
            if self._stochastic:
                tail, tail_method, tail_diag = self._stochastic_boundary_tail(
                    candidates, utilities, byte_sizes, budget_tol,
                    extra_budgets,
                )
            else:
                tail, tail_method, tail_diag = self._base_tail(
                    candidates, utilities, byte_sizes, budget_tol,
                    extra_budgets,
                )
            lp_shed_bytes = _fractional_shed_bytes(
                candidates, utilities, byte_sizes, budget_tol,
                extra_budgets=extra_budgets,
            )

        # --- (b) head: size-descending EFT across classes -------------------
        head = [name for name in layers if name not in tail]
        classes = sorted(bandwidths) or [0]
        head_loads = {c: 0.0 for c in classes}
        head_assignment = _eft_place(
            sorted(head, key=lambda n: (-sizes[n], n)),
            sizes, bandwidths, classes, head_loads,
        )
        has_capacity = any(b > 0 for b in bandwidths.values())
        predicted_t_eps = (
            max(head_loads.values()) if (has_capacity and head_loads) else None
        )

        # Tail rides behind: EFT on top of the head loads.  Per-class FIFO
        # order puts every head layer before every tail layer of its class,
        # so tail placement can never delay the head.
        tail_loads = dict(head_loads)
        tail_assignment = _eft_place(
            sorted(tail, key=lambda n: (-sizes[n], n)),
            sizes, bandwidths, classes, tail_loads,
        )

        # --- (c) transmit order: density-descending (Smith) -----------------
        def _smith_key(name: str) -> tuple[float, str]:
            return (-_density(scores[name], sizes[name]), name)

        assignment: dict[str, int] = {}
        for name in sorted(head, key=_smith_key):
            assignment[name] = head_assignment[name]
        for name in sorted(tail, key=_smith_key):
            assignment[name] = tail_assignment[name]

        head_bytes = sum(sizes[name] for name in head)
        tail_bytes = sum(sizes[name] for name in tail)
        total_bytes = head_bytes + tail_bytes
        shed_utility = float(sum(scores[name] for name in tail))

        coverage_sched = (
            1.0 - shed_utility / utility_total if utility_total > 0 else 1.0
        )
        diagnostics = {
            "tail_method": tail_method,
            "head_bytes": int(head_bytes),
            "tail_bytes": int(tail_bytes),
            "beta_realized": (tail_bytes / total_bytes) if total_bytes else 0.0,
            "utility_total": utility_total,
            "utility_budget": budget,
            "shed_utility": shed_utility,
            # Planned coverage in SCHED units (the strategy's own objective).
            # `coverage_planned` below is the receiver-enforced one and is
            # the number every cross-arm comparison must use (audit TRIG-5:
            # the uniform arm reported a constant 0.7143 here while its head
            # carried a mean 0.509 of the trigger mass the receiver meters).
            "coverage_planned_sched": coverage_sched,
            "budget_metric": self._budget_metric,
            "fluid_bound_head_s": _fluid_seconds(head_bytes, bandwidths),
            "lp_fluid_bound_s": _fluid_seconds(
                total_bytes - lp_shed_bytes, bandwidths,
            ),
            "eft_head_loads_s": {
                str(c): float(load) for c, load in sorted(head_loads.items())
            },
            "class_order": _class_order(assignment),
            **tail_diag,
        }
        # Coverage the RECEIVER will see: trigger mass of the planned head
        # over total trigger mass.  Equals the sched figure whenever the two
        # metrics coincide, so single-metric arms are unaffected.
        if trigger_scores is not None and all(n in trigger_scores for n in layers):
            trigger_sum = float(sum(trigger_scores[name] for name in layers))
            shed_trigger = float(sum(trigger_scores[name] for name in tail))
            diagnostics["shed_trigger_mass"] = shed_trigger
            diagnostics["trigger_total"] = trigger_sum
            diagnostics["coverage_planned"] = (
                1.0 - shed_trigger / trigger_sum if trigger_sum > 0 else 1.0
            )
        else:
            diagnostics["coverage_planned"] = coverage_sched

        return AssignmentResult(
            assignment=assignment,
            head=set(head),
            tail=set(tail),
            predicted_t_eps=predicted_t_eps,
            diagnostics=diagnostics,
        )


class StochasticTailStrategy(CoverageEFTStrategy):
    """Coverage-EFT with randomized head/tail boundary membership.

    FedLUAR-style starvation control, the competitor to additive aging
    (gate §7): instead of boosting starved layers, the *choice* of which
    near-tied layer sheds is re-sampled every round, so no layer is shed
    deterministically forever.  Near-tied = scheduling density within
    ``tie_window`` (default 5 %, relative) of the deterministic boundary.
    Realized planned coverage ≥ (1−ε) is enforced during sampling — see
    :meth:`CoverageEFTStrategy._stochastic_boundary_tail`.

    Not separately registered: the schema selects this arm as
    ``assignment_strategy='coverage_eft'`` + ``aging_mode='stochastic_tail'``,
    which the engine maps to ``make_strategy('coverage_eft',
    stochastic_tail=True, seed=<node seed>)`` (interface-doc contract §1.6
    keeps registry names == config literals).  This subclass is the same
    object with the flag pre-set, for direct construction and tests.
    """

    def __init__(
        self,
        *,
        tie_window: float = 0.05,
        seed: int | None = None,
        exact_tail_limit: int | None = None,
    ) -> None:
        super().__init__(
            stochastic_tail=True,
            tie_window=tie_window,
            seed=seed,
            exact_tail_limit=exact_tail_limit,
        )


# ---------------------------------------------------------------------------
# Cyclic control arm (network-blind FedPart-style rotation)
# ---------------------------------------------------------------------------

@register_strategy("cyclic")
class CyclicStrategy(AssignmentStrategy):
    """Network-blind rotation: k layers per round, depth-order round-robin.

    The dissemination-pattern attribution control (gate §7): it shares the
    *partial-update* trait with coverage scheduling while ignoring scores,
    sizes-vs-bandwidth structure, and ε alike — so any gap between this arm
    and coverage-EFT is attributable to importance/network awareness, not to
    sending fewer bytes.  (FedPart qualification, gate §5.3: we emulate only
    the dissemination pattern; FedPart additionally freezes local training,
    a compute-side mechanism out of scope here.)

    Round r transmits the ``k = cyclic_k`` layers at rotation positions
    ``(r·k + i) mod L`` in depth order (the engine passes scores in model
    order; dict insertion order carries it).  All other layers are **omitted**
    — not transmitted at all (interface-doc omission semantics §1.3), which
    is what makes this FedPart-like rather than a priority scheme; the
    omitted set is recorded in ``diagnostics``.  ``must_receive`` layers
    (e.g. τ_max promotions) are force-included beyond the k so the
    ``must_receive ⊆ head`` contract can never break, and the manifest stays
    truthful.  Scores are ignored by design; placement of the selected
    layers is byte-balanced (bytes ∝ B_c) in rotation order, and the state
    is the internal round counter (``assign()`` is called exactly once per
    round per node — interface-doc contract §1.2).
    """

    def __init__(self, *, cyclic_k: int) -> None:
        if int(cyclic_k) < 1:
            raise ValueError(
                f"cyclic strategy requires cyclic_k >= 1 (got {cyclic_k})"
            )
        self._k = int(cyclic_k)
        self._round_index = 0

    def assign(
        self,
        *,
        scores: dict[str, float],
        sizes: dict[str, int],
        bandwidths: dict[int, float],
        epsilon: float,
        must_receive: set[str],
        ages: dict[str, int] | None = None,
        trigger_scores: dict[str, float] | None = None,
    ) -> AssignmentResult:
        round_index = self._round_index
        self._round_index += 1

        depth_order = list(scores)
        num_layers = len(depth_order)
        if num_layers == 0:
            return AssignmentResult(
                assignment={}, head=set(), tail=set(),
                predicted_t_eps=None,
                diagnostics={"round_index": round_index},
            )

        k = min(self._k, num_layers)
        start = (round_index * k) % num_layers
        selected = [depth_order[(start + i) % num_layers] for i in range(k)]

        must_eff = set(must_receive) & set(depth_order)
        forced = [
            name for name in depth_order
            if name in must_eff and name not in set(selected)
        ]
        transmit = selected + forced

        assignment = _byte_balanced_fill(transmit, sizes, bandwidths)
        transmit_bytes = sum(sizes[name] for name in transmit)
        omitted = [name for name in depth_order if name not in assignment]

        return AssignmentResult(
            assignment=assignment,
            head=set(transmit),
            tail=set(),
            predicted_t_eps=_fluid_seconds(transmit_bytes, bandwidths),
            diagnostics={
                "round_index": round_index,
                "selected": list(selected),
                "forced_must_receive": forced,
                "omitted_layers": omitted,
                "transmit_bytes": int(transmit_bytes),
                "class_order": _class_order(assignment),
            },
        )
