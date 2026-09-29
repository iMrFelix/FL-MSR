"""Shared parsing and cost-model helpers for the overnight analysis scripts.

The offline analysis pipeline (``eps_knee_selection``, ``posthoc_cost``,
``offline_opt_gap``) consumes three artifacts that every run directory
produced by ``python -m scripts.run --output-dir <dir>`` contains:

- ``<dir>/results/report.json`` — the monitor's report.  Per-layer arms log
  ``layer_comm_metrics`` (layer name, importance, traffic class, bytes) per
  node per round, which doubles as the logged round manifest ``(u_l, s_l)``;
  the aggregator's entries additionally carry the receiver-side uplink
  telemetry sidecar (proto field ``uplink_telemetry_json``, pre-run fix 1 in
  ``writeup/01-candidate-selection.md`` §3).
- ``<dir>/configs/node-*.yaml`` — the generated per-node configs, used to
  recover roles, ε, and the nominal per-class uplink bandwidths.

Everything here is intentionally defensive about shapes: the telemetry
sidecar is schema-light by design (analysis scripts parse it, the monitor
stores it verbatim), so helpers tolerate both the JSON-string form and an
already-decoded dict, and missing fields degrade to ``None`` rather than
raising mid-analysis.

The affine per-class cost model (gate ruling G6) is

    t_c = alpha_c * n_msgs + bytes / B_hat_c

fitted by least squares over receiver-side arrival intervals.  ``alpha_c``
absorbs per-message overhead (framing, latency, manifest head-of-line) and
``B_hat_c`` is the *effective* drain rate — deliberately fitted rather than
taken from the tc config, because the wire KPI must measure reality, not
the shaping intent (gate ruling G1).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: bytes/s per Mbps (decimal megabits, matching tc rate semantics).
BYTES_PER_S_PER_MBPS = 1e6 / 8.0


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassCost:
    """Affine drain-time model of one traffic class: ``alpha*n + bytes/rate``.

    Attributes:
        alpha_s: Per-message overhead in seconds (>= 0).
        bytes_per_s: Effective drain rate in bytes/second (> 0).
        source: ``"fit"`` when estimated from telemetry, ``"nominal"`` when
            falling back to the configured shaped bandwidth.  Carried so the
            analysis outputs can flag which runs were fitted vs assumed.
    """

    alpha_s: float
    bytes_per_s: float
    source: str = "nominal"

    def drain_time(self, n_msgs: int, n_bytes: float) -> float:
        """Predicted time to push ``n_msgs`` messages totalling ``n_bytes``."""
        if n_msgs <= 0:
            return 0.0
        return self.alpha_s * n_msgs + n_bytes / self.bytes_per_s

    @property
    def mbps(self) -> float:
        """Effective rate expressed back in Mbps (for strategy bandwidth maps)."""
        return self.bytes_per_s / BYTES_PER_S_PER_MBPS


def nominal_costs(bandwidths_mbps: dict[int, float]) -> dict[int, ClassCost]:
    """Build zero-overhead :class:`ClassCost` entries from nominal Mbps values."""
    return {
        cls: ClassCost(
            alpha_s=0.0,
            bytes_per_s=float(mbps) * BYTES_PER_S_PER_MBPS,
            source="nominal",
        )
        for cls, mbps in bandwidths_mbps.items()
    }


def fit_affine_class_costs(
    observations: dict[int, list[tuple[int, int, float]]],
    nominal_bw_mbps: dict[int, float],
) -> dict[int, ClassCost]:
    """Least-squares fit of the affine cost model, one fit per traffic class.

    Args:
        observations: class -> list of ``(n_msgs, n_bytes, t_observed_s)``
            tuples, one per (run, round, source) interval measured at the
            receiver.
        nominal_bw_mbps: class -> configured shaped bandwidth, the fallback
            when a class has too little (or degenerate) data to fit.

    Returns:
        class -> :class:`ClassCost` covering every class present in either
        input.  Fit results are sanity-clamped: a fit producing a negative
        per-message overhead or a non-positive rate is rejected in favour of
        a 1-parameter rate-only fit, and finally the nominal fallback —
        a misfit silently inverting the cost order between classes would be
        an E1-style artifact reproduced offline.
    """
    import numpy as np

    costs: dict[int, ClassCost] = {}
    classes = sorted(set(observations) | set(nominal_bw_mbps))
    for cls in classes:
        obs = [
            (n, b, t)
            for (n, b, t) in observations.get(cls, [])
            if n > 0 and b > 0 and t > 0
        ]
        fitted: ClassCost | None = None

        if len(obs) >= 3:
            a_mat = np.array([[n, b] for (n, b, _) in obs], dtype=float)
            t_vec = np.array([t for (_, _, t) in obs], dtype=float)
            sol, _, rank, _ = np.linalg.lstsq(a_mat, t_vec, rcond=None)
            alpha, inv_rate = float(sol[0]), float(sol[1])
            if rank == 2 and alpha >= 0.0 and inv_rate > 0.0:
                fitted = ClassCost(
                    alpha_s=alpha, bytes_per_s=1.0 / inv_rate, source="fit"
                )

        if fitted is None and len(obs) >= 1:
            # Rate-only fit through the origin: inv_rate = sum(b*t)/sum(b^2).
            num = sum(b * t for (_, b, t) in obs)
            den = sum(b * b for (_, b, _) in obs)
            if den > 0 and num > 0:
                fitted = ClassCost(
                    alpha_s=0.0, bytes_per_s=den / num, source="fit"
                )

        if fitted is None:
            mbps = nominal_bw_mbps.get(cls)
            if mbps is None:
                logger.warning(
                    "Class %d: no usable observations and no nominal "
                    "bandwidth; skipping",
                    cls,
                )
                continue
            fitted = ClassCost(
                alpha_s=0.0,
                bytes_per_s=float(mbps) * BYTES_PER_S_PER_MBPS,
                source="nominal",
            )
        costs[cls] = fitted
    return costs


def eft_makespan(
    layer_sizes: list[int],
    costs: dict[int, ClassCost],
) -> tuple[float, dict[int, tuple[int, int]]]:
    """Size-descending earliest-finish-time assignment under the affine model.

    This is the discrete head-placement rule adopted by the gate (theory
    repair: proportional fill is the fluid optimum only; size-descending EFT
    is <= 1.38x OPT at C=3, Gonzalez-Ibarra-Sahni 1977).

    Args:
        layer_sizes: Byte sizes of the layers to place (head set).
        costs: class -> :class:`ClassCost`.

    Returns:
        ``(makespan_s, per_class)`` where ``per_class`` maps each class to
        its ``(n_msgs, n_bytes)`` load.  Empty input gives makespan 0.
    """
    loads = {cls: 0.0 for cls in costs}
    per_class = {cls: (0, 0) for cls in costs}
    # Descending size; deterministic tie-break on the value itself suffices
    # because equal sizes are interchangeable for the makespan.
    for size in sorted(layer_sizes, reverse=True):
        best_cls = min(
            costs,
            key=lambda c: (loads[c] + costs[c].drain_time(1, size), c),
        )
        loads[best_cls] += costs[best_cls].drain_time(1, size)
        n, b = per_class[best_cls]
        per_class[best_cls] = (n + 1, b + size)
    makespan = max(loads.values()) if loads else 0.0
    return makespan, per_class


def fluid_lower_bound(
    layer_sizes: list[int],
    costs: dict[int, ClassCost],
) -> float:
    """LP/fluid lower bound on the head drain time (per-message overhead
    excluded by construction — this is the pure-bandwidth relaxation).

    ``max(total/sum-of-rates, largest-layer/fastest-rate)``: the first term
    is the divisible-load bound, the second holds because layers are atomic
    (sub-layer striping is rejected for tonight, gate §7), so no layer can
    finish faster than on the fastest pipe alone.
    """
    if not layer_sizes or not costs:
        return 0.0
    agg_rate = sum(c.bytes_per_s for c in costs.values())
    max_rate = max(c.bytes_per_s for c in costs.values())
    return max(sum(layer_sizes) / agg_rate, max(layer_sizes) / max_rate)


def makespan_of_assignment(
    assignment: dict[str, int],
    sizes: dict[str, int],
    head: set[str],
    costs: dict[int, ClassCost],
) -> float:
    """Affine makespan of a concrete head assignment (strategy output)."""
    per_class_n: dict[int, int] = {}
    per_class_b: dict[int, int] = {}
    for name in head:
        cls = assignment[name]
        per_class_n[cls] = per_class_n.get(cls, 0) + 1
        per_class_b[cls] = per_class_b.get(cls, 0) + sizes[name]
    makespan = 0.0
    for cls, n in per_class_n.items():
        cost = costs.get(cls)
        if cost is None:
            logger.warning("Assignment uses class %d with no cost model", cls)
            continue
        makespan = max(makespan, cost.drain_time(n, per_class_b[cls]))
    return makespan


# ---------------------------------------------------------------------------
# Run-directory parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifestRecord:
    """One logged round manifest: the (u_l, s_l) pairs a worker shipped.

    Reconstructed from a worker node's ``layer_comm_metrics``: ``importance``
    is the logged per-layer score and ``bytes_sent`` the on-wire payload
    size, so β(ε) computed from these is the realized byte geometry, not the
    in-memory tensor size.
    """

    run_id: str
    round: int
    source: str
    scores: dict[str, float]
    sizes: dict[str, int]
    classes: dict[str, int] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(self.sizes.values())

    @property
    def total_score(self) -> float:
        return sum(self.scores.values())


def load_report(run_dir: Path) -> dict[str, Any] | None:
    """Load ``<run_dir>/results/report.json``; None when absent/corrupt."""
    path = Path(run_dir) / "results" / "report.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Unreadable report %s: %s", path, exc)
        return None


def iter_round_nodes(report: dict[str, Any]) -> Iterator[tuple[int, str, dict]]:
    """Yield ``(round, node_id, node_entry)`` in deterministic order."""
    for round_entry in report.get("per_round", []):
        round_num = int(round_entry.get("round", -1))
        for node_id in sorted(round_entry.get("nodes", {})):
            yield round_num, node_id, round_entry["nodes"][node_id]


def parse_uplink_telemetry(node_entry: dict[str, Any]) -> dict[str, Any] | None:
    """Decode the receiver-side telemetry sidecar from a node entry.

    Accepts either the raw proto field (``uplink_telemetry_json`` as a JSON
    string) or an already-decoded ``uplink_telemetry`` dict, whichever the
    collector stored.  Returns ``{source_node_id: {...}}`` or None.
    """
    raw = node_entry.get("uplink_telemetry_json")
    if raw is None:
        raw = node_entry.get("uplink_telemetry")
    if raw is None:
        return None
    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Undecodable uplink telemetry: %s", exc)
            return None
    if isinstance(raw, dict) and raw:
        return raw
    return None


def find_aggregator(run_dir: Path) -> str:
    """Resolve the aggregator node id from the generated per-node configs.

    Falls back to ``node-0`` (the fixed FedAvg convention in every committed
    experiment config) when configs are missing.
    """
    configs_dir = Path(run_dir) / "configs"
    for cfg_path in sorted(configs_dir.glob("node-*.yaml")):
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except (yaml.YAMLError, OSError):
            continue
        if cfg.get("role") == "aggregator":
            return str(cfg.get("node_id", cfg_path.stem))
    return "node-0"


def load_node_config(run_dir: Path, node_id: str) -> dict[str, Any] | None:
    """Load ``<run_dir>/configs/<node_id>.yaml`` (generated per-node config)."""
    path = Path(run_dir) / "configs" / f"{node_id}.yaml"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Unreadable node config %s: %s", path, exc)
        return None


def run_epsilon(run_dir: Path, worker: str = "node-1") -> float | None:
    """Recover the run's configured ε from a worker's generated config."""
    cfg = load_node_config(run_dir, worker)
    if not cfg:
        return None
    training = cfg.get("training") or {}
    eps = training.get("epsilon_deadline")
    return float(eps) if eps is not None else None


def nominal_uplink_bandwidths(
    run_dir: Path,
    worker: str = "node-1",
) -> dict[int, float]:
    """Nominal per-class uplink (worker -> aggregator) bandwidths in Mbps.

    Reads the worker's outgoing edge towards the aggregator from its
    generated config.  Unshaped classes (``bandwidth_mbps: null``) are
    encoded as ``float("inf")`` to match the AssignmentStrategy contract.
    """
    cfg = load_node_config(run_dir, worker)
    if not cfg:
        return {}
    agg = find_aggregator(run_dir)
    agg_idx_match = re.search(r"(\d+)$", agg)
    agg_idx = int(agg_idx_match.group(1)) if agg_idx_match else 0

    edges = cfg.get("outgoing_edges") or []
    chosen = None
    for edge in edges:
        if edge.get("dst") == agg_idx:
            chosen = edge
            break
    if chosen is None and edges:
        chosen = edges[0]
    if chosen is None:
        return {}

    bandwidths: dict[int, float] = {}
    for cls, params in (chosen.get("classes") or {}).items():
        mbps = (params or {}).get("bandwidth_mbps")
        bandwidths[int(cls)] = float("inf") if mbps is None else float(mbps)
    return bandwidths


def worker_manifests(
    run_id: str,
    report: dict[str, Any],
    aggregator: str,
) -> list[ManifestRecord]:
    """Extract one :class:`ManifestRecord` per (round, worker) from a report.

    Aggregator entries are skipped: their ``layer_comm_metrics`` describe the
    downlink broadcast (one metric per destination per layer), which is not
    an uplink manifest.
    """
    records: list[ManifestRecord] = []
    for round_num, node_id, entry in iter_round_nodes(report):
        if node_id == aggregator:
            continue
        metrics = entry.get("layer_comm_metrics") or []
        if not metrics:
            continue
        scores: dict[str, float] = {}
        sizes: dict[str, int] = {}
        classes: dict[str, int] = {}
        for lm in metrics:
            name = lm.get("layer_name")
            if not name or name in scores:
                # First occurrence wins: a worker has a single destination,
                # so duplicates would indicate a multi-dest topology where
                # the per-dest copies are identical anyway.
                continue
            scores[name] = float(lm.get("importance", 0.0))
            sizes[name] = int(lm.get("bytes_sent", 0))
            classes[name] = int(lm.get("traffic_class", 0))
        if scores:
            records.append(
                ManifestRecord(
                    run_id=run_id,
                    round=round_num,
                    source=node_id,
                    scores=scores,
                    sizes=sizes,
                    classes=classes,
                )
            )
    return records


#: Telemetry keys renamed by the NT-01/NT-03 clock-domain pass, mapped to
#: their pre-fix spelling.  Reports written before the pass (every campaign
#: on disk) carry the old keys, so every read goes through `_flow_time`.
_LEGACY_FLOW_KEYS = {
    "t_eps_local_receiver_s": "t_eps_local",
    "manifest_arrival_rel_receiver_s": "manifest_arrival_rel",
    "trigger_fire_rel_receiver_s": "trigger_fire_rel",
    "layer_arrivals_rel_receiver_s": "layer_arrivals_rel",
}


def _flow_time(flow: Mapping[str, Any], key: str) -> Any:
    """Read a domain-tagged flow key, falling back to its pre-fix spelling."""
    if key in flow:
        return flow[key]
    legacy = _LEGACY_FLOW_KEYS.get(key)
    return flow.get(legacy) if legacy is not None else None


@dataclass(frozen=True)
class UplinkObservation:
    """One receiver-side (round, source) telemetry record, flattened.

    Field names carry their clock domain (audit NT-03): ``*_receiver_s`` is
    measured entirely on the receiver, ``*_oneway_s`` is a sender-to-receiver
    wire time on the shared host clock.  The two are different quantities and
    must never be pooled or plotted on one axis.  ``*_oneway_s`` is None for
    runs predating the one-way instrumentation.

    The two κ fields are likewise different quantities (audit TRIG-2/BYTE-04):
    ``kappa_realized`` is TOTAL shed mass in every generation of report, and is
    NOT the quantity ε bounds; ``kappa_slip`` is the coverage slippage that is,
    and is None for reports predating the split.  Decomposing a pre-split
    record needs the layer-count fallback in
    ``scripts.analysis_common.kappa_split`` — do not treat a None here as zero.
    """

    run_id: str
    round: int
    source: str
    t_eps_local_receiver_s: float | None
    watchdog_fired: bool
    manifest_arrival_rel_receiver_s: float | None
    layer_arrivals_rel_receiver_s: dict[str, float]
    shed_layers: list[str]
    kappa_realized: float | None
    t_cover_oneway_s: float | None = None
    layer_wire_times_oneway_s: dict[str, float] = field(default_factory=dict)
    kappa_slip: float | None = None


def uplink_observations(
    run_id: str,
    report: dict[str, Any],
    aggregator: str,
) -> list[UplinkObservation]:
    """Flatten the aggregator's telemetry sidecars across rounds."""
    out: list[UplinkObservation] = []
    for round_num, node_id, entry in iter_round_nodes(report):
        if node_id != aggregator:
            continue
        telemetry = parse_uplink_telemetry(entry)
        if not telemetry:
            continue
        for source in sorted(telemetry):
            if source.startswith("_"):
                # Reserved bookkeeping blocks ("_sender", "_aggregation"),
                # not source flows — schema rule in
                # docs/extensions/04-overnight-interfaces.md §2.
                continue
            td = telemetry[source] or {}
            out.append(
                UplinkObservation(
                    run_id=run_id,
                    round=round_num,
                    source=source,
                    t_eps_local_receiver_s=_flow_time(
                        td, "t_eps_local_receiver_s"
                    ),
                    watchdog_fired=bool(td.get("watchdog_fired", False)),
                    manifest_arrival_rel_receiver_s=_flow_time(
                        td, "manifest_arrival_rel_receiver_s"
                    ),
                    layer_arrivals_rel_receiver_s=dict(
                        _flow_time(td, "layer_arrivals_rel_receiver_s") or {}
                    ),
                    shed_layers=list(td.get("shed_layers") or []),
                    kappa_realized=td.get("kappa_realized"),
                    t_cover_oneway_s=td.get("t_cover_oneway_s"),
                    layer_wire_times_oneway_s=dict(
                        td.get("layer_wire_times_oneway_s") or {}
                    ),
                    kappa_slip=td.get("kappa_slip"),
                )
            )
    return out


def collect_cost_observations(
    manifests: list[ManifestRecord],
    telemetry: list[UplinkObservation],
) -> dict[int, list[tuple[int, int, float]]]:
    """Join manifests with receiver arrivals into affine-fit observations.

    For each (round, source) pair present in both inputs, and for each
    traffic class used by that manifest, the observation is::

        (n_msgs, n_bytes, last_arrival - manifest_arrival)

    i.e. the receiver-local interval from manifest arrival to the last layer
    of that class — exactly the wire-drain interval the fitted alpha/B_hat
    must explain (gate ruling G1: receiver-local, no cross-clock).
    """
    by_key = {(m.round, m.source): m for m in manifests}
    observations: dict[int, list[tuple[int, int, float]]] = {}
    for obs in telemetry:
        manifest = by_key.get((obs.round, obs.source))
        if manifest is None or obs.manifest_arrival_rel_receiver_s is None:
            continue
        per_class_layers: dict[int, list[str]] = {}
        for name, cls in manifest.classes.items():
            per_class_layers.setdefault(cls, []).append(name)
        for cls, layers in sorted(per_class_layers.items()):
            arrivals = [
                obs.layer_arrivals_rel_receiver_s[name]
                for name in layers
                if name in obs.layer_arrivals_rel_receiver_s
            ]
            if not arrivals:
                continue
            t_obs = max(arrivals) - obs.manifest_arrival_rel_receiver_s
            if t_obs <= 0:
                continue
            n_bytes = sum(manifest.sizes[name] for name in layers)
            observations.setdefault(cls, []).append(
                (len(arrivals), n_bytes, t_obs)
            )
    return observations


# ---------------------------------------------------------------------------
# Assignment-strategy loading
# ---------------------------------------------------------------------------

#: Modules that may host strategy registrations beyond the registry module
#: itself.  Probed lazily so the analysis scripts keep working no matter
#: which module the coverage-EFT implementation finally lands in.
_STRATEGY_MODULE_CANDIDATES = (
    "src.importance.assignment",
    "src.importance.strategies",
    "src.importance.coverage_eft",
    "src.importance.assignment_strategies",
)


def load_assignment_strategy(name: str, **kwargs):
    """Instantiate a registered AssignmentStrategy by name.

    Imports every known strategy-hosting module first (registration happens
    at import time), then resolves through the canonical registry so the
    analysis scripts run the *same* code path as the engine — selector
    consistency is the entire point of gate ruling G6.

    Raises:
        ValueError: when the name is not registered anywhere (e.g. the
            coverage-EFT module has not been integrated yet).
    """
    import importlib

    for module_name in _STRATEGY_MODULE_CANDIDATES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue
    from src.importance.assignment import make_strategy

    return make_strategy(name, **kwargs)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

_REPEAT_SUFFIX = re.compile(r"_(?:r|s)\d+$")


def repeat_group(run_id: str) -> str:
    """Strip the repeat/seed suffix: ``a01_coveft_x_r2`` -> ``a01_coveft_x``.

    Repeats of an identical config differ only in this suffix (generator
    convention in ``scripts/gen_overnight_specs.py``), which is what makes
    the noise-floor grouping reconstructible from output dirs alone.
    """
    return _REPEAT_SUFFIX.sub("", run_id)


def find_run_dirs(paths: list[str | Path]) -> list[Path]:
    """Expand CLI run arguments into run dirs that contain a report.

    Each argument may be a run dir itself or a parent whose immediate
    children are run dirs.  Sorted for deterministic output ordering.
    """
    runs: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if (path / "results" / "report.json").exists():
            runs.append(path)
            continue
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if (child / "results" / "report.json").exists():
                    runs.append(child)
    return runs
