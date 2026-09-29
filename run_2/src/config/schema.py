"""Pydantic v2 models for validating the federated learning experiment YAML config."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Traffic / Topology
# ---------------------------------------------------------------------------

class TrafficClassParams(BaseModel):
    """QoS parameters for a single traffic class on an edge."""

    bandwidth_mbps: float | None = Field(
        default=None,
        description="Bandwidth limit in Mbps. None means unlimited.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="One-way latency in milliseconds.",
    )
    drop_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Packet drop probability in [0, 1].",
    )


class TopologyEdge(BaseModel):
    """A directed edge between two nodes with per-class QoS parameters."""

    src: int = Field(..., ge=0, description="Source node ID.")
    dst: int = Field(..., ge=0, description="Destination node ID.")
    classes: dict[int, TrafficClassParams] = Field(
        ...,
        description="Mapping from traffic-class index to QoS parameters.",
    )


class TopologyConfig(BaseModel):
    """Topology of the federation expressed as a list of directed edges."""

    edges: list[TopologyEdge] = Field(
        ...,
        min_length=1,
        description="Directed edges with per-class QoS parameters.",
    )


class FederationConfig(BaseModel):
    """Top-level federation section."""

    num_nodes: int = Field(..., ge=1, description="Total number of nodes.")
    roles: dict[int, str] = Field(
        default_factory=dict,
        description=(
            "Optional mapping from node_id -> role string "
            '(e.g. "aggregator", "worker"). '
            "Nodes not listed default to 'worker'."
        ),
    )
    topology: TopologyConfig


# ---------------------------------------------------------------------------
# Traffic classes
# ---------------------------------------------------------------------------

class TrafficClassesConfig(BaseModel):
    """Global traffic class definitions and DSCP mappings."""

    num_classes: int = Field(..., ge=1, description="Number of traffic classes.")
    dscp_mapping: dict[int, int] = Field(
        ...,
        description="Mapping from class index -> DSCP value.",
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class DatasetPartitionConfig(BaseModel):
    """Controls how the dataset is partitioned across nodes."""

    strategy: Literal["iid", "dirichlet", "pathological", "natural"] = Field(
        ...,
        description=(
            "Partitioning strategy.  'natural' assigns the dataset's "
            "intrinsic units to nodes (FEMNIST: whole writers — the "
            "canonical LEAF non-IID partition); only datasets with a "
            "natural keying support it."
        ),
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    alpha: float | None = Field(
        default=None,
        description="Dirichlet concentration parameter (required for 'dirichlet').",
    )
    classes_per_node: int | None = Field(
        default=None,
        description="Classes assigned per node (required for 'pathological').",
    )
    max_writers: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional writer-subsampling cap for naturally-keyed datasets "
            "(FEMNIST: keep a seeded subsample of N writers, LEAF "
            "'small'-version style).  None = use all writers.  Ignored by "
            "datasets without natural keying."
        ),
    )

    workers_only: bool = Field(
        default=False,
        description=(
            "Partition training data over worker nodes only "
            "(num_nodes - 1 shards) instead of stranding one shard on the "
            "never-training aggregator (node-0).  The aggregator receives "
            "a small copied sliver for schema compliance; workers jointly "
            "train on 100% of the data.  Baseline fix (writeup/16 §5.3)."
        ),
    )

    @model_validator(mode="after")
    def _check_strategy_params(self) -> DatasetPartitionConfig:
        if self.strategy == "dirichlet" and self.alpha is None:
            raise ValueError(
                "'alpha' must be set when strategy is 'dirichlet'."
            )
        if self.strategy == "pathological" and self.classes_per_node is None:
            raise ValueError(
                "'classes_per_node' must be set when strategy is 'pathological'."
            )
        return self


class DatasetConfig(BaseModel):
    """Dataset selection and partitioning."""

    name: str = Field(..., description="Dataset name (e.g. 'mnist', 'cifar10').")
    partition: DatasetPartitionConfig


class TrainingConfig(BaseModel):
    """Hyper-parameters and algorithm selection."""

    dataset: DatasetConfig
    model: str = Field(..., description="Model architecture name (e.g. 'mlp', 'cnn').")
    algorithm: str = Field(
        ...,
        description="FL algorithm. Currently supported: 'dpsgd', 'adpsgd', 'gossip_sgd', 'fedavg'.",
    )
    epochs_per_round: int = Field(..., ge=1)
    total_rounds: int = Field(..., ge=1)
    batch_size: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0.0)
    optimizer: str = Field(..., description="Optimizer name ('sgd', 'adam', ...).")
    momentum: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="SGD momentum coefficient. Ignored by adam.",
    )
    clipnorm: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Per-gradient-tensor L2 clipping norm passed to the Keras "
            "optimizer.  None disables clipping (legacy behaviour).  "
            "Baseline fix for the exploding-gradient collapses "
            "(writeup/16 §5.1)."
        ),
    )
    lr_schedule: Literal["constant", "cosine", "step"] = Field(
        default="constant",
        description=(
            "Round-indexed learning-rate schedule applied at the start of "
            "every round on every node.  'constant' is the legacy "
            "behaviour; 'cosine' anneals base_lr -> 0 over total_rounds; "
            "'step' multiplies by lr_decay_factor every lr_decay_every "
            "rounds.  Baseline fix (writeup/16 §5.2)."
        ),
    )
    lr_decay_factor: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description=(
            "Multiplicative decay per step for lr_schedule='step' "
            "(lr = base * factor^(round // every)).  Ignored otherwise."
        ),
    )
    lr_decay_every: int = Field(
        default=0,
        ge=0,
        description=(
            "Round interval between step decays for lr_schedule='step'.  "
            "Must be >= 1 in step mode; ignored by the other schedules."
        ),
    )
    lr_eta_min: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Absolute learning-rate floor for the non-constant schedules "
            "(audit ML-04/ML-03).  None (the default) means 0.01 * "
            "learning_rate: without a floor the cosine schedule reaches "
            "EXACTLY 0 on round total_rounds-1 (cos(pi) == -1.0), so the "
            "final round ships an all-zero delta — degenerate manifest, "
            "epsilon-trigger disabled, full model transmitted — in the very "
            "round endpoint accuracy and steady-state bytes are read from.  "
            "Cosine anneals to lr_eta_min instead of 0; 'step' is clamped "
            "from below by it.  Set 0.0 to reproduce pre-fix campaigns "
            "exactly; ignored by lr_schedule='constant' (legacy-exact)."
        ),
    )
    server_momentum: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "FedAvgM server momentum beta (Hsu et al. 2019, "
            "arXiv:1909.06335): the aggregator applies "
            "v_t = beta*v_{t-1} + delta_fresh_t; "
            "theta_t = theta_{t-1} + delta_fill_t + v_t.  Only the "
            "fresh-arrival component of the round's motion enters the "
            "velocity — slippage fills (recycled or stale) pass through "
            "unaccelerated, or the momentum and the fill compound at a "
            "per-round gain of beta+f and shed layers blow up (audit "
            "ML-02).  0.0 keeps plain FedAvg bit-identical.  The largest "
            "known accuracy lever at Dirichlet(0.1) (writeup/16 §1)."
        ),
    )
    update_mode: Literal["monolithic", "per_layer"] = Field(
        default="monolithic",
        description="Whether to treat model updates as a single blob or per-layer.",
    )
    importance_metric: str | None = Field(
        default=None,
        description=(
            "Optional importance metric for prioritised communication "
            "('gradient_norm', 'uniform', or None)."
        ),
    )
    staleness_threshold: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum iteration gap for accepting stale updates in async "
            "algorithms (A-DPSGD, Gossip-SGD).  0 means accept all updates "
            "regardless of staleness.  Ignored by synchronous algorithms."
        ),
    )
    epsilon_deadline: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Importance-coverage tolerance for round completion in per-layer "
            "mode.  A round completes once (1 - epsilon_deadline) of total "
            "raw importance has been received from a sender (subject to "
            "must_receive flags in the manifest).  epsilon_deadline=0 means "
            "wait for every layer (synchronous behaviour, the default).  "
            "Ignored in monolithic mode.  See "
            "docs/extensions/02-epsilon-trigger.md."
        ),
    )
    late_layer_policy: Literal["drop", "recycle_last_delta", "renormalize"] = Field(
        default="drop",
        description=(
            "How to handle layers that arrive after the round was completed "
            "by the epsilon-deadline trigger.  'drop' discards (control); "
            "'recycle_last_delta' re-applies the layer's previous aggregated "
            "delta; 'renormalize' re-weights the aggregate over arrived "
            "mass only.  Gate ruling G3 in "
            "writeup/01-candidate-selection.md; see "
            "docs/extensions/03-late-layer-policy.md."
        ),
    )
    eval_every: int = Field(
        default=1,
        ge=1,
        description=(
            "Evaluate validation metrics every N iterations in async mode.  "
            "Reduces overhead when iterations are fast.  The last iteration "
            "always evaluates regardless.  Ignored in synchronous mode."
        ),
    )
    global_eval: bool = Field(
        default=False,
        description=(
            "K0 (writeup/12 §3): when True, evaluate val metrics on the "
            "shared global held-out test set (identical across all nodes, "
            "already shipped in every partition's .npz) instead of the "
            "per-node local validation split.  The local split under a "
            "skewed Dirichlet partition is tiny and non-representative, "
            "driving ~4pp cross-seed accuracy noise that swamps the "
            "non-inferiority margin; the balanced global set certifies "
            "accuracy at low variance.  Reported through the existing "
            "val_accuracy field, so no downstream analysis change is needed. "
            "Default False preserves the campaigns/p1 local-val behaviour."
        ),
    )

    # ------------------------------------------------------------------
    # Overnight-evaluation knobs (writeup/01-candidate-selection.md §7).
    # All defaults preserve pre-existing behaviour so committed configs and
    # the ε=0 bug-detector arm keep running unchanged.
    # ------------------------------------------------------------------
    importance_metric_v2: Literal[
        "raw_norm", "delta_sq_norm", "relative", "snr_reweight", "uniform"
    ] = Field(
        default="raw_norm",
        description=(
            "Scheduling-score metric computed on the shipped multi-epoch "
            "delta (NOT the last minibatch gradient).  Feeds "
            "ImportanceEntry.sched_score and the assignment strategy only; "
            "the ε-trigger accounting stays frozen on delta-sq-norm for all "
            "arms (gate ruling G2).  'raw_norm' is the degenerate-control "
            "default matching the legacy gradient-norm behaviour."
        ),
    )
    assignment_strategy: Literal[
        "gap_based", "byte_balanced", "coverage_eft", "cyclic"
    ] = Field(
        default="gap_based",
        description=(
            "Layer -> traffic-class assignment strategy "
            "(src/importance/assignment.py registry).  'gap_based' is the "
            "naive control preserving the legacy mapper; 'byte_balanced' is "
            "the makespan/ε=0 optimum; 'coverage_eft' minimises t_ε "
            "(density order + EFT head + complement-knapsack tail); "
            "'cyclic' is the network-blind FedPart-style attribution "
            "control (requires cyclic_k >= 1)."
        ),
    )
    aging_mode: Literal["none", "additive_capped", "stochastic_tail"] = Field(
        default="none",
        description=(
            "Starvation control for tail layers.  'additive_capped' adds "
            "aging_lambda per starved round to the scheduling score and "
            "hard-caps staleness at aging_tau_max via must_receive "
            "(the multiplicative form is not starvation-safe); "
            "'stochastic_tail' randomises tail membership FedLUAR-style.  "
            "Acts on sched scores only — never on trigger accounting (G2)."
        ),
    )
    aging_lambda: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Additive aging boost per round of staleness for "
            "aging_mode='additive_capped'.  0.0 means no boost."
        ),
    )
    aging_tau_max: int = Field(
        default=0,
        ge=0,
        description=(
            "Hard staleness cap in rounds: a layer aged >= aging_tau_max is "
            "flagged must_receive, blocking round completion until it "
            "arrives.  0 disables the cap."
        ),
    )
    aging_age_basis: Literal["inclusion", "head_placement"] = Field(
        default="inclusion",
        description=(
            "Which event resets a layer's staleness counter (audit "
            "TRIG-1/ML-01).  'inclusion' (default, the fixed semantics): the "
            "aggregator acknowledges — on the skip-advice downlink — which of "
            "the sender's layers actually entered the aggregate, and only an "
            "acknowledged inclusion resets the age, so aging_tau_max bounds "
            "REALIZED staleness.  'head_placement' is the pre-fix behaviour "
            "(reset on the sender's own head placement, which carries no "
            "delivery guarantee: a layer can be head-placed every round and "
            "shed at the receiver every round, and the measured absence runs "
            "then reach 4-20 rounds at tau_max=3).  Keep it only to reproduce "
            "pre-fix campaigns.  Nodes with no ack channel (decentralised "
            "algorithms) fall back to head placement with a loud warning and "
            "record age_basis in the report's _sender block."
        ),
    )
    epsilon_budget_metric: Literal["trigger", "sched"] = Field(
        default="trigger",
        description=(
            "Units the coverage strategy's epsilon shed budget is metered in "
            "(audit TRIG-5/ML-07).  'trigger' (default, the fixed semantics): "
            "the tail must also fit inside epsilon * (FROZEN trigger mass), "
            "the quantity the receiver's epsilon-trigger actually enforces, "
            "so arms with different sched metrics are coverage-matched.  With "
            "the pre-fix 'sched' metering, importance_metric_v2='uniform' "
            "turned the budget into 'epsilon * L layers' and the byte-greedy "
            "tail selector shed the four largest kernels every round (79% of "
            "bytes, planned head below 1-epsilon of trigger mass in 88% of "
            "rounds) — the control changed the shed set and volume, not just "
            "the ordering.  Both budgets are enforced under 'trigger', so a "
            "sched metric (e.g. the aging boost) still shapes the shed set; "
            "when sched == trigger the two constraints coincide and plans are "
            "bit-identical.  Keep 'sched' only to reproduce pre-fix campaigns."
        ),
    )
    cyclic_k: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of layers transmitted per round by the 'cyclic' "
            "assignment strategy (round-robin over the layer list).  "
            "Ignored by other strategies; must be >= 1 when "
            "assignment_strategy='cyclic'."
        ),
    )
    epsilon_warmup_rounds: int = Field(
        default=0,
        ge=0,
        description=(
            "Round-indexed ε schedule (warm-up): rounds 0..N-1 run with "
            "ε=0 (full coverage during the critical learning period), "
            "later rounds use epsilon_deadline.  0 disables warm-up."
        ),
    )
    watchdog_factor: float = Field(
        default=3.0,
        description=(
            "Receiver-side watchdog (gate ruling G4): a round is force-"
            "completed T_max = watchdog_factor x fluid-bound seconds after "
            "its manifest arrived, bounding the unbounded worst case of the "
            "pure coverage trigger.  Values <= 0 disable the watchdog."
        ),
    )
    skip_feedback: Literal[
        "off", "shed", "fedluar", "fedluar_random", "fedluar_cyclic"
    ] = Field(
        default="off",
        description=(
            "Skip-feedback v2 (sender omission, writeup/04-phase1/plan.md "
            "T1): the aggregator piggybacks per-sender skip advice on its "
            "broadcast; advised layers are omitted next round but remain "
            "manifest-listed with their trigger mass (counted "
            "covered-by-recycling).  'shed' advises each sender its missing "
            "layers; 'fedluar' advises a fixed count sampled "
            "inverse-ratio (the FedLUAR-native baseline, requires "
            "skip_fedluar_count >= 1); 'fedluar_random' is FedLUAR's own "
            "metric ablation (their Table 4 'Random') — same fixed count, "
            "same global set, same recycle fill, same unbounded staleness, "
            "but drawn UNIFORMLY, so any gap to 'fedluar' is attributable "
            "to the importance metric alone; 'fedluar_cyclic' is the "
            "control FedLUAR never ran — the same fixed count rotated "
            "ROUND-ROBIN, which bounds staleness for free (no tau_max) at "
            "the same expected bytes.  All three fedluar* modes require "
            "skip_fedluar_count >= 1.  Incompatible with "
            "late_layer_policy='renormalize'."
        ),
    )
    skip_fedluar_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Fixed number of layers advised per round by "
            "skip_feedback='fedluar' or 'fedluar_random' (clamped to L-1 at "
            "advice time).  Must be >= 1 in those modes; ignored otherwise."
        ),
    )

    @model_validator(mode="after")
    def _check_overnight_knobs(self) -> TrainingConfig:
        # cyclic_k=0 under the cyclic strategy would transmit nothing every
        # round — always a misconfiguration, so fail at load time rather
        # than producing empty manifests at runtime.
        if self.assignment_strategy == "cyclic" and self.cyclic_k < 1:
            raise ValueError(
                "assignment_strategy='cyclic' requires cyclic_k >= 1 "
                f"(got cyclic_k={self.cyclic_k})."
            )
        if self.lr_schedule == "step" and self.lr_decay_every < 1:
            raise ValueError(
                "lr_schedule='step' requires lr_decay_every >= 1 "
                f"(got lr_decay_every={self.lr_decay_every})."
            )
        # Skip-feedback v2 (plan T1).  FedAvg fails fast on both conditions
        # at construction; validating here as well moves the failure to
        # config-load time (before any container starts).
        if (
            self.skip_feedback in (
                "fedluar", "fedluar_random", "fedluar_cyclic",
            )
            and self.skip_fedluar_count < 1
        ):
            raise ValueError(
                f"skip_feedback={self.skip_feedback!r} requires "
                "skip_fedluar_count >= 1 "
                f"(got skip_fedluar_count={self.skip_fedluar_count})."
            )
        if (
            self.skip_feedback != "off"
            and self.late_layer_policy == "renormalize"
        ):
            raise ValueError(
                "skip_feedback cannot be combined with "
                "late_layer_policy='renormalize': skipped layers are "
                "declared covered-by-recycling in the manifest, but "
                "renormalization re-weights their mass away instead of "
                "recycling it (use 'drop' or 'recycle_last_delta')."
            )
        return self


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

class MonitoringConfig(BaseModel):
    """Monitoring / reporting settings."""

    tensorboard: bool = Field(default=True)
    tensorboard_port: int = Field(default=6006, ge=1, le=65535)
    report_output: str = Field(
        default="./results/",
        description="Directory for experiment reports.",
    )


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    """Root configuration model for a federated learning experiment."""

    federation: FederationConfig
    traffic_classes: TrafficClassesConfig
    training: TrainingConfig
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @model_validator(mode="after")
    def _cross_validate(self) -> ExperimentConfig:
        num_nodes: int = self.federation.num_nodes
        num_classes: int = self.traffic_classes.num_classes
        edges = self.federation.topology.edges

        # 1. Every edge must define exactly num_classes traffic classes.
        for idx, edge in enumerate(edges):
            if set(edge.classes.keys()) != set(range(num_classes)):
                raise ValueError(
                    f"Edge {idx} (src={edge.src}, dst={edge.dst}) defines "
                    f"class keys {sorted(edge.classes.keys())}, but expected "
                    f"{list(range(num_classes))} (num_classes={num_classes})."
                )

        # 2. dscp_mapping must cover every class index 0 .. num_classes-1.
        expected_keys = set(range(num_classes))
        actual_keys = set(self.traffic_classes.dscp_mapping.keys())
        if actual_keys != expected_keys:
            raise ValueError(
                f"dscp_mapping keys {sorted(actual_keys)} do not match "
                f"expected class indices {sorted(expected_keys)}."
            )

        # 3. All src / dst must be in [0, num_nodes).
        valid_ids = set(range(num_nodes))
        for idx, edge in enumerate(edges):
            if edge.src not in valid_ids:
                raise ValueError(
                    f"Edge {idx}: src={edge.src} is out of range "
                    f"[0, {num_nodes})."
                )
            if edge.dst not in valid_ids:
                raise ValueError(
                    f"Edge {idx}: dst={edge.dst} is out of range "
                    f"[0, {num_nodes})."
                )

        # 4. No duplicate edges (same src + dst).
        seen: set[tuple[int, int]] = set()
        for idx, edge in enumerate(edges):
            pair = (edge.src, edge.dst)
            if pair in seen:
                raise ValueError(
                    f"Edge {idx}: duplicate edge ({edge.src} -> {edge.dst})."
                )
            seen.add(pair)

        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config and validate it.

    Parameters
    ----------
    path:
        Filesystem path to the YAML configuration file.

    Returns
    -------
    ExperimentConfig
        A fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    yaml.YAMLError
        If the file is not valid YAML.
    pydantic.ValidationError
        If the parsed data fails schema validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return ExperimentConfig.model_validate(raw)
