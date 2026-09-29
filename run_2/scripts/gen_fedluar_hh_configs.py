#!/usr/bin/env python3
"""Generate `fedluar_hh` — FedLUAR-native vs ImpRoute, head to head, at
matched communication budgets, on OUR workload (2026-08-05).

WHY THIS CAMPAIGN EXISTS
------------------------
FedLUAR (NeurIPS 2025) reports accuracy at a fixed ROUND budget against a
`Comm` ratio (byte-weighted per-layer aggregation count / rounds, uplink
only) and never measures wall-clock.  Our own arms were published against
wall-clock in a regime where the wire is ~2.7% of a round, which is why they
looked bad.  Re-analysis of the existing runs on a bytes axis moved the
shedding arms by +9 to +24.5 pp — but it also showed that at Comm ~= 0.5 our
best arm (-7.48 pp) sits on FedLUAR's own *blind Random control* (-7.33 pp),
not on their method (-1.12 pp).  Nothing on disk can say whether that gap is
FedLUAR's importance metric earning its keep or a property of our workload,
because FedLUAR-native has never been run here.  This campaign runs it.

THE FOUR-WAY THE CAMPAIGN IS BUILT AROUND
-----------------------------------------
Every arm in the `fedluar*` family is IDENTICAL in delta (the fixed skipped
count), in the one-global-set broadcast, in the recycle fill (theta +
Delta_prev, FedLUAR Eq. 3-5), in epsilon (0 — no latency trigger, FedLUAR has
none), and in aging (off — FedLUAR §3.3 bounds staleness nowhere).  They
differ in ONE thing, the selection rule:

  fedluar_d<d>   inverse-ratio sampling ||Delta_l|| / ||theta_l||  (their method)
  luarand_d<d>   uniform i.i.d. resampling                          (their Table 4 "Random")
  luarcyc_d<d>   deterministic round-robin rotation                 (the control they NEVER ran)

`luarcyc` is the scientific point of the campaign.  FedLUAR's Table 4 beats
i.i.d. uniform Random, but i.i.d. Random shares their own unbounded-staleness
pathology.  Round-robin rotation bounds staleness for free, needs no metric
and no tau_max — and it is the control that dominates OUR frontier.  If
`luarcyc` matches or beats `fedluar` at matched bytes on this workload, the
paper's headline ablation does not survive the harder control.

Against those sit our own arms and the legacy blind control:

  improute_tau<t>  ImpRoute shedding path (skip_feedback=shed + aging cap)
  cyclic_k<k>      blind rotation with FREEZE fill (the cp2fix control)
  eps0             per-layer transport, nothing shed — the INSTRUMENTED Comm=1.0
                   denominator (mono carries no layer_comm_metrics at all)
  mono             monolithic FedAvg — the accuracy anchor, pairs with w3/mono

`cyclic_k` vs `luarcyc_d(14-k)` is a bonus we get for free: same rotation,
same bytes, FREEZE fill vs RECYCLE fill — our analogue of FedLUAR's Table 5,
which is the one place they hold bytes fixed and vary only the fill rule.

BYTE BUDGETS ARE MATCHED BY CONSTRUCTION FOR THE BYTE-BLIND ARMS ONLY
---------------------------------------------------------------------
deep_cnn ships L=14 parameter tensors.  `luarand`, `luarcyc` and `cyclic` all
select byte-blind, so their uplink share is exactly (1 - delta/14) resp.
(k/14) of the model — `luarcyc`/`cyclic` exactly over a rotation, `luarand`
in expectation.  The delta grid {4,6,7,9,11} was chosen so those land on
{0.714, 0.571, 0.500, 0.357, 0.214}, which byte-matches our measured tau_max
frontier (tau2 0.722, tau3 0.585, tau5 0.344, tau8 0.227) to within 0.014 of
Comm, and puts an exact 0.500 anchor opposite cyclic_k7.

`fedluar`'s own budget is NOT settable and NOT predictable: it is whatever
the inverse-ratio sampler ends up skipping, anywhere in [0.003, 0.999] at
delta=7 depending on whether the draws land on the five 147.5 kB conv kernels
(98.4% of the payload) or on the seven biases (0.28%).  That is exactly the
paper's own situation — their `Comm` column is an OUTCOME they grid-search
delta to land, not a setting.  So the design matches on COUNT and the
ANALYSIS matches on MEASURED Comm, interpolating along the grid.

SCOPE CONDITION, STATED UP FRONT: our sampling universe is per-TENSOR (14),
not per-LAYER (7 kernel+bias groups), so half the slots carry ~0.3% of the
bytes.  That is a deviation from the paper — but it is applied IDENTICALLY to
all four selection rules, so the head-to-head still isolates the selection
rule, which is the question.  Do not compare these absolute Comm values to
the paper's.

Emits: configs/experiments/fedluar/<arm>/seed<seed>.yaml + specs.txt
Every file is validated through src.config.schema.load_config before it is
kept, and the emitted specs.txt is tier-ordered so a short window still lands
the powered headline first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config.schema import load_config  # noqa: E402

PL_BASE = ROOT / "configs/experiments/phase1b/exp1_grid/coverage_eft_eps03/seed41.yaml"
MONO_BASE = ROOT / "configs/experiments/phase1b/exp4_monolithic/mono/seed41.yaml"
OUT = ROOT / "configs/experiments/fedluar"

SEEDS6 = [41, 42, 43, 44, 45, 46]   # powered headline
SEEDS3 = [41, 42, 43]               # trend / Pareto wings

# --- measured model constants (campaigns/w3/byte_balanced_eps0/seed41 r5) ---
NUM_LAYERS = 14                      # deep_cnn parameter TENSORS
FULL_UPLINK_B = 749_401              # one worker, whole model, measured
NUM_WORKERS = 3
DOWNLINK_B = 2_248_191               # aggregator broadcast, measured, invariant

# FedLUAR-family common bundle: one shedding mechanism, no latency trigger,
# no staleness cap, recycle fill.  `byte_balanced` is the epsilon=0 optimum
# and only ORDERS/PLACES the surviving layers (the skip set is removed from
# the strategy's universe upstream), so it adds no second omission channel —
# unlike `coverage_eft`, which would.
LUAR_COMMON = {
    "late_layer_policy": "recycle_last_delta",
    "epsilon_deadline": 0.0,
    "epsilon_warmup_rounds": 0,
    "assignment_strategy": "byte_balanced",
    "aging_mode": "none",
    "aging_lambda": 0.0,
    "aging_tau_max": 0,
}

# ImpRoute shedding bundle — the EXACT w3 frontier patch, deliberately
# unchanged.  Note it keeps epsilon=0.3: `shed` advice is "the layers your
# update was missing", so epsilon slippage is what BOOTSTRAPS the shedding.
# Setting epsilon=0 here would not make the arm cleaner, it would silently
# turn it into eps0.  The consequence — our path couples two mechanisms
# where FedLUAR's couples none — is a finding, not a defect to configure away.
SHED_COMMON = {
    "skip_feedback": "shed",
    "late_layer_policy": "recycle_last_delta",
    "aging_mode": "additive_capped",
    "aging_lambda": 0.5,
}

DELTAS = [4, 6, 7, 9, 11]            # skipped-tensor counts
HEADLINE_DELTA = 7                   # the Comm = 0.500 anchor

# arm -> (tier, base, seeds, training patch)
ARMS: dict[str, tuple] = {}

# --- tier 0: the powered headline four-way at Comm ~= 0.500 ---------------
ARMS["mono"] = (0, MONO_BASE, SEEDS6, {})
for mode, tag in (
    ("fedluar", "fedluar"), ("fedluar_random", "luarand"),
    ("fedluar_cyclic", "luarcyc"),
):
    ARMS[f"{tag}_d{HEADLINE_DELTA}"] = (
        0, PL_BASE, SEEDS6,
        {"skip_feedback": mode, "skip_fedluar_count": HEADLINE_DELTA,
         **LUAR_COMMON},
    )

# --- tier 1: byte denominator, nearest ImpRoute arms, fill ablation -------
ARMS["eps0"] = (1, PL_BASE, SEEDS3,
                {"epsilon_deadline": 0.0,
                 "assignment_strategy": "byte_balanced"})
ARMS["improute_tau3"] = (1, PL_BASE, SEEDS3,
                         {**SHED_COMMON, "aging_tau_max": 3})
ARMS["improute_tau5"] = (1, PL_BASE, SEEDS3,
                         {**SHED_COMMON, "aging_tau_max": 5})
ARMS["cyclic_k7"] = (1, PL_BASE, SEEDS3,
                     {"assignment_strategy": "cyclic", "cyclic_k": 7,
                      "epsilon_deadline": 0.0, "late_layer_policy": "drop"})

# --- tier 2: the Pareto wings of the delta grid ---------------------------
for delta in DELTAS:
    if delta == HEADLINE_DELTA:
        continue
    for mode, tag in (
        ("fedluar", "fedluar"), ("fedluar_random", "luarand"),
        ("fedluar_cyclic", "luarcyc"),
    ):
        ARMS[f"{tag}_d{delta}"] = (
            2, PL_BASE, SEEDS3,
            {"skip_feedback": mode, "skip_fedluar_count": delta,
             **LUAR_COMMON},
        )

# --- tier 3: frontier completion ------------------------------------------
ARMS["improute_tau2"] = (3, PL_BASE, SEEDS3,
                         {**SHED_COMMON, "aging_tau_max": 2})
ARMS["improute_tau8"] = (3, PL_BASE, SEEDS3,
                         {**SHED_COMMON, "aging_tau_max": 8})
ARMS["cyclic_k3"] = (3, PL_BASE, SEEDS3,
                     {"assignment_strategy": "cyclic", "cyclic_k": 3,
                      "epsilon_deadline": 0.0, "late_layer_policy": "drop"})


def expected_comm(arm: str, patch: dict) -> tuple[str, str]:
    """(expected steady-state Comm_uplink, provenance) for one arm.

    Byte-blind selection rules have an analytic budget; `fedluar` and the
    ImpRoute shedding path do not, and are reported as EMERGENT so nobody
    can mistake a design intention for a measurement.
    """
    mode = patch.get("skip_feedback")
    if arm == "mono":
        return "1.0000", "DERIVED (no telemetry; use eps0)"
    if mode in ("fedluar_random", "fedluar_cyclic"):
        delta = patch["skip_fedluar_count"]
        exact = "exact/rotation" if mode == "fedluar_cyclic" else "expectation"
        return f"{1 - delta / NUM_LAYERS:.4f}", f"ANALYTIC ({exact})"
    if mode == "fedluar":
        return "EMERGENT", "MEASURE IT (inverse-ratio draw)"
    if mode == "shed":
        return "EMERGENT", (
            "MEASURE IT (w3 prior: tau2 .722 tau3 .585 tau5 .344 tau8 .227)"
        )
    if patch.get("assignment_strategy") == "cyclic":
        return f"{patch['cyclic_k'] / NUM_LAYERS:.4f}", "ANALYTIC (rotation)"
    return "1.0000", "MEASURED (full model)"


def build(base_path: Path, seed: int, patch: dict) -> dict:
    cfg = yaml.safe_load(base_path.read_text())
    tr = cfg["training"]
    tr["dataset"]["partition"]["seed"] = seed
    tr["total_rounds"] = 20
    tr["momentum"] = 0.0          # same optimizer as w3 — numbers pair
    tr["global_eval"] = True      # shared 10k held-out set, not local val
    tr.update(patch)
    return cfg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs: list[tuple] = []
    budget_rows: list[tuple] = []
    n_ok = 0

    for arm, (tier, base, seeds, patch) in ARMS.items():
        arm_dir = OUT / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            cfg = build(base, seed, patch)
            path = arm_dir / f"seed{seed}.yaml"
            path.write_text(
                yaml.dump(cfg, default_flow_style=False, sort_keys=False)
            )
            # Validate: pydantic rejects a bad knob/enum HERE, on the laptop,
            # not three hours into a queued campaign on the box.
            try:
                v = load_config(str(path))
                assert v.training.global_eval is True
                assert v.training.total_rounds == 20
                assert v.training.momentum == 0.0
                assert v.training.dataset.partition.seed == seed
                if "skip_feedback" in patch:
                    assert v.training.skip_feedback == patch["skip_feedback"]
                if "skip_fedluar_count" in patch:
                    assert (v.training.skip_fedluar_count
                            == patch["skip_fedluar_count"])
                    # FedLUAR has no staleness bound and no latency trigger;
                    # either one leaking in would make the arm not-FedLUAR.
                    assert v.training.aging_mode == "none"
                    assert v.training.aging_tau_max == 0
                    assert v.training.epsilon_deadline == 0.0
                    assert (v.training.late_layer_policy
                            == "recycle_last_delta")
            except Exception as e:  # noqa: BLE001
                print(f"INVALID {path}: {e}")
                return 1
            specs.append((tier, arm, seed, str(path.relative_to(ROOT))))
            n_ok += 1

        comm, prov = expected_comm(arm, patch)
        up_mb = (
            float(comm) * FULL_UPLINK_B * NUM_WORKERS / 1e6
            if comm not in ("EMERGENT",) else None
        )
        budget_rows.append((tier, arm, len(seeds), comm, up_mb, prov))

    specs.sort(key=lambda r: (r[0], r[1], r[2]))
    specs_txt = OUT / "specs.txt"
    with specs_txt.open("w") as f:
        for tier, arm, seed, cfgpath in specs:
            f.write(f"concurrent\t{tier}\t{arm}\t{seed}\t{cfgpath}\n")

    print(f"OK: wrote + validated {n_ok} configs across {len(ARMS)} arms")
    print(f"  specs -> {specs_txt.relative_to(ROOT)}")
    print()
    print("EXPECTED STEADY-STATE BYTE BUDGETS "
          f"(full model = {FULL_UPLINK_B} B/worker, "
          f"{FULL_UPLINK_B * NUM_WORKERS / 1e6:.4f} MB/round over "
          f"{NUM_WORKERS} workers; downlink {DOWNLINK_B / 1e6:.4f} MB/round "
          "in EVERY arm)")
    print(f"{'tier':>4} {'arm':<16} {'n':>2} {'Comm_up':>9} "
          f"{'MB/round':>9} {'Comm_tot':>9}  provenance")
    for tier, arm, n, comm, up_mb, prov in sorted(budget_rows):
        if up_mb is None:
            print(f"{tier:>4} {arm:<16} {n:>2} {comm:>9} {'—':>9} "
                  f"{'—':>9}  {prov}")
        else:
            tot = (up_mb * 1e6 + DOWNLINK_B) / (
                FULL_UPLINK_B * NUM_WORKERS + DOWNLINK_B
            )
            print(f"{tier:>4} {arm:<16} {n:>2} {comm:>9} {up_mb:>9.4f} "
                  f"{tot:>9.4f}  {prov}")
    print()
    print("NOTE round 0 carries no advice (R_0 = empty, as in the paper), so "
          "every fedluar* arm sends the FULL model once; the 20-round mean is "
          "0.95*steady + 0.05.  Read steady state with rounds>=3 and "
          "analysis_common.cycle_stats, never a naive mean.")
    est_min = (n_ok / 3) * 11
    print(f"  rough estimate: {n_ok} runs = {n_ok / 3:.0f} batches @3-way "
          f"~= {est_min / 60:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
