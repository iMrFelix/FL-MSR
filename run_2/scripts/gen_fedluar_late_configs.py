#!/usr/bin/env python3
"""Generate `fedluar_late` — the SAME selection-rule contrast as `fedluar_hh`,
run to a late-training horizon (2026-08-06).

WHY THIS CAMPAIGN EXISTS
------------------------
`fedluar_hh` answered `09` Q3: at matched delta, FedLUAR's update-to-weight
importance metric does NOT beat blind selection on our workload.  Every point
estimate favours blind (-2.54 / -7.92 / -8.92 pp vs i.i.d. uniform), against
their own Table 4 which has the same contrast at +6.21 pp.

But every run in `fedluar_hh` is TWENTY ROUNDS, and that is the one thing that
could explain the whole result away.  FedLUAR selects on
s_l = ||Delta_l|| / ||theta_l||, the update-to-weight ratio.  Early in training
every layer has a large relative update, so the metric has little dynamic range
to discriminate on -- the signal it exists to exploit is a LATE-training
phenomenon.  Our `eps0` reaches 42.09% at r=20 against a measured 76.80%
ceiling: we tested their mechanism nowhere near the regime it targets.

This campaign runs the identical contrast at 100 rounds.  It separates:

  "the metric carries no signal in this workload"   (result stands, stronger)
  "the metric carries no signal YET at r=20"        (result was our artefact)

Either answer is publishable and the second one is the honest risk.  It is
registered as `09` Q3b.

SCOPE -- deliberately narrow, because the box time is not free
--------------------------------------------------------------
delta = 7 ONLY (the Comm = 0.500 anchor, and the ONLY tier in `fedluar_hh`
with real power: n=5-6 there vs n=2-3 elsewhere).  Three selection rules,
three seeds, nine runs.

  fedluar_d7   inverse-ratio sampling ||Delta_l|| / ||theta_l||   (their method)
  luarand_d7   uniform i.i.d. resampling                          (their Table 4)
  luarcyc_d7   deterministic round-robin rotation                 (never theirs)

ONE BUNDLE CHANGE FROM `fedluar_hh`, AND IT MUST BE DECLARED
-------------------------------------------------------------
`clipnorm=1.0` is ON here and was OFF in `fedluar_hh`.  Reason: without it the
collapse mechanism (`05` Sec 5.1) makes long runs a coin flip -- all three
delta=11 arms already went non-finite inside 20 rounds -- and a campaign whose
runs die answers nothing.  With clipping we have 0 collapses in 9 runs at 150
rounds.

Consequence, and it is a real one: the delta=7 numbers here are NOT a pure
horizon extension of `fedluar_hh`'s.  The CONTRAST is still clean, because all
three arms share the bundle and differ only in the selection rule -- that is
the quantity Q3b asks about.  But any statement of the form "the gap moved
from X at r=20 to Y at r=100" is a BETWEEN-CAMPAIGN comparison across two
bundles and must be labelled as such, never presented as one trend line.

Emits: configs/experiments/fedluar_late/<arm>/seed<seed>.yaml + specs.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config.schema import load_config  # noqa: E402

PL_BASE = ROOT / "configs/experiments/phase1b/exp1_grid/coverage_eft_eps03/seed41.yaml"
OUT = ROOT / "configs/experiments/fedluar_late"

SEEDS = [41, 42, 43]
DELTA = 7                              # the Comm = 0.500 anchor
ROUNDS = 100
NUM_LAYERS = 14

# Identical to gen_fedluar_hh_configs.LUAR_COMMON -- do not drift.  FedLUAR has
# no latency trigger and bounds staleness nowhere; either one leaking in would
# make the arm not-FedLUAR, which is what the asserts below defend.
LUAR_COMMON = {
    "late_layer_policy": "recycle_last_delta",
    "epsilon_deadline": 0.0,
    "epsilon_warmup_rounds": 0,
    "assignment_strategy": "byte_balanced",
    "aging_mode": "none",
    "aging_lambda": 0.0,
    "aging_tau_max": 0,
}

# The long-horizon survival bundle (h150).  Applied IDENTICALLY to all three
# arms, so it cannot favour a selection rule.
LATE_COMMON = {
    "clipnorm": 1.0,
    "server_momentum": 0.9,
    "lr_schedule": "cosine",
}

MODES = [("fedluar", "fedluar"),
         ("fedluar_random", "luarand"),
         ("fedluar_cyclic", "luarcyc")]


def build(seed: int, mode: str) -> dict:
    cfg = yaml.safe_load(PL_BASE.read_text())
    tr = cfg["training"]
    tr["dataset"]["partition"]["seed"] = seed
    tr["total_rounds"] = ROUNDS
    tr["momentum"] = 0.0            # client optimizer as in fedluar_hh
    tr["global_eval"] = True        # shared 10k held-out set
    tr.update(LUAR_COMMON)
    tr.update(LATE_COMMON)
    tr["skip_feedback"] = mode
    tr["skip_fedluar_count"] = DELTA
    return cfg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs: list[tuple] = []
    for mode, tag in MODES:
        arm = f"{tag}_d{DELTA}"
        (OUT / arm).mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            path = OUT / arm / f"seed{seed}.yaml"
            path.write_text(yaml.dump(build(seed, mode),
                                      default_flow_style=False, sort_keys=False))
            try:
                v = load_config(str(path))
                t = v.training
                assert t.total_rounds == ROUNDS
                assert t.global_eval is True
                assert t.momentum == 0.0
                assert t.dataset.partition.seed == seed
                assert t.skip_feedback == mode
                assert t.skip_fedluar_count == DELTA
                assert t.clipnorm == 1.0
                # FedLUAR fidelity guards -- same as the hh generator.
                assert t.aging_mode == "none"
                assert t.aging_tau_max == 0
                assert t.epsilon_deadline == 0.0
                assert t.late_layer_policy == "recycle_last_delta"
            except Exception as e:  # noqa: BLE001
                print(f"INVALID {path}: {e}")
                return 1
            specs.append((arm, seed, str(path.relative_to(ROOT))))

    # SEED-BLOCKED, not arm-blocked. The runner takes rows three at a time, so
    # arm-major order would put one whole arm in each concurrent batch and alias
    # measurement occasion onto arm -- the exact confound `commbound2` was built
    # to remove from `commbound`. Accuracy is bit-deterministic here so it would
    # not corrupt the contrast, but seed-blocking costs nothing and keeps any
    # timing side-observation usable.
    specs.sort(key=lambda r: (r[1], r[0]))
    with (OUT / "specs.txt").open("w") as f:
        for arm, seed, cfgpath in specs:
            f.write(f"concurrent\t1\t{arm}\t{seed}\t{cfgpath}\n")

    print(f"OK: wrote + validated {len(specs)} configs "
          f"({len(MODES)} arms x {len(SEEDS)} seeds, {ROUNDS} rounds)")
    print(f"  specs -> {(OUT / 'specs.txt').relative_to(ROOT)}")
    print(f"  byte budget: luarand/luarcyc = {1 - DELTA / NUM_LAYERS:.4f} "
          f"Comm_up by construction; fedluar EMERGENT, measure it.")
    print(f"  est wall-clock: {len(specs) / 3:.0f} batches x ~{ROUNDS * 0.35:.0f} "
          f"min = ~{len(specs) / 3 * ROUNDS * 0.35 / 60:.1f} h at 3-way")
    print("\nREAD THE DOCSTRING before comparing these to fedluar_hh: "
          "clipnorm differs,\nso cross-campaign deltas are between-bundle, "
          "not a horizon trend line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
