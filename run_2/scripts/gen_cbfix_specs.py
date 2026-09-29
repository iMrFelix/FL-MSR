#!/usr/bin/env python3
"""Stage `cbfix` — the post-ML-02-fix re-run of the quarantined cb recycle arms.

WHY.  `campaigns/cb/recycle_eps03_r20` (n=6) and `_r50` are QUARANTINED: they
ran on pre-fix code (bcac3c5) where FedAvgM's velocity accumulated
recycle/stale-FILLED mass — the positive-feedback loop of audit finding ML-02.
Measured consequence: ΔACC −20.95 ± 6.72pp vs mono, against −3.86pp for the
same arm without momentum, with val_loss RISING r12→r16 (2.17→3.41) while
same-seed mono/drop descended. The fix makes velocity accumulate
fresh-arrival motion only. Nothing replaces those runs until they are re-run.

WHAT THIS STAGES.  r20 only, n=6, TWO arms:

  recycle_eps03_r20   the quarantined arm — the point of the exercise
  mono_r20            its baseline, re-run alongside

MONO IS NOT PADDING — it buys two things for ~35 extra minutes:
  1. a SAME-CODE, SAME-OCCASION paired contrast. Pairing new-recycle against
     the existing pre-fix `cb/mono_r20` would compare across a code change and
     across measurement occasions, which is the confound `commbound2` was built
     to remove from `commbound`.
  2. a FREE DETERMINISM SENTINEL. mono has no recycle fills, so ML-02 cannot
     touch it: its new values should reproduce `cb/mono_r20` closely (bitwise,
     if the partitions match). If they do NOT, the code change moved an arm it
     should not have, and that is a finding we need before the meeting rather
     than after.

CONFIGS ARE REUSED UNCHANGED from `configs/experiments/cb/` — the whole point is
same config, post-fix code. This script validates them and emits a SEED-BLOCKED
specs.txt; it writes no new YAML.

WHY THREE ARMS AND NOT TWO — this is a MEASUREMENT decision, not scope creep.
The runner takes rows THREE at a time. With two arms x six seeds, seed-major
order still straddles batches: seed41's pair lands together but seed42's mono
and recycle fall in different batches, so half the pairs are compared ACROSS
measurement occasions. That is precisely the confound `commbound2` was built to
remove from `commbound`. With the full triptych (mono / drop / recycle) each
batch is EXACTLY ONE SEED, all three arms, same machine conditions — perfect
seed-blocking, which is the design our own record says is correct.

`drop_eps03_r20` is not quarantined (no recycle fills, so ML-02 cannot reach
it), so its re-run is not strictly required — but including it is what makes the
blocking exact, and it costs ~35 minutes. It also completes the triptych
same-code and same-occasion, which is what the accuracy claims actually want.
If the time is not there, the 2-arm variant is one edit (drop "drop_eps03_r20"
from ARMS) and the straddling must then be stated as a caveat.

VIRGIN ROOT: output goes to `campaigns/cbfix/`, never into `campaigns/cb/`.
The pre-fix runs are the exhibit for the ML-02 interaction finding (`10` §10.4)
and the standing evidence-preservation rule forbids overwriting them.

Usage:  python -m scripts.gen_cbfix_specs
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config.schema import load_config  # noqa: E402

ARMS = ["mono_r20", "drop_eps03_r20", "recycle_eps03_r20"]  # = one batch/seed
SEEDS = [41, 42, 43, 44, 45, 46]
SRC = ROOT / "configs/experiments/cb"
OUT = ROOT / "configs/experiments/cbfix"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, problems = [], []
    for seed in SEEDS:                      # seed-major = seed-blocked batches
        for arm in ARMS:
            cfg = SRC / arm / f"seed{seed}.yaml"
            if not cfg.exists():
                problems.append(f"missing {cfg}")
                continue
            try:
                v = load_config(str(cfg))
                t = v.training
                assert t.total_rounds == 20, f"{cfg}: rounds={t.total_rounds}"
                assert t.dataset.partition.seed == seed, f"{cfg}: seed mismatch"
                # D1 — assert the arm is the arm its NAME claims.
                # mono and drop are NOT distinguishable by late_layer_policy or
                # momentum (both carry policy='drop', momentum=0.9). The only
                # discriminator is update_mode: monolithic vs per_layer. Without
                # this, a mislabelled mono config validates silently — and mono
                # is the FREE DETERMINISM SENTINEL, so a mislabel would void the
                # one check that would tell us the ML-02 fix moved an arm it
                # should not have. Validating everything except identity is how
                # you get a clean-looking campaign that measures nothing.
                if arm.startswith("mono"):
                    assert t.update_mode == "monolithic", \
                        f"{cfg}: arm is named mono but update_mode=" \
                        f"{t.update_mode!r} — mislabelled arm, sentinel void"
                else:
                    assert t.update_mode == "per_layer", \
                        f"{cfg}: arm {arm} expects per_layer, got " \
                        f"{t.update_mode!r}"
                # R1 — the arm NAME encodes epsilon; assert the config agrees.
                # Same identity-blindness class as D1: flipping the eps03 arms
                # to 0.0 validates 18/18 silently (the momentum guard still
                # passes) while removing the epsilon trigger entirely — no late
                # layers, no recycle fills, so the ML-02 premise this campaign
                # exists to re-test would be gone and every check would be green.
                if "eps03" in arm:
                    assert t.epsilon_deadline == 0.3, \
                        f"{cfg}: arm is named eps03 but epsilon_deadline=" \
                        f"{t.epsilon_deadline!r} — mislabelled arm, ML-02 premise void"
                    # R1b — a warm-up overrides epsilon for the FIRST rounds, so
                    # a nonzero value means the arm is not at 0.3 when it says it
                    # is. At 20 rounds a warm-up of 5 is a quarter of the campaign
                    # at the wrong operating point, with every other check green.
                    assert t.epsilon_warmup_rounds == 0, \
                        f"{cfg}: eps03 arm has epsilon_warmup_rounds=" \
                        f"{t.epsilon_warmup_rounds!r} — early rounds run at a " \
                        "different epsilon, so the arm is not the arm it claims"
                if arm.startswith("drop"):
                    assert t.late_layer_policy == "drop", \
                        f"{cfg}: drop arm has policy={t.late_layer_policy!r}"
                # The bundle under test: ML-02 only bites when server momentum
                # meets recycle fills, so both must be present on the recycle arm.
                if arm.startswith("recycle"):
                    assert t.late_layer_policy == "recycle_last_delta", \
                        f"{cfg}: not a recycle arm ({t.late_layer_policy})"
                    assert (t.server_momentum or 0) > 0, \
                        f"{cfg}: server_momentum={t.server_momentum} — ML-02 " \
                        "cannot occur without it; wrong config"
            except Exception as e:  # noqa: BLE001
                problems.append(f"INVALID {cfg}: {e}")
                continue
            rows.append((arm, seed, str(cfg.relative_to(ROOT))))

    if problems:
        print("STAGING FAILED:")
        for p in problems:
            print(f"  {p}")
        return 1

    spec = OUT / "specs.txt"
    with spec.open("w") as f:
        for arm, seed, path in rows:
            f.write(f"concurrent\t0\t{arm}\t{seed}\t{path}\n")

    print(f"OK: staged + validated {len(rows)} runs "
          f"({len(ARMS)} arms x {len(SEEDS)} seeds, 20 rounds)")
    print(f"  specs -> {spec.relative_to(ROOT)}   (SEED-BLOCKED)")
    print(f"  output root -> campaigns/cbfix/   (VIRGIN; campaigns/cb is evidence)")
    print("\n  batch order (3-way):")
    for i in range(0, len(rows), 3):
        print("    " + " | ".join(f"{a}/s{s}" for a, s, _ in rows[i:i + 3]))
    print(f"\n  MEASURED runtime basis: cb r20 runs averaged 15.5-16.2 min wall "
          f"(range 13.6-19.8)\n  across all three arms at n=6, 3-way concurrent "
          f"-> {len(rows)}/3 = {len(rows) // 3} batches ~= "
          f"{len(rows) // 3 * 17 / 60:.1f} h training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
