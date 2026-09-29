#!/usr/bin/env python3
"""Analyze the `stab` stability census — does shedding move the collapse boundary?

DESIGN.  2 arms x 3 learning rates x 6 seeds = 36 runs at 50 rounds, on the
LEGACY WEAK BASELINE ON PURPOSE (no clipping).  Clipping is what fixed the
collapse (`kb/05` §5.1), so a census run *with* it would have nothing to
measure.  The question is not "do runs collapse" — we know they do without
clipping — but whether **shedding shifts the boundary relative to monolithic at
the same learning rate**.

WHY THIS IS WORTH A SECTION.  "Shedding does not shift the stability boundary"
is a measurement, not a failure, and it is the one place the collapse story
turns into a contribution.  The inverse finding — that shedding *does*
destabilise — would be equally publishable and considerably more awkward, which
is exactly why the test is pre-specified here rather than eyeballed.

TWO GATES BEFORE ANY CELL IS COUNTED
  * GENERATION.  `campaigns/stab` on the laptop mixes the quarantined first
    attempt with the current one.  Filter of record: a run counts only if it
    carries `results/data_digests.json`, which only the current generation's
    runner writes.
  * INTEGRITY.  The census generation is 3-way CONCURRENT, which is the
    condition under which silent all-zero image injection occurs (`kb/13`
    §13.5).  A corrupt cell must be re-run serially, never counted: injected
    label noise is itself a destabiliser, so counting one would bias the
    collapse rate in whichever direction the corruption happened to land.

    Gate used here is `duplicate_images == 0` from the digest.  Note this is a
    PROXY: k injected all-zero rows are duplicates of one another, so they
    contribute k-1 duplicates and any real injection (hundreds of rows) trips
    it.  It cannot distinguish "exactly one zero row" from clean — harmless at
    the observed scale, but it is a proxy and is labelled as one.  Where the
    shard residue still exists, prefer a direct zero-row scan
    (`scripts/audit_partitions.py`).

STATISTICS.  Collapse is a BINARY outcome, so accuracy-style t-intervals do not
apply.  The seed-paired test is an exact two-sided McNemar (binomial on the
discordant pairs at p=0.5), computed here without scipy.  At n=6 per cell the
smallest attainable two-sided p is 2^-5 = 0.031 with 6/6 discordant pairs, so
anything less lopsided cannot reach alpha — that floor is reported per cell so
a null is never mistaken for equivalence.

Usage:  python -m scripts.analyze_stab [campaigns/stab]
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import analysis_common as ac  # noqa: E402

ALPHA = 0.05


@dataclass
class Cell:
    arm: str          # "mono" | "drop_eps03"
    lr: str           # "005" | "010" | "015"
    seed: int
    collapsed: bool
    why: str
    acc: float | None
    nan_round: int | None
    peak: float | None
    digest_ok: bool
    digest_note: str


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p at p=0.5 — McNemar on discordant pairs."""
    if n == 0:
        return float("nan")
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] + 1e-12))


def mcnemar_floor(n: int) -> float:
    """Smallest two-sided p reachable with n discordant pairs."""
    return binom_two_sided(0, n) if n else float("nan")


def load(campaign: Path) -> tuple[list[Cell], list[str]]:
    cells: list[Cell] = []
    skipped: list[str] = []
    for arm_dir in sorted(p for p in campaign.iterdir() if p.is_dir()):
        name = arm_dir.name
        if "_lr" not in name:
            continue
        arm, lr = name.rsplit("_lr", 1)
        for seed_dir in sorted(arm_dir.glob("seed*")):
            rep_p = seed_dir / "results" / "report.json"
            dig_p = seed_dir / "results" / "data_digests.json"
            if not rep_p.exists():
                continue
            if not dig_p.exists():
                skipped.append(f"{name}/{seed_dir.name} (no data_digests.json "
                               "— quarantined first generation)")
                continue
            digest_ok, note = True, ""
            try:
                dig = json.loads(dig_p.read_text())
                dups = {k: v.get("duplicate_images") for k, v in dig.items()}
                zeros = {k: v.get("zero_rows") for k, v in dig.items()
                         if v.get("zero_rows") is not None}
                if any(v for v in dups.values() if v):
                    digest_ok = False
                    note = f"duplicate_images {dups}"
                elif any(v for v in zeros.values() if v):
                    digest_ok = False
                    note = f"zero_rows {zeros}"
            except Exception as e:  # noqa: BLE001
                digest_ok, note = False, f"unreadable digest: {e!r}"

            rep = json.loads(rep_p.read_text())
            seed = int(seed_dir.name.replace("seed", ""))
            rr = ac.read_run(rep, seed=seed)
            cells.append(Cell(
                arm=arm, lr=lr, seed=seed,
                collapsed=not rr.health.healthy,
                why=rr.health.reason or "healthy",
                acc=None if rr.lastk is None else 100.0 * rr.lastk,
                nan_round=rr.health.nan_round,
                peak=None if rr.health.peak is None else 100.0 * rr.health.peak,
                digest_ok=digest_ok, digest_note=note,
            ))
    return cells, skipped


def main(argv: list[str]) -> int:
    campaign = Path(argv[1]) if len(argv) > 1 else ROOT / "campaigns/stab"
    if not campaign.exists():
        print(f"no campaign at {campaign}")
        return 1
    cells, skipped = load(campaign)
    if not cells:
        print(f"no current-generation runs under {campaign} "
              "(none carry data_digests.json)")
        return 1

    # SCOPE is stated in PAIRABLE tiers, not in tiers merely PRESENT.  Every
    # contrast below is seed-paired within one lr, so a tier holding a single
    # arm yields descriptive rows and no contrast at all.  Deriving the header
    # from lrs_present would print "SCOPE: lr 0.010, 0.015" the moment the first
    # 0.015 arm lands — reading as a two-tier census that does not exist, which
    # is exactly the mis-read a mid-campaign cut invites.  Worse, the old
    # mid-campaign banner keyed on len(lrs_present) < 3, so it would SWITCH OFF
    # once three tiers were present even if two of them were single-arm — the
    # campaign's own arm order (drop before mono within each tier) guarantees
    # that state occurs.
    lrs_present = sorted({c.lr for c in cells})
    arms_at = {l: sorted({c.arm for c in cells if c.lr == l}) for l in lrs_present}
    pairable = [l for l in lrs_present if len(arms_at[l]) >= 2]
    partial = [l for l in lrs_present if len(arms_at[l]) < 2]
    _f = lambda ls: ", ".join("0." + l for l in ls)  # noqa: E731
    scope = ("NO PAIRABLE TIER YET" if not pairable
             else f"lr=0.{pairable[0]} ONLY" if len(pairable) == 1
             else f"lr {_f(pairable)}")
    print("=" * 78)
    print(f"stab — stability census   ({campaign})")
    print(f"SCOPE (pairable tiers): {scope}   "
          f"[{len(cells)} current-generation runs]")
    if partial:
        detail = "; ".join(f"0.{l} ({arms_at[l][0]} only)" for l in partial)
        print(f"   present but NOT pairable: {detail}")
        print("   these appear in §1/§4 as descriptive rows only — no paired "
              "contrast is computed for them.")
    if len(pairable) < 3:
        print("*** MID-CAMPAIGN — this is NOT the full census. Call it a "
              f"'{scope}' census;\n*** the lr sweep is the whole point and it "
              "is incomplete.")
    print("=" * 78)

    # ---- §0 gates ---------------------------------------------------------
    corrupt = [c for c in cells if not c.digest_ok]
    usable = [c for c in cells if c.digest_ok]
    print(f"\n§0 GATES")
    print(f"   current generation (has data_digests.json): {len(cells)} runs")
    print(f"   quarantined/older generation skipped: {len(skipped)}")
    for s in skipped[:8]:
        print(f"      - {s}")
    if len(skipped) > 8:
        print(f"      ... and {len(skipped) - 8} more")
    print(f"   partition-integrity: {len(usable)} pass, {len(corrupt)} FAIL")
    for c in corrupt:
        print(f"      EXCLUDED {c.arm}_lr{c.lr}/seed{c.seed}: {c.digest_note}")
    print("   NOTE the digest gate uses duplicate_images as a PROXY for "
          "zero-image\n   injection; it cannot see a single zero row. Prefer "
          "scripts/audit_partitions.py\n   where the shard residue survives.")

    # ---- §1 collapse census ----------------------------------------------
    by: dict[tuple[str, str], list[Cell]] = defaultdict(list)
    for c in usable:
        by[(c.arm, c.lr)].append(c)
    arms = sorted({c.arm for c in usable})
    lrs = sorted({c.lr for c in usable})

    print("\n§1 COLLAPSE RATE  (legacy weak baseline, no clipping — on purpose)")
    print(f"   {'arm':<14}" + "".join(f"{'lr 0.' + l:>16}" for l in lrs))
    for arm in arms:
        row = f"   {arm:<14}"
        for lr in lrs:
            cs = by.get((arm, lr), [])
            if not cs:
                row += f"{'—':>16}"
                continue
            k = sum(1 for c in cs if c.collapsed)
            row += f"{f'{k}/{len(cs)}':>16}"
        print(row)
    print("\n   failure modes:")
    modes: dict[str, int] = defaultdict(int)
    for c in usable:
        if c.collapsed:
            modes[c.why] += 1
    for m, n in sorted(modes.items(), key=lambda kv: -kv[1]):
        print(f"      {m:<24} {n}")
    if not modes:
        print("      (no collapses among gated runs)")

    # ---- §2 the paired test ----------------------------------------------
    print("\n§2 DOES SHEDDING MOVE THE BOUNDARY?  seed-paired, exact McNemar")
    print("   Pairs are the SAME seed at the SAME lr, so partition and init "
          "are held fixed;\n   only the mechanism differs.")
    if len(arms) < 2:
        print("   (needs both arms; only found: " + ", ".join(arms) + ")")
    else:
        base, treat = "mono", next((a for a in arms if a != "mono"), None)
        if base not in arms or treat is None:
            base, treat = arms[0], arms[1]
        print(f"   {treat} vs {base}\n")
        print(f"   {'lr':<8}{'pairs':>7}{'both ok':>9}{'both die':>10}"
              f"{'only ' + treat[:6]:>13}{'only ' + base[:6]:>13}"
              f"{'exact p':>10}{'min p':>8}")
        for lr in lrs:
            m = {c.seed: c for c in by.get((base, lr), [])}
            t = {c.seed: c for c in by.get((treat, lr), [])}
            seeds = sorted(set(m) & set(t))
            if not seeds:
                continue
            both_ok = sum(1 for s in seeds
                          if not m[s].collapsed and not t[s].collapsed)
            both_die = sum(1 for s in seeds
                           if m[s].collapsed and t[s].collapsed)
            only_t = sum(1 for s in seeds
                         if t[s].collapsed and not m[s].collapsed)
            only_m = sum(1 for s in seeds
                         if m[s].collapsed and not t[s].collapsed)
            disc = only_t + only_m
            p = binom_two_sided(only_t, disc) if disc else float("nan")
            floor = mcnemar_floor(disc)
            ps = "—" if disc == 0 else f"{p:.4f}"
            fs = "—" if disc == 0 else f"{floor:.3f}"
            print(f"   0.{lr:<6}{len(seeds):>7}{both_ok:>9}{both_die:>10}"
                  f"{only_t:>13}{only_m:>13}{ps:>10}{fs:>8}")
        print("\n   'only X' counts seeds where X collapsed and the other did "
              "not — the\n   discordant pairs, which are the ONLY information "
              "McNemar uses.\n   'min p' is the smallest two-sided p those "
              "pairs could ever produce: if it\n   exceeds 0.05 the cell "
              "cannot reach significance at any effect size, and a\n   null "
              "there means UNDERPOWERED, not equivalent.")

    # ---- §3 accuracy among survivors -------------------------------------
    print("\n§3 ACCURACY AMONG SURVIVORS  (both arms healthy at that seed+lr)")
    print("   ⚠ SURVIVORSHIP BIAS — READ THE EXCLUSION LINE BEFORE THE NUMBER.")
    print("   Pairing needs both arms alive, so a seed where the BASELINE "
          "collapsed is\n   dropped. That removes mono's worst outcomes from "
          "mono's mean and therefore\n   INFLATES the baseline and OVERSTATES "
          "shedding's cost. The bias is one-directional\n   and it is against "
          "the treatment arm; never quote this ΔACC without the n and the\n"
          "   named excluded seeds.")
    if len(arms) >= 2:
        base = "mono" if "mono" in arms else arms[0]
        treat = next(a for a in arms if a != base)
        fam, seeds_by, notes = {}, {}, []
        for lr in lrs:
            mcells = by.get((base, lr), [])
            tcells = by.get((treat, lr), [])
            m = {c.seed: c.acc for c in mcells
                 if not c.collapsed and c.acc is not None}
            t = {c.seed: c.acc for c in tcells
                 if not c.collapsed and c.acc is not None}
            ss = sorted(set(m) & set(t))
            total = len({c.seed for c in mcells} | {c.seed for c in tcells})
            lost_base = sorted(c.seed for c in mcells if c.collapsed)
            lost_treat = sorted(c.seed for c in tcells if c.collapsed)
            if ss:
                fam[f"lr0.{lr}"] = [t[s] - m[s] for s in ss]
                seeds_by[f"lr0.{lr}"] = ss
                bits = [f"{len(ss)}-of-{total} seeds"]
                if lost_base:
                    bits.append(f"BASELINE ({base}) collapsed at {lost_base} "
                                "— excluding it biases the contrast AGAINST "
                                f"{treat}")
                if lost_treat:
                    bits.append(f"{treat} collapsed at {lost_treat}")
                notes.append(f"   lr0.{lr}: " + "; ".join(bits))
        if fam:
            print(ac.format_family(ac.contrast_family(
                fam, seeds_by, direction="less")))
            for ln in notes:
                print(ln)
        else:
            print("   (no seed has both arms healthy yet)")

    # ---- §4 per-run detail ------------------------------------------------
    print("\n§4 PER-RUN DETAIL")
    print(f"   {'arm':<14}{'lr':>6}{'seed':>6}{'acc%':>9}{'peak%':>9}"
          f"{'nan@r':>7}  status")
    for c in sorted(usable, key=lambda c: (c.lr, c.arm, c.seed)):
        a = "—" if c.acc is None else f"{c.acc:.2f}"
        pk = "—" if c.peak is None else f"{c.peak:.2f}"
        nr = "—" if c.nan_round is None else str(c.nan_round)
        print(f"   {c.arm:<14}{'0.' + c.lr:>6}{c.seed:>6}{a:>9}{pk:>9}"
              f"{nr:>7}  {c.why}")

    print("\n" + "=" * 78)
    print("SCOPE CONDITIONS")
    print("  * Weak baseline BY DESIGN: no clipping, so collapses can occur "
          "at all.\n    These rates say nothing about the shipped "
          "configuration, which has\n    clipnorm on and 0 collapses in 9 runs "
          "at 150 rounds.")
    print("  * n=6 per cell on a BINARY outcome. Read the 'min p' column "
          "before reading\n    any p: most cells cannot reach alpha "
          "regardless of effect size.")
    print("  * A null here is 'no shift detected at this power', never "
          "'no shift'.\n    Equivalence would need a pre-registered margin on "
          "the collapse-rate\n    difference, which we have not set.")
    print("  * Collapse is scored by the pre-registered health gate "
          "(analysis_common),\n    not by eye: non-finite loss, pinned at "
          "chance, or a read below floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
