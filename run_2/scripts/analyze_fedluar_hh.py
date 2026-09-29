#!/usr/bin/env python3
"""Analyze `fedluar_hh` — FedLUAR-native vs ImpRoute at matched comm budgets.

Reports, in the order the argument has to be made:

  §0 health + provenance     which runs may enter a mean at all
  §1 the FedLUAR table       accuracy at a FIXED ROUND BUDGET vs measured Comm
                             (their Table 2 shape, on our workload)
  §2 the Pareto plane        who is dominated by whom on (Comm_up, accuracy)
  §3 selection-rule family   fedluar vs luarand vs luarcyc at matched delta —
                             the head-to-head the campaign exists for
  §4 fill-rule ablation      luarcyc vs cyclic at matched bytes and matched
                             rotation: recycle fill vs freeze fill, our
                             analogue of FedLUAR's Table 5
  §5 ImpRoute vs FedLUAR     our shedding path against theirs at matched
                             measured Comm

EVERY estimator comes from scripts/analysis_common (health gate, last-k read,
Student-t CIs, Holm family, cycle-aligned byte windows).  Nothing is
hand-rolled — writeup/11 §2 and audit S6/S7/S9 exist precisely because it was.

Byte columns are MEASURED off `layer_comm_metrics[].bytes_sent`, never
asserted: worker entries are the uplink, the aggregator's are the downlink
broadcast, and roles are inferred from telemetry (BYTE-05), not hardcoded to
node-0.  The `mono` arm carries no layer_comm_metrics at all, so its byte
row is DERIVED and the `eps0` arm is the instrumented Comm=1.0 denominator.

Usage:  python -m scripts.analyze_fedluar_hh [campaigns/fedluar_hh]
"""
from __future__ import annotations

import json
import math
import re
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import analysis_common as ac  # noqa: E402

NUM_LAYERS = 14                  # deep_cnn parameter TENSORS
FULL_UPLINK_B = 749_401          # one worker, whole model (measured)
NUM_WORKERS = 3


def _num_after(arm: str, marker: str) -> int | None:
    """First integer following `marker` in an arm name, or None.

    Tolerant on purpose: arm names elsewhere in the ledger carry suffixes
    (`cyclic_k2_eps0`), and an analyzer that crashes on a neighbouring
    campaign's naming is an analyzer nobody runs.
    """
    m = re.search(re.escape(marker) + r"(\d+)", arm)
    return int(m.group(1)) if m else None


def rotation_period(arm: str) -> int | None:
    """Refresh period in rounds, for the cycle-aligned byte window (BYTE-02).

    A fixed rounds>=3 truncation contains a fractional number of refresh
    cycles, which aliases the byte column.  Deterministic rotations have an
    exact period; the stochastic arms have none and use every round.
    """
    for marker, prefix in (("_d", "luarcyc"), ("_k", "cyclic")):
        if arm.startswith(prefix):
            n = _num_after(arm, marker)
            if n:
                return NUM_LAYERS // math.gcd(NUM_LAYERS, n)
    if arm.startswith("improute_tau"):
        return _num_after(arm, "_tau")
    return None                   # fedluar / luarand / eps0 / mono: no cycle


def expected_comm_up(arm: str) -> float | None:
    """Analytic budget where the selection rule is byte-blind; else None."""
    if arm in ("mono", "eps0"):
        return 1.0
    if arm.startswith(("luarand_d", "luarcyc_d")):
        n = _num_after(arm, "_d")
        return None if n is None else 1.0 - n / NUM_LAYERS
    if arm.startswith("cyclic_k"):
        n = _num_after(arm, "_k")
        return None if n is None else n / NUM_LAYERS
    return None                   # fedluar / improute: emergent, measure it


@dataclass
class Run:
    arm: str
    seed: int
    acc: float | None             # last-k read, pp
    healthy: bool
    why: str
    up_mb: float | None           # cycle-aligned worker uplink, MB/round
    down_mb: float | None         # aggregator broadcast, MB/round
    rounds: int
    data: str = "UNVERIFIED"      # partition-integrity verdict (see below)

    @property
    def usable(self) -> bool:
        """Healthy training AND a partition that is not known-corrupt.

        In STRICT mode a partition that was never fingerprinted is also
        refused: "we kept no evidence" is not the same claim as "it was
        clean", and the best-powered contrast in §3 rests on three such cells.
        Run both ways — if a sign flips between them it was never a finding.
        """
        if not self.healthy or self.data == "CORRUPT":
            return False
        return not (STRICT and self.data == "UNVERIFIED")


STRICT = False                    # set by --strict; see Run.usable


# ---------------------------------------------------------------------------
# Partition-integrity gate  (MANDATORY before any cross-arm same-seed pairing)
# ---------------------------------------------------------------------------
# Concurrent launches have produced silently corrupt shards: 1.4–14.0% of a
# worker's training images are replaced by ALL-ZERO images while y_train stays
# byte-identical to the clean partition.  The label histogram — the only thing
# anybody ever checks — therefore looks perfect, while the run is really
# training on several percent injected label noise.  A cell in that state is
# not a mechanism observation, it is a different dataset, and pairing it
# against a clean counterpart at the same seed measures the corruption.
#
# Detector (scratchpad/zeroscan.py, verdicts in <campaign>_partition_gate.json):
# exact all-zero rows in x_train/x_val.  This is arm-independent and needs no
# majority vote, which matters — the cheaper CRC32 "differs from the modal
# shard" screen mis-called seed46 in BOTH directions, because only two arms
# had surviving data there and the tie broke toward the corrupt shard.
#
#   clean          4/4 surviving shards free of zero-image injection
#   clean_partial  <4 shards survived, all clean; unsurveyed nodes unknown
#   CORRUPT        zero-image injection measured           -> EXCLUDED
#   UNVERIFIED     no shard survived locally; unknown, and NOT the same as clean

def load_partition_gate(campaign: Path) -> dict[str, str]:
    """`{arm/seedNN: verdict}`, from whichever gate the campaign actually has.

    Two provenances, because they answer the same question at different times:

    * ``<campaign>_partition_gate.json`` — a FORENSIC gate, reconstructed after
      the fact from whatever shards survived the pull. Used for `fedluar_hh`,
      where materialisation was concurrent and unrecorded, so some cells are
      UNVERIFIED and the verdicts are the best archaeology allows.
    * ``<campaign>/PARTITION_MANIFEST.json`` — a PROSPECTIVE gate, written by
      `scripts/prematerialize.py` at materialisation time, before any training.
      Every shard is accounted for by construction, so there is no UNVERIFIED
      tier: a campaign gated this way was proven clean *before* it ran.

    Prefer the forensic file when present (it may carry hand-adjudicated
    verdicts); otherwise derive from the manifest.
    """
    fp = campaign.parent / f"{campaign.name}_partition_gate.json"
    if fp.exists():
        doc = json.loads(fp.read_text())
        return {k: v["verdict"] for k, v in doc.get("cells", {}).items()}

    mf = campaign / "PARTITION_MANIFEST.json"
    if not mf.exists():
        return {}
    gate: dict[str, str] = {}
    for cell, shards in json.loads(mf.read_text()).items():
        if any(s.get("corrupt") for s in shards.values()):
            gate[cell] = "CORRUPT"
        else:
            gate[cell] = "clean" if len(shards) >= 4 else "clean_partial"
    return gate


def _byte_series(rep: Mapping[str, Any]) -> tuple[dict[int, float], dict[int, float]]:
    """(uplink, downlink) MB per round, measured off bytes_sent.

    Roles come from analysis_common.split_roles — the aggregator is the node
    whose telemetry says so, not node-0 by convention (BYTE-05).
    """
    up: dict[int, float] = {}
    down: dict[int, float] = {}
    for rd in rep.get("per_round", []):
        rnum = rd.get("round")
        nodes = rd.get("nodes", {})
        agg, workers = ac.split_roles(nodes)
        u = d = 0.0
        seen = False
        for nid, nd in nodes.items():
            lcm = nd.get("layer_comm_metrics")
            if not lcm:
                continue
            seen = True
            total = sum(float(e.get("bytes_sent", 0)) for e in lcm)
            if nid == agg:
                d += total
            elif nid in workers:
                u += total
        if seen:
            up[rnum] = u / 1e6
            down[rnum] = d / 1e6
    return up, down


def load(campaign: Path) -> list[Run]:
    runs: list[Run] = []
    gate = load_partition_gate(campaign)
    for arm_dir in sorted(p for p in campaign.iterdir() if p.is_dir()):
        for seed_dir in sorted(arm_dir.glob("seed*")):
            report = seed_dir / "results" / "report.json"
            if not report.exists():
                continue
            rep = json.loads(report.read_text())
            seed = int(seed_dir.name.replace("seed", ""))
            rr = ac.read_run(rep, seed=seed)
            up, down = _byte_series(rep)
            tau = rotation_period(arm_dir.name)
            cs_up = ac.cycle_stats(up, tau) if up else None
            cs_dn = ac.cycle_stats(down, tau) if down else None
            runs.append(Run(
                arm=arm_dir.name, seed=seed,
                acc=None if rr.lastk is None else 100.0 * rr.lastk,
                healthy=rr.health.healthy, why=rr.health.reason,
                up_mb=cs_up.mean if cs_up else None,
                down_mb=cs_dn.mean if cs_dn else None,
                rounds=len(rep.get("per_round", [])),
                data=gate.get(f"{arm_dir.name}/{seed_dir.name}", "UNVERIFIED"),
            ))
    return runs


def by_arm(runs: list[Run]) -> dict[str, list[Run]]:
    out: dict[str, list[Run]] = {}
    for r in runs:
        out.setdefault(r.arm, []).append(r)
    return out


def paired(a: list[Run], b: list[Run]) -> tuple[list[float], list[int]]:
    """Seed-paired ΔACC (a − b) over seeds usable in BOTH arms.

    Usable = healthy training AND a partition not measured corrupt.  Both
    conditions are per-CELL: a corrupt treatment cell is dropped even when its
    baseline counterpart at the same seed is clean, because that pair would
    contrast the arm against a different dataset.
    """
    ha = {r.seed: r.acc for r in a if r.usable and r.acc is not None}
    hb = {r.seed: r.acc for r in b if r.usable and r.acc is not None}
    seeds = sorted(set(ha) & set(hb))
    return [ha[s] - hb[s] for s in seeds], seeds


def gate_note(a: list[Run], b: list[Run], seeds: list[int]) -> str:
    """What the partition gate cost this contrast, for the caller to print."""
    lost = sorted({r.seed for r in a + b if r.healthy and r.data == "CORRUPT"})
    unv = sorted({r.seed for r in a + b
                  if r.healthy and r.data == "UNVERIFIED"} - set(lost))
    bits = []
    if lost:
        bits.append(f"−{len(lost)} seed(s) dropped corrupt: {lost}")
    if unv:
        bits.append(f"{len(unv)} unverified seed(s) "
                    f"{'dropped' if STRICT else 'retained'}: {unv}")
    return ("   [" + "; ".join(bits) + "]") if bits else ""


def fmt(x: float | None, nd: int = 4) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:.{nd}f}"


def main(argv: list[str]) -> int:
    global STRICT
    args = [a for a in argv[1:] if not a.startswith("--")]
    STRICT = "--strict" in argv
    campaign = Path(args[0]) if args else ROOT / "campaigns/fedluar_hh"
    if not campaign.exists():
        print(f"no campaign at {campaign}")
        return 1
    runs = load(campaign)
    if not runs:
        print(f"no reports under {campaign}")
        return 1
    arms = by_arm(runs)

    # ---- byte denominator: eps0 (instrumented) with mono as the fallback ---
    e0 = [r for r in arms.get("eps0", []) if r.usable and r.up_mb]
    full_up = st.mean([r.up_mb for r in e0]) if e0 else (
        FULL_UPLINK_B * NUM_WORKERS / 1e6)
    dn_all = [r.down_mb for r in runs if r.down_mb]
    full_dn = st.mean(dn_all) if dn_all else full_up
    denom_src = "MEASURED (eps0)" if e0 else "DERIVED (analytic 3x749,401 B)"

    print("=" * 78)
    print(f"{campaign.name} — {len(runs)} runs, {len(arms)} arms   ({campaign})"
          f"{'   [STRICT: unverified partitions also excluded]' if STRICT else ''}")
    print("=" * 78)

    # ---- §0 health + partition integrity ----------------------------------
    bad = [r for r in runs if not r.healthy]
    print(f"\n§0a HEALTH GATE — {len(runs) - len(bad)}/{len(runs)} healthy")
    for r in bad:
        print(f"   EXCLUDED {r.arm}/seed{r.seed}: {r.why}")

    gate_seen = any(r.data != "UNVERIFIED" for r in runs)
    corrupt = [r for r in runs if r.data == "CORRUPT"]
    unver = [r for r in runs if r.data == "UNVERIFIED"]
    print(f"\n§0b PARTITION-INTEGRITY GATE — "
          f"{'ACTIVE' if gate_seen else 'NOT AVAILABLE (no gate file!)'}")
    if not gate_seen:
        print("   *** every cross-arm pairing below is UNGATED and may be "
              "comparing\n   *** an arm against a corrupted dataset. Run the "
              "zero-image scan first.")
    else:
        cnt = {v: sum(1 for r in runs if r.data == v)
               for v in ("clean", "clean_partial", "CORRUPT", "UNVERIFIED")}
        print(f"   clean {cnt['clean']} | clean_partial {cnt['clean_partial']} "
              f"| CORRUPT {cnt['CORRUPT']} | UNVERIFIED {cnt['UNVERIFIED']}")
        for r in sorted(corrupt, key=lambda r: (r.seed, r.arm)):
            print(f"   EXCLUDED (corrupt partition) {r.arm}/seed{r.seed}")
        if unver:
            u = ", ".join(f"{r.arm}/seed{r.seed}"
                          for r in sorted(unver, key=lambda r: (r.seed, r.arm)))
            print(f"   RETAINED BUT UNVERIFIED (no shard survived): {u}")
        print("   Corrupt = all-zero training images injected while labels stay "
              "intact\n   (1.4-14.0% of rows). Such a cell trains on injected "
              "label noise, so it is\n   dropped from every mean and every "
              "pairing below.")

    print(f"\n   uplink denominator = {fmt(full_up)} MB/round  [{denom_src}]")
    print(f"   downlink (invariant, no ImpRoute lever) = {fmt(full_dn)} MB/round")

    # ---- §1 the FedLUAR table ---------------------------------------------
    print("\n§1 ACCURACY AT A FIXED ROUND BUDGET vs MEASURED COMM "
          "(FedLUAR Table 2 shape)")
    print(f"   {'arm':<16}{'n/heal':>8}{'Acc% ± t-CI':>18}{'up MB/r':>10}"
          f"{'Comm_up':>9}{'Comm_tot':>10}{'expected':>10}  note")
    table: dict[str, dict[str, Any]] = {}
    for arm in sorted(arms):
        rs = arms[arm]
        heal = [r for r in rs if r.usable]
        accs = [r.acc for r in heal if r.acc is not None]
        mean, ci, n = ac.mean_ci(accs)
        ups = [r.up_mb for r in heal if r.up_mb]
        up = st.mean(ups) if ups else None
        if arm == "mono" and up is None:
            up, note = full_up, "DERIVED byte row (no telemetry)"
        else:
            note = ""
        comm_up = (up / full_up) if up else None
        comm_tot = ((up + full_dn) / (full_up + full_dn)) if up else None
        exp = expected_comm_up(arm)
        if exp is not None and comm_up is not None and abs(comm_up - exp) > 0.02:
            note = (note + " ").strip() + f"DEVIATES from analytic {exp:.4f}"
        table[arm] = {"acc": mean, "ci": ci, "n": n,
                      "comm_up": comm_up, "comm_tot": comm_tot}
        acc_s = (ac.NOT_EVALUABLE if mean is None
                 else f"{mean:>7.2f} ± {ci:>6.2f}")
        print(f"   {arm:<16}{len(rs):>4}/{len(heal):<3}{acc_s:>18}"
              f"{fmt(up):>10}{fmt(comm_up):>9}{fmt(comm_tot):>10}"
              f"{fmt(exp):>10}  {note}")
    print("   Comm_up = uplink / monolithic uplink (FedLUAR's scope: uplink "
          "ONLY).\n   Comm_tot additionally counts the constant downlink "
          "broadcast, which\n   no ImpRoute or FedLUAR knob touches — publish "
          "BOTH, never an unlabelled one.")

    # ---- §2 Pareto --------------------------------------------------------
    print("\n§2 PARETO PLANE (Comm_up, accuracy at fixed rounds) — "
          "who is strictly dominated")
    pts = [(a, v["comm_up"], v["acc"]) for a, v in table.items()
           if v["comm_up"] is not None and v["acc"] is not None]
    for a, c, acc in sorted(pts, key=lambda p: p[1]):
        doms = [b for b, cb, ab in pts
                if b != a and cb <= c and ab >= acc and (cb < c or ab > acc)]
        verdict = "on front" if not doms else f"DOMINATED by {', '.join(doms)}"
        print(f"   {a:<16} Comm_up {c:.4f}  acc {acc:6.2f}%   {verdict}")

    # ---- §3 selection rule at matched delta -------------------------------
    print("\n§3 SELECTION RULE AT MATCHED delta — the campaign's question")
    print("   Same count, same global set, same recycle fill, same eps=0, "
          "same aging=off.\n   The ONLY difference is how the skipped layers "
          "are chosen.")
    deltas = sorted({int(a.split("_d")[1]) for a in arms
                     if a.startswith("fedluar_d")})
    for base_tag, base_name in (("luarand", "uniform i.i.d. (their Table 4)"),
                                ("luarcyc", "round-robin (they never ran it)")):
        fam: dict[str, list[float]] = {}
        seeds_by: dict[str, list[int]] = {}
        notes: list[str] = []
        for d in deltas:
            a, b = f"fedluar_d{d}", f"{base_tag}_d{d}"
            if a not in arms or b not in arms:
                continue
            diffs, seeds = paired(arms[a], arms[b])
            note = gate_note(arms[a], arms[b], seeds)
            if note:
                notes.append(f"   d{d}:{note}")
            if diffs:
                fam[f"d{d}"] = diffs
                seeds_by[f"d{d}"] = seeds
        if not fam:
            continue
        print(f"\n   FedLUAR minus {base_tag}  ({base_name})")
        print("   positive ⇒ the importance metric earns its keep")
        print(ac.format_family(
            ac.contrast_family(fam, seeds_by, direction="greater")))
        for ln in notes:
            print(ln)

    # luarcyc vs luarand: bounded vs unbounded staleness, both blind
    fam, seeds_by, notes = {}, {}, []
    for d in deltas:
        a, b = f"luarcyc_d{d}", f"luarand_d{d}"
        if a in arms and b in arms:
            diffs, seeds = paired(arms[a], arms[b])
            note = gate_note(arms[a], arms[b], seeds)
            if note:
                notes.append(f"   d{d}:{note}")
            if diffs:
                fam[f"d{d}"], seeds_by[f"d{d}"] = diffs, seeds
    if fam:
        print("\n   round-robin minus uniform (both blind; bounded vs "
              "unbounded staleness)")
        print(ac.format_family(
            ac.contrast_family(fam, seeds_by, direction="greater")))
        for ln in notes:
            print(ln)

    # ---- §4 fill-rule ablation -------------------------------------------
    print("\n§4 FILL RULE AT MATCHED BYTES AND MATCHED ROTATION "
          "(our FedLUAR Table 5)")
    print("   luarcyc_d(14-k) vs cyclic_k: identical rotation and identical "
          "bytes;\n   recycle fill (theta + Delta_prev) vs freeze fill "
          "(layer held bitwise).")
    fam, seeds_by, notes = {}, {}, []
    for k in (7, 3):
        a, b = f"luarcyc_d{NUM_LAYERS - k}", f"cyclic_k{k}"
        if a in arms and b in arms:
            diffs, seeds = paired(arms[a], arms[b])
            note = gate_note(arms[a], arms[b], seeds)
            if note:
                notes.append(f"   k{k}:{note}")
            if diffs:
                fam[f"k{k}"], seeds_by[f"k{k}"] = diffs, seeds
    print(ac.format_family(ac.contrast_family(
        fam, seeds_by, direction="greater")) if fam else "   (no pairs yet)")
    for ln in notes:
        print(ln)

    # ---- §5 ImpRoute vs the FedLUAR family at matched measured Comm -------
    print("\n§5 IMPROUTE vs THE FEDLUAR FAMILY, matched on MEASURED Comm_up")
    imp = sorted(a for a in arms if a.startswith("improute_tau"))
    luar = sorted(a for a in arms if a.startswith(("fedluar_d", "luarand_d",
                                                   "luarcyc_d")))
    for a in imp:
        ca = table.get(a, {}).get("comm_up")
        aa = table.get(a, {}).get("acc")
        if ca is None or aa is None:
            print(f"   {a:<16} NOT EVALUABLE")
            continue
        near = sorted(
            (abs(table[b]["comm_up"] - ca), b) for b in luar
            if table.get(b, {}).get("comm_up") is not None
            and table.get(b, {}).get("acc") is not None
        )[:3]
        print(f"   {a:<16} Comm_up {ca:.4f}  acc {aa:6.2f}%   nearest:")
        for gap, b in near:
            db = table[b]["acc"] - aa
            print(f"       {b:<16} Comm_up {table[b]['comm_up']:.4f} "
                  f"(Δcomm {gap:+.4f})  acc {table[b]['acc']:6.2f}%  "
                  f"Δacc {db:+6.2f} pp")
    print("\n   Cross-arm gaps here are UNPAIRED (different byte budgets), so "
          "they are\n   descriptive.  The paired, Holm-corrected inference is "
          "§3 and §4 only.")

    # ---- caveats that must travel with every number ----------------------
    print("\n" + "=" * 78)
    print("SCOPE CONDITIONS")
    print("  * Sampling universe is per-TENSOR (14), not per-LAYER (7 "
          "kernel+bias groups).\n    Applied identically to all four selection "
          "rules, so §3/§4 are still clean —\n    but do NOT compare these "
          "absolute Comm values to the paper's.")
    print("  * Byte axis is a coarse staircase: five 147.5 kB conv kernels are "
          "98.4% of\n    the 749,401 B payload, so uplink takes few distinct "
          "values.")
    print("  * Control-plane bytes (manifest, skip advice) are on the wire and "
          "in no field\n    (BYTE-09): every byte figure is payload-only, "
          "which flatters the shedding arms.")
    print("  * Round 0 carries no advice (R_0 = empty, as in the paper): every "
          "fedluar* arm\n    sends the full model once.  Steady state is read "
          "from rounds>=3, cycle-aligned.")
    print("  * Partition gate applied (§0b): 11 cells trained on shards with "
          "all-zero images\n    injected at 1.4-14.0% and labels left intact. "
          "They are excluded, which\n    lowers n on exactly the arms this "
          "campaign exists to test — read n per row,\n    never the nominal "
          "n=3.  4 further cells kept no shard and are UNVERIFIED:\n    "
          "retained, but they are not evidence of a clean partition.")
    print("  * n=3 on every tier-1..3 arm: the exact sign-flip floor is 0.125 "
          "> alpha, so the\n    ROBUST tier is unreachable there BY "
          "CONSTRUCTION.  Only tier 0 (n=6) can\n    support a headline "
          "claim.")
    print("  * No wall-clock is reported here on purpose.  On this testbed the "
          "wire is ~2.7%\n    of a round and the round-time noise floor "
          "(±10%) exceeds the effects.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
