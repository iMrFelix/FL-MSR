#!/usr/bin/env python3
"""Forensic partition-integrity gate for a campaign whose shards still exist.

This is the REPRODUCIBLE form of the ad-hoc scan that produced
`campaigns/fedluar_hh_partition_gate.json`.  Same detector, same verdicts —
written into the repo so the gate behind a published number can be re-derived
rather than taken on trust.

DETECTOR: count rows of `x_train` / `x_val` that are exactly all-zero.

That is the physical signature of the corruption (`kb/13` §13.5): concurrent
`tf.keras.datasets.cifar10.load_data()` calls race on the keras cache and a
reader observes still-zero-filled pages of a re-extracting archive, so 1.35-14%
of the images become black while `y_train` stays byte-identical to the clean
partition.  Shard size, class balance and the label histogram all look perfect.

WHY NOT COMPARE HASHES AGAINST SIBLING RUNS: because that needs a majority of
clean siblings to vote, and where support was thin it mis-called cells in BOTH
directions — two independent implementations of the hash rule agreed with each
other and were both wrong at the same seed.  Counting zero rows needs no
sibling, works at n=1, and is what the verdicts below actually rest on.

VERDICTS
  clean          4/4 surviving shards free of zero images
  clean_partial  <4 shards survived, all clean; unsurveyed nodes unknown
  CORRUPT        zero-image injection measured  -> excluded from every analysis
  UNVERIFIED     no shard survived locally; unknown, and NOT the same as clean

Note the UNVERIFIED tier exists only because this gate is FORENSIC — run after
the fact on whatever survived the pull.  Campaigns materialised through
`scripts/prematerialize.py` get a prospective gate instead and cannot have one.

Usage:
    python -m scripts.audit_partitions campaigns/fedluar_hh
        [-o campaigns/fedluar_hh_partition_gate.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scan_shard(path: Path) -> dict:
    with np.load(path) as d:
        out: dict = {}
        for split in ("x_train", "x_val"):
            x = d[split]
            n = len(x)
            z = int((x.reshape(n, -1) == 0).all(1).sum()) if n else 0
            out[split] = {"n": int(n), "zero_rows": z,
                          "zero_pct": round(100.0 * z / n, 3) if n else 0.0}
        out["corrupt"] = bool(out["x_train"]["zero_rows"]
                              or out["x_val"]["zero_rows"])
    return out


class ExpectNodesMismatch(RuntimeError):
    """A cell holds MORE shards than --expect-nodes claims the campaign has.

    This is a misconfiguration, not a data condition, and it must be fatal.
    With `len(shards) >= expect_nodes` deciding "clean", a 10-node campaign
    audited at the default 4 reports every cell clean — including cells that
    are missing half their shards — and exits 0. Observed live on `n10pilot`:
    "clean | nodes_scanned=10 | 10/4 shards free", rc 0.

    Refusing to emit a gate file at all is the only safe response: a
    silently-wrong gate is worse than no gate, because everything downstream
    treats it as authoritative.
    """


def audit(campaign: Path, expect_nodes: int = 4) -> dict:
    # Fail BEFORE producing anything. Checking per-cell and continuing would
    # still write a file whose "clean" verdicts cannot be trusted.
    over = {}
    for run_dir in sorted(p for p in campaign.glob("*/seed*") if p.is_dir()):
        n = len(list((run_dir / "data").glob("node-*.npz")))
        if n > expect_nodes:
            over[f"{run_dir.parent.name}/{run_dir.name}"] = n
    if over:
        worst = max(over.values())
        raise ExpectNodesMismatch(
            f"{len(over)} cell(s) hold more shards than --expect-nodes="
            f"{expect_nodes} (largest: {worst}). Every 'clean' verdict would be "
            f"unreliable, so no gate file has been written. Re-run with "
            f"--expect-nodes {worst}. Offending cells: "
            + ", ".join(f"{k}={v}" for k, v in sorted(over.items())[:5])
            + (" …" if len(over) > 5 else "")
        )

    cells: dict = {}
    for run_dir in sorted(p for p in campaign.glob("*/seed*") if p.is_dir()):
        cell = f"{run_dir.parent.name}/{run_dir.name}"
        shards = sorted((run_dir / "data").glob("node-*.npz"))
        if not shards:
            cells[cell] = {"verdict": "UNVERIFIED", "nodes_scanned": 0,
                           "zero_pct_by_node": {},
                           "reason": "no partition survived locally; "
                                     "integrity unknown"}
            continue
        per_node, bad = {}, {}
        for s in shards:
            try:
                r = scan_shard(s)
            except Exception as e:  # truncated / unreadable
                per_node[s.name] = {"error": repr(e)}
                bad[s.name] = -1.0
                continue
            per_node[s.name] = r
            if r["corrupt"]:
                bad[s.name] = r["x_train"]["zero_pct"]
        if bad:
            lo, hi = min(bad.values()), max(bad.values())
            verdict = "CORRUPT"
            reason = (f"all-zero images on {len(bad)}/{len(shards)} surviving "
                      f"shards ({lo:.2f}-{hi:.2f}% of x_train rows)")
        elif len(shards) >= expect_nodes:
            verdict, reason = "clean", (f"{len(shards)}/{expect_nodes} shards "
                                        "free of zero-image injection")
        else:
            verdict, reason = "clean_partial", (
                f"{len(shards)}/{expect_nodes} shards survived, all clean; "
                "unsurveyed nodes unknown")
        cells[cell] = {"verdict": verdict, "nodes_scanned": len(shards),
                       "zero_pct_by_node": bad, "reason": reason,
                       "per_node": per_node}
    counts = {v: sum(1 for c in cells.values() if c["verdict"] == v)
              for v in ("clean", "clean_partial", "CORRUPT", "UNVERIFIED")}
    return {"detector": "exact all-zero rows in x_train/x_val "
                        "(arm-independent; no sibling vote)",
            "campaign": str(campaign), "counts": counts, "cells": cells}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--expect-nodes", type=int, default=4,
                    help="nodes per cell; a cell with MORE than this aborts "
                         "the run (see ExpectNodesMismatch)")
    ap.add_argument("--force", action="store_true",
                    help="permit overwriting an existing gate file")
    a = ap.parse_args()

    try:
        doc = audit(a.campaign, a.expect_nodes)
    except ExpectNodesMismatch as e:
        print(f"ABORTED: {e}")
        return 2

    out = a.out or a.campaign.parent / f"{a.campaign.name}_partition_gate.json"
    # The default path IS the published gate that analyzers read. A bare
    # re-derivation must not silently overwrite the record it is being used to
    # check — that would destroy the artifact and the check in one step.
    if out.exists() and not a.force:
        print(f"REFUSING to overwrite existing gate {out}\n"
              f"  It is the published record that analyzers read. Either send "
              f"the re-derivation elsewhere with -o, or pass --force if you "
              f"genuinely intend to replace it.")
        return 3
    out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps(doc["counts"], indent=1))
    print(f"\n{sum(doc['counts'].values())} cells -> {out}")
    print(f"  nodes expected per cell: {a.expect_nodes} (--expect-nodes)")
    for cell, v in sorted(doc["cells"].items()):
        if v["verdict"] == "CORRUPT":
            print(f"  CORRUPT {cell}: {v['reason']}")
    # Exit nonzero when anything is corrupt, so this can gate a pipeline.
    # Previously it always returned 0, which made `&&` chaining silently unsafe.
    return 1 if doc["counts"]["CORRUPT"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
