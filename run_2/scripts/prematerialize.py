#!/usr/bin/env python3
"""Materialise every run's dataset partition SERIALLY, before any concurrency.

WHY
---
Every confirmed instance of silent shard corruption in this project came from
CONCURRENT materialisation.  Two runs calling
``tf.keras.datasets.cifar10.load_data()`` at the same time race on the keras
cache, and a reader can observe pages of a re-extracting archive that are still
zero-filled.  The result: 1.35-13.97% of ``x_train`` becomes all-zero images
while ``y_train`` stays byte-identical to the clean partition.  Shard size,
class balance and the label histogram all look perfect, so the run trains on
several percent injected label noise and reports a plausible number.  It cost
`fedluar_hh` 11 of 81 cells, and at that rate a 9-run campaign expects one
silent casualty — in the decisive campaign, that is unacceptable.

The single-process CIFAR cache warm-up in the runners was NOT sufficient: the
race re-occurred with the cache already staged (`kb/11` §11.6).

So: do the materialisation itself serially, once, in one process, and let the
concurrent launches reuse it.  ``src/launcher.py:_existing_partitions`` adopts
a cached partition only after verifying it is readable and free of zero
images, so a corrupt shard is regenerated rather than silently reused.

Every shard is then zero-row scanned and its SHA256 recorded.  The script exits
NONZERO if any shard is corrupt — wire it as a gate before the runner, never
alongside it.

Usage:
    python -m scripts.prematerialize configs/experiments/fedluar_late/specs.txt \\
        campaigns/fedluar_late
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.schema import load_config  # noqa: E402
from src.launcher import prepare_data  # noqa: E402


def scan(path: Path) -> dict:
    with np.load(path) as d:
        out: dict = {}
        corrupt = False
        for split in ("x_train", "x_val"):
            x = d[split]
            z = int((x.reshape(len(x), -1) == 0).all(1).sum())
            out[split] = {"n": int(len(x)), "zero_rows": z,
                          "zero_pct": round(100.0 * z / max(len(x), 1), 3)}
            corrupt = corrupt or z > 0
        out["x_sha256"] = hashlib.sha256(
            np.ascontiguousarray(d["x_train"])).hexdigest()
        out["y_sha256"] = hashlib.sha256(
            np.ascontiguousarray(d["y_train"])).hexdigest()
        out["corrupt"] = corrupt
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    specs, out_root = Path(argv[1]), Path(argv[2])
    rows = [ln.split("\t") for ln in specs.read_text().splitlines() if ln.strip()]

    manifest: dict = {}
    bad: list[str] = []
    print(f"pre-materialising {len(rows)} runs SERIALLY (one process)\n")
    for i, row in enumerate(rows, 1):
        arm, seed, cfgpath = row[2], row[3], row[4]
        run_dir = out_root / arm / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = load_config(str(ROOT / cfgpath))
        prepare_data(cfg, run_dir)          # idempotent + integrity-checked

        cell = f"{arm}/seed{seed}"
        manifest[cell] = {}
        for p in sorted((run_dir / "data").glob("node-*.npz")):
            s = scan(p)
            manifest[cell][p.name] = s
            if s["corrupt"]:
                bad.append(f"{cell}/{p.name}")
        flag = "  *** CORRUPT ***" if any(
            v["corrupt"] for v in manifest[cell].values()) else ""
        print(f"  [{i}/{len(rows)}] {cell}: "
              f"{len(manifest[cell])} shards{flag}", flush=True)

    dest = out_root / "PARTITION_MANIFEST.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"\nmanifest -> {dest}")

    if bad:
        print(f"\nFAILED: {len(bad)} corrupt shard(s) — DO NOT LAUNCH:")
        for b in bad:
            print(f"  {b}")
        return 1
    print(f"\nOK: {sum(len(v) for v in manifest.values())} shards across "
          f"{len(manifest)} runs, ZERO corruption. Safe to launch concurrently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
