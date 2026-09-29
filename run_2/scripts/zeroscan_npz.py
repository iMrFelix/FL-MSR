#!/usr/bin/env python
"""Zero-row corruption scan for materialised partition shards.

Detector of record for the silent partition corruption confirmed 2026-08-06
(writeup/23 section 2, corrected signature): a corrupted materialisation has
1.35-14% of its x_train rows replaced by ALL-ZERO images (labels intact).
Clean shards contain exactly zero all-zero rows. The scan therefore needs no
sibling or canonical copy: count rows in x_train/x_val that are exactly zero.

Scans the LOCAL pull mirror (campaigns/<campaign>/<arm>/seed*/data/node-*.npz),
which retains shards after the boxes delete them post-run. Idempotent: results
are keyed by (path, size, mtime) in campaigns/<campaign>_zeroscan.json and
unchanged files are skipped, so a cron/monitor loop only pays for new shards.

Usage:
  .venv/bin/python scripts/zeroscan_npz.py                  # n10pilot + w3clean
  .venv/bin/python scripts/zeroscan_npz.py --campaigns X,Y --quiet-clean

stdout: one line per newly scanned shard ("clean ..." suppressed under
--quiet-clean); CORRUPT/ERROR/EMPTY/NO_FILES lines always print. Intended to
sit under a Monitor that forwards only actionable lines.

Exit codes: 3 = corruption found (dominates); 4 = errors, empty shards, or
nothing matched (NOT a clean verdict); 0 = every matched shard scanned clean.
0 never means "scanned nothing".
"""
import argparse
import json
import os
import sys
import time
from glob import glob

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def zero_rows(arr):
    if arr.ndim < 2 or arr.shape[0] == 0:
        return 0, int(arr.shape[0]) if arr.ndim else 0
    flat = arr.reshape(arr.shape[0], -1)
    return int(np.count_nonzero(~flat.any(axis=1))), int(arr.shape[0])


def scan_campaign(campaign, quiet_clean):
    root = os.path.join(REPO, "campaigns", campaign)
    out_path = os.path.join(REPO, "campaigns", f"{campaign}_zeroscan.json")
    record = {"campaign": campaign,
              "detector": "exact all-zero rows in x_train/x_val; clean == 0 "
                          "(writeup/23 s2, corrected signature 2026-08-06)",
              "script": "scripts/zeroscan_npz.py",
              "scans": {}}
    if os.path.exists(out_path):
        with open(out_path) as f:
            record = json.load(f)
    scans = record.setdefault("scans", {})

    new = corrupt = errors = 0
    files = sorted(glob(os.path.join(root, "*", "seed*", "data", "node-*.npz")))
    if not os.path.isdir(root) or not files:
        # A scan that finds nothing is NOT a clean verdict — say so loudly
        # and exit nonzero so a monitor/caller can tell it from "all clean".
        print(f"NO_FILES {campaign}: root={'missing' if not os.path.isdir(root) else 'present'}, 0 shards matched", flush=True)
        return 0, 0, len(scans), 1
    for npz in files:
        rel = os.path.relpath(npz, REPO)
        st = os.stat(npz)
        prior = scans.get(rel)
        # unchanged files skip — EXCEPT prior errors, which are always retried
        if (prior and prior.get("verdict") != "error"
                and prior.get("size") == st.st_size and prior.get("mtime") == st.st_mtime):
            if prior.get("verdict") == "corrupt":
                corrupt += 1
            continue
        entry = {"size": st.st_size, "mtime": st.st_mtime,
                 "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            with np.load(npz) as z:
                zt, nt = zero_rows(z["x_train"])
                zv, nv = zero_rows(z["x_val"]) if "x_val" in z.files else (0, 0)
            # empty x_train is NOT clean — launcher treats len(x)==0 as
            # unusable; mirror that instead of certifying it.
            entry.update(n_train=nt, zero_train=zt, n_val=nv, zero_val=zv,
                         verdict="corrupt" if (zt or zv)
                         else ("empty" if nt == 0 else "clean"))
        except Exception as e:  # truncated mid-rsync copies etc.
            entry.update(verdict="error", error=f"{type(e).__name__}: {e}")
            print(f"ERROR {rel} {entry['error']}", flush=True)
            scans[rel] = entry
            errors += 1
            continue
        scans[rel] = entry
        new += 1
        if entry["verdict"] == "corrupt":
            corrupt += 1
            print(f"CORRUPT {rel} zero_train={zt}/{nt} zero_val={zv}/{nv}", flush=True)
        elif entry["verdict"] == "empty":
            errors += 1
            print(f"EMPTY {rel} n_train=0 (unusable per launcher semantics)", flush=True)
        elif not quiet_clean:
            print(f"clean {rel} train=0/{nt} val=0/{nv}", flush=True)

    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1, sort_keys=True)
    os.replace(tmp, out_path)
    return new, corrupt, len(scans), errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaigns", default="n10pilot,w3clean")
    ap.add_argument("--quiet-clean", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    total_corrupt = total_errors = 0
    for campaign in args.campaigns.split(","):
        new, corrupt, seen, errs = scan_campaign(campaign.strip(), args.quiet_clean)
        total_corrupt += corrupt
        total_errors += errs
        if args.summary:
            print(f"summary {campaign}: shards_seen={seen} new={new} "
                  f"corrupt_total={corrupt} errors={errs}", flush=True)
    # exit: 3 = corruption found (dominates), 4 = errors/no-files (NOT clean),
    # 0 = every matched shard scanned clean. 0 must never mean "scanned nothing".
    sys.exit(3 if total_corrupt else (4 if total_errors else 0))


if __name__ == "__main__":
    main()
