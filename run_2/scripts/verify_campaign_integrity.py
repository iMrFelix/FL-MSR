#!/usr/bin/env python3
"""Post-run integrity check for a pre-materialised campaign.

Closes the one hole the independent review of the partition fix identified:
if `src/launcher.py:_existing_partitions` REJECTS a shard at adoption time, the
pre-hard-fail code path fell through to `prepare_partitions()` in-process while
sibling runs were live — a silent concurrent regeneration whose only trace is a
warning in `run.log`, and which the pre-run manifest would not reflect.

`FL_REQUIRE_PREMATERIALIZED=1` now turns that into a loud failure, but this
script is the *independent* check: it does not trust the flag, it compares what
was on disk BEFORE training against what was on disk AFTER, per cell.

THREE CHECKS PER CELL
  1. HASH CONTINUITY — every node's `x_sha256`/`y_sha256` in the post-run
     `results/data_digests.json` must equal the pre-run
     `PARTITION_MANIFEST.json`. A mismatch means the partition changed under
     the run: regenerated, swapped, or corrupted mid-flight.
  2. NO REGENERATION TRACE — `run.log` must not contain "regenerating",
     "Discarding", or "Partitioning <dataset>", any of which mean the run
     materialised its own data rather than adopting the verified copy.
  3. ZERO-IMAGE FREEDOM — the post-run digest's `zero_rows` (where the runner
     records it) must be 0.

Any cell failing any check is EXCLUDED from analysis, not repaired: a cell that
re-materialised is exactly a cell that raced.

⚠ KNOWN LIMIT OF CHECK 1 — READ BEFORE TRUSTING IT.
`np.savez` regeneration is BYTE-DETERMINISTIC (established by independent
review, 2026-08-06). So a run that silently re-materialised from the SAME config
produces the SAME hashes, and check 1 cannot detect it — not by oversight, but
in principle. Hash continuity only catches regeneration from a DIFFERENT config,
or a regeneration that was itself corrupted.

The controls that DO work against same-config regeneration are check 2 (the
run.log trace) and, primarily, `FL_REQUIRE_PREMATERIALIZED=1` in
`src/launcher.py`, which refuses to re-materialise at all. This is why the
architecture had to be prevention rather than detection: for the single most
likely failure mode, after-the-fact hashing is blind by construction.

Exit code 0 = every cell clean; 1 = at least one cell must be excluded.

Usage:  python -m scripts.verify_campaign_integrity campaigns/fedluar_late
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REGEN_MARKERS = ("regenerating", "Discarding", "Partitioning cifar10",
                 "Partitioning mnist", "Partitioning femnist")


def check(campaign: Path) -> tuple[dict, list[str]]:
    mf = campaign / "PARTITION_MANIFEST.json"
    if not mf.exists():
        raise SystemExit(f"no PARTITION_MANIFEST.json in {campaign} — this "
                         "campaign was not pre-materialised, so there is "
                         "nothing to verify against. Use "
                         "scripts/audit_partitions.py instead.")
    manifest = json.loads(mf.read_text())
    results: dict = {}
    bad: list[str] = []

    for cell, pre in sorted(manifest.items()):
        run_dir = campaign / cell
        issues: list[str] = []

        # 2. regeneration trace
        log = run_dir / "run.log"
        if log.exists():
            text = log.read_text(errors="replace")
            hits = [m for m in REGEN_MARKERS if m in text]
            if hits:
                issues.append(f"run.log contains {hits} — the run materialised "
                              "its own data instead of adopting the verified copy")
        else:
            issues.append("no run.log")

        # 1. + 3. digest continuity and zero-freedom
        dg = run_dir / "results" / "data_digests.json"
        if not dg.exists():
            issues.append("no results/data_digests.json (cannot confirm the "
                          "partition was unchanged by the run)")
        else:
            post = json.loads(dg.read_text())
            for node, pre_s in sorted(pre.items()):
                post_s = post.get(node)
                if post_s is None:
                    issues.append(f"{node}: missing from post-run digest")
                    continue
                # An ABSENT hash must fail, not pass. Comparing `a and b and
                # a != b` silently accepts a digest that simply omits the
                # field — instrumentation that looks like it checks something
                # and does not (kb/11 §11.4). Found by independent review.
                for key in ("x_sha256", "y_sha256"):
                    a, b = pre_s.get(key), post_s.get(key)
                    if not a:
                        issues.append(f"{node}: {key} absent from the MANIFEST "
                                      "— nothing to compare against")
                    elif not b:
                        issues.append(f"{node}: {key} absent from the POST-RUN "
                                      "digest — cannot confirm continuity")
                    elif a != b:
                        issues.append(f"{node}: {key} CHANGED under the run "
                                      f"({a[:12]}… -> {b[:12]}…)")
                if "zero_rows" not in post_s:
                    issues.append(f"{node}: post-run digest records no "
                                  "zero_rows field (runner predates the "
                                  "detector) — zero-freedom unconfirmed")
                elif post_s["zero_rows"]:
                    issues.append(f"{node}: {post_s['zero_rows']} zero images "
                                  "post-run")

        results[cell] = issues
        if issues:
            bad.append(cell)

    # 4. COVERAGE — a cell present on disk but absent from the manifest was
    # never pre-materialised, so nothing above would ever look at it. Silently
    # skipping it is the same failure as passing an absent hash.
    for run_dir in sorted(p for p in campaign.glob("*/seed*") if p.is_dir()):
        cell = f"{run_dir.parent.name}/{run_dir.name}"
        if cell in manifest:
            continue
        if not (run_dir / "results" / "report.json").exists():
            continue                      # not a completed run; nothing to gate
        results[cell] = ["present on disk with a report, but ABSENT from "
                         "PARTITION_MANIFEST.json — it was never "
                         "pre-materialised, so its partition is unverified"]
        bad.append(cell)
    return results, bad


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    campaign = Path(argv[1])
    results, bad = check(campaign)

    print(f"post-run integrity — {campaign}")
    print(f"  cells checked: {len(results)}   clean: {len(results) - len(bad)}"
          f"   MUST EXCLUDE: {len(bad)}")
    for cell, issues in sorted(results.items()):
        if issues:
            print(f"\n  EXCLUDE {cell}")
            for i in issues:
                print(f"     - {i}")
    if not bad:
        print("\n  Every cell trained on the exact partition that was verified "
              "before launch:\n  hashes continuous, no regeneration trace, no "
              "zero images. Safe to analyse.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
