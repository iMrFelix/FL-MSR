#!/usr/bin/env python3
"""Relocate VERIFIED partition-shard residue to external storage.

WHY THIS EXISTS.  `campaigns/**/data/` is ~1 GB per cell and is the sole
surviving copy of the partitions each run actually trained on — the boxes boot
ephemeral overlays and delete `data/` themselves after every run.  The standing
rule is therefore that this residue is never deleted.  When the internal volume
ran short, the tempting move was to prune cells we had already scanned.  This
script does the safe version of that instead: COPY to the external archive,
VERIFY the copy against the hashes the run itself recorded, and only then
remove the internal original.  No evidence is destroyed at any point; if the
script dies midway the worst outcome is a duplicate, never a loss.

WHAT IT REFUSES TO TOUCH.  A cell is eligible only if ALL of these hold:

  * `results/data_digests.json` exists, parses, and has an entry for every
    shard on disk (a cell whose digest step crashed is exactly the cell whose
    integrity is unknown — those stay put);
  * every shard's sha256(x_train) reproduces the digest's `x_sha256`, and its
    row count reproduces `n`.  This is the identity proof: it establishes that
    the residue IS the data the run used, not a later re-materialisation;
  * a zero-row verdict of 0 exists for the cell — either `zero_rows: 0` in the
    digest itself, or an entry in a --zero-scan JSON showing 0 on both splits.

Anything failing any check is REPORTED AND SKIPPED, never moved.

WHAT IT LEAVES BEHIND.  A `data.RELOCATED.json` breadcrumb where `data/` was,
naming the archive path and carrying the verified hashes.  Without it a later
reader finds a cell with no shards and cannot tell relocation from loss —
which is the same ambiguity that made the first-generation residue confusing.

DEFAULT IS A DRY RUN.  `--apply` is required to touch anything.

Usage:
  python -m scripts.relocate_verified_residue campaigns/stab \\
      --dest EXPERIMENT_DATA/residue-archive \\
      --zero-scan path/to/stab_zero_scan.json
  # add --apply once the dry run reads correctly
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Keep a healthy margin on the destination: refuse to start a copy that could
# fill the archive volume, since a truncated npz in the ARCHIVE would be worse
# than one on the internal disk — it is the copy we would keep.
MIN_DEST_FREE_GIB = 5


def sha256_x(path: Path) -> tuple[str, int]:
    """sha256 of x_train exactly as the runners compute it, plus its row count."""
    with np.load(path) as d:
        x = d["x_train"]
        return hashlib.sha256(np.ascontiguousarray(x)).hexdigest(), int(len(x))


def sha256_file(path: Path) -> str:
    """R-b — sha256 of the WHOLE FILE, streamed.

    The x_train hash above proves IDENTITY (this shard is the one the run
    recorded) but says nothing about the rest of the file: y arrays, x_val,
    x_test, and the zip container itself all sit outside it. A copy corrupted
    anywhere in that region would pass an x_train-only check, the originals
    would be removed, and the damaged copy is the one we would keep.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def free_gib(p: Path) -> float:
    st = shutil.disk_usage(p)
    return st.free / (1024 ** 3)


def eligible(cell: Path, zero_scan: dict | None) -> tuple[bool, str, dict]:
    """Decide whether a cell may be relocated. Returns (ok, reason, digest)."""
    dig_p = cell / "results/data_digests.json"
    if not dig_p.exists():
        return False, "no data_digests.json (integrity unknown)", {}
    try:
        dig = json.loads(dig_p.read_text())
    except Exception as e:  # noqa: BLE001
        return False, f"digest unreadable: {e}", {}

    shards = sorted((cell / "data").glob("node-*.npz"))
    if not shards:
        return False, "no shards on disk", dig
    missing = [s.name for s in shards if s.name not in dig]
    if missing:
        return False, f"shards absent from digest: {missing}", dig
    # R-a — completeness must be checked in BOTH directions. The line above
    # catches a shard the digest never recorded; this one catches the opposite,
    # a cell whose residue is PARTIAL because some shards were pulled and
    # others were not. Without it such a cell relocates looking complete, the
    # breadcrumb records only the shards that happened to be there, and the
    # absence becomes indistinguishable from a partition that never existed.
    absent = sorted(k for k in dig if k not in {s.name for s in shards})
    if absent:
        return False, (f"partial residue (digest lists {len(dig)} shards, "
                       f"disk has {len(shards)}; missing {absent})"), dig

    # Zero-row verdict. Prefer the digest's own field; fall back to a scan file.
    key = str(cell.relative_to(cell.parents[1]))
    if all("zero_rows" in v for v in dig.values()):
        if any(v["zero_rows"] for v in dig.values()):
            return False, "digest reports zero rows > 0 — CORRUPT, keep in place", dig
    elif zero_scan is not None and key in zero_scan:
        rec = zero_scan[key]
        bad = [s for s, v in rec.items()
               if v.get("x_train", {}).get("zero") or v.get("x_val", {}).get("zero")]
        if bad:
            return False, f"zero-scan reports zero rows in {bad} — keep in place", dig
        if set(rec) != {s.name for s in shards}:
            return False, "zero-scan shard set does not match disk", dig
    else:
        return False, "no zero-row verdict (digest lacks zero_rows, no scan entry)", dig

    return True, "", dig


def verify(shards: list[Path], dig: dict) -> list[str]:
    """Return a list of failures; empty means every shard matches its digest."""
    bad = []
    for s in shards:
        h, n = sha256_x(s)
        if h != dig[s.name].get("x_sha256"):
            bad.append(f"{s.name}: x_sha256 mismatch")
        if n != dig[s.name].get("n"):
            bad.append(f"{s.name}: n {n} != digest {dig[s.name].get('n')}")
    return bad


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", type=Path)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--zero-scan", type=Path, default=None)
    ap.add_argument("--apply", action="store_true",
                    help="actually copy/verify/remove (default is a dry run)")
    a = ap.parse_args(argv[1:])

    campaign = a.campaign if a.campaign.is_absolute() else ROOT / a.campaign
    dest_root = a.dest if a.dest.is_absolute() else ROOT / a.dest
    if a.apply:
        # Resolve through the symlink and prove we can write BEFORE promising
        # anything: an unwritable archive discovered halfway through is how a
        # copy-then-remove turns into a remove.
        dest_root.mkdir(parents=True, exist_ok=True)
        probe = dest_root / ".write-probe"
        try:
            probe.write_text("ok")
            probe.unlink()
        except Exception as e:  # noqa: BLE001
            print(f"REFUSING: destination not writable: {dest_root} ({e})")
            return 2
        print(f"destination OK: {dest_root}  free={free_gib(dest_root):.1f} GiB")

    zero_scan = json.loads(a.zero_scan.read_text()) if a.zero_scan else None

    cells = sorted(p.parent for p in campaign.glob("*/seed*/data") if p.is_dir())
    print(f"{campaign}: {len(cells)} cells with residue"
          f"{'' if a.apply else '   [DRY RUN — nothing will be touched]'}")

    moved = skipped = failed = 0
    for cell in cells:
        name = str(cell.relative_to(campaign))
        ok, why, dig = eligible(cell, zero_scan)
        if not ok:
            print(f"  SKIP  {name:<28} {why}")
            skipped += 1
            continue

        shards = sorted((cell / "data").glob("node-*.npz"))
        bad = verify(shards, dig)
        if bad:
            print(f"  FAIL  {name:<28} identity check failed: {bad}")
            failed += 1
            continue

        target = dest_root / campaign.name / name / "data"
        if not a.apply:
            print(f"  would move {name:<28} {len(shards)} shards -> {target}")
            moved += 1
            continue

        need = sum(s.stat().st_size for s in shards) / (1024 ** 3)
        if free_gib(dest_root) - need < MIN_DEST_FREE_GIB:
            print(f"  STOP  {name}: destination would drop below "
                  f"{MIN_DEST_FREE_GIB} GiB free — refusing")
            return 2

        # Whole-file hashes of the SOURCES, taken before the copy so the
        # comparison afterwards is against what we actually read.
        src_whole = {s.name: sha256_file(s) for s in shards}

        target.mkdir(parents=True, exist_ok=True)
        for s in shards:
            shutil.copy2(s, target / s.name)          # COPY, never mv

        # TWO independent post-copy checks, and the removal is earned only if
        # BOTH pass. `verify` re-binds the copy to the run's own recorded
        # digest (identity); the whole-file comparison catches damage anywhere
        # outside x_train, which the digest schema cannot see at all.
        bad = verify(sorted(target.glob("node-*.npz")), dig)
        cp_bad = [n for n, h in src_whole.items()
                  if sha256_file(target / n) != h]
        if bad or cp_bad:
            print(f"  FAIL  {name}: post-copy verification failed "
                  f"(identity={bad or 'ok'}, whole-file={cp_bad or 'ok'}) — "
                  "originals KEPT, archive copy left for inspection")
            failed += 1
            continue

        (cell / "data.RELOCATED.json").write_text(json.dumps({
            "archived_to": str(target),
            "shards": {s.name: {"x_sha256": dig[s.name]["x_sha256"],
                                "n": dig[s.name]["n"],
                                "file_sha256": src_whole[s.name]} for s in shards},
            "verified": "sha256(x_train) matched the run's digest before and "
                        "after copy; whole-file sha256 of source matched copy",
        }, indent=1))
        shutil.rmtree(cell / "data")
        print(f"  MOVED {name:<28} {len(shards)} shards -> {target}")
        moved += 1

    print(f"\n{'would move' if not a.apply else 'moved'}: {moved}   "
          f"skipped: {skipped}   failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
