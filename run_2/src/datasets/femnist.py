"""LEAF FEMNIST dataset with the natural per-writer federated partition.

FEMNIST (Federated Extended MNIST) is the LEAF benchmark's federated
character-recognition dataset: NIST Special Database 19 handwriting,
62 classes (10 digits, 26 uppercase, 26 lowercase), keyed by the WRITER
of each character.  Partitioning across nodes assigns whole writers to
nodes ("natural" non-IID) — no synthetic Dirichlet skew is needed.

Reference statistics (LEAF paper, Table 1 — Caldas et al., "LEAF: A
Benchmark for Federated Settings", arXiv:1812.01097v3, NeurIPS 2019
workshop; PDF in PRIOR_WORK/LEAF_Fed_Benchmark_2018.pdf):

    FEMNIST: 3,550 devices (writers), 805,263 total samples,
             226.83 mean / 88.94 stdev samples per device.

KNOWN DISCREPANCY (LEAF issue #49, open): the LEAF repository's own
preprocessing, run today on the canonical NIST S3 archives at
``--sf 1.0 -k 0``, yields 3,597 users / 817,851 samples — not the
published 3,550 / 805,263 (independently replicated, e.g. by the
FederatedScope paper, arXiv:2204.05011).  817,851 further includes a
mechanical defect: ``group_by_writer.py`` appends the first image of
every writer after the first twice (817,851 − 814,255 = 3,596 =
3,597 − 1 duplicates).  This module reproduces the LEAF method
*without* that duplication, giving

    3,597 writers / 814,255 samples
    (mean 226.37 / stdev 88.84 per writer)

where 814,255 equals the EMNIST ByClass character total (Cohen et al.
2017) exactly — every SD19 by_write image hash-matches a by_class
label.  A full (non-subsampled) build validates its counts against
these pipeline-expected constants and logs the Table 1 reference
alongside; set ``FEMNIST_STRICT_STATS=1`` to make a mismatch fatal.

Preprocessing pipeline (faithful re-implementation of the LEAF tooling,
``leaf/data/femnist/preprocess``):

    1. Download NIST SD19 ``by_class.zip`` and ``by_write.zip``
       (the same S3 objects the LEAF ``get_data.sh`` fetches).
    2. Index PNG members of both archives.  ``by_class`` provides the
       class label (hexadecimal directory name); only ``hsf_*``
       partitions are indexed — the ``train_<class>`` directories
       duplicate hsf images and are excluded, mirroring LEAF.
    3. MD5-hash every image's bytes in both archives and match
       ``by_write`` images (which carry the writer id) to ``by_class``
       images (which carry the label) by hash — LEAF's
       ``get_hashes.py`` + ``match_hashes.py`` + ``group_by_writer.py``.
       Samples are the matched ``by_write`` images.
    4. Decode each matched PNG, convert to grayscale, resize to 28x28
       with the Lanczos filter (PIL's former ``Image.ANTIALIAS``), keep
       raw 0..255 intensities (white background ~255).  LEAF's
       ``data_to_json.py`` stores the same pixels scaled to [0, 1];
       we store uint8 and scale at load time (``load_from_file``).
    5. Cache the consolidated result (one compressed .npz keyed by
       writer) so steps 1-4 run once per machine.

Partition strategies:

    - ``natural`` (canonical): writers are shuffled with the partition
      seed and split into ``total_nodes`` contiguous groups — each node
      holds whole writers; sample counts are naturally imbalanced.
    - ``iid`` (control): all samples pooled, shuffled, split evenly via
      the shared IID partitioner.
    - ``dirichlet`` / ``pathological`` are rejected: the natural
      partition is the point of FEMNIST (LEAF builds the federation
      from real writers).

Spec-first configuration: ``natural`` is a first-class
``dataset.partition.strategy`` literal in the experiment-config schema,
and writer subsampling is the first-class ``dataset.partition
.max_writers`` knob (LEAF's "small" versions subsample writers the same
way) — both flow through the launcher into ``prepare_partitions``.
The retired ``FEMNIST_PARTITION_OVERRIDE`` env knob now raises (spec
provenance must match realized behavior).  Remaining env knobs are
host-cache/debug only and must stay out of confirmatory-run
provenance:

    FEMNIST_CACHE_DIR           cache root (default ~/.cache/leaf_femnist)
    FEMNIST_MAX_WRITERS         debug fallback for max_writers (the spec
                                knob, when set, takes precedence)
    FEMNIST_SUBSAMPLE_FRACTION  keep a fraction of writers (seeded)
    FEMNIST_TEST_FRACTION       per-writer global-test holdout (default 0.1)
    FEMNIST_WORKERS             hash/decode worker processes
    FEMNIST_STRICT_STATS        1 = fail on full-build stat mismatch

Test-set semantics: LEAF's standard evaluation splits each user's
samples ("-t sample").  This framework requires one global test set
shared by all nodes, so a seeded ``test_fraction`` of every writer's
samples is held out and pooled into the global (x_test, y_test); the
remaining per-writer samples go to the writer's node, which then takes
the same 90/10 train/val split the sibling loaders use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

from .base import FederationDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LEAF paper Table 1 reference statistics (verified against
# PRIOR_WORK/LEAF_Fed_Benchmark_2018.pdf, page 3).
# ---------------------------------------------------------------------------
LEAF_FEMNIST_TOTAL_WRITERS = 3_550
LEAF_FEMNIST_TOTAL_SAMPLES = 805_263
LEAF_FEMNIST_MEAN_SAMPLES_PER_WRITER = 226.83
LEAF_FEMNIST_STDEV_SAMPLES_PER_WRITER = 88.94

# What the LEAF method actually yields on the canonical NIST archives
# (module docstring, "KNOWN DISCREPANCY"): the duplication-free count
# equals the EMNIST ByClass total; the LEAF repo's own pipeline adds
# one duplicate per writer beyond the first (issue #49).
PIPELINE_EXPECTED_WRITERS = 3_597
PIPELINE_EXPECTED_SAMPLES = 814_255          # == EMNIST ByClass total
EMNIST_BYCLASS_TOTAL = 814_255               # Cohen et al. 2017
LEAF_REPO_PIPELINE_SAMPLES = 817_851         # LEAF issue #49 observation

NUM_CLASSES = 62
IMAGE_SIZE = 28

# NIST SD19 archives — the same objects LEAF's get_data.sh downloads.
_NIST_BASE_URL = "https://s3.amazonaws.com/nist-srd/SD19"
_ARCHIVES = {
    # name -> (url, expected size in bytes, observed 2026-06-10)
    "by_class.zip": (f"{_NIST_BASE_URL}/by_class.zip", 1_031_576_378),
    "by_write.zip": (f"{_NIST_BASE_URL}/by_write.zip", 568_113_446),
}

_INTERMEDIATE_NAME = "femnist_writers.npz"
_STATS_NAME = "femnist_stats.json"


def relabel_class(c: str) -> int:
    """Map a NIST hexadecimal class directory name to 0..61.

    Exactly LEAF's ``relabel_class`` (data_to_json.py):
    0-9 digits, 10-35 uppercase letters, 36-61 lowercase letters.
    """
    if c.isdigit() and int(c) < 40:
        return int(c) - 30
    elif int(c, 16) <= 90:  # uppercase A-Z (0x41..0x5a)
        return int(c, 16) - 55
    else:  # lowercase a-z (0x61..0x7a)
        return int(c, 16) - 61


def default_cache_dir() -> Path:
    env = os.environ.get("FEMNIST_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "leaf_femnist"


# ---------------------------------------------------------------------------
# Worker-process helpers (top-level so macOS spawn can pickle them).
# Each worker opens its own zip handle once.
# ---------------------------------------------------------------------------
_WORKER_ZIP: zipfile.ZipFile | None = None


def _worker_init(zip_path: str) -> None:
    global _WORKER_ZIP
    _WORKER_ZIP = zipfile.ZipFile(zip_path, "r")


def _hash_member(name: str) -> tuple[str, str]:
    """Return (member name, md5 hex of decompressed bytes)."""
    assert _WORKER_ZIP is not None
    with _WORKER_ZIP.open(name) as fh:
        return name, hashlib.md5(fh.read()).hexdigest()


def _decode_member(name: str) -> tuple[str, bytes]:
    """Return (member name, 28*28 uint8 grayscale bytes).

    LEAF: Image.open -> convert('L') -> resize((28, 28), ANTIALIAS);
    ANTIALIAS has been the LANCZOS filter since Pillow 2.7.
    """
    from PIL import Image

    assert _WORKER_ZIP is not None
    with _WORKER_ZIP.open(name) as fh:
        img = Image.open(BytesIO(fh.read()))
        gray = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return name, np.asarray(gray, dtype=np.uint8).tobytes()


def _map_over_zip(
    zip_path: Path,
    names: list[str],
    func,
    workers: int,
    description: str,
) -> list:
    """Apply ``func`` to zip members, multiprocessed when worthwhile."""
    if workers <= 1 or len(names) < 2_000:
        _worker_init(str(zip_path))
        try:
            return [func(n) for n in names]
        finally:
            global _WORKER_ZIP
            if _WORKER_ZIP is not None:
                _WORKER_ZIP.close()
                _WORKER_ZIP = None
    logger.info(
        "FEMNIST: %s — %d members with %d workers", description, len(names), workers
    )
    chunk = max(1, len(names) // (workers * 16))
    with multiprocessing.get_context("spawn").Pool(
        processes=workers, initializer=_worker_init, initargs=(str(zip_path),)
    ) as pool:
        return pool.map(func, names, chunksize=chunk)


class FEMNISTDataset(FederationDataset):
    """LEAF FEMNIST: 62-class handwriting, naturally partitioned by writer.

    Data is stored on disk as uint8 28x28x1 images (raw 0..255 LEAF
    pixels, white background) and scaled to float32 [0, 1] at load
    time, matching LEAF's data_to_json scaling.
    """

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    # ------------------------------------------------------------------
    # Host-side preparation (download → match → convert → partition)
    # ------------------------------------------------------------------

    @classmethod
    def prepare_partitions(
        cls,
        total_nodes,
        output_dir,
        partition_strategy="natural",
        seed=42,
        **kwargs,
    ):
        """Build per-node .npz partitions of LEAF FEMNIST.

        kwargs (all optional, env-var fallbacks documented in the
        module docstring): ``cache_dir``, ``max_writers``,
        ``subsample_fraction``, ``test_fraction``, ``workers``.
        Foreign partitioner kwargs (``alpha``, ``classes_per_node``)
        are rejected together with their strategies.
        """
        if os.environ.get("FEMNIST_PARTITION_OVERRIDE"):
            raise RuntimeError(
                "FEMNIST_PARTITION_OVERRIDE is retired: 'natural' is now a "
                "first-class dataset.partition.strategy literal in the "
                "experiment-config schema.  Set partition.strategy in the "
                "spec instead and unset the environment variable (spec "
                "provenance must match realized behavior)."
            )
        if partition_strategy not in ("natural", "iid"):
            raise ValueError(
                f"FEMNIST supports partition strategies 'natural' (per-writer, "
                f"canonical) and 'iid' (pooled control); got "
                f"'{partition_strategy}'. Dirichlet/pathological skew is not "
                f"meaningful here — the writer partition IS the non-IID-ness."
            )

        cache_dir = Path(
            kwargs.get("cache_dir") or default_cache_dir()
        ).expanduser()
        test_fraction = float(
            kwargs.get("test_fraction")
            or os.environ.get("FEMNIST_TEST_FRACTION", 0.1)
        )
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction}")
        max_writers = kwargs.get("max_writers")
        if max_writers is None and os.environ.get("FEMNIST_MAX_WRITERS"):
            max_writers = int(os.environ["FEMNIST_MAX_WRITERS"])
        subsample_fraction = kwargs.get("subsample_fraction")
        if subsample_fraction is None and os.environ.get(
            "FEMNIST_SUBSAMPLE_FRACTION"
        ):
            subsample_fraction = float(os.environ["FEMNIST_SUBSAMPLE_FRACTION"])
        workers = int(
            kwargs.get("workers")
            or os.environ.get("FEMNIST_WORKERS", min(8, os.cpu_count() or 1))
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        x, y, writer_index, writers = cls._load_or_build_intermediate(
            cache_dir, workers
        )

        # --- Writer-level subsampling (LEAF "small" semantics) ---
        rng = np.random.RandomState(seed)
        keep = np.arange(len(writers))
        if subsample_fraction is not None:
            n_keep = max(1, int(round(len(writers) * float(subsample_fraction))))
            keep = np.sort(rng.permutation(len(writers))[:n_keep])
        if max_writers is not None and int(max_writers) < len(keep):
            keep = np.sort(rng.permutation(keep)[: int(max_writers)])
        if len(keep) < len(writers):
            logger.info(
                "FEMNIST: subsampled %d/%d writers (seed=%d)",
                len(keep),
                len(writers),
                seed,
            )
        kept_set = set(keep.tolist())
        sample_mask = np.isin(writer_index, keep)
        x, y, writer_index = x[sample_mask], y[sample_mask], writer_index[sample_mask]
        kept_writers = sorted(kept_set)

        # --- Global test holdout: seeded test_fraction of every writer ---
        test_rng = np.random.RandomState(seed)
        test_mask = np.zeros(len(x), dtype=bool)
        for w in kept_writers:
            idx = np.where(writer_index == w)[0]
            if len(idx) < 2:
                continue  # keep singletons in training
            n_test = max(1, int(round(len(idx) * test_fraction)))
            n_test = min(n_test, len(idx) - 1)
            test_mask[test_rng.permutation(idx)[:n_test]] = True
        x_test, y_test = x[test_mask], y[test_mask]
        x_rest, y_rest = x[~test_mask], y[~test_mask]
        writer_rest = writer_index[~test_mask]

        # --- Writers → nodes ---
        if partition_strategy == "natural":
            if len(kept_writers) < total_nodes:
                raise ValueError(
                    f"FEMNIST natural partition needs at least one writer per "
                    f"node: {len(kept_writers)} writers < {total_nodes} nodes. "
                    f"Raise FEMNIST_MAX_WRITERS / FEMNIST_SUBSAMPLE_FRACTION."
                )
            order = rng.permutation(np.asarray(kept_writers))
            node_writer_groups = np.array_split(order, total_nodes)
            partitions = []
            for group in node_writer_groups:
                m = np.isin(writer_rest, group)
                partitions.append((x_rest[m], y_rest[m]))
        else:  # iid control
            from .partitioner import create_partitioner

            node_writer_groups = [np.asarray([], dtype=int)] * total_nodes
            partitions = create_partitioner("iid").partition(
                x_rest, y_rest, total_nodes, seed
            )

        # --- Per-node 90/10 train/val + save (same scheme as siblings) ---
        paths = []
        for node_id, (x_node, y_node) in enumerate(partitions):
            node_rng = np.random.RandomState(seed + node_id)
            n = len(x_node)
            if n == 0:
                raise ValueError(
                    f"FEMNIST partition produced an empty node {node_id}; "
                    f"increase writers or samples."
                )
            indices = node_rng.permutation(n)
            val_size = max(1, n // 10)
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]
            path = output_path / f"node-{node_id}.npz"
            np.savez_compressed(
                path,
                x_train=x_node[train_idx],
                y_train=y_node[train_idx],
                x_val=x_node[val_idx],
                y_val=y_node[val_idx],
                x_test=x_test,  # shared global test set
                y_test=y_test,
            )
            paths.append(path)

        # --- Provenance sidecar (validation evidence, ignored by nodes) ---
        per_writer_counts = [
            int((writer_index == w).sum()) for w in kept_writers
        ]
        provenance = {
            "dataset": "femnist",
            "partition_strategy": partition_strategy,
            "seed": seed,
            "test_fraction": test_fraction,
            "writers_used": len(kept_writers),
            "writers_total_in_cache": len(writers),
            "samples_used": int(len(x)),
            "global_test_samples": int(len(x_test)),
            "node_writer_ids": {
                f"node-{i}": sorted(writers[w] for w in group)
                for i, group in enumerate(node_writer_groups)
            },
            "node_sample_counts": {
                f"node-{i}": int(len(px)) for i, (px, _) in enumerate(partitions)
            },
            "leaf_table1_reference": {
                "writers": LEAF_FEMNIST_TOTAL_WRITERS,
                "samples": LEAF_FEMNIST_TOTAL_SAMPLES,
                "mean_per_writer": LEAF_FEMNIST_MEAN_SAMPLES_PER_WRITER,
                "stdev_per_writer": LEAF_FEMNIST_STDEV_SAMPLES_PER_WRITER,
            },
            "subsampled": len(kept_writers) != len(writers),
            "per_writer_count_mean": float(np.mean(per_writer_counts)),
            "per_writer_count_stdev": float(np.std(per_writer_counts)),
        }
        with open(output_path / "femnist_partition.json", "w") as fh:
            json.dump(provenance, fh, indent=2)
        logger.info(
            "FEMNIST: %d nodes, %d writers, %d samples (%d global test), "
            "strategy=%s, seed=%d",
            total_nodes,
            len(kept_writers),
            len(x),
            len(x_test),
            partition_strategy,
            seed,
        )
        return paths

    # ------------------------------------------------------------------
    # LEAF pipeline: download → index → hash-match → decode → cache
    # ------------------------------------------------------------------

    @classmethod
    def _load_or_build_intermediate(
        cls, cache_dir: Path, workers: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Return (x uint8 [N,28,28,1], y int64 [N], writer_index int32 [N],
        writer ids [S]) — building and caching them on first use."""
        inter = cache_dir / _INTERMEDIATE_NAME
        if inter.exists():
            logger.info("FEMNIST: loading cached intermediate %s", inter)
            with np.load(inter, allow_pickle=False) as data:
                return (
                    data["x"],
                    data["y"].astype(np.int64),
                    data["writer_index"].astype(np.int32),
                    [w for w in data["writers"].tolist()],
                )

        raw_dir = cache_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        by_class = cls._download(raw_dir, "by_class.zip")
        by_write = cls._download(raw_dir, "by_write.zip")

        class_members, write_members = cls._index_archives(by_class, by_write)

        logger.info("FEMNIST: hashing %d by_class images", len(class_members))
        class_hashes = _map_over_zip(
            by_class,
            [name for name, _ in class_members],
            _hash_member,
            workers,
            "hash by_class",
        )
        class_label = {name: lbl for name, lbl in class_members}
        hash_to_label = {h: class_label[name] for name, h in class_hashes}

        logger.info("FEMNIST: hashing %d by_write images", len(write_members))
        write_hashes = _map_over_zip(
            by_write,
            [name for name, _ in write_members],
            _hash_member,
            workers,
            "hash by_write",
        )
        writer_of = {name: w for name, w in write_members}

        # LEAF match_hashes: a sample is a by_write image whose byte hash
        # appears in by_class (where the label lives).
        matched = [
            (name, writer_of[name], hash_to_label[h])
            for name, h in write_hashes
            if h in hash_to_label
        ]
        matched.sort(key=lambda t: (t[1], t[0]))  # deterministic order
        logger.info(
            "FEMNIST: matched %d/%d by_write images to labels",
            len(matched),
            len(write_members),
        )

        decoded = _map_over_zip(
            by_write,
            [name for name, _, _ in matched],
            _decode_member,
            workers,
            "decode matched images",
        )
        pixel_bytes = dict(decoded)

        writers = sorted({w for _, w, _ in matched})
        writer_to_idx = {w: i for i, w in enumerate(writers)}
        n = len(matched)
        x = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        y = np.zeros(n, dtype=np.int64)
        widx = np.zeros(n, dtype=np.int32)
        for i, (name, writer, label) in enumerate(matched):
            x[i, :, :, 0] = np.frombuffer(
                pixel_bytes[name], dtype=np.uint8
            ).reshape(IMAGE_SIZE, IMAGE_SIZE)
            y[i] = label
            widx[i] = writer_to_idx[writer]

        cls._validate_full_stats(writers, widx)

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            inter,
            x=x,
            y=y,
            writer_index=widx,
            writers=np.asarray(writers),
        )
        counts = np.bincount(widx, minlength=len(writers))
        stats = {
            "writers": len(writers),
            "samples": int(n),
            "mean_samples_per_writer": float(counts.mean()) if len(counts) else 0.0,
            "stdev_samples_per_writer": float(counts.std()) if len(counts) else 0.0,
            "classes_present": sorted(int(c) for c in np.unique(y)),
        }
        with open(cache_dir / _STATS_NAME, "w") as fh:
            json.dump(stats, fh, indent=2)
        logger.info("FEMNIST: intermediate cached at %s (%s)", inter, stats)
        return x, y, widx, writers

    @staticmethod
    def _download(raw_dir: Path, archive: str) -> Path:
        url, expected_size = _ARCHIVES[archive]
        target = raw_dir / archive
        if target.exists():
            actual = target.stat().st_size
            if actual == expected_size:
                logger.info("FEMNIST: %s already downloaded", archive)
            else:
                logger.warning(
                    "FEMNIST: existing %s is %d bytes (NIST reference: %d); "
                    "using it anyway — delete the file to force a re-download.",
                    archive,
                    actual,
                    expected_size,
                )
            return target
        logger.info("FEMNIST: downloading %s (%d bytes)", url, expected_size)
        tmp = target.with_suffix(".part")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
        actual = tmp.stat().st_size
        if actual != expected_size:
            logger.warning(
                "FEMNIST: %s size %d != expected %d (NIST may have "
                "re-uploaded; continuing)",
                archive,
                actual,
                expected_size,
            )
        tmp.rename(target)
        return target

    @staticmethod
    def _index_archives(
        by_class: Path, by_write: Path
    ) -> tuple[list[tuple[str, int]], list[tuple[str, str]]]:
        """List labelled by_class members and writer-tagged by_write members.

        by_class/<hexclass>/hsf_*/...png   → (member, relabel_class(hexclass))
          (train_<class> duplicate partitions excluded, as in LEAF)
        by_write/hsf_*/<writer>/...png     → (member, writer id)
        """
        class_members: list[tuple[str, int]] = []
        with zipfile.ZipFile(by_class) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".png"):
                    continue
                parts = name.split("/")
                try:
                    root = parts.index("by_class")
                except ValueError:
                    continue
                if len(parts) < root + 4:
                    continue
                hex_class, partition = parts[root + 1], parts[root + 2]
                if not partition.startswith("hsf_"):
                    continue  # skip train_<class> duplicates
                class_members.append((name, relabel_class(hex_class)))

        write_members: list[tuple[str, str]] = []
        with zipfile.ZipFile(by_write) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".png"):
                    continue
                parts = name.split("/")
                try:
                    root = parts.index("by_write")
                except ValueError:
                    continue
                if len(parts) < root + 4:
                    continue
                writer = parts[root + 2]
                write_members.append((name, writer))
        return class_members, write_members

    @staticmethod
    def _validate_full_stats(writers: list[str], writer_index: np.ndarray) -> None:
        """Compare a full-pipeline build against the pipeline-expected
        counts; log the LEAF Table 1 reference and its known gap."""
        n_writers, n_samples = len(writers), int(len(writer_index))
        matches = (
            n_writers == PIPELINE_EXPECTED_WRITERS
            and n_samples == PIPELINE_EXPECTED_SAMPLES
        )
        msg = (
            f"FEMNIST full-build stats: {n_writers} writers / {n_samples} "
            f"samples; pipeline-expected {PIPELINE_EXPECTED_WRITERS} / "
            f"{PIPELINE_EXPECTED_SAMPLES} (= EMNIST ByClass total); "
            f"LEAF paper Table 1 reports {LEAF_FEMNIST_TOTAL_WRITERS} / "
            f"{LEAF_FEMNIST_TOTAL_SAMPLES}, which is not reproducible from "
            f"the canonical archives (LEAF issue #49 — its own pipeline "
            f"yields {PIPELINE_EXPECTED_WRITERS} / "
            f"{LEAF_REPO_PIPELINE_SAMPLES} incl. a per-writer first-image "
            f"duplication this implementation corrects)"
        )
        if matches:
            logger.info("%s — MATCH", msg)
        else:
            logger.warning("%s — MISMATCH", msg)
            if os.environ.get("FEMNIST_STRICT_STATS") == "1":
                raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # Node-side loading
    # ------------------------------------------------------------------

    def load_from_file(self, path):
        """Load a pre-partitioned .npz dataset file.

        Images are stored uint8 (raw 0..255); they are scaled here to
        float32 [0, 1], reproducing LEAF's data_to_json scaling.
        """
        data = np.load(path)
        expected_keys = {"x_train", "y_train", "x_val", "y_val", "x_test", "y_test"}
        missing = expected_keys - set(data.keys())
        if missing:
            raise ValueError(
                f"Corrupted or incomplete dataset file {path}: "
                f"missing keys {sorted(missing)}. "
                f"Available keys: {sorted(data.keys())}. "
                f"Re-run the launcher to regenerate partitions."
            )

        def _to_float(arr: np.ndarray) -> np.ndarray:
            if np.issubdtype(arr.dtype, np.integer):
                return arr.astype(np.float32) / 255.0
            return arr.astype(np.float32)

        self.x_train = _to_float(data["x_train"])
        self.y_train = data["y_train"].astype(np.int64)
        self.x_val = _to_float(data["x_val"])
        self.y_val = data["y_val"].astype(np.int64)
        self.x_test = _to_float(data["x_test"])
        self.y_test = data["y_test"].astype(np.int64)

    def get_input_shape(self):
        return (IMAGE_SIZE, IMAGE_SIZE, 1)

    def get_num_classes(self):
        return NUM_CLASSES

    def get_name(self):
        return "femnist"

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_val_data(self):
        return self.x_val, self.y_val

    def get_test_data(self):
        return self.x_test, self.y_test
