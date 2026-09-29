"""Tests for the LEAF FEMNIST loader and the FedLUAR 4-layer CNN.

The preprocessing pipeline is exercised end-to-end against synthetic
NIST-shaped zip fixtures (no network, no real download).  The real
LEAF Table 1 statistics are asserted as cited constants; the full-build
comparison against them happens inside the loader itself
(``_validate_full_stats``) when the genuine archives are processed.

Fixture trick for writer-attribution checks: every synthetic image is a
solid 128x128 PNG whose constant intensity encodes (writer, image)
uniquely — Lanczos-resizing a constant image is constant, so each
partitioned sample's pixel value identifies its writer.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from src.datasets.femnist import (
    EMNIST_BYCLASS_TOTAL,
    FEMNISTDataset,
    LEAF_FEMNIST_MEAN_SAMPLES_PER_WRITER,
    LEAF_FEMNIST_STDEV_SAMPLES_PER_WRITER,
    LEAF_FEMNIST_TOTAL_SAMPLES,
    LEAF_FEMNIST_TOTAL_WRITERS,
    LEAF_REPO_PIPELINE_SAMPLES,
    PIPELINE_EXPECTED_SAMPLES,
    PIPELINE_EXPECTED_WRITERS,
    default_cache_dir,
    relabel_class,
)
from src.models.base import FederationModel
from src.models.femnist_cnn import (
    EXPECTED_PARAMETER_COUNT,
    FEMNISTCNN,
    FEMNISTCNNModelDef,
)

# ---------------------------------------------------------------------------
# Synthetic NIST SD19 fixture
# ---------------------------------------------------------------------------

#: writer id -> (number of images, base intensity).  Intensities are
#: spaced so writer = value // 20 after the constant-image resize.
_FIXTURE_WRITERS = {
    "f0000_00": (12, 0),
    "f0001_00": (8, 20),
    "f0002_00": (20, 40),
    "f0003_00": (10, 60),
    "f0004_00": (6, 80),
    "f0005_00": (4, 100),
}
_FIXTURE_CLASSES = ["30", "39", "41", "5a", "61", "7a"]  # 0, 9, 10, 35, 36, 61


def _solid_png(value: int) -> bytes:
    from PIL import Image

    img = Image.new("L", (128, 128), color=value)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _writer_of_pixel(value: float) -> int:
    """Inverse of the fixture encoding (value in [0, 1] after loading)."""
    return int(round(value * 255.0)) // 20


def _build_fixture_archives(raw_dir: Path) -> dict[str, dict]:
    """Write by_class.zip / by_write.zip mimicking NIST SD19.

    Returns ground truth: {writer: {"labels": [...], "values": [...]}}.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    truth: dict[str, dict] = {}
    class_entries: list[tuple[str, bytes]] = []
    write_entries: list[tuple[str, bytes]] = []

    for w_idx, (writer, (count, base)) in enumerate(
        sorted(_FIXTURE_WRITERS.items())
    ):
        labels, values = [], []
        for i in range(count):
            value = base + i  # unique bytes per image, constant per image
            hex_class = _FIXTURE_CLASSES[(w_idx + i) % len(_FIXTURE_CLASSES)]
            png = _solid_png(value)
            class_entries.append(
                (f"by_class/{hex_class}/hsf_0/img_{writer}_{i}.png", png)
            )
            # train_<class> duplicates must be ignored by the indexer
            class_entries.append(
                (f"by_class/{hex_class}/train_{hex_class}/dup_{writer}_{i}.png", png)
            )
            write_entries.append(
                (f"by_write/hsf_0/{writer}/c000_{writer}/img_{i}.png", png)
            )
            labels.append(relabel_class(hex_class))
            values.append(value)
        truth[writer] = {"labels": labels, "values": values}

    # An unmatched by_write image (no by_class counterpart): dropped.
    write_entries.append(
        ("by_write/hsf_0/f0000_00/c000_f0000_00/unmatched.png", _solid_png(250))
    )
    # Non-PNG members: ignored.
    class_entries.append(("by_class/readme.txt", b"not a png"))
    write_entries.append(("by_write/readme.txt", b"not a png"))

    for name, entries in (
        ("by_class.zip", class_entries),
        ("by_write.zip", write_entries),
    ):
        with zipfile.ZipFile(raw_dir / name, "w") as zf:
            for member, payload in entries:
                zf.writestr(member, payload)
    return truth


@pytest.fixture()
def femnist_cache(tmp_path):
    """A cache dir whose raw/ already contains the fixture archives."""
    cache = tmp_path / "cache"
    _build_fixture_archives(cache / "raw")
    return cache


def _prepare(cache, out_dir, total_nodes=3, strategy="natural", seed=42, **kw):
    return FEMNISTDataset.prepare_partitions(
        total_nodes=total_nodes,
        output_dir=str(out_dir),
        partition_strategy=strategy,
        seed=seed,
        cache_dir=cache,
        **kw,
    )


def _load_provenance(out_dir) -> dict:
    with open(Path(out_dir) / "femnist_partition.json") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# LEAF paper Table 1 citation constants
# ---------------------------------------------------------------------------

class TestLEAFTable1Constants:
    """The constants the loader validates against must be the numbers in
    PRIOR_WORK/LEAF_Fed_Benchmark_2018.pdf, Table 1 (page 3)."""

    def test_cited_values(self):
        assert LEAF_FEMNIST_TOTAL_WRITERS == 3550
        assert LEAF_FEMNIST_TOTAL_SAMPLES == 805263
        assert LEAF_FEMNIST_MEAN_SAMPLES_PER_WRITER == 226.83
        assert LEAF_FEMNIST_STDEV_SAMPLES_PER_WRITER == 88.94

    def test_internal_consistency(self):
        # 805,263 / 3,550 = 226.834... — Table 1's own mean.
        assert (
            abs(
                LEAF_FEMNIST_TOTAL_SAMPLES / LEAF_FEMNIST_TOTAL_WRITERS
                - LEAF_FEMNIST_MEAN_SAMPLES_PER_WRITER
            )
            < 0.01
        )


class TestPipelineExpectedStats:
    """The LEAF method on the canonical archives cannot reproduce Table 1
    (LEAF issue #49); the loader validates against the pipeline-expected
    counts, which sit on the EMNIST ByClass anchor."""

    def test_emnist_byclass_anchor(self):
        assert PIPELINE_EXPECTED_SAMPLES == EMNIST_BYCLASS_TOTAL == 814_255
        assert PIPELINE_EXPECTED_WRITERS == 3_597

    def test_leaf_issue49_duplication_arithmetic(self):
        """LEAF's own pipeline output (817,851; issue #49) equals our
        duplication-free count plus exactly one duplicated first image
        per writer beyond the first (group_by_writer.py defect)."""
        assert (
            LEAF_REPO_PIPELINE_SAMPLES - PIPELINE_EXPECTED_SAMPLES
            == PIPELINE_EXPECTED_WRITERS - 1
        )

    def test_table1_gap_is_small(self):
        """The published Table 1 sits within ~1.2% of the pipeline truth."""
        assert (
            abs(PIPELINE_EXPECTED_SAMPLES - LEAF_FEMNIST_TOTAL_SAMPLES)
            / LEAF_FEMNIST_TOTAL_SAMPLES
            < 0.012
        )
        assert (
            abs(PIPELINE_EXPECTED_WRITERS - LEAF_FEMNIST_TOTAL_WRITERS)
            / LEAF_FEMNIST_TOTAL_WRITERS
            < 0.014
        )

    @pytest.mark.skipif(
        not (default_cache_dir() / "femnist_stats.json").exists(),
        reason="full FEMNIST cache not built on this machine",
    )
    def test_real_full_build_stats(self):
        """Validates the genuine NIST-archive build (runs only where the
        intermediate cache exists, e.g. the experiment host)."""
        stats = json.loads(
            (default_cache_dir() / "femnist_stats.json").read_text()
        )
        assert stats["writers"] == PIPELINE_EXPECTED_WRITERS
        assert stats["samples"] == PIPELINE_EXPECTED_SAMPLES
        # Per-writer distribution agrees with Table 1 to <0.25%.
        assert (
            abs(
                stats["mean_samples_per_writer"]
                - LEAF_FEMNIST_MEAN_SAMPLES_PER_WRITER
            )
            < 0.5
        )
        assert (
            abs(
                stats["stdev_samples_per_writer"]
                - LEAF_FEMNIST_STDEV_SAMPLES_PER_WRITER
            )
            < 0.2
        )
        assert stats["classes_present"] == list(range(62))


class TestRelabelClass:
    """LEAF data_to_json class mapping: digits, upper, lower → 0..61."""

    def test_digits(self):
        assert relabel_class("30") == 0
        assert relabel_class("39") == 9

    def test_uppercase(self):
        assert relabel_class("41") == 10  # 'A'
        assert relabel_class("5a") == 35  # 'Z'

    def test_lowercase(self):
        assert relabel_class("61") == 36  # 'a'
        assert relabel_class("7a") == 61  # 'z'

    def test_all_62_distinct_and_complete(self):
        hex_classes = (
            [f"3{d}" for d in range(10)]
            + [format(c, "x") for c in range(0x41, 0x5A + 1)]
            + [format(c, "x") for c in range(0x61, 0x7A + 1)]
        )
        labels = [relabel_class(c) for c in hex_classes]
        assert sorted(labels) == list(range(62))


# ---------------------------------------------------------------------------
# Archive indexing
# ---------------------------------------------------------------------------

class TestIndexArchives:
    def test_excludes_train_dirs_and_non_png(self, femnist_cache):
        raw = femnist_cache / "raw"
        class_members, write_members = FEMNISTDataset._index_archives(
            raw / "by_class.zip", raw / "by_write.zip"
        )
        n_images = sum(c for c, _ in _FIXTURE_WRITERS.values())
        # hsf_0 images only — the train_<class> duplicates are excluded.
        assert len(class_members) == n_images
        assert all("/hsf_" in name for name, _ in class_members)
        # by_write: every image incl. the unmatched one; readme ignored.
        assert len(write_members) == n_images + 1
        writers = {w for _, w in write_members}
        assert writers == set(_FIXTURE_WRITERS.keys())

    def test_class_labels_are_relabelled(self, femnist_cache):
        raw = femnist_cache / "raw"
        class_members, _ = FEMNISTDataset._index_archives(
            raw / "by_class.zip", raw / "by_write.zip"
        )
        labels = {lbl for _, lbl in class_members}
        assert labels == {0, 9, 10, 35, 36, 61}


# ---------------------------------------------------------------------------
# Natural partition end-to-end (fixture pipeline, no network)
# ---------------------------------------------------------------------------

class TestNaturalPartition:
    def test_schema_and_writer_disjointness(self, femnist_cache, tmp_path):
        out = tmp_path / "out"
        paths = _prepare(femnist_cache, out, total_nodes=3)
        assert [p.name for p in paths] == [
            "node-0.npz",
            "node-1.npz",
            "node-2.npz",
        ]

        prov = _load_provenance(out)
        assert prov["partition_strategy"] == "natural"
        assert prov["writers_used"] == len(_FIXTURE_WRITERS)

        # Writers are disjoint across nodes and cover all fixture writers.
        groups = [set(v) for v in prov["node_writer_ids"].values()]
        assert set.union(*groups) == set(_FIXTURE_WRITERS.keys())
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                assert groups[i] & groups[j] == set()

        # Per-sample attribution: pixel value encodes the writer; every
        # node's train/val samples must come from that node's writers only.
        writer_names = sorted(_FIXTURE_WRITERS.keys())
        ds = FEMNISTDataset()
        global_tests = []
        for node_id, group in enumerate(prov["node_writer_ids"].values()):
            ds.load_from_file(str(out / f"node-{node_id}.npz"))
            for x_arr in (ds.x_train, ds.x_val):
                assert x_arr.dtype == np.float32
                assert x_arr.shape[1:] == (28, 28, 1)
                assert x_arr.min() >= 0.0 and x_arr.max() <= 1.0
                for sample in x_arr:
                    w = writer_names[_writer_of_pixel(float(sample[0, 0, 0]))]
                    assert w in set(group)
            assert ds.y_train.dtype == np.int64
            assert set(np.unique(ds.y_train)) <= set(range(62))
            global_tests.append((ds.x_test, ds.y_test))

        # The global test set is identical on every node.
        for x_t, y_t in global_tests[1:]:
            assert np.array_equal(x_t, global_tests[0][0])
            assert np.array_equal(y_t, global_tests[0][1])
        assert len(global_tests[0][0]) == prov["global_test_samples"] > 0

    def test_natural_skew_preserved(self, femnist_cache, tmp_path):
        """Nodes hold whole writers, so sample counts are imbalanced."""
        out = tmp_path / "out"
        _prepare(femnist_cache, out, total_nodes=3)
        counts = list(_load_provenance(out)["node_sample_counts"].values())
        assert len(set(counts)) > 1  # not an even split

    def test_train_val_test_account_for_all_samples(
        self, femnist_cache, tmp_path
    ):
        out = tmp_path / "out"
        _prepare(femnist_cache, out, total_nodes=2)
        prov = _load_provenance(out)
        ds = FEMNISTDataset()
        total = 0
        for node_id in range(2):
            ds.load_from_file(str(out / f"node-{node_id}.npz"))
            total += len(ds.x_train) + len(ds.x_val)
        total += prov["global_test_samples"]
        assert total == sum(c for c, _ in _FIXTURE_WRITERS.values())

    def test_too_few_writers_raises(self, femnist_cache, tmp_path):
        with pytest.raises(ValueError, match="at least one writer per"):
            _prepare(femnist_cache, tmp_path / "out", total_nodes=10)

    def test_dirichlet_rejected(self, femnist_cache, tmp_path):
        with pytest.raises(ValueError, match="natural"):
            _prepare(
                femnist_cache, tmp_path / "out", strategy="dirichlet", alpha=0.5
            )


class TestDeterminism:
    def test_same_seed_byte_identical(self, femnist_cache, tmp_path):
        out_a, out_b = tmp_path / "a", tmp_path / "b"
        _prepare(femnist_cache, out_a, seed=7)
        _prepare(femnist_cache, out_b, seed=7)
        for node_id in range(3):
            with np.load(out_a / f"node-{node_id}.npz") as da, np.load(
                out_b / f"node-{node_id}.npz"
            ) as db:
                assert set(da.keys()) == set(db.keys())
                for key in da.keys():
                    assert np.array_equal(da[key], db[key]), (node_id, key)

    def test_different_seed_changes_assignment(self, femnist_cache, tmp_path):
        out_a, out_b = tmp_path / "a", tmp_path / "b"
        _prepare(femnist_cache, out_a, seed=7)
        _prepare(femnist_cache, out_b, seed=8)
        assert (
            _load_provenance(out_a)["node_writer_ids"]
            != _load_provenance(out_b)["node_writer_ids"]
        )


class TestControlsAndKnobs:
    def test_iid_control_splits_evenly(self, femnist_cache, tmp_path):
        out = tmp_path / "out"
        _prepare(femnist_cache, out, total_nodes=3, strategy="iid")
        prov = _load_provenance(out)
        counts = list(prov["node_sample_counts"].values())
        assert max(counts) - min(counts) <= 1
        assert prov["partition_strategy"] == "iid"

    def test_partition_override_env_retired(
        self, femnist_cache, tmp_path, monkeypatch
    ):
        """The interim override knob is retired now that 'natural' is a
        schema literal: setting it is a hard error (PM gate-1 fix)."""
        monkeypatch.setenv("FEMNIST_PARTITION_OVERRIDE", "natural")
        with pytest.raises(RuntimeError, match="retired"):
            _prepare(femnist_cache, tmp_path / "out", strategy="iid")

    def test_max_writers_subsample(self, femnist_cache, tmp_path):
        out = tmp_path / "out"
        _prepare(femnist_cache, out, total_nodes=2, max_writers=3)
        prov = _load_provenance(out)
        assert prov["writers_used"] == 3
        assert prov["subsampled"] is True

    def test_intermediate_cache_reused(self, femnist_cache, tmp_path):
        """After the first build, the raw archives are no longer needed."""
        _prepare(femnist_cache, tmp_path / "a")
        (femnist_cache / "raw" / "by_class.zip").unlink()
        (femnist_cache / "raw" / "by_write.zip").unlink()
        _prepare(femnist_cache, tmp_path / "b")  # served from intermediate
        assert (tmp_path / "b" / "node-0.npz").exists()

    def test_unmatched_by_write_image_dropped(self, femnist_cache, tmp_path):
        """The fixture's unmatched image must not appear in any split."""
        out = tmp_path / "out"
        _prepare(femnist_cache, out, total_nodes=2)
        stats = json.loads((femnist_cache / "femnist_stats.json").read_text())
        assert stats["samples"] == sum(c for c, _ in _FIXTURE_WRITERS.values())
        assert stats["writers"] == len(_FIXTURE_WRITERS)


class TestLoadFromFile:
    def test_missing_keys_error(self, tmp_path):
        bad = tmp_path / "bad.npz"
        np.savez(bad, x_train=np.zeros((1, 28, 28, 1)))
        ds = FEMNISTDataset()
        with pytest.raises(ValueError, match="missing keys"):
            ds.load_from_file(str(bad))

    def test_metadata_accessors(self):
        ds = FEMNISTDataset()
        assert ds.get_input_shape() == (28, 28, 1)
        assert ds.get_num_classes() == 62
        assert ds.get_name() == "femnist"


# ---------------------------------------------------------------------------
# FedLUAR 4-layer CNN
# ---------------------------------------------------------------------------

class TestFEMNISTCNN:
    @pytest.fixture(scope="class")
    def model(self):
        return FEMNISTCNN()

    def test_parameter_count_matches_fedluar_table1(self, model):
        """6,603,710 params = 25.19 MiB; x32 clients = 806.1 MiB, the
        FedLUAR Table 1 FEMNIST (CNN) FedAvg memory footprint."""
        total = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
        assert total == EXPECTED_PARAMETER_COUNT == 6_603_710
        mib_32_clients = total * 4 * 32 / 2**20
        assert abs(mib_32_clients - 806.11) < 0.01

    def test_four_weight_layers_eight_variables(self, model):
        params = FederationModel.get_parameters(model)
        assert sorted(params.keys()) == [
            "conv_0/bias",
            "conv_0/kernel",
            "conv_1/bias",
            "conv_1/kernel",
            "dense_0/bias",
            "dense_0/kernel",
            "head/bias",
            "head/kernel",
        ]
        layers = {path.split("/")[0] for path in params}
        assert len(layers) == 4  # FedLUAR: "4 layers in CNN"

    def test_layer_shapes(self, model):
        params = FederationModel.get_parameters(model)
        assert params["conv_0/kernel"].shape == (5, 5, 1, 32)
        assert params["conv_1/kernel"].shape == (5, 5, 32, 64)
        assert params["dense_0/kernel"].shape == (3136, 2048)  # 7*7*64
        assert params["head/kernel"].shape == (2048, 62)

    def test_dense_kernel_dominates_bytes(self, model):
        """FEMNIST is the single-dominant-layer regime (FedLUAR §4.3)."""
        params = FederationModel.get_parameters(model)
        total = sum(p.size for p in params.values())
        assert params["dense_0/kernel"].size / total > 0.97

    def test_forward_pass_logits(self, model):
        out = model(np.zeros((5, 28, 28, 1), dtype=np.float32))
        assert out.shape == (5, 62)
        flat = model(np.zeros((5, 784), dtype=np.float32))
        assert flat.shape == (5, 62)

    def test_modeldef_roundtrip(self):
        model_def = FEMNISTCNNModelDef()
        assert model_def.get_name() == "femnist_cnn"
        m = model_def.build(input_shape=(28, 28, 1), num_classes=62)
        assert FederationModel.get_parameter_count(m) == EXPECTED_PARAMETER_COUNT


# ---------------------------------------------------------------------------
# Schema knobs (PM gate-1 required fixes: first-class 'natural' +
# max_writers in DatasetPartitionConfig)
# ---------------------------------------------------------------------------

class TestPartitionSchemaKnobs:
    def test_natural_strategy_is_schema_literal(self):
        from src.config.schema import DatasetPartitionConfig

        cfg = DatasetPartitionConfig(strategy="natural", seed=7)
        assert cfg.strategy == "natural"
        assert cfg.max_writers is None  # no extra params required

    def test_max_writers_field(self):
        from pydantic import ValidationError

        from src.config.schema import DatasetPartitionConfig

        cfg = DatasetPartitionConfig(
            strategy="natural", seed=7, max_writers=40
        )
        assert cfg.max_writers == 40
        with pytest.raises(ValidationError):
            DatasetPartitionConfig(strategy="natural", max_writers=0)

    def test_smoke_spec_validates_and_is_natural(self):
        from src.config.schema import load_config

        cfg = load_config(
            "configs/experiments/phase1/t3_femnist_smoke.yaml"
        )
        part = cfg.training.dataset.partition
        assert part.strategy == "natural"
        assert part.max_writers == 40
        assert cfg.training.dataset.name == "femnist"


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

class TestRegistries:
    def test_node_registries(self):
        from src.node import DATASET_REGISTRY, MODEL_REGISTRY

        assert DATASET_REGISTRY["femnist"] is FEMNISTDataset
        assert MODEL_REGISTRY["femnist_cnn"] is FEMNISTCNNModelDef

    def test_launcher_registry(self):
        from src.launcher import DATASET_REGISTRY

        assert DATASET_REGISTRY["femnist"] is FEMNISTDataset
