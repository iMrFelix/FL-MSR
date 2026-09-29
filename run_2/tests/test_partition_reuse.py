"""Tests for the partition reuse guard — the silent-corruption fix.

`src/launcher.py:_existing_partitions` decides whether a run may adopt
partitions already on disk instead of re-materialising them.  That decision is
what removes the concurrent-materialisation race that replaced 1.35-13.97% of
`x_train` with all-zero images while leaving `y_train` byte-identical, and cost
`fedluar_hh` 11 of its 81 cells.

The guard has exactly one dangerous failure mode: adopting a corrupt shard.
These tests pin that shut.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.launcher import _existing_partitions

NODES = 3
N_TRAIN, N_VAL = 40, 6


def _write(path, *, zero_rows_train=0, zero_rows_val=0, n_train=N_TRAIN):
    """A shard that looks exactly like a real one, optionally corrupted."""
    rng = np.random.RandomState(0)
    # +1 keeps every legitimate pixel non-zero, so an all-zero row is
    # unambiguously injected rather than a plausible dark image.
    x_train = rng.rand(n_train, 4, 4, 3).astype(np.float32) + 1.0
    x_val = rng.rand(N_VAL, 4, 4, 3).astype(np.float32) + 1.0
    x_train[:zero_rows_train] = 0.0
    x_val[:zero_rows_val] = 0.0
    np.savez(
        path,
        x_train=x_train,
        y_train=rng.randint(0, 10, n_train).astype(np.int64),
        x_val=x_val,
        y_val=rng.randint(0, 10, N_VAL).astype(np.int64),
        x_test=x_val,
        y_test=rng.randint(0, 10, N_VAL).astype(np.int64),
    )


def _make_clean(tmp_path, nodes=NODES):
    for i in range(nodes):
        _write(tmp_path / f"node-{i}.npz")
    return tmp_path


def test_adopts_a_complete_clean_set(tmp_path):
    _make_clean(tmp_path)
    paths = _existing_partitions(tmp_path, NODES)
    assert paths is not None
    assert [p.name for p in paths] == [f"node-{i}.npz" for i in range(NODES)]


def test_refuses_when_a_shard_is_missing(tmp_path):
    _make_clean(tmp_path)
    (tmp_path / f"node-{NODES - 1}.npz").unlink()
    assert _existing_partitions(tmp_path, NODES) is None


def test_refuses_when_more_nodes_are_requested_than_exist(tmp_path):
    _make_clean(tmp_path)
    assert _existing_partitions(tmp_path, NODES + 1) is None


@pytest.mark.parametrize("node", range(NODES))
def test_refuses_a_shard_with_zero_images_in_train(tmp_path, node):
    """The corruption signature: labels intact, images zeroed."""
    _make_clean(tmp_path)
    _write(tmp_path / f"node-{node}.npz", zero_rows_train=1)
    assert _existing_partitions(tmp_path, NODES) is None


def test_refuses_a_shard_with_zero_images_in_val(tmp_path):
    """x_val is materialised by the same racing code path as x_train."""
    _make_clean(tmp_path)
    _write(tmp_path / "node-1.npz", zero_rows_val=1)
    assert _existing_partitions(tmp_path, NODES) is None


def test_detects_a_single_zero_row_among_many(tmp_path):
    """Observed corruption ran 1.35-13.97%; one row must still trip it."""
    _make_clean(tmp_path)
    _write(tmp_path / "node-0.npz", zero_rows_train=1, n_train=5000)
    assert _existing_partitions(tmp_path, NODES) is None


def test_refuses_an_unreadable_shard(tmp_path):
    """A half-written npz must be regenerated, never adopted."""
    _make_clean(tmp_path)
    (tmp_path / "node-2.npz").write_bytes(b"PK\x03\x04 truncated garbage")
    assert _existing_partitions(tmp_path, NODES) is None


def test_refuses_an_empty_shard(tmp_path):
    _make_clean(tmp_path)
    _write(tmp_path / "node-0.npz", n_train=0)
    assert _existing_partitions(tmp_path, NODES) is None


# ---------------------------------------------------------------------------
# The back door: what happens when the guard REJECTS mid-campaign.
# ---------------------------------------------------------------------------
# A rejection falls through to prepare_partitions -> cifar10.load_data(), and
# inside a concurrent batch that is exactly the keras-cache race that injects
# the zero images. FL_REQUIRE_PREMATERIALIZED turns that silent re-materialise
# into a loud single-cell failure.

def _cfg(tmp_path, nodes=NODES):
    from types import SimpleNamespace
    return SimpleNamespace(
        federation=SimpleNamespace(num_nodes=nodes),
        training=SimpleNamespace(dataset=SimpleNamespace(
            name="cifar10",
            partition=SimpleNamespace(strategy="dirichlet", seed=41, alpha=0.1,
                                      classes_per_node=None, max_writers=None,
                                      workers_only=False))),
    )


def test_require_prematerialized_raises_instead_of_regenerating(tmp_path, monkeypatch):
    from src import launcher
    monkeypatch.setenv("FL_REQUIRE_PREMATERIALIZED", "1")
    out = tmp_path / "run"
    (out / "data").mkdir(parents=True)
    _make_clean(out / "data")
    _write(out / "data" / "node-0.npz", zero_rows_train=3)   # corrupt one shard

    called = []
    monkeypatch.setattr(launcher.CIFAR10Dataset, "prepare_partitions",
                        classmethod(lambda cls, **kw: called.append(kw)))
    with pytest.raises(RuntimeError, match="FL_REQUIRE_PREMATERIALIZED"):
        launcher.prepare_data(_cfg(tmp_path), out)
    assert called == [], "must NOT re-materialise when the flag is set"


def test_require_prematerialized_accepts_clean_partitions(tmp_path, monkeypatch):
    from src import launcher
    monkeypatch.setenv("FL_REQUIRE_PREMATERIALIZED", "1")
    out = tmp_path / "run"
    (out / "data").mkdir(parents=True)
    _make_clean(out / "data")

    called = []
    monkeypatch.setattr(launcher.CIFAR10Dataset, "prepare_partitions",
                        classmethod(lambda cls, **kw: called.append(kw)))
    assert launcher.prepare_data(_cfg(tmp_path), out) == out / "data"
    assert called == [], "clean partitions must be reused, not regenerated"


@pytest.mark.parametrize("val", ["", "0"])
def test_flag_unset_or_zero_keeps_legacy_regeneration(tmp_path, monkeypatch, val):
    """Default behaviour is unchanged: serial callers still regenerate."""
    from src import launcher
    monkeypatch.setenv("FL_REQUIRE_PREMATERIALIZED", val)
    out = tmp_path / "run"
    (out / "data").mkdir(parents=True)
    _write(out / "data" / "node-0.npz", zero_rows_train=3)

    called = []
    monkeypatch.setattr(launcher.CIFAR10Dataset, "prepare_partitions",
                        classmethod(lambda cls, **kw: called.append(kw) or []))
    launcher.prepare_data(_cfg(tmp_path), out)
    assert len(called) == 1, "without the flag, regeneration must still happen"
