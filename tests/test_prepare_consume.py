"""Ownership tests for destructive, low-peak prepared-trait pairing."""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import bipred.prepare as prepare_module
from bipred import PreparedTrait, pair_prepared_traits
from ldpred3 import save_ld_blocks
from ldpred3.interop import PreparedLDCache, prepare_ld_cache


def _ar1(k, rho=0.55):
    i = np.arange(k)
    return np.ascontiguousarray(
        (rho ** np.abs(i[:, None] - i[None, :])).astype(np.float32))


def _reference(tmp_path, *, mmap=False):
    k, m = 6, 12
    blocks = [
        (_ar1(k), np.arange(k)),
        (_ar1(k, 0.7), np.arange(k, m)),
    ]
    ids = np.array([f"rs{i}" for i in range(m)], dtype=object)
    path = tmp_path / ("mapped.npz" if mmap else "ordinary.npz")
    save_ld_blocks(
        path, blocks, ids, mmap=mmap,
        counted_allele=np.array(["A"] * m, dtype=object),
        other_allele=np.array(["G"] * m, dtype=object),
        chrom=np.array(["1"] * m, dtype=object),
        pos=np.arange(1, m + 1),
        reference_af=np.linspace(0.1, 0.45, m), n_ref=500, ridge=0.0)
    return str(path), m


def _trait(indices, m, label):
    indices = np.asarray(indices, dtype=np.int64)
    x = np.linspace(-0.08, 0.09, len(indices))
    return PreparedTrait(
        indices=indices,
        beta_hat=np.ascontiguousarray(x),
        n_eff=np.full(len(indices), 10_000.0),
        z=np.ascontiguousarray(x * 100.0),
        eaf=np.linspace(0.15, 0.4, len(indices)),
        n_cache=m, log={"label": label})


def _assert_same_panel(left, right):
    for name in (
            "beta_hat1", "beta_hat2", "n_eff1", "n_eff2", "id",
            "chrom", "pos", "effect_allele", "other_allele", "af",
            "cache_indices"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    assert len(left.blocks) == len(right.blocks)
    for (R_left, i_left), (R_right, i_right) in zip(
            left.blocks, right.blocks):
        np.testing.assert_array_equal(i_left, i_right)
        np.testing.assert_array_equal(R_left, R_right)


def test_consuming_pair_matches_default_and_releases_each_source_block(
        monkeypatch, tmp_path):
    path, m = _reference(tmp_path)
    # Nonconsecutive rows force an independent dense subset in both blocks.
    indices = np.array([0, 2, 3, 5, 6, 8, 9, 11])
    trait1 = _trait(indices, m, "trait 1")
    trait2 = _trait(indices, m, "trait 2")

    with prepare_ld_cache(path) as ordinary:
        expected = pair_prepared_traits(ordinary, trait1, trait2)
        assert not ordinary.closed and len(ordinary.blocks) == 2
        assert "ld_cache_consumed" not in expected.log

    consumed = prepare_ld_cache(path)
    source_refs = [weakref.ref(R) for R, _ in consumed.blocks]
    original_subset = prepare_module.subset_ld_blocks
    partial_calls = 0

    def observed_subset(blocks, selection, **kwargs):
        nonlocal partial_calls
        # The first two-block call is the validation-only 1x1 probe. Every
        # later one-block call materializes one requested principal subset.
        if len(blocks) == 1:
            partial_calls += 1
            if partial_calls == 2:
                gc.collect()
                assert source_refs[0]() is None
        return original_subset(blocks, selection, **kwargs)

    monkeypatch.setattr(prepare_module, "subset_ld_blocks", observed_subset)
    actual = pair_prepared_traits(
        consumed, trait1, trait2, consume_ld_cache=True)

    _assert_same_panel(actual, expected)
    assert partial_calls == 2
    assert consumed.closed and consumed.blocks == []
    assert type(actual.blocks) is list and actual._ld_owner is None
    assert actual.log["ld_cache_consumed"] is True
    gc.collect()
    assert all(ref() is None for ref in source_refs)


def test_consuming_pair_rejects_mmap_without_damaging_its_owner(tmp_path):
    path, m = _reference(tmp_path, mmap=True)
    indices = np.array([0, 2, 3, 5, 6, 8, 9, 11])
    trait1 = _trait(indices, m, "trait 1")
    trait2 = _trait(indices, m, "trait 2")

    with prepare_ld_cache(path) as mapped:
        source_count = len(mapped.blocks)
        with pytest.raises(ValueError, match="ordinary in-memory"):
            pair_prepared_traits(
                mapped, trait1, trait2, consume_ld_cache=True)
        assert not mapped.closed and len(mapped.blocks) == source_count

        # The established non-consuming path remains valid: a caller-owned
        # mmap cache owns the returned views until its surrounding context ends.
        prepared = pair_prepared_traits(mapped, trait1, trait2)
        assert prepared._ld_owner is None
        assert len(prepared.blocks) == 2


def test_consuming_pair_failure_closes_cache_and_releases_partial_work(
        monkeypatch, tmp_path):
    path, m = _reference(tmp_path)
    indices = np.array([0, 2, 3, 5, 6, 8, 9, 11])
    trait1 = _trait(indices, m, "trait 1")
    trait2 = _trait(indices, m, "trait 2")
    consumed = prepare_ld_cache(path)
    source_refs = [weakref.ref(R) for R, _ in consumed.blocks]
    original_subset = prepare_module.subset_ld_blocks
    partial_calls = 0

    def fail_on_second_block(blocks, selection, **kwargs):
        nonlocal partial_calls
        if len(blocks) == 1:
            partial_calls += 1
            if partial_calls == 2:
                raise MemoryError("simulated second-block allocation failure")
        return original_subset(blocks, selection, **kwargs)

    monkeypatch.setattr(
        prepare_module, "subset_ld_blocks", fail_on_second_block)
    try:
        pair_prepared_traits(
            consumed, trait1, trait2, consume_ld_cache=True)
    except MemoryError as exc:
        assert "second-block" in str(exc)
    else:  # pragma: no cover - documents the required injected failure
        raise AssertionError("expected the simulated allocation failure")

    assert partial_calls == 2
    assert consumed.closed and consumed.blocks == []
    gc.collect()
    assert all(ref() is None for ref in source_refs)


def test_consuming_pair_validates_opt_in_before_mutation(tmp_path):
    path, m = _reference(tmp_path)
    indices = np.array([0, 2, 3, 5, 6, 8, 9, 11])
    trait1 = _trait(indices, m, "trait 1")
    trait2 = _trait(indices, m, "trait 2")

    with prepare_ld_cache(path) as ordinary:
        with pytest.raises(TypeError, match="must be a boolean"):
            pair_prepared_traits(
                ordinary, trait1, trait2, consume_ld_cache=1)
        with pytest.raises(ValueError, match="requires screen=False"):
            pair_prepared_traits(
                ordinary, trait1, trait2, screen=True,
                consume_ld_cache=True)
        assert not ordinary.closed and len(ordinary.blocks) == 2

    malformed = PreparedLDCache(
        "not-on-disk", [(np.eye(1), np.array([0]))],
        np.array(["rs0", "rs1"]), {})
    short1 = _trait([0, 1], 2, "trait 1")
    short2 = _trait([0, 1], 2, "trait 2")
    with pytest.raises(ValueError, match="cover 1 variants, expected 2"):
        pair_prepared_traits(
            malformed, short1, short2, consume_ld_cache=True)
    assert not malformed.closed and len(malformed.blocks) == 1
