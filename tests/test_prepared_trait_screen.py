"""Trait-local LD-consistency screening before bivariate pairing."""

from copy import deepcopy

import numpy as np
import pytest

import bipred
import bipred.prepare as prepare_module
from bipred import PreparedTrait, screen_prepared_trait
from ldpred3 import save_ld_blocks
from ldpred3.interop import prepare_ld_cache


def _reference(tmp_path, *, m=8, mmap=False):
    positions = np.arange(m)
    corr = 0.65 ** np.abs(positions[:, None] - positions[None, :])
    corr = np.ascontiguousarray(corr, dtype=np.float32)
    ids = np.array([f"rs{i}" for i in positions], dtype=object)
    cache = tmp_path / ("ld-mmap.npz" if mmap else "ld.npz")
    save_ld_blocks(
        str(cache), [(corr, positions)], ids, mmap=mmap,
        counted_allele=np.array(["A"] * m, dtype=object),
        other_allele=np.array(["G"] * m, dtype=object),
        chrom=np.array(["1"] * m, dtype=object), pos=positions + 1,
        n_ref=500, ridge=0.0)
    return cache, corr


def _trait(*, n_cache=8, label="reusable CAD"):
    indices = np.array([0, 2, 3, 6], dtype=np.int64)
    return PreparedTrait(
        indices=indices,
        beta_hat=np.array([0.01, 0.02, -0.03, 0.04]),
        n_eff=np.array([100_000.0] * 4),
        z=np.array([1.0, 8.0, -2.0, 0.5]),
        eaf=np.array([0.2, 0.3, np.nan, 0.4]),
        n_cache=n_cache,
        log={"label": label, "qc": {"n_input": 10, "n_kept": 4}})


def test_screen_uses_raw_z_and_selected_rows_without_principal_panel(
        monkeypatch, tmp_path):
    cache, corr = _reference(tmp_path)
    trait = _trait()
    arrays_before = {
        name: getattr(trait, name).copy()
        for name in ("indices", "beta_hat", "n_eff", "z", "eaf")
    }
    log_before = deepcopy(trait.log)
    observed = {}
    events = []

    def fake_screen(blocks, selection, z, **kwargs):
        observed["blocks"] = [
            (np.asarray(matrix).copy(), np.asarray(local).copy())
            for matrix, local in blocks
        ]
        observed["selection"] = np.asarray(selection).copy()
        observed["z"] = np.asarray(z).copy()
        observed["kwargs"] = kwargs
        return np.array([True, False, True, True])

    def forbid_principal_subset(*args, **kwargs):
        raise AssertionError("screening must not materialize a principal panel")

    monkeypatch.setattr(
        "bipred.qc._ld_consistency_screen_selected", fake_screen)
    monkeypatch.setattr(prepare_module, "subset_blocks", forbid_principal_subset)
    monkeypatch.setattr(
        prepare_module, "subset_ld_blocks", forbid_principal_subset)
    screened = screen_prepared_trait(
        cache, trait, rounds=3, window=25, threshold=10.0,
        eigenvalue_floor=2e-3, seed=7, ncores=2, verbose=True,
        progress=events.append)

    source = arrays_before["indices"]
    assert len(observed["blocks"]) == 1
    matrix, local = observed["blocks"][0]
    np.testing.assert_array_equal(matrix, corr)
    np.testing.assert_array_equal(local, np.arange(corr.shape[0]))
    np.testing.assert_array_equal(observed["selection"], source)
    np.testing.assert_array_equal(observed["z"], arrays_before["z"])
    assert observed["kwargs"]["progress"] == events.append
    assert observed["kwargs"]["progress_label"] == (
        "LD consistency screen, reusable CAD")

    keep = np.array([True, False, True, True])
    for name, original in arrays_before.items():
        np.testing.assert_array_equal(
            getattr(screened, name), original[keep])
        np.testing.assert_array_equal(getattr(trait, name), original)
        assert not np.shares_memory(getattr(screened, name), getattr(trait, name))
    assert trait.log == log_before
    assert screened.log["screen"] is True
    record = screened.log["ld_consistency_screen"]
    assert record == {
        "n_input": 4, "n_tested": 4, "n_untested": 0,
        "n_kept": 3, "n_dropped": 1,
        "parameters": {
            "rounds": 3, "window": 25, "threshold": 10.0,
            "eigenvalue_floor": 2e-3, "seed": 7, "ncores": 2,
            "verbose": True,
        },
    }


@pytest.mark.parametrize("mmap", [False, True])
def test_path_loaded_cache_owner_is_closed(monkeypatch, tmp_path, mmap):
    cache, _ = _reference(tmp_path, mmap=mmap)
    opened = []
    real_open = prepare_module.prepare_ld_cache

    def tracked_open(path):
        owner = real_open(path)
        opened.append(owner)
        return owner

    monkeypatch.setattr(prepare_module, "prepare_ld_cache", tracked_open)
    monkeypatch.setattr(
        "bipred.qc._ld_consistency_screen_selected",
        lambda blocks, selection, z, **kwargs: np.ones(len(z), dtype=bool))

    screened = screen_prepared_trait(cache, _trait())
    assert len(screened) == 4
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.parametrize("mmap", [False, True])
def test_caller_owned_cache_remains_open_and_unchanged(tmp_path, mmap):
    path, _ = _reference(tmp_path, mmap=mmap)
    owner = prepare_ld_cache(path)
    source = [block for block, _idx in owner.blocks]
    try:
        with pytest.warns(RuntimeWarning, match="never entered a window"):
            screened = screen_prepared_trait(owner, _trait())
        assert len(screened) == 4
        assert not owner.closed
        assert len(owner.blocks) == 1 and owner.blocks[0][0] is source[0]
        assert np.isfinite(np.asarray(owner.blocks[0][0])[0, 0])
    finally:
        owner.close()


def test_too_few_retained_variants_names_trait_and_closes_mmap(
        monkeypatch, tmp_path):
    cache, _ = _reference(tmp_path, mmap=True)
    opened = []
    real_open = prepare_module.prepare_ld_cache

    def tracked_open(path):
        owner = real_open(path)
        opened.append(owner)
        return owner

    monkeypatch.setattr(prepare_module, "prepare_ld_cache", tracked_open)
    monkeypatch.setattr(
        "bipred.qc._ld_consistency_screen_selected",
        lambda blocks, selection, z, **kwargs: np.array(
            [False, True, False, False]))

    with pytest.raises(
            ValueError,
            match="reusable CAD: LD-consistency screening left fewer than two"):
        screen_prepared_trait(cache, _trait())
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.parametrize("mmap", [False, True])
def test_screen_failure_releases_path_loaded_owner(monkeypatch, tmp_path, mmap):
    cache, _ = _reference(tmp_path, mmap=mmap)
    opened = []
    real_open = prepare_module.prepare_ld_cache

    def tracked_open(path):
        owner = real_open(path)
        opened.append(owner)
        return owner

    def failed_screen(blocks, selection, z, **kwargs):
        raise RuntimeError("injected selected-row failure")

    monkeypatch.setattr(prepare_module, "prepare_ld_cache", tracked_open)
    monkeypatch.setattr(
        "bipred.qc._ld_consistency_screen_selected", failed_screen)

    with pytest.raises(RuntimeError, match="selected-row failure"):
        screen_prepared_trait(cache, _trait())
    assert len(opened) == 1 and opened[0].closed


def test_screen_validates_trait_against_selected_cache(tmp_path):
    cache, _ = _reference(tmp_path)
    with pytest.raises(
            ValueError, match="reusable CAD: prepared against 9 cache variants"):
        screen_prepared_trait(cache, _trait(n_cache=9))


def test_screen_helper_is_public():
    assert bipred.screen_prepared_trait is screen_prepared_trait
    assert "screen_prepared_trait" in bipred.__all__
