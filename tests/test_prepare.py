"""Public on-ramp: subset_blocks, prepare_bivariate_sumstats, write_weights."""

import warnings

import numpy as np
import pytest

from bipred import (BivariateResult, prepare_bivariate_sumstats, subset_blocks,
                    ldpred3_auto_bivariate_blocks)
from ldpred3 import LowRankLD, save_ld_blocks
from ldpred3.interop import prepare_ld_cache
from ldpred3.weights import read_weights


def _ar1(k, rho=0.6):
    i = np.arange(k)
    R = rho ** np.abs(i[:, None] - i[None, :])
    return np.ascontiguousarray(R.astype(np.float32))


def test_subset_blocks_retils_to_a_contiguous_cover():
    R0, R1 = _ar1(6), _ar1(6)
    blocks = [(R0, np.arange(6)), (R1, np.arange(6, 12))]
    keep = np.zeros(12, dtype=bool)
    keep[[1, 2, 3, 8, 9, 10]] = True
    tiled, kept = subset_blocks(blocks, keep)
    assert list(kept) == [1, 2, 3, 8, 9, 10]
    assert [tuple(idx) for _, idx in tiled] == [(0, 1, 2), (3, 4, 5)]
    assert tiled[0][0].shape == (3, 3)


def test_subset_blocks_keeps_lowrank_rows_when_rank_fits():
    k = 8
    R = _ar1(k, 0.8).astype(np.float64)
    w, V = np.linalg.eigh(R)
    U = (V[:, -3:] * np.sqrt(np.maximum(w[-3:], 0))).astype(np.float32)
    lr = LowRankLD(U, k, scale=1.0)
    tiled, kept = subset_blocks([(lr, np.arange(k))], [0, 1, 2, 3, 4])
    assert isinstance(tiled[0][0], LowRankLD)
    assert tiled[0][0].U.shape == (5, 3)
    assert list(kept) == [0, 1, 2, 3, 4]


def _write_sumstats(path, ids, a1, a2, beta, se, n, chrom="1", pos=None):
    pos = np.arange(1, len(ids) + 1) if pos is None else pos
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN\n")
        for i, sid in enumerate(ids):
            fh.write(f"{sid}\t{chrom}\t{int(pos[i])}\t{a1[i]}\t{a2[i]}\t"
                     f"{beta[i]:.8g}\t{se[i]:.8g}\t{n}\n")


def _cache_and_sumstats(tmp_path, m=20, *, with_af=True, mmap=False):
    rng = np.random.default_rng(1)
    R = _ar1(m)
    ids = np.array([f"rs{i}" for i in range(m)], dtype=object)
    a1 = np.array(["A"] * m, dtype=object)
    a2 = np.array(["G"] * m, dtype=object)
    chrom = np.array(["1"] * m, dtype=object)
    pos = np.arange(1, m + 1)
    af = np.full(m, 0.3)
    cache = tmp_path / "ld.npz"
    save_ld_blocks(
        str(cache), [(R, np.arange(m))], ids, mmap=mmap,
        counted_allele=a1, other_allele=a2, chrom=chrom, pos=pos,
        reference_af=af if with_af else None, n_ref=500, ridge=0.0)
    n = 10_000
    se = np.full(m, 1.0 / np.sqrt(n))
    b1 = rng.normal(scale=0.02, size=m)
    b2 = 0.6 * b1 + rng.normal(scale=0.015, size=m)
    # Flip trait 2 alleles on a few SNPs so alignment has to sign-flip.
    a1_2, a2_2 = a1.copy(), a2.copy()
    a1_2[::5], a2_2[::5] = a2[::5], a1[::5]
    b2_file = b2.copy()
    b2_file[::5] *= -1
    p1 = tmp_path / "t1.tsv"
    p2 = tmp_path / "t2.tsv"
    _write_sumstats(p1, ids, a1, a2, b1, se, n, pos=pos)
    _write_sumstats(p2, ids, a1_2, a2_2, b2_file, se, n, pos=pos)
    return str(cache), str(p1), str(p2), b1, b2, n, ids, af


def test_prepare_aligns_and_standardizes(tmp_path):
    cache, p1, p2, b1, b2, n, ids, af = _cache_and_sumstats(tmp_path)
    prep = prepare_bivariate_sumstats(cache, p1, p2, n_eff1=n, n_eff2=n, qc=False)
    assert prep.beta_hat1.shape == (20,)
    assert list(prep.id) == list(ids)
    # Trait 2 was allele-flipped on every 5th SNP; standardized effects
    # should still match the cache-oriented truth.
    from ldpred3 import standardize_betas
    truth2 = standardize_betas(b2, np.full(20, 1.0 / np.sqrt(n)), n)[0]
    assert np.allclose(prep.beta_hat2, truth2, atol=1e-6)
    assert np.allclose(prep.af, af)


def test_prepare_n_cases_uses_ldpred3_n_eff(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path)
    prep = prepare_bivariate_sumstats(
        cache, p1, p2, n_cases2=100, n_controls2=300, n_eff1=10_000, qc=False)
    expected = 4.0 / (1 / 100 + 1 / 300)
    assert np.allclose(prep.n_eff2, expected)


def test_prepare_rejects_scalar_n_eff_with_case_controls(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path)
    with pytest.raises(ValueError, match="scalar n_eff or n_cases"):
        prepare_bivariate_sumstats(
            cache, p1, p2, n_eff1=80_000, n_cases1=12_000, n_controls1=38_000,
            n_eff2=10_000, qc=False)


def test_prepare_names_the_trait_with_no_usable_variants(tmp_path):
    """A trait that loses every variant names itself, for per-trait blame."""
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path)
    m = 20
    off = tmp_path / "off-reference.tsv"
    # Nothing may match: distinct ids, chromosome, positions, and alleles
    # (harmonize falls back to chrom:pos:alleles when ids do not match).
    _write_sumstats(off, [f"off{i}" for i in range(m)], ["C"] * m, ["T"] * m,
                    [0.01] * m, [0.001] * m, 10_000, chrom="2",
                    pos=np.arange(1000, 1000 + m))
    with pytest.raises(ValueError,
                       match="trait1: all GWAS variants were removed"):
        prepare_bivariate_sumstats(cache, str(off), p2, n_eff1=10_000,
                                   n_eff2=10_000, qc=False)
    with pytest.raises(ValueError,
                       match="trait2: all GWAS variants were removed"):
        prepare_bivariate_sumstats(cache, p1, str(off), n_eff1=10_000,
                                   n_eff2=10_000, qc=False)


def test_prepare_accepts_distinct_column_mappings_without_mutating_them(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path)
    path = tmp_path / "trait1-custom.tsv"
    path.write_text(
        (tmp_path / "t1.tsv").read_text(encoding="utf-8").replace(
            "\tN\n", "\tSAMPLES\n", 1),
        encoding="utf-8")
    columns = {"n_eff": "SAMPLES"}
    prep = prepare_bivariate_sumstats(
        cache, path, p2, n_eff2=10_000, columns1=columns, qc=False)
    assert columns == {"n_eff": "SAMPLES"}
    assert np.all(prep.n_eff1 == 10_000)
    with pytest.raises(ValueError, match="either a scalar n_eff or an n_eff column"):
        prepare_bivariate_sumstats(
            cache, path, p2, n_eff1=10_000, n_eff2=10_000,
            columns1=columns, qc=False)


def test_prepare_reads_an_n_eff_column_by_index_name_and_digit_string(tmp_path):
    """columns={"n_eff": 7} must read column 7, not force n_eff=7.0."""
    cache, _, p2, *_ = _cache_and_sumstats(tmp_path)
    m = 20
    n_col = 5_000 + 100 * np.arange(m)          # distinct per variant
    path = tmp_path / "t1-neff-column.tsv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tSAMPLES\n")
        for i in range(m):
            fh.write(f"rs{i}\t1\t{i + 1}\tA\tG\t0.01\t0.001\t{n_col[i]}\n")
    for override in (7, "7", "SAMPLES", np.int64(7)):
        prep = prepare_bivariate_sumstats(
            cache, str(path), p2, n_eff2=10_000,
            columns1={"n_eff": override}, qc=False)
        # The per-variant column comes through; a forced scalar would make
        # every entry equal.
        np.testing.assert_array_equal(prep.n_eff1, n_col)
    with pytest.raises(ValueError, match="column name or a zero-based"):
        prepare_bivariate_sumstats(
            cache, str(path), p2, n_eff2=10_000,
            columns1={"n_eff": 7.5}, qc=False)


def test_write_weights_uses_hwe_sd_from_cache_af(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, m=16)
    prep = prepare_bivariate_sumstats(cache, p1, p2, n_eff1=10_000,
                                      n_eff2=10_000, qc=False)
    res = ldpred3_auto_bivariate_blocks(
        prep.blocks, prep.beta_hat1, prep.beta_hat2, prep.n_eff1, prep.n_eff2,
        burn_in=5, num_iter=5, seed=0)
    path = tmp_path / "t1.weights"
    with pytest.warns(RuntimeWarning, match="HWE reference-panel scale"):
        res.write_weights(str(path), trait=1, id=prep.id, chrom=prep.chrom,
                          pos=prep.pos, effect_allele=prep.effect_allele,
                          other_allele=prep.other_allele, af=prep.af)
    wt = read_weights(str(path))
    assert wt.has_scale
    assert np.allclose(wt.weight, res.beta1_est)
    assert np.allclose(wt.sd_ref, np.sqrt(2 * 0.3 * 0.7))


def test_write_weights_rejects_a_length_mismatch():
    res = BivariateResult(
        beta1_est=np.ones(3), beta2_est=np.ones(3), h2=(0.1, 0.1),
        rg=0.0, p=0.02, sigma=np.eye(2))
    with pytest.raises(ValueError, match="provenance"):
        res.write_weights("x", trait=1, id=["a"], chrom=["1"], pos=[1],
                          effect_allele=["A"], other_allele=["G"])


def test_unstandardized_z_scores_warn(tmp_path):
    R = _ar1(12)
    z = np.linspace(-2.0, 2.5, 12)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ldpred3_auto_bivariate_blocks(
            [(R, np.arange(12))], z, z, 20_000, 20_000,
            burn_in=3, num_iter=3, seed=0)
    messages = " ".join(str(w.message) for w in caught)
    assert "|beta_hat| >= 1" in messages


def test_subset_blocks_strict_indices_sets_singletons_and_views():
    first, second = _ar1(4), _ar1(3)
    blocks = [(first, np.arange(4)), (second, np.arange(4, 7))]

    tiled, kept = subset_blocks(blocks, {1, 6})
    assert kept.tolist() == [1, 6]
    assert [idx.tolist() for _, idx in tiled] == [[0], [1]]
    assert [R.shape for R, _ in tiled] == [(1, 1), (1, 1)]

    whole, kept = subset_blocks(blocks, np.ones(7, dtype=bool))
    assert whole is blocks
    assert whole[0][0] is first
    assert kept.tolist() == list(range(7))
    consecutive, _ = subset_blocks(blocks, [1, 2])
    assert np.shares_memory(consecutive[0][0], first)

    with pytest.raises(ValueError, match="length"):
        subset_blocks(blocks, np.ones(3, dtype=bool))
    with pytest.raises(TypeError, match="integer"):
        subset_blocks(blocks, [1.0])
    with pytest.raises(ValueError, match="duplicate"):
        subset_blocks(blocks, [1, 1])
    with pytest.raises(IndexError, match=r"\[0, 7\)"):
        subset_blocks(blocks, [-1])
    with pytest.raises(IndexError, match=r"\[0, 7\)"):
        subset_blocks(blocks, [7])


def test_prepare_screen_uses_joint_principal_panel_not_zero_filling(tmp_path):
    """A missing neighbour must not become an observed z=0 in the screen."""
    m, n = 120, 10_000
    R = _ar1(m, rho=0.5)
    ids = np.array([f"rs{i}" for i in range(m)], dtype=object)
    alleles1 = np.array(["A"] * m, dtype=object)
    alleles2 = np.array(["G"] * m, dtype=object)
    cache = tmp_path / "screen.npz"
    save_ld_blocks(
        cache, [(R, np.arange(m))], ids,
        counted_allele=alleles1, other_allele=alleles2,
        chrom=np.array(["1"] * m), pos=np.arange(1, m + 1),
        reference_af=np.full(m, 0.3), n_ref=500, ridge=0.0)

    rng = np.random.default_rng(6)
    observed = np.sort(rng.choice(m, 84, replace=False))
    z = rng.normal(size=m)
    outlier = int(rng.integers(observed.size))
    z[observed[outlier]] = rng.choice([-1, 1]) * 8.0
    assert observed[outlier] == 8 and z[8] == -8.0
    se = np.full(observed.size, 1.0 / np.sqrt(n))
    beta = z[observed] * se
    p1, p2 = tmp_path / "screen1.tsv", tmp_path / "screen2.tsv"
    for path in (p1, p2):
        _write_sumstats(
            path, ids[observed], alleles1[observed], alleles2[observed],
            beta, se, n, pos=observed + 1)

    prep = prepare_bivariate_sumstats(
        cache, p1, p2, n_eff1=n, n_eff2=n, qc=False,
        screen=True, screen_seed=11)
    # Correct principal-panel screening drops the injected outlier and its
    # inconsistent observed neighbour. The old zero-filled full-cache screen
    # dropped rs8 only and incorrectly retained rs9.
    assert prep.log["n_joint"] == 84
    assert prep.log["n_screen_drop"] == 2
    assert "rs8" not in prep.id and "rs9" not in prep.id


def test_prepare_screen_subsets_a_complete_joint_cache(monkeypatch, tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, mmap=True)

    def screen(blocks, z, **kwargs):
        keep = np.ones(len(z), dtype=bool)
        keep[3] = False
        return keep

    monkeypatch.setattr("bipred.qc.ld_consistency_screen", screen)
    prep = prepare_bivariate_sumstats(
        cache, p1, p2, n_eff1=10_000, n_eff2=10_000,
        qc=False, screen=True)
    assert prep.log["n_joint"] == 20
    assert prep.log["n_screen_drop"] == 1
    assert len(prep.id) == 19 and "rs3" not in prep.id
    owner = prep._ld_owner
    prep.close()
    assert owner.closed


def test_prepare_keeps_mmap_owner_alive_until_close(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, mmap=True)
    prep = prepare_bivariate_sumstats(
        cache, p1, p2, n_eff1=10_000, n_eff2=10_000, qc=False)
    owner = prep._ld_owner
    assert owner is not None and not owner.closed
    # The fit reads the mmap-backed view before explicit release.
    result = ldpred3_auto_bivariate_blocks(
        prep.blocks, prep.beta_hat1, prep.beta_hat2,
        prep.n_eff1, prep.n_eff2, burn_in=2, num_iter=2, seed=0)
    assert np.all(np.isfinite(result.beta1_est))
    prep.close()
    assert owner.closed and prep.blocks == []


def test_prepare_reuses_a_caller_owned_validated_cache(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, mmap=True)
    with prepare_ld_cache(cache) as shared:
        prep = prepare_bivariate_sumstats(
            shared, p1, p2, n_eff1=10_000, n_eff2=10_000, qc=False)
        assert prep.log["prepared_cache"] is True
        assert prep._ld_owner is None
        prep.close()
        assert not shared.closed
        assert len(prep.blocks) == 1
    assert shared.closed


def test_missing_cache_af_writes_safe_target_scaled_weights(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, with_af=False)
    prep = prepare_bivariate_sumstats(
        cache, p1, p2, n_eff1=10_000, n_eff2=10_000, qc=False)
    assert prep.af is None
    res = BivariateResult(
        beta1_est=np.full(len(prep.id), 0.01),
        beta2_est=np.full(len(prep.id), -0.01), h2=(0.1, 0.1),
        rg=0.0, p=0.02, sigma=np.eye(2))
    path = tmp_path / "target.weights"
    res.write_weights(
        path, trait=1, id=prep.id, chrom=prep.chrom, pos=prep.pos,
        effect_allele=prep.effect_allele, other_allele=prep.other_allele,
        af=prep.af)
    assert not read_weights(path).has_scale
