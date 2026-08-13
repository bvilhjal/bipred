"""Public on-ramp: subset_blocks, prepare_bivariate_sumstats, write_weights."""

import warnings

import numpy as np
import pytest

from bipred import (BivariateResult, prepare_bivariate_sumstats, subset_blocks,
                    ldpred3_auto_bivariate_blocks)
from ldpred3 import LowRankLD, save_ld_blocks
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


def _cache_and_sumstats(tmp_path, m=20):
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
        str(cache), [(R, np.arange(m))], ids,
        counted_allele=a1, other_allele=a2, chrom=chrom, pos=pos,
        reference_af=af, n_ref=500, ridge=0.0)
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


def test_write_weights_uses_hwe_sd_from_cache_af(tmp_path):
    cache, p1, p2, *_ = _cache_and_sumstats(tmp_path, m=16)
    prep = prepare_bivariate_sumstats(cache, p1, p2, n_eff1=10_000,
                                      n_eff2=10_000, qc=False)
    res = ldpred3_auto_bivariate_blocks(
        prep.blocks, prep.beta_hat1, prep.beta_hat2, prep.n_eff1, prep.n_eff2,
        burn_in=5, num_iter=5, seed=0)
    path = tmp_path / "t1.weights"
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
