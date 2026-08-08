"""LD-consistency screening: does it keep clean variants and catch broken ones."""

import numpy as np
import pytest

from bipred.qc import dentist, dentist_statistic


def _ar1(rho, k):
    pos = np.arange(k)
    return rho ** np.abs(pos[:, None] - pos[None, :])


def _clean_panel(k=400, blocks=3, rho=0.7, seed=0):
    """Blocks plus z-scores drawn from the model DENTIST assumes: z ~ N(R b, R)."""
    rng = np.random.default_rng(seed)
    R = _ar1(rho, k)
    chol = np.linalg.cholesky(R + 1e-8 * np.eye(k))
    blocks_out, z = [], []
    for b in range(blocks):
        beta = np.zeros(k)
        beta[rng.choice(k, 4, replace=False)] = rng.standard_normal(4) * 0.05
        z.append(R @ beta * np.sqrt(20000) + chol @ rng.standard_normal(k))
        blocks_out.append((R.astype(np.float32), np.arange(b * k, (b + 1) * k)))
    return blocks_out, np.concatenate(z)


def test_clean_data_survives_screening():
    """A false-positive rate this test would catch: p=5e-8 should drop ~nobody."""
    blocks, z = _clean_panel()
    keep = dentist(blocks, z, rounds=2)
    assert keep.mean() > 0.99, f"dropped {100*(1-keep.mean()):.1f}% of clean data"


def test_a_sign_flipped_variant_is_caught():
    """The error harmonisation cannot see: an allele flip inside strong LD.

    Its own z stays a plausible size, so no per-variant filter (frequency,
    imputation quality, a chi-square cap) can flag it. Only its disagreement
    with the correlated variants around it gives it away.
    """
    blocks, z = _clean_panel(rho=0.9, seed=1)
    victim = 150
    z = z.copy()
    z[victim] = -z[victim] - 8.0            # inconsistent with its neighbours
    keep = dentist(blocks, z, rounds=2)
    assert not keep[victim]


def test_statistic_is_calibrated_under_the_null():
    """T should look like chi2_1 when the z really are consistent with the LD."""
    rng = np.random.default_rng(3)
    k = 300
    R = _ar1(0.6, k)
    chol = np.linalg.cholesky(R + 1e-8 * np.eye(k))
    stats = []
    for _ in range(30):
        z = chol @ rng.standard_normal(k)
        order = rng.permutation(k)
        stats.append(dentist_statistic(R, z, order[:k // 2], order[k // 2:]))
    stats = np.concatenate(stats)
    # chi2_1 has median 0.455; allow generous slack for a finite, LD-correlated
    # sample -- this catches a statistic that is wrong by a factor, not by noise.
    assert 0.2 < np.median(stats) < 1.2, np.median(stats)


def test_length_and_finiteness_are_checked():
    blocks, z = _clean_panel(k=100, blocks=2)
    with pytest.raises(ValueError, match="blocks span"):
        dentist(blocks, z[:-1])
    bad = z.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        dentist(blocks, bad)


def test_low_rank_blocks_need_no_densifying():
    """A low-rank block is screened through its factor, not a dense expansion."""
    from ldpred3 import lowrank_ld
    k = 300
    R = _ar1(0.7, k)
    factor = lowrank_ld(R, variance=0.99, quantize=True)
    rng = np.random.default_rng(5)
    chol = np.linalg.cholesky(R + 1e-8 * np.eye(k))
    z = chol @ rng.standard_normal(k)
    keep = dentist([(factor, np.arange(k))], z, rounds=1)
    assert keep.shape == (k,)
    assert keep.mean() > 0.95
