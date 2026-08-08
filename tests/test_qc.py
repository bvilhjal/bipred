"""LD-consistency screening: does it keep clean variants and catch broken ones."""

import numpy as np
import pytest

from bipred.qc import (
    dentist, dentist_statistic, implied_sample_size, in_long_range_ld,
    sd_consistency,
)


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


def _sumstats(n_variants=5000, n_eff=100_000.0, af=None, seed=0, binary=False):
    """Summary statistics whose implied SD matches the reference by construction."""
    rng = np.random.default_rng(seed)
    if af is None:
        af = rng.uniform(0.05, 0.95, n_variants)
    sd_ref = np.sqrt(2 * af * (1 - af))
    scale = 2.0 if binary else 1.0
    # Invert sd_ss = scale / sqrt(n se^2 + beta^2) at sd_ss == sd_ref.
    beta = rng.standard_normal(n_variants) * 1e-3
    se = np.sqrt(np.maximum(scale**2 / sd_ref**2 - beta**2, 1e-12) / n_eff)
    return beta, se, af


def test_sd_consistency_keeps_self_consistent_sumstats():
    beta, se, af = _sumstats()
    keep, offset = sd_consistency(beta, se, np.full(beta.size, 100_000.0), af)
    assert keep.mean() > 0.99, f"dropped {100*(1-keep.mean()):.1f}% of clean data"
    assert 0.9 < offset < 1.1, offset


def test_sd_consistency_flags_a_wrong_sample_size():
    """The error it exists for: N inflated, so the implied SD is too small."""
    beta, se, af = _sumstats(seed=1)
    n_true = np.full(beta.size, 100_000.0)
    # Claim 9x the sample size actually used; nothing else changes. The
    # implied SD falls by sqrt(9) = 3, which clears the 0.5x lower bound --
    # a 4x overstatement lands at exactly 0.5 and is deliberately not caught.
    keep, offset = sd_consistency(beta, se, 9.0 * n_true, af, normalise=False)
    assert 0.3 < offset < 0.4, offset
    assert keep.mean() < 0.05


def test_sd_consistency_normalisation_makes_traits_comparable():
    """A binary trait on a mis-specified N must not be judged on a shifted scale.

    Without normalisation the same threshold means different things for two
    traits: on real CARDIoGRAMplusC4D data the unnormalised ratio sat at 0.755
    while a well-specified trait sat near 1, so tightening the bound removed
    83% of the genome from one and almost nothing from the other.
    """
    beta, se, af = _sumstats(seed=2, binary=True)
    n = np.full(beta.size, 100_000.0)
    raw_keep, offset = sd_consistency(beta, se, 9.0 * n, af, binary=True,
                                      normalise=False)
    norm_keep, _ = sd_consistency(beta, se, 9.0 * n, af, binary=True,
                                  normalise=True)
    assert offset < 0.4                        # the scale really is shifted
    assert raw_keep.mean() < norm_keep.mean()  # normalising rescues them
    assert norm_keep.mean() > 0.9


def test_implied_sample_size_recovers_a_known_n():
    beta, se, af = _sumstats(n_eff=80_000.0, seed=3, binary=True)
    out = implied_sample_size(beta, se, af, binary=True,
                              reported_n=np.full(beta.size, 80_000.0))
    assert abs(out["median"] - 80_000.0) / 80_000.0 < 0.02
    assert out["consistent"] is True


def test_implied_sample_size_catches_an_overstated_n():
    """CAD's case: reported 162,973, implied 92,966, ratio 0.570."""
    beta, se, af = _sumstats(n_eff=92_966.0, seed=4, binary=True)
    reported = np.full(beta.size, 162_973.0)
    out = implied_sample_size(beta, se, af, binary=True, reported_n=reported)
    assert out["consistent"] is False
    assert 0.5 < out["ratio"] < 0.62, out["ratio"]


def test_long_range_regions_cover_the_expected_span():
    from bipred.qc import APOE_HG19, LONG_RANGE_LD_HG19
    assert len(LONG_RANGE_LD_HG19) == 24            # Price et al. 2008
    chrom = np.array(["6", "6", "19", "1", "22"])
    pos = np.array([30_000_000, 40_000_000, 45_400_000, 50_000_000, 20_000_000])
    with_apoe = in_long_range_ld(chrom, pos)
    without = in_long_range_ld(chrom, pos, include_apoe=False)
    assert with_apoe.tolist() == [True, False, True, True, False]
    # APOE is not one of the 24; excluding it must free chr19 alone.
    assert without.tolist() == [True, False, False, True, False]
    assert APOE_HG19[0] == "19"
