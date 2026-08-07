"""Regional genetic correlation: exactness, LD-representation agreement, recovery."""

import numpy as np
import pytest

from ldpred3 import LowRankLD

from bipred import (
    RegionalRgResult,
    ldpred3_auto_bivariate_blocks,
    regional_rg,
)
from bipred.regional import _Q8_SCALE


def _ar1(rho, k):
    d = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    return (rho ** d).astype(np.float32)


def test_matches_hand_computed_quadratics():
    # Two regions inside one block; the estimate must equal the LD-aware ratio
    # computed directly on each region's sub-block.
    R = np.array([[1.0, 0.5, 0.0, 0.0],
                  [0.5, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.3],
                  [0.0, 0.0, 0.3, 1.0]], dtype=np.float32)
    b1 = np.array([1.0, 2.0, 3.0, 4.0])
    b2 = np.array([2.0, 1.0, -1.0, 1.0])
    res = regional_rg(b1, b2, [(R, np.arange(4))], [0, 0, 1, 1])

    assert isinstance(res, RegionalRgResult) and len(res) == 2
    np.testing.assert_array_equal(res.region, [0, 1])
    np.testing.assert_array_equal(res.n_variants, [2, 2])
    for c, sl in enumerate((slice(0, 2), slice(2, 4))):
        Rr = R[sl, sl].astype(float)
        x, y = b1[sl], b2[sl]
        q11, q12, q22 = x @ Rr @ x, x @ Rr @ y, y @ Rr @ y
        assert res.gvar1[c] == pytest.approx(q11)
        assert res.gcov[c] == pytest.approx(q12)
        assert res.gvar2[c] == pytest.approx(q22)
        assert res.rg[c] == pytest.approx(q12 / np.sqrt(q11 * q22))


def test_block_diagonal_sum_is_exact_across_blocks():
    # A region spanning two blocks must equal the sum of its within-block
    # quadratics -- LD is block-diagonal, so no cross-block term is dropped.
    k = 5
    A, B = _ar1(0.6, k), _ar1(0.2, k)
    rng = np.random.default_rng(0)
    b1, b2 = rng.normal(size=2 * k), rng.normal(size=2 * k)
    blocks = [(A, np.arange(k)), (B, np.arange(k, 2 * k))]

    whole = regional_rg(b1, b2, blocks, np.zeros(2 * k, dtype=int))
    split = regional_rg(b1, b2, blocks, np.repeat([0, 1], k))

    assert whole.gvar1[0] == pytest.approx(split.gvar1.sum())
    assert whole.gcov[0] == pytest.approx(split.gcov.sum())
    assert whole.gvar2[0] == pytest.approx(split.gvar2.sum())


def test_int8_blocks_track_float_blocks():
    # int8 LD is dequantised on the fly; it should agree closely with float LD.
    k = 40
    R = _ar1(0.5, k)
    q8 = np.rint(np.clip(R, -1.0, 1.0) * 127.0).astype(np.int8)
    rng = np.random.default_rng(1)
    b1, b2 = rng.normal(size=k), rng.normal(size=k)
    reg = np.repeat([0, 1], k // 2)

    flt = regional_rg(b1, b2, [(R, np.arange(k))], reg)
    i8 = regional_rg(b1, b2, [(q8, np.arange(k))], reg)
    assert np.max(np.abs(flt.rg - i8.rg)) < 0.01
    assert _Q8_SCALE == pytest.approx(1.0 / 127.0)


def test_lowrank_matches_its_dense_equivalent():
    # A LowRankLD factor and the dense matrix it represents must give the same
    # regional quadratics, without the low-rank path densifying anything.
    rng = np.random.default_rng(2)
    k, rank = 12, 4
    # float LowRankLD requires scale == 1, so scale the factor itself and keep a
    # strictly positive residual diagonal (the low-rank-plus-diagonal contract).
    U = rng.normal(size=(k, rank))
    U /= 2.0 * np.linalg.norm(U, axis=1, keepdims=True)      # row norm^2 = 0.25
    U = U.astype(np.float32)
    W = U.astype(np.float64)
    resid = 1.0 - np.einsum("ij,ij->i", W, W)
    assert np.all(resid > 0), "fixture must keep a positive residual diagonal"
    dense = (W @ W.T + np.diag(resid)).astype(np.float32)

    lr = LowRankLD(U, k, scale=1.0, residual_diag=resid)
    b1, b2 = rng.normal(size=k), rng.normal(size=k)
    reg = np.repeat([0, 1], k // 2)

    got = regional_rg(b1, b2, [(lr, np.arange(k))], reg)
    ref = regional_rg(b1, b2, [(dense, np.arange(k))], reg)
    np.testing.assert_allclose(got.rg, ref.rg, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got.gcov, ref.gcov, rtol=1e-5, atol=1e-6)


def test_recovers_heterogeneous_regional_rg():
    # Regions with genuinely different genetic correlations must come out
    # ordered and close to truth when the effects are the truth themselves
    # (this isolates the estimator from sampler shrinkage).
    rng = np.random.default_rng(3)
    k = 60
    truth = [0.0, 0.0, 0.5, 0.5, 0.9, 0.9]
    blocks, b1, b2, reg = [], [], [], []
    for i, rg in enumerate(truth):
        R = _ar1(0.4, k)
        L = np.array([[1.0, 0.0], [rg, np.sqrt(1 - rg * rg)]])
        raw = L @ rng.normal(size=(2, k))
        blocks.append((R, np.arange(i * k, (i + 1) * k)))
        b1.append(raw[0]); b2.append(raw[1]); reg.append(np.full(k, i))
    res = regional_rg(np.concatenate(b1), np.concatenate(b2), blocks,
                      np.concatenate(reg))

    assert np.mean(res.rg[:2]) == pytest.approx(0.0, abs=0.25)
    assert np.mean(res.rg[2:4]) == pytest.approx(0.5, abs=0.25)
    assert np.mean(res.rg[4:]) == pytest.approx(0.9, abs=0.15)
    assert np.mean(res.rg[:2]) < np.mean(res.rg[2:4]) < np.mean(res.rg[4:])


def test_region_labels_may_be_strings_and_keep_first_appearance_order():
    R = _ar1(0.3, 6)
    rng = np.random.default_rng(4)
    b1, b2 = rng.normal(size=6), rng.normal(size=6)
    res = regional_rg(b1, b2, [(R, np.arange(6))],
                      ["chr2", "chr2", "chr1", "chr1", "chr3", "chr3"])
    np.testing.assert_array_equal(res.region, ["chr2", "chr1", "chr3"])
    np.testing.assert_array_equal(res.n_variants, [2, 2, 2])


def test_min_variants_blanks_small_regions():
    R = _ar1(0.3, 6)
    rng = np.random.default_rng(5)
    b1, b2 = rng.normal(size=6), rng.normal(size=6)
    reg = [0, 0, 0, 0, 1, 1]
    res = regional_rg(b1, b2, [(R, np.arange(6))], reg, min_variants=3)
    assert np.isfinite(res.rg[0]) and np.isnan(res.rg[1])
    # The quadratics are still reported for an under-sized region, so a caller
    # can inspect it. They are not additive across regions inside one LD block
    # (see RegionalRgResult) -- to merge regions, re-run with merged labels.
    assert np.isfinite(res.gcov[1])


def test_degenerate_region_gives_nan_not_an_exception():
    # A region whose trait-2 effects are all zero has zero variance; rg is
    # undefined there and must be NaN rather than an error or a spurious value.
    R = _ar1(0.3, 4)
    b1 = np.array([1.0, 1.0, 1.0, 1.0])
    b2 = np.array([1.0, 1.0, 0.0, 0.0])
    res = regional_rg(b1, b2, [(R, np.arange(4))], [0, 0, 1, 1])
    assert np.isfinite(res.rg[0]) and np.isnan(res.rg[1])


def test_nonpositive_regional_variances_give_nan():
    # An indefinite supplied LD matrix can make both quadratic variances
    # negative. Their positive product must not manufacture a finite rg.
    R = np.array([[1.0, 0.9, 0.9],
                  [0.9, 1.0, -0.9],
                  [0.9, -0.9, 1.0]], dtype=np.float32)
    beta = np.array([-1.0, 1.0, 1.0])
    res = regional_rg(beta, beta, [(R, np.arange(3))], [0, 0, 0])
    assert res.gvar1[0] < 0.0 and res.gvar2[0] < 0.0
    assert np.isnan(res.rg[0])


@pytest.mark.parametrize("kwargs,match", [
    (dict(regions=[0, 0, 1]), "one label per variant"),
    (dict(regions=np.array([[0, 0], [1, 1]])), "one-dimensional"),
    (dict(regions=[0, 0, None, 1]), "None labels"),
    (dict(regions=[0.0, np.nan, 1.0, 1.0]), "non-finite labels"),
    (dict(beta1=np.ones((2, 2))), "one-dimensional"),
    (dict(beta2=np.ones((2, 2))), "one-dimensional"),
    (dict(min_variants=0), "min_variants must be >= 1"),
    (dict(min_variants=1.5), "min_variants must be an integer"),
    (dict(min_variants=np.bool_(True)), "min_variants must be an integer"),
    (dict(clip=1), "clip.*boolean"),
])
def test_validation(kwargs, match):
    R = _ar1(0.3, 4)
    b1 = b2 = np.ones(4)
    call = dict(beta1=b1, beta2=b2, blocks=[(R, np.arange(4))],
                regions=[0, 0, 1, 1])
    call.update(kwargs)
    with pytest.raises((ValueError, TypeError), match=match):
        regional_rg(**call)


def test_rejects_mismatched_or_nonfinite_effects():
    R = _ar1(0.3, 4)
    with pytest.raises(ValueError, match="same length"):
        regional_rg(np.ones(4), np.ones(3), [(R, np.arange(4))], [0] * 4)
    bad = np.array([1.0, np.nan, 1.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        regional_rg(bad, np.ones(4), [(R, np.arange(4))], [0] * 4)


def test_clip_flag_exposes_out_of_range_values():
    # With clip=False a non-PD sub-block may push |rg| past 1; the flag exists so
    # that symptom is visible rather than silently hidden.
    R = np.array([[1.0, 0.9, 0.9],
                  [0.9, 1.0, -0.9],
                  [0.9, -0.9, 1.0]], dtype=np.float32)
    b1 = np.array([-2.0, 0.0, 2.0])
    b2 = np.array([-2.0, 2.0, 0.0])
    clipped = regional_rg(b1, b2, [(R, np.arange(3))], [0, 0, 0], clip=True)
    raw = regional_rg(b1, b2, [(R, np.arange(3))], [0, 0, 0], clip=False)
    assert -1.0 <= clipped.rg[0] <= 1.0
    assert abs(raw.rg[0]) > 1.0


def test_default_fit_and_regional_rg_evaluate_the_same_dense_blocks():
    """The fit consumes dense blocks as given, so passing them here is aligned.

    ``ld_int8`` defaulted to auto-quantising float blocks at or below a cutoff,
    which meant the ordinary call pattern -- the same float blocks to both --
    silently evaluated different LD in each, and regional_rg warned about it.
    The fit now copies nothing, so that call pattern is correct and silent.
    """
    import warnings as _w

    rng = np.random.default_rng(0)
    k, nb = 200, 3
    m = k * nb
    pos = np.arange(k)
    R = (0.5 ** np.abs(pos[:, None] - pos[None, :])).astype(np.float32)
    blocks = [(R, np.arange(b * k, (b + 1) * k)) for b in range(nb)]
    bh1 = rng.normal(scale=0.01, size=m)
    bh2 = 0.6 * bh1 + 0.8 * rng.normal(scale=0.01, size=m)

    res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 20000, 20000,
                                        burn_in=5, num_iter=5, seed=0)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        regional_rg(res.beta1_est, res.beta2_est, blocks,
                    np.repeat(np.arange(nb), k))
    assert caught == [], [str(w.message) for w in caught]

    # The fit holds the caller's own block object, not a converted copy -- which
    # is both what makes the two calls agree and what keeps the fit from
    # allocating a second genome-scale payload.
    from bipred.bivariate import _prepare_block

    prepared, scale = _prepare_block(R, False)
    assert prepared is R and scale == 1.0
    # int8 blocks are likewise consumed as-is.
    R8 = np.rint(R * 127).astype(np.int8)
    prepared8, scale8 = _prepare_block(R8, False)
    assert prepared8 is R8 and scale8 == 1.0 / 127.0


def test_no_warn_on_lowrank_blocks():
    import warnings as _w

    k, r = 100, 5
    rng = np.random.default_rng(0)
    U = rng.standard_normal((k, r)).astype(np.float32) * 0.1
    lr = LowRankLD(U=U, m=k, residual_diag=1.0 - np.einsum("ij,ij->i", U, U))
    b1 = rng.standard_normal(k) * 0.01
    b2 = rng.standard_normal(k) * 0.01
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        regional_rg(b1, b2, [(lr, np.arange(k))], np.ones(k, dtype=int))
    assert not any("auto-quantises" in str(w.message) for w in caught)
