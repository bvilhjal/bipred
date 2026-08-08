"""Cross-trait LD Score regression (genetic correlation): recovery, overlap,
and a bit-exact golden.

``ldsc_rg`` / ``estimate_sample_overlap`` moved to bipred with the rest of the
genetic-correlation machinery; univariate LD scores (``ld_scores``) still come
from ldpred3. The golden value pins the iterated cross-trait LDSC variance
weight, including the ``E[z1*z2] ** 2`` term used by reference LDSC.
"""

import warnings

import numpy as np
import pytest

from bipred import ldsc_rg, LDSCRgResult, estimate_sample_overlap
from bipred.ldsc import _z_from_standardized
from ldpred3 import ld_scores, standardize_betas


def _ar1(k, rho):
    d = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    return (rho ** d).astype(np.float64)


def _varied_blocks(n_blocks, k, seed=0):
    """Block-diagonal AR(1) with rho varying per block, so LD scores span a real
    range (LDSC needs LD-score variation to identify the slope/intercept)."""
    rng = np.random.default_rng(seed)
    blocks, chols = [], []
    for b in range(n_blocks):
        rho = rng.uniform(0.0, 0.9)
        R = _ar1(k, rho)
        blocks.append((R.astype(np.float32), np.arange(b * k, (b + 1) * k)))
        chols.append(np.linalg.cholesky(R))
    return blocks, chols


def _simple_inputs():
    """Exactly linear, positive-h2 LDSC inputs for validation tests."""
    ell = np.array([1.0, 2.0, 4.0, 7.0])
    n = 100.0
    x = n * ell / ell.size
    chisq = 1.0 + 0.2 * x
    beta = np.sqrt(chisq / (n + chisq))
    return beta, ell, n


def test_ldsc_rg_recovers_genetic_correlation():
    k, nb, n1, n2 = 200, 60, 40000, 20000
    blocks, chols = _varied_blocks(nb, k, seed=5)
    m = nb * k
    idxs = [np.arange(b * k, (b + 1) * k) for b in range(nb)]
    ell = ld_scores(blocks)

    def gv(a, b):
        return sum(a[ix] @ (blocks[i][0].astype(float) @ b[ix]) for i, ix in enumerate(idxs))

    def sumstats(beta, n, rng):
        bh = np.empty(m)
        for i, ix in enumerate(idxs):
            bh[ix] = blocks[i][0].astype(float) @ beta[ix] + \
                (chols[i] @ rng.standard_normal(k)) / np.sqrt(n)
        return bh

    for rg_true in (0.0, 0.6):
        ests = []
        for rep in range(5):
            rng = np.random.default_rng(80 + rep)
            c = rng.random(m) < 0.05
            L = np.linalg.cholesky([[1, rg_true], [rg_true, 1]])
            raw = L @ rng.standard_normal((2, c.sum()))
            b1 = np.zeros(m); b2 = np.zeros(m); b1[c] = raw[0]; b2[c] = raw[1]
            b1 *= np.sqrt(0.5 / gv(b1, b1)); b2 *= np.sqrt(0.5 / gv(b2, b2))
            res = ldsc_rg(sumstats(b1, n1, rng), sumstats(b2, n2, rng), ell, n1, n2,
                          n_blocks=60)
            ests.append(res.rg)
        assert abs(np.mean(ests) - rg_true) < 0.15, (rg_true, np.mean(ests))


def test_estimate_sample_overlap_inversion():
    # estimate_sample_overlap inverts the cross-trait intercept:
    # N_shared = intercept * sqrt(N1 N2) / rho_pheno.
    n1, n2, rho_ph = 60000.0, 40000.0, 0.5
    icpt = rho_ph * 30000.0 / np.sqrt(n1 * n2)        # a "true" N_shared = 30000
    res = LDSCRgResult(rg=0.0, rg_se=0.0, gcov=0.0, gcov_intercept=icpt,
                       h2=(0.5, 0.5))
    out = estimate_sample_overlap(res, n1, n2, pheno_corr=rho_ph)
    assert abs(out["n_shared"] - 30000.0) < 1.0
    assert abs(out["overlap_frac"] - 30000.0 / n2) < 1e-6
    assert out["effective_overlap"] == pytest.approx(rho_ph * 30000.0)
    assert out["n_shared_raw"] == pytest.approx(30000.0)
    assert out["cross_corr_valid"] is True
    assert out["physically_consistent"] is True
    # With an unknown nonnegative phenotypic correlation, rho=1 gives the
    # overlap-only lower bound, not an upper bound.
    lower = estimate_sample_overlap(res, n1, n2)
    assert lower["n_shared"] == pytest.approx(rho_ph * 30000.0)
    assert out["sign_consistent"] is True
    with pytest.raises(ValueError):
        estimate_sample_overlap(res, n1, n2, pheno_corr=0.0)


def test_sign_mismatch_is_unidentified_rather_than_zero_overlap():
    """A negative intercept under the default pheno_corr is not "no overlap".

    Measured on GLGC HDL x TG, two lipids assayed in the *same* individuals:
    the intercept is -0.352 and the default pheno_corr=1.0 inverted it to a
    negative count, which the old clip reported as ``n_shared`` 0.0 and
    ``overlap_frac`` 0.0. Read literally that says the studies share nobody,
    for a pair that shares everybody. The quantity is unidentified because the
    phenotypic correlation is negative, so it is nan now, and the intercept --
    the thing you actually pass as ``cross_corr`` -- is reported regardless.
    """
    n1 = n2 = 90000.0
    res = LDSCRgResult(rg=-0.7, rg_se=0.04, gcov=-0.1, gcov_intercept=-0.3521,
                       h2=(0.14, 0.12))
    with pytest.warns(RuntimeWarning, match="opposite sign"):
        out = estimate_sample_overlap(res, n1, n2)
    assert out["sign_consistent"] is False
    assert out["physically_consistent"] is False
    assert np.isnan(out["n_shared"]) and np.isnan(out["overlap_frac"])
    assert out["n_shared_raw"] < 0                      # kept, for diagnosis
    assert out["overlap_corr"] == pytest.approx(-0.3521)

    # Supplying the real (negative) phenotypic correlation identifies it again.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fixed = estimate_sample_overlap(res, n1, n2, pheno_corr=-0.45)
    assert fixed["sign_consistent"] is True
    assert fixed["n_shared"] > 0
    assert 0.0 < fixed["overlap_frac"] <= 1.0


def test_impossible_shared_count_is_nan_not_an_overlap_above_one():
    n1, n2, rho = 100.0, 80.0, 0.1
    res = LDSCRgResult(rg=0.0, rg_se=0.0, gcov=0.0, gcov_intercept=0.5,
                       h2=(0.5, 0.5))
    with pytest.warns(RuntimeWarning, match="exceeds the smaller cohort"):
        out = estimate_sample_overlap(res, n1, n2, pheno_corr=rho)
    assert out["sign_consistent"] is True
    assert out["cross_corr_valid"] is True
    assert out["physically_consistent"] is False
    assert out["n_shared_raw"] > min(n1, n2)
    assert np.isnan(out["n_shared"]) and np.isnan(out["overlap_frac"])


@pytest.mark.parametrize("intercept", [-1.0, 1.0, -1.2, 1.2])
def test_intercept_outside_fit_cross_corr_interval_is_reported(intercept):
    res = LDSCRgResult(rg=0.0, rg_se=0.0, gcov=0.0,
                       gcov_intercept=intercept, h2=(0.5, 0.5))
    with pytest.warns(RuntimeWarning) as caught:
        out = estimate_sample_overlap(
            res, 100.0, 100.0, pheno_corr=np.copysign(1.0, intercept))
    assert any("valid cross_corr interval" in str(w.message) for w in caught)
    assert out["overlap_corr"] == intercept
    assert out["cross_corr_valid"] is False
    # Exactly +/-1 describes singular, full overlap and is physically possible,
    # but the joint fit deliberately requires a positive-definite covariance.
    assert out["physically_consistent"] is (abs(intercept) == 1.0)
    if abs(intercept) > 1.0:
        assert np.isnan(out["n_shared"]) and np.isnan(out["overlap_frac"])


@pytest.mark.parametrize("name,bad", [
    ("beta_hat1", np.empty(0)),
    ("beta_hat1", np.ones((4, 1))),
    ("beta_hat2", np.ones(1)),
    ("ld_scores", np.ones(3)),
    ("beta_hat1", np.array([1.0, 2.0, np.inf, 4.0])),
    ("beta_hat2", np.array([1.0, 2.0, np.nan, 4.0])),
    ("beta_hat1", np.array([0.1, 0.2, 1.0, 0.4])),
    ("beta_hat2", np.array([0.1, 0.2, -1.0, 0.4])),
    ("beta_hat2", np.array([0.1, 0.2, 1.1, 0.4])),
    ("ld_scores", np.array([1.0, 2.0, np.nan, 4.0])),
    ("ld_scores", np.array([1.0, 2.0, 0.0, 4.0])),
    ("ld_scores", np.array([1.0, 2.0, -0.1, 4.0])),
])
def test_ldsc_rg_validates_summary_statistic_vectors(name, bad):
    beta, ell, n = _simple_inputs()
    values = {"beta_hat1": beta, "beta_hat2": beta, "ld_scores": ell,
              "n_eff1": n, "n_eff2": n}
    values[name] = bad
    with pytest.raises(ValueError):
        ldsc_rg(**values)


@pytest.mark.parametrize("name,bad", [
    ("n_eff1", 0.0),
    ("n_eff1", np.inf),
    ("n_eff1", True),
    ("n_eff1", np.ones(3)),
    ("n_eff2", -1.0),
    ("n_eff2", np.ones(4, dtype=bool)),
    ("n_eff1", [True, 100.0, 100.0, 100.0]),
    ("n_eff2", "100"),
    ("n_eff2", np.array([100.0, 100.0, np.nan, 100.0])),
])
def test_ldsc_rg_validates_sample_sizes(name, bad):
    beta, ell, n = _simple_inputs()
    values = {"beta_hat1": beta, "beta_hat2": beta, "ld_scores": ell,
              "n_eff1": n, "n_eff2": n}
    values[name] = bad
    with pytest.raises(ValueError):
        ldsc_rg(**values)


@pytest.mark.parametrize("name,bad", [
    ("m_snps", 0.0),
    ("m_snps", np.nan),
    ("m_snps", [4.0]),
    ("n_iter", -1),
    ("n_iter", 1.0),
    ("n_iter", True),
    ("n_blocks", 0),
    ("n_blocks", 2.0),
    ("n_blocks", False),
    ("constrain_intercept", np.inf),
    ("constrain_intercept", [0.0]),
])
def test_ldsc_rg_validates_control_parameters(name, bad):
    beta, ell, n = _simple_inputs()
    kwargs = {name: bad}
    with pytest.raises(ValueError):
        ldsc_rg(beta, beta, ell, n, n, **kwargs)


def test_ldsc_rg_accepts_per_variant_sample_sizes_and_one_block():
    beta, ell, n = _simple_inputs()
    res = ldsc_rg(beta, beta, ell, np.full(ell.size, n), np.full(ell.size, n),
                  n_blocks=1)
    assert res.rg == pytest.approx(1.0)
    assert np.isnan(res.rg_se)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_ldsc_rg_uses_exact_signed_standardized_effect_to_z_relation(sign):
    ell = np.array([1.0, 2.0, 4.0, 7.0])
    n1, n2 = 10.0, 100.0
    chisq = 1.0 + 2.0 * ell
    z = np.sqrt(chisq)
    beta1 = z / np.sqrt(n1 + chisq)
    beta2 = sign * z / np.sqrt(n2 + chisq)

    res = ldsc_rg(beta1, beta2, ell, n1, n2, n_blocks=1, n_iter=0)

    assert res.h2 == pytest.approx((0.8, 0.08))
    assert res.gcov == pytest.approx(sign * 8.0 / np.sqrt(n1 * n2))
    assert res.gcov_intercept == pytest.approx(sign)
    assert res.rg == pytest.approx(sign)


def test_exact_z_conversion_round_trips_ldpred3_standardize_betas():
    beta = np.array([-0.8, -0.03, 0.02, 1.2])
    se = np.array([0.2, 0.01, 0.04, 0.3])
    n = np.array([25.0, 1000.0, 250.0, 16.0])
    beta_std, _scale = standardize_betas(beta, se, n)

    observed = _z_from_standardized(beta_std, n, "beta_hat")

    np.testing.assert_allclose(observed, beta / se, rtol=2e-15, atol=0.0)


def test_ldsc_rg_common_permutation_changes_only_jackknife_grouping():
    # ldsc_rg has no genomic coordinates to sort or validate. A common
    # permutation therefore preserves the regressions but changes which
    # variants form each contiguous delete-a-block jackknife group.
    rng = np.random.default_rng(0)
    m, n = 40, 1000.0
    ell = np.linspace(1.0, 10.0, m)
    chi1 = 1.0 + 0.30 * ell + rng.normal(0.0, 0.10, m)
    chi2 = 1.0 + 0.25 * ell + rng.normal(0.0, 0.15, m)
    beta1 = np.sqrt(chi1 / (n + chi1))
    beta2 = np.sqrt(chi2 / (n + chi2))
    beta2 *= rng.choice([-1.0, 1.0], m, p=[0.08, 0.92])

    ordered = ldsc_rg(beta1, beta2, ell, n, n, n_blocks=4, n_iter=0)
    perm = np.random.default_rng(5).permutation(m)
    shuffled = ldsc_rg(
        beta1[perm], beta2[perm], ell[perm], n, n, n_blocks=4, n_iter=0)

    assert shuffled.rg == pytest.approx(ordered.rg)
    assert shuffled.gcov == pytest.approx(ordered.gcov)
    assert shuffled.h2 == pytest.approx(ordered.h2)
    assert shuffled.rg_se != pytest.approx(ordered.rg_se)


def test_ldsc_rg_rejects_unidentified_full_fit():
    beta = np.array([0.01, 0.02, 0.03, 0.04])
    with pytest.raises(ValueError, match="singular.*LD scores must vary"):
        ldsc_rg(beta, beta, np.ones(4), 100.0, 100.0, n_blocks=2)


def test_ldsc_rg_singular_jackknife_replicate_makes_se_undefined():
    ell = np.array([1.0, 1.0, 2.0, 2.0])
    n = 100.0
    x = n * ell / ell.size
    chisq = 1.0 + 0.2 * x
    beta = np.sqrt(chisq / (n + chisq))
    res = ldsc_rg(beta, beta, ell, n, n, n_blocks=2, n_iter=0)
    assert res.rg == pytest.approx(1.0)
    assert np.isnan(res.rg_se)


def test_ldsc_rg_nonpositive_h2_is_undefined():
    ell = np.array([1.0, 2.0, 3.0, 4.0])
    chisq = np.array([4.0, 3.0, 2.0, 1.0])
    beta = np.sqrt(chisq / (100.0 + chisq))
    res = ldsc_rg(beta, beta, ell, 100.0, 100.0, n_blocks=2, n_iter=0)
    assert res.h2[0] < 0.0 and res.h2[1] < 0.0
    assert np.isnan(res.rg)
    assert np.isnan(res.rg_se)


def test_ldsc_rg_invalid_jackknife_replicate_makes_se_undefined():
    ell = np.array([1.0, 2.0, 3.0, 4.0])
    chisq = np.array([4.0, 3.0, 2.0, 20.0])
    beta = np.sqrt(chisq / (100.0 + chisq))
    # Deleting the final observation makes both h2 estimates negative.
    invalid = ldsc_rg(beta[:-1], beta[:-1], ell[:-1], 100.0, 100.0,
                      n_blocks=1, n_iter=0)
    assert np.isnan(invalid.rg)
    # A valid-subset SE would look more certain than the failed jackknife is.
    res = ldsc_rg(beta, beta, ell, 100.0, 100.0, n_blocks=4, n_iter=0)
    assert np.isfinite(res.rg)
    assert np.isnan(res.rg_se)


@pytest.mark.parametrize("n1,n2,rho,intercept", [
    (0.0, 10.0, 0.5, 0.1),
    (10.0, np.inf, 0.5, 0.1),
    (10.0, 10.0, np.nan, 0.1),
    (10.0, 10.0, 1.01, 0.1),
    (10.0, 10.0, -1.01, 0.1),
    (10.0, 10.0, 0.5, np.inf),
])
def test_estimate_sample_overlap_validation(n1, n2, rho, intercept):
    res = LDSCRgResult(0.0, 0.0, 0.0, intercept, (0.5, 0.5))
    with pytest.raises(ValueError):
        estimate_sample_overlap(res, n1, n2, pheno_corr=rho)


# --- bit-exact golden for the reference-formula cross-trait WLS weights ---
_LDSC_RG = 1.1078743967665865


def _golden_fixtures():
    rng = np.random.default_rng(0)
    m = 20
    R = (0.4 ** np.abs(np.subtract.outer(np.arange(m), np.arange(m)))).astype(
        np.float64)
    beta_hat = rng.standard_normal(m) * 0.05
    blocks = [(R[:10, :10].astype(np.float32), np.arange(10)),
              (R[10:, 10:].astype(np.float32), np.arange(10, 20))]
    return beta_hat, blocks, m


def _beta_hat2(beta_hat, m):
    """A second, correlated trait for the rg golden."""
    rng2 = np.random.default_rng(1)
    return 0.7 * beta_hat + 0.3 * rng2.standard_normal(m) * 0.05


def test_golden_ldsc_rg():
    beta_hat, blocks, m = _golden_fixtures()
    ell = ld_scores(blocks)
    res = ldsc_rg(beta_hat, _beta_hat2(beta_hat, m), ell, 10000, 10000, m_snps=m)
    np.testing.assert_allclose(res.rg, _LDSC_RG, rtol=1e-6)


def test_estimate_sample_overlap_requires_result_type():
    with pytest.raises(ValueError, match="LDSCRgResult"):
        estimate_sample_overlap(0.1, 60000.0, 40000.0)


def test_ldsc_rg_constrain_intercept_recovers_rg():
    # Fixing the cross-trait intercept at its true value (0: no sample
    # overlap) leaves the genetic-correlation estimate on target.
    k, nb, n1, n2 = 200, 60, 40000, 20000
    blocks, chols = _varied_blocks(nb, k, seed=5)
    m = nb * k
    idxs = [np.arange(b * k, (b + 1) * k) for b in range(nb)]
    ell = ld_scores(blocks)

    def gv(a, b):
        return sum(a[ix] @ (blocks[i][0].astype(float) @ b[ix])
                   for i, ix in enumerate(idxs))

    def sumstats(beta, n, rng):
        bh = np.empty(m)
        for i, ix in enumerate(idxs):
            bh[ix] = blocks[i][0].astype(float) @ beta[ix] + \
                (chols[i] @ rng.standard_normal(k)) / np.sqrt(n)
        return bh

    ests = []
    for rep in range(5):
        rng = np.random.default_rng(180 + rep)
        c = rng.random(m) < 0.05
        L = np.linalg.cholesky([[1, 0.6], [0.6, 1]])
        raw = L @ rng.standard_normal((2, c.sum()))
        b1 = np.zeros(m); b2 = np.zeros(m); b1[c] = raw[0]; b2[c] = raw[1]
        b1 *= np.sqrt(0.5 / gv(b1, b1)); b2 *= np.sqrt(0.5 / gv(b2, b2))
        res = ldsc_rg(sumstats(b1, n1, rng), sumstats(b2, n2, rng), ell,
                      n1, n2, n_blocks=60, constrain_intercept=0.0)
        assert res.gcov_intercept == 0.0
        ests.append(res.rg)
    assert abs(np.mean(ests) - 0.6) < 0.15, np.mean(ests)
