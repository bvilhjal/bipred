"""Cross-trait LD Score regression (genetic correlation): recovery, overlap,
and a bit-exact golden.

``ldsc_rg`` / ``estimate_sample_overlap`` moved to bipred with the rest of the
genetic-correlation machinery; univariate LD scores (``ld_scores``) still come
from ldpred3. The golden value pins the iterated cross-trait LDSC variance
weight, including the ``E[z1*z2] ** 2`` term used by reference LDSC.
"""

import numpy as np
import pytest

from bipred import ldsc_rg, LDSCRgResult, estimate_sample_overlap
from ldpred3 import ld_scores


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
    beta = np.sqrt((1.0 + 0.2 * x) / n)
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
    # With an unknown nonnegative phenotypic correlation, rho=1 gives the
    # overlap-only lower bound, not an upper bound.
    lower = estimate_sample_overlap(res, n1, n2)
    assert lower["n_shared"] == pytest.approx(rho_ph * 30000.0)
    # non-positive intercept -> clipped to no overlap; pheno_corr=0 is rejected.
    z = estimate_sample_overlap(LDSCRgResult(0, 0, 0, -0.01, (0.5, 0.5)), n1, n2)
    assert z["n_shared"] == 0.0
    with pytest.raises(ValueError):
        estimate_sample_overlap(res, n1, n2, pheno_corr=0.0)


@pytest.mark.parametrize("name,bad", [
    ("beta_hat1", np.empty(0)),
    ("beta_hat1", np.ones((4, 1))),
    ("beta_hat2", np.ones(1)),
    ("ld_scores", np.ones(3)),
    ("beta_hat1", np.array([1.0, 2.0, np.inf, 4.0])),
    ("beta_hat2", np.array([1.0, 2.0, np.nan, 4.0])),
    ("ld_scores", np.array([1.0, 2.0, np.nan, 4.0])),
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


def test_ldsc_rg_nonpositive_h2_is_undefined():
    ell = np.array([1.0, 2.0, 3.0, 4.0])
    beta = np.sqrt(np.array([4.0, 3.0, 2.0, 1.0]) / 100.0)
    res = ldsc_rg(beta, beta, ell, 100.0, 100.0, n_blocks=2, n_iter=0)
    assert res.h2[0] < 0.0 and res.h2[1] < 0.0
    assert np.isnan(res.rg)
    assert np.isnan(res.rg_se)


def test_ldsc_rg_invalid_jackknife_replicate_makes_se_undefined():
    ell = np.array([1.0, 2.0, 3.0, 4.0])
    beta = np.sqrt(np.array([4.0, 3.0, 2.0, 20.0]) / 100.0)
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
_LDSC_RG = 1.1208147743678865


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
