"""Cross-trait LD Score regression (genetic correlation): recovery, overlap,
and a bit-exact golden.

``ldsc_rg`` / ``estimate_sample_overlap`` moved to bipred with the rest of the
genetic-correlation machinery; univariate LD scores (``ld_scores``) still come
from ldpred3. The golden value is bit-identical to the one previously frozen in
ldpred3's ``test_golden_estimators.py``.
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
    # non-positive intercept -> clipped to no overlap; pheno_corr=0 is rejected.
    z = estimate_sample_overlap(LDSCRgResult(0, 0, 0, -0.01, (0.5, 0.5)), n1, n2)
    assert z["n_shared"] == 0.0
    with pytest.raises(ValueError):
        estimate_sample_overlap(res, n1, n2, pheno_corr=0.0)


# --- bit-exact golden (fully fixed input; matches the value frozen in ldpred3) ---
_LDSC_RG = 1.1049275624398285


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
