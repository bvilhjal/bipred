"""Golden (characterization) test for bivariate LDpred-auto.

Freezes the exact bivariate outputs on a fully fixed input (the same AR(1) block
and seeded ``beta_hat`` used by ldpred3's ``test_golden.py``), so a silent 3-5%
math drift (a dropped term, a wrong scale, an off-by-one) fails immediately
instead of hiding under a ~0.1-of-truth statistical tolerance.

The inputs are deterministic, so the outputs are too. The frozen values were
captured from known-good code and are **bit-identical** between the Numba and
pure-Python paths — so the same goldens hold on both CI legs.
"""

import numpy as np

from bipred import ldpred3_auto_bivariate


def _fixtures():
    rng = np.random.default_rng(0)
    m = 20
    R = (0.4 ** np.abs(np.subtract.outer(np.arange(m), np.arange(m)))).astype(
        np.float64)
    beta_hat = rng.standard_normal(m) * 0.05
    return R, beta_hat, m


def _beta_hat2(beta_hat, m):
    """A second, correlated trait for the rg / bivariate goldens."""
    rng2 = np.random.default_rng(1)
    return 0.7 * beta_hat + 0.3 * rng2.standard_normal(m) * 0.05


# --- frozen outputs (captured from known-good code; see module docstring) ---
_BIVAR_RG = 0.9188617923882318
_BIVAR_BETA1 = np.array([
    2.24980429356143091e-04, -4.71318734363154659e-03, 3.82386860491423647e-02,
    1.20993362589549760e-03, -4.40882017432814993e-02, 3.21806600089039063e-04,
    6.02421384388605596e-02, 5.14601663474335766e-02, -4.14631419685409092e-02,
    -6.00695234537987968e-02, -1.25160958116114340e-03, 6.80057596524489966e-02,
    -1.55258265176938082e-01, 6.88338653808804740e-02, -6.69930483977646957e-02,
    -9.18437856678719963e-04, -3.43253457709516039e-03, -5.19305928056782926e-03,
    2.44874320398874221e-04, 5.37048461086601508e-02,
])


# --- int8 goldens: the default LD representation quantises R to int8, so the
# outputs differ from the exact-float path by the tiny quantisation error
# (~0.003 here) and get their own frozen values. Captured from the same known-good
# code and likewise bit-identical between the Numba and pure-Python paths. ---
_BIVAR_RG_INT8 = 0.9154684936365023
_BIVAR_BETA1_INT8 = np.array([
    1.38541208831286400e-04, -5.06160560064855012e-03, 3.81781609778040066e-02,
    1.43713713680043638e-03, -4.44134263101568412e-02, 1.98906980801838095e-04,
    6.26889932693942266e-02, 4.80481567695626410e-02, -3.76714678348756369e-02,
    -6.03920892468849874e-02, -1.33960008900301767e-03, 6.83385105299416059e-02,
    -1.55845155473349725e-01, 6.87212146357265677e-02, -6.83453198381011314e-02,
    -6.80252553139864800e-04, -3.63073348093846004e-03, -5.27057238631533789e-03,
    3.84290947948995106e-04, 5.29414794508034616e-02,
])


def test_golden_bivariate():
    # Exact dense-float32 path (ld_int8=False): the reference math, frozen to the
    # pre-int8 goldens so any drift in the sampler itself is caught exactly.
    R, beta_hat, m = _fixtures()
    res = ldpred3_auto_bivariate(R, beta_hat, _beta_hat2(beta_hat, m), 10000,
                                 10000, burn_in=50, num_iter=150, seed=42,
                                 p_init=0.1, ld_int8=False)
    np.testing.assert_allclose(res.rg, _BIVAR_RG, rtol=1e-6)
    np.testing.assert_allclose(res.beta1_est, _BIVAR_BETA1, rtol=1e-6, atol=1e-9)


def test_golden_bivariate_int8():
    # Default int8-quantised LD path: its own frozen goldens (drift detector for
    # the quantise + dequantise-in-loop machinery).
    R, beta_hat, m = _fixtures()
    res = ldpred3_auto_bivariate(R, beta_hat, _beta_hat2(beta_hat, m), 10000,
                                 10000, burn_in=50, num_iter=150, seed=42,
                                 p_init=0.1)
    np.testing.assert_allclose(res.rg, _BIVAR_RG_INT8, rtol=1e-6)
    np.testing.assert_allclose(res.beta1_est, _BIVAR_BETA1_INT8, rtol=1e-6,
                               atol=1e-9)
    # int8 stays close to the exact float fit (quantisation error is small).
    np.testing.assert_allclose(res.rg, _BIVAR_RG, atol=0.02)
    np.testing.assert_allclose(res.beta1_est, _BIVAR_BETA1, atol=0.01)
