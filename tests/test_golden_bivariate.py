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


def test_golden_bivariate():
    R, beta_hat, m = _fixtures()
    res = ldpred3_auto_bivariate(R, beta_hat, _beta_hat2(beta_hat, m), 10000,
                                 10000, burn_in=50, num_iter=150, seed=42)
    np.testing.assert_allclose(res.rg, _BIVAR_RG, rtol=1e-6)
    np.testing.assert_allclose(res.beta1_est, _BIVAR_BETA1, rtol=1e-6, atol=1e-9)
