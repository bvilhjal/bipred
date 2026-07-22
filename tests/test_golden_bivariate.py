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


# --- frozen outputs (captured from coherent-initialization code; see docstring) ---
_BIVAR_RG = 0.9198872305774631
_BIVAR_BETA1 = np.array([
    1.57189097964960945e-04, -4.28077816352409683e-03, 3.80018899847359216e-02,
    6.20080476301159430e-04, -4.37157272401155910e-02, 2.13502259491582478e-04,
    6.04232920672766787e-02, 5.06057436216132421e-02, -3.92474140338508501e-02,
    -6.08150556365017754e-02, -7.17828492204075908e-04, 6.84831219183628408e-02,
    -1.55927514443277019e-01, 7.02096540396927270e-02, -6.83101144578948078e-02,
    -7.99270116102478669e-04, -3.04057175433961442e-03, -4.10055068429184974e-03,
    1.84725979127333327e-04, 5.32542022673826696e-02,
])


# --- int8 goldens: the automatic policy quantises this small R to int8, so the
# outputs differ from the exact-float path by the tiny quantisation error
# (~0.003 here) and get their own frozen values. Captured from the same known-good
# code and likewise bit-identical between the Numba and pure-Python paths. ---
_BIVAR_RG_INT8 = 0.914588545485982
_BIVAR_BETA1_INT8 = np.array([
    1.00715912615757303e-04, -3.96884812733992103e-03, 3.81784298769406531e-02,
    1.00972625655868887e-03, -4.43440277260852350e-02, 1.82360316614327191e-04,
    6.19566604288621032e-02, 4.78451454975686178e-02, -3.84271936653311151e-02,
    -6.11066443832446714e-02, -1.00261733912600518e-03, 6.84254249994016031e-02,
    -1.55871072601938966e-01, 6.97465970888896021e-02, -6.86066680166198746e-02,
    -5.47859240316516112e-04, -3.04406402266507066e-03, -3.95344960911508188e-03,
    3.24734576740398688e-04, 5.28174127155385009e-02,
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
    # Automatic small-block int8 path: its own frozen goldens (drift detector
    # for the quantise + dequantise-in-loop machinery).
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
