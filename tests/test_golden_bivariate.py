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
_BIVAR_BETA2 = np.array([
    9.18440875132913695e-05, -1.25642487517512119e-04, 3.12476925618041211e-02,
    -1.50669158736608140e-02, -4.87582115673005061e-03, 2.96888441841096460e-04,
    1.90612357018260133e-02, 5.01923279358683916e-02, -2.29932496838740723e-02,
    -4.20566729021603705e-02, -6.26576098905062199e-04, 6.25668350189269584e-02,
    -1.26660340430480300e-01, 5.36907636546743658e-02, -5.50249677920169322e-02,
    9.90437042971750443e-05, -8.65444991076825794e-04, -2.30433652022696148e-03,
    -2.15184420544339605e-04, 3.26256805437314806e-02,
])
_BIVAR_H2 = (0.04008305500773926, 0.022890700274269222)
_BIVAR_P = 0.6425691261548463


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
_BIVAR_BETA2_INT8 = np.array([
    1.02115180212825803e-04, -2.15675249986642197e-04, 3.26065818535915403e-02,
    -1.71594465410478195e-02, -4.28552685647558187e-03, 3.06730247384675203e-04,
    2.19670240197312357e-02, 4.68249600796658699e-02, -2.06410510093719760e-02,
    -4.27037674985630486e-02, -9.82513475312560324e-04, 6.31044299056693070e-02,
    -1.28363358225860258e-01, 5.50089385418068497e-02, -5.64463129787121098e-02,
    1.29402916155430090e-04, -8.57908853788176659e-04, -2.77631690475894010e-03,
    -1.94790235242917401e-04, 3.23914799079089780e-02,
])
_BIVAR_H2_INT8 = (0.0397042716216955, 0.023440858650269677)
_BIVAR_P_INT8 = 0.6474505667358872


def _assert_rg_matches_its_definition(res):
    """rg must equal the quadratic ratio it is defined as (algorithm.md Eq. 6).

    Freezing rg alone cannot catch a rescaled denominator: clamping h2 into
    h2_bounds and dividing the raw covariance by it left every golden value
    intact while saturating rg on any fit whose h2 hit a bound.
    """
    gvar1, gcov, gvar2 = res.genetic_samples.mean(axis=0)
    np.testing.assert_allclose(res.rg, gcov / np.sqrt(gvar1 * gvar2), rtol=1e-12)


def test_golden_bivariate():
    # Exact dense-float32 path (ld_int8=False): the reference math, frozen to the
    # pre-int8 goldens so any drift in the sampler itself is caught exactly.
    R, beta_hat, m = _fixtures()
    res = ldpred3_auto_bivariate(R, beta_hat, _beta_hat2(beta_hat, m), 10000,
                                 10000, burn_in=50, num_iter=150, seed=42,
                                 p_init=0.1, ld_int8=False)
    np.testing.assert_allclose(res.rg, _BIVAR_RG, rtol=1e-6)
    np.testing.assert_allclose(res.beta1_est, _BIVAR_BETA1, rtol=1e-6, atol=1e-9)
    # Trait 2 and the heritability scale are frozen too: with only rg and
    # beta1_est pinned, a scale error confined to trait 2 or to h2 passed.
    np.testing.assert_allclose(res.beta2_est, _BIVAR_BETA2, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(res.h2, _BIVAR_H2, rtol=1e-6)
    np.testing.assert_allclose(res.p, _BIVAR_P, rtol=1e-6)
    _assert_rg_matches_its_definition(res)


def test_golden_bivariate_int8():
    # Opt-in int8 path (ld_int8=True): its own frozen goldens, a drift detector
    # for the quantise + dequantise-in-loop machinery. Explicit since the
    # fit-time default became ld_int8=False.
    R, beta_hat, m = _fixtures()
    res = ldpred3_auto_bivariate(R, beta_hat, _beta_hat2(beta_hat, m), 10000,
                                 10000, burn_in=50, num_iter=150, seed=42,
                                 p_init=0.1, ld_int8=True)
    np.testing.assert_allclose(res.rg, _BIVAR_RG_INT8, rtol=1e-6)
    np.testing.assert_allclose(res.beta1_est, _BIVAR_BETA1_INT8, rtol=1e-6,
                               atol=1e-9)
    np.testing.assert_allclose(res.beta2_est, _BIVAR_BETA2_INT8, rtol=1e-6,
                               atol=1e-9)
    np.testing.assert_allclose(res.h2, _BIVAR_H2_INT8, rtol=1e-6)
    np.testing.assert_allclose(res.p, _BIVAR_P_INT8, rtol=1e-6)
    _assert_rg_matches_its_definition(res)
    # int8 stays close to the exact float fit (quantisation error is small).
    np.testing.assert_allclose(res.rg, _BIVAR_RG, atol=0.02)
    np.testing.assert_allclose(res.beta1_est, _BIVAR_BETA1, atol=0.01)
    np.testing.assert_allclose(res.beta2_est, _BIVAR_BETA2, atol=0.01)
    np.testing.assert_allclose(res.h2, _BIVAR_H2, atol=0.01)
