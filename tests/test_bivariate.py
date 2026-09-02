"""Bivariate LDpred-auto: rg / h2 recovery and cross-trait borrowing."""

import warnings

import numpy as np
import pytest

import bipred.bivariate as bivariate
from bipred import ldpred3_auto_bivariate, ldpred3_auto_bivariate_blocks
from ldpred3 import ldpred3_by_blocks, ldpred3_auto_infer


def _ar1_chol(rho, k):
    """Exact Cholesky factor of the AR(1) correlation ``rho**|i-j|``."""
    L = np.zeros((k, k))
    L[:, 0] = rho ** np.arange(k)
    scale = np.sqrt(1.0 - rho * rho)
    for j in range(1, k):
        L[j:, j] = scale * rho ** np.arange(k - j)
    return L


def _blocks(n_blocks=12, k=200, seed=0):
    rng = np.random.default_rng(seed)
    blocks, chols, idxs = [], [], []
    for b in range(n_blocks):
        rho = rng.uniform(0.0, 0.8)
        d = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
        R = (rho ** d).astype(np.float64)
        blocks.append((R.astype(np.float32), np.arange(b * k, (b + 1) * k)))
        chols.append(_ar1_chol(rho, k))
        idxs.append(np.arange(b * k, (b + 1) * k))
    return blocks, chols, idxs


def _gv(blocks, idxs, a, b):
    return sum(a[ix] @ (blocks[i][0].astype(float) @ b[ix])
               for i, ix in enumerate(idxs))


def _sim(blocks, chols, idxs, m, *, p, h2, rg, rng):
    """Shared-causal bivariate effects scaled to (h2[0], h2[1]) with corr rg."""
    causal = rng.random(m) < p
    nc = causal.sum()
    L = np.array([[1.0, 0.0], [rg, np.sqrt(1.0 - rg * rg)]])
    raw = (L @ rng.standard_normal((2, nc)))
    b1 = np.zeros(m); b2 = np.zeros(m)
    b1[causal] = raw[0]; b2[causal] = raw[1]
    b1 *= np.sqrt(h2[0] / _gv(blocks, idxs, b1, b1))
    b2 *= np.sqrt(h2[1] / _gv(blocks, idxs, b2, b2))
    return b1, b2


def _sumstats(blocks, chols, idxs, beta, n, k, rng):
    bhat = np.empty(beta.shape[0])
    for i, ix in enumerate(idxs):
        bhat[ix] = blocks[i][0].astype(float) @ beta[ix] + \
            (chols[i] @ rng.standard_normal(k)) / np.sqrt(n)
    return bhat


def _genetic_r2(b_est, beta, blocks, idxs):
    num = _gv(blocks, idxs, b_est, beta)
    den = _gv(blocks, idxs, b_est, b_est) * _gv(blocks, idxs, beta, beta)
    return float(num * num / den) if den > 0 else 0.0


def test_recovers_rg_and_h2():
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=1)
    m = nb * k
    rgs, h1s, h2s = [], [], []
    for rep in range(3):
        rng = np.random.default_rng(10 + rep)
        b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.7, rng=rng)
        bh1 = _sumstats(blocks, chols, idxs, b1, 40000, k, rng)
        bh2 = _sumstats(blocks, chols, idxs, b2, 40000, k, rng)
        res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 40000, 40000,
                                            burn_in=120, num_iter=150, seed=rep)
        rgs.append(res.rg); h1s.append(res.h2[0]); h2s.append(res.h2[1])
    assert abs(np.mean(rgs) - 0.7) < 0.2, np.mean(rgs)
    assert abs(np.mean(h1s) - 0.5) < 0.12
    assert abs(np.mean(h2s) - 0.5) < 0.12


def _lr8_blocks(k=200, nb=3, rank=64, rho=0.6):
    """int8 low-rank blocks whose rank clears the widening gate."""
    from ldpred3 import lowrank_ld

    pos = np.arange(k)
    dense = (rho ** np.abs(np.subtract.outer(pos, pos))).astype(np.float64)
    factor = lowrank_ld(dense, variance=0.999, max_rank=rank, quantize=True)
    blocks = [(factor, np.arange(b * k, (b + 1) * k)) for b in range(nb)]
    return blocks, factor, k * nb


@pytest.mark.skipif(not bivariate.HAVE_NUMBA, reason="widening is Numba-only")
def test_lr8_widening_fires_and_preserves_the_fit():
    """The int8 low-rank factor is widened once per sweep, above the gate.

    Every other low-rank test in this suite uses rank <= 4, far below
    ``_LR8_DEQUANTISE_MIN_RANK``, so without this the widening branch would
    ship untested.
    """
    blocks, factor, m = _lr8_blocks(rank=64)
    rank = factor.U.shape[1]
    assert factor.U.dtype == np.int8
    assert rank >= bivariate._LR8_DEQUANTISE_MIN_RANK, rank

    # The gate is reached, and the scratch is sized per thread, not per block.
    payloads = [factor.U] * len(blocks)
    sizes = np.array([len(idx) for _, idx in blocks])
    scratch, stride, min_rank = bivariate._lr8_dequant_scratch(payloads, sizes, 1)
    assert min_rank == bivariate._LR8_DEQUANTISE_MIN_RANK
    assert stride == sizes[0] * rank
    assert scratch.size == stride and scratch.dtype == np.float32

    rng = np.random.default_rng(0)
    bh1 = rng.normal(scale=0.01, size=m)
    bh2 = 0.6 * bh1 + 0.8 * rng.normal(scale=0.01, size=m)
    kw = dict(burn_in=8, num_iter=12, seed=3)

    widened = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 50_000, 50_000, **kw)

    # Widening is an exact int8 -> float32 conversion, so the fit may move only
    # at the level fastmath reassociates the reduction -- not at the int8
    # quantisation resolution, which would mean the factor itself had changed.
    monkey = bivariate._LR8_DEQUANTISE_MIN_RANK
    try:
        bivariate._LR8_DEQUANTISE_MIN_RANK = 10 ** 9      # disables the branch
        plain = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 50_000, 50_000,
                                              **kw)
    finally:
        bivariate._LR8_DEQUANTISE_MIN_RANK = monkey
    scale = max(float(np.max(np.abs(plain.beta1_est))), 1e-300)
    assert abs(widened.rg - plain.rg) < 1e-12
    assert np.max(np.abs(widened.beta1_est - plain.beta1_est)) / scale < 1e-10
    np.testing.assert_allclose(widened.h2, plain.h2, rtol=1e-12)


@pytest.mark.skipif(not bivariate.HAVE_NUMBA, reason="widening is Numba-only")
def test_lr8_widening_keeps_ncores_results_identical():
    """Both drivers widen, so the seeded ncores contract still holds above the
    gate. A serial-only widening would reassociate one path and not the other.
    """
    blocks, _factor, m = _lr8_blocks(rank=64)
    rng = np.random.default_rng(1)
    bh1 = rng.normal(scale=0.01, size=m)
    bh2 = 0.6 * bh1 + 0.8 * rng.normal(scale=0.01, size=m)
    kw = dict(burn_in=6, num_iter=10, seed=5)

    one = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 50_000, 50_000,
                                        ncores=1, **kw)
    four = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 50_000, 50_000,
                                         ncores=4, **kw)
    assert one.rg == four.rg
    np.testing.assert_array_equal(one.beta1_est, four.beta1_est)
    np.testing.assert_array_equal(one.beta2_est, four.beta2_est)


def test_lr8_widening_gate_ignores_float_and_low_rank_factors():
    """The branch stays off where it cannot pay: float factors and small ranks."""
    from ldpred3 import lowrank_ld

    pos = np.arange(120)
    dense = (0.6 ** np.abs(np.subtract.outer(pos, pos))).astype(np.float64)
    sizes = np.array([120])

    f32 = lowrank_ld(dense, variance=0.999, max_rank=64, quantize=False)
    _, _, min_rank = bivariate._lr8_dequant_scratch([f32.U], sizes, 1)
    assert min_rank == 0, "float32 factors are already the fast specialisation"

    small = lowrank_ld(dense, variance=0.5, max_rank=4, quantize=True)
    scratch, stride, min_rank = bivariate._lr8_dequant_scratch([small.U], sizes, 1)
    assert (min_rank, stride, scratch.size) == (0, 0, 0)


def test_vectorised_rg_matches_the_scalar_elementwise():
    """The split-Rhat trace and the reported rg must use one convention.

    Including the degenerate cases: a non-positive variance reports 0.0, and a
    NaN variance stays NaN rather than being laundered into an rg of 0.
    """
    cases = [
        (0.1, 0.3, 0.4), (0.1, -0.001, 0.3), (0.5, 0.1, 0.1), (0.0, 0.0, 0.0),
        (-0.5, 0.1, 0.1), (0.1, 0.0, 0.4), (0.1, 0.3, -1.0),
        (np.nan, 0.3, 0.4), (0.1, np.nan, 0.4), (0.1, 0.3, np.nan),
        (np.inf, 0.3, 0.4), (0.1, np.inf, 0.4),
    ]
    g12, g1, g2 = (np.array([c[i] for c in cases]) for i in range(3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # no sqrt/divide RuntimeWarnings
        vector = bivariate._rg_from_quadratics_array(g12, g1, g2)
    expected = np.array([bivariate._rg_from_quadratics(*c) for c in cases])
    np.testing.assert_array_equal(vector, expected)      # NaN-aware


def test_option_defaults_match_the_public_signature():
    """The multi-chain path reads defaults from a dict, not the signature.

    ``_BIVARIATE_OPTION_DEFAULTS`` duplicates the defaults of
    ``ldpred3_auto_bivariate_blocks``. Without this gate, changing one default
    silently gives the single-chain and multi-chain entry points different
    behaviour.
    """
    import inspect

    signature = inspect.signature(ldpred3_auto_bivariate_blocks)
    for name, default in bivariate._BIVARIATE_OPTION_DEFAULTS.items():
        assert name in signature.parameters, f"{name} is not a public option"
        assert signature.parameters[name].default == default, name


def test_rg_is_invariant_to_h2_bounds():
    """rg is a ratio of the raw quadratics, not of the clamped h2.

    Dividing the raw genetic covariance by the h2_bounds-clamped variances
    made a binding bound drive rg toward +/-1: on this fixture a true rg of
    ~0.7 was reported as 1.0 once h2_bounds capped h2 below its fitted value.
    h2 itself is still clamped for reporting.
    """
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=1)
    m = nb * k
    rng = np.random.default_rng(10)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.7, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 40000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 40000, k, rng)

    def fit(h2_bounds):
        return ldpred3_auto_bivariate_blocks(
            blocks, bh1, bh2, 40000, 40000, burn_in=120, num_iter=150,
            seed=0, h2_bounds=h2_bounds, h2_init=0.1)

    loose = fit((1e-4, 1.0))
    # The tight ceiling binds, which the diagnostic now reports from the raw
    # quadratics rather than from the already-clamped h2.
    with pytest.warns(RuntimeWarning, match="h2 reached its upper bound"):
        tight = fit((1e-4, 0.1))

    # The clamp binds: reported h2 is pinned to the ceiling ...
    assert tight.h2 == (0.1, 0.1)
    assert loose.h2[0] > 0.1 and loose.h2[1] > 0.1
    # ... but the sampled quadratics, and therefore rg, are untouched.
    np.testing.assert_allclose(tight.genetic_samples, loose.genetic_samples)
    assert tight.rg == pytest.approx(loose.rg)
    assert abs(tight.rg) < 0.99, "rg saturated against the h2 bound"


def test_nonfinite_fit_raises_instead_of_returning_nan(monkeypatch):
    """A diverged chain must fail loudly, not return NaN estimates.

    NaN does not announce itself in the sweep: the log-sum-exp leaves
    ``wmax = w0``, every state probability becomes NaN, all three ``u < p``
    tests are False, and the variant falls through to the both-causal branch.
    """
    from bipred.bivariate import _check_fit_is_finite

    finite = np.zeros(3)
    _check_fit_is_finite((0.3, 0.1, 0.4), finite, finite)      # no raise

    with pytest.raises(FloatingPointError, match="non-finite genetic quadratic"):
        _check_fit_is_finite((np.nan, 0.1, 0.4), finite, finite)
    with pytest.raises(FloatingPointError, match="non-finite genetic quadratic"):
        _check_fit_is_finite((np.inf, 0.1, 0.4), finite, finite)
    with pytest.raises(FloatingPointError, match="non-finite posterior-mean"):
        _check_fit_is_finite((0.3, 0.1, 0.4), np.array([np.nan]), finite)
    with pytest.raises(FloatingPointError, match="non-finite posterior-mean"):
        _check_fit_is_finite((0.3, 0.1, 0.4), finite, np.array([np.inf]))

    # And the driver actually consults it, rather than clamping NaN away.
    blocks, chols, idxs = _blocks(1, 40, seed=0)
    rng = np.random.default_rng(0)
    m = 40
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.2, h2=(0.3, 0.3), rg=0.5, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 10000, 40, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 10000, 40, rng)
    real = bivariate._check_fit_is_finite

    def poisoned(quadratics, beta1, beta2):
        return real((np.nan,) * 3, beta1, beta2)

    monkeypatch.setattr(bivariate, "_check_fit_is_finite", poisoned)
    with pytest.raises(FloatingPointError):
        ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 10000, 10000,
                                      burn_in=5, num_iter=5, seed=0)


def test_implausible_fit_warning_is_two_sided():
    """A floored h2 is as suspect as an inflated one, and used to be silent."""
    bounds = (1e-4, 1.0)
    m = bivariate._DIAGNOSTIC_MIN_VARIANTS

    with pytest.warns(RuntimeWarning, match="upper bound"):
        bivariate._warn_if_implausible_fit((1.0, 0.3), 0.01, bounds, m)
    # A non-positive quadratic is degenerate and does zero rg ...
    with pytest.warns(RuntimeWarning, match="non-positive.*reports rg as 0"):
        bivariate._warn_if_implausible_fit((-1e-9, 0.3), 0.01, bounds, m)
    # ... but a small strictly-positive one only means h2 was clamped. Saying
    # "non-positive" there would be a false statement about a healthy
    # low-heritability fit, and rg is computed from the raw quadratics.
    with pytest.warns(RuntimeWarning, match="lower bound.*rg is unaffected"):
        bivariate._warn_if_implausible_fit((5e-5, 0.3), 0.01, (1e-4, 1.0), m)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bivariate._warn_if_implausible_fit((5e-5, 0.3), 0.01, (1e-4, 1.0), m)
    assert "non-positive" not in str(caught[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # ordinary fit: silent
        bivariate._warn_if_implausible_fit((0.3, 0.4), 0.01, bounds, m)
        # and still quiet below the panel-size floor
        bivariate._warn_if_implausible_fit((1e-4, 0.3), 0.01, bounds, m - 1)


def test_rg_zero_is_recovered():
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=4)
    m = nb * k
    rng = np.random.default_rng(0)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.0, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 40000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 40000, k, rng)
    res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 40000, 40000,
                                        burn_in=120, num_iter=150, seed=1)
    assert abs(res.rg) < 0.25, res.rg


def test_int8_ld_matches_float_and_accepts_prequantized():
    # Quantising in the fit (ld_int8=True) tracks the exact float32 fit closely.
    # A block handed in already int8 is consumed as-is -- bit-identical to
    # quantising the float block on the fly, and without the extra payload.
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=2)
    m = nb * k
    rng = np.random.default_rng(3)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.6, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 60000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 60000, k, rng)
    kw = dict(burn_in=120, num_iter=150, seed=1)

    flt = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 60000, 60000,
                                        ld_int8=False, **kw)
    q8 = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 60000, 60000,
                                       ld_int8=True, **kw)
    # Opt-in int8 stays close to the exact float fit.
    assert abs(q8.rg - flt.rg) < 0.05, (q8.rg, flt.rg)
    assert abs(q8.h2[0] - flt.h2[0]) < 0.05 and abs(q8.h2[1] - flt.h2[1]) < 0.05
    assert np.max(np.abs(q8.beta1_est - flt.beta1_est)) < 0.02

    # pre-quantised int8 blocks (what ldpred3.compute_ld_blocks(quantize=True)
    # emits, and the recommended way to get int8) are detected by dtype and
    # consumed as-is, so the fit is bit-identical to quantising on the fly --
    # under the ld_int8=False default, which copies nothing.
    pre = [(np.rint(np.clip(R, -1.0, 1.0) * 127.0).astype(np.int8), ix)
           for (R, ix) in blocks]
    q8_pre = ldpred3_auto_bivariate_blocks(pre, bh1, bh2, 60000, 60000,
                                           ld_int8=False, **kw)
    assert q8_pre.rg == q8.rg
    assert np.array_equal(q8_pre.beta1_est, q8.beta1_est)


def test_dense_ld_auto_storage_policy_and_explicit_overrides(monkeypatch):
    import bipred.bivariate as bivariate

    assert bivariate._AUTO_INT8_MAX_BLOCK == 1500
    monkeypatch.setattr(bivariate, "_AUTO_INT8_MAX_BLOCK", 2)
    small = np.eye(2, dtype=np.float32)
    large = np.eye(3, dtype=np.float32)

    auto_small, small_scale = bivariate._prepare_block(small, None)
    auto_large, large_scale = bivariate._prepare_block(large, None)
    forced_int8, forced_int8_scale = bivariate._prepare_block(large, True)
    forced_float, forced_float_scale = bivariate._prepare_block(small, False)

    assert auto_small.dtype == np.int8 and small_scale == 1.0 / 127.0
    assert auto_large.dtype == np.float32 and large_scale == 1.0
    assert forced_int8.dtype == np.int8 and forced_int8_scale == 1.0 / 127.0
    assert forced_float.dtype == np.float32 and forced_float_scale == 1.0


def test_mixer_overlap_params():
    # The 4-state result exposes MiXeR-style overlap params: pi sums to 1, the
    # mixer summary has the expected keys, and the rg decomposition
    # (rho_beta * pi11/sqrt(pi1 pi2)) is consistent with the reported rg.
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=2)
    m = nb * k
    rng = np.random.default_rng(3)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.6, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 60000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 60000, k, rng)
    res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 60000, 60000,
                                        burn_in=150, num_iter=200, seed=1)
    assert res.pi is not None and abs(res.pi.sum() - 1.0) < 1e-6
    mx = res.mixer
    assert set(mx) == {"polygenicity", "n_causal", "n_shared", "frac_shared",
                       "rho_beta", "rg_from_overlap"}
    assert 0.0 <= mx["frac_shared"] <= 1.0
    assert -1.0 <= mx["rho_beta"] <= 1.0
    # the overlap-decomposition rg matches the reported rg to within MC noise
    assert abs(mx["rg_from_overlap"] - res.rg) < 0.15, (mx["rg_from_overlap"], res.rg)


def test_pi_prior_default_and_validation():
    # Default pi_prior reproduces the historical Dirichlet(1,1,1,1) sampler
    # bit-for-bit; the Jeffreys concentration still yields a valid mixture and
    # leaves rg essentially unchanged; improper concentrations are rejected.
    import pytest
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=2)
    m = nb * k
    rng = np.random.default_rng(3)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.6, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 60000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 60000, k, rng)
    kw = dict(burn_in=120, num_iter=180, seed=1)
    default = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 60000, 60000, **kw)
    uni = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 60000, 60000,
                                        pi_prior=1.0, **kw)
    jef = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 60000, 60000,
                                        pi_prior=0.5, **kw)
    assert np.allclose(default.pi, uni.pi)
    assert abs(jef.pi.sum() - 1.0) < 1e-6
    assert abs(jef.rg - uni.rg) < 0.1
    with pytest.raises(ValueError, match="pi_prior"):
        ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 60000, 60000,
                                      pi_prior=0.0, **kw)


def test_initial_hyperparameters_match_documented_genetic_moments():
    """The four-state start must encode h2_init and rg_init, not fractions of them."""
    m = 100
    pi, s1, s2, s12 = bivariate._initial_hyperparameters(
        m, (0.4, 0.2), 0.3, 0.4,
    )
    p1, p2, shared = pi[1] + pi[3], pi[2] + pi[3], pi[3]
    h1, h2 = m * p1 * s1, m * p2 * s2
    rg = m * shared * s12 / np.sqrt(h1 * h2)
    np.testing.assert_allclose((h1, h2, rg), (0.4, 0.2, 0.4), rtol=1e-12)
    np.testing.assert_allclose(pi, (0.7, 0.1, 0.1, 0.1), rtol=1e-12)

    # A large genetic correlation needs more initial shared mass than the equal
    # non-null split; the helper increases it while preserving the union p.
    high, hs1, hs2, hs12 = bivariate._initial_hyperparameters(
        m, 0.3, 0.3, 0.9,
    )
    hp = high[1] + high[3]
    hrg = m * high[3] * hs12 / np.sqrt(
        (m * hp * hs1) * (m * hp * hs2)
    )
    assert high[3] > 0.1
    assert abs(hs12 / np.sqrt(hs1 * hs2)) <= bivariate._INIT_RHO_MAX
    np.testing.assert_allclose(hrg, 0.9, rtol=1e-12)


def test_explicit_pi_init_controls_overlap_and_validates_rg_feasibility():
    m = 200
    pi0 = np.array([0.78, 0.02, 0.12, 0.08])  # p1=.10, p2=.20
    pi, s1, s2, s12 = bivariate._initial_hyperparameters(
        m, (0.5, 0.25), 0.02, 0.3, pi_init=pi0,
    )
    p1, p2, shared = pi[1] + pi[3], pi[2] + pi[3], pi[3]
    h1, h2 = m * p1 * s1, m * p2 * s2
    rg = m * shared * s12 / np.sqrt(h1 * h2)
    np.testing.assert_allclose((h1, h2, rg), (0.5, 0.25, 0.3), rtol=1e-12)

    # Float32 simplex rounding is accepted and normalised. Explicit pi_init
    # also makes the scalar p_init shorthand irrelevant at the public boundary.
    pi32 = np.array(
        [0.37767145, 0.10645247, 0.46477157, 0.05110449],
        dtype=np.float32,
    )
    normalized, *_ = bivariate._initial_hyperparameters(
        m, 0.2, 0.0, 0.0, pi_init=pi32,
    )
    np.testing.assert_allclose(normalized.sum(), 1.0, rtol=0.0, atol=1e-15)
    public = ldpred3_auto_bivariate(
        np.eye(3), np.zeros(3), np.zeros(3), 1000, 1000,
        h2_init=0.1, p_init=0.0, pi_init=(0.7, 0.1, 0.1, 0.1),
        burn_in=0, num_iter=1, seed=0,
    )
    assert np.isfinite(public.rg)

    with pytest.raises(ValueError, match="cannot represent rg_init"):
        bivariate._initial_hyperparameters(
            m, (0.5, 0.25), 0.02, 0.9, pi_init=pi0,
        )
    for bad in ([0.8, 0.1, 0.1], [0.8, 0.1, 0.1, 0.1],
                [0.8, -0.1, 0.2, 0.1], [1.0, 0.0, 0.0, 0.0]):
        with pytest.raises(ValueError, match="pi_init"):
            bivariate._initial_hyperparameters(
                m, 0.2, 0.02, 0.0, pi_init=bad,
            )


def test_mixer_calibrated_uses_univariate_polygenicity():
    # mixer_calibrated keeps the joint fit's reliable ratios (frac_shared,
    # rho_beta) but replaces per-trait polygenicity with two univariate runs'
    # learned p, rebuilding the absolute counts on that scale.
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=2)
    m = nb * k
    rng = np.random.default_rng(3)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.6, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 60000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 60000, k, rng)
    res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 60000, 60000,
                                        burn_in=150, num_iter=200, seed=1)
    n = np.full(m, 60000.0)
    i1 = ldpred3_auto_infer(blocks, bh1, n, n_chains=4, burn_in=120,
                            num_iter=150, seed=1)
    i2 = ldpred3_auto_infer(blocks, bh2, n, n_chains=4, burn_in=120,
                            num_iter=150, seed=1)
    mj, mc = res.mixer, res.mixer_calibrated(i1, i2)
    # ratios are taken from the joint fit unchanged
    assert abs(mc["frac_shared"] - mj["frac_shared"]) < 1e-9
    assert abs(mc["rho_beta"] - mj["rho_beta"]) < 1e-9
    # polygenicity is exactly the univariate learned p; counts follow
    assert abs(mc["polygenicity"][0] - i1.p_est) < 1e-9
    assert abs(mc["polygenicity"][1] - i2.p_est) < 1e-9
    assert abs(mc["n_causal"][0] - i1.p_est * m) < 1e-6
    assert abs(mc["n_shared"] - mc["frac_shared"] * min(i1.p_est, i2.p_est) * m) < 1e-6
    # floats are accepted in place of InferResult objects
    mf = res.mixer_calibrated(0.1, 0.1)
    assert abs(mf["n_causal"][0] - 0.1 * m) < 1e-6
    for bad in (-0.1, 1.1, np.nan, True):
        with pytest.raises(ValueError, match="polygenic"):
            res.mixer_calibrated(bad, 0.1)


def test_mixer_iterate_intervals_and_point_summaries():
    # pi and Sigma points both summarize the retained hybrid iterates. The
    # accurately named API reports empirical central iterate intervals; the old
    # posterior/CI spelling remains a warning-emitting compatibility alias.
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=2)
    m = nb * k
    rng = np.random.default_rng(3)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.6, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 60000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 60000, k, rng)
    res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 60000, 60000,
                                        burn_in=150, num_iter=200, seed=1)
    assert res.pi_samples is not None and res.pi_samples.shape == (200, 4)
    assert res.sigma_samples.shape == (200, 3)
    assert np.allclose(res.pi, res.pi_samples.mean(axis=0))
    s1, s2, s12 = res.sigma_samples.mean(axis=0)
    assert np.allclose(res.sigma, [[s1, s12], [s12, s2]])

    post = res.mixer_iterate_summary(level=0.95)
    assert set(post) == {"n_causal", "polygenicity", "n_shared", "frac_shared",
                         "rho_beta", "rg_from_overlap", "level"}
    point = res.mixer
    for i in (0, 1):
        entry = post["n_causal"][i]
        lo, hi = entry["interval"]
        assert lo <= entry["mean"] <= hi                 # interval brackets mean
        assert lo <= point["n_causal"][i] <= hi                # and the point est
    for key in ("n_shared", "frac_shared", "rho_beta", "rg_from_overlap"):
        lo, hi = post[key]["interval"]
        assert lo <= post[key]["mean"] <= hi
        assert post[key]["sd"] >= 0.0
    # frac_shared is a probability in [0, 1]
    lo, hi = post["frac_shared"]["interval"]
    assert 0.0 <= lo <= hi <= 1.0

    with pytest.raises(ValueError, match="level"):
        res.mixer_iterate_summary(level=1.0)


def test_noise_inflation_calibrates_counts_under_mismatch():
    # The learned noise-inflation lambda is ~1 (a no-op) when the fit LD matches
    # the GWAS sample, but rises under a finite-reference-panel LD and deflates the
    # mismatch-inflated causal count back toward the truth, leaving h2/rg intact.
    k, nb = 200, 10
    m = nb * k
    n_causal = int(0.05 * m)
    rng = np.random.default_rng(0)
    # population LD (AR1 per block) + a finite reference-panel estimate (mismatch)
    pop, chol, ref = [], [], []
    for b in range(nb):
        rho = rng.uniform(0.3, 0.85)
        R = (rho ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))).astype(float)
        pop.append(R); chol.append(_ar1_chol(rho, k))
        Z = rng.standard_normal((2000, k)) @ chol[b].T
        Z = (Z - Z.mean(0)) / Z.std(0)
        Rr = 0.95 * (Z.T @ Z) / 2000 + 0.05 * np.eye(k)
        ref.append((Rr.astype(np.float32), np.arange(b * k, (b + 1) * k)))
    idx = [np.arange(b * k, (b + 1) * k) for b in range(nb)]

    def gv(a, bb):
        return sum(a[ix] @ (pop[i] @ bb[ix]) for i, ix in enumerate(idx))

    causal = rng.choice(m, 2 * n_causal, replace=False)
    b1 = np.zeros(m); b2 = np.zeros(m)
    b1[causal[:n_causal]] = rng.standard_normal(n_causal)
    b2[causal[n_causal:]] = rng.standard_normal(n_causal)
    b1 *= np.sqrt(0.5 / gv(b1, b1)); b2 *= np.sqrt(0.5 / gv(b2, b2))
    N = 200000
    bh1 = np.empty(m); bh2 = np.empty(m)
    for i, ix in enumerate(idx):
        bh1[ix] = pop[i] @ b1[ix] + (chol[i] @ rng.standard_normal(k)) / np.sqrt(N)
        bh2[ix] = pop[i] @ b2[ix] + (chol[i] @ rng.standard_normal(k)) / np.sqrt(N)

    matched = [(pop[i].astype(np.float32), idx[i]) for i in range(nb)]
    r_match = ldpred3_auto_bivariate_blocks(matched, bh1, bh2, N, N, burn_in=120,
                                            num_iter=180, noise_inflation=True, seed=1)
    with pytest.warns(RuntimeWarning, match="Implausible bivariate fit"):
        off = ldpred3_auto_bivariate_blocks(
            ref, bh1, bh2, N, N, burn_in=120, num_iter=180, seed=1
        )
    on = ldpred3_auto_bivariate_blocks(ref, bh1, bh2, N, N, burn_in=120,
                                       num_iter=180, noise_inflation=True, seed=1)
    # matched LD -> lambda ~ 1 (near no-op)
    assert max(r_match.noise_scale) < 1.25, r_match.noise_scale
    # mismatch -> lambda well above 1
    assert max(on.noise_scale) > 1.3, on.noise_scale
    # the inflated count is deflated toward the truth (2*n_causal total causal)
    n_off = off.mixer["n_causal"][0] + off.mixer["n_causal"][1]
    n_on = on.mixer["n_causal"][0] + on.mixer["n_causal"][1]
    assert n_on < n_off                         # fix reduces the inflated count
    assert n_on < 0.85 * n_off                  # ... substantially
    # h2 and rg are preserved (not wrecked by the deflation)
    assert abs(on.rg - off.rg) < 0.1
    assert on.h2[0] > 0.2 and on.h2[1] > 0.2


def test_h2_cap_skips_prepass_and_validations():
    import pytest
    k, nb = 200, 8
    blocks, chols, idxs = _blocks(nb, k, seed=9)
    m = nb * k
    rng = np.random.default_rng(0)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.6, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 40000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 40000, k, rng)

    # h2_cap path (skips the univariate pre-pass) still recovers rg
    res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 40000, 40000,
                                        burn_in=80, num_iter=120,
                                        h2_cap=(0.5, 0.5), seed=1)
    assert abs(res.rg - 0.6) < 0.25

    with pytest.raises(ValueError, match="cross_corr"):
        ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 40000, 40000,
                                      cross_corr=1.0, h2_cap=(0.5, 0.5))

    overlap = [(blocks[0][0], np.arange(0, k)),
               (blocks[1][0], np.arange(k // 2, k // 2 + k))] + \
        [(blocks[i][0], np.arange(i * k, (i + 1) * k)) for i in range(2, nb)]
    with pytest.raises(ValueError, match="overlap|repeat"):
        ldpred3_auto_bivariate_blocks(overlap, bh1, bh2, 40000, 40000,
                                      h2_cap=(0.5, 0.5))


def test_h2_cap_binds_implied_heritability_not_raw_slab_variance():
    """h2_cap is s ≤ h2_cap / n_causal, not a raw slab-variance ceiling.

    A 0.02 ceiling must shrink both slab variances. If the cap were ignored,
    or applied as ``s ≤ 0.02`` (typical s is a few 1e-3 here), Sigma would
    not move.
    """
    k, nb = 200, 8
    blocks, chols, idxs = _blocks(nb, k, seed=9)
    m = nb * k
    rng = np.random.default_rng(0)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.6, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, 40000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 40000, k, rng)
    kw = dict(burn_in=80, num_iter=120, seed=1)
    uncapped = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 40000, 40000, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        capped = ldpred3_auto_bivariate_blocks(
            blocks, bh1, bh2, 40000, 40000, h2_cap=(0.02, 0.02), **kw)
    assert min(uncapped.h2) > 0.2
    assert capped.sigma[0, 0] < uncapped.sigma[0, 0]
    assert capped.sigma[1, 1] < uncapped.sigma[1, 1]
    assert max(capped.h2) < min(uncapped.h2)


def test_borrows_strength_for_low_power_trait():
    """With high rg, a low-N trait should predict better jointly than alone."""
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=2)
    m = nb * k
    N1, N2 = 100000, 3000       # trait 1 well powered, trait 2 weak
    bi, uni = [], []
    for rep in range(4):
        rng = np.random.default_rng(20 + rep)
        b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.9, rng=rng)
        bh1 = _sumstats(blocks, chols, idxs, b1, N1, k, rng)
        bh2 = _sumstats(blocks, chols, idxs, b2, N2, k, rng)
        res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, N1, N2,
                                            burn_in=120, num_iter=150, seed=rep)
        bi.append(_genetic_r2(res.beta2_est, b2, blocks, idxs))
        solo = ldpred3_by_blocks(blocks, bh2, np.full(m, float(N2)),
                                 method="auto", burn_in=120, num_iter=150, seed=rep)
        uni.append(_genetic_r2(solo, b2, blocks, idxs))
    assert np.mean(bi) > np.mean(uni) + 0.02, (np.mean(bi), np.mean(uni))


def _effective_lowrank(R):
    """Materialise the effective matrix under the installed ldpred3 contract."""
    from ldpred3.ld_repr import dense_ld

    return dense_ld(R, dtype=np.float64)


def test_lowrank_matmul_includes_diagonal_residual_and_zero_factor_row():
    from types import SimpleNamespace

    rng = np.random.default_rng(11)
    U = rng.standard_normal((9, 4)).astype(np.float32)
    U *= np.sqrt(0.6) / np.linalg.norm(U, axis=1)[:, None]
    U[-1] = 0.0
    supplied_residual = np.r_[np.full(8, 0.40001), 1.0]
    R = SimpleNamespace(U=U, scale=1.0,
                        residual_diag=supplied_residual)
    payload, scale, residual = bivariate._prepare_bivariate_lowrank_block(R)
    fblocks = [(bivariate._LOWRANK, payload, 0, 9, scale, residual,
                np.zeros(4), np.zeros(4))]
    V = rng.standard_normal((5, 9)).astype(np.float32)

    observed = bivariate._apply_R_rows(fblocks, V)
    expected_R = U @ U.T + np.diag(supplied_residual)
    expected = V @ expected_R.astype(np.float32)
    assert residual[-1] == 1.0
    assert residual.dtype == np.float32
    np.testing.assert_allclose(residual, supplied_residual, rtol=0, atol=2e-8)
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)


def test_prepared_lowrank_payload_is_shared_but_chain_scratch_is_not():
    from types import SimpleNamespace

    data = np.ones((5, 2), dtype=np.float32)
    scale = 1.0
    residual = np.zeros(5, dtype=np.float32)
    prepared = SimpleNamespace(
        blocks=((bivariate._LOWRANK, data, 0, 5, scale, residual),)
    )

    first = bivariate._instantiate_chain_blocks(prepared)[0]
    second = bivariate._instantiate_chain_blocks(prepared)[0]
    assert first[1] is second[1]
    assert first[4] == second[4] == 1.0
    assert first[5] is second[5]
    assert first[6] is not second[6]
    assert first[7] is not second[7]
    first[6][0] = 1.0
    assert second[6][0] == 0.0


def test_bivariate_lowrank_keeps_scalar_scale_and_float32_residual():
    from ldpred3 import lowrank_ld

    k = 12
    corr = 0.5 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    compact = lowrank_ld(corr, variance=0.7, max_rank=4, quantize=True)
    payload, scale, residual = bivariate._prepare_bivariate_lowrank_block(
        compact
    )

    assert payload.dtype == np.int8
    assert isinstance(scale, float)
    assert scale == compact.scale
    assert residual.dtype == np.float32
    assert np.shares_memory(residual, compact.residual_diag)


def test_lowrank_kernel_diagonal_residual_matches_dense_sweep():
    rng = np.random.default_rng(111)
    k, rank = 7, 3
    U = rng.standard_normal((k, rank)).astype(np.float32)
    U *= np.sqrt(0.6) / np.linalg.norm(U, axis=1)[:, None]
    factor_scale = 1.0
    residual = (1.0 - np.einsum(
        "ij,ij->i", U, U, dtype=np.float64
    )).astype(np.float32)
    dense = (U @ U.T + np.diag(residual)).astype(np.float32)
    bh1 = rng.standard_normal(k) * 0.01
    bh2 = rng.standard_normal(k) * 0.01
    n1 = np.full(k, 20_000.0)
    n2 = np.full(k, 18_000.0)
    curr1 = rng.standard_normal(k) * 0.002
    curr2 = rng.standard_normal(k) * 0.002
    unif = np.full(k, 1.0 - 1e-12)
    z1 = rng.standard_normal(k)
    z2 = rng.standard_normal(k)
    lpi = np.log(np.array([1e-6, 1e-6, 1e-6, 1.0 - 3e-6]))

    dense_buffers = [curr1.copy(), curr2.copy(), np.zeros(k), np.zeros(k),
                     np.zeros(k), np.zeros(k)]
    lowrank_buffers = [a.copy() for a in dense_buffers]
    dcurr1, dcurr2, drb1, drb2, drbs1, drbs2 = dense_buffers
    lcurr1, lcurr2, lrb1, lrb2, lrbs1, lrbs2 = lowrank_buffers

    dense_result = bivariate._bivar_one_sweep(
        dense, bh1, bh2, n1, n2, dcurr1, dcurr2, drb1, drb2, drbs1,
        drbs2, unif, z1, z2, *lpi, 8e-5, 9e-5, 2e-5, 0.1,
        1.0, True, True)
    lowrank_result = bivariate._bivar_one_sweep_lowrank(
        U, factor_scale, residual, bh1, bh2, n1, n2, lcurr1, lcurr2,
        np.zeros(rank), np.zeros(rank), lrb1, lrb2, lrbs1, lrbs2,
        unif, z1, z2, *lpi, 8e-5, 9e-5, 2e-5, 0.1,
        True, True, True)

    np.testing.assert_allclose(lowrank_result, dense_result,
                               rtol=3e-6, atol=3e-9)
    for observed, expected in zip(lowrank_buffers, dense_buffers):
        np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=3e-9)


def test_truncated_float_lowrank_matches_its_effective_dense_matrix():
    from ldpred3 import lowrank_ld

    k = 14
    corr = 0.45 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    R = lowrank_ld(corr, variance=0.7, max_rank=4)
    if hasattr(R, "residual_diag"):
        assert np.any(np.asarray(R.residual_diag) > 0.0)
    dense = _effective_lowrank(R).astype(np.float32)
    rng = np.random.default_rng(12)
    bh1 = rng.standard_normal(k) * 0.025
    bh2 = 0.6 * bh1 + rng.standard_normal(k) * 0.015
    kwargs = dict(burn_in=12, num_iter=18, seed=9, ld_int8=False)

    expected = ldpred3_auto_bivariate_blocks(
        [(dense, np.arange(k))], bh1, bh2, 30000, 25000, **kwargs)
    observed = ldpred3_auto_bivariate_blocks(
        [(R, np.arange(k))], bh1, bh2, 30000, 25000,
        **{**kwargs, "ld_int8": True})

    np.testing.assert_allclose(observed.beta1_est, expected.beta1_est,
                               rtol=3e-5, atol=3e-7)
    np.testing.assert_allclose(observed.beta2_est, expected.beta2_est,
                               rtol=3e-5, atol=3e-7)
    np.testing.assert_allclose(observed.pi_samples, expected.pi_samples,
                               rtol=3e-5, atol=3e-7)
    np.testing.assert_allclose(observed.sigma_samples, expected.sigma_samples,
                               rtol=3e-5, atol=3e-7)


def test_lr8_matches_effective_dense_and_ignores_dense_storage_policy():
    from ldpred3 import lowrank_ld

    k = 18
    corr = 0.55 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    R = lowrank_ld(corr, variance=0.7, max_rank=4, quantize=True)
    if hasattr(R, "residual_diag"):
        assert np.any(np.asarray(R.residual_diag) > 0.0)
    dense = _effective_lowrank(R).astype(np.float32)
    rng = np.random.default_rng(13)
    bh1 = rng.standard_normal(k) * 0.02
    bh2 = 0.5 * bh1 + rng.standard_normal(k) * 0.018
    kwargs = dict(burn_in=10, num_iter=16, seed=4)

    expected = ldpred3_auto_bivariate_blocks(
        [(dense, np.arange(k))], bh1, bh2, 20000, 18000,
        ld_int8=False, **kwargs)
    observed = ldpred3_auto_bivariate_blocks(
        [(R, np.arange(k))], bh1, bh2, 20000, 18000,
        ld_int8=True, **kwargs)
    automatic = ldpred3_auto_bivariate_blocks(
        [(R, np.arange(k))], bh1, bh2, 20000, 18000,
        ld_int8=None, **kwargs)

    np.testing.assert_allclose(observed.beta1_est, expected.beta1_est,
                               rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(observed.beta2_est, expected.beta2_est,
                               rtol=2e-5, atol=2e-7)
    np.testing.assert_array_equal(automatic.beta1_est, observed.beta1_est)
    np.testing.assert_array_equal(automatic.pi_samples, observed.pi_samples)


def test_mixed_lowrank_dense_supports_optional_estimators():
    from ldpred3 import LowRankLD

    k = 10
    corr1 = 0.4 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    corr2 = 0.3 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    lowrank = LowRankLD(np.linalg.cholesky(corr1).astype(np.float32), k)
    blocks = [(lowrank, np.arange(k)),
              (corr2.astype(np.float32), np.arange(k, 2 * k))]
    rng = np.random.default_rng(14)
    bh1 = rng.standard_normal(2 * k) * 0.04
    bh2 = 0.8 * bh1 + rng.standard_normal(2 * k) * 0.002

    result = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, 25000, 15000, burn_in=8, num_iter=12, seed=7,
        rg_decorrelated=True, sample_every=2, noise_inflation=True)

    assert np.all(np.isfinite(result.beta1_est))
    assert np.all(np.isfinite(result.beta2_est))
    assert np.isfinite(result.rg)
    assert np.all(np.isfinite(result.h2))
    assert np.all(np.isfinite(result.noise_scale))
    np.testing.assert_allclose(
        result.noise_scale, result.noise_scale_samples.mean(axis=0)
    )


def _assert_bivariate_result_array_equal(observed, expected):
    for name in ("beta1_est", "beta2_est", "sigma", "pi", "pi_samples",
                 "sigma_samples", "genetic_samples", "noise_scale_samples"):
        np.testing.assert_array_equal(getattr(observed, name),
                                      getattr(expected, name))
    assert observed.h2 == expected.h2
    assert observed.rg == expected.rg
    assert observed.p == expected.p
    assert observed.noise_scale == expected.noise_scale


@pytest.mark.skipif(not bivariate.HAVE_NUMBA, reason="Numba is required")
def test_ncores_two_matches_one_for_readonly_variable_size_d8_blocks():
    sizes = (7, 10, 6)
    blocks = []
    start = 0
    for k, rho in zip(sizes, (0.25, 0.45, 0.6)):
        corr = rho ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
        payload = np.rint(corr * 127.0).astype(np.int8)
        payload.setflags(write=False)
        blocks.append((payload, np.arange(start, start + k)))
        start += k
    rng = np.random.default_rng(21)
    bh1 = rng.standard_normal(start) * 0.02
    bh2 = 0.4 * bh1 + rng.standard_normal(start) * 0.015
    kwargs = dict(burn_in=5, num_iter=8, seed=22)

    serial = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, 20_000, 18_000, ncores=1, **kwargs)
    parallel = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, 20_000, 18_000, ncores=2, **kwargs)

    _assert_bivariate_result_array_equal(parallel, serial)
    assert all(not block.flags.writeable for block, _idx in blocks)


@pytest.mark.skipif(not bivariate.HAVE_NUMBA, reason="Numba is required")
def test_ncores_one_fuses_homogeneous_blocks(monkeypatch):
    k = 6
    corr = 0.3 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    payload = np.rint(corr * 127.0).astype(np.int8)
    blocks = [
        (payload.copy(), np.arange(i * k, (i + 1) * k))
        for i in range(4)
    ]
    rng = np.random.default_rng(214)
    bh1 = rng.standard_normal(4 * k) * 0.02
    bh2 = rng.standard_normal(4 * k) * 0.02
    calls = 0
    fused = bivariate._bivar_dense_sweep_all_jit

    def counting_fused(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fused(*args, **kwargs)

    monkeypatch.setattr(
        bivariate, "_bivar_dense_sweep_all_jit", counting_fused
    )
    fused_result = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, 20_000, 18_000,
        burn_in=3, num_iter=5, ncores=1, seed=215,
    )

    assert calls == 8
    monkeypatch.setattr(bivariate, "HAVE_NUMBA", False)
    per_block_result = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, 20_000, 18_000,
        burn_in=3, num_iter=5, ncores=1, seed=215,
    )
    _assert_bivariate_result_array_equal(fused_result, per_block_result)


@pytest.mark.skipif(not bivariate.HAVE_NUMBA, reason="Numba is required")
def test_all_lowrank_fit_skips_genome_length_rb_buffers(monkeypatch):
    from ldpred3 import lowrank_ld

    k = 12
    corr = 0.4 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    compact = lowrank_ld(corr, variance=0.7, max_rank=4, quantize=True)
    rng = np.random.default_rng(216)
    bh1 = rng.standard_normal(k) * 0.02
    bh2 = rng.standard_normal(k) * 0.02
    observed_sizes = []
    fused = bivariate._bivar_lowrank_sweep_all_jit

    def recording_fused(*args, **kwargs):
        observed_sizes.append((args[13].size, args[14].size))
        return fused(*args, **kwargs)

    monkeypatch.setattr(
        bivariate, "_bivar_lowrank_sweep_all_jit", recording_fused
    )
    ldpred3_auto_bivariate_blocks(
        [(compact, np.arange(k))], bh1, bh2, 20_000, 18_000,
        burn_in=2, num_iter=3, ncores=1, seed=217,
    )

    assert observed_sizes == [(0, 0)] * 5


@pytest.mark.skipif(not bivariate.HAVE_NUMBA, reason="Numba is required")
def test_ncores_restores_numba_thread_mask_after_exception(monkeypatch):
    from numba import config, get_num_threads, set_num_threads

    if config.NUMBA_NUM_THREADS < 2:
        pytest.skip("Numba thread pool has only one thread")
    original = get_num_threads()
    set_num_threads(1)

    def fail_sweep(*_args, **_kwargs):
        raise RuntimeError("deliberate sweep failure")

    monkeypatch.setattr(
        bivariate, "_bivar_dense_sweep_all_par_jit", fail_sweep
    )
    beta_hat = np.full(4, 0.02)
    try:
        with pytest.raises(RuntimeError, match="deliberate sweep failure"):
            ldpred3_auto_bivariate(
                np.eye(4), beta_hat, beta_hat, 1000, 1000,
                ld_int8=False, burn_in=0, num_iter=1,
                ncores=2, seed=218,
            )
        assert get_num_threads() == 1
    finally:
        set_num_threads(original)


@pytest.mark.skipif(not bivariate.HAVE_NUMBA, reason="Numba is required")
def test_ncores_two_matches_one_for_float32_blocks_with_per_variant_n():
    sizes = (8, 11)
    blocks = []
    start = 0
    for k, rho in zip(sizes, (0.2, 0.5)):
        corr = rho ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
        blocks.append((corr.astype(np.float32), np.arange(start, start + k)))
        start += k
    rng = np.random.default_rng(211)
    bh1 = rng.standard_normal(start) * 0.02
    bh2 = 0.3 * bh1 + rng.standard_normal(start) * 0.016
    n1 = np.linspace(12_000.0, 20_000.0, start)
    n2 = np.linspace(10_000.0, 18_000.0, start)[::-1].copy()
    kwargs = dict(burn_in=4, num_iter=7, seed=212, ld_int8=False)

    serial = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, n1, n2, ncores=1, **kwargs)
    parallel = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, n1, n2, ncores=2, **kwargs)

    _assert_bivariate_result_array_equal(parallel, serial)


@pytest.mark.skipif(not bivariate.HAVE_NUMBA, reason="Numba is required")
@pytest.mark.parametrize("quantize", [False, True], ids=["lr32", "lr8"])
def test_ncores_two_matches_one_for_variable_rank_lowrank_blocks(quantize):
    from ldpred3 import lowrank_ld

    blocks = []
    start = 0
    for k, rho, rank in ((8, 0.35, 2), (11, 0.55, 4), (7, 0.25, 3)):
        corr = rho ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
        compact = lowrank_ld(corr, variance=0.7, max_rank=rank,
                             quantize=quantize)
        compact.U.setflags(write=False)
        blocks.append((compact, np.arange(start, start + k)))
        start += k
    rng = np.random.default_rng(23)
    bh1 = rng.standard_normal(start) * 0.018
    bh2 = 0.5 * bh1 + rng.standard_normal(start) * 0.012
    kwargs = dict(
        burn_in=5, num_iter=8, seed=24, noise_inflation=True,
    )

    serial = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, 22_000, 17_000, ncores=1, **kwargs)
    parallel = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, 22_000, 17_000, ncores=2, **kwargs)

    _assert_bivariate_result_array_equal(parallel, serial)
    assert all(not block.U.flags.writeable for block, _idx in blocks)


def test_ncores_mixed_blocks_bucket_into_parallel_calls(monkeypatch):
    from ldpred3 import lowrank_ld

    k = 6
    corr = 0.3 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    d8 = np.rint(corr * 127.0).astype(np.int8)
    f32 = corr.astype(np.float32)
    lr8 = lowrank_ld(corr, variance=0.7, max_rank=3, quantize=True)
    blocks = [(d8, np.arange(k)),
              (f32, np.arange(k, 2 * k)),
              (lr8, np.arange(2 * k, 3 * k))]
    rng = np.random.default_rng(25)
    bh1 = rng.standard_normal(3 * k) * 0.02
    bh2 = rng.standard_normal(3 * k) * 0.02
    kwargs = dict(
        burn_in=3, num_iter=5, seed=26, ld_int8=False,
    )
    serial = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, 16_000, 14_000, ncores=1, **kwargs)

    # A mixed panel cannot share one typed.List element type, but it is bucketed
    # by (kind, dtype, scale) and each bucket gets its own fused parallel call --
    # int8 dense, float32 dense and LR8 low-rank here, so three buckets. Blocks
    # partition the variants, and the reduction still reads per-block statistics
    # in genome order, so the result stays bit-identical to the serial sweep.
    calls = {"dense": 0, "lowrank": 0}
    dense_jit = bivariate._bivar_dense_sweep_all_par_jit
    lowrank_jit = bivariate._bivar_lowrank_sweep_all_par_jit

    def counting_dense(*args, **kw):
        calls["dense"] += 1
        return dense_jit(*args, **kw)

    def counting_lowrank(*args, **kw):
        calls["lowrank"] += 1
        return lowrank_jit(*args, **kw)

    monkeypatch.setattr(bivariate, "_bivar_dense_sweep_all_par_jit",
                        counting_dense)
    monkeypatch.setattr(bivariate, "_bivar_lowrank_sweep_all_par_jit",
                        counting_lowrank)
    requested = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, 16_000, 14_000, ncores=2, **kwargs)

    sweeps = kwargs["burn_in"] + kwargs["num_iter"]
    if bivariate.HAVE_NUMBA:
        assert calls["dense"] == 2 * sweeps     # int8 and float32 buckets
        assert calls["lowrank"] == sweeps       # the LR8 bucket
    else:
        # Without Numba there is no fused kernel to call; every block is swept
        # by the serial driver, which is what ncores > 1 falls back to.
        assert calls == {"dense": 0, "lowrank": 0}
    _assert_bivariate_result_array_equal(requested, serial)
    monkeypatch.setattr(bivariate, "HAVE_NUMBA", False)
    per_block = ldpred3_auto_bivariate_blocks(
        blocks, bh1, bh2, 16_000, 14_000, ncores=1, **kwargs
    )
    _assert_bivariate_result_array_equal(serial, per_block)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"ld_int8": 1}, "ld_int8.*boolean"),
        ({"ld_int8": "auto"}, "ld_int8.*boolean"),
        ({"h2_init": 0.0}, "h2"),
        ({"h2_init": np.nan}, "h2"),
        ({"h2_init": (0.1,)}, "h2_init"),
        ({"h2_init": (0.1, -0.2)}, "h2_init"),
        ({"h2_init": (0.1, True)}, "h2_init"),
        ({"h2_init": "0.1"}, "h2_init"),
        ({"sigma_prior_scale": "0.1"}, "sigma_prior_scale"),
        ({"p_init": 0.0}, "p"),
        ({"p_init": 1.1}, "p"),
        ({"rg_init": 1.0}, "rg_init"),
        ({"rg_init": np.nan}, "rg_init"),
        ({"pi_init": (0.8, 0.1, 0.1)}, "pi_init"),
        ({"pi_init": (0.8, 0.05, 0.05, 0.1), "rg_init": 0.9},
         "cannot represent rg_init"),
        ({"cross_corr": 1.0}, "cross_corr"),
        ({"cross_corr": np.nan}, "cross_corr"),
        ({"burn_in": -1}, "burn_in"),
        ({"burn_in": 1.5}, "burn_in"),
        ({"num_iter": 0}, "num_iter"),
        ({"num_iter": True}, "num_iter"),
        ({"h2_bounds": (0.1,)}, "h2_bounds"),
        ({"h2_bounds": (0.2, 0.5)}, "h2_bounds"),
        ({"h2_bounds": (0.0, 1.0)}, "h2_bounds"),
        ({"h2_bounds": (-1.0, 1.0)}, "h2_bounds"),
        ({"h2_bounds": (1e-4, np.inf)}, "h2_bounds"),
        ({"h2_cap": (0.2,)}, "h2_cap"),
        ({"h2_cap": (0.0, 0.2)}, "h2_cap"),
        ({"h2_cap": (0.2, np.nan)}, "h2_cap"),
        ({"iw_df": 0.0}, "iw_df"),
        ({"iw_df": np.inf}, "iw_df"),
        ({"rg_decorrelated": 1}, "rg_decorrelated.*boolean"),
        ({"noise_inflation": 0}, "noise_inflation.*boolean"),
        ({"ni_damp": 0.0}, "ni_damp"),
        ({"ni_damp": 1.1}, "ni_damp"),
        ({"pi_prior": 0.0}, "pi_prior"),
        ({"pi_prior": np.nan}, "pi_prior"),
        ({"sigma_prior_scale": 0.0}, "sigma_prior_scale"),
        ({"sigma_prior_scale": (0.1,)}, "sigma_prior_scale"),
        ({"sigma_prior_scale": (0.1, True)}, "sigma_prior_scale"),
        ({"sample_every": 0}, "sample_every"),
        ({"sample_every": 1.5}, "sample_every"),
        ({"ncores": 0}, "ncores"),
        ({"ncores": 1.5}, "ncores"),
        ({"ncores": True}, "ncores"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_bivariate_validates_public_controls(overrides, match):
    R = np.eye(3)
    beta = np.zeros(3)
    kwargs = {"burn_in": 0, "num_iter": 1, "h2_cap": (0.2, 0.2)}
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        ldpred3_auto_bivariate(R, beta, beta, 1000, 1000, **kwargs)


@pytest.mark.parametrize(
    "beta1, beta2, n1, n2, match",
    [
        (np.zeros((1, 2)), np.zeros(2), 1000, 1000, "one-dimensional"),
        (np.zeros(2), np.zeros((1, 2)), 1000, 1000, "one-dimensional"),
        (np.zeros(2), np.zeros(3), 1000, 1000, "same length"),
        (np.array([0.0, np.nan]), np.zeros(2), 1000, 1000, "finite"),
        (np.zeros(0), np.zeros(0), 1000, 1000, "at least one"),
        (np.zeros(2), np.zeros(2), 0, 1000, "finite positive"),
        (np.zeros(2), np.zeros(2), [1000], 1000, "length-2"),
        (np.zeros(2), np.zeros(2), 1000, [1000, np.inf], "finite positive"),
        (np.zeros(2), np.zeros(2), True, 1000, "finite positive"),
    ],
)
def test_bivariate_validates_effect_and_sample_size_vectors(
        beta1, beta2, n1, n2, match):
    with pytest.raises(ValueError, match=match):
        ldpred3_auto_bivariate(
            np.eye(2), beta1, beta2, n1, n2,
            burn_in=0, num_iter=1, h2_cap=(0.2, 0.2),
        )


@pytest.mark.parametrize(
    "blocks, m, match",
    [
        ([(np.eye(3), np.arange(2))], 2, "shape"),
        ([(np.ones((2, 3)), np.arange(2))], 2, "shape"),
        ([(np.array([[1.0, np.nan], [np.nan, 1.0]]), np.arange(2))], 2,
         "finite"),
        ([(np.array([[1.0, 0.2], [0.3, 1.0]]), np.arange(2))], 2,
         "symmetric"),
        ([(np.array([[0.9, 0.2], [0.2, 1.0]]), np.arange(2))], 2,
         "diagonal"),
        ([(np.array([[1.0, 1.2], [1.2, 1.0]]), np.arange(2))], 2,
         r"\[-1, 1\]"),
        ([(np.array([[126, 0], [0, 127]], dtype=np.int8), np.arange(2))], 2,
         "diagonal"),
        ([(np.array([[127, -128], [-128, 127]], dtype=np.int8), np.arange(2))], 2,
         "out-of-range"),
        ([(np.eye(2), np.array([0.0, 1.0]))], 2, "integer"),
        ([(np.empty((0, 0)), np.array([], dtype=int)),
          (np.eye(2), np.arange(2))], 2, "must not be empty"),
        ([(np.eye(2), np.array([0, 2])),
          (np.eye(1), np.array([1]))], 3, "contiguous"),
    ],
)
def test_bivariate_validates_dense_ld_block_geometry(blocks, m, match):
    beta = np.zeros(m)
    with pytest.raises(ValueError, match=match):
        ldpred3_auto_bivariate_blocks(
            blocks, beta, beta, 1000, 1000,
            burn_in=0, num_iter=1, h2_cap=(0.2, 0.2),
        )


def test_per_variant_n_controls_variant_specific_shrinkage():
    # One retained sweep is enough: Rao-Blackwellized effects are computed before
    # the stochastic state draw. Equal marginal effects get much less shrinkage
    # at the high-N variant, exercising the per-variant-N kernel branch.
    beta_hat = np.full(2, 0.02)
    n_eff = np.array([100.0, 100_000.0])
    res = ldpred3_auto_bivariate(
        np.eye(2), beta_hat, beta_hat, n_eff, n_eff,
        ld_int8=False, h2_init=0.1, p_init=0.5,
        burn_in=0, num_iter=1, h2_cap=(0.2, 0.2), seed=1,
    )
    assert np.all(np.isfinite(res.beta1_est))
    assert res.beta1_est[1] > 5.0 * res.beta1_est[0]


def test_decorrelated_rg_accumulator_is_opt_in_and_path_is_used(monkeypatch):
    assert bivariate._decorrelated_accumulator(False, 10_000_000) is None
    accumulator = bivariate._decorrelated_accumulator(True, 4)
    assert accumulator.sum1.shape == accumulator.sum2.shape == (4,)
    assert accumulator.sum1.dtype == accumulator.sum2.dtype == np.float64
    assert accumulator.diagonal.shape == (3,)

    calls = []

    def fake_decorrelated(_blocks, moments):
        calls.append((moments.count, moments.sum1.shape, moments.sum2.shape))
        return 0.25, 1.0, 1.0

    monkeypatch.setattr(bivariate, "_decorrelated_cov", fake_decorrelated)
    beta_hat = np.full(4, 0.02)
    kwargs = dict(
        ld_int8=False, h2_init=0.1, p_init=0.5, burn_in=0,
        num_iter=3, sample_every=2, h2_cap=(0.2, 0.2), seed=1,
    )
    ldpred3_auto_bivariate(np.eye(4), beta_hat, beta_hat, 1000, 1000, **kwargs)
    assert calls == []
    res = ldpred3_auto_bivariate(
        np.eye(4), beta_hat, beta_hat, 1000, 1000,
        rg_decorrelated=True, **kwargs,
    )
    assert calls == [(2, (4,), (4,))]
    assert res.rg == 0.25


def test_online_decorrelated_moments_match_materialised_formula():
    rng = np.random.default_rng(123)
    k = 9
    corr = (0.4 ** np.abs(
        np.subtract.outer(np.arange(k), np.arange(k))
    )).astype(np.float32)
    fblocks = [
        (bivariate._DENSE, corr, 0, k, 1.0, None, None, None)
    ]
    samples1 = rng.standard_normal((7, k))
    samples2 = rng.standard_normal((7, k))

    moments = bivariate._decorrelated_accumulator(True, k)
    for beta1, beta2 in zip(samples1, samples2):
        Rbeta1, Rbeta2 = bivariate._apply_R_rows(
            fblocks, np.vstack([beta1, beta2])
        )
        bivariate._accumulate_decorrelated(
            moments, beta1, beta2,
            (beta1 @ Rbeta1, beta1 @ Rbeta2, beta2 @ Rbeta2),
        )
    observed = bivariate._decorrelated_cov(fblocks, moments)

    sum1 = samples1.sum(axis=0, keepdims=True)
    sum2 = samples2.sum(axis=0, keepdims=True)
    Rsum1, Rsum2 = bivariate._apply_R_rows(
        fblocks, np.vstack([sum1, sum2])
    )
    all11 = float(sum1[0] @ Rsum1)
    all12 = float(sum1[0] @ Rsum2)
    all22 = float(sum2[0] @ Rsum2)
    Rs1 = bivariate._apply_R_rows(fblocks, samples1)
    Rs2 = bivariate._apply_R_rows(fblocks, samples2)
    diagonal = (
        float(np.einsum("ij,ij->", samples1, Rs1)),
        float(np.einsum("ij,ij->", samples1, Rs2)),
        float(np.einsum("ij,ij->", samples2, Rs2)),
    )
    npairs = len(samples1) * (len(samples1) - 1)
    expected = (
        (all12 - diagonal[1]) / npairs,
        (all11 - diagonal[0]) / npairs,
        (all22 - diagonal[2]) / npairs,
    )
    np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=2e-14)


def test_decorrelated_rg_applies_ld_only_once_at_finalization(monkeypatch):
    apply_R_rows = bivariate._apply_R_rows
    calls = 0

    def counting_apply(*args, **kwargs):
        nonlocal calls
        calls += 1
        return apply_R_rows(*args, **kwargs)

    monkeypatch.setattr(bivariate, "_apply_R_rows", counting_apply)
    beta_hat = np.full(20, 0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = ldpred3_auto_bivariate(
            np.eye(20), beta_hat, beta_hat, 10_000, 10_000,
            ld_int8=False, h2_init=0.2, p_init=0.5, burn_in=5,
            num_iter=12, sample_every=2, h2_cap=(0.5, 0.5),
            rg_decorrelated=True, seed=2,
        )

    # calls == 1 is the invariant under test: the LD is applied exactly once at
    # finalization. This deliberately tiny config produces degenerate sparse
    # support on some platforms' float ordering, where the decorrelated rg is
    # NaN via the documented degrade path; that is acceptable here, so allow
    # either a finite estimate or NaN (but never an infinity).
    assert calls == 1
    assert not np.isinf(result.rg)


@pytest.mark.parametrize(
    "num_iter,sample_every",
    [(1, 1), (5, 5), (5, 10)],
)
def test_decorrelated_rg_requires_two_retained_effect_samples(
        num_iter, sample_every):
    beta_hat = np.full(4, 0.02)
    with pytest.raises(
        ValueError,
        match="at least two retained effect samples.*num_iter.*sample_every",
    ):
        ldpred3_auto_bivariate(
            np.eye(4), beta_hat, beta_hat, 1000, 1000,
            ld_int8=False, h2_init=0.1, p_init=0.5, burn_in=0,
            num_iter=num_iter, sample_every=sample_every,
            h2_cap=(0.2, 0.2), rg_decorrelated=True, seed=1,
        )


def test_decorrelated_rg_raises_on_non_finite_cross_sweep_quadratics(monkeypatch):
    # Non-finite quadratics signal a broken computation and still hard-raise.
    monkeypatch.setattr(bivariate, "_decorrelated_cov",
                        lambda *_args: (np.nan, 1.0, 1.0))
    beta_hat = np.full(4, 0.02)
    with pytest.raises(ValueError,
                       match="non-finite cross-sweep quadratic forms"):
        ldpred3_auto_bivariate(
            np.eye(4), beta_hat, beta_hat, 1000, 1000,
            ld_int8=False, h2_init=0.1, p_init=0.5, burn_in=0,
            num_iter=3, sample_every=2, h2_cap=(0.2, 0.2),
            rg_decorrelated=True, seed=1,
        )


@pytest.mark.parametrize("cov", [(0.25, 0.0, 1.0), (0.25, 1.0, -0.1)])
def test_decorrelated_rg_degrades_to_nan_on_non_positive_variance(
        monkeypatch, cov):
    # A non-positive cross-sweep genetic variance leaves the decorrelated rg
    # undefined; the fit warns and reports rg as NaN rather than aborting.
    monkeypatch.setattr(bivariate, "_decorrelated_cov", lambda *_args: cov)
    beta_hat = np.full(4, 0.02)
    with pytest.warns(RuntimeWarning,
                      match="non-positive cross-sweep genetic variance"):
        result = ldpred3_auto_bivariate(
            np.eye(4), beta_hat, beta_hat, 1000, 1000,
            ld_int8=False, h2_init=0.1, p_init=0.5, burn_in=0,
            num_iter=3, sample_every=2, h2_cap=(0.2, 0.2),
            rg_decorrelated=True, seed=1,
        )
    assert np.isnan(result.rg)


def test_cross_corr_explains_correlated_sampling_signal():
    # Identical small marginal effects are consistent with correlated sampling
    # noise. Supplying a strong positive cross_corr therefore reduces the joint
    # posterior effects relative to incorrectly assuming independent noise.
    beta_hat = np.full(4, 0.03)
    kwargs = dict(
        ld_int8=False, h2_init=0.1, p_init=0.5, burn_in=0,
        num_iter=1, h2_cap=(0.2, 0.2), seed=1,
    )
    independent = ldpred3_auto_bivariate(
        np.eye(4), beta_hat, beta_hat, 1000, 1000, cross_corr=0.0, **kwargs)
    corrected = ldpred3_auto_bivariate(
        np.eye(4), beta_hat, beta_hat, 1000, 1000, cross_corr=0.8, **kwargs)
    assert np.linalg.norm(corrected.beta1_est) < 0.25 * np.linalg.norm(
        independent.beta1_est)


def test_initial_hyperparameters_extreme_rg_saturates_shared():
    # |rg_init| above the 0.999 boundary would require more shared mass than
    # the union probability; the shorthand must saturate at an all-shared
    # start (a valid probability vector) while keeping the implied moments
    # exact.
    m = 1000
    for rg_init in (0.999, 0.9999, -0.9999):
        pi, s1, s2, s12 = bivariate._initial_hyperparameters(
            m, (0.1, 0.05), 0.02, rg_init,
        )
        assert np.all(pi >= 0.0)
        np.testing.assert_allclose(pi, (0.98, 0.0, 0.0, 0.02), atol=1e-15)
        p1, p2, shared = pi[1] + pi[3], pi[2] + pi[3], pi[3]
        h1, h2 = m * p1 * s1, m * p2 * s2
        rg = m * shared * s12 / np.sqrt(h1 * h2)
        np.testing.assert_allclose((h1, h2, rg), (0.1, 0.05, rg_init),
                                   rtol=1e-12)
        assert abs(s12 / np.sqrt(s1 * s2)) < 1.0


def test_rg_decorrelated_recovers_rg_for_asymmetric_power():
    # With one strong and one weak trait the decorrelated estimator recovers
    # rg (and exercises the retained effect-sample path end to end).
    k, nb = 200, 12
    blocks, chols, idxs = _blocks(nb, k, seed=3)
    m = nb * k
    rgs = []
    for rep in range(3):
        rng = np.random.default_rng(30 + rep)
        b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.8,
                      rng=rng)
        bh1 = _sumstats(blocks, chols, idxs, b1, 100000, k, rng)
        bh2 = _sumstats(blocks, chols, idxs, b2, 5000, k, rng)
        res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 100000, 5000,
                                            rg_decorrelated=True,
                                            burn_in=150, num_iter=200, seed=rep)
        assert np.isfinite(res.rg)
        rgs.append(res.rg)
    assert abs(np.mean(rgs) - 0.8) < 0.15, np.mean(rgs)


def test_scalar_n_matches_constant_vector_n_bit_for_bit():
    # When N is a shared scalar the residual-independent constants are hoisted out
    # of the per-SNP loop (the n_const path in _bivar_const / _bivar_one_sweep),
    # documented as leaving the per-SNP arithmetic unchanged and therefore
    # bit-identical to the per-variant path. Pin that invariant: a scalar N and a
    # constant per-variant N vector must agree exactly, not merely approximately.
    rng = np.random.default_rng(0)
    m = 300
    corr = np.eye(m)
    beta1 = rng.normal(0.0, 0.05, m)
    beta2 = rng.normal(0.0, 0.05, m)
    n1, n2 = 50_000.0, 40_000.0
    scalar = ldpred3_auto_bivariate(corr, beta1, beta2, n1, n2, seed=7,
                                    burn_in=40, num_iter=40)
    vector = ldpred3_auto_bivariate(corr, beta1, beta2, np.full(m, n1),
                                    np.full(m, n2), seed=7, burn_in=40,
                                    num_iter=40)
    assert scalar.rg == vector.rg
    assert scalar.h2 == vector.h2
    np.testing.assert_array_equal(scalar.beta1_est, vector.beta1_est)
    np.testing.assert_array_equal(scalar.beta2_est, vector.beta2_est)


def test_cross_corr_with_per_variant_n():
    # The per-SNP branch of the noise covariance (E12 != 0 with per-variant N)
    # stays finite and still shrinks correlated sampling noise.
    beta_hat = np.full(4, 0.03)
    n_vec = np.array([500.0, 2000.0, 1000.0, 4000.0])
    kwargs = dict(ld_int8=False, h2_init=0.1, p_init=0.5, burn_in=0,
                  num_iter=1, h2_cap=(0.2, 0.2), seed=1)
    independent = ldpred3_auto_bivariate(
        np.eye(4), beta_hat, beta_hat, n_vec, n_vec, cross_corr=0.0, **kwargs)
    corrected = ldpred3_auto_bivariate(
        np.eye(4), beta_hat, beta_hat, n_vec, n_vec, cross_corr=0.6, **kwargs)
    assert np.all(np.isfinite(corrected.beta1_est))
    assert np.all(np.isfinite(corrected.beta2_est))
    assert np.isfinite(corrected.rg)
    assert np.linalg.norm(corrected.beta1_est) < np.linalg.norm(
        independent.beta1_est)


def test_noise_inflation_with_per_variant_n():
    # noise_inflation with per-variant N runs the deflation loop per SNP and
    # stays sane on matched LD (lambda near 1).
    k, nb = 200, 4
    blocks, chols, idxs = _blocks(nb, k, seed=6)
    m = nb * k
    rng = np.random.default_rng(7)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.5, 0.5), rg=0.6, rng=rng)
    n_vec = np.full(m, 40000.0)
    n_vec[::7] = 15000.0
    bh1 = _sumstats(blocks, chols, idxs, b1, 40000, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, 40000, k, rng)
    res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, n_vec, n_vec,
                                        burn_in=60, num_iter=80,
                                        noise_inflation=True, seed=1)
    assert np.all(np.isfinite(res.beta1_est))
    assert -1.0 <= res.rg <= 1.0
    assert all(1.0 <= lam < 2.0 for lam in res.noise_scale)


def test_single_variant_fit_is_well_formed():
    # Degenerate one-SNP input must not crash or produce NaNs; with no signal
    # the heritabilities sit at the lower bound and rg stays in [-1, 1].
    kw = dict(burn_in=0, num_iter=2, h2_cap=(0.2, 0.2), seed=0)
    beta = np.array([0.05])
    single = ldpred3_auto_bivariate(np.eye(1), beta, beta, 1000, 1000, **kw)
    blocked = ldpred3_auto_bivariate_blocks(
        [(np.eye(1), np.arange(1))], beta, beta, 1000, 1000, **kw)
    for res in (single, blocked):
        assert np.all(np.isfinite(res.beta1_est))
        assert np.all(np.isfinite(res.beta2_est))
        assert -1.0 <= res.rg <= 1.0
        assert res.h2[0] >= 1e-4 and res.h2[1] >= 1e-4


def _diverged_args(**over):
    """Arguments to ``_warn_if_fit_diverged`` describing a healthy fit.

    The defaults are the real post-QC LDL x CAD numbers: sum(beta^2)/h2 of
    0.65, a largest posterior mean 4.1 slab SDs out, and a flat trace. Each
    test perturbs one of them to the value the *diverged* fit had.
    """
    m = bivariate._DIAGNOSTIC_MIN_VARIANTS
    beta1 = np.zeros(m)
    beta1[0] = 0.0239                             # 4.1 slab SDs at sigma below
    beta1[1:] = np.sqrt(max(0.0573 - 0.0239 ** 2, 0.0) / (m - 1))
    args = dict(beta1=beta1, beta2=np.zeros(m),
                raw_h2=(0.0882, 0.0706),
                sigma_diag=(0.0239 / 4.1) ** 2,   # placeholder, replaced below
                genetic_samples=np.tile([0.0882, 0.01, 0.0706], (80, 1)),
                m=m)
    args["sigma_diag"] = ((0.0239 / 4.1) ** 2, 1.0)
    args.update(over)
    return args


def test_no_divergence_warning_on_a_healthy_fit():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        diagnostic = bivariate._warn_if_fit_diverged(
            **_diverged_args(largest_block=12_169))
    assert diagnostic["evaluated"] is True
    assert diagnostic["flagged"] is False
    assert diagnostic["largest_block_variants"] == 12_169
    assert diagnostic["trace_iterations"] == 80
    assert diagnostic["trace_evaluated"] is True
    assert diagnostic["thresholds"] == {
        "minimum_variants": 1000,
        "minimum_trace_iterations": 40,
        "effect_energy_ratio": 10.0,
        "max_effect_slab_sd": 25.0,
        "trace_drift_fold": 1.25,
    }
    trait = diagnostic["traits"]["trait1"]
    assert trait["sum_beta_squared"] == pytest.approx(0.0573)
    assert trait["raw_genetic_variance"] == pytest.approx(0.0882)
    assert trait["effect_energy_ratio"] == pytest.approx(0.0573 / 0.0882)
    assert trait["max_effect_slab_sd"] == pytest.approx(4.1)
    assert trait["trace_first_quarter_mean"] == pytest.approx(0.0882)
    assert trait["trace_last_quarter_mean"] == pytest.approx(0.0882)
    assert trait["trace_drift_fold"] == pytest.approx(1.0)
    assert not any(trait["flags"].values())


def test_divergence_warning_catches_cancelling_effects():
    """The failure this exists for: h2 and p both looked fine.

    The real fit reported h2 0.6732 and a causal fraction of 0.00076 -- inside
    every bound, so ``_warn_if_implausible_fit`` stayed silent -- while
    sum(beta^2) was 171.8, a ratio of 255.
    """
    m = bivariate._DIAGNOSTIC_MIN_VARIANTS
    beta1 = np.full(m, np.sqrt(171.8 / m))
    with pytest.warns(RuntimeWarning, match="cancelling through LD"):
        diagnostic = bivariate._warn_if_fit_diverged(**_diverged_args(
            beta1=beta1, raw_h2=(0.6732, 0.0706), sigma_diag=(1.0, 1.0)))
    assert diagnostic["flagged"] is True
    assert diagnostic["traits"]["trait1"]["flags"] == {
        "nonpositive_genetic_variance": False,
        "effect_energy_ratio": True,
        "max_effect_slab_sd": False,
        "trace_drift": False,
    }


def test_divergence_warning_structures_nonpositive_genetic_variance():
    beta1 = np.zeros(bivariate._DIAGNOSTIC_MIN_VARIANTS)
    with pytest.warns(RuntimeWarning, match=(
            "non-positive sampled genetic variance.*h2 and rg.*not valid")):
        diagnostic = bivariate._warn_if_fit_diverged(**_diverged_args(
            beta1=beta1, raw_h2=(-0.01, 0.0706)))
    trait = diagnostic["traits"]["trait1"]
    assert diagnostic["flagged"] is True
    assert trait["raw_genetic_variance"] == pytest.approx(-0.01)
    assert trait["effect_energy_ratio"] is None
    assert trait["flags"] == {
        "nonpositive_genetic_variance": True,
        "effect_energy_ratio": False,
        "max_effect_slab_sd": False,
        "trace_drift": False,
    }


def test_divergence_warning_catches_effects_beyond_the_fitted_slab():
    """A posterior mean 103 slab SDs out contradicts the fit's own prior."""
    beta1 = np.zeros(bivariate._DIAGNOSTIC_MIN_VARIANTS)
    beta1[0] = 3.1929
    with pytest.warns(RuntimeWarning, match="times the per-causal effect SD"):
        diagnostic = bivariate._warn_if_fit_diverged(**_diverged_args(
            beta1=beta1, raw_h2=(1e6, 0.0706),    # keep the ratio arm quiet
            sigma_diag=((3.1929 / 103.0) ** 2, 1.0)))
    assert diagnostic["traits"]["trait1"]["flags"][
        "max_effect_slab_sd"] is True


def test_divergence_warning_catches_a_trace_that_never_settled():
    """Post-burn-in drift: the real fit's gvar1 rose by a factor of 1.63."""
    rising = np.column_stack([np.linspace(0.42, 0.82, 80),
                              np.full(80, 0.01), np.full(80, 0.0706)])
    with pytest.warns(RuntimeWarning, match="rose .* had not settled"):
        diagnostic = bivariate._warn_if_fit_diverged(**_diverged_args(
            genetic_samples=rising))
    trait = diagnostic["traits"]["trait1"]
    assert trait["flags"]["trace_drift"] is True
    assert trait["trace_direction"] == "rising"
    assert trait["trace_drift_fold"] == pytest.approx(
        trait["trace_last_quarter_mean"]
        / trait["trace_first_quarter_mean"])


def test_divergence_warning_catches_a_collapsing_trace():
    falling = np.column_stack([np.linspace(0.82, 0.42, 80),
                               np.full(80, 0.01), np.full(80, 0.0706)])
    with pytest.warns(RuntimeWarning, match="fell .* had not settled"):
        diagnostic = bivariate._warn_if_fit_diverged(**_diverged_args(
            genetic_samples=falling))
    trait = diagnostic["traits"]["trait1"]
    assert trait["flags"]["trace_drift"] is True
    assert trait["trace_direction"] == "falling"
    assert trait["trace_drift_fold"] == pytest.approx(
        trait["trace_first_quarter_mean"]
        / trait["trace_last_quarter_mean"])


def test_divergence_warning_is_silent_on_small_panels():
    """Same ratios, too few variants to mean anything."""
    m = bivariate._DIAGNOSTIC_MIN_VARIANTS - 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        diagnostic = bivariate._warn_if_fit_diverged(
            beta1=np.full(m, np.sqrt(171.8 / m)), beta2=np.zeros(m),
            raw_h2=(0.6732, 0.0706), sigma_diag=(1e-6, 1.0),
            genetic_samples=None, m=m)
    assert diagnostic["evaluated"] is False
    assert diagnostic["flagged"] is False
    assert diagnostic["traits"]["trait1"]["effect_energy_ratio"] > 10
    assert not any(diagnostic["traits"]["trait1"]["flags"].values())


def test_fit_reports_every_sweep_and_changes_nothing():
    """Progress is a side channel: the chain must come out bit-identical."""
    blocks, chols, idxs = _blocks(4, 100, seed=3)
    m, k, n = 400, 100, 20_000
    rng = np.random.default_rng(5)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.05, h2=(0.4, 0.4), rg=0.5, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, n, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, n, k, rng)
    kw = dict(burn_in=8, num_iter=12, seed=4)
    events = []
    loud = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, n, n,
                                         progress=events.append, **kw)
    quiet = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, n, n, **kw)
    assert [e["done"] for e in events] == list(range(1, 21))
    assert [e["phase"] for e in events] == ["burn-in"] * 8 + ["sampling"] * 12
    assert {e["total"] for e in events} == {20}
    assert {e["unit"] for e in events} == {"sweep"}
    assert (loud.rg, loud.p) == (quiet.rg, quiet.p)
    assert loud.h2 == quiet.h2
    assert np.array_equal(loud.beta1_est, quiet.beta1_est)
    assert np.array_equal(loud.beta2_est, quiet.beta2_est)


def test_fit_rejects_a_non_callable_progress():
    blocks, chols, idxs = _blocks(2, 60, seed=0)
    m, k, n = 120, 60, 10_000
    rng = np.random.default_rng(0)
    b1, b2 = _sim(blocks, chols, idxs, m, p=0.1, h2=(0.3, 0.3), rg=0.3, rng=rng)
    bh1 = _sumstats(blocks, chols, idxs, b1, n, k, rng)
    bh2 = _sumstats(blocks, chols, idxs, b2, n, k, rng)
    with pytest.raises(TypeError, match="callable"):
        ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, n, n, burn_in=2,
                                      num_iter=2, progress=object())


def test_tol_warns_when_the_schedule_cannot_stop_early():
    with pytest.warns(RuntimeWarning, match="cannot stop this run early"):
        ldpred3_auto_bivariate(
            np.eye(4), np.full(4, 0.02), np.full(4, 0.02), 1000, 1000,
            ld_int8=False, burn_in=0, num_iter=40, check_every=50, tol=1e-3,
            h2_cap=(0.5, 0.5), seed=1)


def test_divergence_warning_catches_a_sign_crossing_trace():
    crossing = np.column_stack([
        np.concatenate([np.full(40, 0.4), np.full(40, -0.3)]),
        np.full(80, 0.01), np.full(80, 0.0706)])
    with pytest.warns(RuntimeWarning, match="changed sign"):
        diagnostic = bivariate._warn_if_fit_diverged(**_diverged_args(
            genetic_samples=crossing))
    trait = diagnostic["traits"]["trait1"]
    assert trait["flags"]["trace_drift"] is True
    assert trait["trace_direction"] == "sign-crossing"


def test_mixer_rho_beta_ratio_of_means_is_not_mean_of_ratios():
    from bipred.bivariate import BivariateResult
    sigma_samples = np.array([[1.0, 1.0, 0.2], [4.0, 1.0, 0.2]])
    s1, s2, s12 = sigma_samples.mean(axis=0)
    ratio_of_means = s12 / np.sqrt(s1 * s2)
    mean_of_ratios = float(np.mean(
        sigma_samples[:, 2] / np.sqrt(sigma_samples[:, 0] * sigma_samples[:, 1])))
    assert abs(ratio_of_means - mean_of_ratios) > 1e-6
    res = BivariateResult(
        beta1_est=np.zeros(4), beta2_est=np.zeros(4),
        h2=(0.1, 0.1), rg=0.0, p=0.1,
        sigma=np.array([[s1, s12], [s12, s2]]),
        pi=np.array([0.7, 0.1, 0.1, 0.1]),
        pi_samples=np.tile([0.7, 0.1, 0.1, 0.1], (2, 1)),
        sigma_samples=sigma_samples,
    )
    assert res.mixer["rho_beta"] == pytest.approx(ratio_of_means)
    assert res.mixer_iterate_summary()["rho_beta"]["mean"] == pytest.approx(
        mean_of_ratios)


def test_nonfinite_quadratics_are_rejected_as_divergence():
    with pytest.raises(FloatingPointError, match="not positive"):
        bivariate._check_fit_is_finite(
            (np.nan, 0.1, 0.1), np.ones(4), np.ones(4))
    with pytest.raises(FloatingPointError, match="posterior-mean"):
        bivariate._check_fit_is_finite(
            (0.1, 0.0, 0.1), np.array([np.inf, 0.0]), np.zeros(2))


def test_per_variant_n_memo_hit_path_stays_finite():
    """Consecutive equal n_eff entries reuse _bivar_const; mixed runs hit both."""
    beta_hat = np.full(8, 0.03)
    n_runs = np.array([500.0, 500.0, 500.0, 2000.0, 2000.0, 4000.0, 4000.0, 4000.0])
    kwargs = dict(ld_int8=False, h2_init=0.1, p_init=0.5, burn_in=0,
                  num_iter=2, h2_cap=(0.2, 0.2), seed=1)
    result = ldpred3_auto_bivariate(
        np.eye(8), beta_hat, beta_hat, n_runs, n_runs, cross_corr=0.3, **kwargs)
    assert np.all(np.isfinite(result.beta1_est))
    assert np.isfinite(result.rg)


def test_cross_corr_reduces_ld_structured_sampling_noise():
    """Supplying the true sampling correlation recovers rg nearer the true 0."""
    k, n, rho = 60, 8_000, 0.75
    pos = np.arange(k)
    R = (0.6 ** np.abs(pos[:, None] - pos[None, :])).astype(np.float64)
    rng = np.random.default_rng(4)
    L = np.linalg.cholesky(R + 1e-8 * np.eye(k))
    e1 = L @ rng.standard_normal(k) / np.sqrt(n)
    e2 = rho * e1 + np.sqrt(1.0 - rho * rho) * (L @ rng.standard_normal(k) / np.sqrt(n))
    kwargs = dict(ld_int8=False, h2_init=0.05, p_init=0.2, burn_in=15,
                  num_iter=25, seed=2, h2_cap=(0.4, 0.4))
    ignore = ldpred3_auto_bivariate(R.astype(np.float32), e1, e2, n, n,
                                    cross_corr=0.0, **kwargs)
    corrected = ldpred3_auto_bivariate(R.astype(np.float32), e1, e2, n, n,
                                       cross_corr=rho, **kwargs)
    # True genetic rg is 0: only correlated sampling errors were planted.
    assert abs(corrected.rg) < abs(ignore.rg)
    assert abs(corrected.rg) < 0.15


def test_structurally_indefinite_correlation_is_rejected():
    """Entries in [-1, 1] are not enough; a 3x3 can still have min eig ~-1."""
    R = np.array([[1.0, 0.9, -0.9],
                  [0.9, 1.0, 0.9],
                  [-0.9, 0.9, 1.0]], dtype=np.float32)
    assert float(np.linalg.eigvalsh(R.astype(np.float64)).min()) < -0.5
    beta = np.full(3, 0.02)
    with pytest.raises(ValueError, match="structurally indefinite"):
        ldpred3_auto_bivariate(
            R, beta, beta, 10_000, 10_000, burn_in=2, num_iter=2, seed=0)


def test_d8_block_is_not_rejected_for_quantization_indefiniteness():
    """Ordinary D8 rounding is expected to be slightly indefinite."""
    rng = np.random.default_rng(7)
    k, n = 410, 40
    X = rng.standard_normal((n, k))
    R = X.T @ X
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    Q = np.rint(np.clip(R, -1.0, 1.0) * 127).astype(np.int8)
    deq = Q.astype(np.float64) / 127.0
    assert float(np.linalg.eigvalsh(deq).min()) < -0.05
    beta = np.full(k, 0.01)
    ldpred3_auto_bivariate(
        Q, beta, beta, 10_000, 10_000, burn_in=2, num_iter=2, seed=0,
        ld_int8=False)


def test_large_float_block_rejects_an_embedded_indefinite_3x3():
    """Random quadratic probes used to miss a localized defect above k=1024."""
    k = 1025
    R = np.eye(k, dtype=np.float32)
    R[:3, :3] = np.array([[1.0, 0.9, -0.9],
                          [0.9, 1.0, 0.9],
                          [-0.9, 0.9, 1.0]], dtype=np.float32)
    beta = np.full(k, 0.01)
    with pytest.raises(ValueError, match="structurally indefinite"):
        ldpred3_auto_bivariate(
            R, beta, beta, 10_000, 10_000, burn_in=1, num_iter=1, seed=0,
            ld_int8=False)


def _random_corr(k, rng):
    A = rng.normal(size=(k, k))
    C = A @ A.T / k
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def _oracle_bivar_sweep(R, bh1, bh2, n1, n2, curr1, curr2, unif, z1, z2,
                        lpi, s1, s2, s12, cross_corr):
    """One sweep by textbook Gaussian conditioning, written from the model
    rather than from ``_bivar_const``'s closed forms. ``R @ beta`` is
    recomputed from scratch at every variant, so the incremental residual
    update in the kernel under test is also checked."""
    k = len(bh1)
    c1, c2 = curr1.copy(), curr2.copy()
    rbsum1, rbsum2 = np.zeros(k), np.zeros(k)
    c10 = c01 = c11 = 0
    sum1sq = sum2sq = sum12 = 0.0
    nn1, nn2 = np.asarray(n1, float), np.asarray(n2, float)
    lpi = np.asarray(lpi, float)
    slabs = [np.zeros((2, 2)),
             np.array([[s1, 0.0], [0.0, 0.0]]),
             np.array([[0.0, 0.0], [0.0, s2]]),
             np.array([[s1, s12], [s12, s2]])]
    for j in range(k):
        E = np.array(
            [[1.0 / nn1[j], cross_corr / np.sqrt(nn1[j] * nn2[j])],
             [cross_corr / np.sqrt(nn1[j] * nn2[j]), 1.0 / nn2[j]]])
        d = np.array([bh1[j] - (R @ c1)[j] + c1[j],
                      bh2[j] - (R @ c2)[j] + c2[j]])
        ws, mus, covs = [], [], []
        for S in slabs:
            C = E + S
            _, ldet = np.linalg.slogdet(C)
            ws.append(-0.5 * ldet - 0.5 * float(d @ np.linalg.solve(C, d)))
            if S.any():
                G = S @ np.linalg.inv(C)
                mus.append(G @ d)
                covs.append(S - G @ S)
            else:
                mus.append(np.zeros(2))
                covs.append(np.zeros((2, 2)))
        w = lpi + np.asarray(ws)
        w -= w.max()
        pr = np.exp(w)
        pr /= pr.sum()
        rbsum1[j] += pr[1] * mus[1][0] + pr[3] * mus[3][0]
        rbsum2[j] += pr[2] * mus[2][1] + pr[3] * mus[3][1]
        cum = np.cumsum(pr)
        u = unif[j]
        if u < cum[0]:
            st = 0
        elif u < cum[1]:
            st = 1
        elif u < cum[2]:
            st = 2
        else:
            st = 3
        if st == 0:
            new1 = new2 = 0.0
        elif st == 1:
            new1 = mus[1][0] + np.sqrt(max(covs[1][0, 0], 0.0)) * z1[j]
            new2 = 0.0
        elif st == 2:
            new1 = 0.0
            new2 = mus[2][1] + np.sqrt(max(covs[2][1, 1], 0.0)) * z2[j]
        else:
            L = np.linalg.cholesky(covs[3])
            new1 = mus[3][0] + L[0, 0] * z1[j]
            new2 = mus[3][1] + L[1, 0] * z1[j] + L[1, 1] * z2[j]
        if st == 1:
            c10 += 1
            sum1sq += new1 * new1
        elif st == 2:
            c01 += 1
            sum2sq += new2 * new2
        elif st == 3:
            c11 += 1
            sum1sq += new1 * new1
            sum2sq += new2 * new2
            sum12 += new1 * new2
        c1[j], c2[j] = new1, new2
    return ((c10, c01, c11, sum1sq, sum2sq, sum12,
             float(c1 @ (R @ c1)), float(c1 @ (R @ c2)), float(c2 @ (R @ c2))),
            c1, c2, rbsum1, rbsum2)


def _run_sweep_vs_oracle(k, seed, cross_corr, per_variant_n, quantise):
    rng = np.random.default_rng(seed)
    R = _random_corr(k, rng)
    corr = (R * 127.0).astype(np.int8) if quantise else R
    scale = 1.0 / 127.0 if quantise else 1.0
    R_eff = corr * scale
    bh1, bh2 = rng.normal(0, 0.05, k), rng.normal(0, 0.05, k)
    if per_variant_n:
        n1, n2 = rng.uniform(2e4, 9e4, k), rng.uniform(1e4, 6e4, k)
        n_const = False
    else:
        n1, n2 = np.full(k, 60000.0), np.full(k, 45000.0)
        n_const = True
    s1, s2, s12 = 2e-4, 3e-4, 4e-5
    lpi = np.log(np.array([0.90, 0.04, 0.03, 0.03]))
    unif, z1, z2 = rng.random(k), rng.standard_normal(k), rng.standard_normal(k)
    init1, init2 = rng.normal(0, 0.01, k), rng.normal(0, 0.01, k)

    exp, oc1, oc2, ors1, ors2 = _oracle_bivar_sweep(
        R_eff, bh1, bh2, n1, n2, init1.copy(), init2.copy(),
        unif, z1, z2, lpi, s1, s2, s12, cross_corr)

    def run(kernel):
        c1, c2 = init1.copy(), init2.copy()
        rb1, rb2 = np.zeros(k), np.zeros(k)
        rs1, rs2 = np.zeros(k), np.zeros(k)
        got = kernel(corr, bh1, bh2, n1, n2, c1, c2, rb1, rb2, rs1, rs2,
                     unif, z1, z2, lpi[0], lpi[1], lpi[2], lpi[3],
                     s1, s2, s12, cross_corr, scale, n_const, True)
        return got, c1, c2, rb1, rb2, rs1, rs2

    got, c1, c2, rb1, rb2, rs1, rs2 = run(bivariate._bivar_one_sweep)
    assert tuple(got[:3]) == tuple(exp[:3])
    for label, g, e in zip(
            ("sum1sq", "sum2sq", "sum12", "gv11", "gv12", "gv22"),
            got[3:9], exp[3:9]):
        np.testing.assert_allclose(g, e, rtol=1e-9, atol=1e-12, err_msg=label)
    np.testing.assert_allclose(c1, oc1, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(c2, oc2, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rs1, ors1, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rs2, ors2, rtol=1e-9, atol=1e-12)
    # The sweep's incremental R@beta equals a from-scratch recompute.
    np.testing.assert_allclose(rb1, R_eff @ c1, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rb2, R_eff @ c2, rtol=1e-9, atol=1e-12)
    # The JIT kernel agrees with the Python kernel.
    got_j, c1_j, c2_j, _, _, _, _ = run(bivariate._bivar_one_sweep_jit)
    assert tuple(got_j[:3]) == tuple(got[:3])
    np.testing.assert_allclose(got_j[3:9], got[3:9], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(c1_j, c1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(c2_j, c2, rtol=1e-10, atol=1e-12)


def test_bivar_sweep_matches_independent_gaussian_conditioning():
    """The four-state conditional likelihood, posterior means, and draws agree
    with an independent Gaussian-conditioning oracle. Every other check on the
    shared algebra (low-rank vs dense, ncores, vectorised rg, per-variant-N
    memo) reuses ``_bivar_const``, so only this test would catch a systematic
    error in it."""
    for seed, cc, per_variant, quantise in (
            (7, 0.0, False, False),
            (11, 0.3, False, False),
            (13, -0.45, False, False),
            (17, 0.2, True, False),
            (19, 0.15, True, True),
            (23, -0.2, True, False)):
        _run_sweep_vs_oracle(40, seed, cc, per_variant, quantise)
