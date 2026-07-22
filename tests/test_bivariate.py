"""Bivariate LDpred-auto: rg / h2 recovery and cross-trait borrowing."""

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
    # int8-quantised LD (the default) tracks the exact float32 fit closely, and a
    # block handed in already int8 is detected and consumed as-is -- bit-identical
    # to quantising the float block on the fly.
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
    q8 = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, 60000, 60000, **kw)
    # int8 (default) stays close to the exact float fit -- quantisation error only
    assert abs(q8.rg - flt.rg) < 0.05, (q8.rg, flt.rg)
    assert abs(q8.h2[0] - flt.h2[0]) < 0.05 and abs(q8.h2[1] - flt.h2[1]) < 0.05
    assert np.max(np.abs(q8.beta1_est - flt.beta1_est)) < 0.02

    # pre-quantised int8 blocks (what ldpred3.compute_ld_blocks(quantize=True)
    # emits) are detected by dtype and consumed as-is, so the fit is bit-identical
    # to the default on-the-fly quantisation -- even with ld_int8=False.
    pre = [(np.rint(np.clip(R, -1.0, 1.0) * 127.0).astype(np.int8), ix)
           for (R, ix) in blocks]
    q8_pre = ldpred3_auto_bivariate_blocks(pre, bh1, bh2, 60000, 60000,
                                           ld_int8=False, **kw)
    assert q8_pre.rg == q8.rg
    assert np.array_equal(q8_pre.beta1_est, q8.beta1_est)


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

    with pytest.deprecated_call(match="mixer_posterior"):
        legacy = res.mixer_posterior(level=0.95)
    assert legacy["n_shared"]["ci"] == post["n_shared"]["interval"]
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
    off = ldpred3_auto_bivariate_blocks(ref, bh1, bh2, N, N, burn_in=120,
                                        num_iter=180, seed=1)
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


def test_bivariate_rejects_lowrank_blocks():
    # Compact (low-rank) LD blocks must fail loudly, not crash with a cryptic
    # float() TypeError inside np.ascontiguousarray.
    from ldpred3 import LowRankLD
    rng = np.random.default_rng(0)
    b1 = rng.standard_normal(40) * 0.02
    b2 = rng.standard_normal(40) * 0.02
    blocks = [(LowRankLD(np.ones((40, 1), dtype=np.float32), 40),
               np.arange(40))]
    with pytest.raises(NotImplementedError, match="dense LD"):
        ldpred3_auto_bivariate_blocks(blocks, b1, b2, 10000, 10000,
                                      burn_in=5, num_iter=5,
                                      h2_cap=(0.1, 0.1))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"ld_int8": 1}, "ld_int8.*boolean"),
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


def test_decorrelated_rg_buffers_are_opt_in_and_path_is_used(monkeypatch):
    assert bivariate._effect_sample_buffers(False, 10, 2, 10_000_000) == (None, None)
    s1, s2 = bivariate._effect_sample_buffers(True, 3, 2, 4)
    assert s1.shape == s2.shape == (2, 4)
    assert s1.dtype == s2.dtype == np.float32

    calls = []

    def fake_decorrelated(_blocks, samples1, samples2):
        calls.append((samples1.shape, samples2.shape))
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
    assert calls == [((2, 4), (2, 4))]
    assert res.rg == 0.25


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
