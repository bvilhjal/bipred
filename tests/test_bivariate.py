"""Bivariate LDpred-auto: rg / h2 recovery and cross-trait borrowing."""

import numpy as np

from bipred import ldpred3_auto_bivariate_blocks
from ldpred3 import ldpred3_by_blocks, ldpred3_auto_infer


def _blocks(n_blocks=12, k=200, seed=0):
    rng = np.random.default_rng(seed)
    blocks, chols, idxs = [], [], []
    for b in range(n_blocks):
        rho = rng.uniform(0.0, 0.8)
        d = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
        R = (rho ** d).astype(np.float64)
        blocks.append((R.astype(np.float32), np.arange(b * k, (b + 1) * k)))
        chols.append(np.linalg.cholesky(R + 1e-6 * np.eye(k)))
        idxs.append(np.arange(b * k, (b + 1) * k))
    return blocks, chols, idxs


def _gv(blocks, idxs, a, b):
    return sum(a[ix] @ (blocks[i][0].astype(float) @ b[ix])
               for i, ix in enumerate(idxs))


def _sim(blocks, chols, idxs, m, *, p, h2, rg, rng):
    """Shared-causal bivariate effects scaled to (h2[0], h2[1]) with corr rg."""
    causal = rng.random(m) < p
    nc = causal.sum()
    L = np.linalg.cholesky([[1.0, rg], [rg, 1.0]])
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


def test_mixer_posterior_intervals():
    # mixer_posterior turns the retained (pi, Sigma) draws into a posterior mean
    # + credible interval per overlap quantity; the mean matches the point
    # estimate, the CI brackets it, and under matched LD the interval covers the
    # truth (which the point estimate is calibrated to here).
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
    post = res.mixer_posterior(level=0.95)
    assert set(post) == {"n_causal", "polygenicity", "n_shared", "frac_shared",
                         "rho_beta", "rg_from_overlap", "level"}
    point = res.mixer
    for i in (0, 1):
        entry = post["n_causal"][i]
        lo, hi = entry["ci"]
        assert lo <= entry["mean"] <= hi                       # CI brackets mean
        assert lo <= point["n_causal"][i] <= hi                # and the point est
    for key in ("n_shared", "frac_shared", "rho_beta", "rg_from_overlap"):
        lo, hi = post[key]["ci"]
        assert lo <= post[key]["mean"] <= hi
        assert post[key]["sd"] >= 0.0
    # frac_shared is a probability in [0, 1]
    assert 0.0 <= post["frac_shared"]["ci"][0] <= post["frac_shared"]["ci"][1] <= 1.0


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
        pop.append(R); chol.append(np.linalg.cholesky(R + 1e-6 * np.eye(k)))
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
    with pytest.raises(ValueError, match="partition"):
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


def test_bivariate_rejects_compact_blocks():
    # Compact (low-rank) LD blocks must fail loudly, not crash with a cryptic
    # float() TypeError inside np.ascontiguousarray.
    import pytest
    from ldpred3 import lowrank_ld
    rng = np.random.default_rng(0)
    R = (0.3 ** np.abs(np.subtract.outer(np.arange(40), np.arange(40)))).astype(float)
    b1 = rng.standard_normal(40) * 0.02
    b2 = rng.standard_normal(40) * 0.02
    for conv in (lowrank_ld,):
        blocks = [(conv(R), np.arange(40))]
        with pytest.raises(NotImplementedError, match="dense LD"):
            ldpred3_auto_bivariate_blocks(blocks, b1, b2, 10000, 10000,
                                          burn_in=5, num_iter=5, h2_cap=(0.1, 0.1))
