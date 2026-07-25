"""Prototype + benchmark: a **hierarchical per-region effect covariance** for
regional genetic correlation.

STATUS: research prototype. Not a shipped feature.

The problem (RESULTS_REGIONAL.md section 5). bipred's sampler carries ONE 2x2
effect covariance Sigma for the whole genome, so every per-SNP posterior borrows
across traits in proportion to the *genome-wide* effect correlation. Regional r_g
read out of that model is therefore pulled toward the genome-wide value: in the
regional benchmark, null regions read 0.037 against a realized 0.003 and
r_g = 0.8 regions read 0.725 against 0.796. The oracle-cross_corr arm shows the
same compression, so this is NOT an overlap artifact -- correcting cross_corr
fixes contamination, not shrinkage. The bias is also flat in region size (it is
applied per SNP in the same direction for every SNP in a region, so averaging
more variants cancels noise but not the offset).

The naive fix -- an independent Sigma per region -- trades bias for variance: a
2x2 covariance estimated from ~100 shrunken effects is very noisy. The principled
fix is partial pooling.

Model. Give each region its own Sigma_r with a hierarchical prior centred on a
genome-wide Psi:

    Sigma_r ~ InverseWishart(nu, (nu - 3) * Psi)        E[Sigma_r] = Psi
    Sigma_r | beta_r ~ IW(nu + k_r, (nu - 3) * Psi + S_r)

where ``S_r`` is the region's 2x2 effect scatter and ``k_r`` its variant count.
``nu`` is the pooling strength and is **estimated** each sweep from a grid, by
its exact conditional given {Sigma_r} and Psi (flat prior over the grid). Psi is
updated empirical-Bayes as the mean of the current Sigma_r draws.

This makes the three models one code path with different ``nu`` handling, so the
comparison isolates exactly one thing:

    global     nu -> infinity   all regions share one Sigma (bipred today)
    perregion  nu = 4           minimal pooling, independent Sigma_r
    hier       nu estimated     partial pooling (the proposal)

All three estimate ``cross_corr`` jointly (established as necessary in
RESULTS.md), so overlap handling is held constant.

Run:  OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/bench_hier_sigma.py [main|size] [reps]
Writes bench_hier_<grid>.csv.
"""
import csv
import os
import sys
import time

import numpy as np
from scipy.special import gammaln

from bench_cross_corr import HAVE_NUMBA, _seed_numba, make_ld, make_sumstats, njit
from bench_regional_rg import (REGION_RGS, make_regional_effects,  # noqa: E402
                               regional_realized_rg)

H1, H2 = 0.5, 0.5
N_DEF = 50_000.0
CC_TRUE = 0.4
NB_DEF, K_DEF = 60, 100
N_SWEEP, BURN = 900, 300
REPS = 10
MODELS = ("global", "perregion", "hier", "rho")
RHO_GRID = np.linspace(-0.95, 0.95, 191)
NU_GRID = np.array([4.0, 6.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0, 1e4, 1e6])


@njit
def _sweep_hier(R, bhat1, bhat2, b1, b2, rb1, rb2, ve, lv, s11, s12, s22):
    """One sweep with a PER-REGION posterior (``ve``/``lv`` indexed by block).

    Also returns each region's 2x2 effect scatter, which is the sufficient
    statistic for that region's Sigma_r update."""
    nb, k, _ = R.shape
    for i in range(nb):
        a00 = ve[i, 0, 0]; a01 = ve[i, 0, 1]
        a10 = ve[i, 1, 0]; a11 = ve[i, 1, 1]
        l00 = lv[i, 0, 0]; l10 = lv[i, 1, 0]; l11 = lv[i, 1, 1]
        t11 = 0.0; t12 = 0.0; t22 = 0.0
        for j in range(k):
            d1 = bhat1[i, j] - rb1[i, j] + b1[i, j]
            d2 = bhat2[i, j] - rb2[i, j] + b2[i, j]
            m1 = a00 * d1 + a01 * d2
            m2 = a10 * d1 + a11 * d2
            z1 = np.random.normal(); z2 = np.random.normal()
            n1 = m1 + l00 * z1
            n2 = m2 + l10 * z1 + l11 * z2
            db1 = n1 - b1[i, j]; db2 = n2 - b2[i, j]
            for t in range(k):
                rb1[i, t] += R[i, t, j] * db1
                rb2[i, t] += R[i, t, j] * db2
            b1[i, j] = n1; b2[i, j] = n2
            t11 += n1 * n1; t22 += n2 * n2; t12 += n1 * n2
        s11[i] = t11; s12[i] = t12; s22[i] = t22


def _iw_draw(df, scale, rng):
    """2x2 inverse-Wishart draw via Bartlett on the inverse."""
    L = np.linalg.cholesky(np.linalg.inv(scale))
    A = np.zeros((2, 2))
    A[0, 0] = np.sqrt(rng.chisquare(df))
    A[1, 1] = np.sqrt(rng.chisquare(df - 1.0))
    A[1, 0] = rng.standard_normal()
    return np.linalg.inv(L @ A @ A.T @ L.T)


def _log_iw_pdf(S, df, scale):
    """log IW(S; df, scale) for 2x2, up to terms constant in df/scale."""
    sgn_l, logdet_l = np.linalg.slogdet(scale)
    sgn_s, logdet_s = np.linalg.slogdet(S)
    if sgn_l <= 0 or sgn_s <= 0:
        return -np.inf
    tr = np.trace(scale @ np.linalg.inv(S))
    lg2 = 0.5 * np.log(np.pi) + gammaln(df / 2.0) + gammaln(df / 2.0 - 0.5)
    return (0.5 * df * logdet_l - 0.5 * (df + 3.0) * logdet_s
            - 0.5 * tr - df * np.log(2.0) - lg2)


def run_hier(R, Linv, bh1, bh2, n1, n2, model, seed,
             n_sweep=N_SWEEP, burn=BURN):
    """Bivariate Gibbs with per-region Sigma_r and a global estimated cross_corr.

    ``model`` selects the pooling strength only."""
    nb, k, _ = R.shape
    m = nb * k
    rng = np.random.default_rng(seed)
    if HAVE_NUMBA:
        _seed_numba(seed + 1)
    else:                                                  # pragma: no cover
        np.random.seed(seed + 1)
    b1 = np.zeros((nb, k)); b2 = np.zeros((nb, k))
    rb1 = np.zeros((nb, k)); rb2 = np.zeros((nb, k))
    s11 = np.zeros(nb); s12 = np.zeros(nb); s22 = np.zeros(nb)

    Psi = np.array([[H1 / m, 0.0], [0.0, H2 / m]])
    Sig = np.repeat(Psi[None, :, :], nb, axis=0)
    if model.startswith("fix"):                 # fixed pooling strength
        nu = float(model.split(":")[1])
    else:
        nu = 1e6 if model == "global" else (4.0 if model == "perregion" else 50.0)
    cc = 0.0
    sd1, sd2 = 1.0 / np.sqrt(n1), 1.0 / np.sqrt(n2)
    cc_grid = np.linspace(-0.95, 0.95, 191)

    q11 = np.zeros(nb); q22 = np.zeros(nb); q12 = np.zeros(nb)
    acc_b1 = np.zeros((nb, k)); acc_b2 = np.zeros((nb, k))
    acc_rb1 = np.zeros((nb, k)); acc_rb2 = np.zeros((nb, k))
    nu_keep, cc_keep, kept = [], [], 0

    for sweep in range(n_sweep):
        E = np.array([[1.0 / n1, cc * sd1 * sd2], [cc * sd1 * sd2, 1.0 / n2]])
        Einv = np.linalg.inv(E)
        ve = np.empty((nb, 2, 2)); lv = np.zeros((nb, 2, 2))
        for i in range(nb):
            V = np.linalg.inv(np.linalg.inv(Sig[i]) + Einv)
            ve[i] = V @ Einv
            lv[i] = np.linalg.cholesky(V)
        _sweep_hier(R, bh1, bh2, b1, b2, rb1, rb2, ve, lv, s11, s12, s22)

        # ---- per-region Sigma_r | beta_r, nu, Psi -------------------------- #
        prior_scale = (nu - 3.0) * Psi
        if model == "rho" or model.startswith("rhoz"):
            tau = float(model.split(":")[1]) if model.startswith("rhoz") else None
            # Per-region CORRELATION only: the per-trait scale is held at the
            # global estimate and just rho_r varies by region. Letting the whole
            # 2x2 Sigma_r float (the `hier`/`perregion` arms) destabilises the
            # sampler -- a region's sampled scatter feeds back into its own
            # posterior on the covariance scale, and no pooling strength nu
            # rescues it (see the nusweep grid). Reducing each region to ONE
            # bounded parameter removes that feedback path, and the conditional
            # has the same closed form as the cross_corr update.
            S = np.array([[s11.sum(), s12.sum()], [s12.sum(), s22.sum()]])
            Sg = _iw_draw(4.0 + m, np.eye(2) * 1e-12 + S, rng)
            v1 = max(Sg[0, 0], 1e-300); v2 = max(Sg[1, 1], 1e-300)
            sd_1, sd_2 = np.sqrt(v1), np.sqrt(v2)
            for i in range(nb):
                t11 = s11[i] / v1
                t22 = s22[i] / v2
                t12 = s12[i] / (sd_1 * sd_2)
                llr = (-0.5 * k * np.log(1 - RHO_GRID ** 2)
                       - (t11 - 2 * RHO_GRID * t12 + t22)
                       / (2 * (1 - RHO_GRID ** 2)))
                if tau is not None:
                    # Partial pooling on rho ALONE: Fisher-z prior centred on the
                    # genome-wide rho. One bounded parameter per region, unlike
                    # the three-parameter Sigma_r pooling that failed.
                    zbar = np.arctanh(np.clip(
                        Sg[0, 1] / np.sqrt(v1 * v2), -0.999, 0.999))
                    llr = llr - 0.5 * ((np.arctanh(RHO_GRID) - zbar) / tau) ** 2
                llr -= llr.max()
                wr = np.exp(llr)
                rho_i = float(rng.choice(RHO_GRID, p=wr / wr.sum()))
                off = rho_i * sd_1 * sd_2
                Sig[i] = np.array([[v1, off], [off, v2]])
            Psi = Sg
        elif model == "global" or (model.startswith("fix") and nu >= 1e5):                 # full pooling == one shared Sigma
            S = np.array([[s11.sum(), s12.sum()], [s12.sum(), s22.sum()]])
            Sg = _iw_draw(4.0 + m, np.eye(2) * 1e-12 + S, rng)
            Sig[:] = Sg
            Psi = Sg
        else:
            for i in range(nb):
                Si = np.array([[s11[i], s12[i]], [s12[i], s22[i]]])
                Sig[i] = _iw_draw(nu + k, prior_scale + Si + 1e-14 * np.eye(2),
                                  rng)
            Psi = Sig.mean(axis=0)
            # ---- pooling strength nu | {Sigma_r}, Psi (exact 1-D grid) ----- #
            if model == "hier":
                ll = np.empty(NU_GRID.size)
                for gi, g in enumerate(NU_GRID):
                    sc = (g - 3.0) * Psi
                    ll[gi] = sum(_log_iw_pdf(Sig[i], g, sc) for i in range(nb))
                ll -= ll.max()
                w = np.exp(ll)
                if np.isfinite(w).all() and w.sum() > 0:
                    nu = float(rng.choice(NU_GRID, p=w / w.sum()))

        # ---- global cross_corr (whitened marginal residuals) --------------- #
        w1 = np.einsum("ikl,il->ik", Linv, bh1 - rb1) * np.sqrt(n1)
        w2 = np.einsum("ikl,il->ik", Linv, bh2 - rb2) * np.sqrt(n2)
        S11 = float((w1 * w1).sum()); S22 = float((w2 * w2).sum())
        S12 = float((w1 * w2).sum())
        llc = (-0.5 * m * np.log(1 - cc_grid ** 2)
               - (S11 - 2 * cc_grid * S12 + S22) / (2 * (1 - cc_grid ** 2)))
        llc -= llc.max()
        wc = np.exp(llc)
        cc = float(rng.choice(cc_grid, p=wc / wc.sum()))

        if sweep >= burn:
            q11 += np.einsum("ik,ik->i", b1, rb1)
            q22 += np.einsum("ik,ik->i", b2, rb2)
            q12 += np.einsum("ik,ik->i", b1, rb2)
            acc_b1 += b1; acc_b2 += b2
            acc_rb1 += rb1; acc_rb2 += rb2
            nu_keep.append(nu); cc_keep.append(cc); kept += 1

    pm1, pm2 = acc_b1 / kept, acc_b2 / kept
    prb1, prb2 = acc_rb1 / kept, acc_rb2 / kept
    p11 = np.einsum("ik,ik->i", pm1, prb1)
    p22 = np.einsum("ik,ik->i", pm2, prb2)
    p12 = np.einsum("ik,ik->i", pm1, prb2)
    postmean = p12 / np.sqrt(np.maximum(p11 * p22, 1e-300))
    return (np.clip(postmean, -1.5, 1.5), float(np.mean(cc_keep)),
            float(np.median(nu_keep)))


def one_cell(nb, k, n1, n2, reps, ld_seed=7):
    R, L, Linv, ell = make_ld(nb, k, ld_seed)
    rg_by_block = np.array([REGION_RGS[i % len(REGION_RGS)] for i in range(nb)])
    out = {mo: {"est": [], "cc": [], "nu": []} for mo in MODELS}
    realized = []
    for rep in range(reps):
        beta1, beta2 = make_regional_effects(R, rg_by_block, H1, H2,
                                             1234 + 977 * rep)
        realized.append(regional_realized_rg(R, beta1, beta2))
        bh1, bh2 = make_sumstats(R, L, beta1, beta2, CC_TRUE, n1, n2,
                                 5000 + 31 * rep)
        for mo in MODELS:
            est, cc, nu = run_hier(R, Linv, bh1, bh2, n1, n2, mo, seed=rep)
            out[mo]["est"].append(est)
            out[mo]["cc"].append(cc)
            out[mo]["nu"].append(nu)
    return out, np.array(realized), rg_by_block


def summarise(out, realized, rg_by_block, k, n1, reps):
    rows = []
    for mo in MODELS:
        E = np.array(out[mo]["est"])
        lo = E[:, rg_by_block == REGION_RGS[0]].ravel()
        hi = E[:, rg_by_block == REGION_RGS[-1]].ravel()
        pooled = np.sqrt((np.nanvar(lo, ddof=1) + np.nanvar(hi, ddof=1)) / 2)
        d = float((np.nanmean(hi) - np.nanmean(lo)) / pooled)
        for cls in REGION_RGS:
            sel = rg_by_block == cls
            err = (E[:, sel] - realized[:, sel]).ravel()
            err = err[np.isfinite(err)]
            rows.append({
                "model": mo, "region_rg": cls, "N": int(n1),
                "region_size": k, "n_regions": int(sel.sum()), "reps": reps,
                "est_mean": round(float(np.nanmean(E[:, sel])), 4),
                "est_sd": round(float(np.nanstd(E[:, sel], ddof=1)), 4),
                "realized_mean": round(float(np.nanmean(realized[:, sel])), 4),
                "bias": round(float(err.mean()), 4),
                "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
                "separation_d": round(d, 3),
                "cc_mean": round(float(np.mean(out[mo]["cc"])), 4),
                "nu_median": round(float(np.median(out[mo]["nu"])), 1),
            })
    return rows


def write(rows, name):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, f"bench_hier_{name}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    print(f"wrote {os.path.basename(path)} ({len(rows)} rows)")


def show(rows):
    print(f"{'model':10s} {'rg':>4s} {'est_mean':>9s} {'realized':>9s} "
          f"{'bias':>8s} {'rmse':>7s} {'sd':>7s} {'sep_d':>6s} {'nu':>8s}")
    for r in rows:
        print(f"{r['model']:10s} {r['region_rg']:>4} {r['est_mean']:>9} "
              f"{r['realized_mean']:>9} {r['bias']:>8} {r['rmse']:>7} "
              f"{r['est_sd']:>7} {r['separation_d']:>6} {r['nu_median']:>8}")


def grid_main(reps=REPS):
    t0 = time.time()
    out, real, rgb = one_cell(NB_DEF, K_DEF, N_DEF, N_DEF, reps)
    print(f"  main cell done in {time.time()-t0:.0f}s")
    return summarise(out, real, rgb, K_DEF, N_DEF, reps)


def grid_nusweep(reps=6):
    """Fixed-nu sweep: does ANY pooling strength beat the global model?

    The `hier` arm estimates nu from the dispersion of the Sigma_r draws, but
    that dispersion contains estimation noise as well as real heterogeneity, so
    it can be driven to too little pooling. This sweep removes nu estimation from
    the question: if some fixed nu clearly beats `global`, the model is sound and
    only the nu update needs fixing; if none does, partial pooling of Sigma is
    the wrong tool here."""
    global MODELS
    keep = MODELS
    MODELS = tuple(f"fix:{v}" for v in (4, 20, 50, 100, 300, 1000, 1000000))
    rows = []
    t0 = time.time()
    out, real, rgb = one_cell(NB_DEF, K_DEF, N_DEF, N_DEF, reps)
    rows += summarise(out, real, rgb, K_DEF, N_DEF, reps)
    print(f"  nu sweep done in {time.time()-t0:.0f}s")
    MODELS = keep
    return rows


def grid_rhotau(reps=6):
    """Partial pooling on rho ALONE, swept over the prior width tau.

    Sigma-pooling failed outright (grid_nusweep). The natural next question is
    whether pooling the single bounded parameter rho_r -- via a Fisher-z prior of
    width tau centred on the genome-wide correlation -- pins the null regions
    without reintroducing that instability. tau is swept rather than estimated,
    so that a null result cannot be blamed on the estimator (the mistake we made
    once already with nu)."""
    global MODELS
    keep = MODELS
    MODELS = ("global", "rho", "rhoz:0.15", "rhoz:0.3", "rhoz:0.6")
    t0 = time.time()
    out, real, rgb = one_cell(NB_DEF, K_DEF, N_DEF, N_DEF, reps)
    rows = summarise(out, real, rgb, K_DEF, N_DEF, reps)
    print(f"  rho-tau sweep done in {time.time()-t0:.0f}s")
    MODELS = keep
    return rows


def grid_size(reps=REPS):
    rows = []
    for nb, k in ((30, 200), (60, 100), (120, 50)):
        t0 = time.time()
        out, real, rgb = one_cell(nb, k, N_DEF, N_DEF, reps)
        rows += summarise(out, real, rgb, k, N_DEF, reps)
        print(f"  region_size={k} done in {time.time()-t0:.0f}s")
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else REPS
    print(f"hierarchical-Sigma regional benchmark grid={which} reps={reps}")
    todo = {"main": [("main", grid_main)], "size": [("size", grid_size)],
            "nusweep": [("nusweep", grid_nusweep)],
            "rhotau": [("rhotau", grid_rhotau)]}[which]
    for name, fn in todo:
        print(f"\n=== grid: {name} ===")
        rows = fn(reps)
        show(rows)
        write(rows, name)


if __name__ == "__main__":
    main()
