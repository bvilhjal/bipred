"""Benchmark: estimating the GWAS-overlap noise correlation (``cross_corr``)
inside the bivariate Gibbs sampler, versus the alternatives.

STATUS: research benchmark for the prototype in this directory. Not a shipped
feature; the only `bipred` import is `ldsc_rg`, used to build the competing arm.

The question is NOT "does estimating cross_corr beat ignoring it" (it does, and
that is a weak claim). It is: **does estimating cross_corr in the sampler match
or beat bipred's existing recommended practice** -- deriving cross_corr from the
cross-trait LDSC intercept (`ldsc_rg(...).gcov_intercept`, the quantity
`estimate_sample_overlap` inverts)? So all four arms run the *same* sampler on
the *same* simulated data and differ in exactly one thing, the cross_corr value:

    naive   cross_corr = 0                 (bipred's default when unspecified)
    ldsc    cross_corr = gcov_intercept    (bipred's recommended practice today)
    joint   cross_corr estimated per sweep (the proposed method)
    oracle  cross_corr = the true value    (upper bound; not attainable)

Identity used by the `ldsc` arm: with noise variances 1/N_t and covariance
cross_corr/sqrt(N1 N2), the noise *correlation* is exactly cross_corr, which is
also what the cross-trait LDSC intercept estimates -- so gcov_intercept is
directly comparable to cross_corr, no rescaling.

Metrics, per grid cell over replicates: mean and SD of the estimated r_g (bias
against the true r_g) and, for `joint`, of the estimated cross_corr.

Grids:
  main  r_g x cross_corr at fixed N and m -- recovery, de-biasing, and the
        false-positive check at cross_corr = 0 (does the method invent overlap,
        and what does freeing the parameter cost when there is none?)
  n     N sweep at fixed (r_g, cross_corr) -- overlap noise shrinks relative to
        genetic signal as N grows, so identification should get harder
  m     m sweep -- the LDSC intercept needs many SNPs spanning a wide LD-score
        range, so a small-m win for `joint` would be an artifact, not a result

Run:  OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/bench_cross_corr.py [main|n|m|all]
Writes bench_cross_corr_<grid>.csv. Numba-accelerated; falls back to pure Python.
"""
import csv
import os
import sys
import time

import numpy as np

from bipred import ldsc_rg

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                        # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco(a[0]) if a and callable(a[0]) else deco


# ---- defaults ------------------------------------------------------------ #
NB, K = 60, 100                    # blocks x variants -> m = 6000
N_DEF = 8_000.0                    # per-trait GWAS N
H1, H2 = 0.5, 0.5                  # SNP heritabilities
REPS = 10
N_SWEEP, BURN = 900, 300
ARMS = ("naive", "ldsc", "joint", "oracle")


@njit
def _sweep(R, bhat1, bhat2, b1, b2, rb1, rb2,
           ve00, ve01, ve10, ve11, l00, l10, l11):
    """One Gibbs sweep over all blocks; updates beta and R@beta in place.

    Returns the accumulated 2x2 effect cross-products (for the genetic
    covariance update)."""
    nb, k, _ = R.shape
    s11 = 0.0
    s22 = 0.0
    s12 = 0.0
    for i in range(nb):
        for j in range(k):
            d1 = bhat1[i, j] - rb1[i, j] + b1[i, j]
            d2 = bhat2[i, j] - rb2[i, j] + b2[i, j]
            m1 = ve00 * d1 + ve01 * d2
            m2 = ve10 * d1 + ve11 * d2
            z1 = np.random.normal()
            z2 = np.random.normal()
            n1 = m1 + l00 * z1
            n2 = m2 + l10 * z1 + l11 * z2
            db1 = n1 - b1[i, j]
            db2 = n2 - b2[i, j]
            for t in range(k):
                rb1[i, t] += R[i, t, j] * db1
                rb2[i, t] += R[i, t, j] * db2
            b1[i, j] = n1
            b2[i, j] = n2
            s11 += n1 * n1
            s22 += n2 * n2
            s12 += n1 * n2
    return s11, s12, s22


@njit
def _seed_numba(seed):
    np.random.seed(seed)


def make_ld(nb, k, seed, rho_lo=0.2, rho_hi=0.8):
    """Block-diagonal AR(1) LD, its simulation Cholesky, whitening inverse, and
    LD scores (sum of r^2 within block, including self).

    ``rho_lo``/``rho_hi`` set the per-block AR(1) decay and therefore the spread
    of LD scores. The default (0.2-0.8) gives a NARROW range (ell about 1-4.5),
    which handicaps LDSC: the cross-trait intercept is identified only by
    variation in ell, so a narrow range makes the intercept hard to separate from
    the slope. The `ldwide` grid raises rho_hi to widen ell to a more realistic
    span and re-runs the comparison -- see README.md."""
    r = np.random.default_rng(seed)
    R = np.empty((nb, k, k))
    L = np.empty((nb, k, k))
    Linv = np.empty((nb, k, k))
    d = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    for i in range(nb):
        rho = r.uniform(rho_lo, rho_hi)
        Ri = rho ** d
        R[i] = Ri
        L[i] = np.linalg.cholesky(Ri + 1e-6 * np.eye(k))
        Linv[i] = np.linalg.inv(np.linalg.cholesky(Ri + 1e-8 * np.eye(k)))
    ell = (R ** 2).sum(axis=2).ravel()
    return R, L, Linv, ell


def make_effects(R, rg, h1, h2, seed):
    """Bivariate infinitesimal effects scaled to (h1, h2) with correlation rg."""
    nb, k, _ = R.shape
    r = np.random.default_rng(seed)
    Lg = np.array([[1.0, 0.0], [rg, np.sqrt(max(0.0, 1 - rg * rg))]])
    raw = Lg @ r.standard_normal((2, nb * k))
    b1 = raw[0].reshape(nb, k).copy()
    b2 = raw[1].reshape(nb, k).copy()

    def gv(x, y):
        return float(sum(x[i] @ (R[i] @ y[i]) for i in range(nb)))

    b1 *= np.sqrt(h1 / gv(b1, b1))
    b2 *= np.sqrt(h2 / gv(b2, b2))
    return b1, b2


def make_sumstats(R, L, beta1, beta2, cc, n1, n2, seed):
    """bhat_t = R beta_t + L u_t / sqrt(N_t), corr(u1_j, u2_j) = cc."""
    nb, k, _ = R.shape
    r = np.random.default_rng(seed)
    Ln = np.array([[1.0, 0.0], [cc, np.sqrt(max(0.0, 1 - cc * cc))]])
    bh1 = np.empty((nb, k))
    bh2 = np.empty((nb, k))
    for i in range(nb):
        u = Ln @ r.standard_normal((2, k))
        bh1[i] = R[i] @ beta1[i] + L[i] @ u[0] / np.sqrt(n1)
        bh2[i] = R[i] @ beta2[i] + L[i] @ u[1] / np.sqrt(n2)
    return bh1, bh2


def run_sampler(R, Linv, bh1, bh2, n1, n2, arm, cc_fixed, seed,
                n_sweep=N_SWEEP, burn=BURN, h1=H1, h2=H2):
    """Infinitesimal bivariate LDpred Gibbs. `arm` only controls cross_corr."""
    nb, k, _ = R.shape
    m = nb * k
    r = np.random.default_rng(seed)
    if HAVE_NUMBA:
        _seed_numba(seed + 1)
    else:                                                  # pragma: no cover
        np.random.seed(seed + 1)
    b1 = np.zeros((nb, k))
    b2 = np.zeros((nb, k))
    rb1 = np.zeros((nb, k))
    rb2 = np.zeros((nb, k))
    G = np.array([[h1 / m, 0.0], [0.0, h2 / m]])
    cc = 0.0 if arm == "joint" else float(cc_fixed)
    sd1, sd2 = 1.0 / np.sqrt(n1), 1.0 / np.sqrt(n2)
    grid = np.linspace(-0.95, 0.95, 191)
    keep_cc, keep_G = [], []

    for sweep in range(n_sweep):
        E = np.array([[1.0 / n1, cc * sd1 * sd2], [cc * sd1 * sd2, 1.0 / n2]])
        V = np.linalg.inv(np.linalg.inv(G) + np.linalg.inv(E))
        Lv = np.linalg.cholesky(V)
        VE = V @ np.linalg.inv(E)
        s11, s12, s22 = _sweep(R, bh1, bh2, b1, b2, rb1, rb2,
                               VE[0, 0], VE[0, 1], VE[1, 0], VE[1, 1],
                               Lv[0, 0], Lv[1, 0], Lv[1, 1])

        # genetic covariance ~ IW(4 + m, Sbeta) via Bartlett
        S = np.array([[s11, s12], [s12, s22]]) + 1e-10 * np.eye(2)
        Lc = np.linalg.cholesky(np.linalg.inv(S))
        A = np.zeros((2, 2))
        A[0, 0] = np.sqrt(r.chisquare(4.0 + m))
        A[1, 1] = np.sqrt(r.chisquare(3.0 + m))
        A[1, 0] = r.standard_normal()
        G = np.linalg.inv(Lc @ A @ A.T @ Lc.T)

        if arm == "joint":
            # whitened marginal residuals: z_t = sqrt(N_t) Linv (bhat_t - R beta_t)
            w1 = np.einsum("ikl,il->ik", Linv, bh1 - rb1) * np.sqrt(n1)
            w2 = np.einsum("ikl,il->ik", Linv, bh2 - rb2) * np.sqrt(n2)
            S11 = float((w1 * w1).sum())
            S22 = float((w2 * w2).sum())
            S12 = float((w1 * w2).sum())
            ll = (-0.5 * m * np.log(1 - grid ** 2)
                  - (S11 - 2 * grid * S12 + S22) / (2 * (1 - grid ** 2)))
            ll -= ll.max()
            wts = np.exp(ll)
            cc = float(r.choice(grid, p=wts / wts.sum()))

        if sweep >= burn:
            keep_cc.append(cc)
            keep_G.append(G.copy())

    Gm = np.mean(keep_G, axis=0)
    return (float(Gm[0, 1] / np.sqrt(Gm[0, 0] * Gm[1, 1])),
            float(np.mean(keep_cc)))


def ldsc_cross_corr(bh1, bh2, ell, n1, n2, nb):
    """cross_corr from the cross-trait LDSC intercept (bipred's current path).

    Returns ``(value, status)``. A failure returns NaN with status "fail" rather
    than silently substituting 0.0 -- collapsing the competing arm into `naive`
    and then reporting it as an LDSC result would flatter the proposed method."""
    m = bh1.size
    try:
        res = ldsc_rg(bh1.ravel(), bh2.ravel(), ell, n1, n2,
                      m_snps=m, n_blocks=min(nb, 50))
        ic = float(res.gcov_intercept)
    except Exception:
        return np.nan, "fail"
    if not np.isfinite(ic):
        return np.nan, "fail"
    if abs(ic) > 0.95:                       # sampler needs |cross_corr| < 1
        return float(np.clip(ic, -0.95, 0.95)), "clip"
    return ic, "ok"


def realized_rg(R, b1, b2):
    """LD-aware genetic correlation of a *finite* effect draw -- the quantity the
    estimators actually target for that dataset, which differs slightly from the
    nominal r_g used to generate it."""
    nb = R.shape[0]

    def q(x, y):
        return float(sum(x[i] @ (R[i] @ y[i]) for i in range(nb)))

    return q(b1, b2) / np.sqrt(q(b1, b1) * q(b2, b2))


def one_cell(rg, cc_true, n1, n2, nb, k, reps, ld_seed=7,
             rho_lo=0.2, rho_hi=0.8):
    """All four arms on identical data, `reps` independent replicates.

    Each replicate redraws BOTH the causal effects and the GWAS sampling noise.
    Freezing the effects across replicates (an earlier version of this script)
    makes every cell a single architecture draw: the LDSC intercept is a
    functional of that architecture, so its per-cell mean then reflects one draw
    rather than the method's behaviour, and across-draw SD can exceed the effect
    being reported. Redrawing per replicate makes the replicates independent and
    the reported SDs meaningful."""
    R, L, Linv, ell = make_ld(nb, k, ld_seed, rho_lo, rho_hi)
    out = {a: {"rg": [], "cc": []} for a in ARMS}
    realized, n_fail, n_clip = [], 0, 0
    for rep in range(reps):
        beta1, beta2 = make_effects(R, rg, H1, H2, ld_seed + 1 + 1000 * rep)
        realized.append(realized_rg(R, beta1, beta2))
        bh1, bh2 = make_sumstats(R, L, beta1, beta2, cc_true, n1, n2,
                                 1000 + 17 * rep)
        cc_ldsc, status = ldsc_cross_corr(bh1, bh2, ell, n1, n2, nb)
        n_fail += status == "fail"
        n_clip += status == "clip"
        for arm in ARMS:
            fixed = {"naive": 0.0, "ldsc": cc_ldsc,
                     "joint": 0.0, "oracle": cc_true}[arm]
            if arm == "ldsc" and not np.isfinite(fixed):
                out[arm]["rg"].append(np.nan)   # excluded, not silently naive
                out[arm]["cc"].append(np.nan)
                continue
            rg_hat, cc_hat = run_sampler(R, Linv, bh1, bh2, n1, n2, arm,
                                         fixed, seed=rep)
            out[arm]["rg"].append(rg_hat)
            out[arm]["cc"].append(cc_hat if arm == "joint" else fixed)
    return out, np.array(realized), n_fail, n_clip


def _rows(cell, rg, cc_true, n1, m, reps, realized, n_fail=0, n_clip=0):
    """Per-arm summary. Bias/RMSE are scored against each replicate's REALIZED
    r_g (the estimand for that finite effect draw), not the nominal r_g used to
    generate it; `rg_bias_nominal` keeps the nominal-scored value for reference.
    Dispersions use ddof=1. NaN entries (excluded LDSC failures) are dropped and
    counted in `n_used`."""
    rows = []
    for arm in ARMS:
        rgs = np.array(cell[arm]["rg"], dtype=float)
        ccs = np.array(cell[arm]["cc"], dtype=float)
        ok = np.isfinite(rgs)
        err = rgs[ok] - realized[ok]              # vs the realized estimand
        ccs_ok = ccs[np.isfinite(ccs)]
        rows.append({
            "arm": arm, "rg_true": rg, "cc_true": cc_true, "N": int(n1),
            "m": m, "reps": reps, "n_used": int(ok.sum()),
            "rg_mean": round(float(rgs[ok].mean()), 4),
            "rg_sd": round(float(rgs[ok].std(ddof=1)), 4),
            "rg_bias": round(float(err.mean()), 4),
            "rg_rmse": round(float(np.sqrt((err ** 2).mean())), 4),
            "rg_bias_nominal": round(float(rgs[ok].mean() - rg), 4),
            "cc_mean": round(float(ccs_ok.mean()), 4) if ccs_ok.size else "",
            "cc_sd": (round(float(ccs_ok.std(ddof=1)), 4)
                      if ccs_ok.size > 1 else ""),
            "realized_rg_mean": round(float(realized.mean()), 4),
            "realized_rg_sd": round(float(realized.std(ddof=1)), 4),
            "ldsc_fail": n_fail if arm == "ldsc" else "",
            "ldsc_clip": n_clip if arm == "ldsc" else "",
        })
    return rows


def write(rows, name):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, f"bench_cross_corr_{name}.csv")
    fields = ["arm", "rg_true", "cc_true", "N", "m", "reps", "rg_mean",
              "rg_sd", "rg_bias", "rg_rmse", "cc_mean", "cc_sd"]
    extra = [k for k in rows[0] if k not in fields] if rows else []
    fields = fields + extra
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.basename(path)} ({len(rows)} rows)")


def show(rows):
    print(f"{'arm':7s} {'rg_true':>7s} {'cc_true':>7s} {'N':>7s} {'m':>6s} "
          f"{'rg_mean':>9s} {'rg_sd':>7s} {'rg_bias':>8s} {'rg_rmse':>8s} "
          f"{'cc_mean':>8s}")
    for r in rows:
        print(f"{r['arm']:7s} {r['rg_true']:7.2f} {r['cc_true']:7.2f} "
              f"{r['N']:7d} {r['m']:6d} {r['rg_mean']:9.4f} {r['rg_sd']:7.4f} "
              f"{r['rg_bias']:8.4f} {r['rg_rmse']:8.4f} {r['cc_mean']:8.4f}")


def grid_main(reps=REPS):
    rows = []
    for rg in (0.0, 0.3, 0.6):
        for cc in (0.0, 0.2, 0.4):
            t0 = time.time()
            cell, real, nf, nc = one_cell(rg, cc, N_DEF, N_DEF, NB, K, reps)
            rows += _rows(cell, rg, cc, N_DEF, NB * K, reps, real, nf, nc)
            print(f"  cell rg={rg} cc={cc} done in {time.time()-t0:.0f}s")
    return rows


def grid_n(reps=REPS):
    rows = []
    for n in (4_000.0, 8_000.0, 20_000.0, 50_000.0):
        t0 = time.time()
        cell, real, nf, nc = one_cell(0.6, 0.4, n, n, NB, K, reps)
        rows += _rows(cell, 0.6, 0.4, n, NB * K, reps, real, nf, nc)
        print(f"  cell N={n:.0f} done in {time.time()-t0:.0f}s")
    return rows


def grid_m(reps=REPS):
    rows = []
    for nb, k in ((15, 100), (30, 100), (60, 100), (120, 100)):
        t0 = time.time()
        cell, real, nf, nc = one_cell(0.6, 0.4, N_DEF, N_DEF, nb, k, reps)
        rows += _rows(cell, 0.6, 0.4, N_DEF, nb * k, reps, real, nf, nc)
        print(f"  cell m={nb*k} done in {time.time()-t0:.0f}s")
    return rows


def grid_scale(reps=4, only_nb=None):
    """Large-m extension of the `m` sweep.

    RESULTS.md shows the LDSC intercept converging roughly as 1/sqrt(m), so the
    honest question is whether the advantage survives toward realistic variant
    counts.

    Sweeps are RAISED, not lowered, for this grid. Per-SNP power (N*h2/m) falls
    as m grows -- 0.04 at m = 100k -- and the sampler needs more sweeps to
    converge there, not fewer. A first version of this grid used 400 sweeps and
    produced a large apparent degradation at m >= 50k *in every arm including
    the oracle*, which is diagnostic of a convergence artifact rather than a
    cross_corr effect: at m = 100k the oracle's bias is -0.246 at 400 sweeps and
    -0.040 at 1200. Do not lower these.
    """
    rows = []
    global N_SWEEP, BURN
    keep = (N_SWEEP, BURN)
    N_SWEEP, BURN = 1500, 500
    todo = (only_nb,) if only_nb else (125, 250, 500, 1000)
    for nb in todo:                             # m = 12.5k .. 100k at K = 100
        t0 = time.time()
        cell, real, nf, nc = one_cell(0.6, 0.4, N_DEF, N_DEF, nb, 100, reps)
        rows += _rows(cell, 0.6, 0.4, N_DEF, nb * 100, reps, real, nf, nc)
        print(f"  cell m={nb*100} done in {time.time()-t0:.0f}s")
    N_SWEEP, BURN = keep
    return rows


def grid_ldwide(reps=REPS):
    """Fairness re-test: widen the LD-score range so LDSC's intercept is properly
    identified. The default grids use rho in [0.2, 0.8] (ell about 1-4.5), far
    narrower than a real genome; the cross-trait LDSC intercept is identified
    only by variation in ell, so the narrow default disadvantages the `ldsc` arm.
    Here rho reaches 0.99, giving ell up to ~100."""
    rows = []
    for lo, hi in ((0.2, 0.8), (0.5, 0.95), (0.5, 0.99)):
        R, _, _, ell = make_ld(60, 100, 7, lo, hi)
        t0 = time.time()
        cell, real, nf, nc = one_cell(0.6, 0.4, N_DEF, N_DEF, 60, 100, reps,
                                      rho_lo=lo, rho_hi=hi)
        r = _rows(cell, 0.6, 0.4, N_DEF, 6000, reps, real, nf, nc)
        for row in r:
            row["ell_min"] = round(float(ell.min()), 2)
            row["ell_max"] = round(float(ell.max()), 2)
            row["rho_hi"] = hi
        rows += r
        print(f"  cell rho<= {hi} (ell {ell.min():.1f}-{ell.max():.1f}) "
              f"done in {time.time()-t0:.0f}s")
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else REPS
    only = int(sys.argv[3]) if len(sys.argv) > 3 else None
    print(f"cross_corr benchmark (numba={HAVE_NUMBA}) grid={which} reps={reps}")
    todo = {"main": [("main", grid_main)], "n": [("n", grid_n)],
            "m": [("m", grid_m)], "ldwide": [("ldwide", grid_ldwide)],
            "scale": [(f"scale{sys.argv[3]}" if len(sys.argv) > 3 else "scale",
                       lambda r: grid_scale(r, only))],
            "all": [("main", grid_main), ("n", grid_n), ("m", grid_m),
                    ("ldwide", grid_ldwide)]}[which]
    for name, fn in todo:
        print(f"\n=== grid: {name} ===")
        rows = fn(reps)
        show(rows)
        write(rows, name)


if __name__ == "__main__":
    main()
