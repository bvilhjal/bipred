"""Benchmark: **regional** genetic correlation from the bivariate Gibbs sampler,
and why estimating ``cross_corr`` in the sampler is what makes it possible.

STATUS: research benchmark. Not a shipped feature.

Motivation. Local/regional r_g (per LD block or locus) is the use case that makes
the in-sampler ``cross_corr`` estimator worth building, for two reasons the
genome-wide benchmark cannot show:

  1. **A region is small by construction** (10^2-10^3 variants). That is exactly
     the regime where the cross-trait LDSC intercept is useless -- at m = 1,500
     its SD was 0.625 in bench_cross_corr.py -- so "just use the LDSC intercept"
     is not an option per region.
  2. **Overlap cannot be estimated within a region** for the same reason, so
     regional r_g needs a *genome-wide* cross_corr. Uncorrected sample overlap
     adds the SAME spurious covariance to EVERY region, so it does not cancel
     when you compare regions: it inflates every regional r_g at once and
     confounds genuine regional heterogeneity.

The claim under test: with cross_corr estimated genome-wide in the sampler,
regional r_g is recovered and regional heterogeneity is preserved; without it,
every region is contaminated.

Design. Each LD block is one region. Regions are assigned heterogeneous true
r_g (0.0 / 0.4 / 0.8 in equal thirds) with an equal heritability share, and the
GWAS pair has real sample overlap. All four arms of bench_cross_corr.py are run
(naive / ldsc / joint / oracle), differing only in the cross_corr value, and per
region we estimate r_g two ways from the retained sweeps:

  sampled  mean(b1' R b2) / sqrt(mean(b1' R b1) * mean(b2' R b2))
           -- the sampled-quadratic ratio, matching bipred's genome-wide `rg`.
  postmean pm1' R pm2 / sqrt(pm1' R pm1 * pm2' R pm2) using posterior-mean
           effects -- shrunk, but without the same-sweep noise inflation.

Both are scored against each region's **realized** r_g (computed from that
replicate's true effects with the same LD), never the nominal label.

A caveat this benchmark is designed to expose, not hide: the sampler carries a
single genome-wide effect covariance Sigma, so every per-SNP draw is shrunk
toward the genome-wide r_g. Regional estimates are therefore expected to be
pulled toward the genome-wide mean regardless of cross_corr. The `shrink`
columns quantify that; see RESULTS_REGIONAL.md.

Run:  OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/bench_regional_rg.py [main|size|all] [reps]
Writes bench_regional_<grid>.csv.
"""
import csv
import os
import sys
import time

import numpy as np

from bench_cross_corr import (ARMS, HAVE_NUMBA, _seed_numba, _sweep,  # noqa: E402
                              ldsc_cross_corr, make_ld, make_sumstats)

H1, H2 = 0.5, 0.5
N_DEF = 50_000.0
REGION_RGS = (0.0, 0.4, 0.8)       # true r_g classes, equal numbers of regions
NB_DEF, K_DEF = 60, 100            # 60 regions x 100 variants
CC_TRUE = 0.4
N_SWEEP, BURN = 900, 300
REPS = 10


def make_regional_effects(R, rg_by_block, h1, h2, seed):
    """Effects whose *per-block* genetic correlation is ``rg_by_block[i]``.

    Each block gets an equal share of heritability, so regions differ in r_g but
    not in signal strength -- otherwise a region's r_g accuracy would be confounded
    with its power."""
    nb, k, _ = R.shape
    r = np.random.default_rng(seed)
    b1 = np.empty((nb, k))
    b2 = np.empty((nb, k))
    for i in range(nb):
        rg = rg_by_block[i]
        Lg = np.array([[1.0, 0.0], [rg, np.sqrt(max(0.0, 1 - rg * rg))]])
        raw = Lg @ r.standard_normal((2, k))
        x, y = raw[0], raw[1]
        x *= np.sqrt((h1 / nb) / float(x @ (R[i] @ x)))
        y *= np.sqrt((h2 / nb) / float(y @ (R[i] @ y)))
        b1[i], b2[i] = x, y
    return b1, b2


def regional_realized_rg(R, b1, b2):
    """Realized r_g within each block -- the estimand for that replicate."""
    nb = R.shape[0]
    out = np.empty(nb)
    for i in range(nb):
        q11 = float(b1[i] @ (R[i] @ b1[i]))
        q22 = float(b2[i] @ (R[i] @ b2[i]))
        q12 = float(b1[i] @ (R[i] @ b2[i]))
        out[i] = q12 / np.sqrt(q11 * q22) if q11 > 0 and q22 > 0 else np.nan
    return out


def run_regional(R, Linv, bh1, bh2, n1, n2, arm, cc_fixed, seed,
                 n_sweep=N_SWEEP, burn=BURN):
    """Gibbs sampler returning PER-REGION r_g by two estimators.

    Per-block quadratics come free from the tracked ``R @ beta``:
    ``b1[i] @ rb1[i]`` is exactly ``b1' R b1`` restricted to block i."""
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
    G = np.array([[H1 / m, 0.0], [0.0, H2 / m]])
    cc = 0.0 if arm == "joint" else float(cc_fixed)
    sd1, sd2 = 1.0 / np.sqrt(n1), 1.0 / np.sqrt(n2)
    grid = np.linspace(-0.95, 0.95, 191)

    q11 = np.zeros(nb); q22 = np.zeros(nb); q12 = np.zeros(nb)
    acc_b1 = np.zeros((nb, k)); acc_b2 = np.zeros((nb, k))
    acc_rb1 = np.zeros((nb, k)); acc_rb2 = np.zeros((nb, k))
    cc_keep = []
    kept = 0

    for sweep in range(n_sweep):
        E = np.array([[1.0 / n1, cc * sd1 * sd2], [cc * sd1 * sd2, 1.0 / n2]])
        V = np.linalg.inv(np.linalg.inv(G) + np.linalg.inv(E))
        Lv = np.linalg.cholesky(V)
        VE = V @ np.linalg.inv(E)
        s11, s12, s22 = _sweep(R, bh1, bh2, b1, b2, rb1, rb2,
                               VE[0, 0], VE[0, 1], VE[1, 0], VE[1, 1],
                               Lv[0, 0], Lv[1, 0], Lv[1, 1])

        S = np.array([[s11, s12], [s12, s22]]) + 1e-10 * np.eye(2)
        Lc = np.linalg.cholesky(np.linalg.inv(S))
        A = np.zeros((2, 2))
        A[0, 0] = np.sqrt(r.chisquare(4.0 + m))
        A[1, 1] = np.sqrt(r.chisquare(3.0 + m))
        A[1, 0] = r.standard_normal()
        G = np.linalg.inv(Lc @ A @ A.T @ Lc.T)

        if arm == "joint":
            w1 = np.einsum("ikl,il->ik", Linv, bh1 - rb1) * np.sqrt(n1)
            w2 = np.einsum("ikl,il->ik", Linv, bh2 - rb2) * np.sqrt(n2)
            S11 = float((w1 * w1).sum()); S22 = float((w2 * w2).sum())
            S12 = float((w1 * w2).sum())
            ll = (-0.5 * m * np.log(1 - grid ** 2)
                  - (S11 - 2 * grid * S12 + S22) / (2 * (1 - grid ** 2)))
            ll -= ll.max()
            wts = np.exp(ll)
            cc = float(r.choice(grid, p=wts / wts.sum()))

        if sweep >= burn:
            q11 += np.einsum("ik,ik->i", b1, rb1)      # b1' R b1 per block
            q22 += np.einsum("ik,ik->i", b2, rb2)
            q12 += np.einsum("ik,ik->i", b1, rb2)
            acc_b1 += b1; acc_b2 += b2
            acc_rb1 += rb1; acc_rb2 += rb2
            cc_keep.append(cc)
            kept += 1

    sampled = q12 / np.sqrt(np.maximum(q11 * q22, 1e-300))
    pm1, pm2 = acc_b1 / kept, acc_b2 / kept
    prb1, prb2 = acc_rb1 / kept, acc_rb2 / kept
    p11 = np.einsum("ik,ik->i", pm1, prb1)
    p22 = np.einsum("ik,ik->i", pm2, prb2)
    p12 = np.einsum("ik,ik->i", pm1, prb2)
    postmean = p12 / np.sqrt(np.maximum(p11 * p22, 1e-300))
    return (np.clip(sampled, -1.5, 1.5), np.clip(postmean, -1.5, 1.5),
            float(np.mean(cc_keep)))


def one_cell(nb, k, n1, n2, reps, ld_seed=7):
    """All arms, `reps` replicates; effects and noise redrawn every replicate."""
    R, L, Linv, ell = make_ld(nb, k, ld_seed)
    rg_by_block = np.array([REGION_RGS[i % len(REGION_RGS)] for i in range(nb)])
    res = {a: {"sampled": [], "postmean": [], "cc": []} for a in ARMS}
    realized = []
    for rep in range(reps):
        beta1, beta2 = make_regional_effects(R, rg_by_block, H1, H2,
                                             1234 + 977 * rep)
        realized.append(regional_realized_rg(R, beta1, beta2))
        bh1, bh2 = make_sumstats(R, L, beta1, beta2, CC_TRUE, n1, n2,
                                 5000 + 31 * rep)
        cc_ldsc, status = ldsc_cross_corr(bh1, bh2, ell, n1, n2, nb)
        for arm in ARMS:
            fixed = {"naive": 0.0, "ldsc": cc_ldsc,
                     "joint": 0.0, "oracle": CC_TRUE}[arm]
            if arm == "ldsc" and not np.isfinite(fixed):
                continue
            s, p, cc = run_regional(R, Linv, bh1, bh2, n1, n2, arm, fixed,
                                    seed=rep)
            res[arm]["sampled"].append(s)
            res[arm]["postmean"].append(p)
            res[arm]["cc"].append(cc)
    return res, np.array(realized), rg_by_block


def summarise(res, realized, rg_by_block, nb, k, n1, reps):
    """Per (arm, estimator, true-rg class): bias/RMSE vs the realized regional rg,
    plus a discrimination measure between the null and the strongest class."""
    rows = []
    for arm in ARMS:
        if not res[arm]["sampled"]:
            continue
        for est in ("sampled", "postmean"):
            E = np.array(res[arm][est])          # (reps, nb)
            for cls in REGION_RGS:
                sel = rg_by_block == cls
                err = (E[:, sel] - realized[:, sel]).ravel()
                err = err[np.isfinite(err)]
                vals = E[:, sel].ravel()
                rows.append({
                    "arm": arm, "estimator": est, "region_rg": cls,
                    "N": int(n1), "n_regions": int(sel.sum()),
                    "region_size": k, "reps": reps,
                    "est_mean": round(float(np.nanmean(vals)), 4),
                    "est_sd": round(float(np.nanstd(vals, ddof=1)), 4),
                    "realized_mean": round(float(np.nanmean(realized[:, sel])), 4),
                    "bias": round(float(err.mean()), 4),
                    "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
                    "cc_mean": round(float(np.mean(res[arm]["cc"])), 4),
                })
            # discrimination: separation between null and strongest region class
            lo = np.array(res[arm][est])[:, rg_by_block == REGION_RGS[0]].ravel()
            hi = np.array(res[arm][est])[:, rg_by_block == REGION_RGS[-1]].ravel()
            pooled = np.sqrt((np.nanvar(lo, ddof=1) + np.nanvar(hi, ddof=1)) / 2)
            for r in rows[-len(REGION_RGS):]:
                r["separation_d"] = round(
                    float((np.nanmean(hi) - np.nanmean(lo)) / pooled), 3)
    return rows


def write(rows, name):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, f"bench_regional_{name}.csv")
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.basename(path)} ({len(rows)} rows)")


def show(rows):
    print(f"{'arm':7s} {'estimator':9s} {'rg':>4s} {'est_mean':>9s} {'realized':>9s} "
          f"{'bias':>8s} {'rmse':>7s} {'sep_d':>6s} {'cc':>6s}")
    for r in rows:
        print(f"{r['arm']:7s} {r['estimator']:9s} {r['region_rg']:>4} "
              f"{r['est_mean']:>9} {r['realized_mean']:>9} {r['bias']:>8} "
              f"{r['rmse']:>7} {r.get('separation_d',''):>6} {r['cc_mean']:>6}")


def grid_main(reps=REPS):
    t0 = time.time()
    res, real, rgb = one_cell(NB_DEF, K_DEF, N_DEF, N_DEF, reps)
    rows = summarise(res, real, rgb, NB_DEF, K_DEF, N_DEF, reps)
    print(f"  main cell done in {time.time()-t0:.0f}s")
    return rows


def grid_size(reps=REPS):
    """Region size sweep at fixed total m: smaller regions are the hard case."""
    rows = []
    for nb, k in ((30, 200), (60, 100), (120, 50)):
        t0 = time.time()
        res, real, rgb = one_cell(nb, k, N_DEF, N_DEF, reps)
        rows += summarise(res, real, rgb, nb, k, N_DEF, reps)
        print(f"  region_size={k} ({nb} regions) done in {time.time()-t0:.0f}s")
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else REPS
    print(f"regional r_g benchmark (numba={HAVE_NUMBA}) grid={which} reps={reps}")
    todo = {"main": [("main", grid_main)], "size": [("size", grid_size)],
            "all": [("main", grid_main), ("size", grid_size)]}[which]
    for name, fn in todo:
        print(f"\n=== grid: {name} ===")
        rows = fn(reps)
        show(rows)
        write(rows, name)


if __name__ == "__main__":
    main()
