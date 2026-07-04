"""Genetic-correlation estimators compared — accuracy and running time.

Four ways to estimate the genetic correlation from GWAS summary statistics, all
on the same realistic non-repeating coalescent LD (reusing ``rg_architectures``'
cached segments + finite reference panels):

  * ``ldsc``    -- cross-trait LD Score regression (``ldsc_rg``); cheapest.
  * ``uni_gv``  -- two independent univariate LDpred3-auto runs, genetic
    correlation of the posterior-mean effects, self-normalized:
    ``b1'R b2 / sqrt(b1'R b1 . b2'R b2)``.
  * ``uni_r2``  -- same numerator, but the denominator uses each run's
    *decorrelated* out-of-sample r² (``InferResult.r2_est``, a cross-chain
    quadratic with the within-trait self-noise removed):
    ``b1'R b2 / sqrt(r2_1 . r2_2)``.
  * ``biv``     -- the bivariate joint fit's genetic correlation
    (``ldpred3_auto_bivariate_blocks(...).rg``).

``uni_gv`` and ``uni_r2`` are read off the *same* pair of univariate runs, so they
share a running time (the two ``ldpred3_auto_infer`` calls); only a cheap
denominator differs. Reports mean±sd accuracy over replicates (fresh phenotypes,
fixed genotypes) and mean wall-clock per method, at two power settings
(symmetric and asymmetric). A second pass times every method across ``m`` to show
scaling.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/rg_methods.py

Env: ``NB`` / ``K`` / ``MUT_RATE`` (via rg_architectures) set ``m``; ``REPS``;
``RHO_BETA``; ``N_CHAINS``; ``SCALE_SIZES`` (comma list of NB for the timing
scan); ``OUT`` basename; ``SWEEP_ONLY`` / ``SCALE_ONLY`` to run one pass.
"""
import os
import sys
import csv
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rg_architectures as R                                            # noqa: E402
from ldpred3 import ldsc_rg, ldpred3_auto_infer                         # noqa: E402
from bipred import ldpred3_auto_bivariate_blocks                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = int(os.environ.get("REPS", "6"))
RHO_BETA = float(os.environ.get("RHO_BETA", "0.9"))
N_CHAINS = int(os.environ.get("N_CHAINS", "6"))
BURN, ITER = 150, 250


def _p_causal():
    return int(round(0.10 * R.M))


def qform(ref, a, b):
    return float(sum(a[ix] @ (Rr.astype(np.float64) @ b[ix]) for Rr, ix in ref))


def sim(rng, frac_shared, rho_beta=RHO_BETA):
    m = R.M
    nc = _p_causal()
    n_sh = int(round(frac_shared * nc)); n_u = nc - n_sh
    picks = rng.choice(m, n_sh + 2 * n_u, replace=False)
    sh, u1, u2 = picks[:n_sh], picks[n_sh:n_sh + n_u], picks[n_sh + n_u:]
    b1 = np.zeros(m); b2 = np.zeros(m)
    b1[u1] = rng.standard_normal(n_u); b2[u2] = rng.standard_normal(n_u)
    L = np.linalg.cholesky([[1, rho_beta], [rho_beta, 1]])
    raw = L @ rng.standard_normal((2, n_sh)); b1[sh] = raw[0]; b2[sh] = raw[1]
    b1 *= np.sqrt(0.5 / R.gv(b1, b1)); b2 *= np.sqrt(0.5 / R.gv(b2, b2))
    return b1, b2, rho_beta * (n_sh / nc)


def estimate_all(ref, ell, bh1, bh2, N1, N2, rep):
    """Return {method: rg} and {method: seconds}. uni_gv/uni_r2 share a run."""
    m = R.M
    rg, t = {}, {}
    t0 = time.perf_counter()
    try:
        rg["ldsc"] = ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=R.NB).rg
    except Exception:
        rg["ldsc"] = np.nan
    t["ldsc"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    i1 = ldpred3_auto_infer(ref, bh1, np.full(m, float(N1)), n_chains=N_CHAINS,
                            burn_in=BURN, num_iter=ITER, seed=rep)
    i2 = ldpred3_auto_infer(ref, bh2, np.full(m, float(N2)), n_chains=N_CHAINS,
                            burn_in=BURN, num_iter=ITER, seed=rep)
    t_uni = time.perf_counter() - t0
    e1, e2 = i1.beta_est, i2.beta_est
    cov = qform(ref, e1, e2)
    v1, v2 = qform(ref, e1, e1), qform(ref, e2, e2)
    rg["uni_gv"] = cov / np.sqrt(max(v1 * v2, 1e-30))
    rg["uni_r2"] = cov / np.sqrt(max(i1.r2_est * i2.r2_est, 1e-30))
    t["uni_gv"] = t["uni_r2"] = t_uni      # shared cost

    t0 = time.perf_counter()
    rg["biv"] = ldpred3_auto_bivariate_blocks(ref, bh1, bh2, N1, N2, burn_in=BURN,
                                              num_iter=ITER, seed=rep).rg
    t["biv"] = time.perf_counter() - t0
    return rg, t


METHODS = ["ldsc", "uni_gv", "uni_r2", "biv"]


def sweep(rows, N1, N2, tag):
    print(f"\n== accuracy: {tag} (N1={N1} N2={N2}, rho_beta={RHO_BETA}, m={R.M}, "
          f"{REPS} reps) ==", flush=True)
    hdr = " ".join(f"{mth:>12}" for mth in METHODS)
    print(f"{'rg_true':>7} | {hdr}", flush=True)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        acc = {mth: [] for mth in METHODS}
        tim = {mth: [] for mth in METHODS}
        tru = None
        for rep in range(REPS):
            ref, ell = R.ref_panel(rep)
            rng = np.random.default_rng(6000 + rep)
            b1, b2, tru = sim(rng, frac)
            bh1, bh2 = R.sumstats_pair(b1, b2, N1, N2, rng)
            rg, t = estimate_all(ref, ell, bh1, bh2, N1, N2, rep)
            for mth in METHODS:
                acc[mth].append(rg[mth]); tim[mth].append(t[mth])
        row = {"tag": tag, "N1": N1, "N2": N2, "rg_true": round(tru, 3)}
        cells = []
        for mth in METHODS:
            a = np.array(acc[mth], float)
            ok = np.isfinite(a) & (np.abs(a) <= 1.5)
            mean = float(np.mean(a[ok])) if ok.any() else np.nan
            sd = float(np.std(a[ok])) if ok.any() else np.nan
            row[f"{mth}_rg"] = round(mean, 3)
            row[f"{mth}_sd"] = round(sd, 3)
            row[f"{mth}_t"] = round(float(np.mean(tim[mth])), 3)
            cells.append(f"{mean:>6.2f}±{sd:<5.2f}")
        rows.append(row)
        print(f"{tru:>7.2f} | " + " ".join(cells), flush=True)
    t_line = " ".join(f"{mth} {np.mean([r[f'{mth}_t'] for r in rows if r['tag']==tag]):.2f}s"
                      for mth in METHODS)
    print(f"  mean time/fit: {t_line}", flush=True)


def scale(rows_t, sizes):
    print(f"\n== timing scan vs m (rho_beta={RHO_BETA}, N=50000, 1 rep) ==", flush=True)
    print(f"{'m':>8} | " + " ".join(f"{mth:>10}" for mth in METHODS), flush=True)
    import importlib
    for nb in sizes:
        os.environ["NB"] = str(nb)
        importlib.reload(R)                      # rebuild pop LD / m for this NB
        m = R.M
        ref, ell = R.ref_panel(0)
        rng = np.random.default_rng(12345)
        b1, b2, _ = sim(rng, 0.5)
        bh1, bh2 = R.sumstats_pair(b1, b2, 50000, 50000, rng)
        # warm JIT at this size
        ldpred3_auto_bivariate_blocks(ref, bh1, bh2, 50000, 50000, burn_in=3, num_iter=3)
        ldpred3_auto_infer(ref, bh1, np.full(m, 5e4), n_chains=2, burn_in=3, num_iter=3)
        _, t = estimate_all(ref, ell, bh1, bh2, 50000, 50000, 0)
        row = {"m": m, "nb": nb, **{f"{mth}_t": round(t[mth], 3) for mth in METHODS}}
        rows_t.append(row)
        print(f"{m:>8} | " + " ".join(f"{t[mth]:>9.3f}s" for mth in METHODS), flush=True)


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def make_figure(rows, rows_t, base):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    npan = len(tags) + (1 if rows_t else 0)
    fig, ax = plt.subplots(1, npan, figsize=(4.2 * npan, 3.8))
    if npan == 1:
        ax = [ax]
    colors = {"ldsc": "C0", "uni_gv": "C2", "uni_r2": "C1", "biv": "C3"}
    mark = {"ldsc": "o", "uni_gv": "^", "uni_r2": "v", "biv": "s"}
    for p, tag in enumerate(tags):
        rr = sorted([r for r in rows if r["tag"] == tag], key=lambda r: r["rg_true"])
        x = [r["rg_true"] for r in rr]
        ax[p].plot([0, 1], [0, 1], "k--", lw=1, alpha=.5)
        for mth in METHODS:
            ax[p].errorbar(x, [r[f"{mth}_rg"] for r in rr],
                           [r[f"{mth}_sd"] for r in rr], fmt=mark[mth] + "-", ms=4,
                           capsize=2, color=colors[mth], label=mth, alpha=.85)
        ax[p].set_title(f"accuracy — {tag}", fontsize=9)
        ax[p].set_xlabel("true r_g"); ax[p].grid(alpha=.3)
        if p == 0:
            ax[p].set_ylabel("estimated r_g"); ax[p].legend(fontsize=8)
    if rows_t:
        a = ax[-1]
        xs = [r["m"] for r in rows_t]
        for mth in METHODS:
            a.plot(xs, [r[f"{mth}_t"] for r in rows_t], mark[mth] + "-", ms=4,
                   color=colors[mth], label=mth)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("m (variants)"); a.set_ylabel("seconds / fit")
        a.set_title("running time vs m", fontsize=9); a.grid(alpha=.3, which="both")
        a.legend(fontsize=8)
    fig.suptitle(f"Genetic-correlation estimators — accuracy & running time "
                 f"({REPS} reps)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, base + ".png"), dpi=130)


def main():
    base = os.environ.get("OUT", "rg_methods")
    rows, rows_t = [], []
    if not os.environ.get("SCALE_ONLY"):
        print(f"Genetic-correlation estimators — realistic LD (m={R.M}, {R.NB} "
              f"blocks, Nref={R.NREF})", flush=True)
        # warm JIT
        ref, ell = R.ref_panel(0); rng = np.random.default_rng(0)
        b1, b2, _ = sim(rng, 0.5); bh1, bh2 = R.sumstats_pair(b1, b2, 50000, 50000, rng)
        ldpred3_auto_bivariate_blocks(ref, bh1, bh2, 50000, 50000, burn_in=3, num_iter=3)
        ldpred3_auto_infer(ref, bh1, np.full(R.M, 5e4), n_chains=2, burn_in=3, num_iter=3)
        t0 = time.time()
        sweep(rows, 50000, 50000, "symmetric")
        sweep(rows, 50000, 10000, "asymmetric")
        _write_csv(os.path.join(HERE, base + ".csv"), rows)
        print(f"  [accuracy done {time.time()-t0:.0f}s]", flush=True)
    if not os.environ.get("SWEEP_ONLY"):
        sizes = [int(x) for x in os.environ.get("SCALE_SIZES", "25,100,250").split(",")]
        scale(rows_t, sizes)
        _write_csv(os.path.join(HERE, base + "_timing.csv"), rows_t)
    if not os.environ.get("OUT"):
        make_figure(rows, rows_t, base)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
