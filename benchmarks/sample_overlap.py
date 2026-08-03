"""Sample-overlap validation for the bivariate genetic-correlation estimators.

Overlapping GWAS samples induce a correlation in the two traits' sampling noise
(controlled here by ``rho_e``), which can shift a genetic-correlation estimate.
This checks how the estimators respond to their overlap corrections on the
**same realistic
non-repeating coalescent LD** as
``rg_architectures.py`` (whose population blocks and helpers are reused):

  - bivariate LDSC: compare a free cross-trait *intercept* with an intercept
    constrained to zero.
  - bivariate LDpred3: compare ``cross_corr=rho_e`` with ``cross_corr=0``.

Also reports per-fit running time and writes ``sample_overlap.csv``. Needs
``msprime``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/sample_overlap.py
"""
import os
import sys
import csv
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bipred import ldsc_rg, ldpred3_auto_bivariate_blocks                  # noqa: E402
import rg_architectures as R                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N1, N2 = 15000, 15000                 # equal, fully overlapping cohorts
RHO_E = float(os.environ.get("RHO_E", "0.5"))   # cross-trait sampling-noise corr
REPS = int(os.environ.get("REPS", "8"))
RGS = [0.0, 0.3, 0.6]


def main():
    ref, ell = R.ref_panel(0)         # one reference panel, shared across cells
    # warm the bivariate sampler's JIT
    r0 = np.random.default_rng(0)
    e1, e2 = R.sim_effects("polygenic", 0.3, r0)
    h1, h2 = R.sumstats_pair(e1, e2, N1, N2, r0, rho_e=RHO_E)
    ldpred3_auto_bivariate_blocks(ref, h1, h2, N1, N2, burn_in=5, num_iter=5)

    print(f"Sample overlap (rho_e={RHO_E}) on realistic non-repeating LD "
          f"(m={R.M}, {R.NB} unique coalescent blocks, N1={N1}/N2={N2}, {REPS} reps)\n")
    print(f"{'target':>6} | {'realized rg':>13} | {'LDSC icpt=0':>11} | "
          f"{'LDSC free icpt':>14} | "
          f"{'biv cc=0':>9} | {'biv cc=rho':>10} | {'t LDSC':>7} | {'t biv':>7}")
    print("-" * 104)
    t0 = time.time()
    rows = []
    for rg_target in RGS:
        realized, a, b, c, d, tl, tb = [], [], [], [], [], [], []
        for rep in range(REPS):
            rng = np.random.default_rng(400 + rep)
            b1, b2 = R.sim_effects("sparse", rg_target, rng)
            realized.append(R.realized_rg(b1, b2))
            bh1, bh2 = R.sumstats_pair(b1, b2, N1, N2, rng, rho_e=RHO_E)
            t = time.perf_counter()
            a.append(ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=R.NB,
                             constrain_intercept=0.0).rg)
            b.append(ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=R.NB).rg)
            tl.append((time.perf_counter() - t) / 2)
            t = time.perf_counter()
            c.append(ldpred3_auto_bivariate_blocks(ref, bh1, bh2, N1, N2, burn_in=150,
                                                   num_iter=180, cross_corr=0.0,
                                                   seed=rep).rg)
            d.append(ldpred3_auto_bivariate_blocks(ref, bh1, bh2, N1, N2, burn_in=150,
                                                   num_iter=180, cross_corr=RHO_E,
                                                   seed=rep).rg)
            tb.append((time.perf_counter() - t) / 2)

        truth = np.asarray(realized, float)
        finite_truth = truth[np.isfinite(truth)]
        row = {
            "rg_target": rg_target,
            "rg_realized_mean": round(float(np.mean(finite_truth)), 4),
            "rg_realized_sd": round(float(np.std(finite_truth)), 4),
            "rho_e": RHO_E, "n1": N1, "n2": N2, "reps": REPS,
            "ldsc_t": round(float(np.mean(tl)), 4),
            "biv_t": round(float(np.mean(tb)), 4),
        }
        for name, values in (("ldsc_con", a), ("ldsc_free", b),
                             ("biv_cc0", c), ("biv_cctrue", d)):
            mean, sd, mae, fail = R._estimate_summary(
                np.asarray(values, float), truth)
            row[f"{name}_rg"] = mean
            row[f"{name}_sd"] = sd
            row[f"{name}_mae_realized"] = mae
            row[f"{name}_fail"] = fail
        rows.append(row)

        def display(name):
            mean = row[f"{name}_rg"]
            sd = row[f"{name}_sd"]
            return f"{mean:.3f}±{sd:.3f}" if mean != "" else "—"

        print(f"{rg_target:>6.2f} | {row['rg_realized_mean']:>6.3f}"
              f"±{row['rg_realized_sd']:<5.3f} | "
              f"{display('ldsc_con'):>11} | {display('ldsc_free'):>14} | "
              f"{display('biv_cc0'):>9} | {display('biv_cctrue'):>10} | "
              f"{np.mean(tl)*1000:>5.1f}ms | {np.mean(tb):>5.2f}s")
    with open(os.path.join(HERE, "sample_overlap.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCompare corrected and uncorrected columns against realized rg; "
          f"Monte Carlo variation can dominate the expected overlap shift at "
          f"this power. Inspect MAE and failure counts. ({time.time()-t0:.0f}s)\n"
          "wrote sample_overlap.csv")


if __name__ == "__main__":
    main()
