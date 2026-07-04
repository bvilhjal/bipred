"""Sample-overlap validation for the bivariate genetic-correlation estimators.

Overlapping GWAS samples induce a correlation in the two traits' sampling noise
(controlled here by ``rho_e``), which inflates a naive genetic-correlation
estimate even when the traits are genetically uncorrelated. This checks that the
corrections handle it, on the **same realistic non-repeating coalescent LD** as
``rg_architectures.py`` (whose population blocks and helpers are reused):

  - bivariate LDSC: a free cross-trait *intercept* should absorb the overlap;
    constraining it to 0 should leave the bias.
  - bivariate LDpred3: passing ``cross_corr=rho_e`` should remove the bias;
    leaving ``cross_corr=0`` should not.

Also reports the per-fit running time. Needs ``msprime``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/sample_overlap.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bipred import ldsc_rg, ldpred3_auto_bivariate_blocks                  # noqa: E402
import rg_architectures as R                                              # noqa: E402

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
    print(f"{'rg':>4} | {'LDSC icpt=0':>11} | {'LDSC free icpt':>14} | "
          f"{'biv cc=0':>9} | {'biv cc=rho':>10} | {'t LDSC':>7} | {'t biv':>7}")
    print("-" * 82)
    t0 = time.time()
    for rg in RGS:
        a, b, c, d, tl, tb = [], [], [], [], [], []
        for rep in range(REPS):
            rng = np.random.default_rng(400 + rep)
            b1, b2 = R.sim_effects("sparse", rg, rng)          # p=0.01 (overlap bites hardest)
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

        def m(x):
            x = np.asarray(x, float)
            x = x[np.abs(x) <= 1.5]
            return f"{np.mean(x):.3f}±{np.std(x):.3f}" if x.size else "  —  "
        print(f"{rg:>4} | {m(a):>11} | {m(b):>14} | {m(c):>9} | {m(d):>10} | "
              f"{np.mean(tl)*1000:>5.1f}ms | {np.mean(tb):>5.2f}s")
    print(f"\nUncorrected columns (LDSC icpt=0, biv cc=0) should be biased upward by "
          f"the overlap; the corrected ones should recover the truth.  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
