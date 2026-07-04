"""Sample overlap and the bivariate genetic-correlation estimate — how much it
matters and how to set ``cross_corr``.

Overlapping GWAS samples correlate the two studies' sampling noise. Bivariate
LDpred3 models that as the noise-covariance term ``cross_corr/sqrt(N1_j N2_j)``,
structurally separate from the LD-mediated genetic covariance, so:

1. **The rg estimate is only mildly sensitive to overlap.** Fitting with
   ``cross_corr=0`` (ignoring overlap) inflates rg by a small amount even at
   strong overlap; supplying the true value removes it.
2. **Setting ``cross_corr``**: use the known overlap
   (``N_shared·rho_pheno/sqrt(N1 N2)``) when you have it, otherwise the
   cross-trait LDSC intercept (``ldsc_rg(...).gcov_intercept``, inverted by
   ``estimate_sample_overlap``) — the standard estimator, well-anchored at real
   GWAS scale. ``cross_corr=0`` is a safe default; the residual bias is small.

Realistic non-repeating coalescent LD (``rg_architectures``). Needs ``msprime``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/overlap_estimation.py
"""
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bipred import ldpred3_auto_bivariate_blocks                             # noqa: E402
import rg_architectures as R                                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N1, N2 = 60000, 40000
REPS = int(os.environ.get("REPS", "8"))
RG_GRID = [0.0, 0.5]
RHO_GRID = [0.0, 0.2, 0.4]              # true overlap-induced noise correlation
_REF, _ = R.ref_panel(0)


def fit_rg(bh1, bh2, cross_corr, seed):
    return ldpred3_auto_bivariate_blocks(_REF, bh1, bh2, N1, N2, burn_in=150,
                                         num_iter=180, cross_corr=cross_corr,
                                         seed=seed).rg


def main():
    print(f"Bivariate rg vs sample overlap (m={R.M}, N1={N1}, N2={N2}, {REPS} reps)\n")
    print(f"{'rg_true':>7} {'overlap ρ':>9} | {'rg (cc=0)':>10} | {'rg (cc=true)':>12} "
          f"| {'bias from ρ':>11}")
    print("-" * 60)
    rows = []
    for rgt in RG_GRID:
        base0 = None
        for rho in RHO_GRID:
            r0, rt = [], []
            for rep in range(REPS):
                rng = np.random.default_rng(20 + rep)
                b1, b2 = R.sim_effects("polygenic", rgt, rng)
                bh1, bh2 = R.sumstats_pair(b1, b2, N1, N2, rng, rho_e=rho)
                r0.append(fit_rg(bh1, bh2, 0.0, rep))
                rt.append(fit_rg(bh1, bh2, rho, rep))
            m0, mt = float(np.mean(r0)), float(np.mean(rt))
            if base0 is None:
                base0 = m0
            print(f"{rgt:>7} {rho:>9} | {m0:>10.3f} | {mt:>12.3f} | {m0 - base0:>+11.3f}")
            rows.append({"rg_true": rgt, "overlap_rho": rho,
                         "rg_cc0": round(m0, 4), "rg_cctrue": round(mt, 4),
                         "bias_vs_no_overlap": round(m0 - base0, 4)})
    with open(os.path.join(HERE, "overlap_estimation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nIgnoring overlap (cc=0) adds a small upward bias that grows with ρ; "
          "supplying the\ntrue cross_corr removes it. The effect is second-order for "
          "a strong true rg.\nwrote overlap_estimation.csv")


if __name__ == "__main__":
    main()
