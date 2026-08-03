"""Sample overlap and the bivariate genetic-correlation estimate — how much it
matters and how to set ``cross_corr``.

Overlapping GWAS samples correlate the two studies' sampling noise. Bivariate
LDpred3 models that as the noise-covariance term ``cross_corr/sqrt(N1_j N2_j)``,
structurally separate from the LD-mediated genetic covariance, so:

1. Fitting with ``cross_corr=0`` exposes the overlap-induced change in rg;
   supplying the generating noise correlation tests the correction.
2. **Setting ``cross_corr``**: use the known overlap
   (``N_shared·rho_pheno/sqrt(N1 N2)``) when you have it, otherwise the
   cross-trait LDSC intercept (``ldsc_rg(...).gcov_intercept``, inverted by
   ``estimate_sample_overlap``) under the assumptions described in
   ``docs/rg.md``. This benchmark reports the effect of a known correction; it
   does not validate intercept inversion or promote a universal default.

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


def _paired_delta(values, baseline):
    """Mean and SD of paired changes inside the diagnostic rg window."""
    ok = (np.isfinite(values) & np.isfinite(baseline)
          & (np.abs(values) <= 1.5) & (np.abs(baseline) <= 1.5))
    if not ok.any():
        return "", ""
    delta = values[ok] - baseline[ok]
    return (round(float(np.mean(delta)), 4),
            round(float(np.std(delta)), 4))


def _display(value):
    return f"{value:.3f}" if value != "" else "—"


def main():
    print(f"Bivariate rg vs sample overlap (m={R.M}, N1={N1}, N2={N2}, "
          f"{REPS} reps)\n")
    print(f"{'target':>7} {'overlap ρ':>9} | {'realized rg':>13} | "
          f"{'rg (cc=0)':>10} | {'rg (cc=true)':>12} | {'Δ cc=0':>8}")
    print("-" * 78)
    rows = []
    for rg_target in RG_GRID:
        base0 = None
        baset = None
        for rho in RHO_GRID:
            realized, r0, rt = [], [], []
            for rep in range(REPS):
                rng = np.random.default_rng(20 + rep)
                b1, b2 = R.sim_effects("polygenic", rg_target, rng)
                realized.append(R.realized_rg(b1, b2))
                bh1, bh2 = R.sumstats_pair(b1, b2, N1, N2, rng, rho_e=rho)
                r0.append(fit_rg(bh1, bh2, 0.0, rep))
                rt.append(fit_rg(bh1, bh2, rho, rep))
            truth = np.asarray(realized, float)
            r0 = np.asarray(r0, float)
            rt = np.asarray(rt, float)
            m0, s0, mae0, fail0 = R._estimate_summary(r0, truth)
            mt, st, maet, failt = R._estimate_summary(rt, truth)
            if base0 is None:
                base0 = r0.copy()
                baset = rt.copy()
            delta0, delta0_sd = _paired_delta(r0, base0)
            deltat, deltat_sd = _paired_delta(rt, baset)
            finite_truth = truth[np.isfinite(truth)]
            row = {
                "rg_target": rg_target, "overlap_rho": rho,
                "rg_realized_mean": round(float(np.mean(finite_truth)), 4),
                "rg_realized_sd": round(float(np.std(finite_truth)), 4),
                "rg_cc0": m0, "rg_cc0_sd": s0,
                "rg_cc0_mae_realized": mae0,
                "rg_cc0_fail": fail0,
                "rg_cctrue": mt, "rg_cctrue_sd": st,
                "rg_cctrue_mae_realized": maet,
                "rg_cctrue_fail": failt,
                "delta_cc0_vs_no_overlap": delta0,
                "delta_cc0_sd": delta0_sd,
                "delta_cctrue_vs_no_overlap": deltat,
                "delta_cctrue_sd": deltat_sd,
            }
            rows.append(row)
            print(f"{rg_target:>7.2f} {rho:>9.2f} | "
                  f"{row['rg_realized_mean']:>6.3f}±"
                  f"{row['rg_realized_sd']:<5.3f} | {_display(m0):>10} | "
                  f"{_display(mt):>12} | "
                  f"{_display(row['delta_cc0_vs_no_overlap']):>8}")
    with open(os.path.join(HERE, "overlap_estimation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nThe delta columns are paired against the rho=0 run. Interpret the "
          "correction from their means, SDs, and failure counts rather than as a "
          "general recovery guarantee.\nwrote overlap_estimation.csv")


if __name__ == "__main__":
    main()
