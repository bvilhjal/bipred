"""Genetic correlation under **environmental** correlation from overlapping samples.

The hard case for rg: two traits measured on the *same* individuals whose
**environments** are correlated (`re`) even though the traits may be genetically
uncorrelated. The shared environment makes the phenotypes correlate, which a
naive genetic-correlation estimate reads as genetic — a false positive. This
demonstrates that the corrections recover the true **genetic** rg regardless of
`re`:

  - bivariate **LDSC** with a free cross-trait intercept — the intercept absorbs
    the overlap automatically (no knowledge of `re` needed); constraining it to 0
    leaves the environmental confounding in the estimate.
  - bivariate **LDpred3** with ``cross_corr`` set to the phenotypic correlation on
    the overlap (here read straight off the shared cohort, or from the LDSC
    intercept); ``cross_corr=0`` leaves the bias.

Real individual-level genotypes/phenotypes (the independent-block coalescent genome
of ``infer_vs_ldsc_sbayes``) are used so the confounding arises mechanistically:
both GWAS run on the *same* people, genetic effects have correlation ``rg`` and the
residual environments have correlation ``re``. Needs ``msprime``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/rg_env_overlap.py
"""
import os
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bipred import ldsc_rg, ldpred3_auto_bivariate_blocks                 # noqa: E402
import infer_vs_ldsc_sbayes as G                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = G.N_GWAS                          # both traits on the SAME N individuals (full overlap)
H2 = 0.5
P = 0.1                               # shared causal fraction
REPS = int(os.environ.get("REPS", "20"))
# (rg, re): the headline is rg=0 with re>0 — genetically independent, environment-correlated.
CELLS = [(0.0, 0.0), (0.0, 0.3), (0.0, 0.6), (0.5, 0.0), (0.5, 0.6)]


def simulate(gg, rg, re, rng):
    """Two phenotypes on the shared GWAS cohort with genetic corr rg + env corr re;
    return the two traits' standardized marginal GWAS and the realized phenotypic
    correlation (the sampling-noise correlation induced by the full overlap)."""
    Zg, m = gg["Zg"], gg["m"]
    good = gg["good"]
    Lg = np.linalg.cholesky([[1.0, rg], [rg, 1.0]])
    c = (rng.random(m) < P) & good
    if not c.any():
        c[np.flatnonzero(good)[0]] = True
    raw = Lg @ rng.standard_normal((2, int(c.sum())))
    b1 = np.zeros(m); b2 = np.zeros(m)
    b1[c] = raw[0]; b2[c] = raw[1]
    g1 = Zg @ b1; g2 = Zg @ b2
    b1 *= np.sqrt(H2 / g1.var()); b2 *= np.sqrt(H2 / g2.var())
    g1 = Zg @ b1; g2 = Zg @ b2
    # environments correlated by re, on the SAME individuals
    Le = np.linalg.cholesky([[1.0, re], [re, 1.0]])
    e = (Le @ rng.standard_normal((2, N))) * np.sqrt(1.0 - H2)
    y1 = g1 + e[0]; y2 = g2 + e[1]
    y1 = (y1 - y1.mean()) / y1.std(); y2 = (y2 - y2.mean()) / y2.std()
    rho_pheno = float(np.corrcoef(y1, y2)[0, 1])          # overlap noise correlation
    bhat1 = (Zg.T @ y1) / N                                # standardized marginal effects
    bhat2 = (Zg.T @ y2) / N
    return bhat1, bhat2, rho_pheno


def run_cell(rg, re, seed):
    out = {k: [] for k in ("ldsc_free", "ldsc_con", "biv_cc0", "biv_cc")}
    icpt = []
    gg = G.genome(0)                       # one fixed LD genome; average over the
    for rep in range(REPS):                # phenotype / effect / environment draw
        rng = np.random.default_rng(seed + rep)
        bhat1, bhat2, rho = simulate(gg, rg, re, rng)
        ld, ell, n = gg["ld"], gg["ell"], float(N)
        free = ldsc_rg(bhat1, bhat2, ell, n, n, n_blocks=G.NB)
        out["ldsc_free"].append(free.rg)
        icpt.append(free.gcov_intercept)
        out["ldsc_con"].append(ldsc_rg(bhat1, bhat2, ell, n, n, n_blocks=G.NB,
                                       constrain_intercept=0.0).rg)
        out["biv_cc0"].append(ldpred3_auto_bivariate_blocks(
            ld, bhat1, bhat2, n, n, burn_in=150, num_iter=180,
            cross_corr=0.0, seed=rep).rg)
        out["biv_cc"].append(ldpred3_auto_bivariate_blocks(
            ld, bhat1, bhat2, n, n, burn_in=150, num_iter=180,
            cross_corr=rho, seed=rep).rg)
    return out, float(np.mean(icpt))


def agg(x):
    x = np.asarray(x, float)
    x = x[np.abs(x) <= 1.5]
    return (float(np.mean(x)), float(np.std(x))) if x.size else (float("nan"), float("nan"))


def main():
    rows = []
    print(f"Genetic correlation under environmental overlap — real genotypes, both "
          f"traits on the SAME N={N} individuals (m={G.M}, h2={H2}, {REPS} reps)\n")
    print(f"{'rg':>4} {'re':>4} | {'LDSC free':>13} | {'LDSC icpt=0':>13} | "
          f"{'biv cc=0':>13} | {'biv cc=rho':>13} | {'LDSC icpt':>9}")
    print("-" * 86)
    for rg, re in CELLS:
        out, icpt = run_cell(rg, re, seed=2000 + int(rg * 10) * 100 + int(re * 10))
        r = {"rg_true": rg, "re": re}
        for k in ("ldsc_free", "ldsc_con", "biv_cc0", "biv_cc"):
            mu, sd = agg(out[k])
            r[k] = round(mu, 4); r[k + "_sd"] = round(sd, 4)
        r["ldsc_intercept"] = round(icpt, 4)
        rows.append(r)
        print(f"{rg:>4} {re:>4} | {r['ldsc_free']:>6}±{r['ldsc_free_sd']:<5} | "
              f"{r['ldsc_con']:>6}±{r['ldsc_con_sd']:<5} | "
              f"{r['biv_cc0']:>6}±{r['biv_cc0_sd']:<5} | "
              f"{r['biv_cc']:>6}±{r['biv_cc_sd']:<5} | {r['ldsc_intercept']:>9}")
    with open(os.path.join(HERE, "rg_env_overlap.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nThe two corrected columns (LDSC free intercept, biv cc=rho) recover the "
          "true genetic rg;\nthe naive columns (icpt=0, cc=0) inflate with re — a "
          "spurious genetic correlation from shared environment.")
    print("wrote rg_env_overlap.csv")


if __name__ == "__main__":
    main()
