"""Genetic correlation under **environmental** correlation from overlapping samples.

The hard case for rg: two traits measured on the *same* individuals whose
**environments** are correlated (`re`) even though the traits may be genetically
uncorrelated. The shared environment makes the phenotypes correlate, which a
naive genetic-correlation estimate reads as genetic — a false positive. This
stress-tests the proposed corrections; it does not assume that they recover the
true **genetic** rg in every cell:

  - bivariate **LDSC** with a free or constrained cross-trait intercept.
  - bivariate **LDpred3** with ``cross_corr`` set to the phenotypic correlation on
    the overlap (here read straight off the shared cohort, or from the LDSC
    intercept), compared with ``cross_corr=0``.

Real individual-level genotypes/phenotypes (the independent-block coalescent genome
of ``infer_vs_ldsc_sbayes``) are used so the confounding arises mechanistically:
both GWAS run on the *same* people, genetic effects have target correlation
``rg`` and the residual environments have correlation ``re``. The finite
effects' realized genetic correlation is recorded separately. Needs ``msprime``.

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
    """Simulate two phenotypes with target genetic corr ``rg`` and env corr ``re``.

    Return the standardized marginal GWAS, realized phenotypic correlation, and
    realized genetic correlation of the finite simulated genetic components.
    """
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
    rg_realized = float(np.corrcoef(g1, g2)[0, 1])
    # environments correlated by re, on the SAME individuals
    Le = np.linalg.cholesky([[1.0, re], [re, 1.0]])
    e = (Le @ rng.standard_normal((2, N))) * np.sqrt(1.0 - H2)
    y1 = g1 + e[0]; y2 = g2 + e[1]
    y1 = (y1 - y1.mean()) / y1.std(); y2 = (y2 - y2.mean()) / y2.std()
    rho_pheno = float(np.corrcoef(y1, y2)[0, 1])          # overlap noise correlation
    bhat1 = (Zg.T @ y1) / N                                # standardized marginal effects
    bhat2 = (Zg.T @ y2) / N
    return bhat1, bhat2, rho_pheno, rg_realized


def run_cell(rg, re, seed):
    out = {k: [] for k in ("ldsc_free", "ldsc_con", "biv_cc0", "biv_cc")}
    icpt, realized = [], []
    gg = G.genome(0)                       # one fixed LD genome; average over the
    for rep in range(REPS):                # phenotype / effect / environment draw
        rng = np.random.default_rng(seed + rep)
        bhat1, bhat2, rho, rg_realized = simulate(gg, rg, re, rng)
        realized.append(rg_realized)
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
    return out, np.asarray(realized, float), float(np.mean(icpt))


def agg(x, realized):
    """Summarize retained estimates and their per-replicate realized-truth error."""
    x = np.asarray(x, float)
    realized = np.asarray(realized, float)
    keep = np.isfinite(x) & np.isfinite(realized) & (np.abs(x) <= 1.5)
    if not keep.any():
        return float("nan"), float("nan"), float("nan"), int(x.size)
    return (float(np.mean(x[keep])), float(np.std(x[keep])),
            float(np.mean(np.abs(x[keep] - realized[keep]))),
            int((~keep).sum()))


def main():
    rows = []
    print(f"Genetic correlation under environmental overlap — real genotypes, both "
          f"traits on the SAME N={N} individuals (m={G.M}, h2={H2}, {REPS} reps)\n")
    print("Summary means/SDs exclude non-finite estimates and |rg| > 1.5; inspect "
          "raw runs when diagnosing divergence.\n")
    print(f"{'rg target/real':>14} {'re':>4} | {'LDSC free':>13} | "
          f"{'LDSC icpt=0':>13} | {'biv cc=0':>13} | {'biv cc=rho':>13} | "
          f"{'LDSC icpt':>9}")
    print("-" * 98)
    for rg, re in CELLS:
        out, realized, icpt = run_cell(
            rg, re, seed=2000 + int(rg * 10) * 100 + int(re * 10))
        r = {"rg_target": rg,
             "rg_realized": round(float(np.mean(realized)), 4),
             "rg_realized_sd": round(float(np.std(realized)), 4),
             "re": re}
        for k in ("ldsc_free", "ldsc_con", "biv_cc0", "biv_cc"):
            mu, sd, mae, fail = agg(out[k], realized)
            r[k] = round(mu, 4); r[k + "_sd"] = round(sd, 4)
            r[k + "_mae_realized"] = round(mae, 4)
            r[k + "_fail"] = fail
        r["ldsc_intercept"] = round(icpt, 4)
        rows.append(r)
        print(f"{rg:>5.2f}/{r['rg_realized']:<5.2f} {re:>4} | "
              f"{r['ldsc_free']:>6}±{r['ldsc_free_sd']:<5} | "
              f"{r['ldsc_con']:>6}±{r['ldsc_con_sd']:<5} | "
              f"{r['biv_cc0']:>6}±{r['biv_cc0_sd']:<5} | "
              f"{r['biv_cc']:>6}±{r['biv_cc_sd']:<5} | {r['ldsc_intercept']:>9}")
    with open(os.path.join(HERE, "rg_env_overlap.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nCSV MAE columns compare each retained estimate with that replicate's "
          "realized genetic rg. This stress test can expose an unstable or biased "
          "correction.")
    print("wrote rg_env_overlap.csv")


if __name__ == "__main__":
    main()
