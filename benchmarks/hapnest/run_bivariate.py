"""Bivariate LDpred3 on HAPNEST-simulated genotypes + phenotypes (real-LD benchmark).

The bivariate benchmarks elsewhere in this directory simulate their genome with a
coalescent (``msprime``) or a cached ``ld_library.npz``. This one instead consumes
a **HAPNEST** dataset -- synthetic genotypes resampled from a real 1000G+HGDP
reference (so they carry real LD, MAF spectra and population structure) together
with HAPNEST's own multi-trait phenotypes -- and scores bipred against the ground
truth HAPNEST was configured with.

HAPNEST (Wharrie et al., Bioinformatics 2023;
https://github.com/intervene-EU-H2020/synthetic_data) writes the ground truth into
the config used to generate the data, so nothing needs to be inferred from truth
files:

  * genetic correlation  <- ``TraitCorr``     (flattened nTrait x nTrait matrix)
  * per-trait h2          <- ``ProportionGeno``
  * causal overlap        <- ``Pleiotropy`` / the shared ``causal_list``

Pass those truths on the command line (they must match the config used for
``generate_pheno``) so the run is scored automatically. See ``README.md`` for the
two-step workflow (run HAPNEST, then this script) and ``config_bivariate.yaml``
for a ready two-trait config.

What it does with one HAPNEST fileset (``--geno PREFIX``, ``--pheno FILE``):

  1. split samples into GWAS / LD-reference / held-out test sets,
  2. run a per-trait marginal GWAS on the GWAS split (standardized effects),
  3. build per-block reference LD + LD scores from the reference split
     (or a separate ``--ref`` panel),
  4. estimate rg with cross-trait LDSC (``ldsc_rg``) and jointly with
     ``ldpred3_auto_bivariate_blocks`` (overlap term from the LDSC intercept),
  5. score estimated rg / h2 / MiXeR overlap against the passed truths and report
     out-of-sample PRS R2 on the test split -- bivariate vs a univariate ldpred3
     baseline, the gain that motivates the joint fit.

Writes ``run_bivariate.csv``. Needs ``ldpred3`` and ``bipred`` installed; HAPNEST
is only needed to produce the inputs.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/hapnest/run_bivariate.py \
        --geno data/output/synthetic --pheno data/output/synthetic.pheno \
        --rg-true 0.5 --h2-true 0.5 0.5 --overlap-true 1.0
"""
import os
import sys
import csv
import time
import argparse
import dataclasses

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from ldpred3.genotype_io import read_plink                    # noqa: E402
from ldpred3.ld import compute_ld_blocks                      # noqa: E402
from ldpred3 import ld_scores                                 # noqa: E402
from bipred import ldsc_rg, ldpred3_auto_bivariate_blocks     # noqa: E402

try:                                                          # univariate PRS baseline
    from ldpred3 import ldpred3_by_blocks
    HAVE_UNI = True
except Exception:                                             # pragma: no cover
    HAVE_UNI = False

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
#  Genotype / phenotype loading                                               #
# --------------------------------------------------------------------------- #
def read_pheno(path, iids, traits):
    """Read a whitespace-delimited HAPNEST phenotype file aligned to ``iids``.

    Assumes ``FID IID <trait0> <trait1> ...`` (a header row is auto-detected and
    skipped). ``traits`` are 0-based indices into the trait columns (the columns
    after FID/IID). Returns an ``(n, 2)`` float array in the order of ``iids``.
    """
    by_iid = {}
    with open(path) as fh:
        for ln, line in enumerate(fh):
            f = line.split()
            if len(f) < 3:
                continue
            if ln == 0:                                       # header?
                try:
                    float(f[2])
                except ValueError:
                    continue
            vals = []
            for c in traits:
                try:
                    vals.append(float(f[2 + c]))
                except (IndexError, ValueError):
                    vals.append(np.nan)
            by_iid[f[1]] = vals
    y = np.array([by_iid.get(str(i), [np.nan, np.nan]) for i in iids], float)
    if np.isnan(y).any():
        raise SystemExit("phenotype file does not cover all samples (or --traits "
                         "columns are wrong)")
    return y


def standardize(dosage, f=None, sd=None):
    """Standardize an ``(n, m)`` A1-dosage matrix (``-1`` = missing).

    Missing entries are mean-imputed. When ``f``/``sd`` are given (e.g. from the
    training split) they are reused instead of recomputed -- so a test set is
    standardized on training-set statistics, as a deployed PRS would be.
    """
    X = dosage.astype(np.float64)
    miss = X < 0
    if f is None:
        col = np.where(miss, np.nan, X)
        f = np.nanmean(col, 0) / 2.0
    X[miss] = np.take(2.0 * f, np.where(miss)[1])
    if sd is None:
        sd = X.std(0)
    good = sd > 0
    Z = np.zeros_like(X)
    Z[:, good] = (X[:, good] - 2.0 * f[good]) / sd[good]
    return Z, f, sd, good


def gwas(Z, y):
    """Standardized marginal effects for a standardized phenotype (``Z.T y / n``)."""
    ys = (y - y.mean()) / (y.std() + 1e-300)
    return (Z.T @ ys) / Z.shape[0]


# --------------------------------------------------------------------------- #
#  Driver                                                                      #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geno", required=True, help="PLINK prefix of the HAPNEST fileset")
    ap.add_argument("--pheno", required=True, help="HAPNEST phenotype file")
    ap.add_argument("--ref", default=None,
                    help="optional separate PLINK prefix for the LD reference "
                         "(default: a held-out subset of --geno)")
    ap.add_argument("--traits", default="0,1",
                    help="0-based trait-column pair, e.g. '0,1'")
    ap.add_argument("--rg-true", type=float, required=True)
    ap.add_argument("--h2-true", type=float, nargs=2, required=True)
    ap.add_argument("--overlap-true", type=float, default=float("nan"),
                    help="true frac_shared = shared / min(causal1, causal2)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--ref-n", type=int, default=2000,
                    help="reference-panel size when --ref is not given")
    ap.add_argument("--block-size", type=int, default=500)
    ap.add_argument("--burn-in", type=int, default=200)
    ap.add_argument("--num-iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    traits = [int(c) for c in args.traits.split(",")]

    t0 = time.time()
    g = read_plink(args.geno)
    n, m = g.dosage.shape
    chrom = np.asarray(g.variants.chrom)
    y = read_pheno(args.pheno, g.samples.iid, traits)
    print(f"loaded {n} samples x {m} variants; phenotypes for traits {traits}",
          flush=True)

    # sample split: test (held out for PRS) / ref (LD) / gwas (rest) --------- #
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_test = int(round(args.test_frac * n))
    test_idx = perm[:n_test]
    rest = perm[n_test:]
    if args.ref is None:
        ref_n = min(args.ref_n, rest.size - 1)
        ref_idx = rest[:ref_n]
        gwas_idx = rest[ref_n:]
        ref_dosage = g.dosage[ref_idx]
    else:
        gwas_idx = rest
        gr = read_plink(args.ref)
        # HAPNEST panels share the variant set; intersect by id and keep --geno order
        pos = {vid: i for i, vid in enumerate(gr.variants.id)}
        take = np.array([pos[v] for v in g.variants.id if v in pos], np.int64)
        keep = np.array([i for i, v in enumerate(g.variants.id) if v in pos], np.int64)
        if keep.size != m:
            g = _subset_variants(g, keep)
            chrom = chrom[keep]
            m = keep.size
        ref_dosage = gr.dosage[:, take]
    print(f"split: {gwas_idx.size} GWAS / {ref_dosage.shape[0]} ref / "
          f"{test_idx.size} test", flush=True)

    # standardize the GWAS split (freq/sd reused for the test set) ----------- #
    Zg, f, sd, good = standardize(g.dosage[gwas_idx])
    bhat1 = gwas(Zg[:, good], y[gwas_idx, 0])
    bhat2 = gwas(Zg[:, good], y[gwas_idx, 1])
    # keep only polymorphic variants throughout
    chrom, ref_dosage = chrom[good], ref_dosage[:, good]
    n_gwas = float(gwas_idx.size)

    # reference LD + LD scores ---------------------------------------------- #
    blocks = compute_ld_blocks(ref_dosage, chrom=chrom, block_size=args.block_size)
    n_blocks = len(blocks)
    ell = ld_scores(blocks, n_ref=ref_dosage.shape[0])
    print(f"built {n_blocks} LD blocks over {good.sum()} polymorphic variants "
          f"({time.time() - t0:.0f}s)", flush=True)

    # rg: cross-trait LDSC (free intercept) then the joint fit --------------- #
    ld = ldsc_rg(bhat1, bhat2, ell, n_gwas, n_gwas,
                 n_blocks=min(200, n_blocks))
    cross = float(ld.gcov_intercept)                          # sample-overlap term
    res = ldpred3_auto_bivariate_blocks(blocks, bhat1, bhat2, n_gwas, n_gwas,
                                        cross_corr=cross, burn_in=args.burn_in,
                                        num_iter=args.num_iter, seed=args.seed)
    mix = res.mixer

    # out-of-sample PRS on the held-out test split -------------------------- #
    Zt, _, _, _ = standardize(g.dosage[test_idx][:, good], f=f[good], sd=sd[good])
    r2_biv = (_r2(Zt @ res.beta1_est, y[test_idx, 0]),
              _r2(Zt @ res.beta2_est, y[test_idx, 1]))
    if HAVE_UNI:
        s1 = ldpred3_by_blocks(blocks, bhat1, np.full(good.sum(), n_gwas),
                               method="auto", burn_in=args.burn_in,
                               num_iter=args.num_iter, seed=args.seed)
        s2 = ldpred3_by_blocks(blocks, bhat2, np.full(good.sum(), n_gwas),
                               method="auto", burn_in=args.burn_in,
                               num_iter=args.num_iter, seed=args.seed)
        r2_uni = (_r2(Zt @ s1, y[test_idx, 0]), _r2(Zt @ s2, y[test_idx, 1]))
    else:
        r2_uni = (float("nan"), float("nan"))

    row = {
        "rg_true": args.rg_true, "rg_ldsc": round(ld.rg, 4),
        "rg_joint": round(res.rg, 4),
        "h2_true1": args.h2_true[0], "h2_est1": round(res.h2[0], 4),
        "h2_true2": args.h2_true[1], "h2_est2": round(res.h2[1], 4),
        "overlap_true": args.overlap_true,
        "frac_shared": round(mix["frac_shared"], 4),
        "rho_beta": round(mix["rho_beta"], 4),
        "rg_from_overlap": round(mix["rg_from_overlap"], 4),
        "cross_corr": round(cross, 4),
        "prs_r2_uni1": round(r2_uni[0], 4), "prs_r2_biv1": round(r2_biv[0], 4),
        "prs_r2_uni2": round(r2_uni[1], 4), "prs_r2_biv2": round(r2_biv[1], 4),
        "prs_gain2": round(r2_biv[1] - r2_uni[1], 4),
        "time_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(HERE, "run_bivariate.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader(); w.writerow(row)

    print(f"\n  rg   true={row['rg_true']}  ldsc={row['rg_ldsc']}  "
          f"joint={row['rg_joint']}")
    print(f"  h2   true=({row['h2_true1']},{row['h2_true2']})  "
          f"est=({row['h2_est1']},{row['h2_est2']})")
    print(f"  overlap true={row['overlap_true']}  frac_shared={row['frac_shared']}"
          f"  rho_beta={row['rho_beta']}")
    print(f"  trait2 PRS R2  uni={row['prs_r2_uni2']}  biv={row['prs_r2_biv2']}  "
          f"gain={row['prs_gain2']:+}")
    print(f"wrote {os.path.join(HERE, 'run_bivariate.csv')}  ({row['time_s']}s)")


def _r2(pred, y):
    """Squared Pearson correlation between a PRS and a phenotype (0 if degenerate)."""
    if np.std(pred) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(pred, y)[0, 1] ** 2)


def _subset_variants(g, keep):
    """Restrict a Genotypes bundle to variant indices ``keep`` (kept in order)."""
    return dataclasses.replace(g, dosage=g.dosage[:, keep],
                               variants=g.variants.subset(keep))


if __name__ == "__main__":
    main()
