"""Heritability & polygenicity inference: LDSC vs LDpred3-auto vs GCTB SBayesS.

Sweeps a 2-D grid of **genetic architecture x true h2** on realistic coalescent
LD (fit from a finite **reference panel**, the dominant real-world error) and
estimates the summary-statistic quantities each tool reports:

* ``ldsc``     -- LD Score regression (``ldsc_h2``): h2 only.
* ``ldpred3``  -- LDpred3-auto (``ldpred3_auto_infer``): h2 + polygenicity p,
                  each with a 95% credible interval (so we also score coverage).
* ``sbayess``  -- GCTB ``--sbayes S`` (mixture + MAF/selection ``S`` term): its
                  ``.parRes`` reports h2 (``hsq``) and polygenicity (``Pi``).

All three consume the **same** simulated GWAS: LDSC/LDpred3 use the reference-panel
per-block LD, GCTB builds its **shrunk** LDM (``--make-shrunk-ldm``) from the same
PLINK reference fileset. Architectures include a **Laplace** (double-exponential)
effect distribution -- the true model behind the Laplace / lassosum2 prior -- and a
fat-tailed **t** architecture, alongside the usual infinitesimal / sparse /
polygenic / major-locus set.

Reports each estimate as mean +/- SD over reps against the known truth, the
LDpred3 interval coverage, and wall-clock time. Needs ``msprime`` and GCTB
(``GCTB=/path/to/gctb``; bioconda ``gctb``). Writes ``infer_ldsc_sbayes.{csv,png}``.

    GCTB=/opt/gctb/bin/gctb OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        python benchmarks/infer_vs_ldsc_sbayes.py
"""
import os
import sys
import csv
import math
import time
import tempfile
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldpred3.genotype_io import VariantTable, SampleTable, write_plink   # noqa: E402
from ldpred3.simulate import simulate_genotypes_by_mutation_rate         # noqa: E402
from ldpred3 import (ld_scores, ldsc_h2, ldpred3_auto_infer,             # noqa: E402
                     standardize_betas)
from ldpred3.ld import compute_ld_blocks                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GCTB = os.environ.get("GCTB", "gctb")
N_REF, N_GWAS = 4000, 20000
# Genome = NB *independent* coalescent segments, each trimmed to BLOCK_SIZE common
# SNPs, so the block-diagonal LD is exact (a continuous chromosome chopped into
# arbitrary blocks leaks cross-block LD that corrupts h2 inference).
NB, BLOCK_SIZE = 20, 150
SEG_LEN, MUT_RATE = 0.6e6, 3e-8              # per-segment length / SNP density
M = NB * BLOCK_SIZE
NE = 11400                                   # shrunk-LDM effective sample size
CHAIN, BURN = 6000, 2000                     # GCTB MCMC
REPS = int(os.environ.get("REPS", "5"))
H2_GRID = [0.2, 0.5, 0.8]
_ERF = np.vectorize(math.erf)

# name -> (causal fraction p, effect-size sampler on the causal SNPs)
ARCHS = {
    "infinitesimal": (1.0, lambda rng, k: rng.standard_normal(k)),
    "sparse":        (0.01, lambda rng, k: rng.standard_normal(k)),
    "polygenic":     (0.2, lambda rng, k: rng.standard_normal(k)),
    "laplace":       (0.1, lambda rng, k: rng.laplace(0.0, 1.0, k)),
    "fat_tail_t":    (0.1, lambda rng, k: rng.standard_t(3, k)),
    "major_locus":   (0.02, None),           # special-cased in make_beta
}


def make_beta(arch, m, good, rng):
    """Causal effect vector (unscaled) for one architecture."""
    p, sampler = ARCHS[arch]
    beta = np.zeros(m)
    if arch == "major_locus":                # few huge effects on a sparse bg
        c = (rng.random(m) < 0.02) & good
        beta[c] = rng.standard_normal(int(c.sum())) * 0.3
        maj = rng.choice(np.flatnonzero(good), 3, replace=False)
        beta[maj] = rng.choice([-1.0, 1.0], 3) * 4.0
        return beta
    c = (rng.random(m) < p) & good
    if not c.any():
        c[np.flatnonzero(good)[0]] = True
    beta[c] = sampler(rng, int(c.sum()))
    return beta


def true_p(arch, m, good):
    """Expected causal fraction (for coverage scoring of LDpred3's p)."""
    p, _ = ARCHS[arch]
    return 0.02 if arch == "major_locus" else p


_GENOME = {}


def genome(rep):
    """Genome for one replicate: ``NB`` independent coalescent segments (each an
    own msprime run, trimmed to ``BLOCK_SIZE`` SNPs so the block-diagonal LD is
    exact), plus everything that depends only on the genotypes -- standardized
    GWAS genotypes and the reference-panel LD blocks / LD scores.

    Memoised on ``rep`` and shared across all (arch, h2) cells so the estimators
    are compared on the *same* genotypes (a paired comparison) and each genome is
    simulated once, not once per cell.
    """
    if rep in _GENOME:
        return _GENOME[rep]
    cols = []
    for b in range(NB):
        mut = MUT_RATE
        for _ in range(4):                   # bump density until the segment fills
            Gb = simulate_genotypes_by_mutation_rate(
                N_REF + N_GWAS, SEG_LEN, mut_rate=mut, min_maf=0.02,
                seed=rep * NB + b + 1)
            if Gb.shape[1] >= BLOCK_SIZE:
                break
            mut *= 1.6
        if Gb.shape[1] < BLOCK_SIZE:
            raise RuntimeError("segment produced too few SNPs; raise MUT_RATE")
        cols.append(Gb[:, :BLOCK_SIZE])
    G = np.concatenate(cols, axis=1)
    m = G.shape[1]
    Gref = G[:N_REF].astype(np.int8)
    Gg = G[N_REF:].astype(float)
    f = Gg.mean(0) / 2.0
    sd = np.sqrt(2 * f * (1 - f))
    good = sd > 0
    Zg = np.where(good, (Gg - 2 * f) / np.where(good, sd, 1.0), 0.0)
    ld = [(R.astype(np.float64), idx) for R, idx in
          compute_ld_blocks(Gref, block_size=BLOCK_SIZE)]
    ell = ld_scores(ld, n_ref=N_REF)
    _GENOME[rep] = dict(m=m, Gref=Gref, Gg=Gg, Zg=Zg, f=f, sd=sd, good=good,
                        ld=ld, ell=ell)
    return _GENOME[rep]


def simulate(arch, h2, rep, pheno_seed):
    """A GWAS for one (arch, h2) on the replicate's fixed genome.

    Only the phenotype / summary statistics vary here; the genotypes and
    reference-panel LD come from the memoised :func:`genome`. Returns the pieces
    the three estimators consume (standardized effects + LD for LDSC/LDpred3, the
    per-allele freq/b/se/p for GCTB's ``.ma``).
    """
    gg = genome(rep)
    m, Gg, Zg, f, sd, good = (gg["m"], gg["Gg"], gg["Zg"], gg["f"], gg["sd"],
                              gg["good"])
    rng = np.random.default_rng(pheno_seed)
    beta = make_beta(arch, m, good, rng)
    gv = float(np.var(Zg @ beta))
    beta *= np.sqrt(h2 / gv) if gv > 0 else 1.0
    g = Zg @ beta
    y = g + rng.normal(0, np.sqrt(max(1e-6, 1 - g.var())), N_GWAS)

    Gc = Gg - Gg.mean(0)
    vG = Gc.var(0)
    vG[vG == 0] = 1.0
    b_allele = (Gc.T @ (y - y.mean())) / (N_GWAS * vG)
    se_allele = np.sqrt(np.maximum(1e-12, (y.var() - b_allele ** 2 * vG)
                                   / (N_GWAS * vG)))
    z = b_allele / se_allele
    pval = np.clip(2 * (1 - 0.5 * (1 + _ERF(np.abs(z) / 2 ** 0.5))), 1e-300, 1.0)
    bhat_std, _ = standardize_betas(b_allele, se_allele, np.full(m, float(N_GWAS)))
    return dict(m=m, f=f, sd=sd, Gref=gg["Gref"], b_allele=b_allele,
                se_allele=se_allele, pval=pval, bhat_std=bhat_std,
                ld=gg["ld"], ell=gg["ell"], p_true=true_p(arch, m, good))


# --------------------------------------------------------------------------- #
#  Estimators                                                                  #
# --------------------------------------------------------------------------- #
def est_ldsc(sim):
    n = float(N_GWAS)
    t0 = time.perf_counter()
    r = ldsc_h2(n * sim["bhat_std"] ** 2, sim["ell"], n, n_blocks=100)
    return dict(h2=r.h2, p=float("nan"), h2_lo=r.h2_ci[0], h2_hi=r.h2_ci[1],
                p_lo=float("nan"), p_hi=float("nan"), t=time.perf_counter() - t0)


def est_ldpred3(sim, seed):
    n = np.full(sim["m"], float(N_GWAS))
    r = ldpred3_auto_infer(sim["ld"], sim["bhat_std"], n, n_chains=8,
                           burn_in=150, num_iter=180, seed=seed)
    t0 = time.perf_counter()
    r = ldpred3_auto_infer(sim["ld"], sim["bhat_std"], n, n_chains=8,
                           burn_in=150, num_iter=180, seed=seed)
    return dict(h2=r.h2_est, p=r.p_est, h2_lo=r.h2_ci[0], h2_hi=r.h2_ci[1],
                p_lo=r.p_ci[0], p_hi=r.p_ci[1], t=time.perf_counter() - t0)


def _parse_parres(path):
    """GCTB ``.parRes``: label -> Mean (first numeric column on the row)."""
    out = {}
    with open(path) as fh:
        for line in fh:
            c = line.split()
            if len(c) >= 2:
                try:
                    out[c[0]] = float(c[1])
                except ValueError:
                    continue
    return out


def est_sbayess(sim, work, seed):
    m = sim["m"]
    # Independent blocks -> put a large genetic-distance gap between blocks so
    # GCTB's distance-shrunk LDM never couples cross-block pairs.
    blk = np.arange(m) // BLOCK_SIZE
    within = np.arange(m) % BLOCK_SIZE
    cm = blk * 20.0 + within * 0.001
    pos = (blk.astype(np.int64) * 10_000_000 + (within + 1) * 15000)
    variants = VariantTable(
        chrom=np.array(["1"] * m, object),
        id=np.array([f"rs{i}" for i in range(m)], object),
        cm=cm, pos=pos,
        a1=np.array(["A"] * m, object), a2=np.array(["G"] * m, object))
    smp = SampleTable(
        fid=np.array([f"R{i}" for i in range(N_REF)], object),
        iid=np.array([f"R{i}" for i in range(N_REF)], object),
        sex=np.ones(N_REF, np.int64), pheno=np.full(N_REF, np.nan))
    ref = os.path.join(work, "ref")
    write_plink(ref, sim["Gref"], variants, smp)
    with open(os.path.join(work, "gwas.ma"), "w") as fh:
        fh.write("SNP A1 A2 freq b se p N\n")
        for i in range(m):
            fh.write(f"rs{i} A G {sim['f'][i]:.6g} {sim['b_allele'][i]:.6g} "
                     f"{sim['se_allele'][i]:.6g} {sim['pval'][i]:.4g} {N_GWAS}\n")
    ldm = os.path.join(work, "L")
    b = subprocess.run([GCTB, "--bfile", ref, "--make-shrunk-ldm", "--ne", str(NE),
                        "--out", ldm], capture_output=True, text=True)
    if not os.path.exists(ldm + ".ldm.shrunk.bin"):
        sys.stderr.write(b.stdout + b.stderr)
        raise RuntimeError("gctb make-shrunk-ldm failed")
    out = os.path.join(work, "sbS")
    t0 = time.perf_counter()
    r = subprocess.run([GCTB, "--sbayes", "S", "--ldm", ldm + ".ldm.shrunk",
                        "--gwas-summary", os.path.join(work, "gwas.ma"),
                        "--chain-length", str(CHAIN), "--burn-in", str(BURN),
                        "--seed", str(seed + 1), "--out", out],
                       capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if not os.path.exists(out + ".parRes"):
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError("gctb sbayes S failed")
    par = _parse_parres(out + ".parRes")
    h2 = par.get("hsq", float("nan"))
    pi = par.get("Pi", float("nan"))
    return dict(h2=h2, p=pi, h2_lo=float("nan"), h2_hi=float("nan"),
                p_lo=float("nan"), p_hi=float("nan"), t=dt)


# --------------------------------------------------------------------------- #
#  Driver                                                                      #
# --------------------------------------------------------------------------- #
def run_cell(arch, h2, base_seed):
    """Return {method: list-of-per-rep dicts} for one (arch, h2) cell."""
    acc = {"ldsc": [], "ldpred3": [], "sbayess": []}
    for rep in range(REPS):
        sim = simulate(arch, h2, rep, pheno_seed=base_seed + rep)
        acc["ldsc"].append(est_ldsc(sim))
        acc["ldpred3"].append(est_ldpred3(sim, seed=rep))
        work = tempfile.mkdtemp(prefix="infer_sb_")
        try:
            acc["sbayess"].append(est_sbayess(sim, work, seed=rep))
        except RuntimeError as e:
            sys.stderr.write(f"  sbayess failed ({arch} h2={h2}): {e}\n")
            acc["sbayess"].append(dict(h2=float("nan"), p=float("nan"),
                                       h2_lo=float("nan"), h2_hi=float("nan"),
                                       p_lo=float("nan"), p_hi=float("nan"), t=0.0))
    return acc, sim["p_true"]


def summarize(arch, h2, acc, p_true, rows):
    def col(method, key):
        return np.array([d[key] for d in acc[method]], float)

    for method in ("ldsc", "ldpred3", "sbayess"):
        h2e = col(method, "h2")
        pe = col(method, "p")
        h2_cov = np.mean([(d["h2_lo"] <= h2 <= d["h2_hi"]) for d in acc[method]
                          if not math.isnan(d["h2_lo"])]) if not math.isnan(
                              acc[method][0]["h2_lo"]) else float("nan")
        p_cov = np.mean([(d["p_lo"] <= p_true <= d["p_hi"]) for d in acc[method]
                         if not math.isnan(d["p_lo"])]) if not math.isnan(
                             acc[method][0]["p_lo"]) else float("nan")
        rows.append({
            "arch": arch, "h2_true": h2, "p_true": round(p_true, 4),
            "method": method,
            "h2_mean": round(float(np.nanmean(h2e)), 4),
            "h2_sd": round(float(np.nanstd(h2e)), 4),
            "h2_cov": round(float(h2_cov), 2) if not math.isnan(h2_cov) else "",
            "p_mean": round(float(np.nanmean(pe)), 4) if not np.all(np.isnan(pe)) else "",
            "p_sd": round(float(np.nanstd(pe)), 4) if not np.all(np.isnan(pe)) else "",
            "p_cov": round(float(p_cov), 2) if not math.isnan(p_cov) else "",
            "time_s": round(float(np.mean(col(method, "t"))), 2),
        })


def main():
    # ARCH may be a comma-separated subset (parallel workers); OUT overrides the
    # CSV basename so concurrent workers don't clobber each other.
    only_arch = os.environ.get("ARCH")
    only_h2 = os.environ.get("H2")
    archs = only_arch.split(",") if only_arch else list(ARCHS)
    h2s = [float(only_h2)] if only_h2 else H2_GRID
    all_archs = list(ARCHS)
    rows = []
    base = os.environ.get("OUT", "infer_ldsc_sbayes")
    csv_path = os.path.join(HERE, base + ".csv")
    for arch in archs:
        ai = all_archs.index(arch)                # stable global index -> seed
        for hi, h2 in enumerate(h2s):
            print(f"=== {arch}  h2={h2} ===", flush=True)
            acc, p_true = run_cell(arch, h2, base_seed=1000 + 100 * ai + 10 * hi)
            summarize(arch, h2, acc, p_true, rows)
            for r in rows[-3:]:
                print(f"  {r['method']:>8}  h2={r['h2_mean']}±{r['h2_sd']} "
                      f"(cov {r['h2_cov']})  p={r['p_mean']}  ({r['time_s']}s)",
                      flush=True)
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    if not os.environ.get("OUT"):                 # full run -> also draw the figure
        make_figure(rows)
    print(f"wrote {csv_path}")


def make_figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    archs = sorted({r["arch"] for r in rows})
    h2s = sorted({r["h2_true"] for r in rows})
    methods = ["ldsc", "ldpred3", "sbayess"]
    colors = {"ldsc": "C0", "ldpred3": "C3", "sbayess": "C1"}
    fig, axes = plt.subplots(1, len(h2s), figsize=(4.6 * len(h2s), 4.4), sharey=True)
    if len(h2s) == 1:
        axes = [axes]
    x = np.arange(len(archs))
    w = 0.25
    for ax, h2 in zip(axes, h2s):
        for j, mth in enumerate(methods):
            ys = [next((r["h2_mean"] for r in rows if r["arch"] == a
                        and r["h2_true"] == h2 and r["method"] == mth), np.nan)
                  for a in archs]
            es = [next((r["h2_sd"] for r in rows if r["arch"] == a
                        and r["h2_true"] == h2 and r["method"] == mth), 0)
                  for a in archs]
            ax.bar(x + (j - 1) * w, ys, w, yerr=es, capsize=2,
                   label=mth, color=colors[mth])
        ax.axhline(h2, ls="--", c="k", lw=1, alpha=.6)
        ax.set_xticks(x)
        ax.set_xticklabels(archs, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"true h²={h2}")
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("estimated h²")
    axes[0].legend()
    fig.suptitle("h² inference — LDSC vs LDpred3-auto vs SBayesS (realistic LD, ref panel)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "infer_ldsc_sbayes.png"), dpi=130)


if __name__ == "__main__":
    main()
