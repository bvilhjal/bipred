"""Genetic-correlation recovery across polygenicity, at a larger m.

Sweeps the causal fraction ``p`` in {0.1, 0.01, 0.001, 0.0001} at a fixed true
``rg``, comparing bivariate LDSC (``ldsc_rg``) and bivariate LDpred3
(``ldpred3_auto_bivariate_blocks``) against the truth. The LD is the realistic
non-repeating coalescent model of ``rg_architectures.py``, pushed to a **larger
m** by raising msprime's mutation rate so each segment yields more SNPs (larger
``K`` per block) rather than by adding many small blocks.

The population LD (the "genotypes for LD") is **simulated once** and cached on
disk by ``rg_architectures`` (keyed by the geometry + mutation rate); every
polygenicity and replicate reuses it. Within a cell the **genotypes are held
fixed** (one reference panel) and each replicate redraws only the **phenotype**
(a fresh causal set, effect sizes and GWAS sampling noise), so the spread across
reps measures each method's phenotype-sampling accuracy at that polygenicity —
not genotype-panel noise. So the expensive coalescent simulation runs once for
the whole sweep.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        NB=100 K=1000 MUT_RATE=3e-7 python benchmarks/rg_polygenicity.py
    # m = NB*K = 100,000 here; raise MUT_RATE with K so segments yield >= K SNPs.
Needs ``msprime``. Single core recommended.
"""
import os
import sys
import csv
import time
import resource

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rg_architectures as R                                      # noqa: E402
from bipred import ldsc_rg, ldpred3_auto_bivariate_blocks         # noqa: E402

RG = float(os.environ.get("RG", "0.5"))          # true genetic correlation
# Polygenicity sweep; env P runs a single value (one worker per p, for parallel
# runs), else the full set. env OUT sets the csv basename (and skips the figure).
PS = ([float(os.environ["P"])] if os.environ.get("P")
      else [0.1, 0.01, 0.001, 0.0001])
REPS = int(os.environ.get("REPS", "5"))
BURN, ITER = 150, 180


def run_cell(p, base_seed):
    # Hold the genotypes / LD fixed across reps (one reference panel) and redraw
    # only the PHENOTYPE each rep -- a fresh causal set, effect sizes and GWAS
    # sampling noise. The spread across reps then reflects phenotype-sampling
    # variability of each method's rg estimate (its accuracy at this p), not
    # genotype-panel variability.
    ref, ell = R.ref_panel(0)                    # same genotypes for every rep
    ld, bp, t_ld, t_bp = [], [], [], []
    for rep in range(REPS):
        rng = np.random.default_rng(base_seed + rep)
        b1, b2 = R.sim_effects("polygenic", RG, rng, p=p)       # new phenotype
        bh1, bh2 = R.sumstats_pair(b1, b2, R.N1, R.N2, rng)
        t0 = time.perf_counter()
        ld.append(ldsc_rg(bh1, bh2, ell, R.N1, R.N2, n_blocks=R.NB).rg)
        t_ld.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        bp.append(ldpred3_auto_bivariate_blocks(ref, bh1, bh2, R.N1, R.N2,
                                                burn_in=BURN, num_iter=ITER,
                                                seed=rep).rg)
        t_bp.append(time.perf_counter() - t0)
    return (np.array(ld, float), np.array(bp, float),
            float(np.mean(t_ld)), float(np.mean(t_bp)))


def main():
    csv_path = os.path.join(HERE, os.environ.get("OUT", "rg_polygenicity") + ".csv")
    print(f"Genetic correlation vs polygenicity — realistic non-repeating LD "
          f"(m={R.M:,} = {R.NB}x{R.K}, true rg={RG}, Nref={R.NREF}, "
          f"N1={R.N1}/N2={R.N2}, {REPS} reps)\n", flush=True)
    # Warm the sampler's Numba kernels (also forces the one-time LD build/load).
    _rng = np.random.default_rng(0)
    _ref, _ = R.ref_panel(0)
    _b1, _b2 = R.sim_effects("polygenic", RG, _rng, p=0.01)
    _bh1, _bh2 = R.sumstats_pair(_b1, _b2, R.N1, R.N2, _rng)
    ldpred3_auto_bivariate_blocks(_ref, _bh1, _bh2, R.N1, R.N2,
                                  burn_in=5, num_iter=5)

    print(f"{'p':>8} | {'n_causal':>8} | {'rg LDSC':>14} | {'rg LDpred3':>14} | "
          f"{'t LDSC':>7} | {'t LDpred3':>9}")
    print("-" * 78)
    rows = []
    t0 = time.time()
    for pi, p in enumerate(PS):
        ld, bp, t_ld, t_bp = run_cell(p, base_seed=1000 + 100 * pi)
        ok = np.isfinite(ld) & (np.abs(ld) <= 1.5)
        ldv = ld[ok]
        row = {"p": p, "n_causal": int(round(p * R.M)),
               "rg_true": RG,
               "ldsc_rg": round(float(np.mean(ldv)), 4) if ldv.size else "",
               "ldsc_sd": round(float(np.std(ldv)), 4) if ldv.size else "",
               "ldsc_fail": int((~ok).sum()),
               "ldpred3_rg": round(float(np.mean(bp)), 4),
               "ldpred3_sd": round(float(np.std(bp)), 4),
               "ldsc_t": round(t_ld, 4), "ldpred3_t": round(t_bp, 3)}
        rows.append(row)
        fail = f" [{row['ldsc_fail']}f]" if row["ldsc_fail"] else ""
        print(f"{p:>8} | {row['n_causal']:>8} | "
              f"{row['ldsc_rg']!s:>7}±{row['ldsc_sd']!s:<6}{fail}"
              f" | {row['ldpred3_rg']!s:>7}±{row['ldpred3_sd']!s:<6}"
              f" | {t_ld:>6.2f}s | {t_bp:>8.2f}s", flush=True)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_gb = mem / 1e9 if sys.platform == "darwin" else mem / 1e6
    print(f"\npeak RSS {mem_gb:.2f} GB  (incl. the one-time LD build)  "
          f"| total {time.time() - t0:.0f}s")
    if not os.environ.get("OUT"):                # single-p workers skip the figure
        make_figure(rows)
    print(f"wrote {csv_path}")


def make_figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib absent: no figure)")
        return

    def num(v):
        return float(v) if v not in ("", None) else np.nan

    xs = [r["p"] for r in rows]
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # accuracy: estimated rg vs polygenicity
    ax.axhline(RG, ls="--", c="k", lw=1, alpha=.5, label=f"true rg={RG}")
    ax.errorbar(xs, [num(r["ldsc_rg"]) for r in rows],
                [num(r["ldsc_sd"]) for r in rows], fmt="o-", ms=5, capsize=3,
                color="C0", label="bivariate LDSC")
    ax.errorbar(xs, [num(r["ldpred3_rg"]) for r in rows],
                [num(r["ldpred3_sd"]) for r in rows], fmt="s-", ms=5, capsize=3,
                color="C3", label="bivariate LDpred3")
    ax.set_xscale("log")
    ax.set_xlabel("polygenicity p (causal fraction)")
    ax.set_ylabel("estimated rg")
    ax.set_title("Accuracy")
    ax.grid(alpha=.3, which="both")
    ax.legend()

    # running time per fit vs polygenicity (log-y: LDSC and LDpred3 differ ~10x)
    axt.semilogy(xs, [r["ldpred3_t"] for r in rows], "s-", ms=5, color="C3",
                 label="bivariate LDpred3")
    axt.semilogy(xs, [r["ldsc_t"] for r in rows], "o-", ms=5, color="C0",
                 label="bivariate LDSC")
    axt.set_xscale("log")
    axt.set_xlabel("polygenicity p (causal fraction)")
    axt.set_ylabel("time per fit (s)")
    axt.set_title("Running time")
    axt.grid(alpha=.3, which="both")
    axt.legend()

    fig.suptitle(f"Genetic correlation vs polygenicity "
                 f"(m={R.M:,}, realistic non-repeating LD, single core)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "rg_polygenicity.png"), dpi=130)


if __name__ == "__main__":
    main()
