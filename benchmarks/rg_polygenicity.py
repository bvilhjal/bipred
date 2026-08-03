"""Genetic-correlation recovery across polygenicity.

Sweeps the causal fraction ``p`` in {0.1, 0.01, 0.001, 0.0001} at a fixed target
effect correlation, comparing bivariate LDSC (``ldsc_rg``) and bivariate
LDpred3 (``ldpred3_auto_bivariate_blocks``) against each replicate's realized
genetic correlation. The LD is the realistic non-repeating coalescent model of
``rg_architectures.py``. The committed
``rg_polygenicity.csv`` uses that benchmark's default geometry (m = 5,000);
raising msprime's mutation rate (with a larger ``K``) makes each segment yield
more SNPs for an optional larger-m run (see the invocation below), rather than
adding many small blocks.

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

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _benchmark_utils import peak_rss_bytes                         # noqa: E402
import rg_architectures as R                                      # noqa: E402
from bipred import ldsc_rg, ldpred3_auto_bivariate_blocks         # noqa: E402

RG = float(os.environ.get("RG", "0.5"))          # target effect correlation
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
    realized, n_causal, ld, bp, t_ld, t_bp = [], [], [], [], [], []
    for rep in range(REPS):
        rng = np.random.default_rng(base_seed + rep)
        b1, b2 = R.sim_effects("polygenic", RG, rng, p=p)       # new phenotype
        realized.append(R.realized_rg(b1, b2))
        n_causal.append(np.count_nonzero(b1))
        bh1, bh2 = R.sumstats_pair(b1, b2, R.N1, R.N2, rng)
        t0 = time.perf_counter()
        ld.append(ldsc_rg(bh1, bh2, ell, R.N1, R.N2, n_blocks=R.NB).rg)
        t_ld.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        bp.append(ldpred3_auto_bivariate_blocks(ref, bh1, bh2, R.N1, R.N2,
                                                burn_in=BURN, num_iter=ITER,
                                                seed=rep).rg)
        t_bp.append(time.perf_counter() - t0)
    return (np.array(realized, float), np.array(n_causal, int),
            np.array(ld, float), np.array(bp, float),
            float(np.mean(t_ld)), float(np.mean(t_bp)))


def main():
    csv_path = os.path.join(HERE, os.environ.get("OUT", "rg_polygenicity") + ".csv")
    print(f"Genetic correlation vs polygenicity — realistic non-repeating LD "
          f"(m={R.M:,} = {R.NB}x{R.K}, target effect corr={RG}, Nref={R.NREF}, "
          f"N1={R.N1}/N2={R.N2}, {REPS} reps)\n", flush=True)
    # Warm the sampler's Numba kernels (also forces the one-time LD build/load).
    _rng = np.random.default_rng(0)
    _ref, _ = R.ref_panel(0)
    _b1, _b2 = R.sim_effects("polygenic", RG, _rng, p=0.01)
    _bh1, _bh2 = R.sumstats_pair(_b1, _b2, R.N1, R.N2, _rng)
    ldpred3_auto_bivariate_blocks(_ref, _bh1, _bh2, R.N1, R.N2,
                                  burn_in=5, num_iter=5)

    print(f"{'p':>8} | {'n causal':>12} | {'realized rg':>14} | "
          f"{'rg LDSC':>14} | {'rg LDpred3':>14}")
    print("-" * 78)
    rows = []
    t0 = time.time()
    for pi, p in enumerate(PS):
        realized, n_causal, ld, bp, t_ld, t_bp = run_cell(
            p, base_seed=1000 + 100 * pi)
        truth = realized[np.isfinite(realized)]
        ld_mean, ld_sd, ld_mae, ld_fail = R._estimate_summary(ld, realized)
        bp_mean, bp_sd, bp_mae, bp_fail = R._estimate_summary(bp, realized)
        # sim_effects forces one causal variant when the Bernoulli draw is empty.
        n_expected = p * R.M + (1.0 - p) ** R.M
        row = {"p": p, "n_causal_expected": round(n_expected, 3),
               "n_causal_mean": round(float(np.mean(n_causal)), 3),
               "n_causal_sd": round(float(np.std(n_causal)), 3),
               "rg_target": RG,
               "rg_realized_mean": (round(float(np.mean(truth)), 4)
                                    if truth.size else ""),
               "rg_realized_sd": (round(float(np.std(truth)), 4)
                                  if truth.size else ""),
               "ldsc_rg": ld_mean, "ldsc_sd": ld_sd,
               "ldsc_mae_realized": ld_mae, "ldsc_fail": ld_fail,
               "ldpred3_rg": bp_mean, "ldpred3_sd": bp_sd,
               "ldpred3_mae_realized": bp_mae, "ldpred3_fail": bp_fail,
               "ldsc_t": round(t_ld, 4), "ldpred3_t": round(t_bp, 3)}
        rows.append(row)
        ld_fail_text = f" [{ld_fail}f]" if ld_fail else ""
        bp_fail_text = f" [{bp_fail}f]" if bp_fail else ""
        print(f"{p:>8} | {row['n_causal_mean']:>6.1f}±{row['n_causal_sd']:<4.1f}"
              f" | {row['rg_realized_mean']!s:>7}±"
              f"{row['rg_realized_sd']!s:<6}"
              f" | {row['ldsc_rg']!s:>7}±{row['ldsc_sd']!s:<6}{ld_fail_text}"
              f" | {row['ldpred3_rg']!s:>7}±"
              f"{row['ldpred3_sd']!s:<6}{bp_fail_text}", flush=True)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    mem_gb = peak_rss_bytes() / 1e9
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

    # Estimates versus realized genetic correlation at each polygenicity.
    ax.axhline(RG, ls=":", c="k", lw=1, alpha=.4,
               label=f"effect-correlation target={RG}")
    ax.errorbar(xs, [num(r["rg_realized_mean"]) for r in rows],
                [num(r["rg_realized_sd"]) for r in rows], fmt="x--", ms=5,
                capsize=3, color="0.35", label="realized genetic rg")
    ax.errorbar(xs, [num(r["ldsc_rg"]) for r in rows],
                [num(r["ldsc_sd"]) for r in rows], fmt="o-", ms=5, capsize=3,
                color="C0", label="bivariate LDSC")
    ax.errorbar(xs, [num(r["ldpred3_rg"]) for r in rows],
                [num(r["ldpred3_sd"]) for r in rows], fmt="s-", ms=5, capsize=3,
                color="C3", label="bivariate LDpred3")
    ax.set_xscale("log")
    ax.set_xlabel("polygenicity p (causal fraction)")
    ax.set_ylabel("estimated rg")
    ax.set_title("Recovery")
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
