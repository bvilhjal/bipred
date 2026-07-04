"""Scaling of genetic-correlation estimation with m: bivariate LDSC vs LDpred3.

Mirrors the software's other scaling simulations (``infer_scaling.py``,
``lowrank_scaling.py``): a geometric sweep of the variant count ``m``, each size
run in its **own subprocess** so the reported peak RSS is that size's own
footprint, writing ``{csv,png}`` incrementally. The LD is the realistic
non-repeating coalescent model of ``rg_architectures.py`` (K=200 SNPs/block, one
unique msprime segment per block), reused here so the scaling curve is measured on
the same realistic LD as the accuracy benchmark — each size builds ``m/K`` unique
blocks (``NB`` via the env hook), cached per size like the accuracy benchmark.

For each size one polygenic two-trait pair at a fixed true ``rg`` is simulated
from the population LD and both estimators are fitted against a finite reference
panel: bivariate LDSC (``ldsc_rg``) and bivariate LDpred3
(``ldpred3_auto_bivariate_blocks``, the default path incl. its two univariate
h²-cap pre-passes). Reports per-fit wall time, peak RSS and the recovered rg, so
the curve shows both the cost growth and that accuracy holds as m grows.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/rg_scaling.py
    python benchmarks/rg_scaling.py 5000 10000 20000 40000 80000   # custom m
Needs ``msprime``. Single core recommended.
"""
import os
import sys
import csv
import json
import time
import resource
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
K = 200                                # SNPs per (unique coalescent) block
RG = 0.5                               # true genetic correlation for the sweep
BURN, ITER = 150, 180                  # per-fit sampler sweeps (as rg_architectures)
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def _worker(m):
    """One bivariate LDSC + LDpred3 rg fit at size ``m`` in this process; prints a
    ``RESULT <json>`` line with the two fit times, peak RSS and recovered rg."""
    import numpy as np
    os.environ["NB"] = str(m // K)
    os.environ["K"] = str(K)
    sys.path.insert(0, HERE)
    import rg_architectures as R                       # builds NB unique blocks
    from bipred import ldsc_rg, ldpred3_auto_bivariate_blocks

    ref, ell = R.ref_panel(0)
    rng = np.random.default_rng(123)
    b1, b2 = R.sim_effects("polygenic", RG, rng)
    bh1, bh2 = R.sumstats_pair(b1, b2, R.N1, R.N2, rng)
    nb = m // K

    # Warm the Numba kernels once so the timed fits are steady-state.
    ldpred3_auto_bivariate_blocks(ref, bh1, bh2, R.N1, R.N2,
                                  burn_in=3, num_iter=3)

    t = time.perf_counter()
    rg_ldsc = ldsc_rg(bh1, bh2, ell, R.N1, R.N2, n_blocks=nb).rg
    t_ldsc = time.perf_counter() - t

    t = time.perf_counter()
    rg_bp = ldpred3_auto_bivariate_blocks(ref, bh1, bh2, R.N1, R.N2,
                                          burn_in=BURN, num_iter=ITER, seed=1).rg
    t_ldpred3 = time.perf_counter() - t

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_gb = mem / 1e9 if sys.platform == "darwin" else mem / 1e6
    print("RESULT " + json.dumps(
        {"m": m, "nb": nb, "t_ldsc": t_ldsc, "t_ldpred3": t_ldpred3,
         "mem_gb": mem_gb, "rg_ldsc": rg_ldsc, "rg_ldpred3": rg_bp,
         "rg_true": RG}))


def run(m):
    """Run one size in its own process; classify OOM vs other failure."""
    r = subprocess.run([sys.executable, __file__, "--worker", str(m)],
                       env=ENV, capture_output=True, text=True)
    if r.returncode == 0:
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")][-1]
        res = json.loads(line[7:])
        res["ok"] = True
        return res
    err = (r.stderr or "") + (r.stdout or "")
    killed = r.returncode in (-9, 137) or "MemoryError" in err or "Killed" in err
    return {"ok": False, "killed": killed,
            "msg": (err.strip().splitlines()[-1:] or [""])[0]}


def write_csv(rows):
    with open(os.path.join(HERE, "rg_scaling.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["m", "nb", "t_ldsc_s", "t_ldpred3_s", "peak_gb",
                    "rg_ldsc", "rg_ldpred3", "rg_true"])
        w.writerows(rows)


def main():
    sizes = [int(float(a)) for a in sys.argv[1:] if not a.startswith("--")] or \
            [5000, 10000, 20000, 40000, 80000]
    print(f"Genetic-correlation scaling with m — realistic non-repeating LD "
          f"(K={K}/block, true rg={RG}, {BURN}+{ITER} sweeps, single core)\n",
          flush=True)
    print(f"{'m':>8} | {'LDSC (s)':>9} | {'LDpred3 (s)':>11} | {'peak GB':>7} | "
          f"{'rg LDSC':>8} | {'rg LDpred3':>10}")
    print("-" * 72)
    rows = []
    for m in sizes:
        res = run(m)
        if res["ok"]:
            print(f"{m:>8,} | {res['t_ldsc']:>9.3f} | {res['t_ldpred3']:>11.2f} | "
                  f"{res['mem_gb']:>7.2f} | {res['rg_ldsc']:>8.3f} | "
                  f"{res['rg_ldpred3']:>10.3f}", flush=True)
            rows.append([m, m // K, round(res["t_ldsc"], 4),
                         round(res["t_ldpred3"], 3), round(res["mem_gb"], 3),
                         round(res["rg_ldsc"], 4), round(res["rg_ldpred3"], 4), RG])
        else:
            tag = "OOM" if res["killed"] else "FAIL"
            print(f"{m:>8,} | {tag}  ({res['msg'][:70]})", flush=True)
            rows.append([m, m // K, tag, tag, tag, tag, tag, RG])
        write_csv(rows)
    make_figure(rows)
    print("\nwrote rg_scaling.csv and rg_scaling.png")


def make_figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib absent: no figure)")
        return
    ok = [r for r in rows if isinstance(r[2], (int, float))]
    if not ok:
        print("(no successful sizes: no figure)")
        return
    xs = [r[0] for r in ok]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]                       # fit time vs m (log-log): show the slope
    ax.loglog(xs, [r[3] for r in ok], "s-", color="C3", label="LDpred3 (bivariate)")
    ax.loglog(xs, [r[2] for r in ok], "o-", color="C0", label="LDSC")
    if len(xs) >= 2:                   # linear-in-m reference line through point 1
        base = ok[0]
        ax.loglog(xs, [base[3] * (x / base[0]) for x in xs], "--", color="gray",
                  lw=1, alpha=.7, label="linear in m")
    ax.set_xlabel("m (variants)"); ax.set_ylabel("fit time (s)")
    ax.set_title("Running time"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)

    ax = axes[1]                       # peak RSS vs m
    ax.plot([x / 1000 for x in xs], [r[4] for r in ok], "o-", color="C2")
    ax.set_xlabel("m (thousands)"); ax.set_ylabel("peak RSS (GB)")
    ax.set_title("Memory"); ax.grid(alpha=.3)

    ax = axes[2]                       # recovered rg vs m (accuracy holds)
    ax.axhline(RG, ls="--", c="k", lw=1, alpha=.5, label=f"true rg={RG}")
    ax.plot([x / 1000 for x in xs], [r[6] for r in ok], "s-", color="C3",
            label="LDpred3")
    ax.plot([x / 1000 for x in xs], [r[5] for r in ok], "o-", color="C0",
            label="LDSC")
    ax.set_xlabel("m (thousands)"); ax.set_ylabel("estimated rg")
    ax.set_title("Accuracy"); ax.grid(alpha=.3); ax.legend(fontsize=8)

    fig.suptitle("Genetic-correlation estimation: scaling with m "
                 "(realistic non-repeating LD, single core)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "rg_scaling.png"), dpi=130)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        _worker(int(sys.argv[2]))
    else:
        main()
