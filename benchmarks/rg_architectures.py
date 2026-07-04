"""Genetic-correlation (rg) estimation across architectures — realistic LD.

For a sweep of true genetic correlations across several architectures, two
correlated traits are simulated (shared causal variants with bivariate-normal
effects of correlation rg), a GWAS is produced from the population LD, and rg is
estimated by **bivariate LDSC** (`ldsc_rg`) and **bivariate LDpred3**
(`ldpred3_auto_bivariate_blocks`) against the truth. Both fit an LD estimated from
a finite **reference panel** (the dominant real-world error).

The LD is **realistic and non-repeating**: each of the ``NB`` blocks is its own
coalescent-with-recombination segment (`simulate_genotypes_by_mutation_rate`), so
no block is a copy of another — unlike a small tiled LD library. The population
block correlations are built once and cached on disk; the reference panel is a
fresh finite resample each replicate.

Extensive by default: five architectures × six rg × ten reps. Parallelise by
passing a comma-separated ``ARCH`` subset and an ``OUT`` basename to separate
workers. Needs ``msprime``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/rg_architectures.py
"""
import os
import sys
import csv
import time
import resource

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldpred3 import ld_scores                                            # noqa: E402
from bipred import ldsc_rg, ldpred3_auto_bivariate_blocks                # noqa: E402
from ldpred3.simulate import simulate_genotypes_by_mutation_rate          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".rg_cache")
NB = int(os.environ.get("NB", "25"))  # unique coalescent blocks (env-configurable)
K = int(os.environ.get("K", "200"))   # SNPs per block; m = NB * K
M = NB * K
N_POP = 10000                         # sample for the "population" block LD
NREF = 4000                           # reference-panel size (finite, noisy)
N1, N2 = 50000, 20000                 # per-trait GWAS sizes
H2 = 0.5
SEG_LEN = float(os.environ.get("SEG_LEN", "0.6e6"))    # per-segment length
MUT_RATE = float(os.environ.get("MUT_RATE", "4e-8"))   # SNP density (raise for
#                                                        larger K per block -> larger m)
SHRINK = 0.05                         # ref-LD shrinkage toward I (sampler stability)
REPS = int(os.environ.get("REPS", "10"))
RGS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
ARCHS = {                             # name -> causal fraction (major_locus special)
    "infinitesimal": 1.0,
    "sparse":        0.01,
    "moderate":      0.05,
    "polygenic":     0.2,
    "major_locus":   0.02,
}
IDX = [np.arange(b * K, (b + 1) * K) for b in range(NB)]


def _build_segment(b):
    """One coalescent block's population correlation ``R`` + Cholesky ``C``.

    Cached on disk **per segment** (keyed by its seed, ``K``, ``N_POP``, segment
    length and mutation rate), so the expensive msprime simulation runs once per
    segment ever: a smaller ``NB`` just uses the first few cached segments and a
    larger ``NB`` reuses those and simulates only the new ones. Raise ``MUT_RATE``
    (denser segments) to afford a larger ``K`` per block -> larger ``m``.
    """
    path = os.path.join(CACHE, f"seg{b + 1}_k{K}_npop{N_POP}_seg{int(SEG_LEN)}"
                               f"_mu{MUT_RATE:g}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["R"], d["C"]
    mut = MUT_RATE
    for _ in range(4):
        G = simulate_genotypes_by_mutation_rate(N_POP, SEG_LEN, mut_rate=mut,
                                                min_maf=0.02, seed=b + 1)
        if G.shape[1] >= K:
            break
        mut *= 1.6
    if G.shape[1] < K:
        raise ValueError(f"segment {b} has only {G.shape[1]} SNPs (< K={K}); "
                         "raise MUT_RATE or SEG_LEN")
    G = G[:, :K].astype(np.float64)
    Z = (G - G.mean(0)) / G.std(0)
    R = ((Z.T @ Z) / N_POP).astype(np.float64)
    C = np.linalg.cholesky(R + 1e-4 * np.eye(K))
    tmp = path.replace(".npz", f".tmp{os.getpid()}.npz")   # savez keeps a .npz name
    np.savez(tmp, R=R, C=C)
    os.replace(tmp, path)
    return R, C


def _build_pop():
    """``NB`` unique coalescent block correlations + their Cholesky factors.

    Each block is an independent msprime segment trimmed to ``K`` SNPs, so the
    genome is realistic LD with no repeated blocks. Segments are cached
    individually (see :func:`_build_segment`), so varying ``NB`` reuses them.
    """
    os.makedirs(CACHE, exist_ok=True)
    Rs, Cs = [], []
    for b in range(NB):
        R, C = _build_segment(b)
        Rs.append(R)
        Cs.append(C)
    return Rs, Cs


POP_R, POP_C = _build_pop()


_REF_CACHE = {}


def ref_panel(rep):
    """A finite reference panel for one replicate: per-block noisy LD + LD scores
    (latent-Gaussian resample of the population LD, shrunk toward I).

    Memoised by ``rep``: the panel depends only on the replicate (seed
    ``90000 + rep``), not on the architecture or rg, so the same panels are reused
    across every cell of the sweep instead of being rebuilt once per cell — 10
    builds for the default 5×6×10 grid rather than 300. Consumers treat the LD as
    read-only, so sharing one copy is safe."""
    cached = _REF_CACHE.get(rep)
    if cached is not None:
        return cached
    rng = np.random.default_rng(90000 + rep)
    ref = []
    for b in range(NB):
        Z = rng.standard_normal((NREF, K)) @ POP_C[b].T
        Z = (Z - Z.mean(0)) / Z.std(0)
        Rr = (1 - SHRINK) * ((Z.T @ Z) / NREF) + SHRINK * np.eye(K)
        ref.append((Rr.astype(np.float32), IDX[b]))
    _REF_CACHE[rep] = (ref, ld_scores(ref, n_ref=NREF))
    return _REF_CACHE[rep]


def gv(a, b):
    return sum(a[ix] @ (POP_R[i] @ b[ix]) for i, ix in enumerate(IDX))


def sim_effects(arch, rg, rng, p=None):
    """Two traits: shared causal set, bivariate-normal effects of correlation rg,
    each scaled to h²=``H2`` under the population LD.

    ``p`` overrides the architecture's causal fraction (for a polygenicity sweep);
    ``None`` uses ``ARCHS[arch]``."""
    L = np.linalg.cholesky([[1.0, rg], [rg, 1.0]])
    b1 = np.zeros(M)
    b2 = np.zeros(M)
    if p is None:
        p = ARCHS[arch]
    if arch == "infinitesimal":
        c = np.ones(M, bool)
    else:
        c = rng.random(M) < p
    if not c.any():
        c[rng.integers(M)] = True
    raw = L @ rng.standard_normal((2, int(c.sum())))
    b1[c] = raw[0]
    b2[c] = raw[1]
    if arch == "major_locus":                     # a few large shared effects
        maj = rng.choice(M, 3, replace=False)
        rawm = (L @ rng.standard_normal((2, 3))) * 4.0
        b1[maj] = rawm[0]
        b2[maj] = rawm[1]
    b1 *= np.sqrt(H2 / gv(b1, b1))
    b2 *= np.sqrt(H2 / gv(b2, b2))
    return b1, b2


def sumstats_pair(b1, b2, n1, n2, rng, rho_e=0.0):
    """Both traits' marginal GWAS from the population LD. ``rho_e`` correlates the
    two traits' sampling noise (the effect of overlapping GWAS samples); the main
    sweep here uses ``rho_e=0`` (disjoint samples), while ``sample_overlap.py`` /
    ``overlap_estimation.py`` reuse this helper with ``rho_e>0``."""
    bh1 = np.empty(M)
    bh2 = np.empty(M)
    for i, ix in enumerate(IDX):
        u1 = rng.standard_normal(K)
        u2 = (rho_e * u1 + np.sqrt(1 - rho_e ** 2) * rng.standard_normal(K)
              if rho_e else rng.standard_normal(K))
        bh1[ix] = POP_R[i] @ b1[ix] + (POP_C[i] @ u1) / np.sqrt(n1)
        bh2[ix] = POP_R[i] @ b2[ix] + (POP_C[i] @ u2) / np.sqrt(n2)
    return bh1, bh2


def run_cell(arch, rg, base_seed):
    ld, bp, t_ld, t_bp = [], [], [], []
    for rep in range(REPS):
        ref, ell = ref_panel(rep)
        rng = np.random.default_rng(base_seed + rep)
        b1, b2 = sim_effects(arch, rg, rng)
        bh1, bh2 = sumstats_pair(b1, b2, N1, N2, rng)
        t0 = time.perf_counter()
        ld.append(ldsc_rg(bh1, bh2, ell, N1, N2, n_blocks=NB).rg)
        t_ld.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        bp.append(ldpred3_auto_bivariate_blocks(ref, bh1, bh2, N1, N2, burn_in=150,
                                                num_iter=180, seed=rep).rg)
        t_bp.append(time.perf_counter() - t0)
    return (np.array(ld, float), np.array(bp, float),
            float(np.mean(t_ld)), float(np.mean(t_bp)))


def _warmup():
    """Trigger the bivariate sampler's Numba compilation once so the reported
    per-fit times are steady-state, not first-call JIT overhead."""
    rng = np.random.default_rng(0)
    ref, _ = ref_panel(0)
    b1, b2 = sim_effects("polygenic", 0.5, rng)
    bh1, bh2 = sumstats_pair(b1, b2, N1, N2, rng)
    ldpred3_auto_bivariate_blocks(ref, bh1, bh2, N1, N2, burn_in=5, num_iter=5)


def _write_csv(csv_path, rows):
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _summarise_cell(arch, rg, ld, bp, t_ld, t_bp):
    """One CSV row for a cell. LDSC's rg = rho_g / sqrt(h2_1 h2_2) diverges when a
    marginal h2 estimate is ~0 (its documented failure mode); those reps are
    counted as failures and only the in-range ones are summarised."""
    ok = np.isfinite(ld) & (np.abs(ld) <= 1.5)
    ldv = ld[ok]
    return {"arch": arch, "rg_true": rg,
            "ldsc_rg": round(float(np.mean(ldv)), 4) if ldv.size else "",
            "ldsc_sd": round(float(np.std(ldv)), 4) if ldv.size else "",
            "ldsc_fail": int((~ok).sum()),
            "ldpred3_rg": round(float(np.mean(bp)), 4),
            "ldpred3_sd": round(float(np.std(bp)), 4),
            "ldsc_t": round(t_ld, 4), "ldpred3_t": round(t_bp, 3)}


def main():
    only_arch = os.environ.get("ARCH")
    archs = only_arch.split(",") if only_arch else list(ARCHS)
    all_archs = list(ARCHS)
    base = os.environ.get("OUT", "rg_architectures")
    csv_path = os.path.join(HERE, base + ".csv")
    rows = []
    print(f"Genetic correlation across architectures — realistic non-repeating LD "
          f"(m={M}, {NB} unique coalescent blocks, Nref={NREF}, N1={N1}/N2={N2}, "
          f"{REPS} reps)\n", flush=True)
    _warmup()
    t0 = time.time()
    for arch in archs:
        ai = all_archs.index(arch)
        for ri, rg in enumerate(RGS):
            ld, bp, t_ld, t_bp = run_cell(arch, rg, base_seed=700 + 100 * ai + 10 * ri)
            r = _summarise_cell(arch, rg, ld, bp, t_ld, t_bp)
            rows.append(r)
            fail = f" [{r['ldsc_fail']} fail]" if r["ldsc_fail"] else ""
            print(f"  {arch:>13} rg={rg:>4} | LDSC {r['ldsc_rg']:>6}±{r['ldsc_sd']:<5}{fail}"
                  f" | LDpred3 {r['ldpred3_rg']:>6}±{r['ldpred3_sd']:<5}"
                  f" | t: LDSC {r['ldsc_t']:.3f}s / LDpred3 {r['ldpred3_t']:.2f}s",
                  flush=True)
            _write_csv(csv_path, rows)          # checkpoint after every cell
    if not os.environ.get("OUT"):
        make_figure(rows)
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_gb = mem / 1e9 if sys.platform == "darwin" else mem / 1e6
    t_ld = np.mean([r["ldsc_t"] for r in rows])
    t_bp = np.mean([r["ldpred3_t"] for r in rows])
    print(f"\nmean time/fit: LDSC {t_ld*1000:.1f} ms, bivariate LDpred3 {t_bp:.2f} s "
          f"| peak RSS {mem_gb:.2f} GB", flush=True)
    print(f"wrote {csv_path}  ({time.time() - t0:.0f}s)")


def make_figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    archs = sorted({r["arch"] for r in rows})
    fig, axes = plt.subplots(1, len(archs), figsize=(3.3 * len(archs), 3.6),
                             sharex=True, sharey=True)
    if len(archs) == 1:
        axes = [axes]
    def num(v):
        return float(v) if v not in ("", None) else np.nan

    for ax, a in zip(axes, archs):
        rr = sorted([r for r in rows if r["arch"] == a], key=lambda r: float(r["rg_true"]))
        x = [float(r["rg_true"]) for r in rr]
        ax.plot([0, 1], [0, 1], ls="--", c="k", lw=1, alpha=.5)
        ax.errorbar(x, [num(r["ldsc_rg"]) for r in rr], [num(r["ldsc_sd"]) for r in rr],
                    fmt="o-", ms=4, capsize=2, label="LDSC", color="C0")
        ax.errorbar(x, [num(r["ldpred3_rg"]) for r in rr], [num(r["ldpred3_sd"]) for r in rr],
                    fmt="s-", ms=4, capsize=2, label="LDpred3", color="C3")
        ax.set_title(a, fontsize=9)
        ax.set_xlabel("true r_g")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("estimated r_g")
    axes[0].legend(fontsize=8)
    fig.suptitle("Genetic correlation — bivariate LDSC vs LDpred3 (realistic LD)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "rg_architectures.png"), dpi=130)


if __name__ == "__main__":
    main()
