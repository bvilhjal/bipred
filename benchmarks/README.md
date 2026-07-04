# bipred benchmarks

The **bivariate-LDpred** benchmarks, split out of the `ldpred3` repository. They
exercise the genetic-correlation / polygenic-overlap functionality now provided
by the [`bipred`](../) package — the joint fit (`ldpred3_auto_bivariate`,
`ldpred3_auto_bivariate_blocks`, `BivariateResult`) and cross-trait LDSC
(`ldsc_rg`) — while still importing the *univariate* pieces they build on
(LD scores, `ldpred3_auto_infer` / `ldpred3_by_blocks`, simulation helpers) from
`ldpred3`. Both `bipred` and `ldpred3` must be installed to run them.

Run single-core for stable timings:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/<script>.py
```

Most scripts simulate a **realistic non-repeating coalescent** genome, so they
need `msprime` (`pip install msprime`). Population LD is cached under
`benchmarks/.rg_cache/`. `bivariate_demo.py` instead reads a cached
`ld_library.npz` (100 blocks × 500×500 correlation matrices) from the working
directory rather than simulating.

## Scripts

| Script | What it measures | Needs msprime |
|--------|------------------|:---:|
| `rg_architectures.py` | Genetic correlation (bivariate LDSC vs bivariate LDpred3) vs truth across a six-point r_g grid × five architectures, on realistic non-repeating coalescent LD; timing + memory (→ `rg_architectures.{csv,png}`) | ✓ |
| `rg_polygenicity.py` | Genetic-correlation recovery vs polygenicity (p=0.1…1e-4) at larger m (denser blocks via higher mutation rate); LD simulated once, reused across p (→ `rg_polygenicity.{csv,png}`) | ✓ |
| `rg_methods.py` | rg estimators compared — cross-trait LDSC / `uni_gv` / `uni_r2` / the bivariate joint fit: accuracy (symmetric & asymmetric power) + running time + a timing scan across m (→ `rg_methods.{csv,png}`, `rg_methods_timing.csv`) | ✓ |
| `rg_scaling.py` | Genetic-correlation estimation scaling with m (bivariate LDSC vs LDpred3): per-fit time / peak RSS / accuracy, one subprocess per size (→ `rg_scaling.{csv,png}`) | ✓ |
| `mixer_overlap.py` | MiXeR-style polygenic-overlap recovery (`res.mixer`): overlap fraction, within-shared ρ_β, the r_g decomposition and relative polygenicity across overlap / ρ_β / power sweeps (→ `mixer_overlap.{csv,png}`) | ✓ |
| `overlap_estimation.py` | Bivariate rg sensitivity to sample overlap and how to set `cross_corr` (→ `overlap_estimation.csv`) | ✓ |
| `sample_overlap.py` | Validates the sample-overlap corrections (free LDSC intercept, bivariate `cross_corr`) on the realistic non-repeating rg LD; also per-fit timing | ✓ |
| `bivariate_demo.py` | Bivariate prediction gain for a weak trait across two-trait architectures (needs `ld_library.npz` in the cwd) | — |
| `rg_env_overlap.py` | Genetic rg recovered under **environmental** correlation on shared samples (real individual-level genotypes) (→ `rg_env_overlap.csv`) | ✓ |

`rg_env_overlap.py` reuses the univariate `infer_vs_ldsc_sbayes.py` benchmark
(which stays a univariate benchmark and imports only from `ldpred3`) for its
real-genotype coalescent genome; a copy is included here so `rg_env_overlap.py`
can `import infer_vs_ldsc_sbayes` at runtime.
