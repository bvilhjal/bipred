# bipred benchmarks

The **bivariate-LDpred** benchmarks, split out of the `ldpred3` repository. They
exercise the genetic-correlation / polygenic-overlap functionality now provided
by the [`bipred`](../) package — the joint fit (`ldpred3_auto_bivariate`,
`ldpred3_auto_bivariate_blocks`, `BivariateResult`) and cross-trait LDSC
(`ldsc_rg`) — while still importing the *univariate* LD scores and
`ldpred3_auto_infer` / `ldpred3_by_blocks` from `ldpred3`. Coalescent scripts
simulate with msprime where installed (the `[sim]` extra); without it they
fall back to a bundled Numba coalescent vendored from ldpred3
(`benchmarks/_coalescent.py`), which is dependency-free and measured about
2x msprime's per-segment time at the benchmark shape. Both packages must be
installed to run the benchmarks.

[`RESULTS.md`](RESULTS.md) is the bipred 0.2.0 snapshot generated on 2026-08-03
from benchmark source revision `17d6ae2`. The CSV files are the authoritative
numeric record. Accuracy is paired against each finite effect draw's realized
population-LD genetic correlation; `rg_target` records the generating
effect-correlation parameter separately.

From a checkout, `[bench]` installs the dependencies for the self-contained
simulation and plotting scripts:

```bash
python -m pip install -e ".[fast,bench]"
```

`[sim]` is deliberately narrower: it installs `msprime` only. HAPNEST and
`bivariate_demo.py` still require the external inputs described below.

Run single-core for stable timings on POSIX shells:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/bipred-mpl \
  python benchmarks/<script>.py
```

The PowerShell equivalent on Windows is:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMBA_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
python benchmarks/<script>.py
```

Run timing scripts sequentially. Concurrent benchmark processes compete for
memory bandwidth and turn timing columns into decorative fiction.

Most scripts simulate a **realistic non-repeating coalescent** genome, so they
need `msprime` (`pip install msprime`). Population LD is cached under
`benchmarks/.rg_cache/`. `bivariate_demo.py` instead reads a cached
`ld_library.npz` (12 blocks × 500×500 correlation matrices) from the working
directory rather than simulating.

Peak RSS is measured with `resource.getrusage` on POSIX and the Windows process
API on Windows. Wall time and process-start overhead are platform-dependent;
model validation, interval semantics, and sampler allocation behavior are not
Windows-specific package defects. Windows is an operating system, not an
alibi.

## Scripts

**Table 1. Benchmark scripts and their external simulation requirements.**

| Script | What it measures | msprime |
|--------|------------------|:---:|
| `rg_architectures.py` | LDSC and joint-fit genetic correlation across six targets × five architectures; paired realized-truth error, failures, timing, and memory (→ `rg_architectures.{csv,png}`) | opt |
| `rg_polygenicity.py` | Realized-truth recovery as the causal fraction falls from 0.1 to 1e-4; expected and observed causal counts (→ `rg_polygenicity.{csv,png}`) | opt |
| `rg_methods.py` | LDSC, `uni_gv`, `uni_r2`, the default joint fit, and `rg_decorrelated=True` under symmetric and asymmetric power, plus timing versus m (→ `rg_methods.{csv,png}`, `rg_methods_timing.csv`) | opt |
| `rg_scaling.py` | Per-fit time, peak RSS, and single-draw recovery versus m, one subprocess per size (→ `rg_scaling.{csv,png}`) | opt |
| `mixer_overlap.py` | MiXeR-style overlap, effect correlation, joint and overlap-derived r_g, LD matching, noise inflation, and univariate count anchoring (→ `mixer_overlap.{csv,png}`) | opt |
| `overlap_estimation.py` | Paired effect of known sample-overlap `cross_corr`; it does not validate LDSC-intercept inversion (→ `overlap_estimation.csv`) | opt |
| `sample_overlap.py` | Lower-power comparison of free/constrained LDSC and unset/set bivariate overlap corrections (→ `sample_overlap.csv`) | opt |
| `bivariate_demo.py` | Bivariate prediction gain for a weak trait across two-trait architectures (needs `ld_library.npz` in the cwd) | — |
| `rg_env_overlap.py` | Individual-genotype stress test under **environmental** correlation on shared samples; records paired MAE and failures after the `|r_g| <= 1.5` diagnostic window (→ `rg_env_overlap.csv`) | opt |
| `hapnest/run_bivariate.py` | rg / h² / MiXeR-overlap recovery **and** out-of-sample PRS gain (bivariate vs univariate) on **HAPNEST** genotypes+phenotypes — synthetic genomes resampled from a real 1000G+HGDP reference, so real LD/MAF/structure with known truth (→ `hapnest/run_bivariate.csv`). See [`hapnest/README.md`](hapnest/README.md). | — (needs HAPNEST) |

`rg_env_overlap.py` reuses the univariate `infer_vs_ldsc_sbayes.py` benchmark
(which stays a univariate benchmark and imports only from `ldpred3`) for its
real-genotype coalescent genome; a copy is included here so `rg_env_overlap.py`
can `import infer_vs_ldsc_sbayes` at runtime.

## Committed artifact contract

**Table 2. Generated artifacts and data-row counts.**

| Stem | CSV rows | PNG |
|---|---:|:---:|
| `rg_architectures` | 30 | ✓ |
| `rg_polygenicity` | 4 | ✓ |
| `rg_methods` | 10 | ✓ |
| `rg_methods_timing` | 3 | — |
| `rg_scaling` | 5 | ✓ |
| `mixer_overlap` | 29 | ✓ |
| `overlap_estimation` | 6 | — |
| `sample_overlap` | 3 | — |
| `rg_env_overlap` | 5 | — |

That is nine CSVs and five PNGs. `bivariate_demo.py`, HAPNEST, and standalone
`infer_vs_ldsc_sbayes.py` require the external inputs named in
[`RESULTS.md`](RESULTS.md).
