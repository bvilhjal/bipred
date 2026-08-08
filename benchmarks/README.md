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

[`RESULTS.md`](RESULTS.md) is the narrative record; CSV files are the
authoritative numeric record. A release-quality regeneration consists of the
CSV/PNG artifacts, the `run_all.log` that names their clean source revision and
environment, and matching prose tables. Timing and memory columns must come
from that same sweep rather than from snapshots spliced together. Tests in
`tests/test_benchmark_simulate.py` check the cheap artifact and harness
contracts and re-derive important prose tables from their CSVs.

`run_all.sh` drives the regeneration in order, single-core:

```bash
BIPRED_PYTHON=/path/to/python bash benchmarks/run_all.sh
```

It refuses a dirty source tree: a commit hash does not identify staged,
unstaged, or untracked code. After that preflight it records the full source
revision, interpreter, package versions and resolved simulator backend at the
top of its log. A run whose `import msprime` fails falls back to the bundled
coalescent, which shares no cached segments with an msprime record and is not
directly comparable to it. Accuracy is paired against each finite effect draw's
realized population-LD genetic correlation; `rg_target` records the generating
effect-correlation parameter separately.

From a checkout, `[bench]` installs the dependencies for the self-contained
simulation and plotting scripts:

```bash
python -m pip install -e ".[fast,bench]"
```

`[sim]` is deliberately narrower: it installs `msprime` only. HAPNEST still
requires the external inputs described below; `bivariate_demo.py` needs only
`make_ld_library.py` run first and writes full-precision per-replicate results
to `benchmarks/bivariate_demo.csv`.

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
| `mixer_overlap.py` | MiXeR-style overlap, effect correlation, shared-fraction bias versus per-trait polygenicity, LD matching, noise inflation, and univariate count anchoring (→ `mixer_overlap.{csv,png}`) | opt |
| `overlap_estimation.py` | Paired effect of known sample-overlap `cross_corr`; it does not validate LDSC-intercept inversion (→ `overlap_estimation.csv`) | opt |
| `sample_overlap.py` | Lower-power comparison of free/constrained LDSC and unset/set bivariate overlap corrections (→ `sample_overlap.csv`) | opt |
| `sweep_cost.py` | Per-sweep median ± MAD by LD representation and core count, with one distinct payload per synthetic block (→ `sweep_cost.csv`) | — |
| `fit_memory.py` | Caller LD payload and Python allocation added by a fit, again with distinct block payloads (→ `fit_memory.csv`) | — |
| `bivariate_demo.py` | Bivariate prediction gain for a weak trait across two-trait architectures, pairing true zero- and 5%-shrinkage references and a disjoint-causal control (needs `ld_library.npz` from `make_ld_library.py`; → `bivariate_demo.csv`) | — |
| `qc_factorial.py` | **QC sensitivity**: 2x2x2 over strict thresholds, long-range LD exclusion and the LD-consistency screen, on three real trait pairs; records bipred and LDSC on identical variants and uses each arm's raw in-range LDSC intercept as a `cross_corr` sensitivity value (→ `qc_factorial.csv`). This assumes the whole intercept is correlated sampling noise; confounding may also contribute. ~75 min, same inputs as `real_ldl_cad.py` plus GLGC HDL/TG and GIANT height | — |
| `real_ldl_cad.py` | **Real GWAS**: LDL x CAD on a UK Biobank HapMap3 LD reference, fitted at three cleaning stages; checks numerical diagnostics and the divergence warning, with external estimates used only as rough context (→ `real_ldl_cad.csv`). Needs ~9 GB of downloads, see its docstring | — |
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
| `mixer_overlap` | 41 | ✓ |
| `overlap_estimation` | 6 | — |
| `sample_overlap` | 3 | — |
| `rg_env_overlap` | 5 | — |
| `sweep_cost` | 10 | — |
| `fit_memory` | 8 | — |

That is eleven CSVs and five PNGs generated by `run_all.sh`. The following
artifacts have separate input or runtime contracts and are regenerated
manually:

**Table 3. Manually regenerated artifacts.**

| Stem | CSV rows | Requirement |
|---|---:|---|
| `bivariate_demo` | 60 | locally generated `ld_library.npz` |
| `qc_factorial` | 24 | public real-GWAS inputs; about 75 minutes |
| `real_ldl_cad` | 3 | public real-GWAS inputs; about 25 minutes |

**Table 4. Acquisition record for the six checksummed real-data inputs.**

| Manifest entry | Canonical source | Release, citation, and terms |
|---|---|---|
| `ldref-hm3/ldpred3_ldref_hm3.npz` | [European LD reference with blocks, Figshare article 19213299](https://figshare.com/articles/dataset/European_LD_reference_with_blocks_/19213299) | Download `map.rds` and `ldref_with_blocks.zip`; CC BY 4.0. Convert them with `ldpred3/benchmarks/convert_bigsnpr_ldref.py` at ldpred3 revision `5d86ac9d97e42c57fa31d84ff093d3bf637dc0e6`. |
| `sumstats/jointGwasMc_LDL.txt.gz` | [GLGC 2013 LDL](http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_LDL.txt.gz) | Willer et al. (2013); follow the provider's data-use terms. |
| `sumstats/jointGwasMc_HDL.txt.gz` | [GLGC 2013 HDL](http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_HDL.txt.gz) | Willer et al. (2013); follow the provider's data-use terms. |
| `sumstats/jointGwasMc_TG.txt.gz` | [GLGC 2013 triglycerides](http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_TG.txt.gz) | Willer et al. (2013); follow the provider's data-use terms. |
| `sumstats/cad.add.160614.website.txt` | [CARDIoGRAMplusC4D 2015, GCST003116](https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST003001-GCST004000/GCST003116/cad.add.160614.website.txt) | Nikpay et al. (2015), GWAS Catalog study GCST003116; follow the archive's terms. |
| `sumstats/GIANT_HEIGHT_2014.txt.gz` | [GIANT height 2014 public release](https://giant-consortium.web.broadinstitute.org/images/0/01/GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz) | Wood et al. (2014); save under the manifest filename and follow the provider's terms. |

For example, the five summary-statistic files can be staged without changing
their compressed bytes:

```bash
BIPRED_BENCH_WORK=/absolute/path/to/bipred-benchmark-work
mkdir -p "$BIPRED_BENCH_WORK/sumstats"
curl -L http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_LDL.txt.gz -o "$BIPRED_BENCH_WORK/sumstats/jointGwasMc_LDL.txt.gz"
curl -L http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_HDL.txt.gz -o "$BIPRED_BENCH_WORK/sumstats/jointGwasMc_HDL.txt.gz"
curl -L http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_TG.txt.gz -o "$BIPRED_BENCH_WORK/sumstats/jointGwasMc_TG.txt.gz"
curl -L https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST003001-GCST004000/GCST003116/cad.add.160614.website.txt -o "$BIPRED_BENCH_WORK/sumstats/cad.add.160614.website.txt"
curl -L https://giant-consortium.web.broadinstitute.org/images/0/01/GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz -o "$BIPRED_BENCH_WORK/sumstats/GIANT_HEIGHT_2014.txt.gz"
export BIPRED_WORK="$BIPRED_BENCH_WORK"
```

For the LD artifact, place Figshare's `map.rds` and extracted
`ldref_with_blocks.zip` contents under `$BIPRED_BENCH_WORK/ldref-hm3`, then run
the converter from the exact ldpred3 revision in Table 4:

```bash
python /path/to/ldpred3/benchmarks/convert_bigsnpr_ldref.py \
  --work "$BIPRED_BENCH_WORK/ldref-hm3" --test
```

The converter revision fixes the transformation; the manifest hash fixes the
resulting NPZ bytes. The other providers do not publish a uniform license in
these download files, so this repository does not invent one on their behalf.

`real_ldl_cad.py` and `qc_factorial.py` are the two real-data benchmarks. They
exist because a defect shipped in 0.3.0 that no simulation drawing `beta_hat`
from the fitted model could have caught. Both are excluded from `run_all.sh`
because their inputs are large external downloads.

Both scripts validate every input acquired through Table 4 against
[`real_data_inputs.sha256`](real_data_inputs.sha256) before parsing. The LD
reference entry hashes the converted NPZ itself, not merely its download or
converter version; this pins the bytes used by the fit even if the sibling
ldpred3 converter later changes.

Future runs also refuse staged, unstaged, or untracked source changes and write
`<stem>.provenance.json` beside the CSV. That sidecar records the clean source
revision, Python/platform and package versions, observed input hashes, and the
loaded ldpred3 source identity. A Git checkout must be clean and at the tested
revision; a non-VCS installation is pinned by a complete package-tree hash and,
when available, its PEP 610 commit. The real LDL/CAD sidecar also records its
command-line screening-round count. The committed real-data CSVs predate this
contract and have no sidecars; none were invented after the fact.

The `bivariate_demo.csv` rows retain full precision, causal-set overlap and the
realized LD-aware genetic correlation, as well as the reference shrinkage
setting; [`RESULTS.md`](RESULTS.md) §8 is a rounded summary, not the primary
evidence. A disjoint causal support need not have realized genetic correlation
exactly zero because LD connects variants across the two supports. Future runs
refuse dirty source and write a provenance sidecar that also hashes the local
`ld_library.npz`. The current CSV was regenerated during the 0.3.4 working-tree
review before that preflight existed, so it is a provisional numerical artifact
without a clean-revision sidecar.

The current `sweep_cost.csv` and `fit_memory.csv` were likewise regenerated to
repair their payload and timing contracts, but not as part of the historical
`run_all.log`. They are provisional until the next clean `run_all.sh`
regeneration; the log must not be cited as their provenance.

The `qc_factorial.csv` header was mechanically renamed from `dentist` to
`ld_screen` with the public API, and its `expected` labels were recast as rough
context rather than truth ranges. In both real-data CSVs, `warned` was renamed
to `divergence_warned`: the scripts count only warnings whose text contains
`diverged`, not every warning emitted by a fit. Both CSVs remain numerical
records of the earlier screen, which could stop after the first empty random
partition; the current screen always runs every requested partition. They were not
regenerated after that semantic correction, so treat Tables 13--16 as
historical case studies until a clean rerun replaces the files. The factorial's
saved `cross_corr` is the arm-specific free LDSC intercept; all saved values are
inside `(-1, 1)`, and the current script rejects rather than clips an invalid
one. HAPNEST and standalone `infer_vs_ldsc_sbayes.py` require the external
inputs named in `RESULTS.md`.
