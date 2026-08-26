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
| `qc_factorial.py` | **QC sensitivity**: 2x2x2 over strict thresholds, long-range LD exclusion and the LD-consistency screen, on three real trait pairs; records bipred and LDSC on identical variants and uses each arm's raw in-range LDSC intercept as a `cross_corr` sensitivity value (→ `qc_factorial.csv`). This assumes the whole intercept is correlated sampling noise; confounding may also contribute. ~126 min on the benchmark's 10-core Apple M2 Pro, same inputs as `real_ldl_cad.py` plus GLGC HDL/TG and GIANT height | — |
| `real_ldl_cad.py` | **Real GWAS**: LDL x CAD on a UK Biobank HapMap3 LD reference, fitted at three cleaning stages; checks numerical diagnostics and the divergence warning, with external estimates used only as rough context (→ `real_ldl_cad.csv` and step-level `real_ldl_cad_timing.csv`). Needs ~9 GB of downloads, see its docstring | — |
| `rg_env_overlap.py` | Individual-genotype stress test under **environmental** correlation on shared samples; records paired MAE and failures after the `|r_g| <= 1.5` diagnostic window (→ `rg_env_overlap.csv`) | opt |
| `external_overlap.py` | **External-tool validation on simulated truth**: bipred vs the original MiXeR (gsa-mixer v2.2.1, source-built) vs the original LDSC (CBIIT Python-3 port) on one small coalescent panel with known causal overlap, in-sample LD (→ `external_overlap.csv` + sidecar). Needs the `.venv-ldsc` and `.mixer` environments (see *External tools* below); absent tools become NaN rows | opt |
| `external_hdl_tg.py` | **External-tool validation on real data**: GLGC 2013 HDL x TG through the original LDSC (standard `eur_w_ld_chr` weights) against bipred's `ldsc_rg` and the joint fit with and without the overlap correction (→ `external_hdl_tg.csv` + sidecar). MiXeR cells need a user-supplied 1000G.EUR.QC bundle via `MIXER_REF` and skip otherwise | — |
| `hapnest/run_bivariate.py` | rg / h² / MiXeR-overlap recovery **and** out-of-sample PRS gain (bivariate vs univariate) on **HAPNEST** genotypes+phenotypes — synthetic genomes resampled from a real 1000G+HGDP reference, so real LD/MAF/structure with known truth (→ `hapnest/run_bivariate.csv`). See [`hapnest/README.md`](hapnest/README.md). | — (needs HAPNEST) |

## External tools

`external_overlap.py` and `external_hdl_tg.py` compare against the *original*
MiXeR and LDSC implementations rather than bipred's reimplementations. Both
tools live in isolated, gitignored environments inside this directory:

- **LDSC** — `bash benchmarks/external_setup.sh ldsc` creates
  `.venv-ldsc/`, installs the CBIIT/ldsc PyPI port (`ldsc` 2.0.1), and applies
  two one-line compatibility patches for modern bitarray/numpy (documented in
  the script; unfixed upstream as of 2026-08). Override discovery with
  `LDSC_BIN=/path/to/bin`.
- **MiXeR** — gsa-mixer v2.2.1 is built from source into `.mixer/` (macOS
  arm64 is not a supported platform; the exact working recipe and its
  hello-world validation output are in `.mixer/BUILD_LOG.txt` on machines that
  ran it). Discovery env vars: `MIXER_PY`, `MIXER_SRC`, `MIXER_LIB`.
- The real-arm MiXeR cells additionally need `MIXER_REF` pointing at a
  1000G.EUR.QC reference bundle (GB-scale; never downloaded by the scripts).

The LDSC 1000G EUR weights come from the Zenodo mirror named in Table 4 (the
Broad host is requester-pays since 2026); `w_hm3.snplist` ships inside the same
tarball. Both files are pinned in
[`real_data_inputs.sha256`](real_data_inputs.sha256).

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

| Stem | CSV rows | Provenance sidecar | Requirement |
|---|---:|---|---|
| `bivariate_demo` | 60 | `bivariate_demo.provenance.json` | locally generated `ld_library.npz` |
| `qc_factorial` | 24 | `qc_factorial.provenance.json` | public real-GWAS inputs; ~126 min on a 10-core Apple M2 Pro |
| `real_ldl_cad` | 3 + 34 timing rows | `real_ldl_cad.provenance.json` | public real-GWAS inputs; 1,365.693 s on the same machine |
| `external_overlap` | 40 (2 cells x 5 reps x 4 methods) | `external_overlap.provenance.json` | `.venv-ldsc` / `.mixer` tool environments (see *External tools*) |
| `external_overlap_ldsc200k` | 24 (2 cells x 3 reps x 4 methods) | `external_overlap_ldsc200k.provenance.json` | same; the LDSC-scale panel (m = 200k) variant of the above |
| `external_hdl_tg` | 5 | `external_hdl_tg.provenance.json` | GLGC HDL/TG inputs + LDSC weights from Table 4 |

**The artifact record is currently split across two ldpred3 versions.** The
ten `run_all.sh` scripts were regenerated from clean revision `bf5236a`
(bipred 0.3.10.dev0) against the current pin, ldpred3 0.6.1 at
`af5d92c7aab6a5b67d15c94ebe28b89e33f5d69d`, with Python 3.14.6, NumPy 2.4.6 and
Numba 0.66.0; `run_all.sh` selected msprime and all ten completed. The manual
benchmarks -- `qc_factorial`, `real_ldl_cad`, `external_overlap`,
`external_overlap_ldsc200k`, `external_hdl_tg` and `bivariate_demo` -- still
carry the earlier 0.3.5 record (revision `5c06ec7`, ldpred3 0.4.5), because
they need real-data inputs and tool environments that sweep did not have. They
must be regenerated against the current pin before this counts as a
release-quality record; until then, do not read a manual artifact and a
`run_all.sh` artifact as one sweep.

For the record: moving the pin left every bipred-owned estimate unchanged.
What moved was ldpred3's univariate inference -- the `calib_*` columns of
`mixer_overlap`, which are the only ones fed by `ldpred3_auto_infer`, and the
`uni_gv`/`uni_r2` rows of `rg_methods` -- alongside timing and peak RSS
throughout. The joint-fit columns beside them are identical.

Wall times are measurements of that 10-core Apple M2 Pro under one-thread
controls, not portable expectations.

**Table 4. Acquisition record for the six checksummed real-data inputs.**

| Manifest entry | Canonical source | Release, citation, and terms |
|---|---|---|
| `ldref-hm3/ldpred3_ldref_hm3.npz` | [European LD reference with blocks, Figshare article 19213299](https://figshare.com/articles/dataset/European_LD_reference_with_blocks_/19213299) | Download `map.rds` and `ldref_with_blocks.zip`; CC BY 4.0. Convert them with `ldpred3/benchmarks/convert_bigsnpr_ldref.py` at ldpred3 revision `5d86ac9d97e42c57fa31d84ff093d3bf637dc0e6`. |
| `sumstats/jointGwasMc_LDL.txt.gz` | [GLGC 2013 LDL](http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_LDL.txt.gz) | Willer et al. (2013); follow the provider's data-use terms. |
| `sumstats/jointGwasMc_HDL.txt.gz` | [GLGC 2013 HDL](http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_HDL.txt.gz) | Willer et al. (2013); follow the provider's data-use terms. |
| `sumstats/jointGwasMc_TG.txt.gz` | [GLGC 2013 triglycerides](http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_TG.txt.gz) | Willer et al. (2013); follow the provider's data-use terms. |
| `sumstats/cad.add.160614.website.txt` | [CARDIoGRAMplusC4D 2015, GCST003116](https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST003001-GCST004000/GCST003116/cad.add.160614.website.txt) | Nikpay et al. (2015), GWAS Catalog study GCST003116; follow the archive's terms. |
| `sumstats/GIANT_HEIGHT_2014.txt.gz` | [GIANT height 2014 public release](https://giant-consortium.web.broadinstitute.org/images/0/01/GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz) | Wood et al. (2014); save under the manifest filename and follow the provider's terms. |
| `ldsc-weights/eur_w_ld_chr.tar.gz` | [Zenodo record 8182036](https://zenodo.org/records/8182036), a mirror of the Broad Institute's 1000 Genomes European LD scores (the Broad host is requester-pays since 2026) | Bulik-Sullivan et al. (2015); verify against the manifest hash after download. |
| `ldsc-weights/w_hm3.snplist` | Ships inside the `eur_w_ld_chr` tarball above | HapMap3 allele list used by `munge_sumstats.py --merge-alleles`. |

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

The manual scripts refuse staged, unstaged, or untracked source changes and
write the sidecars named in Table 3. Each records the clean source revision,
Python/platform and package versions, observed input hashes, and loaded
ldpred3 source identity. A Git checkout must be clean and at the tested
revision; a non-VCS installation is pinned by a complete package-tree hash and,
when available, its PEP 610 commit. The real LDL/CAD sidecar also records its
screening-round count, CPU count, thread environment, and
`real_ldl_cad_timing.csv`. That 34-row artifact separates reference loading,
GWAS harmonisation, preparation, both LD-consistency screens, LD scores, LDSC,
bivariate fitting, diagnostics, and output. Only its final total overlaps the
leaf rows; no machine-dependent pass/fail threshold is imposed.

The `bivariate_demo.csv` rows retain full precision, causal-set overlap and the
realized LD-aware genetic correlation, as well as the reference shrinkage
setting; [`RESULTS.md`](RESULTS.md) §8 is a rounded summary, not the primary
evidence. A disjoint causal support need not have realized genetic correlation
exactly zero because LD connects variants across the two supports. The script
refuses dirty source and writes a provenance sidecar that also hashes the local
`ld_library.npz`; the current pair comes from the clean 0.3.5 regeneration.

The current `sweep_cost.csv` and `fit_memory.csv` were produced by the same
clean `run_all.sh` regeneration recorded in `run_all.log`.

Both real-data CSVs were regenerated with the always-run partition screen.
Their `divergence_warned` field counts only warnings whose text contains
`diverged`. The factorial's saved `cross_corr` is the arm-specific free LDSC
intercept; all saved values are inside `(-1, 1)`, and the script rejects rather
than clips an invalid one. HAPNEST and standalone
`infer_vs_ldsc_sbayes.py` require the external inputs named in `RESULTS.md`.
