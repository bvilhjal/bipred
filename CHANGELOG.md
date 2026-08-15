# Changelog

User-visible changes to **bipred** are recorded here. The project is currently
`0.3.9.dev0`.

## [Unreleased]

### Changed

- The LD-consistency screen explains blocked parallelism. Requesting
  `ncores > 1` and silently getting a serial screen read as a no-op flag; the
  conservative BLAS gate is unchanged, but a blocked run now warns which
  condition failed (unpinned BLAS threads, missing `threadpoolctl`, or a
  non-reentrant BLAS) and how to enable the pool. The gate opening, and
  `ncores=1`, stay quiet.
- `ldpred3_auto_bivariate_blocks` warns once per process when the pure-Python
  no-Numba fallback is active (`warn_no_numba`, re-exported through the
  `ldpred3._numba` seam) — the fallback is numerically identical but orders of
  magnitude slower, and was previously indistinguishable from the compiled
  path at runtime.
- The development line now requires LDpred3 0.5 and delegates strict principal
  LD subsetting to its public interoperability API.
- CI installs the current sibling through the declared resolver contract; the
  now-redundant second `ldpred3-head` suite has been removed.
- The command line exposes column mappings, summary-statistic QC, allele-
  frequency concordance, LD-screen controls, sampling-error correlation and
  deterministic multi-chain inference. Target-scaled weight files are now the
  safe CLI default; HWE-derived frozen scaling requires an explicit flag.

### Fixed

- Missing or QC-dropped variants are intersected before LD-consistency
  screening and are never imputed as z=0 observations.
- Principal subsetting validates masks and indices, retains singleton blocks,
  supports set callers, reuses complete mmap blocks, and never expands a whole
  low-rank parent merely to select a small principal submatrix.
- Prepared mmap panels retain their cache owner until explicitly closed.
  A caller-owned `ldpred3.interop.PreparedLDCache` can instead be shared across
  sibling fits without rescanning the complete payload.
- Threaded multi-chain fitting retains at most one completed genome-wide result
  per worker, and single-chain finalization no longer normalizes posterior
  vectors twice.
- A cache without reference allele frequencies writes target-scaled weights
  instead of failing after the fit. HWE-derived dosage SD is labeled as an
  approximation rather than an observed fit-cohort scale.

## [0.3.8] - 2026-08-13

### Added

- `prepare_bivariate_sumstats` loads an ldpred3 LD cache and two GWAS files,
  harmonizes both to the cache allele, standardizes, and returns contiguous
  blocks plus provenance. `subset_blocks` is the public retile used after QC.
  `BivariateResult.write_weights` writes one trait as an ldpred3 weight file
  (HWE `SD_REF` from cache AF when `sd` is omitted). `python -m bipred`
  runs that path.
- `ldsc_chi2_mask` returns the reference LDSC chi-square row filter
  (`chi2 > max(0.001 N, 80)`). It is for subsetting `ldsc_rg` arguments
  only.

### Fixed

- The user guide no longer lists a chi-square cap among the per-variant
  filters for the joint-fit panel. The cap is an LDSC-row filter; applying
  the same mask to `ldpred3_auto_bivariate_blocks` is the 0.3.7 failure
  mode. `docs/rg.md` and the `qc` module intro now say the same thing.
- Table 3 documents `h2_cap` as a ceiling on implied per-trait heritability
  (`s_t ≤ h2_cap_t / n_causal,t`), matching the sampler, not as a raw slab
  variance.
- The fit warns when `beta_hat` looks like a z-score (`|beta| >= 1`) or
  like unstandardized per-allele effects, instead of silently returning a
  plausible `h2`/`rg` on the wrong scale.

## [0.3.7] - 2026-08-09

A documentation defect with real consequences: `ldsc_rg` told callers to cap
extreme chi-square before calling it, and did not say that the cap belongs to
the regression alone. Both real-data drivers read it the other way, built one
per-variant mask, and handed the same capped variant set to LD Score regression
and to the bivariate fit.

### Changed

- **`ldsc_rg` now says which variants the chi-square cap is for.** The advice
  exists because this is a one-step unfiltered LDSC whose weights come from the
  fitted means, so a variant far above the line keeps near-full leverage on the
  slope. It is about a linear regression and applies to nothing else. Applying
  it to the fit as well deletes large effects from a mixture model whose slab
  component exists to hold them.

  How much that costs depends entirely on the trait's architecture. Over the
  HapMap3 reference, `chi2 > 80` discards 0.4% of CAD's summed chi-square and
  7.6% of LDL's, but 47% of urate's, 51% of lipoprotein(a)'s and 73% of total
  bilirubin's -- from under 3,000 variants in a million. `studies/_lib/
  chi2_cap_audit.py` measures it for a registered trait.

  No estimator behaviour changed. Fits that did not apply a cap are unaffected,
  and passing the same arguments as before gives the same answer.

### Fixed

- **`benchmarks/real_ldl_cad.py` and `benchmarks/qc_factorial.py` take
  `--chi2-cap {both,regression,none}`.** `both` is the default and reproduces every
  committed number; `regression` holds high-chi-square rows out of the regression
  while the fit keeps every variant. In the factorial the distinction reaches
  further than the fit: `per_variant()` runs upstream of the LD-consistency
  screen, so under `both` the screen never evaluates the variants the cap
  removed.

  On LDL x CAD the change is immaterial, which is consistent with those two
  traits being near the bottom of the loss table: after screening, `rg` moves
  +0.2658 to +0.2589, less than the fit's own iterate spread of 0.0148 there.
  The two fit sets differ by 426 variants before the screen and by one after
  it, so the screen was already removing what the cap would have.

- **`qc_factorial.py` has a CLI.** It had none, so every knob was a module
  global a caller had to monkeypatch, and any run overwrote the committed
  `qc_factorial.csv`. It now takes `--out`, `--chi2-cap` and `--rounds`, and
  `main()` accepts a parsed namespace.

- **`qc_factorial.py` no longer raises `KeyError: 'LDL'` on a narrowed trait
  set.** Its checksum set was built from hardcoded trait names; it now derives
  from the traits in `PAIRS`.

- **`real_ldl_cad.py` honours `BIPRED_WORK`,** as `qc_factorial.py` already
  did. Pointing it at a checkout outside `~/REPOS` previously meant setting
  `BIPRED_LDREF`, `BIPRED_LDL` and `BIPRED_CAD` individually.

- **Two tests failed for environment reasons, one of them by segfault.**
  `test_pooled_screen_matches_the_serial_one` forces the block pool on to check
  that `ncores` cannot change the mask, but forcing it skipped the gate's other
  precondition: the pool is only ever entered with BLAS pinned to one thread.
  Against numpy's bundled OpenBLAS at its default thread count, nesting over it
  segfaulted rather than failing, so the process died mid-suite and the tests
  ordered after it never ran. It now pins BLAS for the duration.
  `test_run_all_refuses_an_untracked_source_file` derived Git Bash from
  `git.exe` at a fixed depth that only holds when `git` resolves to
  `<Git>/cmd/git.exe`.

### Added

- **`real_ldl_cad.py` records `rg_iterate_sd`, `h2_ldl_iterate_sd`,
  `h2_cad_iterate_sd` and `retained_iterations`,** matching what
  `qc_factorial.py` already wrote and carrying the same caveat: this is
  approximate MCMC, so these are the spread of autocorrelated iterates from one
  chain rather than posterior standard deviations. `ldpred3_auto_bivariate_
  chains` and its split-Rhat remain the way to get a defensible interval.

## [0.3.6] - 2026-08-08

Ports what carries over from ldpred3 0.4.6's DENTIST work. Only one of its
three changes applies here, because the two screens are different statistics:
ldpred3 inverts a whole block and drops the single worst variant per pass,
while this one predicts random half-windows from each other. Its SPD-Cholesky
inverse route has no counterpart in a truncated `eigh` pseudo-inverse, and its
fused Numba Schur kernel downdates a precision matrix this screen never forms.
Its factored low-rank route is not wanted here either: `_window_ld` already
reads the exact windowed submatrix of `U U' + diag(d)` at window scale, where
ldpred3 needs an approximation whose own benchmark shows it discarding up to
seven times more clean variants.

### Added

- **`ld_consistency_screen(..., ncores=N)` settles blocks concurrently.**
  Blocks tile disjoint variant ranges and each round re-reads only its own
  survivors, so a block's entire schedule depends on no other block. Measured
  at 16 x k=2,000 over 3 rounds with BLAS pinned, on four cores: 1.69x at
  `ncores=2`, 2.17x at 3, 2.49x at 4, with the keep mask identical at every
  core count. Runtime is monotone in `ncores`.

  The pool nests over BLAS, so it is taken only when BLAS is pinned to one
  thread *and* `threadpoolctl` confirms the loaded library is reentrant. This
  screen's concurrent call is `np.linalg.eigh` -- the routine ldpred3 measured
  returning silently wrong answers under an OpenMP-layer OpenBLAS -- so it
  takes the conservative branch of ldpred3's gate and never nests on the
  environment-variable hint alone. Without `threadpoolctl` the screen stays
  serial whatever `ncores` says. Peak memory rises by one dense window per
  worker, not one dense block.

### Changed

- **A given `seed` no longer reproduces the masks of 0.3.5 and earlier.**
  Making the pool possible required this: the old single generator was
  consumed in block order, so which draws a block saw depended on how many the
  blocks before it had made, and the splits would have followed whatever order
  the workers finished in. Each block now derives its own stream per round
  from a spawned `SeedSequence`, keyed to its position in `blocks`. The mask
  is a function of the seed alone and identical at every `ncores`. A shorter
  run remains a prefix of a longer one, so comparing `rounds=3` against
  `rounds=4` still isolates the fourth split.
- Under `verbose`, per-round drop counts are printed once the screen finishes
  rather than as each round completes, since a block now runs all of its
  rounds together. The counts themselves are unchanged.
- The split-half statistic forms one `t x p x r` product instead of two.
  `retained / values` scales columns, so `across @ (retained / values)` is
  `(across @ retained) / values`: one product serves both the prediction and
  the leverage, and the prediction reads off the smaller `t x r` result rather
  than `across`. Measured 1.10x on the screen end-to-end, serial, with the
  mask unchanged; `eigh` dominates what is left. The reassociation moves the
  statistic by float64 rounding against a threshold of 29.72, and a test pins
  it to the two-product form.

### Benchmarks

- Nothing was regenerated for this release, and the committed record stays a
  0.3.5 one. The re-keying above changes which variants a given seed drops, so
  every screen-dependent figure in `benchmarks/RESULTS.md` — the retained
  counts and `rg` of Sections 9 and 10, and Tables 13--17 — would move on a
  re-run. Those sections report qualitative separations (12/12 unscreened arms
  warned against 0/12 screened) that are not expected to turn on the exact
  mask, but the numbers are 0.3.5 measurements and are now labelled as such.

## [0.3.5] - 2026-08-08

### Added

- The LDL-CAD real-data benchmark now persists six-decimal wall times for
  source and input checks, reference loading, harmonisation, preparation,
  LD-consistency screening, LD scores, LDSC, each bivariate fit, diagnostics,
  output, and the inclusive end-to-end total. The timing artifact records the
  same clean source, inputs, thread controls, and environment as the result.

### Benchmarks

- The complete suite was regenerated from clean revision `5c06ec7` with
  bipred 0.3.5: all ten `run_all.sh` scripts and the three manual benchmarks.
  The manual outputs now include clean-source provenance sidecars; the LDL-CAD
  run also includes its 34-row timing artifact.
- With numerical thread counts fixed at one on the 10-core Apple M2 Pro used
  for this refresh, LDL-CAD took 1,365.693 seconds (22.8 minutes) and the
  24-arm QC factorial about 126 minutes. These are hardware-specific
  measurements, not performance thresholds.

## [0.3.4] - 2026-08-08

This release tightens the contracts exposed by the 0.3.3 real-data review and
corrects claims that the evidence did not support.

### Fixed

- The LD-consistency screen now dequantizes dense D8 blocks, slices a window
  before widening it, runs every requested random partition, and validates
  finite inputs and controls. Dense floats and floating low-rank factors are
  normalised to the fitter's D32 values. The old dense path could both
  misclassify D8 LD and allocate a float64 copy of an entire block per window.
- `implied_sample_size` no longer reports an absolute effective sample size for
  a quantitative trait. Its phenotype scale is unidentified from `beta`, `se`,
  and allele frequency alone; calibrating that scale from the reported N made
  the earlier agreement circular. The binary-trait inversion remains available.
- `estimate_sample_overlap` now distinguishes an intercept inside the numeric
  domain of `cross_corr` from a physically possible shared-person inversion.
  Domain validity does not prove that the intercept is sampling noise. Counts
  with the wrong sign or larger than the smaller cohort are reported as `nan`,
  while raw diagnostic quantities remain visible.
- Dense regional quadratics avoid quadratic temporaries. Contiguous regions use
  a view; interleaved regions are multiplied in bounded row slabs. Floating
  low-rank factors are evaluated after the same D32 normalisation as the fit.

### Changed

- `ld_consistency_screen` is the primary name for the lightweight,
  DENTIST-inspired diagnostic. `dentist` remains a compatibility alias; neither
  name claims reproduction or calibration of the full published procedure.
- Benchmark controls now distinguish exact zero shrinkage and genuinely
  disjoint causal supports, preserve small CSV effects, avoid repeated-block and
  hot-cache timing artifacts. Core-count speedups use the actual one-core arm;
  peak RSS is reported only at process scope. The self-contained runner records
  clean-source provenance, including the exact ldpred3 source identity, and
  real-data scripts checksum externally sourced inputs whose acquisition record
  is explicit; historical artifacts retain their own snapshot identity.
- Real-data conclusions are scoped to three related pairs that all contain a
  GLGC file. Long-range-LD exclusion is an estimator-specific sensitivity, and
  external-value agreement is not a convergence diagnostic. Their saved rows
  predate the always-run partition correction and were not relabelled as a
  current-screen validation.

## [0.3.3] - 2026-08-08

A 24-arm factorial over three related real-data pairs tested three
summary-statistic QC choices. The result motivated a narrower diagnostic; it did
not settle a universal QC procedure.

### Added

- **`bipred.qc.implied_sample_size`** introduced an inversion of the SD
  relation. Its binary-trait use identified a 0.570 ratio for the
  CARDIoGRAMplusC4D file (162,973 reported against 92,966 implied). The four
  quantitative-trait matches were not independent checks: their unknown scale
  had been calibrated from the reported N, forcing agreement. Version 0.3.4
  makes that unidentified contract explicit.
- **`bipred.qc.sd_consistency`** -- LDpred2's SD check with both trait types put
  on a common scale. The published form normalises only quantitative traits, on
  the reasoning that a binary trait's effective N fixes the scale; on
  GC-corrected statistics that fails, and the unnormalised ratio sat at 0.755
  against ~1 for a well-specified trait. A threshold calibrated on one then
  removes 83% of the other.
- **`bipred.qc.in_long_range_ld`** with the 24 Price et al. 2008 regions and
  APOE, 136 Mb in total.
- **`benchmarks/qc_factorial.py`** -- the factorial, recording bipred and
  cross-trait LDSC on identical variants and sample sizes, plus the overlap
  readouts and MCMC iterate spreads.

### Changed

- In these 24 arms, the LD-consistency screen separated the warning outcome:
  12/12 unscreened arms warned and 0/12 screened arms did. Strict thresholds
  and long-range exclusion did not change the warning count, but long-range
  exclusion shifted screened `rg` by 0.0001--0.0067 for LDL x CAD, about 0.012
  for height x LDL, and 0.021--0.023 for HDL x TG.
- `docs/guide.md`'s QC section is rewritten around that result.

### Withdrawn

- **"A bivariate fit tolerates less summary-statistic error than a univariate
  one" was too general.** The failure follows the *file*: all three GLGC lipid files
  diverged in every pairing, height and CAD in none, and on height x LDL one
  trait diverged while the other was estimated correctly in the same fit.
- **"bipred's h2 runs systematically below LDSC" is false.** It is below for
  LDL and CAD and 57% *above* for height, where 0.415 sits against rough
  external context around 0.45 and LDSC's 0.264 looks low. The
  truncated-LD-score mechanism proposed
  for it cannot produce both directions.

### Measured

Reported and corrected: an earlier note claimed the effective-N fix reconciled
CAD's h2 with LDSC. It does not -- the correction moves both estimates and the
ratio stays near 0.58. Across the four screened arms, the joint and LDSC `rg`
estimates differ by 1.16--1.43 LDSC standard errors for LDL x CAD and
2.93--4.74 for HDL x TG.

For HDL x TG, all four screened arms fall inside the rough external range of
-0.5 to -0.6 used by the historical study, while only one of four unscreened
arms does. That uncited range is context rather than ground truth. External
agreement alone still cannot establish convergence; fit diagnostics and data
checks are required.

## [0.3.2] - 2026-08-08

A second real trait pair, chosen to test what LDL x CAD could not: GLGC HDL x
TG, where the genetic correlation is strongly *negative* and the two GWAS share
their individuals rather than being disjoint.

Both new checks held up. The sign came back negative throughout, and the
divergence guard fired on the harmonisation-only stage of a pair it was not
calibrated against. But the run exposed two further things.

**Uncorrected sample overlap produces a converged fit with a wrong answer.**
The cross-trait LDSC intercept is -0.352 here, against +0.02 for the disjoint
consortia. Fitting with `cross_corr=0` gives `rg` -0.90; supplying the
intercept gives **-0.52**, against rough external context of -0.5 to -0.6.
Neither fit warned,
and neither should have: cancellation was 0.3, the trace was flat, the sampler
did its job on the inputs it was given. Divergence detection cannot see
mis-specification, and the two failures need separate treatment.

### Fixed

- **`estimate_sample_overlap` reported "no overlap" for negatively correlated
  traits.** It clipped a negative shared-sample inversion to `0.0`, so HDL x
  TG — two lipids measured in the *same people* — came back with `n_shared`
  0.0 and `overlap_frac` 0.0. That reads as a finding rather than as an
  unidentified quantity. The intercept identifies only the product
  `N_shared * rho_pheno`, so when its sign disagrees with `pheno_corr` there is
  no solution; `n_shared` and `overlap_frac` are now `nan`, a `RuntimeWarning`
  names the likely cause, and a new `sign_consistent` key lets callers branch.
  Under the overlap-only model, supplying an external sensitivity value
  (`pheno_corr=-0.45`) gives 72,454 shared samples, 78% of each study. The
  intercept does not identify that phenotypic correlation.
- Documented `overlap_corr` as the direct intercept, independent of
  `rho_pheno`. Version 0.3.4 adds the missing open-interval gate before it can
  be used as a `cross_corr` sensitivity value.

## [0.3.1] - 2026-08-08

First application of bipred to real GWAS, and everything in this release comes
out of that one experiment. The headline is uncomfortable and worth stating
plainly: **on real summary statistics bipred produced a silently diverged fit,
where ldpred3's univariate sampler on the identical LD blocks and the identical
unfiltered data was entirely well behaved.**

The fit reported `h2` 0.64 and a causal fraction of 0.00075 — both inside every
bound — and returned without a warning. It had in fact diverged: posterior
means reached 3.33 against the per-causal effect SD of 0.030 it had itself
inferred, `sum(beta^2)` was 157.5 against a genetic variance of 0.64, and the
genetic variance was still climbing at the final iteration. The runaway effects
sat on variants in near-perfect LD and cancelled inside the quadratic form,
which is exactly why `h2` looked plausible throughout.

None of the thirty architecture cells in `benchmarks/` could have caught it.
They all simulate `beta_hat ~ N(R beta, R/N)` from the model the sampler
assumes, on well-conditioned coalescent LD. The failure needs summary
statistics that disagree with the LD reference, which a simulation drawing both
from the same model cannot produce.

### Added

- **`bipred.qc`** — `dentist` and `dentist_statistic`, a lightweight
  DENTIST-inspired LD-consistency screen. Within a window it splits variants at
  random,
  predicts each z-score from the opposite half through the LD, and drops those
  too far from their neighbourhood's prediction. This is the only check that
  can see the failure: frequency, imputation-quality and chi-square filters all
  judge a variant in isolation. It runs against the blocks you will fit with,
  because an inconsistency only means anything relative to the LD the model
  actually uses. Among the listed per-variant filters, this is the check that
  sees neighbourhood disagreement. Low-rank blocks are screened through their
  factor and never densified.
- **Divergence detection.** A fit now warns when effects are cancelling through
  LD (`sum(beta^2)` against the genetic variance), when a posterior mean
  exceeds the slab the fit itself inferred, or when the retained genetic
  variance drifts systematically. Thresholds are calibrated on the real fit
  above and its repaired counterpart, so each separates the two regimes by more
  than an order of magnitude. This is distinct from the existing
  implausible-fit warning, which keys on a *large* causal fraction or a bound
  being touched and was silent here.
- **`benchmarks/real_ldl_cad.py`** — an end-to-end real-data benchmark on
  public GWAS (GLGC 2013 LDL, CARDIoGRAMplusC4D 2015 CAD) against a UK Biobank
  HapMap3 LD reference, fitting all three cleaning stages so the contrast is
  the result. At that release it asserted the final `rg` landed in an external
  range; 0.3.4 recasts that range as rough context rather than a pass/fail
  check. Not part of `run_all.sh`: it needs about 9 GB of downloads.

### Changed

- `docs/guide.md` gains a *Quality control before fitting real data* section.
  Its original general claim that bivariate fitting tolerates less error was
  withdrawn in 0.3.3: the observed failure followed one GLGC file.
- The divergence warning reports the largest block size when one is large.
  Block size alone is deliberately *not* warned about: the same reference, with
  a 12,169-variant MHC block, fitted cleanly once the summary statistics were
  screened, so a size warning would fire on healthy fits too.
- Documented that `ldpred3.shrink_ld_blocks` keys its shrinkage on `k / n_ref`,
  which assumes finite-panel noise. Against a reference whose correlations were
  thresholded to zero the distortion is structural and does not fall with
  `n_ref`, so the default intensity under-shrinks.

### Measured

The same fit at three levels of cleaning, 924,254 variants before filtering:

**Table 1. Historical LDL x CAD analysis by cleaning stage.**

| | harmonised | + per-variant QC | + LD-consistency screen |
|---|---:|---:|---:|
| `rg` | +0.0558 | +0.1390 | **+0.2856** |
| `h2` LDL | 0.6732 | 0.5688 | **0.0882** |
| `h2` CAD | 0.1713 | 0.0594 | **0.0706** |
| `sum(beta^2)/h2`, LDL | 255 | 271 | **0.7** |
| largest \|effect\|, LDL | 3.193 | 3.322 | **0.024** |
| genetic-variance trace | rising | rising | **settled** |
| cross-trait LDSC `rg` | +0.2238 | +0.1973 | +0.1851 ± 0.052 |

Per-variant filters removed 4.0% and repaired CAD alone (cancellation 91.2 to
4.9) while leaving LDL untouched (255 to 271) — LDL's problem is not a property
of any single variant. The LD-consistency screen removed a further 4.7% and
resolved this case's diagnostics. CAD is fitted at its implied effective sample
size throughout, which is why its `h2` differs from the figures published for
0.3.0.

## [0.3.0] - 2026-08-07

Performance and benchmark-evidence release. **The minor version moves because
two changes are not backward compatible**: the `ld_int8` default now uses the
dense D32 representation rather than quantising it inside the fit, which changes
results for callers who passed float blocks, and the `ldpred3` floor rises to
`>=0.4.5`. The sampler's estimates are otherwise unchanged except where a defect
made them wrong. The self-contained artifact log records one 0.3.0 dirty-tree
sweep; later real-data sections were added separately.

The `ld_int8` change also retires the package's worst documented failure mode;
see the environmental-overlap note below and `benchmarks/RESULTS.md` Table 11.

### Performance

- **Block parallelism silently stopped working after any serial run.** The two
  fused sweep drivers are each jitted twice from one Python function —
  `parallel=True` and `nogil=True` — and Numba keys its on-disk cache on
  (source file, qualname, first line, signature) but *not* on the compilation
  flags. The twins therefore shared one cache entry, and whichever compiled
  first was served to both. Since the cache lives in `__pycache__` beside
  `bivariate.py` and persists, one `ncores=1` run disabled `ncores>1` for every
  later run on that checkout. Measured at m=20,000 / k=500 on LR8: from a cache a
  serial run had touched, `ncores=4` ran at 9.50 ms/sweep against a 9.54 serial
  baseline — no scaling whatever — where a clean cache gave 2.76. The parallel
  twins now opt out of the on-disk cache, which restores full scaling (1.16
  ms/sweep against a 1.12 private-cache reference) bit-identically, at the cost
  of one compilation per process when `ncores>1`.
- **The low-rank sweep kernel is compiled with `fastmath`**, mirroring ldpred3's
  scoping (`_kernels.py:1277`): its O(rank) projection dots are the bulk of a
  low-rank sweep and are add-latency-bound, so letting LLVM reassociate and
  vectorise the reduction is most of the available win. Measured at m=20,000,
  k=500, rank 481, serial: **2.34x** on LR8 (9.46 -> 4.03 ms/sweep) and
  **3.79x** on float32 low-rank (9.36 -> 2.47). Results move at the
  reassociation level (`rg` by 2e-16 relative). The dense kernel is deliberately
  left plain: its O(k) row update is guarded on a variant's effect changing, so
  at a realistic causal fraction it fires on about 1% of visits and the sweep is
  dominated by the four `exp()` calls instead.
- **Per-variant `n_eff` recomputes the sweep's residual-independent scalars only
  when N changes**, rather than once per variant. `_bivar_const` is ~29
  quantities including four logs, and real summary statistics carry long runs of
  identical `n_eff`. It is a pure function of its arguments, so a memo hit is
  bit-identical. Measured **1.69x** on a dense per-variant-N fit with runs of 250
  (1.442 -> 0.853 ms/sweep), which brings per-variant `n_eff` to parity with a
  shared scalar N (0.833) — the penalty for varying N is essentially gone. The
  constant-N path is unchanged.
- **int8 low-rank (LR8) factors are widened into a float32 scratch once per
  sweep**, ported from ldpred3 0.4.5. The projection dots read `U[j, c]` for
  every element of every O(rank) dot, for every variant, every sweep, paying an
  int8 sign-extend-and-convert each time; widening once per block amortises it.
  The scratch is one stride per *thread*, so it stays O(k × rank) and int8
  remains the storage format. Measured **4.03 → 2.97 ms/sweep (1.36x)** at rank
  481. Cumulative with `fastmath`, **9.46 → 2.97 (3.19x)**.
  The conversion is exact, so the fit moves only where `fastmath` reassociates
  (1.1e-15 relative), and both drivers widen, so seeded results remain identical
  across `ncores`.

  The rank gate is bipred's own measurement, not ldpred3's: upstream gates at 64
  because rank 32 was a loss for its kernel, whereas here widening won at every
  rank tested (1.07x at 16, 1.16x at 32, 1.31x at 128, 1.43x at 256), so the
  gate is 32.
- `_pinned_numba_threads` now touches Numba's threading layer even at
  `ncores=1`. The low-rank kernel references `_get_thread_id`, and loading it
  from the on-disk cache before that layer exists **segfaults the interpreter**
  — reproduced on numba 0.66 as an exit-139 on a warm cache with a serial-only
  run. This is why ldpred3 pins the mask unconditionally.
- `benchmarks/sweep_cost.py` measures per-sweep cost by representation and core
  count, giving each cell a private Numba cache in a subprocess — without that
  isolation a grid measures its first arm repeatedly, for the reason above.
  `benchmarks/sweep_cost.csv` records the grid at m=20,000 / k=500.

### Changed

- **`ld_int8` now defaults to `False`, which also resolves the package's worst
  documented failure mode.** The environmental-overlap stress test
  (`benchmarks/RESULTS.md` Table 11) recorded joint-fit MAE up to 0.86 under
  strong shared environment, and that was read as a limit of the model. It was
  not: it was in-fit LD quantization. Re-running the current code with
  `ld_int8=True` reproduces the old row byte for byte (0.1733 / 0.0166 / 0.3101
  / 0.8343 / 0.8603); with the float32 default every cell lands between 0.0072
  and 0.0242, a 36x improvement in the worst one. The int8 resolution (~4e-3 per
  LD entry) is enough to destabilize a fit whose conditioning is already
  stressed, while costing nothing measurable on the well-conditioned panels of
  Tables 2-7 (joint MAE across all 30 architecture cells moved 0.0122 to 0.0123).

  Mechanically, dense blocks are now consumed in the representation they arrive
  in. The previous default quantised float blocks
  of at most 1,500 variants *inside the fit*, allocating a second genome-scale
  int8 payload while the caller's panel was still alive. Measured peak inside
  the call at m=100,000 / k=500: **62.1 MB before, 12.1 MB after** — the extra
  payload is `k` bytes per variant -- measured 121 bytes/variant by default
  against 621 with `ld_int8=True` at k=500 -- so roughly 500 MB at m=1,000,000. This
  follows ldpred3, whose fit-time default is also `False` and which quantises at
  LD-build time, where the float source is private and discardable: prefer
  `ldpred3.compute_ld_blocks(quantize=True)`. `ld_int8=True` and `None` are
  retained for the old behaviour. **This changes results** for callers who
  relied on the default with float blocks, by the int8 quantisation resolution
  (~4e-3 on LD entries) and in the direction of more accuracy.
- **The pinned ldpred3 revision moves to `5d86ac9` (0.4.5)** and the dependency
  floor to `>=0.4.5,<0.5`. The seam was audited against it first, as the README
  requires: all eighteen borrowed symbols still resolve. The bump is what makes
  `_get_thread_id`, and therefore the LR8 widening above, available.
- `regional_rg` no longer warns when handed float blocks at or below the fit's
  old auto-quantise cutoff. That warning existed because the fit-time default
  quantised them into a private copy; with the default consuming blocks as
  given, the pattern it flagged — the same float blocks to both calls — is now
  the aligned one, and the warning was firing on the correct usage.
- Documentation corrections: `docs/rg.md` said a degenerate `rg_decorrelated`
  fit raises (it warns and returns NaN); `docs/guide.md` and `docs/rg.md`
  offered "pass the fit's prepared blocks" as a remedy with no public API
  behind it; `docs/guide.md` Table 3 now marks which options the chains driver
  accepts, rejects, or renames, documents `sample_every`, and gives the chains
  `seed` contract; `ldsc_rg` documents that it is one-step and unfiltered (no
  chi-square cap, so single large-effect loci keep full leverage) and that
  `m_snps` and `ld_scores` must describe the same variant map;
  `RegionalRgResult` no longer claims regions can be aggregated by summing
  (that drops cross-region LD within a block).
- `benchmarks/RESULTS.md` Tables 4–6 were still printing 0.2.0 timing and
  peak-memory numbers against regenerated CSVs, including a 2.51 GB memory
  spike at 80k variants that the current data does not reproduce. Accuracy
  columns were correct and unchanged. A test now re-derives Table 6 from
  `rg_scaling.csv` cell by cell; Tables 4 and 5 were corrected by hand and
  remain unguarded.
- The README gained a citation section pointing at LDpred/LDpred2, cross-trait
  LDSC, and MiXeR.

### Fixed

- **`rg` no longer saturates against `h2_bounds`.** The reported genetic
  correlation divided the raw genetic covariance by the *clamped* per-trait
  heritabilities, so a binding bound drove `rg` toward ±1 while the underlying
  quadratics were unchanged — on one 2,000-variant fixture a true `rg` of 0.43
  was reported as 1.00 after tightening `h2_bounds`. `rg` is now the ratio of
  the raw quadratics, exactly as `docs/algorithm.md` Equation 6 defines it;
  `h2` is still clamped for reporting. The same correction applies to the
  adaptive-stopping check, the pooled multi-chain `rg`, and the per-draw `rg`
  trace feeding split-Rhat (where saturation also faked between-chain
  agreement). Fits whose `h2` stayed inside its bounds are unaffected.
- **`from bipred import estimate_sample_overlap, ldsc_rg` bound the module, not
  the function**, so the documented snippet in `docs/rg.md` raised
  `TypeError: 'module' object is not callable`. The LDSC module was named
  `ldsc_rg`, colliding with the `ldsc_rg` function it exports, and which one
  `bipred.ldsc_rg` resolved to depended on import order. The module is now
  `bipred/ldsc.py`; the public API is unchanged. Submodules
  (`bipred.bivariate`, `bipred.ldsc`, `bipred.multichain`, `bipred.regional`)
  are now reachable as attributes and appear in `dir(bipred)`, and a name
  collision is asserted against at import. `import bipred.ldsc_rg` no longer
  resolves; the function `bipred.ldsc_rg` and every other public name are
  unchanged.
- A diverged chain raised through `ldpred3_auto_bivariate_chains` now keeps its
  `FloatingPointError` type instead of being flattened into `RuntimeError` by
  the chain wrapper, matching the type `_validated_chain_traces` already raises
  for a non-finite trace.
- A diverged fit now raises `FloatingPointError` instead of returning NaN or a
  floored `h2`. NaN was not self-announcing in the sweep: the log-sum-exp left
  `wmax` at the first state, every state probability became NaN, and each
  variant fell through to the both-causal branch, so `h2`, `rg`, `sigma` and
  both effect vectors came back NaN with no error. The *implausible fit*
  warning is also two-sided now — a floored `h2` was previously silent — and
  reads the raw quadratics, since a clamped `h2` cannot show that a bound was
  reached. The low end distinguishes the two cases: a non-positive sampled
  genetic variance is degenerate and also reports `rg` as 0, while a small but
  strictly positive quadratic under the caller's own floor only means the
  reported `h2` is clamped. The warning is still suppressed below 1,000
  variants.
- The benchmark simulator resolved its backend twice: frozen into
  `SIMULATOR_CACHE_TAG` at import, then re-probed per call. `import msprime` is
  not idempotent — a failed attempt can leave partial state that lets a retry
  succeed — so segments could be simulated by msprime and cached under the
  `numba` tag. The backend is now resolved once per process.
- `rg_decorrelated=True` no longer aborts an otherwise usable fit when the
  cross-sweep genetic-variance estimate is degenerate. A non-positive variance
  (exactly zero in finite samples, or slightly negative — e.g. a sparse, weakly
  powered fit whose retained states share no causal support) now warns and
  reports `rg` as `NaN` rather than raising `ValueError`; only non-finite
  cross-sweep quadratics still raise. This removes a platform-dependent crash:
  the degenerate case is reachable through floating-point ordering differences
  (observed on Windows for a `seed` that is fine on the Linux CI).
- The seam gate `test_private_ldpred3_imports_stay_centralised` now reads each
  module as UTF-8 instead of the platform default, so it no longer raises
  `UnicodeDecodeError` on a cp1252 (Windows) checkout over the Greek letters in
  the LDSC module (`bipred/ldsc.py`, then named `ldsc_rg.py`).
- `regional_rg` rejects a float `regions` array containing non-finite labels
  (`NaN`/`inf`) instead of silently pooling every `NaN`-marked variant into one
  spurious cross-genome region — matching the existing rejection of `None`
  labels in object arrays.
- The weekly `ldpred3-head` drift-watch CI leg installs bipred with
  `--no-deps`. It deliberately runs an ldpred3 whose version differs from
  bipred's declared range, so letting pip re-resolve the dependency failed at
  resolution and the leg never reached `pytest` — silently, because it is
  `continue-on-error`.
- Passing `tol > 0` together with `rg_decorrelated=True` to the single-chain
  `ldpred3_auto_bivariate[_blocks]` now emits a `RuntimeWarning` and is
  documented as a no-op (the thinned decorrelated-rg estimator needs the full
  retained schedule). Previously the positive `tol` was silently ignored, while
  the multichain driver already rejects the pairing.

## [0.2.1] - 2026-08-04

Review-hardening release: no estimator changes, no public-API changes.

### Added

- A seam-centralisation gate: `test_private_ldpred3_imports_stay_centralised`
  fails if any underscore-private `ldpred3` import lives outside
  `bipred/_ldpred3_compat.py` (previously uncovered modules such as
  `multichain.py` are now scanned).
- A weekly `ldpred3-head` CI leg (also manually dispatchable) that runs the
  suite against `ldpred3@master` as an informational drift early-warning, plus
  a pin-bump checklist in the README.
- `regional_rg` now warns once when handed a float dense block at or below the
  fit's auto-quantise cutoff — the common call pattern that evaluates different
  LD than a default fit used — naming the three ways to keep the
  representations aligned. Silent for int8, larger float blocks, and
  `LowRankLD`.
- A dependency-free fallback for benchmark simulation: ldpred3's bundled
  Numba coalescent, vendored as `benchmarks/_coalescent.py`, used when
  msprime is absent. msprime stays the default where installed (measured
  0.80 s against 1.64 s for a 10,000-sample, 5 Mb segment), and cached
  segments are tagged per backend so the two never mix.

### Changed

- `rg_decorrelated`'s documentation now leads with "sensitivity diagnostic
  only — do not use for production estimates", carries the measured MAE from
  `RESULTS.md` Table 4, and states the multichain/adaptive-stopping
  incompatibility.
- The environmental-overlap failure mode (joint-fit MAE up to 0.86 in the 0.2.0
  stress test, `RESULTS.md` Table 11) is cross-referenced wherever
  `cross_corr` is documented, with the set-from-external-evidence instruction.
- `test_public_api.py` asserts the exact 13-name public surface instead of
  non-None resolution.
- Benchmark evidence regenerated against ldpred3 0.4.3 with this release.

## [0.2.0] - 2026-08-03

### Added

- Bivariate LDpred for dense and blockwise LD:
  `ldpred3_auto_bivariate`, `ldpred3_auto_bivariate_blocks`, and
  `BivariateResult`.
- Four-state polygenic-overlap summaries, retained-iterate summaries, optional
  residual-noise inflation, and calibration against two univariate ldpred3
  fits.
- Cross-trait LD Score regression (`ldsc_rg`) and the
  `estimate_sample_overlap` helper.
- Deterministic multi-chain fitting with dispersed starts, equal-weight pooling,
  basic split-Rhat diagnostics, and optional chain concurrency through
  `chain_ncores`.
- Regional exploratory genetic correlation (`regional_rg`) from posterior-mean
  effects. Results expose regional covariance, variances, and variant counts;
  known sample-overlap contamination and genome-wide shrinkage are documented.
- Optional single-chain adaptive stopping through `tol` and `check_every`, with
  explicit retained-iteration metadata. This is a stabilization heuristic, not
  a convergence test.

### Changed

- Coherent initialization now makes `h2_init`, `p_init`, and `rg_init` match the
  implied starting moments. `pi_init` exposes overlap directly, and
  `sigma_prior_scale` can keep the shrinkage target fixed across starts.
- The default union-causal start is `p_init=0.02`.
- Dense-LD storage defaults to `ld_int8=None`: supplied int8 remains int8,
  float blocks up to 1,500 variants are quantized, and larger blocks remain
  float32.
- Compact float and LR8 `LowRankLD` blocks use ldpred3's current scalar-scale,
  float32-residual contract without expanding per-variant float64 adapters.
- With Numba, one-core fits fuse each homogeneous block bucket into one native
  call; `ncores>1` uses the parallel twin while preserving seeded results.
- Private ldpred3 helpers are routed through one lazy compatibility module;
  `LowRankLD` is imported from ldpred3's public API.
- The tested ldpred3 dependency revision is
  `5436dcc8152531000b223c5088f726d588d8a8cd` (`ldpred3==0.4.3`).
- Repository benchmark artifacts were regenerated against 0.2.0. They now
  separate generating targets from realized LD-adjusted genetic correlation,
  report paired errors and failures, and directly compare the default and
  cross-sweep sensitivity estimators.

### Removed

- `BivariateResult.mixer_posterior` (deprecated alias of
  `mixer_iterate_summary`) and the unreferenced `LDSCRgResult.rg_ci` property.
- `ldsc_rg`'s local scalar/sample-size validators in favour of the ldpred3
  compatibility seam (`_as_n_vector`, `_finite_control`, `_integer_at_least`).
- The obsolete `allow_legacy_lowrank` fitting and regional options; rebuild old
  row-normalized factors with the pinned ldpred3 version.
- Benchmark scripts and result CSVs are no longer shipped in the sdist; they
  remain in the repository with their README.

### Fixed

- The sdist check now matches the repository-only benchmark policy, and shipped
  documentation links to benchmark material through repository URLs.
- `rg_decorrelated=True` no longer silently substitutes the default estimator
  when its cross-sweep estimate is unavailable.
- Sample-overlap documentation now distinguishes scalar cohort counts from
  effective/sample-varying `N` and states the `noise_inflation` interaction.
- Cross-trait LDSC documents the genome-order requirement for block-jackknife
  standard errors, and `regional_rg` states exactly which supplied LD it
  evaluates and reports undefined ratios for non-positive regional variances.
- Cross-trait LDSC recovers exact signed z statistics from standardized effects
  (`|beta_hat| < 1`); its genotype benchmark supplies exact chi-square input.
- Repository benchmarks use their own lazy-msprime simulator after ldpred3
  removed simulation helpers from its installed package, and tag cached
  segments by simulator schema.
- Within-chain parallel fits restore the caller's Numba thread mask, including
  when a sweep raises.
- Multi-chain fitting rejects adaptive stopping instead of failing later on
  unequal trace lengths; `rg_decorrelated=True` remains unsupported there.
- Input validation now rejects invalid booleans, non-finite controls, malformed
  regional vectors, and non-positive LD scores at public boundaries.
- The fallback `r_g` ratio uses the reported heritability scale, avoiding
  boundary artifacts from non-positive-definite quantized LD.
- Public exports and the private ldpred3 compatibility seam have behavioral
  regression tests.

### Performance

- Per-sweep random and residual buffers are reused.
- Mixed LD panels use fused serial or block-parallel sweeps; regional
  quadratics and LDSC jackknife paths avoid repeated gathers and masks.
- Decorrelated `r_g` stores O(M) online sufficient statistics instead of
  O(retained samples × M) effect traces.
- All-low-rank fits without noise inflation omit two unused genome-length
  residual vectors.

Performance depends on hardware, block balance, LD representation, and current
code. Reproduce the benchmark suite rather than treating development anecdotes
as release guarantees.
