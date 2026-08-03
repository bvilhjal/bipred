# Changelog

User-visible changes to **bipred** are recorded here. The project is currently
`0.2.0`.

## [Unreleased]

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
