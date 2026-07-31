# Changelog

User-visible changes to **bipred** are recorded here. The project is currently
`0.1.0.dev1`; all entries remain unreleased.

## [Unreleased]

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
- Compact float and LR8 `LowRankLD` blocks use ldpred3's
  low-rank-plus-diagonal contract. Legacy row-normalized factors require
  `allow_legacy_lowrank=True`.
- `ncores>1` buckets blocks by representation, dtype, and scale, so mixed LD
  panels remain parallel while seeded results match `ncores=1`.
- The tested ldpred3 dependency revision is
  `db3ebd2385b7e3f347712f8761682c0eb49df3e4` (`ldpred3>=0.2.13,<0.3`).

### Removed

- `BivariateResult.mixer_posterior` (deprecated alias of
  `mixer_iterate_summary`) and the unreferenced `LDSCRgResult.rg_ci` property.
- `ldsc_rg`'s local scalar/sample-size validators in favour of the ldpred3
  compatibility seam (`_as_n_vector`, `_finite_control`, `_integer_at_least`).
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
- Mixed LD panels use fused block-parallel sweeps; regional quadratics and LDSC
  jackknife paths avoid repeated gathers and masks.
- Decorrelated `r_g` uses fewer simultaneous effect-sized arrays.

Performance depends on hardware, block balance, LD representation, and current
code. Reproduce the benchmark suite rather than treating development anecdotes
as release guarantees.
