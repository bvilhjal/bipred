# Changelog

All notable changes to **bipred** are recorded here. The format is loosely based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **LDpred3 0.2.13 compatibility.** The tested dependency revision is now
  `e0f6171a2635f87d60134b6d76bc8cdb7ab81119`. BiPred no longer imports or
  advertises LDpred3's removed packed-triangular D8T representation, and its
  declared minimum is now `ldpred3>=0.2.13`. Cross-trait LDSC explicitly checks
  predictor variation before calling LDpred3's WLS helper, preserving the
  documented full-fit error and undefined jackknife SE for unidentified fits.
  A public-export smoke test now catches lazy import failures at the dependency
  seam.
- **Coherent four-state initialization.** `h2_init` (now scalar or per-trait),
  `p_init`, and `rg_init` now initialize a `pi`/`Sigma` pair whose implied
  marginal heritabilities and genetic correlation equal those values exactly.
  Previously, the equal non-null split made the implied marginal `h2` only
  `2/3 * h2_init` and the implied genetic correlation `rg_init / 2`.
  `p_init` is now documented precisely as the union-causal probability;
  `pi_init` exposes the otherwise-unidentified four-state overlap, and
  `sigma_prior_scale` can hold the persistent covariance-prior target fixed
  while starts vary. Seeded outputs change because this corrects the actual
  starting covariance rather than merely relabelling it.
- **Size-aware automatic dense-LD storage.** `ld_int8=None` is now the default:
  supplied int8 blocks stay int8, float blocks with at most 1500 variants are
  int8-quantised, and larger float blocks remain float32. This keeps D8's
  fourfold storage saving on small blocks without exposing large dense blocks to
  conditioning-sensitive quantisation. `ld_int8=True` quantises every float
  block and `False` keeps every float input float32. Existing small-block int8
  goldens are unchanged.
- **Default `p_init` lowered from `0.1` to `0.02`** for the bivariate sampler
  (`ldpred3_auto_bivariate[_blocks]`), matching ldpred3's realistic ~2 %-causal
  starting polygenicity. Besides being a better default, this **fixes the
  low-power absolute-count over-count**: at low per-SNP power the single-chain
  count is influenced by its starting value, and the old `p_init=0.1` inflated it
  ~3×. With `p_init=0.02` the `mixer` `n_causal`/`n_shared` calibrate to ≈1× truth
  across the whole power range (benchmarked below). The mixture is still updated
  each sweep; the bivariate golden test pins `p_init` explicitly, so it is
  unaffected.

### Added
- **Deterministic within-chain block parallelism.** `ncores>1` now fuses
  homogeneous dense or homogeneous low-rank blocks into one Numba `prange`
  sweep. Random arrays are generated before launch and the three counts plus six
  floating statistics are reduced in genome order, preserving seeded
  `ncores=1` results exactly. Mixed representations or dtypes fall back
  serially. Multi-chain inference still runs chains sequentially; this setting
  parallelises blocks within each chain.
- **Sequential multi-chain bivariate inference.**
  `ldpred3_auto_bivariate_chains` runs deterministic dispersed chains one at a
  time under one shared covariance prior and pools every finite, equal-length
  chain with equal weight. Non-finite or unequal chains abort rather than being
  filtered. The result exposes auditable starts/seeds and classical basic
  split-Rhat values with degeneracy metadata, but makes no convergence claim
  and has no `converged` flag. `rg_decorrelated=True` is not supported by
  this driver.
- **Compact low-rank LD inference.** The bivariate block sampler now
  consumes ldpred3 float `LowRankLD` and LR8 factors without materialising dense
  LD, and permits mixed dense/low-rank block lists. It maintains two persistent
  rank-size score vectors per low-rank block, giving `O(k*r)` sweep work and
  storage for block size `k` and rank `r`. The adapter supports both released
  row-normalised factors and ldpred3's newer globally scaled factor plus
  diagonal-residual contract; `ld_int8` continues to control dense storage only.
- **Configurable four-state mixture prior (`pi_prior`).** The Dirichlet
  concentration for the per-sweep `π` draw is now a parameter (default `1.0`,
  the historical uniform prior; `0.5` is Jeffreys). Backward compatible. Mirrors
  ldpred3's univariate `p_prior`; a *minor* lever for the absolute counts under
  real LD (matters mainly in the no-LD limit).
- **Re-characterized and re-benchmarked the absolute polygenic counts (docs +
  benchmark).** Against known truth, with the realistic `p_init=0.02` default the
  `mixer` counts are **well calibrated across the whole per-SNP power range**
  (per-trait `count/true` ≈1.0 up to `N·h²/M~0.5`, ≈1.1–1.2 by `N·h²/M=2`) — the
  large low-power over-count reported in earlier development was a `p_init=0.1`
  artifact. The residual is a mild over-count that **grows with power**
  (LD-spreading — correlated SNPs recruited around each causal), a little larger
  on a finite reference panel. Trimmed by `noise_inflation` (mismatch part) and
  by univariate anchoring via `mixer_calibrated`; `r_g` and the overlap ratios
  cancel it. **New `unical` sweep** in `benchmarks/mixer_overlap.py` benchmarks
  `mixer_calibrated` vs truth (joint vs calibrated per-trait and shared counts
  across power); the `ldmatch` framing was corrected (LD-spreading vs
  reference-mismatch) and all sweeps were rerun. `docs/rg.md` and
  `docs/algorithm.md` (the "Calibrating the counts" table) rewritten accordingly.
- **Initial release: bivariate (two-trait) LDpred, split out of `ldpred3`.**
  The four-state joint sampler (`ldpred3_auto_bivariate` /
  `ldpred3_auto_bivariate_blocks`, returning `BivariateResult`) moves here
  unchanged from `ldpred3/bivariate.py`. It jointly fits two traits sharing one
  LD reference and reports per-trait SNP heritability, the genetic correlation
  `r_g` (two estimators: the same-sweep quadratic ratio and a decorrelated
  variant for asymmetric-power pairs), the four-state causal mixture
  `(π₀₀, π₁₀, π₀₁, π₁₁)`, and posterior-mean effects that let a well-powered
  trait sharpen a correlated under-powered one. Sample overlap is handled via
  `cross_corr`; the effect-covariance `Σ` uses an inverse-Wishart-inspired
  diagonal shrinkage target (`iw_df`) in a damped moment update.
- **MiXeR-style polygenic-overlap parameters** (`BivariateResult.mixer` and
  `.mixer_calibrated`): per-trait and shared polygenicity, the shared fraction,
  the within-shared effect correlation `ρ_β`, and the `r_g` overlap
  decomposition. `mixer_calibrated` anchors the absolute counts on two univariate
  `ldpred3_auto_infer` runs.
- **Retained-iterate summaries for overlap counts** —
  `BivariateResult.mixer_iterate_summary(level=0.95)`. The sampler retains
  post-burn-in mixture draws and effect-covariance iterates (`pi_samples`,
  `sigma_samples`) and maps them through the MiXeR decomposition to return an
  empirical mean and central iterate interval for `n_causal`, `n_shared`,
  `frac_shared`, `ρ_β`, and `rg_from_overlap`. Because `Sigma` receives a damped
  moment update rather than a conditional posterior draw, these are not Bayesian
  credible intervals and no calibration claim is made. They also do not capture
  LD-reference-mismatch bias. The old `mixer_posterior()` name remains as a
  deprecated compatibility alias.
- **Noise-inflation option for calibrated absolute counts** —
  `ldpred3_auto_bivariate*(..., noise_inflation=True)`. Learns a per-trait
  LDSC-intercept-style factor `λ_t ≥ 1` from the residual misfit
  (`b_hat − R·β`) and fits with an effective `N_t / λ_t`. Under a matched LD
  reference the residual is pure sampling noise so `λ ≈ 1` (a no-op); under
  LD-reference mismatch it is inflated, so `λ > 1` makes the sampler stop reading
  the misfit as extra polygenicity. This removes the **N-growing** component of
  the polygenic-overlap count inflation with `h²`/`rg` unchanged: in the
  committed `calibration` sweep (`benchmarks/mixer_overlap.csv`, N = 1k-20k),
  the reference-mismatch count inflation rises to ~1.2× with the option off and
  is pulled back to ~1.0× with it on (learned `λ` ≈ 1.1-1.2; a scalar `λ` can't
  absorb structured mismatch entirely). Off
  by default; the learned factors are on `BivariateResult.noise_scale`. New
  `benchmarks/mixer_overlap.py` `calibration` sweep reports the on/off relative
  polygenicity, `λ`, and retained-iterate interval inclusion across power.
- **Cross-trait LD Score regression** (`ldsc_rg`, `LDSCRgResult`,
  `estimate_sample_overlap`), moved from ldpred3 so bipred owns *all*
  genetic-correlation estimation. It is the fast, moment-based `r_g` estimator and
  the independent cross-check on the joint fit, and reuses ldpred3's univariate
  LDSC internals (`ld_scores`, and the `_wls` / `_weights` helpers from
  `ldpred3.ldsc`). ldpred3 keeps only univariate `ldsc_h2`.
- **Tests** carried over from ldpred3: statistical recovery of `r_g` / `h²` /
  overlap and cross-trait borrowing (`tests/test_bivariate.py`), plus a
  bit-exact golden characterization test (`tests/test_golden_bivariate.py`).
- **Benchmarks and docs** for genetic-correlation accuracy vs bivariate LDSC,
  sample-overlap corrections, MiXeR-style overlap recovery, and weak-trait
  prediction gain (`benchmarks/`, `docs/`).

### Fixed
- **Review follow-ups** (theory / documentation / implementation review; see
  `REVIEW.md`). The shorthand initialization with `|rg_init| > 0.999` (and
  `pi_init=None`) could produce negative single-trait mixture masses; the
  shared mass now saturates at the union probability (an all-shared start),
  keeping the implied moments exact. The fallback `rg` ratio now uses the
  clamped `h2`-scale denominators, so a non-PD (int8-quantised) block can no
  longer slam `rg` to ±1 through the variance floor. Validation tightening:
  `h2_init` / `sigma_prior_scale` reject numeric strings, `ldsc_rg` sample
  sizes reject mixed bool/string sequences, and `estimate_sample_overlap`
  validates its result type. Documentation corrections: the `rg_decorrelated`
  bias mechanism (same-sweep coupling *inflates* the genetic covariance; the
  sampled-quadratic ratio attenuates through the weak trait's sampled
  variance), the `res.h2` estimand, the Equation 1 noise covariance, the
  `noise_inflation` × `cross_corr` interaction, compact-LD rejection, the full
  ldpred3 seam (Notes), and the `mixer_overlap.py` /
  `bivariate_demo.py` benchmark-table rows; the noise-inflation numbers above
  now match the committed calibration sweep.

### Notes
- bipred depends on `ldpred3` (`>=` the release that removes the in-tree
  `bivariate` and cross-trait-`ldsc_rg` code) for the shared LD representations,
  the Numba sampler shim and the univariate LDSC machinery. The private seam:
  `HAVE_NUMBA`, `_jit`, `_jit_parallel`, `_set_threads`, `prange`,
  `_as_n_vector`, `LowRankLD`, `_check_h2_p`,
  `_finite_control`, `_integer_at_least`, `_validate_beta_hat`,
  `_validate_blocks`, `_validate_boolean_controls`, `_validate_iterations` and
  `_validate_seed` from `ldpred3.ldpred3`; `_wls` and `_weights` from
  `ldpred3.ldsc`; and `_Q8` from `ldpred3._kernels`.
