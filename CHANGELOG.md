# Changelog

All notable changes to **bipred** are recorded here. The format is loosely based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
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
  `cross_corr`; the effect-covariance `Σ` is regularised by an inverse-Wishart
  diagonal prior (`iw_df`).
- **MiXeR-style polygenic-overlap parameters** (`BivariateResult.mixer` and
  `.mixer_calibrated`): per-trait and shared polygenicity, the shared fraction,
  the within-shared effect correlation `ρ_β`, and the `r_g` overlap
  decomposition. `mixer_calibrated` anchors the absolute counts on two univariate
  `ldpred3_auto_infer` runs.
- **Posterior distribution of the overlap counts** — `BivariateResult.mixer_posterior(level=0.95)`.
  The sampler now retains the post-burn-in mixture / effect-covariance draws
  (`pi_samples`, `sigma_samples`), and `mixer_posterior` maps each draw through
  the MiXeR decomposition to return the posterior **mean + credible interval**
  for `n_causal`, `n_shared`, `frac_shared`, `ρ_β` and `rg_from_overlap` — the
  posterior overlap counts given the prior and data, rather than only the
  `mixer` point estimate. Validated on known-truth simulations: under a matched
  LD reference the interval is calibrated and covers the truth; the interval
  captures sampling uncertainty, **not** LD-reference-mismatch bias (which
  inflates the absolute counts, growing with N, and is an LD-quality issue — the
  ratios stay reliable regardless).
- **Noise-inflation option for calibrated absolute counts** —
  `ldpred3_auto_bivariate*(..., noise_inflation=True)`. Learns a per-trait
  LDSC-intercept-style factor `λ_t ≥ 1` from the residual misfit
  (`b_hat − R·β`) and fits with an effective `N_t / λ_t`. Under a matched LD
  reference the residual is pure sampling noise so `λ ≈ 1` (a no-op); under
  LD-reference mismatch it is inflated, so `λ > 1` makes the sampler stop reading
  the misfit as extra polygenicity. This removes the **N-growing** component of
  the polygenic-overlap count inflation with `h²`/`rg` unchanged: on
  well-conditioned LD the counts calibrate ~fully (e.g. n₁ 909→309 vs truth 300
  at N=200k), and on realistic coalescent LD it cuts the inflation from ~2.4× to
  ~1.6× at N=200k (a scalar `λ` can't absorb structured mismatch entirely). Off
  by default; the learned factors are on `BivariateResult.noise_scale`. New
  `benchmarks/mixer_overlap.py` `calibration` sweep reports the on/off relative
  polygenicity, `λ`, and `mixer_posterior` coverage across power.
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

### Notes
- bipred depends on `ldpred3` (`>=` the release that removes the in-tree
  `bivariate` and cross-trait-`ldsc_rg` code) for the shared LD representations,
  the Numba sampler shim and the univariate LDSC machinery. It imports `_jit`,
  `_as_n_vector` and `LowRankLD` from `ldpred3.ldpred3`, and `_wls` / `_weights`
  from `ldpred3.ldsc`.
