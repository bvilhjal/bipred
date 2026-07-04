# Changelog

All notable changes to **bipred** are recorded here. The format is loosely based
on [Keep a Changelog](https://keepachangelog.com/), and the project aims to
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
