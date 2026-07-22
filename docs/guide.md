# bipred user guide

bipred is a Python library for fitting two GWAS traits jointly with one matched
LD reference. It returns per-trait SNP heritability, genetic correlation `r_g`,
posterior-mean effects for prediction, and a MiXeR-style overlap summary.

Use this page for day-to-day usage. See [algorithm.md](algorithm.md) for the
model and [rg.md](rg.md) for estimator details.

## When to use bipred

Use bipred when:

- both traits have summary statistics on the same ancestry,
- variants are harmonized to the same order and allele orientation, and
- you want either genetic correlation, cross-trait PRS borrowing, or polygenic
  overlap.

The most reliable outputs are `res.rg`, the posterior-mean effects, and the
overlap ratios. Absolute causal counts are useful but approximate.

## Inputs

You need:

1. `beta_hat1`, `beta_hat2`: standardized marginal effects for the same variants.
2. `n_eff1`, `n_eff2`: scalar or per-variant effective sample sizes.
3. LD for those variants:
   - a dense correlation matrix `corr`, or
   - blocks `[(R, idx), ...]` with contiguous `idx` arrays partitioning
     `0..m-1`, where each `R` is dense or an ldpred3 `LowRankLD`.
4. Optional `cross_corr` if the GWAS share samples.

bipred does not build LD or harmonize summary statistics. Use ldpred3 for that
preparation. The bivariate sampler consumes dense, compact float `LowRankLD`,
and LR8 blocks, including mixed block lists.

The default `ld_int8=None` policy keeps supplied int8 blocks as-is, quantises
float blocks with at most 1500 variants, and keeps larger float blocks float32.
Small D8 blocks use a quarter of float32 storage and are dequantised in the
sampler; the size cutoff avoids quantising large dense blocks where conditioning
can be sensitive. Use `ld_int8=True` to quantise every dense float block or
`False` to keep dense float inputs float32. This setting does not alter
`LowRankLD` factors.

## Quickstart

Single dense LD matrix:

```python
from bipred import ldpred3_auto_bivariate

res = ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n1, n2, seed=0)
print(res)
```

Genome-wide blocks:

```python
from bipred import ldpred3_auto_bivariate_blocks

res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n1, n2,
    burn_in=200, num_iter=200, seed=0,
)
```

For auditable dispersed starts, run chains sequentially:

```python
from bipred import ldpred3_auto_bivariate_chains

fit = ldpred3_auto_bivariate_chains(
    blocks, beta_hat1, beta_hat2, n1, n2,
    n_chains=4, seed=0,
)
res = fit.posterior
```

All finite, equal-length chains contribute equally. Any non-finite or
wrong-length chain aborts instead of being discarded. `fit.basic_split_rhat`
contains classical basic split-Rhat values plus explicit degeneracy flags; it
does not filter chains or claim convergence. The driver does not support
`rg_decorrelated=True`.

## Reading `BivariateResult`

**Table 1. Main result fields.**

| field | meaning |
|---|---|
| `beta1_est`, `beta2_est` | posterior-mean standardized effects for PRS scoring |
| `h2` | `(h2_1, h2_2)` SNP heritabilities |
| `rg` | genetic correlation |
| `p` | total non-null mixture fraction |
| `sigma` | mean of the retained 2x2 effect-covariance iterates |
| `pi` | `(pi00, pi10, pi01, pi11)` mixture: neither / trait 1 / trait 2 / both |
| `noise_scale` | learned `(lambda1, lambda2)`; always present, `(1.0, 1.0)` when `noise_inflation=False` |
| `genetic_samples` | retained raw `(gvar_1, gcov, gvar_2)` quadratic traces |
| `noise_scale_samples` | retained `(lambda1, lambda2)` trace; all ones when inflation is off |

Overlap summary:

```python
mx = res.mixer
mx["polygenicity"]     # per-trait causal fractions
mx["n_causal"]         # per-trait causal counts
mx["n_shared"]         # shared causal count
mx["frac_shared"]      # shared fraction of the less-polygenic trait
mx["rho_beta"]         # effect correlation within the shared component
mx["rg_from_overlap"]  # overlap decomposition of r_g
```

`res.mixer_iterate_summary()` maps retained `pi` draws and covariance iterates
to empirical retained-chain intervals. Because `sigma` uses a damped moment
update rather than a conditional posterior draw, these are not Bayesian credible
intervals. `res.mixer_posterior()` is a deprecated compatibility alias.
`res.mixer_calibrated(infer1, infer2)` rescales counts using two univariate
`ldpred3_auto_infer` fits.

## Genetic Correlation

Use `res.rg` as the main estimate. It uses the joint LD likelihood and is usually
more precise than cross-trait LD Score regression.

Use `rg_decorrelated=True` when one trait is much better powered than the other:

```python
res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n1, n2,
    rg_decorrelated=True,
)
```

For a fast independent screen:

```python
from bipred import ldsc_rg
from ldpred3 import ld_scores

ell = ld_scores(blocks)
rgr = ldsc_rg(beta_hat1, beta_hat2, ell, n1, n2)
rgr.rg, rgr.rg_se, rgr.gcov_intercept
```

If LDSC returns a huge `|rg|` or standard error on a small panel, check its
marginal `h2` estimates. The ratio can blow up when either LDSC heritability is
near zero.

## Prediction

Score `beta1_est` and `beta2_est` like ordinary LDpred weights. The joint model
helps most when one trait is under-powered and genetically correlated with a
better-powered trait. Validate prediction out of sample; when there is little
shared signal, the joint fit should largely decouple the traits but can still add
Monte Carlo noise.

## Polygenic Overlap

The stable overlap outputs are ratios:

- `frac_shared`
- `rho_beta`
- `rg_from_overlap`

Absolute counts (`n_causal`, `n_shared`) are approximate. LD can spread posterior
inclusion mass around causal variants, and finite reference panels can add more
inflation. The default `p_init=0.02` avoids the older low-power over-count seen
with larger initial polygenicity, but very low-power counts are still weakly
identified.

For count-sensitive work:

- set `noise_inflation=True` when fitting on finite reference-panel LD,
- use `res.mixer_calibrated(infer1, infer2)` to anchor per-trait counts on
  univariate ldpred3 fits, and
- treat MiXeR-style counts as descriptive unless validated for the architecture.

## Sample Overlap

If the two GWAS share samples, pass the cross-trait sampling-noise correlation:

```python
res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n1, n2,
    cross_corr=cross_corr,
)
```

If known,

**Equation 1. Sampling-noise correlation from shared samples.**

```text
cross_corr = N_shared * rho_pheno / sqrt(N1 * N2)
```

For fully shared samples, this is the phenotypic correlation among shared
individuals. If unknown, approximate it from the cross-trait LDSC intercept:

```python
from bipred import estimate_sample_overlap

rgr = ldsc_rg(beta_hat1, beta_hat2, ell, n1, n2)
estimate_sample_overlap(rgr, n1, n2, pheno_corr=0.4)
```

This inversion requires an assumed non-zero phenotypic correlation. The LDSC
intercept can also contain correlated population structure, measurement effects,
or other cross-trait confounding, so it does not identify sample overlap by
itself. Under an overlap-only model with a positive intercept and phenotypic
correlation, using `pheno_corr=1` gives the effective overlap and the smallest
shared-sample count compatible with `|rho_pheno| <= 1`--a lower bound, not an
upper bound.

`cross_corr=0` is a reasonable default for independent studies or first-pass
analyses, but it can bias `r_g` upward when samples overlap strongly.

## Main Options

**Table 2. Main sampler options.**

| option | default | use |
|---|---:|---|
| `ld_int8` | `None` | dense only: auto D8 through 1500 variants and float32 above; `True`/`False` force either dense policy |
| `burn_in`, `num_iter` | `200`, `200` | Gibbs burn-in and sampling sweeps |
| `h2_init`, `p_init`, `rg_init` | `0.1`, `0.02`, `0.0` | exact genetic-moment starts; `h2_init` may be a pair and `p_init` is union-causal |
| `pi_init` | `None` | explicit `(pi00, pi10, pi01, pi11)` start for overlap-sensitive work |
| `sigma_prior_scale` | `None` | fixed per-trait slab-variance shrinkage target; decouples starts from the prior |
| `cross_corr` | `0.0` | sample-overlap noise correlation |
| `rg_decorrelated` | `False` | better `r_g` for asymmetric-power pairs |
| `noise_inflation`, `ni_damp` | `False`, `0.1` | learn residual noise inflation |
| `pi_prior` | `1.0` | symmetric Dirichlet mixture prior |
| `h2_bounds`, `h2_cap` | `(1e-4, 1.0)`, `None` | heritability clamps |
| `iw_df` | `10.0` | shrinkage strength for the moment update of `sigma` |
| `sample_every` | `5` | thinning for retained effect samples (only with `rg_decorrelated=True`) |
| `seed` | `None` | RNG seed |

## Pitfalls

- Use the same ancestry LD reference for both traits.
- Harmonize variant order and allele orientation before fitting.
- Keep effects on ldpred3's standardized scale.
- Dense, `LowRankLD`, and mixed block lists are supported. A low-rank fit uses
  the approximation encoded by its retained factor, so choose its rank or
  retained variance deliberately. `ld_int8` controls dense storage only.
- Do not over-interpret absolute overlap counts at low power.
- Increase `burn_in` and `num_iter` if `h2` or `r_g` is unstable across seeds.
- At low power, vary `pi_init` as a pre-specified sensitivity analysis: scalar
  `p_init` cannot determine marginal polygenicity and causal overlap separately.
