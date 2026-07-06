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
   - dense blocks `[(R, idx), ...]` with contiguous `idx` arrays partitioning
     `0..m-1`.
4. Optional `cross_corr` if the GWAS share samples.

bipred does not build LD or harmonize summary statistics. Use ldpred3 for that
preparation. The bivariate sampler currently requires dense LD blocks; ldpred3
`LowRankLD` blocks are rejected.

## Quickstart

Single dense LD matrix:

```python
from bipred import ldpred3_auto_bivariate

res = ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n1, n2, seed=0)
print(res)
```

Genome-wide dense blocks:

```python
from bipred import ldpred3_auto_bivariate_blocks

res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n1, n2,
    burn_in=200, num_iter=200, seed=0,
)
```

## Reading `BivariateResult`

| field | meaning |
|---|---|
| `beta1_est`, `beta2_est` | posterior-mean standardized effects for PRS scoring |
| `h2` | `(h2_1, h2_2)` SNP heritabilities |
| `rg` | genetic correlation |
| `p` | total non-null mixture fraction |
| `sigma` | learned 2x2 effect covariance |
| `pi` | `(pi00, pi10, pi01, pi11)` mixture: neither / trait 1 / trait 2 / both |
| `noise_scale` | learned `(lambda1, lambda2)` if `noise_inflation=True` |

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

`res.mixer_posterior()` maps retained Gibbs draws to posterior intervals.
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

```text
cross_corr = N_shared * rho_pheno / sqrt(N1 * N2)
```

For fully shared samples, this is the phenotypic correlation among shared
individuals. If unknown, estimate it from the cross-trait LDSC intercept:

```python
from bipred import estimate_sample_overlap

rgr = ldsc_rg(beta_hat1, beta_hat2, ell, n1, n2)
estimate_sample_overlap(rgr, n1, n2, pheno_corr=0.4)
```

`cross_corr=0` is a reasonable default for independent studies or first-pass
analyses, but it can bias `r_g` upward when samples overlap strongly.

## Main Options

| option | default | use |
|---|---:|---|
| `burn_in`, `num_iter` | `200`, `200` | Gibbs burn-in and sampling sweeps |
| `h2_init`, `p_init`, `rg_init` | `0.1`, `0.02`, `0.0` | starting values |
| `cross_corr` | `0.0` | sample-overlap noise correlation |
| `rg_decorrelated` | `False` | better `r_g` for asymmetric-power pairs |
| `noise_inflation`, `ni_damp` | `False`, `0.1` | learn residual noise inflation |
| `pi_prior` | `1.0` | symmetric Dirichlet mixture prior |
| `h2_bounds`, `h2_cap` | `(1e-4, 1.0)`, `None` | heritability clamps |
| `iw_df` | `10.0` | shrinkage strength on `sigma` |
| `sample_every` | `5` | thinning for retained effect samples |
| `seed` | `None` | RNG seed |

## Pitfalls

- Use the same ancestry LD reference for both traits.
- Harmonize variant order and allele orientation before fitting.
- Keep effects on ldpred3's standardized scale.
- Use dense LD blocks; low-rank blocks are not supported by the bivariate sampler.
- Do not over-interpret absolute overlap counts at low power.
- Increase `burn_in` and `num_iter` if `h2` or `r_g` is unstable across seeds.
