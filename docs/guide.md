# User guide

Use bipred when two harmonized GWAS traits share one ancestry-matched LD
reference. The package returns genetic correlation, SNP heritability,
posterior-mean effects for prediction, and polygenic-overlap summaries.

For a self-contained first run, use
[`examples/minimal.py`](../examples/minimal.py). Model definitions are in
[`algorithm.md`](algorithm.md); estimator interpretation is in
[`rg.md`](rg.md).

## Inputs

Provide:

1. `beta_hat1`, `beta_hat2`: standardized marginal effects in identical variant
   order and allele orientation.
2. `n_eff1`, `n_eff2`: positive scalar or per-variant effective sample sizes.
3. Either one dense LD correlation matrix or blocks `[(R, idx), ...]` whose
   contiguous indices partition `0..m-1`.
4. A scalar `cross_corr` when the two GWAS have correlated sampling errors;
   shared samples are one possible cause.

bipred does not harmonize summary statistics or build LD. Blocks may be dense
float/int8 matrices or ldpred3 `LowRankLD` objects, including LR8, and
representations may be mixed.

The default `ld_int8=None` keeps supplied int8 blocks, quantizes float blocks of
at most 1,500 variants, and keeps larger float blocks as float32. This is a
storage heuristic, not a real-data accuracy guarantee. Use `True` to quantize
all dense float blocks or `False` to keep them float32; this option does not
change low-rank factors.

## Fit one chain

For one dense matrix:

```python
from bipred import ldpred3_auto_bivariate

res = ldpred3_auto_bivariate(
    corr, beta_hat1, beta_hat2, n_eff1, n_eff2, seed=0
)
```

For genome-wide blocks:

```python
from bipred import ldpred3_auto_bivariate_blocks

res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n_eff1, n_eff2,
    burn_in=200, num_iter=200, ncores=2, seed=0,
)
```

With Numba, dense blocks are grouped by dtype and dequantization scale; low-rank
blocks are grouped by dtype and keep their scalar scales per block. `ncores=1`
sweeps each group in one fused serial call; `ncores>1` parallelizes blocks
within each group, while groups remain sequential. Groups with few or imbalanced
blocks may scale poorly. Seeded results agree across the two paths, and every
sweep still waits for all blocks before updating global parameters.

## Fit multiple chains

```python
from bipred import ldpred3_auto_bivariate_chains

fit = ldpred3_auto_bivariate_chains(
    blocks, beta_hat1, beta_hat2, n_eff1, n_eff2,
    n_chains=4, chain_ncores=4, seed=0,
)
res = fit.posterior
```

The default starts are dispersed over union-causal fractions. All finite,
equal-length chains are pooled with equal weight; a failed or wrong-length chain
aborts the fit. `fit.basic_split_rhat` is a classical scalar diagnostic with
degeneracy flags. It neither filters chains nor certifies convergence, and it
does not cover variant-level effects.

`chain_ncores>1` runs independent chains concurrently. Do not combine it with
`ncores>1`: nested threading is rejected because it oversubscribes cores. The
multi-chain driver also rejects `tol>0` and `rg_decorrelated=True`, both of which
require different trace contracts. The pooled posterior records
`retained_iterations = n_chains * retained_per_chain` and
`stopped_early=False`.

## Read the result

**Table 1. Main `BivariateResult` fields.**

| Field | Meaning |
|---|---|
| `beta1_est`, `beta2_est` | posterior-mean standardized effects for scoring |
| `h2` | SNP heritability pair `(h2_1, h2_2)` |
| `rg` | genome-wide genetic correlation |
| `p` | total non-null mixture fraction |
| `pi` | `(pi00, pi10, pi01, pi11)` mixture |
| `sigma` | mean retained 2 × 2 effect covariance |
| `noise_scale` | learned residual factors; `(1, 1)` when disabled |
| `genetic_samples` | retained `(gvar_1, gcov, gvar_2)` trace |
| `retained_iterations` | post-burn-in sweeps actually retained |
| `stopped_early` | whether single-chain adaptive stopping fired |

The overlap summary is available as:

```python
mx = res.mixer
mx["frac_shared"]
mx["rho_beta"]
mx["rg_from_overlap"]
mx["n_causal"], mx["n_shared"]
```

Ratios avoid the literal causal-count interpretation but still need calibration.
LD can spread inclusion mass to correlated neighbours, while reference mismatch
can add inflation. For count-sensitive work, compare `noise_inflation=True` and
`res.mixer_calibrated(infer1, infer2)` with the unadjusted result; neither is
guaranteed to improve calibration at every power setting.

## Genetic correlation

Use `res.rg` by default. For strongly asymmetric-power pairs,
`rg_decorrelated=True` is an optional sensitivity estimator. It had higher
paired error than the default in the committed 0.2.0 symmetric and asymmetric
synthetic sweeps, so compare it with the default rather than replacing the
default automatically. `ldsc_rg` is a fast independent screen. Interpretation,
sample overlap, and overlap-interval semantics are covered in [`rg.md`](rg.md).

For per-region exploratory estimates:

```python
from bipred import regional_rg

local = regional_rg(
    res.beta1_est, res.beta2_est, blocks, region_labels,
    min_variants=50,
)
```

`region_labels` has one label per variant; labels need not be contiguous.
Regional estimates use posterior-mean effects and expose `rg`, `gcov`, `gvar1`,
`gvar2`, and `n_variants`. They are best used for ranking or comparison:
uncorrected sample overlap contaminates every region, and the genome-wide model
shrinks local estimates toward the genome-wide correlation.

## Options

**Table 2. Main fitting options.**

| Option | Default | Use |
|---|---:|---|
| `burn_in`, `num_iter` | `200`, `200` | burn-in and retained sweeps |
| `h2_init`, `p_init`, `rg_init` | `0.1`, `0.02`, `0` | coherent genetic-moment start; `p_init` is union-causal |
| `pi_init` | `None` | explicit four-state overlap start |
| `sigma_prior_scale` | `None` | fixed covariance shrinkage target across starts |
| `cross_corr` | `0` | correlation of cross-trait sampling noise (set from external evidence when traits share environmental effects — see `docs/rg.md`) |
| `rg_decorrelated` | `False` | cross-sweep sensitivity estimator for asymmetric power |
| `noise_inflation`, `ni_damp` | `False`, `0.1` | learn and damp residual-noise inflation |
| `pi_prior` | `1` | symmetric Dirichlet mixture concentration |
| `h2_bounds`, `h2_cap` | `(1e-4, 1)`, `None` | ordinary bounds and optional expert ceiling |
| `iw_df` | `10` | covariance shrinkage strength |
| `ld_int8` | `None` | automatic, forced-int8, or forced-float dense storage |
| `ncores` | `1` | within-chain block threads |
| `chain_ncores` | `1` | multi-chain concurrency; cannot combine with `ncores>1` |
| `tol`, `check_every` | `0`, `50` | optional single-chain stabilization heuristic |
| `seed` | `None` | random seed |

With `tol>0`, a single chain checks the relative RMS change in both
posterior-mean effect vectors and the change in `r_g` every `check_every`
retained sweeps. Meeting the threshold can stop the run early. This is a
schedule- and seed-dependent stabilization heuristic, not evidence that the
Markov chain converged. It is disabled for `rg_decorrelated=True` and unsupported
by multi-chain inference; use dispersed full-length chains for diagnostics.

## Pitfalls

- Match ancestry and harmonize variant order, alleles, and effect scale before
  fitting.
- Supply a defensible `cross_corr` when cross-trait sampling errors are
  correlated. Sample overlap is one cause; an LDSC intercept can also contain
  confounding and does not identify overlap by itself.
- Treat low-power absolute causal counts and regional absolute `r_g` values as
  exploratory.
- Vary `pi_init` as a prespecified sensitivity analysis when overlap is weakly
  identified.
- Validate prediction out of sample.
- A warning about an *implausible fit* means a large causal fraction or an
  `h2` bound was hit on a sizeable panel. Inspect LD quality, reference size,
  block size, and regularization before interpreting or timing that fit.
