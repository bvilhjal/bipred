# Genetic correlation and overlap

This page covers estimator choice, sample overlap, regional exploration, and
polygenic-overlap interpretation. See [`guide.md`](guide.md) for fitting.

## Genome-wide estimators

**Table 1. Genetic-correlation estimators.**

| Estimator | Use | Main caveat |
|---|---|---|
| `res.rg` | default joint LD estimate | needs well-matched LD |
| `rg_decorrelated=True` | asymmetric-power sensitivity check | limited direct benchmarks; requires a full single-chain schedule |
| `bipred.ldsc_rg` | fast screen or independent check | unstable when marginal LDSC `h2` is near zero |
| two univariate LDpred fits | additional diagnostic | often attenuated under power asymmetry |

Cross-trait LDSC fits the moment relation:

**Equation 1. Cross-trait LD Score regression.**

```text
E[z1_j z2_j] =
    intercept + sqrt(N1 N2) rho_g LD_score_j / M
```

The slope estimates genetic covariance under LDSC assumptions. The intercept
captures correlated sampling noise and cross-trait confounding. `M` is the
variant count defining the heritability and covariance; pass the full count as
`m_snps` when summary statistics are a subset of that map.

For `rg_se`, order rows by chromosome and position. The block jackknife deletes
contiguous ranges of the supplied row order; arbitrary order does not preserve
local LD dependence.

Use the joint fit by default and inspect LDSC as a cheap sensitivity check. The
committed benchmark results are a dated snapshot, not current performance
evidence; see the
[benchmark results](https://github.com/bvilhjal/bipred/blob/main/benchmarks/RESULTS.md).

## Asymmetric-power sensitivity

When one trait is much better powered, compare this optional estimator with the
default:

```python
res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n_eff1, n_eff2,
    rg_decorrelated=True,
)
```

The default sampled-quadratic ratio can attenuate the weak trait through its
posterior-noise-inflated variance. The alternative averages cross-sweep
quadratics while excluding same-sweep pairs. Thinning reduces, but does not
eliminate, dependence between retained MCMC states. Direct benchmark coverage is
limited, so treat the result as a sensitivity analysis rather than an automatic
replacement for `res.rg`. It needs at least two retained effect samples and a
full schedule; adaptive stopping is disabled. Undefined cross-sweep quadratics
raise an error rather than silently returning the default estimator.

## Sample overlap

Shared GWAS samples can correlate the two traits' sampling errors. With scalar
cohort sizes under a homogeneous standardized quantitative-trait sampling model,
the usual mapping is:

**Equation 2. Scalar-N overlap approximation.**

```text
cross_corr = N_shared rho_pheno / sqrt(N1 N2)
```

Here `N1` and `N2` are analyzed cohort sizes, and `rho_pheno` is the phenotypic
correlation among the shared analyzed individuals. With complete overlap and
equal cohort sizes, `cross_corr = rho_pheno`.

When the fitting inputs are effective sample sizes—for example, case-control
GWAS, meta-analyses, or SNP-varying `N`—Equation 2 is an approximation, not a
literal shared-person identity. One scalar `cross_corr` then represents an
assumed sampling-error correlation. Pass that value to the fit:

```python
res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n_eff1, n_eff2,
    cross_corr=cross_corr,
)
```

When overlap is unknown and scalar effective sample sizes are defensible, the
LDSC intercept can provide a sensitivity value:

```python
from bipred import estimate_sample_overlap, ldsc_rg

n1_scalar = 50_000.0
n2_scalar = 40_000.0
rgr = ldsc_rg(beta_hat1, beta_hat2, ld_scores, n1_scalar, n2_scalar)
estimate_sample_overlap(rgr, n1_scalar, n2_scalar, pheno_corr=0.4)
```

This inversion requires a non-zero assumed phenotypic correlation. The intercept
can also contain population structure, measurement effects, and other
confounding, so it does not identify overlap by itself. Under an overlap-only
model with nonnegative phenotypic correlation, `pheno_corr=1` yields the minimum
shared-sample count compatible with `0 < rho_pheno <= 1`, not an upper bound.

Environmental correlation among shared samples belongs in `cross_corr`, not in
genetic covariance. Small-panel intercepts are noisy; use them as diagnostics,
not precise overlap detectors.

With `noise_inflation=True`, bipred replaces each `N_t` by `N_t / lambda_t` in
both the diagonal variances and off-diagonal covariance while holding
`cross_corr` fixed. This assumes the added residual variance has the same
cross-trait correlation. If inflation represents trait-specific noise or
reference mismatch, use the combination as a sensitivity analysis rather than a
literal overlap correction.

## Regional genetic correlation

`regional_rg` restricts LD-aware quadratics of posterior-mean effects to groups
of variants without refitting:

```python
from bipred import regional_rg

local = regional_rg(
    res.beta1_est,
    res.beta2_est,
    blocks,
    region_labels,
    min_variants=50,
)
```

Labels may be strings or integers, need not be contiguous, and appear in
first-observed order. Regions may span LD blocks; within-block contributions are
summed under the block-diagonal LD assumption.

`regional_rg` evaluates the representation encoded by the LD objects passed to
it. A default fit can internally Q8-quantize float blocks of at most 1,500
variants without mutating the originals. Passing those originals therefore
evaluates float LD, not the fit's internal Q8 representation. To use the same
matrix, pass matching int8 blocks to both calls, or use the same float32 blocks
and set `ld_int8=False` during fitting.

**Table 2. `RegionalRgResult` fields.**

| Field | Meaning |
|---|---|
| `region` | labels in first-observed order |
| `rg` | regional genetic-correlation estimate |
| `gcov` | regional LD-aware genetic covariance |
| `gvar1`, `gvar2` | regional LD-aware genetic variances |
| `n_variants` | variants assigned to each region |

Two limitations dominate:

1. If sample overlap was not handled in the genome-wide fit, the same spurious
   covariance contaminates every region and does not average away.
2. The sampler has one genome-wide effect covariance, so local estimates are
   shrunk toward the genome-wide correlation.

Use regional estimates primarily for ranking and comparison, not calibrated
absolute effects or formal testing. Quantization can also make a dense LD block
non-positive-definite; inspect the raw variances and use `clip=False` to expose
out-of-range diagnostic values for the blocks passed to `regional_rg`. This does
not diagnose a different representation used internally by the fit. The research
evidence and its limitations are in the repository's
[`RESULTS_REGIONAL.md`](https://github.com/bvilhjal/bipred/blob/main/research/cross_corr_estimation/RESULTS_REGIONAL.md).

## Polygenic overlap

The four-state mixture gives:

**Equation 3. Polygenic-overlap decomposition.**

```text
pi1             = pi10 + pi11
pi2             = pi01 + pi11
rho_beta        = s12 / sqrt(s1 s2)
rg_from_overlap = rho_beta pi11 / sqrt(pi1 pi2)
```

Start with `frac_shared`, `rho_beta`, and `rg_from_overlap`. Absolute
`n_causal` and `n_shared` counts are approximate because LD can spread
inclusion mass and reference mismatch can inflate it.

For count-sensitive analyses:

- use `noise_inflation=True` when finite-reference mismatch is plausible;
- compare `res.mixer_calibrated(infer1, infer2)` using two univariate ldpred3
  fits; and
- report absolute counts as descriptive MiXeR-style summaries unless
  simulation validates them for the target architecture.

`mixer_iterate_summary()` reports empirical retained-iterate intervals, not
Bayesian credible intervals and not correction for reference mismatch.
