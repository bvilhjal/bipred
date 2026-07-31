# Genetic correlation and overlap

This page covers estimator choice, sample overlap, regional exploration, and
polygenic-overlap interpretation. See [`guide.md`](guide.md) for fitting.

## Genome-wide estimators

**Table 1. Genetic-correlation estimators.**

| Estimator | Use | Main caveat |
|---|---|---|
| `res.rg` | default joint LD estimate | needs well-matched LD |
| `rg_decorrelated=True` | strongly asymmetric power | requires a full single-chain schedule |
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

Use the joint fit by default and inspect LDSC as a cheap sensitivity check. The
committed benchmark results are a dated snapshot, not current performance
evidence; see [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

## Asymmetric power

When one trait is much better powered:

```python
res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n_eff1, n_eff2,
    rg_decorrelated=True,
)
```

The default sampled-quadratic ratio can attenuate the weak trait through its
posterior-noise-inflated variance. The decorrelated estimator removes
same-sweep noise coupling and recovers covariance using posterior-mean
information. It needs a full retained schedule, so adaptive stopping is disabled.

## Sample overlap

Shared GWAS samples correlate the two traits' sampling errors. When overlap is
known:

**Equation 2. Sampling-noise correlation from shared samples.**

```text
cross_corr = N_shared rho_pheno / sqrt(N1 N2)
```

For fully shared samples this is the phenotypic correlation among the shared
individuals. Pass it to the fit:

```python
res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n_eff1, n_eff2,
    cross_corr=cross_corr,
)
```

When overlap is unknown, the LDSC intercept can provide a sensitivity value:

```python
from bipred import estimate_sample_overlap, ldsc_rg

rgr = ldsc_rg(beta_hat1, beta_hat2, ld_scores, n_eff1, n_eff2)
estimate_sample_overlap(rgr, n_eff1, n_eff2, pheno_corr=0.4)
```

This inversion requires a non-zero assumed phenotypic correlation. The intercept
can also contain population structure, measurement effects, and other
confounding, so it does not identify overlap by itself. Under an overlap-only
model with nonnegative phenotypic correlation, `pheno_corr=1` yields the minimum
shared-sample count compatible with `0 < rho_pheno <= 1`, not an upper bound.

Environmental correlation among shared samples belongs in `cross_corr`, not in
genetic covariance. Small-panel intercepts are noisy; use them as diagnostics,
not precise overlap detectors.

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
out-of-range diagnostic values. The research evidence and its limitations are in
the repository's
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
