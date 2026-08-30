# Genetic correlation and overlap

This page covers estimator choice, sample overlap, regional exploration, and
polygenic-overlap interpretation. See [`guide.md`](guide.md) for fitting.

Real summary statistics require harmonisation and checks against the LD
reference. In one public-data analysis, an LD-mismatched lipid file made a joint
fit diverge while its `h2` and causal fraction looked ordinary. That failure
followed the file, not bivariate fitting in general. Inspect fit warnings and
compare the DENTIST-inspired
[`bipred.qc.ld_consistency_screen`](../bipred/qc.py) diagnostic described in the
[guide](guide.md); it is neither the complete published DENTIST procedure nor
proof that a retained variant is correct.

## Genome-wide estimators

**Table 1. Genetic-correlation estimators.**

| Estimator | Use | Main caveat |
|---|---|---|
| `res.rg` | default joint LD estimate | needs well-matched LD |
| `rg_decorrelated=True` | **sensitivity diagnostic only** — the default estimator measured more accurate in both power regimes (0.0086 vs 0.0108, 0.0174 vs 0.0242); incompatible with multichain and adaptive stopping |
| `bipred.ldsc_rg` | fast screen or independent check | unstable when marginal LDSC `h2` is near zero; one-step and unfiltered, so single large-effect loci carry full leverage |
| two univariate LDpred fits | additional diagnostic | often attenuated under power asymmetry |

These are reimplementations, not wrappers: agreement with the original
cross-trait LDSC and MiXeR software has not been validated, and the benchmark
record compares these estimators against simulated truth and each other only
(see the README's *Citing and prior work*).

Cross-trait LDSC fits the moment relation:

**Equation 1. Cross-trait LD Score regression.**

```text
z_tj = sqrt(N_tj) beta_hat_tj / sqrt(1 - beta_hat_tj^2)

E[z1_j z2_j] =
    intercept + sqrt(N_1j N_2j) rho_g LD_score_j / M
```

The first line is the exact signed conversion for effects returned by
`ldpred3.standardize_betas`; all standardized effects must have absolute value
below one. The simpler `z_tj ≈ sqrt(N_tj) beta_hat_tj` is only a weak-effect
approximation. The slope estimates genetic covariance under LDSC assumptions.
The intercept captures correlated sampling noise and cross-trait confounding.
`M` is the variant count defining the heritability and covariance; pass the full
count as `m_snps` when summary statistics are a subset of that map. `m_snps` and
`ld_scores` must describe the same variant map — `ldpred3.ld_scores(blocks)` sums
over exactly the blocks it is handed, so pairing subset-derived LD scores with a
full-map `m_snps` inflates both slopes.

`ldsc_rg` is one-step and applies no chi-square filter, so a few large-effect
variants keep near-full leverage on both the covariance and the marginal
heritabilities. Cap the **rows you pass to `ldsc_rg`**, not the variants you
fit: `bipred.ldsc.ldsc_chi2_mask` is that row filter. The same mask on
`ldpred3_auto_bivariate_blocks` deletes the slab's large effects. Drop
long-range-LD regions (MHC, APOE) before using LDSC as a screen where
individual loci could dominate. Keep `m_snps` at the full map count either way.

For `rg_se`, order rows by chromosome and position. The block jackknife deletes
contiguous ranges of the supplied row order; arbitrary order does not preserve
local LD dependence.

Use the joint fit by default and inspect LDSC as a cheap sensitivity check. The
committed benchmark record states its simulation assumptions, paired
realized-truth errors, failures, and runtime provenance; see the
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
eliminate, dependence between retained MCMC states. In the committed synthetic
sweep (`benchmarks/RESULTS.md`, Table 4), paired MAE was 0.0086 versus 0.0108
under symmetric power and 0.0174 versus 0.0242 under asymmetric power for the
default versus cross-sweep estimator. Treat it as a sensitivity analysis, not
an automatic replacement for `res.rg` — and not as a production estimator at
all. It needs at least two
retained effect samples and a full schedule; adaptive stopping is disabled.
Non-finite cross-sweep quadratics raise an error rather than silently returning
the default estimator. A merely *degenerate* denominator — a non-positive
cross-sweep genetic variance, which a sparse, weakly powered fit can produce —
warns and reports `rg` as `NaN`, leaving the rest of the fit usable.

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
overlap = estimate_sample_overlap(
    rgr, n1_scalar, n2_scalar, pheno_corr=0.4,
)
```

When `overlap["cross_corr_valid"]` is true, its `overlap_corr` is the raw
intercept to use as a `cross_corr` sensitivity value. It needs no assumption
about the phenotypic correlation. A free LDSC intercept can, however, fall
outside the joint fit's required open interval `(-1, 1)`; then
`cross_corr_valid` is false, a warning is raised, and the value must not be
passed unchanged. GLGC measured HDL and triglycerides in the same individuals;
the intercept there is **-0.352**, and the joint fit gives `rg` -0.90 with
`cross_corr=0` against **-0.52** with the correction. A rough external range
used by the historical benchmark is -0.5 to -0.6; it is context, not ground
truth. Neither fit warned — uncorrected overlap produces a converged but
misspecified fit, a different failure from sampler divergence.

The shared-*count* is a weaker claim, because the intercept identifies only the
product `N_shared * rho_pheno`. Splitting it needs `rho_pheno` from outside,
and the default `pheno_corr=1.0` is a placeholder, not a guess: for a
negatively correlated pair it has the wrong sign, the inversion has no
solution, and `n_shared` comes back `nan` with a warning rather than `0.0`.
Under the overlap-only model, supplying an external sensitivity value completes
the inversion: HDL x TG at `pheno_corr=-0.45` gives 72,454 shared samples, 78%
of each study, instead of the "no overlap" that a clipped zero used to imply.
That value is not identified by the intercept. Read the count only when
`overlap["physically_consistent"]` is true. A negative count or a count larger
than the smaller cohort is impossible under the assumed overlap-only model;
the function then warns and returns `nan` for `n_shared` and `overlap_frac`
while retaining `n_shared_raw` for diagnosis.

The intercept can also contain population structure, measurement effects, and
other confounding, so it does not identify overlap by itself. Under an
overlap-only model with nonnegative phenotypic correlation, `pheno_corr=1`
yields the minimum shared-sample count compatible with `0 < rho_pheno <= 1`,
not an upper bound.

Environmental correlation among shared samples belongs in `cross_corr`, not in
genetic covariance. Small-panel intercepts are noisy; use them as diagnostics,
not precise overlap detectors. Through 0.2.1 this regime looked far worse than
it is — the stress test measured joint-fit MAE up to 0.86 — but that was an
artifact of the fit quantizing its LD internally, and with the current
`ld_int8=False` default the same cells land between 0.0072 and 0.0242
(`benchmarks/RESULTS.md`, Table 11). Setting `cross_corr` from external evidence
still helps; it is no longer compensating for a defect.

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
it. The default fit preserves D8/D32 values (copying non-contiguous storage when
needed) but normalises other dense floats and floating low-rank factors to D32;
`regional_rg` makes the same dense and low-rank normalisation, so passing the
same logical blocks to both calls evaluates the same values. In-fit quantization
(`ld_int8=True` or `None`) evaluates a private Q8 copy for float blocks;
pre-quantize and pass those blocks to both calls instead.

**Table 2. `RegionalRgResult` fields.**

| Field | Meaning |
|---|---|
| `region` | labels in first-observed order |
| `rg` | regional genetic-correlation estimate |
| `gcov` | regional LD-aware genetic covariance |
| `gvar1`, `gvar2` | regional LD-aware genetic variances |
| `n_variants` | variants assigned to each region |

Two limitations dominate:

1. If sample overlap was not handled in the genome-wide fit, spurious
   covariance can affect every region and does not average away.
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
frac_shared     = pi11 / min(pi1, pi2)
rho_beta        = s12 / sqrt(s1 s2)
rg_from_overlap = rho_beta pi11 / sqrt(pi1 pi2)
```

`.mixer["rho_beta"]` is the ratio of posterior-mean Sigma entries;
`mixer_iterate_summary` reports the mean of the per-iterate ratios. They
differ by Jensen's inequality.

Start with `frac_shared`, `rho_beta`, and `rg_from_overlap`. Absolute
`n_causal` and `n_shared` counts are approximate because LD can spread
inclusion mass and reference mismatch can inflate it.

`frac_shared` is biased upward, but not monotonically with polygenicity: the
committed sweep measured mean bias **+0.051** at 1% causal variants, **+0.025**
at 3%, **+0.088** at 10%, and **+0.225** at 30%
(`benchmarks/RESULTS.md`, Table 8). Very sparse traits are not automatically
safe: the 1% cells contain only 50 causal variants per trait and show more count
inflation than the 3% cells. Read `frac_shared` beside fitted polygenicity and
the spread expected for the target architecture; treat it as descriptive
unless simulations matched to that architecture establish calibration.

For count-sensitive analyses, compare rather than presume:

- use `noise_inflation=True` when finite-reference mismatch is plausible;
- compare `res.mixer_calibrated(infer1, infer2)` using two univariate ldpred3
  fits; and
- report absolute counts as descriptive MiXeR-style summaries unless
  simulation validates them for the target architecture.

The committed benchmark found that both adjustments helped some power settings
and worsened others.

`mixer_iterate_summary()` reports empirical retained-iterate intervals, not
Bayesian credible intervals and not correction for reference mismatch.
