# Genetic correlation and overlap

This page covers how to choose and interpret bipred's genetic-correlation and
polygenic-overlap outputs. For basic usage, start with [guide.md](guide.md).

## Estimators

**Table 1. Genetic-correlation estimators.**

| estimator | best use | main caveat |
|---|---|---|
| `res.rg` from `ldpred3_auto_bivariate[_blocks]` | default estimate; highest precision in benchmarks | needs dense LD blocks |
| `rg_decorrelated=True` | asymmetric-power pairs | less benchmark coverage than the default estimator |
| `bipred.ldsc_rg` | fast screen and independent cross-check | noisier; can diverge when marginal LDSC `h2` is near zero |
| two univariate LDpred runs | extra diagnostic | often attenuated, especially under power asymmetry |

The joint estimator uses the full LD likelihood. Cross-trait LDSC is a
method-of-moments regression:

**Equation 1. Cross-trait LD Score regression.**

```text
E[z1_j z2_j] = intercept + (sqrt(N1 N2) * rho_g / M) * LD_score_j
```

The intercept captures cross-trait sampling-noise correlation and correlated
confounding; the slope gives genetic covariance under the LDSC assumptions.

## Practical recommendation

Use the joint fit's `res.rg` unless you only need a quick screen. Add
`bipred.ldsc_rg` as a cheap sanity check, especially to inspect the cross-trait
intercept.

Use `rg_decorrelated=True` when a strong trait is being used to boost a much
weaker correlated trait:

```python
res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n1, n2,
    rg_decorrelated=True,
)
```

Benchmarks in `benchmarks/rg_architectures.py` and `benchmarks/rg_methods.py`
show the joint fit is roughly unbiased across tested architectures and is usually
about 1.5-2x more precise than cross-trait LDSC. The advantage is largest for
sparse or major-locus architectures, where LDSC can become unstable through the
`rho_g / sqrt(h2_1 h2_2)` ratio.

## Sample overlap

If GWAS samples overlap, their errors are correlated. Supply that correlation
with `cross_corr`:

```python
res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n1, n2,
    cross_corr=cross_corr,
)
```

When the overlap is known:

**Equation 2. Sampling-noise correlation from shared samples.**

```text
cross_corr = N_shared * rho_pheno / sqrt(N1 * N2)
```

For fully shared samples, this reduces to the phenotypic correlation among the
shared individuals.

When the overlap is unknown, approximate it from cross-trait LDSC:

```python
from bipred import ldsc_rg, estimate_sample_overlap

rgr = ldsc_rg(beta_hat1, beta_hat2, ld_scores, n1, n2)
estimate_sample_overlap(rgr, n1, n2, pheno_corr=0.4)
```

The intercept does not uniquely identify overlap: correlated population
structure, measurement effects, and other cross-trait confounding can contribute
too. The inversion also requires a non-zero assumed phenotypic correlation.
Under an overlap-only model with a positive intercept and phenotypic correlation,
`pheno_corr=1` returns the effective overlap and the minimum shared-sample count
compatible with `|rho_pheno| <= 1`; it is a lower bound, not an upper bound. On
small panels the intercept is noisy; treat it as an indicator of cross-trait
sampling correlation or confounding, not as a specific overlap detector.

`cross_corr=0` is fine for independent studies and first-pass analyses, but
overlap can bias `r_g` upward, especially near true `r_g = 0`.

## Environmental correlation

Shared samples can have correlated non-genetic residuals. In the bivariate model,
that correlation belongs in the sampling-noise term, not in genetic covariance.
Use the same `cross_corr` mechanism. Cross-trait LDSC handles it through the free
intercept.

`benchmarks/rg_env_overlap.py` is a stress test, not evidence of guaranteed
recovery. The committed run shows high variance in the bivariate estimator and
a failed corrected cell at true `r_g=0`, `r_e=0.6`; inspect every row and its
standard deviation before interpreting the correction.

## Polygenic overlap

`res.mixer` derives overlap quantities from the four-state mixture:

**Equation 3. Polygenic-overlap decomposition.**

```text
pi1 = pi10 + pi11
pi2 = pi01 + pi11
rho_beta = s12 / sqrt(s1 * s2)
rg_from_overlap = rho_beta * pi11 / sqrt(pi1 * pi2)
```

Read these first:

- `frac_shared`
- `rho_beta`
- `rg_from_overlap`

These ratios are more stable than absolute causal counts because many LD-related
biases affect numerator and denominator similarly.

## Absolute counts

`n_causal` and `n_shared` are approximate. The main issue is LD-spreading:
posterior inclusion mass can spread from a causal SNP to correlated neighbours.
Finite reference-panel LD can add a smaller mismatch component.

Current benchmarks with `p_init=0.02` show much better low-power calibration than
older settings with `p_init=0.1`, but count estimates are still weakly identified
when per-SNP power is very low.

For count-sensitive analyses:

- set `noise_inflation=True` to reduce reference-mismatch inflation,
- use `res.mixer_calibrated(infer1, infer2)` to anchor per-trait counts on
  univariate ldpred3 fits,
- use `res.mixer_iterate_summary()` for empirical intervals across retained `pi`
  draws and covariance iterates (not Bayesian credible intervals), and
- report that the absolute counts are MiXeR-style summaries, not a replacement
  for a dedicated causal-mixture likelihood.

## Reproducing benchmarks

Relevant scripts:

- `benchmarks/rg_architectures.py`: architecture sweep for LDpred vs LDSC.
- `benchmarks/rg_methods.py`: joint, LDSC, and univariate estimator comparison.
- `benchmarks/sample_overlap.py`: sample-overlap corrections.
- `benchmarks/rg_env_overlap.py`: environmental correlation on shared samples.
- `benchmarks/mixer_overlap.py`: overlap ratios and count calibration.
