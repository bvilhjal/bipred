# Estimating the GWAS-overlap noise correlation (`cross_corr`) in the sampler

**Status: research prototype — not shipped.** This directory records a proof of
concept; it imports nothing from `bipred` and adds no public API. It exists to
answer one design question and to justify (or not) a future production feature.

## Question

bipred treats `cross_corr` — the cross-trait correlation of the *sampling noise*
induced by overlapping GWAS samples — as a **fixed user input** (supplied
directly, or via `estimate_sample_overlap` from the cross-trait LDSC intercept).
Can the bivariate Gibbs sampler instead **estimate it jointly** with the genetic
correlation `r_g`?

(Note: this is the *noise/overlap* correlation, not the environmental
correlation `r_e`. `cross_corr` reflects the phenotypic correlation among shared
samples — genetic + environmental combined; summary statistics cannot separate
the environmental part. See the discussion in the package docs.)

> **The authoritative results are in [`RESULTS.md`](RESULTS.md)** (genome-wide)
> and **[`RESULTS_REGIONAL.md`](RESULTS_REGIONAL.md)** (regional r_g). Both
> compare against bipred's *existing* practice — deriving `cross_corr` from the
> cross-trait LDSC intercept. Read those first.
>
> **Short version of what the benchmarks concluded.** Genome-wide, the in-sampler
> estimator wins only below m ≈ 50,000 variants; past that the LDSC intercept is
> equal or better, so at genome scale this feature adds nothing. The case for
> building it is **regional** r_g, where each region has 10²–10³ variants: there
> the LDSC intercept is not merely worse but actively harmful, and uncorrected
> sample overlap manufactures r_g ≈ 0.26 in regions whose true r_g is zero.
>
> **Caveat on the numbers in this file.** This prototype draws the causal effects
> **once** and redraws only the GWAS sampling noise across its 5 replicates, and
> reports `np.std` with `ddof=0`. Its ± figures therefore understate the true
> replicate-to-replicate variability, and any single-draw quantity should not be
> read as an estimate of a method's behaviour. The direction of its conclusion is
> confirmed by the properly-replicated benchmark in `RESULTS.md`; the dispersions
> here are not. It is retained as the readable derivation of the method.

## Answer: yes, and estimating it de-biases `r_g`

`estimate_cross_corr_prototype.py` simulates two traits with a known genetic
correlation (`r_g = 0.6`) and a known overlap noise correlation
(`cross_corr = 0.4`), then compares a sampler that fixes `cross_corr = 0`
(bipred today) with one that estimates it. Five replicates, committed to
[`results.csv`](results.csv):

| quantity | sampler | truth | mean ± sd |
|---|---|---:|---|
| `r_g` | `cross_corr` fixed at 0 (today) | 0.60 | **0.734 ± 0.015** |
| `r_g` | `cross_corr` estimated | 0.60 | **0.589 ± 0.026** |
| `cross_corr` | estimated | 0.40 | **0.385 ± 0.028** |
| `cross_corr` | control, no overlap | 0.00 | **−0.031 ± 0.073** |

Three findings:

1. **`cross_corr` is identifiable in the sampler** — recovered as 0.385 ± 0.028
   (truth 0.4); the no-overlap control gives −0.031 ± 0.073 (truth 0), so it
   does not invent overlap that isn't there.
2. **Estimating it removes the overlap bias in `r_g`** — fixing `cross_corr = 0`
   inflates `r_g` to 0.734 (a tight, systematic +0.13), and jointly estimating
   it pulls `r_g` back to 0.589 ≈ the true 0.60.
3. The cost is the expected bias–variance trade: slightly higher `r_g` variance
   (0.026 vs 0.015).

## Method (and the correction that matters)

Per SNP, the residualised marginal is `d_j = beta_j + e_j` with noise
`e_j ~ N(0, E)`, where `E` has **fixed** diagonals `1/N1, 1/N2` and the free
off-diagonal `cross_corr / sqrt(N1 N2)`.

The naive statistic — the per-SNP residual `d_j - beta_j` — is **biased** (an
early version gave `cross_corr ≈ -0.43` at truth 0). Two reasons: with
`r_g > 0` the joint per-SNP bivariate draw couples the two residuals, and the
GWAS noise is LD-structured (`R/N`), not per-SNP diagonal.

The correct sufficient statistic **whitens the marginal residual by the LD
Cholesky** `R = L Lᵀ`:

```
z_t = sqrt(N_t) · L⁻¹ (bhat_t − R beta_t)
```

`z` is then i.i.d. bivariate normal with correlation exactly `cross_corr`, so
`cross_corr` is drawn from an exact 1-D conditional — a grid over the
correlation likelihood that respects the **known** unit-variance diagonals.
This is the LDSC-intercept identification (overlap is LD-flat while genetic
covariance scales with LD score), recast as a Gibbs step.

## Scope and limitations of this prototype

- **Infinitesimal** bivariate model (every SNP Gaussian). The `cross_corr`
  update is orthogonal to bipred's four-state mixture, so it should carry over —
  but that is asserted here, not demonstrated on the mixture.
- **Dense** LD with a known Cholesky. The whitening needs `L⁻¹` (or `R⁻¹`).
- One `(N, r_g, cross_corr)` operating point; not a full recovery grid.

## What a production feature would require

1. **Whitening for every LD representation.** Dense is easy; bipred's default is
   **int8** and it also supports **low-rank** LD, which carry no Cholesky — the
   int8 path would dequantise, the low-rank path needs a different whitening
   (e.g. via the factor). This is the main integration design question.
2. **Opt-in flag** (e.g. `estimate_cross_corr=True`), default off, so the
   golden-test-guarded kernels and existing outputs are untouched when off.
3. The sweep-boundary `cross_corr` draw wired into the Numba/threading RNG
   framework, a new `BivariateResult` field (posterior mean + samples), input
   validation, and docs.
4. A validation benchmark on msprime LD **with the four-state mixture** (not the
   infinitesimal model used here).

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/estimate_cross_corr_prototype.py
```

Pure NumPy, deterministic (seeded), a few minutes single-core. Rewrites
`results.csv`.
