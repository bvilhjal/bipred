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

With the default `ld_int8=False`, contiguous D8 and D32 blocks are reused.
Non-contiguous D8/D32 inputs are copied once to contiguous storage; other dense
float inputs are converted once to contiguous D32.
Quantize while building the LD, with
`ldpred3.compute_ld_blocks(quantize=True)`, rather than inside the fit:
`ld_int8=True` and `None` create a private second payload for float blocks and
are retained only for compatibility. D8 uses one quarter of D32's storage. Low-rank storage is
useful when its rank is much smaller than block size; otherwise D8 is usually
smaller. Treat representation timings as machine- and architecture-specific;
the scripts and their limitations are in the
[benchmark record](https://github.com/bvilhjal/bipred/blob/main/benchmarks/RESULTS.md).

## Quality control before fitting real data

Real summary statistics can disagree with the fitted LD reference in ways that
model-matched simulations cannot reproduce. The result may be a divergent fit
whose `h2` and causal fraction still look ordinary. Use the checks below before
interpreting a real-data fit, and always heed the fit's own warnings.

### The recommended procedure

1. **Harmonize, then verify.** Match on rsID, require the reported allele pair
   to equal the reference pair as a set, flip the sign of `beta` when the effect
   allele is the reference's other allele, and drop indels and strand-ambiguous
   A/T and C/G pairs. Then check it: correlate the aligned effect-allele
   frequency against the reference's own. Near +1 is aligned, near −1 inverted.
   The flip *rate* cannot distinguish those — a legitimate effect-allele
   convention produces any rate at all (one real file flipped 69.7%).
2. **Check the effective sample size.** `n_eff` scales every standardized effect.
   For case-control data,
   [`bipred.qc.implied_sample_size`](../bipred/qc.py) can compare the reported
   value with the genotype scale implied by `beta`, `se`, and allele frequency.
   It found 92,966 versus 162,973 for the tested CAD file. For quantitative
   traits, absolute N and phenotype scale are not separately identifiable from
   these columns; the function reports N as unidentified rather than calibrating
   itself to the reported value.
3. **Filter per variant using study-appropriate thresholds.** Minor allele frequency
   ≥ 0.01, allele-frequency concordance, imputation quality where the file
   carries it, per-variant sample size, and
   [`bipred.qc.sd_consistency`](../bipred/qc.py) are useful inputs **for the
   joint-fit panel**. A chi-square cap is not one of those: it is an LDSC-row
   filter (see below and [`ldsc_rg`](rg.md)). Thresholds must reflect the
   study and reference; the committed factorial does not establish a universal
   optimum.
4. **Compare an LD-consistency screen** with
   [`bipred.qc.ld_consistency_screen`](../bipred/qc.py). This lightweight,
   block-based routine is inspired by the DENTIST statistic but does not
   implement the published DENTIST windowing and protected-removal procedure.

```python
from ldpred3 import standardize_betas
from bipred.qc import implied_sample_size, ld_consistency_screen

sized = implied_sample_size(beta2, se2, ref_af, binary=True, reported_n=n2)
if not sized["consistent"]:
    n2 = n2 * sized["ratio"]          # fit what the data behaves like

# Standardized effects depend on N: recompute them after any N adjustment.
beta_hat1 = standardize_betas(beta1, se1, n1)[0]
beta_hat2 = standardize_betas(beta2, se2, n2)[0]

z1, z2 = beta1 / se1, beta2 / se2     # original GWAS columns
keep = (ld_consistency_screen(blocks, z1)
        & ld_consistency_screen(blocks, z2))
```

Use raw GWAS z-scores here, not standardized `beta_hat / se`. After intersecting
the masks, subset and reindex every LD block, both standardized effects, and any
per-variant sample-size arrays before calling the fit. Recompute standardized
effects whenever N changes. With the default fit policy, D8, D32, and low-rank
blocks match the screen numerically; other dense floats are normalised to D32 in
both. The screen cannot replay the private D8 copy made by legacy in-fit
quantisation, so pre-quantise when exact alignment matters. `dentist` remains a
compatibility alias for this routine; it does not turn the approximation into
the full DENTIST method.

A chi-square cap is a leverage filter for [`ldsc_rg`](rg.md), not a joint-fit
mask. The reference exclusion is `chi2 > max(0.001 N, 80)`.
`bipred.ldsc.ldsc_chi2_mask` returns that row filter. Subset the arguments to
`ldsc_rg` and leave the joint fit its full variant set; keep `m_snps` at the
full count. The same mask on `ldpred3_auto_bivariate_blocks` deletes the slab's
large effects — that is the 0.3.7 failure mode.

Blocks are independent, so `ncores` settles several at once — 2.49× on four
cores at 16 × k=2,000, with the mask identical at every core count. The pool
nests over BLAS, so it is taken only when BLAS is pinned to one thread and
`threadpoolctl` confirms the loaded library is reentrant; the concurrent call
is `np.linalg.eigh`, which is exactly the routine that miscomputes under a
non-reentrant BLAS, so it never nests on an environment-variable guess:

```bash
OMP_NUM_THREADS=1 python your_screen.py     # then ld_consistency_screen(..., ncores=4)
```

Note that ldpred3 ships a *different* LD-consistency filter,
`ldpred3.qc.dentist_outlier_mask`. It inverts a whole block and removes the
single worst variant per pass; this one predicts random half-windows from each
other, which is closer to the split-sample procedure Chen et al. published. They
are not interchangeable, and bipred's committed factorial evidence was generated
with the screen documented here.

### What the factorial established

Three related trait pairs spanning the sign range — LDL × CAD (+0.26), height ×
LDL (≈ 0), and HDL × TG (−0.53) — were each fitted under all eight
combinations of stricter per-variant thresholds, long-range-LD exclusion, and
the screen. Every pair contains at least one GLGC lipid file, so these 24 arms
are repeated perturbations of three file combinations, not independent
validation across 24 settings. The saved rows come from a clean 0.3.5 run in
which every requested random partition completed. That makes them current for
these files and this reference, not a general validation of the screen.

**Table 1. Divergence warnings across 24 current-screen arms.**

| factor | off | on |
|---|---:|---:|
| strict per-variant thresholds | 6/12 | 6/12 |
| long-range LD exclusion | 6/12 | 6/12 |
| **LD-consistency screen** | **12/12 diverged** | **0/12 diverged** |

In this run, the screen separated the warnings in these file/reference
combinations. The other factors did not change the warning count, but they did
change estimates; Table 1 cannot establish that they "do nothing." Among its
screened fits,
long-range exclusion moved `rg` by about 0.012 for height × LDL, 0.021–0.023
for HDL × TG, and 0.0001–0.0067 for LDL × CAD. Use
[`bipred.qc.in_long_range_ld`](../bipred/qc.py) as an estimator-specific
sensitivity analysis. Exclusion may protect genome-wide moments, while retaining
APOE may matter for prediction; the appropriate choice depends on the target.

### Why diagnostics matter

A diverged fit can still look plausible. On HDL × TG, all four screened
estimates and only one of four unscreened estimates lay in a rough external
range of −0.5 to −0.6 used by the historical study. That uncited context is not
ground truth, and agreement with any external point or interval cannot by
itself certify a fit.

Nor is the failure uniform. On LDL × CAD divergence halved `rg`; on height ×
LDL it shrank it toward zero; on HDL × TG it inflated it. And it can strike one
trait while sparing the other in the same fit — height × LDL diverged at
cancellation 150–212 on the LDL side while height remained in the rough
external range 0.3–0.5, with `h2` 0.41 against rough context around 0.45.

Within this study, warning status tracked the *summary-statistic file*: all
three GLGC lipid files diverged in every pairing, while height and CAD did not.
Three related pairs do not establish that pattern generally.

Since 0.3.1 a fit that trips a divergence diagnostic raises a `RuntimeWarning`
naming the check. Do not interpret `h2`, `rg`, or the overlap readouts until the
data/LD mismatch has been investigated; passing the diagnostic is necessary
evidence, not proof of correctness.

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

**Table 2. Main `BivariateResult` fields.**

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
can add inflation. In the committed sweep, mean `frac_shared` bias was
nonmonotonic: +0.05, +0.03, +0.09, and +0.23 at causal fractions 1%, 3%, 10%,
and 30%. Very sparse traits are not exempt. Read the estimate beside fitted `p`
rather than applying a fixed offset; see [`rg.md`](rg.md). For
count-sensitive work, compare `noise_inflation=True` and
`res.mixer_calibrated(infer1, infer2)` with the unadjusted result; neither is
guaranteed to improve calibration at every power setting.

## Genetic correlation

Use `res.rg` by default. `rg_decorrelated=True` is a **sensitivity diagnostic
only — do not use it for production estimates**: it had higher paired error
than the default in both the symmetric and asymmetric synthetic sweeps (0.0086
versus 0.0108, 0.0174 versus 0.0242), and it is incompatible with multichain
pooling and adaptive stopping. `ldsc_rg` is a fast independent screen.
Interpretation, sample overlap, and overlap-interval semantics are covered in
[`rg.md`](rg.md).

For per-region exploratory estimates:

```python
from bipred import regional_rg

local = regional_rg(
    res.beta1_est, res.beta2_est, blocks, region_labels,
    min_variants=50,
)
```

Pass the same effective LD representation you fitted with. Under the default,
D8 and D32 retain their numeric values (a non-contiguous input may be copied);
other dense float inputs are normalized to D32. Exact alignment therefore
requires passing D8/D32, or the same normalized D32, to both calls. In-fit
quantization (`ld_int8=True` or `None`) creates a private copy for float blocks:
prefer pre-quantizing with `ldpred3.compute_ld_blocks(..., quantize=True)` and
passing those blocks to both.

`region_labels` has one label per variant; labels need not be contiguous.
Regional estimates use posterior-mean effects and expose `rg`, `gcov`, `gvar1`,
`gvar2`, and `n_variants`. They are best used for ranking or comparison:
uncorrected sample overlap contaminates every region, and the genome-wide model
shrinks local estimates toward the genome-wide correlation.

## Options

**Table 3. Main fitting options.** Unless the *Scope* column says otherwise, an
option is accepted by both `ldpred3_auto_bivariate[_blocks]` and
`ldpred3_auto_bivariate_chains`. Options marked *single* are rejected by the
chains driver, which reserves them for its own dispersal.

| Option | Default | Scope | Use |
|---|---:|---|---|
| `burn_in`, `num_iter` | `200`, `200` | both | burn-in and retained sweeps; chains additionally requires `num_iter` even and ≥ 4 |
| `h2_init`, `rg_init` | `0.1`, `0` | both | coherent genetic-moment start |
| `p_init` | `0.02` | single | union-causal start; chains disperses it over `p_init_range` |
| `pi_init` | `None` | single | explicit four-state overlap start; the chains analogue is `pi_inits`, one row per chain |
| `sigma_prior_scale` | `None` | both | fixed covariance shrinkage target across starts |
| `cross_corr` | `0` | both | correlation of cross-trait sampling noise (set from external evidence when traits share environmental effects — see `docs/rg.md`) |
| `rg_decorrelated` | `False` | single | sensitivity diagnostic only; the default estimator measured more accurate in both power regimes; chains rejects it |
| `noise_inflation`, `ni_damp` | `False`, `0.1` | both | learn and damp residual-noise inflation |
| `pi_prior` | `1` | both | symmetric Dirichlet mixture concentration |
| `h2_bounds` | `(1e-4, 1)` | both | clamp on the **reported** `h2` only. `rg` is a ratio of the raw quadratics, so it is invariant to this |
| `h2_cap` | `None` | both | optional expert ceiling on implied per-trait heritability, enforced *inside* the sampler as `s_t ≤ h2_cap_t / n_causal,t`. Unlike `h2_bounds` it changes the fitted effects, so it moves `rg` and `h2` alike whenever it binds |
| `iw_df` | `10` | both | covariance shrinkage strength |
| `sample_every` | `5` | both | thinning for the effect states the decorrelated `rg` uses; no effect otherwise |
| `ld_int8` | `False` | both | dense storage policy. Contiguous D8/D32 are reused; non-contiguous blocks are copied and other float types normalized to D32. `True`/`None` quantize float blocks inside the fit and build a second payload |
| `ncores` | `1` | both | within-chain block threads |
| `chain_ncores` | `1` | chains | multi-chain concurrency; cannot combine with `ncores>1` |
| `tol` | `0` | single | optional stabilization heuristic; chains rejects `tol>0` |
| `check_every` | `50` | both | retained sweeps between stabilization checks; accepted by chains but inert there, since adaptive stopping is disabled |
| `seed` | `None` (single) / `0` (chains) | both | random seed. The chains driver requires an integer and rejects `None` |

With `tol>0`, a single chain checks the relative RMS change in both
posterior-mean effect vectors and the change in `r_g` every `check_every`
retained sweeps. Meeting the threshold can stop the run early. This is a
schedule- and seed-dependent stabilization heuristic, not evidence that the
Markov chain converged. It is disabled for `rg_decorrelated=True` and unsupported
by multi-chain inference; use dispersed full-length chains for diagnostics.

## Pitfalls

- Match ancestry and harmonize variant order, alleles, and effect scale before
  fitting.
- A chi-square cap is for `ldsc_rg` rows only. Applying it to the joint-fit
  panel deletes the large effects the slab is there to hold.
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
- A warning that the fit *appears to have diverged* is stronger: the effects
  are cancelling through LD, exceed the inferred slab, or the genetic variance
  never settled. Do not interpret `h2` or `rg` from it. Recheck harmonization,
  scaling, sample size, and reference compatibility, then compare
  `bipred.qc.ld_consistency_screen`; passing that approximate screen is not a
  certificate of correctness.
- `ldpred3.shrink_ld_blocks` keys its shrinkage on `k / n_ref`, which assumes
  the distortion is finite-panel noise. Against a reference whose correlations
  were thresholded to zero outside a window, the distortion is structural and
  does not fall with `n_ref`, so the default intensity under-shrinks.
