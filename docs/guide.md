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

The default `ld_int8=False` consumes every dense block in the representation it
arrives in — int8 stays int8, float32 stays float32 — and copies nothing.
Quantize when the LD is *built*, with `ldpred3.compute_ld_blocks(quantize=True)`,
rather than in the fit: quantizing here allocates a second genome-scale payload
while your panel is still alive (62.1 MB of peak against 12.1 MB at m=100,000
with 500-variant blocks, from [`benchmarks/fit_memory.csv`](../benchmarks/fit_memory.csv)). `ld_int8=True` and `None` are retained for that older
behaviour. This option does not change low-rank factors.

**Table 1. Choosing a dense or low-rank representation.** Per-sweep cost at
m=20,000 with 500-variant blocks, one core, fitting a sparse causal model
(p ≈ 0.01, h² = 0.4), from
[`benchmarks/sweep_cost.py`](../benchmarks/sweep_cost.py).

| Representation | ms/sweep | Bytes per variant |
|---|---:|---|
| dense int8 (D8) | 0.82 | `k` = 500 |
| dense float32 (D32) | 0.83 | `4k` = 2,000 |
| low-rank float32, rank 481 | 2.49 | `4·rank` = 1,924 |
| low-rank int8, rank 481 (LR8) | 2.95 | `rank` = 481 |

Dense int8 and float32 sweep within a few per cent of each other, so **int8 is
a memory choice and costs nothing meaningful in time**. The dequantization sits in the
O(k) row update, which is guarded on a variant's effect changing and so fires on
roughly the causal fraction of visits — about 1% here. That guard is why this
table is sensitive to the fit: on a degenerate one, where nearly every variant is
called causal, the same comparison makes int8 look 1.5× slower. Timings taken
against unstructured noise will mislead you for that reason, and quantizing the
LD *inside* the fit does more than mislead — see `benchmarks/RESULTS.md` Table 11.

Low-rank cost scales with rank rather than block size, so it wins on large
blocks and loses on small ones. This benchmark uses a near-full rank (481 of a
500-variant block) to stress the projection dots, which is the *worst* case for
low-rank — at that rank it is both slower than dense and saves nothing over D8.
Use it when `rank ≪ k`.

## Quality control before fitting real data

**A bivariate fit tolerates far less summary-statistic error than a univariate
one does on the same panel.** This is the single most important practical
difference between bipred and ldpred3, it is not intuitive, and it is easy to
be caught by: data clean enough for LDpred is not automatically clean enough
for bipred.

The evidence is a real LDL × CAD analysis on a UK Biobank HapMap3 reference,
924,254 variants. On identical blocks and identical unfiltered summary
statistics, ldpred3's univariate sampler returned `sum(beta^2)` of 0.22 and a
largest effect of 0.082 — entirely healthy. The bivariate fit on the same input
returned `sum(beta^2)` of **157.5** with a largest effect of **3.33**, against a
per-causal effect SD of 0.030 that the fit had itself inferred. It reported
`h2` 0.64 and a causal fraction of 0.00075, both inside every bound, and
completed **without a warning** under 0.3.0.

Do this before fitting, in order:

1. **Harmonize strictly.** Match on rsID, require the reported allele pair to
   equal the reference pair as a set, flip the sign of `beta` when the effect
   allele is the reference's other allele, and drop indels and strand-ambiguous
   A/T and C/G pairs. Then *verify* it: correlate the aligned effect-allele
   frequency against the reference's own. Near +1 means aligned, near −1 means
   inverted. The flip *rate* cannot tell those apart — a legitimate
   effect-allele convention can produce any rate at all.
2. **Filter per variant.** Imputation quality (`INFO ≥ 0.9`), minor allele
   frequency (`≥ 0.01`), a chi-square cap (`≤ 80`), and per-variant sample size
   where a meta-analysis reports one. Exclude the MHC (chr6 25–34 Mb).
3. **Screen for LD consistency** with [`bipred.qc.dentist`](../bipred/qc.py).
   This is the step nothing else can substitute for: every filter in step 2
   judges a variant in isolation, and none can see that a variant's effect
   disagrees with the variants correlated with it.

On that LDL × CAD analysis, steps 1–2 removed 4.0% of variants and repaired one
trait while leaving the other untouched — CAD's cancellation ratio fell from
91.9 to 3.3, LDL's went from 246 to 264. Step 3 removed a further 4.7% and
repaired the fit outright:

**Table 2. The same fit at three levels of cleaning.**

| | harmonized only | + per-variant filters | + `qc.dentist` |
|---|---:|---:|---:|
| `rg` | +0.0675 | +0.1244 | **+0.2822** |
| `h2` trait 1 | 0.6395 | 0.5121 | **0.0861** |
| `sum(beta^2) / h2` | 246 | 264 | **0.7** |
| largest \|effect\| | 3.329 | 3.109 | **0.025** |
| genetic-variance trace | rising | rising | **settled** |
| cross-trait LDSC `rg` | +0.2238 | +0.1973 | +0.1864 ± 0.052 |

```python
from bipred.qc import dentist

keep = dentist(blocks, beta_hat1 / se1) & dentist(blocks, beta_hat2 / se2)
```

Run it once per trait against the blocks you will fit with, and intersect. Then
subset the blocks and both traits to `keep`.

Since 0.3.1 a fit that diverges this way raises a `RuntimeWarning` naming which
check failed. **Do not interpret `h2` or `rg` from a fit that warns.**

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

**Table 3. Main `BivariateResult` fields.**

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
can add inflation. `frac_shared` runs high by an amount that grows with
polygenicity — +0.03 at a 3% causal fraction, +0.23 at 30% — so read it beside
the fitted `p` rather than as a fixed offset; see [`rg.md`](rg.md). For
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

Pass the same blocks you fitted with. Under the default `ld_int8=False` the fit
evaluates them as given, so this is aligned with no further care. Only a fit
that opted into in-fit quantization (`ld_int8=True` or `None`) evaluates
something else, and that copy is private: pre-quantize the blocks yourself
(`ldpred3.compute_ld_blocks(..., quantize=True)`) and pass those to both calls.

`region_labels` has one label per variant; labels need not be contiguous.
Regional estimates use posterior-mean effects and expose `rg`, `gcov`, `gvar1`,
`gvar2`, and `n_variants`. They are best used for ranking or comparison:
uncorrected sample overlap contaminates every region, and the genome-wide model
shrinks local estimates toward the genome-wide correlation.

## Options

**Table 4. Main fitting options.** Unless the *Scope* column says otherwise, an
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
| `h2_cap` | `None` | both | optional expert ceiling on the per-trait slab variance, enforced *inside* the sampler. Unlike `h2_bounds` it changes the fitted effects, so it moves `rg` and `h2` alike whenever it binds |
| `iw_df` | `10` | both | covariance shrinkage strength |
| `sample_every` | `5` | both | thinning for the effect states the decorrelated `rg` uses; no effect otherwise |
| `ld_int8` | `False` | both | dense storage policy. The default copies nothing; `True`/`None` quantize inside the fit and build a second payload |
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
  are cancelling through LD, or exceed the slab the fit itself inferred, or the
  genetic variance never settled. Do not interpret `h2` or `rg` from it. The
  usual cause is summary statistics inconsistent with the LD reference, not the
  reference — screen them with `bipred.qc.dentist` before anything else.
- `ldpred3.shrink_ld_blocks` keys its shrinkage on `k / n_ref`, which assumes
  the distortion is finite-panel noise. Against a reference whose correlations
  were thresholded to zero outside a window, the distortion is structural and
  does not fall with `n_ref`, so the default intensity under-shrinks.
