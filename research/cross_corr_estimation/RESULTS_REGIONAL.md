# Regional genetic correlation and overlap correction

**Status:** research evidence behind the public exploratory `regional_rg`
readout. The benchmark's joint `cross_corr` estimator is not shipped:
production fits require a user-supplied `cross_corr`, then `regional_rg` computes
posterior-mean regional quadratics. Code:
[`bench_regional_rg.py`](bench_regional_rg.py). Genome-wide prototype results:
[`RESULTS.md`](RESULTS.md).

## Why this is the case that matters

The genome-wide benchmark ends on a caveat: the cross-trait LDSC intercept
converges roughly as `1/sqrt(m)`, so at real GWAS scale the in-sampler estimator's
advantage narrows. **For regional r_g that argument runs backwards**, for two
reasons:

1. **A region is small by construction** — 10²–10³ variants. That is precisely
   where the LDSC intercept is unusable: `RESULTS.md` §3 measures its SD at
   **0.625** at m = 1,500. "Just estimate the intercept per region" is not an
   available option.
2. **Overlap cannot be estimated within a region**, so regional r_g needs a
   *genome-wide* `cross_corr` — exactly what the in-sampler estimator produces.
   And overlap does not cancel when comparing regions: it adds the **same**
   spurious covariance to **every** region, so it inflates them all at once and
   **confounds genuine regional heterogeneity**, which is the entire scientific
   signal.

## Design

Each LD block is one region. Regions get heterogeneous true r_g — 0.0 / 0.4 / 0.8
in equal thirds — and an **equal heritability share**, so a region's accuracy is
not confounded with its power. The GWAS pair has real sample overlap
(`cross_corr` = 0.4). All four arms of the genome-wide benchmark are run
(`naive` = 0, `ldsc` = intercept, `joint` = estimated, `oracle` = truth), which
differ **only** in the `cross_corr` value. N = 50,000, 10 replicates, effects and
noise redrawn every replicate, 900 sweeps (300 burn-in).

Every region is scored against its **realized** r_g for that replicate, never the
nominal label. Two estimators are reported:

- **`sampled`** — `mean(b1'Rb2) / sqrt(mean(b1'Rb1)·mean(b2'Rb2))`, the
  sampled-quadratic ratio matching bipred's genome-wide `rg`.
- **`postmean`** — the same ratio built from posterior-mean effects: shrunk, but
  free of the same-sweep noise inflation.

## 1. Main result (60 regions × 100 variants, N = 50,000)

[`bench_regional_main.csv`](bench_regional_main.csv), `postmean` estimator:

**Table 1. Regional estimates by overlap-correction arm.**

| arm | null regions (realized ≈ 0.003) | r_g = 0.4 regions (realized 0.393) | r_g = 0.8 regions (realized 0.796) | null/strong separation *d* |
|---|---:|---:|---:|---:|
| `naive` (cc = 0) | **0.264** | 0.548 | 0.819 | 5.45 |
| `ldsc` (intercept) | **0.515** | 0.709 | 0.893 | 2.99 |
| **`joint` (estimated)** | **0.037** | 0.380 | 0.725 | **6.32** |
| `oracle` (truth) | 0.062 | 0.400 | 0.738 | 6.32 |

**The headline: uncorrected sample overlap manufactures regional genetic
correlation where there is none.** Null regions — true r_g ≈ 0 — read **0.264**
under `naive`. In a real analysis that is a false "shared genetic architecture"
signal at *every* null locus in the genome.

With `cross_corr` estimated in the sampler, null regions read **0.037**, matching
the oracle (0.062) and restoring the heterogeneity structure. `joint` also gives
the **best null-vs-strong separation** (d = 6.32, equal to the oracle) — better
than `naive` (5.45) and far better than the LDSC-intercept route (2.99).

## 2. The LDSC-intercept route is worse than doing nothing here

Feeding the per-dataset cross-trait intercept into the sampler made regional
inference *worse* than leaving `cross_corr` at zero (null regions 0.515 vs 0.264;
separation 2.99 vs 5.45). At N = 50,000 the intercept is extremely noisy —
`RESULTS.md` §2 measures its SD at **0.513** at this N — and in these runs it
came out negative (means −0.50 to −0.62 across the three region sizes), which
*over*-corrects in the wrong direction and inflates every regional estimate.

The robust claim is **not** "the LDSC intercept is systematically negative" — with
an SD of ~0.5, individual replicate values vary widely and the sign is not a
stable property. The robust claim is that **a per-dataset intercept this noisy is
an actively harmful input** to regional inference, because the error is applied
identically to every region.

## 3. Region size (fixed total m; [`bench_regional_size.csv`](bench_regional_size.csv))

`postmean`, null-region estimate (realized ≈ 0) and separation *d*:

**Table 2. Regional estimates by region size.**

| region size | `naive` | `ldsc` | **`joint`** | `oracle` | *d*: naive / ldsc / **joint** / oracle |
|---:|---:|---:|---:|---:|---|
| 200 variants | 0.259 | 0.471 | **0.013** | 0.056 | 7.71 / 2.51 / **9.15** / 9.44 |
| 100 variants | 0.264 | 0.515 | **0.037** | 0.062 | 5.45 / 2.99 / **6.32** / 6.32 |
| 50 variants | 0.254 | 0.518 | **0.036** | 0.055 | 4.03 / 2.23 / **4.71** / 4.65 |

The contamination is **independent of region size** (`naive` ≈ 0.25–0.26
everywhere) because it comes from a genome-wide nuisance, not from regional
noise. `joint` tracks the oracle at every size. Absolute precision degrades as
regions shrink (separation 9.15 → 4.71), as expected from fewer variants.

## 4. Estimator choice: `postmean` beats `sampled` for regions

For `joint`, RMSE by region class:

**Table 3. Regional RMSE by quadratic estimator.**

| estimator | null | r_g = 0.4 | r_g = 0.8 |
|---|---:|---:|---:|
| `sampled` | 0.119 | 0.051 | 0.126 |
| **`postmean`** | **0.070** | 0.059 | **0.082** |

The sampled-quadratic ratio inflates the denominator with posterior noise, which
matters more per region than genome-wide because each region has fewer variants.
**Recommendation: use posterior-mean effects for regional r_g**, and keep the
sampled-quadratic ratio for the genome-wide estimate where bipred already uses it.

## 5. The shrinkage this benchmark was built to expose

The sampler carries a **single genome-wide effect covariance Σ**, so every
per-SNP draw is shrunk toward the genome-wide r_g (here ≈ 0.4, the mean of the
three region classes). That shrinkage is visible and is **not** removed by
correcting `cross_corr`:

- strong regions are attenuated: `joint` gives 0.725 against a realized 0.796
  (bias −0.071), and the oracle is no better (0.738, −0.058);
- null regions are pulled up slightly: 0.037 against 0.003.

So regional estimates are biased *toward the genome-wide mean* by construction.
This is a real limitation of reading regional r_g out of a genome-wide model, and
it is the main argument for a **per-region Σ** (or a hierarchical prior over
regions) if calibrated regional inference is pursued. The public readout exposes
the global-model quantity with this limitation documented. It is orthogonal to
`cross_corr`: correcting overlap fixes contamination, not shrinkage.

## 6. Limitations

1. **Shrinkage toward the genome-wide r_g** (§5) — the dominant remaining bias,
   unaddressed by this change.
2. **Regions are exactly LD blocks.** Real regional analyses use windows that do
   not align perfectly with LD boundaries; cross-region LD leakage is untested.
3. **One operating point** for the regional grid: `cross_corr` = 0.4, N = 50,000,
   h² = 0.5 per trait, equal heritability per region. No negative `cross_corr`,
   no heterogeneous per-region heritability.
4. **In-model, overlap-only, infinitesimal, dense LD** — as in `RESULTS.md`.
   Population stratification would also produce cross-region covariance and is
   not simulated.
5. **The `ldsc` arm's specific values are high-variance** (§2); the direction of
   harm replicates across all three region sizes, the magnitude should not be
   read as a stable property.
6. **10 replicates**, and the separation statistic *d* pools regions within a
   class, so it measures class separation, not per-region testing power. No
   formal null calibration (Type-I) is computed.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/bench_regional_rg.py all 10
```

Writes `bench_regional_{main,size}.csv`. About 20 minutes on 4 cores.
