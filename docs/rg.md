# bipred — genetic correlation and polygenic overlap

`ldpred3_auto_bivariate` reports a genetic correlation `r_g` between two traits
and, from the same four-state fit, a MiXeR-style polygenic-overlap summary. This
is the usage-and-guidance companion to the model reference in
[algorithm.md](algorithm.md#bivariate-two-trait-ldpred3): how accurate `r_g` is,
which estimator to use, how to handle overlapping samples, and how to read the
polygenic-overlap output. bipred owns **all** genetic-correlation estimation:
besides the joint fit it provides cross-trait LD Score regression
(`bipred.ldsc_rg`, with `LDSCRgResult` and `estimate_sample_overlap`) as the
independent moment-based cross-check, used throughout below. It builds on
ldpred3's *univariate* LDSC (`ldpred3.ld_scores` for the LD scores,
`ldpred3.ldsc_h2` for the marginal heritabilities).

## Genetic correlation vs bivariate LDSC

The `r_g` from `ldpred3_auto_bivariate` has an independent external check —
**cross-trait LD Score regression**, `bipred.ldsc_rg`
(`E[z₁z₂] = intercept + (√(N₁N₂)·ρ_g/M)·ℓ`). Under realistic reference-panel LD
both are roughly unbiased and the bivariate sampler is ~2× more precise (at true
r_g=0.8, LDSC 0.80 ± 0.031 vs bivariate LDpred3 0.79 ± 0.016) — and, as below,
numerically more robust.

**Across architectures.** Sweeping true r_g over five architectures (shared causal
variants with bivariate-normal effects, both traits h²=0.5). The LD here is
**realistic and non-repeating** — the genome is 25 *distinct* coalescent segments
(m=5000), not a tiled block library — fit from a finite reference panel
(N₁=50k/N₂=20k, 10 reps; `benchmarks/rg_architectures.py`,
`rg_architectures.png`). Representative r_g = 0.0 / 0.4 / 0.8:

| architecture | bivariate LDSC | bivariate LDpred3 |
|--------------|:--------------:|:-----------------:|
| infinitesimal | 0.06 ± 0.08 / 0.39 ± 0.11 / 0.80 ± 0.03 | 0.01 ± 0.03 / 0.40 ± 0.04 / 0.79 ± 0.02 |
| sparse (p=0.01) | 0.02 ± 0.22 / 0.37 ± 0.26† / 0.65 ± 0.17 | 0.04 ± 0.08 / 0.41 ± 0.20 / 0.80 ± 0.03 |
| moderate (p=0.05) | 0.02 ± 0.13 / 0.37 ± 0.08 / 0.77 ± 0.04 | 0.04 ± 0.08 / 0.38 ± 0.05 / 0.78 ± 0.03 |
| polygenic (p=0.2) | 0.02 ± 0.08 / 0.40 ± 0.08 / 0.79 ± 0.04 | 0.00 ± 0.05 / 0.40 ± 0.04 / 0.80 ± 0.02 |
| major locus | 0.14 ± 0.38† / 0.29 ± 0.24 / 0.76 ± 0.07 | 0.07 ± 0.16 / 0.33 ± 0.23 / 0.77 ± 0.04 |

(± is the across-replicate SD over the in-range reps; † marks cells where one of
the ten LDSC reps diverged and was excluded.)

- **r_g is architecture-robust** and **unbiased at r_g=0** (no spurious
  correlation) for both methods — unlike `h²` and `p`, the genetic *correlation*
  largely cancels the LD-mismatch and architecture effects (they hit numerator and
  denominator alike). Both track the full 0 → 0.95 sweep closely.
- **Bivariate LDpred3 is consistently more precise** (full likelihood vs the
  moment regression) — typically ~1.5–2× smaller SD, most visibly on the sparse
  and major-locus traits where the cross-trait signal is carried by few variants.
- **LDpred3 is also numerically robust.** LDSC's `r_g = ρ_g / √(h²₁h²₂)` divides
  by marginal heritabilities, so when a univariate `h²` estimate lands near zero
  (its noisy regime on sparse / major-locus traits) the ratio **blows up** — a
  handful of reps here returned values in the hundreds and had to be excluded
  (the † cells). The bivariate sampler, which models the joint effect covariance
  directly, **never diverged**: on major-locus at r_g=0.6 it gives 0.59 ± 0.11
  where LDSC's surviving reps scatter 0.48 ± 0.20.
- **Running time & memory.** Per fit on this genome (m=5000, 25 blocks, single
  core): bivariate **LDSC ~25 ms**, bivariate **LDpred3 ~0.23 s** at a **0.27 GB**
  peak RSS. So the sampler's precision and robustness cost ~9× LDSC's time but
  stay sub-second and light — LDSC remains the instant screen, LDpred3 the
  accurate estimate.

For a quick first pass, a **marginal** (no-LD) r_g — the moment estimator that
assumes independent SNPs — is already reasonable (unlike a marginal h², which is
useless), because LD inflates the cross-covariance and both heritabilities
*proportionally* and largely cancels in the ratio. Use it as a sanity check,
`bipred.ldsc_rg` as a fast LD-correct screen with a confounding intercept, and
the bivariate sampler when precision matters.

## Choosing an r_g estimator

Besides LDSC and the bivariate joint fit, you can read `r_g` off **two
independent univariate `-auto` runs** — the self-normalized genetic correlation of
the posterior-mean effects (`uni_gv`), optionally with a decorrelated
out-of-sample-`r²` denominator (`uni_r2`). `benchmarks/rg_methods.py` compares all
four for accuracy **and running time** (m up to 50k): the **bivariate joint fit is
the most accurate and, per fit, ~5× cheaper than the univariate pair** (which must
run several chains per trait), and it is uniquely robust to **power asymmetry** —
where a strong trait boosts a weak one, the univariate estimators attenuate
(e.g. 0.78 vs true 0.90) while the joint fit stays unbiased. `uni_gv` and `uni_r2`
come out numerically identical, and LDSC — being moment-based — does not attenuate
under asymmetry but is noisier at low `r_g`.

| estimator | when to reach for it |
|---|---|
| bivariate joint fit (default `"gv"`) | recommended — most accurate, ~5× cheaper per fit than the univariate pair, robust to power asymmetry |
| bivariate joint fit, `rg_decorrelated=True` | asymmetric-power pairs (strong trait boosting a weak one) — recovers the weak trait's covariance the same-sweep ratio attenuates |
| `bipred.ldsc_rg` (cross-trait LDSC) | instant moment-based screen; no shrinkage attenuation, but noisier at low `r_g` |
| univariate `uni_gv` / `uni_r2` | independent cross-check; attenuates under power asymmetry |

The theory (why each works, the scale-matching principle, and why "calibrating"
the denominator with `h²` *fails*) is in
[algorithm.md](algorithm.md#the-genetic-correlation-estimators-theory--trade-offs).

## Polygenic overlap (MiXeR-style)

The bivariate fit's four-state mixture also yields the quantities MiXeR reports —
the shared-causal fraction, the within-shared effect correlation `ρ_β` and the
decomposition `r_g = ρ_β·π₁₁/√(π₁π₂)` — exposed as `res.mixer` (see
[algorithm.md](algorithm.md#bivariate-two-trait-ldpred3)). The overlap **ratios**
(`frac_shared`, `ρ_β`, `rg_from_overlap`) are reliable; the **absolute counts**
(`n_causal`, `n_shared`) are over-estimated on realistic LD and should be read as
relative unless calibrated (`benchmarks/mixer_overlap.py`, `calibration` sweep).

### Absolute counts at low power

At the **low per-SNP power `N·h²/M < 1` typical of real GWAS** (e.g. `h²=0.3`,
`N=500k`, `M~1M` → ~0.15) the absolute polygenic counts are **over-estimated**,
by ~3× at `p=0.10` and up to ~6× at a sparser `p=0.03`. Benchmarked against a
**known** truth (`benchmarks/mixer_overlap.py`), the count/true ratio at `p=0.10`:

| `N·h²/M` | 0.1 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|
| matched LD | 3.3 | 1.9 | 1.1 | 1.0 | 1.1 |
| ref-panel | 3.1 | 1.9 | 1.2 | 1.2 | 1.3 |

The mechanism, dissected against truth (univariate `ldpred3_auto_infer` and the
exact no-LD posterior; see ldpred3's
[`docs/inference.md`](https://github.com/bvilhjal/ldpred3/blob/master/docs/inference.md)):

- **The dominant cause with real LD is LD-spreading, not the prior or the
  point-estimate.** Even under *matched, in-sample* LD the count is inflated at
  the posterior **mode itself** (mean ≈ median ≈ mode — the posterior is *tight*
  around the wrong value), because correlated SNPs are recruited as causal around
  each true causal. This is a genuine likelihood-level effect, and no prior or
  summary choice removes it.
- **The bivariate over-counts *more* than the univariate** (correcting an earlier
  note): a SNP correlated with signal in *either* trait can enter states `10`,
  `01` or `11`, so the four-state model amplifies the spreading (~3× bivariate vs
  ~1.4–2× univariate at `p=0.10`, `N·h²/M=0.1`). So `mixer_calibrated`'s
  univariate anchoring does help somewhat (the univariate anchor is *less*
  inflated), but it does not fully debias.
- **The prior is a minor lever here.** `pi_prior` sets the symmetric Dirichlet
  concentration (default `1.0`, uniform; `0.5` is Jeffreys). Jeffreys trims only
  ~5% off the count with real LD (`r_g` unchanged) — in the *no-LD* limit the
  prior and the mean-vs-median skew are the whole story and Jeffreys removes them
  (that is where ldpred3's univariate `p_prior=(0.5,0.5)` matters), but realistic
  LD-spreading swamps them.
- **Per-causal power `λ_c = N·h²/(M·p)` governs identifiability but the bias is
  genuinely 2-D** in `(N·h²/M, p)` — worst where `λ_c ≈ 1` *and* the true `p` is
  far below the prior mean — so a *sparse* trait (`λ_c > 1` even at `N·h²/M < 1`)
  is better determined, but no single `λ_c` threshold certifies safety.
- **What actually helps.** `noise_inflation=True` damps the count (~3.4×→2.5× at
  `N·h²/M=0.1`, most of it the reference-mismatch part), with `h²`/`r_g`
  unchanged, and improves the posterior coverage below;
  `res.mixer_posterior()` gives the **credible interval**; LD shrinkage / QC and
  adequate power shrink the spreading. Otherwise **read the absolute counts as
  relative**.
- **The `r_g` and the overlap ratios cancel the bias entirely** and stay reliable
  across the whole power range (`r_g ≈ 0.3–0.4` for true 0.4, even at
  `N·h²/M = 0.1`).

`res.mixer_calibrated(infer1, infer2)` rebuilds the counts on two univariate
`ldpred3.ldpred3_auto_infer` polygenicities while keeping the joint ratios; since
the univariate anchor is less LD-spread-inflated than the joint fit, it *reduces*
(without eliminating) the over-count — for absolute counts prefer
`noise_inflation`, `pi_prior=0.5`, adequate power and good LD (shrinkage / QC),
and lean on the ratios otherwise.

## Handling sample overlap

When the two GWAS share individuals, their sampling noise is correlated (`ρ_e`),
which inflates a naive genetic-correlation estimate even for genetically
uncorrelated traits. Both methods have a correction — a free cross-trait
**intercept** in LDSC, and the **`cross_corr`** parameter in the bivariate sampler
— and both work (sparse trait, N₁=N₂=15k fully overlapping, 8 reps;
`benchmarks/sample_overlap.py`). At true r_g=0, mean estimate:

| ρ_e | LDSC (intercept=0) | LDSC (free intercept) | LDpred3 (cross_corr=0) | LDpred3 (cross_corr=ρ_e) |
|----:|:------------------:|:---------------------:|:----------------------:|:------------------------:|
| 0.5 | 0.078 | 0.027 | 0.056 | 0.042 |
| 0.9 | 0.105 | 0.026 | 0.065 | 0.029 |

The uncorrected columns drift upward with the overlap; supplying the correction
(free intercept / `cross_corr=ρ_e`) brings the spurious correlation back to ~0.
The residual inflation is modest here because the LD-aware estimators load the
overlap onto a term (the intercept / the noise-covariance) that is largely
separable from the genetic signal — but it grows with ρ_e, so pass the known
overlap when you have it.

A second benchmark (`benchmarks/sample_overlap.py`, noise correlation ρ_e=0.5,
N=10000, h²=0.3) confirms the correction across `r_g`:

| true r_g | LDSC, intercept=0 | LDSC, free intercept | bivariate, cross_corr=0 | bivariate, cross_corr=ρ_e |
|---------:|------------------:|---------------------:|------------------------:|--------------------------:|
| 0.0 | 0.101 | 0.051 | 0.026 | −0.025 |
| 0.5 | 0.589 | 0.521 | 0.531 | 0.487 |

Uncorrected, overlap biases `r_g` upward (≈+0.03–0.10 at r_g=0); the free LDSC
intercept and the bivariate `cross_corr=ρ_e` each remove it (both corrected
columns straddle zero). If your two GWAS share samples, leave the LDSC intercept
free and pass `cross_corr` (the overlap-induced noise correlation, ≈ the
cross-trait LDSC intercept) to the bivariate sampler.

### Environmental correlation, mechanistically

The sharper worry is two traits whose *environments* are correlated (`re`)
measured on the **same** people: the shared environment makes the phenotypes
correlate with no genetic basis, which a naive estimate would read as genetic.
`benchmarks/rg_env_overlap.py` builds this from real individual-level
genotypes/phenotypes — both GWAS on the same N=20k individuals, genetic effects
correlated by `rg` and residual environments by `re` — and shows the **genetic**
rg is recovered regardless of `re` (20 reps):

| true rg | re | LDSC free intercept | LDSC intercept=0 | LDpred3 cc=0 | LDpred3 cc=ρ |
|--------:|---:|--------------------:|-----------------:|-------------:|-------------:|
| 0.0 | 0.0 | −0.03 | −0.02 | −0.02 | −0.02 |
| 0.0 | 0.6 | −0.02 | +0.02 | +0.02 | −0.00 |
| 0.5 | 0.0 | 0.49 | 0.51 | 0.51 | 0.50 |
| 0.5 | 0.6 | 0.48 | 0.53 | 0.53 | **0.50** |

Both corrected estimators recover the true genetic rg (0 and 0.5) at every `re`;
the naive ones drift up with the shared environment (rg=0.5, re=0.6 → 0.53 vs the
true 0.50), and LDSC's fitted intercept grows with the overlap that the slope then
ignores. Bivariate **LDpred3 with `cross_corr`** is the tightest and most robust
(it models the noise covariance in the likelihood rather than dividing by two
noisy marginal h² estimates — the quantity whose instability inflates LDSC's rg
when a heritability lands near zero). The remaining scatter (±~0.13 at rg=0) is
this small simulation's few-LD-block sampling variance, not a bias.

### Choosing `cross_corr` in practice

The bivariate sampler takes the overlap as `cross_corr` (the cross-trait
sampling-noise correlation). How much it matters and how to set it
(`benchmarks/overlap_estimation.py`):

- **The rg estimate is only mildly sensitive to overlap.** Fitting with
  `cross_corr=0` inflates rg by ~0.01–0.05 even at strong overlap (ρ=0.4) —
  because the model puts the overlap in the noise-covariance term
  `E12 = cross_corr/√(N₁ⱼN₂ⱼ)`, structurally separate from the LD-mediated genetic
  covariance. So `cross_corr=0` is a safe default; supply a value when you want an
  unbiased rg near zero or with large shared control sets.
- **If you know the overlap** (same cohort, documented shared samples), set
  `cross_corr = N_shared·ρ_pheno/√(N₁N₂)` (for fully shared samples, just the
  phenotypic correlation among them).
- **If you don't**, estimate it from the **cross-trait LDSC intercept** —
  `bipred.ldsc_rg(...).gcov_intercept`, which
  `bipred.estimate_sample_overlap(rg_result, N₁, N₂, pheno_corr)` inverts to a
  shared-sample count. This is the standard estimator and is well-anchored at real
  GWAS scale (millions of SNPs spanning a wide LD-score range); on a small panel
  the intercept is a noisy extrapolation, so treat its sign/magnitude as a
  *detector* there rather than a precise count.
- The overlap **cannot** be read back out of the bivariate sampler itself: fit
  with `cross_corr=0` it absorbs the overlap into (spurious) genetic covariance, so
  the fit residuals carry no overlap signal. Estimate it externally (LDSC) and pass
  it in.
