# Benchmark: estimating `cross_corr` in the bivariate Gibbs sampler

**Status: research benchmark.** Not a shipped feature. See [`README.md`](README.md)
for the method and [`bench_cross_corr.py`](bench_cross_corr.py) for the code.

## What is being compared

`cross_corr` is the cross-trait correlation of the GWAS *sampling noise* induced
by overlapping samples. bipred takes it as a fixed user input today. The question
is not "does estimating it beat ignoring it" — that is a weak claim — but
**does estimating it in the sampler match or beat bipred's existing recommended
practice**, deriving it from the cross-trait LDSC intercept?

All four arms run the **same** sampler on the **same** simulated data and differ
in exactly one thing, the `cross_corr` value:

| arm | `cross_corr` | represents |
|---|---|---|
| `naive` | `0` | bipred's default when unspecified |
| `ldsc` | `ldsc_rg(...).gcov_intercept` | bipred's recommended practice today |
| `joint` | estimated every sweep | **the proposed method** |
| `oracle` | the true simulated value | upper bound; not attainable in practice |

Identity used by the `ldsc` arm: with noise variances `1/N_t` and covariance
`cross_corr/sqrt(N1 N2)`, the noise *correlation* is exactly `cross_corr`, which
is also what the cross-trait LDSC intercept estimates — directly comparable, no
rescaling.

Design: infinitesimal bivariate liability, block-diagonal AR(1) LD, `h² = 0.5`
per trait, **20 replicates per cell**, 900 sweeps (300 burn-in). Each replicate
redraws **both** the causal effects and the GWAS sampling noise, so replicates
are independent. Bias and RMSE are scored against each replicate's **realized**
r_g (the estimand of that finite effect draw), not the nominal r_g used to
generate it. Dispersions are SDs across replicates (ddof=1), not Monte-Carlo
standard errors.

> **Correction.** An earlier version of this benchmark drew the causal effects
> **once** and reused them across every replicate and every cell. Because the
> LDSC intercept is a functional of the genetic architecture, each cell's LDSC
> number was then a single draw rather than an estimate, and the across-draw SD
> exceeded the effects being reported. That version supported claims — "LDSC
> drifts high", "LDSC systematically over-corrects", "the LDSC arm degrades as N
> grows" — which are **false**: with effects redrawn per replicate, the LDSC
> intercept is essentially unbiased everywhere below. The variance results, and
> the headline, survive.

## 1. Main grid — r_g × cross_corr (N = 8,000, m = 6,000)

[`bench_cross_corr_main.csv`](bench_cross_corr_main.csv)

| true r_g | true cc | naive bias | ldsc RMSE | **joint RMSE** | oracle RMSE | ldsc `ĉc` (sd) | joint `ĉc` (sd) |
|---:|---:|---:|---:|---:|---:|---|---|
| 0.0 | 0.0 | −0.002 | 0.139 | 0.036 | 0.022 | 0.001 (0.102) | **0.001 (0.019)** |
| 0.0 | 0.2 | +0.267 | 0.144 | **0.039** | 0.024 | 0.197 (0.107) | **0.201 (0.023)** |
| 0.0 | 0.4 | +0.507 | 0.155 | **0.041** | 0.026 | 0.392 (0.116) | **0.400 (0.025)** |
| 0.3 | 0.0 | +0.003 | 0.138 | 0.034 | 0.020 | −0.001 (0.109) | **−0.002 (0.018)** |
| 0.3 | 0.2 | +0.251 | 0.146 | **0.035** | 0.020 | 0.196 (0.116) | **0.198 (0.021)** |
| 0.3 | 0.4 | +0.450 | 0.161 | **0.036** | 0.021 | 0.391 (0.126) | **0.398 (0.023)** |
| 0.6 | 0.0 | +0.009 | 0.129 | 0.032 | 0.018 | −0.001 (0.124) | **−0.007 (0.018)** |
| 0.6 | 0.2 | +0.219 | 0.138 | **0.030** | 0.016 | 0.195 (0.131) | **0.194 (0.019)** |
| 0.6 | 0.4 | +0.354 | 0.155 | **0.029** | 0.015 | 0.391 (0.141) | **0.395 (0.020)** |

Findings:

1. **`cross_corr` is recovered accurately, with no false positives.** At true
   `cross_corr = 0` the estimate is −0.007 to +0.001; at 0.2 and 0.4 it recovers
   0.194–0.201 and 0.395–0.400, at every r_g.
2. **Estimating it removes the overlap bias in r_g.** Uncorrected, overlap
   inflates r_g by up to **+0.51**; `joint` stays within 0.014 of the realized
   truth and tracks the oracle.
3. **Both `ldsc` and `joint` are unbiased; the difference is variance.** The
   LDSC intercept is essentially unbiased (`ĉc` 0.391–0.392 against 0.4; r_g bias
   ≤ 0.005), but its SD is **5–7× larger** than `joint`'s (0.10–0.14 vs
   0.018–0.025). That variance is what drives the 4–5× RMSE gap.
4. **Freeing the parameter costs precision when there is no overlap.** At
   `cross_corr = 0`, RMSE rises from 0.018–0.022 (`naive`/`oracle`) to
   0.032–0.036 (`joint`) — about 1.7×. This is the argument for keeping any such
   feature **opt-in**, not default-on.

## 2. Sample size (r_g = 0.6, cross_corr = 0.4, m = 6,000)

[`bench_cross_corr_n.csv`](bench_cross_corr_n.csv)

| N | naive bias | ldsc RMSE | **joint RMSE** | oracle RMSE | ldsc `ĉc` (sd) | joint `ĉc` (sd) |
|---:|---:|---:|---:|---:|---|---|
| 4,000 | +0.381 | 0.192 | **0.041** | 0.023 | 0.395 (0.101) | **0.396 (0.016)** |
| 8,000 | +0.354 | 0.155 | **0.029** | 0.015 | 0.391 (0.141) | **0.395 (0.020)** |
| 20,000 | +0.203 | 0.146 | **0.021** | 0.010 | 0.385 (0.259) | **0.394 (0.030)** |
| 50,000 | +0.109 | 0.142 | **0.016** | 0.008 | 0.344 (0.513) | **0.393 (0.051)** |

The uncorrected bias shrinks as N grows (genetic signal grows relative to the
fixed overlap noise) but never vanishes. `joint` improves monotonically. The LDSC
intercept stays roughly unbiased but its **variance grows sharply with N**
(SD 0.10 → 0.51): on the z-scale the LD-score slope term scales with
`sqrt(N1 N2)` while the intercept does not, so a fixed architecture misfit is
amplified linearly in N. `joint`'s SD also grows (0.016 → 0.051) but stays ~10×
smaller.

## 3. Number of variants (r_g = 0.6, cross_corr = 0.4, N = 8,000)

[`bench_cross_corr_m.csv`](bench_cross_corr_m.csv)

| m | ldsc RMSE | **joint RMSE** | ldsc `ĉc` (sd) | joint `ĉc` (sd) | sd ratio |
|---:|---:|---:|---|---|---:|
| 1,500 | 0.244 | **0.034** | 0.379 (0.625) | **0.374 (0.063)** | 9.9× |
| 3,000 | 0.225 | **0.035** | 0.354 (0.284) | **0.399 (0.038)** | 7.4× |
| 6,000 | 0.155 | **0.029** | 0.391 (0.141) | **0.395 (0.020)** | 7.2× |
| 12,000 | 0.140 | **0.044** | 0.391 (0.071) | **0.399 (0.015)** | 4.9× |

The LDSC intercept's SD falls 0.625 → 0.071 as m grows 8× — it is converging,
roughly as `1/sqrt(m)` — and the advantage ratio narrows from 9.9× to 4.9×.
§3b extends this to m = 100,000 and finds the crossover.

## 3b. Large m — where the genome-wide advantage ends

[`bench_cross_corr_scale.csv`](bench_cross_corr_scale.csv), r_g = 0.6,
cross_corr = 0.4, N = 8,000, 4 replicates, **1,500 sweeps** (see the warning
below).

| m | ldsc `ĉc` (sd) | joint `ĉc` (sd) | sd ratio | oracle r_g bias |
|---:|---|---|---:|---:|
| 12,500 | 0.419 (0.076) | **0.385 (0.012)** | **6.3×** | +0.005 |
| 25,000 | 0.398 (0.025) | **0.392 (0.009)** | **3.0×** | +0.026 |
| 50,000 | **0.394 (0.004)** | 0.404 (0.007) | 0.5× | +0.008 |
| 100,000 | **0.398 (0.004)** | 0.411 (0.005) | 0.8× | −0.110 |

**The genome-wide advantage does not survive to large m.** The crossover is near
**m ≈ 50,000**: beyond it the cross-trait LDSC intercept estimates `cross_corr`
as well as or better than the in-sampler update, exactly as the `1/sqrt(m)`
convergence in §3 predicted. For a genome-scale analysis with ~10⁶ variants, the
existing `estimate_sample_overlap` route is the right tool and this feature adds
nothing.

That is the honest genome-wide conclusion. The case for building the in-sampler
estimator is **regional** r_g, where m per region is 10²–10³ — the left-hand end
of this table, where the advantage is 6–10× — and where the LDSC intercept is not
merely worse but actively harmful. See
[`RESULTS_REGIONAL.md`](RESULTS_REGIONAL.md).

> **Warning on the m = 100,000 row.** The oracle — which has no `cross_corr`
> uncertainty at all — carries a bias of −0.110 there, so *every* arm in that row
> is affected by something other than `cross_corr`: at m = 100,000 the per-SNP
> power is `N·h²/m` = 0.04, and the sampler has not fully converged even at 1,500
> sweeps. Do not read the `joint` value in that row as a property of the
> estimator. An earlier version of this grid ran 400 sweeps and showed a much
> larger apparent collapse (oracle bias −0.246 at 400 sweeps versus −0.040 at
> 1,200 in a direct check) — that was a convergence artifact, not a result.

## 4. LD-score range (r_g = 0.6, cross_corr = 0.4, N = 8,000, m = 6,000)

[`bench_cross_corr_ldwide.csv`](bench_cross_corr_ldwide.csv). The cross-trait
LDSC intercept is identified only by variation in LD score, so a narrow range
would disadvantage the `ldsc` arm and make the `joint` win an artifact.

| ell range | span | ldsc RMSE | **joint RMSE** | oracle RMSE | ldsc `ĉc` (sd) | joint `ĉc` (sd) |
|---|---:|---:|---:|---:|---|---|
| 1.04–4.49 | 4.3× | 0.155 | **0.029** | 0.015 | 0.391 (0.141) | **0.395 (0.020)** |
| 1.34–18.65 | 14× | 0.241 | **0.019** | 0.017 | 0.383 (0.226) | **0.399 (0.011)** |
| 1.34–57.59 | 43× | 0.225 | **0.021** | 0.020 | 0.386 (0.213) | **0.398 (0.011)** |

Widening the LD-score range 10× does not close the gap. **But this grid does not
cleanly isolate the LD-score-range effect**: raising the AR(1) ρ widens `ell` and
simultaneously *reduces the effective number of independent SNPs*, which hurts
LDSC for a different reason. Read it as "the result is not obviously an artifact
of narrow LD-score range", not as a controlled test.

## 5. Limitations — read these before believing the headline

1. **The genome-wide advantage ends near m ≈ 50,000** (§3b). Beyond that the
   LDSC intercept matches or beats the in-sampler estimator, so for a
   genome-scale analysis (~10⁶ variants) the existing `estimate_sample_overlap`
   route is the right tool and this feature adds nothing. The defensible
   genome-wide claim is: *below ~50k variants, joint estimation is near-oracle
   and 3–10× lower variance than the LDSC intercept; above it, the two are
   comparable and LDSC is cheaper.* The motivating use case is therefore
   **regional** r_g ([`RESULTS_REGIONAL.md`](RESULTS_REGIONAL.md)), where m per
   region is 10²–10³ by construction.
2. **In-model.** The data are simulated with exactly the noise structure the
   sampler assumes. This is favourable terrain for a model-based estimator — the
   same caveat that applies to bipred's own §1–4 benchmarks.
3. **Overlap only.** The simulated cross-trait noise is caused purely by sample
   overlap. Correlated population stratification also produces cross-trait
   covariance, and this benchmark does **not** test whether the estimator
   misattributes it. Nor does it separate genetic from environmental components:
   `cross_corr` is not `r_e`.
4. **Infinitesimal + dense LD.** bipred's production sampler is a four-state
   mixture over int8 / low-rank LD, and the whitening step needs a per-block
   `L⁻¹` that those representations do not carry. The `cross_corr` update is
   argued to be orthogonal to the mixture, but that is asserted, not shown.
5. **One quadrant of the parameter space.** No negative `cross_corr`, no
   `|cross_corr| > 0.4`, always `N1 = N2`, always `h² = 0.5` for both traits, and
   r_g never near ±1. The true `cross_corr` also always lies exactly on the
   sampler's 0.01 search grid, which slightly flatters `joint`.
6. **Only r_g is scored.** Per-trait h² and out-of-sample PRS accuracy — both
   bipred deliverables — are not measured.
7. **One LD realisation per cell** (`ld_seed = 7`), so LD-structure variability is
   not represented; only effects and sampling noise are redrawn.
8. **20 replicates**, so the SDs are themselves noisy; small differences between
   neighbouring cells should not be over-read.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/bench_cross_corr.py all 20
```

Numba-accelerated (falls back to pure NumPy); about 30 minutes for all four grids
on 4 cores. Writes `bench_cross_corr_{main,n,m,ldwide}.csv`.
