# bipred user guide

bipred fits **two GWAS traits jointly** from their summary statistics and one
shared LD reference, in a single four-state Gibbs sampler. From that one fit you
get each trait's SNP heritability, the **genetic correlation** `r_g`, a
MiXeR-style **polygenic-overlap** summary, and posterior-mean effects for
**prediction** — where a well-powered trait sharpens a correlated under-powered
one. This guide is the practical how-to; for the model and sampler see
[algorithm.md](algorithm.md), and for genetic-correlation accuracy and the
overlap readout in depth see [rg.md](rg.md).

bipred is a **Python library** (no CLI). It depends on
[ldpred3](https://github.com/bvilhjal/ldpred3), which it uses for LD handling,
summary-statistics harmonisation and the Numba sampler internals — so you prepare
inputs with ldpred3 and run the bivariate fit with bipred.

---

## When to use it

Reach for bipred when you have **two traits measured on (mostly) the same
ancestry**, each with GWAS summary statistics, and you expect them to be
genetically correlated. Three things it gives you:

| goal | what to read | notes |
|---|---|---|
| **Estimate `r_g`** between two traits | `res.rg` (and `bipred.ldsc_rg` as a fast cross-check) | the most robust output — accurate and power-stable |
| **Improve a PRS** for an under-powered trait | `res.beta1_est` / `res.beta2_est` | works when the traits are correlated and one is better powered; **no harm** when they are disjoint |
| **Quantify polygenic overlap** | `res.mixer` (shared fraction, `ρ_β`, counts) | the *fractions* are reliable; the absolute *counts* are approximate — see [§6](#6-polygenic-overlap-mixer-style) |

If the two traits turn out to share no causal variants, the joint fit learns
that (`π₁₁ → 0`) and reduces to two independent fits — so running bipred is safe
even when you are not sure there is overlap.

---

## 1. What you need

Three inputs, on the **standardized (allele-frequency-normalised) scale** that
ldpred3 uses throughout:

1. **Two sets of GWAS summary statistics** for the *same variants in the same
   order*: standardized marginal effects `beta_hat1`, `beta_hat2` (each a length-`m`
   array) and per-trait effective sample sizes `n_eff1`, `n_eff2` (a scalar or a
   length-`m` array).
2. **One LD reference** matching the GWAS ancestry, as either
   - a dense `m × m` correlation matrix `corr`, or
   - a list of per-block LD, `blocks = [(R_b, idx_b), ...]`, where each `idx_b` is
     a contiguous index array and together they tile `0 … m-1` (the streaming,
     genome-wide path).
3. **Sample-overlap information**, if the two GWAS share individuals — a single
   `cross_corr` scalar (see [§7](#7-sample-overlap)). Default `0` assumes
   independent samples.

bipred does **not** build LD or harmonise sumstats itself — use ldpred3 for that
(its [user guide](https://github.com/bvilhjal/ldpred3/blob/master/docs/guide.md),
"What you need" and "Working from your own LD blocks"). The two traits must be
harmonised to the **same** variant order and allele orientation as the LD
reference; a flip in one trait silently corrupts the cross-trait signal.

---

## 2. Quickstart

**One LD block (or a small genome packed into one dense matrix):**

```python
import numpy as np
from bipred import ldpred3_auto_bivariate

res = ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n1, n2)
print(res)          # BivariateResult(h2=(0.31, 0.28), rg=+0.42, p=0.03, n_variants=5000)
```

**Genome-wide (stream block by block — never materialises the whole LD):**

```python
from bipred import ldpred3_auto_bivariate_blocks

# blocks = [(R_0, idx_0), (R_1, idx_1), ...] tiling 0..m-1; the two traits share it
res = ldpred3_auto_bivariate_blocks(blocks, beta_hat1, beta_hat2, n1, n2,
                                    burn_in=200, num_iter=200, seed=0)
```

Both calls return a `BivariateResult`. The blocks may be dense float32 or
ldpred3 low-rank (`LowRankLD`) blocks; low-rank scales inference the same way it
scales scoring, so a genome-wide fit stays light (see [§9](#9-scaling--performance)).

---

## 3. Reading the output (`BivariateResult`)

| field | meaning |
|---|---|
| `beta1_est`, `beta2_est` | posterior-mean standardized effects for the two traits — feed to a PRS |
| `h2` | `(h²₁, h²₂)` SNP heritabilities |
| `rg` | genetic correlation (the headline estimate) |
| `p` | overall causal fraction (`π₁₀+π₀₁+π₁₁`) |
| `sigma` | learned `2×2` effect covariance `Σ = [[s₁, s₁₂],[s₁₂, s₂]]` |
| `pi` | posterior-mean four-state mixture `(π₀₀, π₁₀, π₀₁, π₁₁)` = neither / trait-1-only / trait-2-only / both causal |
| `pi_samples`, `sigma_samples` | retained post-burn-in draws (used by `mixer_posterior`) |
| `noise_scale` | learned `(λ₁, λ₂)` if `noise_inflation=True`, else `(1, 1)` |

**Polygenic overlap — `res.mixer`** (a dict) reports the MiXeR quantities derived
from `pi`/`sigma`:

```python
mx = res.mixer
mx["polygenicity"]     # (π₁, π₂) per-trait causal fractions, πₜ = π_t0 + π11
mx["n_causal"]         # (π₁·m, π₂·m) per-trait causal counts
mx["n_shared"]         # π11·m shared causal count
mx["frac_shared"]      # π11 / min(π₁, π₂)  — the polygenic-overlap fraction
mx["rho_beta"]         # within-shared effect correlation ρ_β
mx["rg_from_overlap"]  # ρ_β · π11/√(π₁π₂) — should match res.rg
```

- **`res.mixer_posterior(level=0.95)`** — the same quantities with **credible
  intervals** from the retained draws (a posterior, not a point).
- **`res.mixer_calibrated(infer1, infer2)`** — the counts re-scaled by two
  univariate `ldpred3.ldpred3_auto_infer` runs (see [§6](#6-polygenic-overlap-mixer-style)).

---

## 4. Genetic correlation — which estimator

`res.rg` from the joint fit is the recommended estimate: it is architecture-robust,
unbiased at `r_g = 0`, and typically ~2× more precise than LD Score regression
because it uses the full LD likelihood. Two situations call for a variant:

- **Asymmetric power** (a strong trait boosting a weak one): pass
  `rg_decorrelated=True`. The default same-sweep ratio slightly attenuates a weak
  trait's covariance; the decorrelated estimator (effects sampled at independent
  sweeps) recovers it, at the cost of a small over-estimate when the two traits
  are balanced.
- **A fast screen / independent cross-check**: `bipred.ldsc_rg` implements
  cross-trait LD Score regression (moment-based, no shrinkage attenuation, but
  noisier at low `r_g`):

  ```python
  from bipred import ldsc_rg
  from ldpred3 import ld_scores
  ell = ld_scores(blocks)                       # per-SNP LD scores
  rgr = ldsc_rg(beta_hat1, beta_hat2, ell, n1, n2)
  rgr.rg, rgr.rg_se, rgr.gcov_intercept         # intercept ≈ 0 without sample overlap
  ```

See [rg.md](rg.md) for the full accuracy/timing comparison and the theory of why
each estimator behaves as it does.

---

## 5. Prediction: boosting an under-powered trait

`beta1_est` / `beta2_est` are posterior-mean effects you score exactly like any
LDpred weights. The joint fit's value over two independent runs is **cross-trait
borrowing**: when trait 2 is under-powered but genetically correlated with a
well-powered trait 1, the shared component lets trait 1 sharpen trait 2's effects.
The gain grows with `r_g` and with the power asymmetry, and there is **no harm**
at `r_g ≈ 0` or with disjoint causal variants (the model drives `π₁₁ → 0` and the
traits decouple). See [algorithm.md](algorithm.md) for the benchmarked gains.

---

## 6. Polygenic overlap (MiXeR-style)

The **ratios** — `frac_shared`, `ρ_β`, `rg_from_overlap` — are reliable and
power-stable. The **absolute counts** (`n_causal`, `n_shared`) carry a mild
over-count that grows with per-SNP power (the point-normal counts a causal's LD
neighbours as partly causal — "LD-spreading"), a little larger on a finite
reference panel. With the default `p_init=0.02` the per-trait count is ≈1× truth
up to `N·h²/M ≈ 0.5` and ≈1.1–1.2× by `N·h²/M = 2`. Three ways to handle it:

- **Read the fractions**, treat the counts as approximate — always safe.
- **`noise_inflation=True`** removes the reference-mismatch part of the residual
  (learns a per-trait `λ ≥ 1` from the residual misfit), with `h²`/`r_g`
  unchanged. Recommended when fitting on a finite reference panel.
- **`res.mixer_calibrated(infer1, infer2)`** anchors the per-trait counts on two
  univariate runs (which over-count less than the four-state fit):

  ```python
  from ldpred3 import ldpred3_auto_infer
  inf1 = ldpred3_auto_infer(blocks, beta_hat1, n1)
  inf2 = ldpred3_auto_infer(blocks, beta_hat2, n2)
  cal = res.mixer_calibrated(inf1, inf2)        # keeps the joint frac_shared
  ```

At very low power the absolute count is influenced by `p_init` (the data barely
pin `p`), so it is most trustworthy near the default polygenicity and at adequate
power. Full benchmarks and numbers: [rg.md, "Absolute counts"](rg.md).

---

## 7. Sample overlap

If the two GWAS share individuals, their sampling noise is correlated and inflates
a naive `r_g`. Pass the overlap as `cross_corr` (the cross-trait sampling-noise
correlation, i.e. the bivariate-LDSC intercept):

- **You know the overlap** (same cohort / documented shared controls):
  `cross_corr = N_shared · ρ_pheno / √(N₁N₂)` — for fully shared samples, just the
  phenotypic correlation among them.
- **You don't**: estimate it from the cross-trait LDSC intercept —
  `bipred.ldsc_rg(...).gcov_intercept`, which
  `bipred.estimate_sample_overlap(rgr, n1, n2, pheno_corr)` inverts to a shared-sample
  count.

`cross_corr=0` (the default) is a safe starting point — the `r_g` estimate is only
mildly sensitive to overlap — but supply a value when you need an unbiased `r_g`
near zero or with large shared control sets. Full treatment (including
environment-correlation controls): [rg.md, "Handling sample overlap"](rg.md).

---

## 8. Options reference

All are keyword arguments to `ldpred3_auto_bivariate[_blocks]`:

| option | default | what it does |
|---|---|---|
| `burn_in`, `num_iter` | `200`, `200` | Gibbs burn-in and retained sweeps; raise both if `rg`/`h²` look unconverged |
| `h2_init`, `p_init`, `rg_init` | `0.1`, `0.02`, `0.0` | sampler starting values; `p_init=0.02` is a realistic causal fraction |
| `cross_corr` | `0.0` | sample-overlap noise correlation ([§7](#7-sample-overlap)) |
| `rg_decorrelated` | `False` | use the asymmetric-power `r_g` estimator ([§4](#4-genetic-correlation--which-estimator)) |
| `noise_inflation`, `ni_damp` | `False`, `0.1` | learn `λₜ ≥ 1` to trim mismatch-inflated counts ([§6](#6-polygenic-overlap-mixer-style)) |
| `pi_prior` | `1.0` | Dirichlet concentration for the mixture prior (`0.5` = Jeffreys); minor lever |
| `h2_bounds`, `h2_cap` | `(1e-4, 1.0)`, `None` | clamp the per-sweep `h²` estimate |
| `iw_df` | `10.0` | inverse-Wishart shrinkage strength on `Σ` (larger = more toward independent traits) |
| `sample_every` | `5` | thinning for the retained effect draws (decorrelated `r_g`) |
| `seed` | `None` | RNG seed for reproducibility |

---

## 9. Scaling & performance

- **Genome-wide → use the blocks path.** `ldpred3_auto_bivariate_blocks` streams
  one LD block at a time, so the full LD is never materialised; memory is set by
  the largest block, not the genome.
- **Low-rank LD** blocks (`ldpred3.LowRankLD`) keep the compact eigenspace end to
  end, so inference scales like scoring — prefer them at genome scale.
- **Cost** (single core, realistic LD, `m=5000`, 25 blocks): the joint fit is
  ~0.23 s at ~0.27 GB peak — sub-second and light. `bipred.ldsc_rg` is ~10× faster
  still (use it as the instant screen), and the univariate `ldpred3_auto_infer`
  runs for `mixer_calibrated` add a couple of multi-chain fits.
- Install the `[fast]` extra (Numba) — the pure-NumPy fallback is far slower.

---

## 10. Pitfalls & troubleshooting

- **Same-ancestry LD.** Both GWAS must match the LD reference's ancestry; a
  mismatched panel biases `h²`/counts (though `r_g` is largely immune).
- **Variant order & orientation.** The two traits and the LD reference must share
  one harmonised variant order; a strand/allele flip in one trait corrupts the
  cross-trait signal. Harmonise with ldpred3 before fitting.
- **Standardized scale.** `beta_hat` must be on the allele-standardized scale
  (as ldpred3 produces), not raw per-allele betas.
- **Absolute overlap counts are approximate** at low power — read the ratios, or
  use `noise_inflation` / `mixer_calibrated` ([§6](#6-polygenic-overlap-mixer-style)).
- **Convergence.** If `rg`/`h²` are unstable across seeds, raise `burn_in` /
  `num_iter`; check that `h²` is not pinned at a `h2_bounds` edge.
- **`ldsc_rg` returns `|rg| > 1` or a huge SE** on a small panel — the intercept /
  marginal-`h²` extrapolation is noisy there; trust the joint fit's `rg` and treat
  LDSC as a screen (it is well-anchored only at real GWAS scale).

---

## See also

- [algorithm.md](algorithm.md) — the four-state model, the Gibbs sampler, the two
  `r_g` estimators, and the polygenic-overlap decomposition.
- [rg.md](rg.md) — `r_g` accuracy vs bivariate LDSC, sample-overlap and
  environment-correlation controls, and the calibrated overlap counts.
- [ldpred3](https://github.com/bvilhjal/ldpred3) — LD construction, sumstats
  harmonisation, univariate PRS / inference / fine-mapping that bipred builds on.
