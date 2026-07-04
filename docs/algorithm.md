# bipred — algorithm and model

`bipred` jointly fits **two traits that share one LD reference** and reports their
genetic correlation and polygenic overlap. This is the model and sampler
reference; for genetic-correlation usage and guidance see [rg.md](rg.md). The
bivariate machinery builds on the univariate LDpred3 sampling model (each
marginal effect is the true effect plus LD-weighted spillover, `β̂ = R β + ε` with
`ε ~ N(0, R/N)`); the cross-trait extension is everything below.

## Bivariate (two-trait) LDpred3

`ldpred3_auto_bivariate` jointly fits **two traits that share one LD reference**.
Each variant takes one of **four** states — causal for neither trait, trait 1
only, trait 2 only, or **both** — with probabilities `(π₀₀, π₁₀, π₀₁, π₁₁)`. A
trait-1-causal effect is `N(0, s₁)`, a trait-2-causal one `N(0, s₂)`, and a
*both*-causal pair is `N(0, Σ)` with `Σ = [[s₁, s₁₂],[s₁₂, s₂]]`; the
off-diagonal `s₁₂` is the genetic covariance and the only place the traits
couple. Each Gibbs step evaluates the four bivariate-Gaussian likelihoods of the
residual estimate, samples a state, and draws the effects; `π` and the effect
covariance `Σ` are re-estimated each sweep. By default `r_g = β₁ᵀRβ₂ / √(h²₁h²₂)`
is reported from the same-sweep sampled effects (accurate and tight for
similarly-powered pairs); for **asymmetric-power** pairs — a strong trait boosting
a weak one — set `rg_decorrelated=True` to estimate it from effects sampled at
*different* sweeps (independent noise), which recovers the weak trait's covariance
that the same-sweep ratio attenuates.

```python
from bipred import ldpred3_auto_bivariate
res = ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n1, n2)
res.beta1_est, res.beta2_est      # adjusted effects for the two traits
res.h2, res.rg                    # (h2_1, h2_2) and the genetic correlation
res.pi, res.mixer                 # 4-state mixture + MiXeR-style overlap summary
```

**MiXeR-style polygenic overlap.** The four states — neither / trait-1-only /
trait-2-only / both causal, with posterior-mean probabilities `res.pi =
(π₀₀, π₁₀, π₀₁, π₁₁)` — make this a bivariate **causal mixture model** in the
sense of MiXeR ([Frei et al. 2019, *Nat. Commun.*](https://doi.org/10.1038/s41467-019-10310-0)).
`res.mixer` reports the same quantities: the
per-trait polygenicity `(π₁, π₂)` with `π₁ = π₁₀ + π₁₁`, the shared causal
fraction `π₁₁`, the correlation of effect sizes **within the shared component**
`ρ_β = s₁₂/√(s₁s₂)`, and the decomposition `r_g = ρ_β · π₁₁/√(π₁π₂)` (genetic
correlation = polygenic overlap × within-shared effect correlation). Two caveats:
the point-normal mixture does **not calibrate absolute polygenicity** (it counts
a causal variant's LD neighbours as partly causal too), so the absolute counts
are biased by an architecture- and power-dependent factor — `benchmarks/mixer_overlap.py`
measures anywhere from ~0.3× (clustered causal variants) to ~2.5× (spread across
LD blocks, large `N`) — while the *overlap fraction* `π₁₁/min(π₁,π₂)` and the
`r_g` decomposition **are** recovered across those same sweeps. The dominant term
in that bias is **LD-reference mismatch**, not the sampler: the inflation *grows*
with `N` (a misspecified-likelihood signature — a benign shrinkage bias would
shrink toward the prior as data accrue), and a control that re-fits the same data
on the *exact in-sample LD* collapses it from ~2.5× to ~1.1× at `N·h²/m=20`
(residual: the intrinsic weak identifiability of `π` under a Gaussian slab). `r_g`
and the overlap fraction are **ratios**, so the mismatch cancels and they stay
unbiased — only the absolute counts are exposed. So read the overlap fraction and
`r_g`, treat the absolute counts as relative, and note it needs a **well-powered**
pair (large `N·h²/m`) to separate the states. A dedicated causal-mixture
likelihood (MiXeR) is what calibrates the absolute counts (it fits the full
z-score distribution rather than counting posterior inclusions); here the overlap
comes for free from the joint fit.

**Calibrating the counts with univariate runs (`res.mixer_calibrated`).** Because
the bias is mostly LD-reference mismatch, and the four-state sampler is ~2× *more*
sensitive to it than a univariate fit, you can put the absolute counts on a
calibrated scale cheaply: run univariate `ldpred3.ldpred3_auto_infer` on each trait
and pass the two results to `res.mixer_calibrated(infer1, infer2)`. It keeps the
joint fit's reliable *ratios* (`frac_shared`, `ρ_β`) but replaces `(π₁, π₂)` with
the univariate learned polygenicities and rebuilds `n_causal` / `n_shared` on that
scale — e.g. at `m=5000, N=10⁵`, true 500/500 causal and 250 shared, the joint
`n_causal` (984, 922) / `n_shared` 539 become (497, 530) / 291. (The univariate
`p` is itself unbiased only when the LD matches; it too inflates with `N` under a
mismatched panel, just far less — so this anchors, it does not fully cure.)

The benchmark (realistic non-repeating coalescent LD, `m=5000`, 8 phenotype reps
on fixed genotypes) sweeps overlap, within-shared correlation `ρ_β`, power, and an
LD-match control:

| sweep | true → estimated | verdict |
|---|---|---|
| overlap (`frac_shared` 0→1) | 0.00→0.14, 0.25→0.34, 0.50→0.59, 0.75→0.80, 1.00→0.95; `r_g` tracks 0→0.8 | recovered (slight upward bias at low overlap) |
| `ρ_β` (0→0.9) | 0.00→0.03, 0.30→0.29, 0.60→0.54, 0.90→0.59; `frac_shared`≈0.5 throughout | recovered (attenuates near 1) |
| power (`N·h²/m` 1→20) | `frac_shared`≈0.6 and `r_g`≈0.4 stable; rel. polygenicity 1.3→2.5 | overlap/`r_g` stable, absolute counts drift with power |
| ld-match (`N·h²/m` 2→20) | rel. polygenicity **ref-panel** 1.4→2.6 vs **in-sample LD** ~1.1→1.3; `r_g`≈0.4 in both | the count drift is LD-reference mismatch, not the sampler; `r_g` immune |

**Why per-trait states (and not one shared causal indicator).** An earlier
prototype used a single shared indicator (both traits causal at the same SNPs).
That helps when the assumption holds but **hurts** badly when it doesn't — with
disjoint causal variants it forced sharing and dropped the weak trait's accuracy
by ~0.1. The four-state model *learns* whether causal variants co-occur (`π₁₁`),
so disjoint traits drive `π₁₁ → 0` and the joint fit reduces to the independent
ones. The effect covariance `Σ` is re-estimated each sweep by an
**inverse-Wishart-style shrinkage** update — the standard multiple-trait
animal-model approach (MTGSAM, [Van Tassell & Van Vleck 1996, *J. Anim. Sci.*](https://doi.org/10.2527/1996.74112586x);
[Sorensen & Gianola 2002, *Likelihood, Bayesian and MCMC Methods in Quantitative
Genetics*, Springer](https://doi.org/10.1007/b98952)) — shrinking `(s₁, s₂, s₁₂)`
toward a weak diagonal prior (zero prior genetic covariance). This keeps `Σ`
positive-definite by construction and the covariance off the boundary, and
regularises a weak trait's variance so it does not inflate by borrowing from a
strong correlated one — without the earlier ad-hoc univariate-h² ceiling (which
under-estimated h² on noisy dense LD and biased `r_g` upward) or hard PD cap, and
with no separate univariate pre-pass (so the fit is also faster).

The benchmark is **realistic**: the GWAS is generated from the true population
(coalescent) LD but fitted with an LD matrix estimated from a finite reference
panel (`Nref=2000`). For a genuinely under-powered trait 2 (N=2000, polygenic)
vs a well-powered trait 1 (N=100000), the gain grows with `r_g` and there is **no
harm** at low `r_g` or disjoint architectures:

| architecture | trait-2 alone | trait-2 joint | gain | r_g est |
|--------------|--------------:|--------------:|-----:|--------:|
| shared, r_g=0.0 | 0.641 | 0.636 | −0.005 | +0.02 |
| shared, r_g=0.3 | 0.647 | 0.641 | −0.006 | +0.39 |
| shared, r_g=0.6 | 0.655 | 0.694 | +0.039 | +0.67 |
| shared, r_g=0.9 | 0.658 | 0.830 | **+0.173** | +0.89 |
| disjoint causal | 0.630 | 0.610 | −0.020 | −0.08 |

The benefit is **real and large only where it should be** — a weak trait highly
correlated with a strong one — and negligible otherwise. It scales with how
under-powered trait 2 is: at N=1000 the rg=0.9 gain reaches ~+0.28, while for an
already well-powered trait 2 there is little to borrow and a small overhead, so
use the joint fit to boost an under-powered trait. (An earlier "fit with the true
LD" benchmark overstated the gains — they shrink markedly under realistic
reference-panel LD.) `ldpred3_auto_bivariate_blocks` is the streaming genome-wide
version; both GWAS must use the same LD/ancestry, and sample overlap is handled
via `cross_corr` (default 0). Regenerate with `benchmarks/bivariate_demo.py`.

**Genetic correlation vs bivariate LDSC.** The reported `r_g` has an independent
cross-check in `ldpred3.ldsc_rg` (cross-trait LD Score regression). Under the same
realistic reference-panel LD both are roughly unbiased from the same summary
statistics; bivariate LDpred3 is ~2× more precise (it uses the full LD
likelihood):

| true r_g | bivariate LDSC | bivariate LDpred3 |
|---------:|---------------:|------------------:|
| 0.0 | 0.02 ± 0.11 | −0.00 ± 0.07 |
| 0.4 | 0.40 ± 0.10 | 0.39 ± 0.07 |
| 0.6 | 0.60 ± 0.06 | 0.60 ± 0.04 |
| 0.8 | 0.80 ± 0.03 | 0.79 ± 0.02 |

(An infinitesimal trait on realistic, non-repeating coalescent LD fit from a
finite reference panel; LDpred3's precision edge is largest — and LDSC's `r_g`
most fragile — on sparse and major-locus architectures, detailed in
[rg.md](rg.md).) Regenerate with `benchmarks/rg_architectures.py`.

### The genetic-correlation estimators (theory & trade-offs)

The genetic correlation is `r_g = cov_g / √(h²₁·h²₂)`, where the genetic
covariance is the **LD-aware** quadratic form `cov_g = β₁ᵀRβ₂` in the true
standardized effects and `h²_t = β_tᵀRβ_t`. Every estimator differs only in how
it forms this numerator and denominator from noisy, shrunk effect estimates, and
one principle governs all of them — **scale matching**: numerator and denominator
must be built from the *same* (shrunk) quantities, or the shrinkage fails to
cancel and `r_g` is biased.

1. **Joint sampled-quadratic ("gv", the default).** From the bivariate chain,
   `r_g = β̂₁ᵀRβ̂₂ / √(β̂₁ᵀRβ̂₁·β̂₂ᵀRβ̂₂)` on the sampled posterior-mean effects.
   Each `β̂_t` is a shrinkage estimate of `β_t`, but because numerator and
   denominator are all built from the *same* posterior means the (non-uniform)
   shrinkage largely cancels in the ratio — which is why `r_g` is recovered even
   where the individual `h²` are not.
2. **Decorrelated cross-sweep covariance (`rg_decorrelated=True`).** The
   same-sweep numerator shares one sweep's sampling noise across the two traits;
   for an asymmetric pair (strong trait 1, weak trait 2) this leaks trait-1 signal
   into trait 2 and attenuates the weak trait's covariance. Taking `β₁` and `β₂`
   from *different* sweeps makes their noise independent, so `E[β₁ᵀRβ₂] =
   (Eβ₁)ᵀR(Eβ₂)` with no cross-noise — the same trick that makes LDpred2-auto's
   out-of-sample `r²` a cross-chain product ([Privé et al. 2023,
   *Am. J. Hum. Genet.*](https://doi.org/10.1016/j.ajhg.2023.10.010)).
3. **Two univariate runs** (skip the joint model; combine independent `-auto`
   fits of each trait):
   - `uni_gv = β̂₁ᵀRβ̂₂ / √(β̂₁ᵀRβ̂₁·β̂₂ᵀRβ̂₂)` — the numerator uses posterior
     means from two *independent* runs, so their noise is already decorrelated
     (cross terms vanish in expectation), giving a clean covariance.
   - `uni_r2 = β̂₁ᵀRβ̂₂ / √(r²₁·r²₂)` — same numerator, but the denominator uses
     each run's **decorrelated out-of-sample `r²`** (`InferResult.r2_est`, a
     cross-chain quadratic `bᵢᵀRbⱼ`, `i≠j`), which removes the small positive
     self-noise bias in `β̂ᵀRβ̂`.
   In practice **`uni_gv` and `uni_r2` are numerically identical**: `β̂` is
   averaged over many samples across chains, so its residual variance — the only
   thing the `r²` denominator de-biases — is negligible. Both are **slightly
   attenuated** vs the joint fit, because the residual bias lives in the
   *numerator* (each posterior mean is non-uniformly shrunk across SNPs, so the
   self-normalizing ratio does not perfectly cancel) — which no choice of
   denominator can fix.

**Why "calibrating" the denominator with `h²` fails.** It is tempting to correct
the covariance/count bias by dividing by a well-calibrated `h²` (e.g. from
univariate runs, which estimate `h²` accurately). This *breaks* scale matching
and attenuates `r_g`: (a) a shrunk posterior-mean numerator over an *unshrunk*
`h²` denominator reintroduces the shrinkage; (b) building the covariance from the
mixture variance components, `cov_g ≈ M·π₁₁·s₁₂`, under-estimates it because that
diagonal product **ignores the LD cross-terms** in `βᵀRβ` — it sits at ~0.6× the
true `h²` scale, so dividing by the true-scale `h²` attenuates further. Both were
measured and both attenuate; the lesson is that `r_g` must be a ratio of
*like-scaled* quantities. Cross-trait **LDSC** ([Bulik-Sullivan et al. 2015,
*Nat. Genet.*, "An atlas of genetic correlations"](https://doi.org/10.1038/ng.3406);
h² regression from [Bulik-Sullivan et al. 2015, *Nat. Genet.*](https://doi.org/10.1038/ng.3211))
side-steps shrinkage entirely — it regresses the product of the two traits'
z-scores on the LD score, a method-of-moments covariance — so it does **not**
attenuate under power asymmetry, at the cost of higher variance (noisier at low
`r_g`).

**Accuracy and running time** (`benchmarks/rg_methods.py`, realistic LD, `m` up
to 50k, symmetric vs asymmetric power):

| estimator | symmetric (true 0.90) | asymmetric, weak trait 2 (true 0.90) | time / fit (m=20k) |
|---|---:|---:|---:|
| bivariate joint fit | **0.90** | **0.90** (borrows for the weak trait) | 1.1 s |
| cross-trait LDSC | 0.89 (noisier at low `r_g`) | 0.90 (moment-based, no shrinkage) | 0.3 s |
| `uni_gv` / `uni_r2` | 0.85 (attenuated) | 0.78 (badly attenuated) | 6.3 s |

So the **bivariate joint fit is both the most accurate and, per fit, ~5× cheaper
than the univariate pair** (which must run several chains per trait); it is the
recommended `r_g` estimator, with LDSC as a cheap moment-based screen. The
univariate estimators are a useful independent cross-check but attenuate under
power asymmetry — exactly where the joint fit's cross-trait borrowing helps most.
