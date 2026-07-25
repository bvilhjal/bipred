# Notes on estimating overlap and regional genetic correlation

*An account of the investigation, including the parts that did not work.*

These notes are the connected story; [`RESULTS.md`](RESULTS.md) and
[`RESULTS_REGIONAL.md`](RESULTS_REGIONAL.md) hold the numbers, and
[`README.md`](README.md) is the index of files. Nothing here is shipped: this is
a record of what was tried, what was measured, and what may be concluded.

The reader who wants only the conclusion may skip to §11. The reader who wants to
avoid repeating our mistakes should read §6 and §9, which are the two places we
were wrong.

---

## 1. The problem

Two GWAS that share individuals have *correlated sampling noise*. Write the
residualised marginal effect for variant *j* as

    d_j = beta_j + e_j ,        e_j ~ N(0, E)

with

    E = [[ 1/N1              , c/sqrt(N1 N2) ]
         [ c/sqrt(N1 N2)      , 1/N2         ]] .

The off-diagonal parameter *c* — `cross_corr` in bipred — is the correlation of
that noise, and it is not a nuisance one may politely ignore: it enters the
cross-trait covariance in exactly the same place as the genetic covariance, so
ignoring it inflates the estimated genetic correlation. In our benchmark an
overlap of c = 0.4 inflates a true r_g of 0.0 to **0.507**.

bipred today asks the user to supply *c*, typically obtained from the cross-trait
LDSC intercept. The question that started this work: **can the Gibbs sampler
estimate *c* itself?**

A clarification that has caused confusion, so let us be explicit. *c* is not the
environmental correlation r_e. It is the correlation of the *sampling noise*,
which reflects the phenotypic correlation among the shared individuals — genetic
and environmental together. Summary statistics do not identify the environmental
part separately, and nothing here attempts to.

## 2. A wrong turn, taken first

The obvious sufficient statistic is the per-SNP residual `d_j − beta_j`,
accumulated across variants. Conditional on the sampled effects this ought to be
a draw from N(0, E), and its cross-moment ought to identify *c*.

It does not. Our first implementation returned **c ≈ −0.43 when the truth was
0** — not merely noisy but confidently wrong, and wrong in sign.

Two things are being neglected. First, with r_g > 0 the per-SNP bivariate draw
couples the two traits' residuals, so `d_j − beta_j` is not the noise realisation
one imagines. Second, and more seriously, the GWAS noise is *LD-structured*: its
covariance is R/N, not diagonal. Treating neighbouring variants' residuals as
independent samples of the same 2×2 distribution is simply the wrong likelihood.

> **Lesson 1.** A statistic that is unbiased in the no-LD limit may be badly
> biased under LD. Check it against a known truth before believing it.

## 3. The whitening, and why it is the right statistic

Since Cov(e) = R/N with R = L Lᵀ, the cure is to whiten by the LD Cholesky:

    z_t = sqrt(N_t) · L^{-1} ( bhat_t − R beta_t ) ,     t = 1, 2 .

Now z is i.i.d. bivariate normal with **unit variances** and correlation exactly
*c*. Because the diagonals are *known*, the conditional for *c* is one-dimensional
and we may evaluate it exactly on a grid:

    log p(c | z) = −(k/2) log(1 − c²) − ( S11 − 2 c S12 + S22 ) / (2(1 − c²))

with S the cross-moments of z. Drawing *c* from this grid is a proper Gibbs step.

It is worth pausing to note what identifies *c* here, because it is the same
thing that identifies the LDSC intercept: overlap noise is **flat in LD score**,
whereas genetic covariance **grows with LD score**. Our update is that
identification restated inside the sampler. This is reassuring — the method is
not conjuring information from nowhere — and it also predicts, correctly, that
the two approaches converge as the number of variants grows (§5).

## 4. How to benchmark it honestly

The tempting comparison is against a sampler that ignores overlap (c = 0). That
comparison is worthless: of course estimating a parameter beats pretending it is
zero. The comparison that matters is against **what bipred already recommends**,
namely the LDSC intercept.

We therefore run four arms which differ in *exactly one thing*, the value of *c*:

| arm | *c* | what it represents |
|---|---|---|
| `naive` | 0 | bipred's default when unspecified |
| `ldsc` | `ldsc_rg(...).gcov_intercept` | bipred's recommended practice today |
| `joint` | estimated each sweep | the proposal |
| `oracle` | the true simulated value | an upper bound, not attainable |

Same sampler, same data, same seeds. Whenever one is tempted to add a difference
between arms — a different number of sweeps, a fallback when a method fails — one
is no longer measuring what one claims to measure. (We violated this once; see
§6.)

## 5. The genome-wide answer: it works, and then it stops mattering

Below about 12,000 variants the in-sampler estimator is clearly better: it
recovers *c* with SD 0.012–0.025 against the intercept's 0.07–0.63, removes the
r_g inflation, and tracks the oracle. Both estimators are essentially *unbiased*;
the whole difference is variance.

But the LDSC intercept converges as 1/sqrt(m), and it does not stop:

| m | ldsc SD | joint SD | ratio |
|---:|---:|---:|---:|
| 1,500 | 0.625 | 0.0631 | 9.9× |
| 12,500 | 0.076 | 0.012 | 6.3× |
| 50,000 | **0.004** | 0.007 | 0.5× |
| 100,000 | **0.004** | 0.005 | 0.8× |

The crossover is near **m ≈ 50,000**. A real GWAS has ~10⁶ variants, comfortably
past it.

> **Conclusion 1.** For genome-wide r_g, this feature is not worth building. The
> existing `estimate_sample_overlap` route is adequate and cheaper.

That would have been the end of the matter, were it not for §7.

## 6. Two errors of our own, and their cost

Both were caught before publication — one by an adversarial review, one by a
direct check — and both are recorded because the failure modes are general.

**The frozen architecture.** Our first benchmark drew the causal effects *once*
and reused them across every replicate and every grid cell. Since the LDSC
intercept is a functional of the genetic architecture, each cell's LDSC figure
was then *one draw, not an estimate*, with an across-draw SD (0.073) larger than
the effect we were reporting from it. On that basis we asserted that LDSC
"systematically over-corrects" and "degrades as N grows." Both claims are false.
With effects redrawn per replicate the intercept is unbiased everywhere; what is
real is its *variance*.

**The under-swept grid.** Our first large-m grid used 400 sweeps and showed a
dramatic collapse at m ≥ 50,000 — in *every* arm, including the oracle. An arm
with no `cross_corr` uncertainty cannot degrade because of `cross_corr`, and that
is the tell. At m = 100,000 the oracle's bias is −0.246 at 400 sweeps and −0.040
at 1,200. It was our own convergence artifact.

> **Lesson 2.** Replicates must redraw everything that varies in reality. If a
> quantity is a functional of something you froze, you are reporting a draw.
>
> **Lesson 3.** Keep a control arm that *cannot* be affected by the thing you are
> studying. When it moves, the problem is yours, not the method's.

We also removed a silent `except: return 0.0` in the LDSC arm, which quietly
substituted the `naive` value on failure and reported it as an LDSC result — a
fallback that flattered our own method. Failures are now excluded and counted.

## 7. Regional genetic correlation inverts the argument

Local r_g — per LD block or locus — changes the picture entirely, for two
reasons that are really the same reason.

A region contains 10²–10³ variants **by construction**. That is the left-hand end
of the table in §5, where the intercept's SD is 0.6 and the in-sampler estimate's
is 0.06. So "just use the LDSC intercept per region" is not on the menu.

And overlap cannot be estimated *within* a region either — so regional r_g needs
a **genome-wide** *c*, which is precisely what the sampler now supplies. Worse,
overlap adds the *same* spurious covariance to *every* region, so it does not
cancel when regions are compared: it inflates them all together and confounds the
regional heterogeneity that is the entire object of the exercise.

The measurement is stark. With true regional r_g = 0 and c = 0.4:

| arm | null regions read | separation *d* |
|---|---:|---:|
| `naive` | **0.264** | 5.45 |
| `ldsc` | **0.515** | 2.99 |
| `joint` | **0.037** | **6.32** |
| `oracle` | 0.062 | 6.32 |

Ignoring overlap manufactures r_g ≈ 0.26 at *every null locus*. Feeding in a
noisy per-dataset intercept is worse than doing nothing, because the same wrong
value is applied to every region and therefore never averages away.

> **Conclusion 2.** For regional r_g, the in-sampler estimator is not an
> optimisation; it is a prerequisite.

## 8. But the model shrinks the regions together

Correcting overlap exposes a second, independent problem. bipred's sampler
carries **one** effect covariance Σ for the whole genome, so every per-SNP
posterior,

    V = (Σ^{-1} + E^{-1})^{-1} ,      mean = V E^{-1} d_j ,

borrows across traits in proportion to the *genome-wide* effect correlation. A
variant in a region whose true r_g is zero is nonetheless pulled toward the
global value. Reading regional r_g out of such a model measures a quantity that
has already been shrunk toward the mean.

Two observations make this concrete. First, the compression is visible in both
directions: null regions read 0.037 against a realised 0.003, and r_g = 0.8
regions read 0.725 against 0.796. Second — and this is the decisive check — the
**oracle** arm shows the same compression. The oracle is handed the exactly
correct *c*, so this cannot be an overlap artifact. Correcting *c* cures
contamination; it does nothing about shrinkage.

The bias is also **flat in region size** (−0.084, −0.071, −0.072 for regions of
200, 100 and 50 variants) while precision improves. That is what one expects of a
systematic pull applied identically to every variant in the region: averaging
more variants cancels noise, not an offset. One cannot escape it by using bigger
regions.

## 9. A second wrong turn: pooling the wrong object

The textbook remedy is partial pooling — give each region its own Σ_r with a
hierarchical prior centred on a genome-wide Ψ,

    Σ_r ~ InverseWishart(nu, (nu − 3) Ψ) ,

and estimate the pooling strength *nu*. We implemented this, estimating *nu* each
sweep from its exact conditional. It was worse than doing nothing: mean RMSE
0.100 against the global model's 0.070, with a −0.120 bias at null regions.

Our first instinct was that the *nu* estimator was at fault — it is fitted to the
observed dispersion of the Σ_r, which contains estimation noise as well as
genuine heterogeneity, and so is driven toward too little pooling. A reasonable
theory. We tested it by fixing *nu* on a grid, which removes the estimator from
the question entirely:

| nu | 4 | 20 | 50 | 100 | 300 | 1000 | ∞ (global) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean RMSE | 0.092 | 0.096 | 0.115 | 0.115 | 0.088 | 0.072 | **0.069** |

No pooling strength beats the global model, and the *worst* results are in the
middle. The theory was wrong; the problem is not how much we pool but **what** we
pool. Letting all three free parameters of Σ_r float per region opens a feedback
path on the covariance scale — a region's sampled scatter drives its own
posterior, which drives its scatter — and no prior strength closes it.

> **Lesson 4.** When a method fails, sweep the parameter you suspect before
> theorising about it. A monotone curve refutes a different hypothesis than a
> U-shaped one, and here the U was the answer.

## 10. Constraining the right thing

The heterogeneity we care about is in the *correlation*, not the scale. So hold
the per-trait variances at their global estimates and let only rho_r vary:

    Σ_r = D C_r D ,   D = diag(sigma_1, sigma_2) global,
                      C_r = [[1, rho_r], [rho_r, 1]] .

Each region now carries **one** bounded parameter instead of three, and — pleasingly —
its conditional has exactly the same closed form as the *c* update of §3. The
feedback path of §9 is closed by construction.

This is better than every other regional variant, and it removes the attenuation
it was meant to remove (r_g = 0.8 bias: −0.071 → +0.039; r_g = 0.4 bias: −0.004).
It gives the best separation at every region size. A residual negative bias at
null regions remains (≈ −0.083).

That residual led to the most interesting measurement of the whole exercise.
Running with **no overlap at all**:

| | null bias at c = 0.4 | null bias at c = 0 |
|---|---:|---:|
| `global` | +0.031 | **+0.104** |
| `rho` | −0.087 | **−0.027** |

Two things follow. The `rho` model's null bias is largely an *interaction with
the overlap correction* — remove overlap and it falls threefold. And the global
model's apparently mild +0.031 was **two errors cancelling**: an upward shrinkage
pull offset by a downward over-correction. In the clean case the global model is
four times worse. A comparison made at a single operating point had flattered it.

> **Lesson 5.** When two mechanisms push in opposite directions, a single
> operating point can show a small net error and hide both. Vary the mechanism
> you are not studying, and see whether your conclusion survives.

Having been wrong once about pooling (§9) we were careful to test, rather than
assume, the natural next move: partial pooling on rho *alone*, via a Fisher-z
prior of width tau centred on the genome-wide correlation. It does not help.

| tau | 0.15 | 0.3 | 0.6 | flat | (global) |
|---|---:|---:|---:|---:|---:|
| null bias | −0.108 | −0.101 | −0.091 | −0.085 | +0.032 |
| mean RMSE | 0.099 | 0.083 | 0.081 | 0.084 | **0.069** |
| `cross_corr` est. | 0.593 | 0.500 | 0.438 | **0.407** | 0.444 |

Pooling makes the null bias slightly *worse*, and tight pooling corrupts the
overlap estimate badly — 0.593 against a truth of 0.400 — because squeezing the
rho_r together forces the regional heterogeneity back into the nuisance
parameter, the same leak noted above in reverse. This is consistent with the
zero-overlap diagnostic: the residual null bias is an interaction with the
overlap correction, and pooling is simply not the lever that acts on it.

> **Lesson 6.** Two failures of the same shape are evidence about the diagnosis,
> not just the remedy. Pooling failed on Sigma and again on rho; the common
> factor is that neither addresses the mechanism the zero-overlap experiment
> actually implicated.

There is also a side benefit worth recording: under regional rho the *global*
estimate of *c* is more accurate (0.4045 against a truth of 0.400, versus 0.4407
for the global-Σ model). Heterogeneity the model cannot represent leaks into the
nuisance parameter.

## 11. Where this leaves us

What we believe, with measurements behind it:

1. *c* is identifiable inside the sampler, and the whitened residual is the
   statistic that identifies it (§3).
2. Genome-wide, this does not earn its keep past m ≈ 50,000 (§5).
3. Regionally it is a prerequisite, because uncorrected overlap fabricates
   correlation at every null locus and the per-region intercept is unusable (§7).
4. Regional inference has a *second*, independent problem — shrinkage toward the
   genome-wide correlation — which correcting *c* does not touch (§8).
5. Pooling the full Σ_r does not fix it and cannot be made to (§9); constraining
   the regions to share a scale and differ only in correlation does much better
   (§10).

What remains open:

- How to remove the residual null bias. It is an interaction between the global
  overlap correction and free regional rho (§10), and **not** a pooling problem:
  neither Sigma-pooling (§9) nor rho-pooling touches it. The next thing we would
  try is updating *c* and {rho_r} jointly rather than in sequence, since they are
  partially confounded — both add cross-trait covariance, and the sampler
  currently lets one absorb what the other should explain.
- Whether any of this survives bipred's four-state mixture, its int8 and low-rank
  LD representations — the whitening of §3 needs a per-block L^{-1}, which those
  representations do not carry — and realistic LD.
- Everything is simulated in-model, with overlap as the only source of
  cross-trait noise. Population stratification would produce similar covariance
  and is untested.

We would not build the production feature on the strength of §5. We would build
it on §7, and we would not ship regional r_g without resolving §8.
