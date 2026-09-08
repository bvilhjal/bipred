# Model and algorithm

bipred extends LDpred3-auto to two traits that share one LD reference. This page
defines the statistical model and estimators; see [`guide.md`](guide.md) for the
API and implementation options.

## Summary-statistic model

For trait `t`, standardized marginal effects follow the usual LDpred working
model. With scalar sample sizes:

**Equation (1). Per-trait summary-statistic model.**

```text
beta_hat_t = R beta_t + epsilon_t
Cov(epsilon_t) = R / N_t
Cov(epsilon_1, epsilon_2) = cross_corr R / sqrt(N_1 N_2)
```

`R` is the LD correlation matrix and `beta_t` is the vector of joint effects.
The LD correlation in the sampling noise is why a coordinate's conditional
variance is `1 / N`, not an assumption that marginal errors are independent.
At variant `j`, bipred uses the following conditional two-trait covariance:

**Equation (2). Per-variant sampling-noise covariance.**

```text
E_j = [[1 / N_1j,                         cross_corr / sqrt(N_1j N_2j)],
       [cross_corr / sqrt(N_1j N_2j),     1 / N_2j]]
```

`cross_corr=0` assumes uncorrelated cross-trait sampling errors. Non-overlapping
GWAS samples are sufficient, but not necessary, for that condition. Equation (2)
is the per-coordinate conditional covariance implied by Equation (1) for scalar
N. Supplying SNP-varying N is a working generalization of that likelihood.

The `1 / N` sampler variance is a weak-effect approximation. Cross-trait LDSC is
a separate estimator: it reconstructs exact signed z scores from
LDpred3-standardized effects as described in [`rg.md`](rg.md). This correction
does not change the Gibbs likelihood.

## Four-state effect prior

Each variant has one latent state.

**Table 1. Four-state effect prior.**

| State | Meaning | Effect prior |
|---|---|---|
| `00` | neither trait causal | `(0, 0)` |
| `10` | trait 1 only | `beta1 ~ N(0, s1)`, `beta2 = 0` |
| `01` | trait 2 only | `beta1 = 0`, `beta2 ~ N(0, s2)` |
| `11` | both traits causal | `(beta1, beta2) ~ N(0, Sigma)` |

Here `Sigma = [[s1, s12], [s12, s2]]` and
`pi = (pi00, pi10, pi01, pi11)`. Shared variants are learned through `pi11`;
they are not forced.

## Coherent initialization

`p_init` is the union probability that either trait is causal. Its shorthand
divides non-null mass among the three causal states, increasing the shared mass
only when required to represent `rg_init`. Supply `pi_init` when the overlap
itself matters. With union mass `q = p_init`, the shorthand is exact:

**Equation (8). Union-probability shorthand start.**

```text
shared = max(q / 3, |rg_init| q / (2 * 0.999 - |rg_init|))
shared = min(shared, q)
single = (q - shared) / 2
pi     = (1 - q, single, single, shared)
```

At modest `|rg_init|` the three causal states split evenly; at large
`|rg_init|` the shared mass grows just enough to keep the implied
within-shared effect correlation of Equation (4) at or below the 0.999
safety boundary, saturating at an all-shared start. The slab covariance is
then calibrated by Equation (4), so the implied starting heritabilities and
genetic correlation equal the requested values exactly.

For an explicit mixture:

**Equation (3). Marginal and shared causal probabilities.**

```text
p1 = pi10 + pi11
p2 = pi01 + pi11
u  = pi11
```

The initial slab covariance is calibrated as:

**Equation (4). Initial slab calibration.**

```text
s1       = h2_init_1 / (M p1)
s2       = h2_init_2 / (M p2)
rho_beta = rg_init sqrt(p1 p2) / u
s12      = rho_beta sqrt(s1 s2)
```

Here `M` is the modeled variant count. This calibration makes the implied
starting heritabilities and genetic correlation equal the requested values.
Invalid combinations requiring `|rho_beta| >= 1` are rejected.
`sigma_prior_scale` separates the persistent shrinkage target from the initial
state, which is important when comparing dispersed starts.

## Gibbs updates

For variant `j`, the sampler first removes current LD spillover from the
marginal effects, forming the leave-one-out residual `d = beta_hat - R @ beta
+ beta_j` per trait. With the noise covariance `E_j` of Equation (2) and the
slab `(s1, s2, s12)`, each non-null state implies a Gaussian posterior mean
`mu_s` for the variant's effect(s): a univariate conditional for the
trait-1-only and trait-2-only states, a bivariate conditional for the
both-traits state. State weights follow from Gaussian conditioning relative
to the null state:

**Equation (9). Four-state log weights.**

```text
g      = E_j^-1 @ d
log w_s = log pi_s - (ldet_s - ldet_0) / 2 + g' mu_s / 2
```

where `ldet_s` is the log determinant of the state's marginal covariance
(`E_j` plus the state prior covariance) and the null state's
`-0.5 (ldet_0 + d' E_j^-1 d)` term is common to all four states and cancels
in normalization. The sampler draws a state from the normalized weights,
draws the state's effect(s) from its Gaussian conditional (Cholesky draw for
the both-traits state), and updates the persistent `R @ beta` projections of
both traits by the change — one rank-two LD update per variant, never a full
matrix-vector product.

Effect reporting is Rao–Blackwellized, not sampled:

**Equation (10). Rao–Blackwell effect accumulation.**

```text
beta1_est[j] += P(state 10) E[beta1_j | state 10]
              + P(state 11) E[beta1_j | state 11]
```

and symmetrically for trait 2, averaged over retained sweeps. The
heritability quadratics instead use the *sampled* effects, which is why
`res.h2` can be mildly upward-biased at low power (see Genetic correlation).

After each sweep, the hyperparameters update from the sampled state counts
`(c00, c10, c01, c11)` and effect (co)moments `(S1, S2, S12)`:

**Equation (11). Mixture Dirichlet draw.**

```text
pi ~ Dirichlet(pi_prior + c00, pi_prior + c10,
               pi_prior + c01, pi_prior + c11)
```

**Equation (12). Damped slab-covariance update.**

```text
s1  <- 0.8 s1  + 0.2 (iw_df * psi1 + S1)  / (iw_df + c10 + c11)
s2  <- 0.8 s2  + 0.2 (iw_df * psi2 + S2)  / (iw_df + c01 + c11)
s12 <- 0.8 s12 + 0.2  S12 / (iw_df + c11)
```

This is inverse-Wishart-style shrinkage toward a weak diagonal prior with
`(psi1, psi2)` slab scales, `iw_df` pseudo-counts, and zero prior covariance:
marginal variances pool all trait-causal variants while the covariance is
pulled toward zero, keeping `s12` off the positive-definiteness boundary
(enforced explicitly at `|s12| <= 0.999 sqrt(s1 s2)`, with slab floors at
1e-12). `(psi1, psi2)` equal the coherently calibrated initial slab
variances unless `sigma_prior_scale` overrides them. An optional expert
`h2_cap` additionally clamps the implied per-trait heritability inside the
sampler. The covariance update is deterministic conditional on sampled
effects; it is not a conditional inverse-Wishart draw. Consequently,
intervals from `mixer_iterate_summary()` summarize retained iterates but are
not Bayesian credible intervals.

With `noise_inflation=True`, per-trait residual factors `lambda_t >= 1`
deflate the sample sizes used on the next sweep:

**Equation (5). Noise-inflation effective sample size.**

```text
N_eff,t = N_t / lambda_t
lambda_t <- (1 - ni_damp) lambda_t + ni_damp max(mean(n_t r_t^2), 1)
```

where `r_t = beta_hat_t - R @ beta_t` is the post-sweep residual: under
matched LD it is pure sampling noise (`mean(n r^2) ~ 1`) and inflated
otherwise. When `cross_corr` is non-zero, these deflated sample sizes also
enter the off-diagonal term in Equation (2).

**Algorithm 1. One bivariate Gibbs sweep with hyperparameter updates.**

```text
Input:  beta_hat1, beta_hat2, LD blocks, N1, N2 (scalar or per-variant),
        state (beta1, beta2, q1 = R @ beta1, q2 = R @ beta2, pi, s1, s2, s12,
        lambda1, lambda2); retain is true for a post-burn-in sweep.
Output: updated state; one Rao–Blackwell contribution per variant when retain.

1.  for each LD block, in fixed order, do
2.      for each variant j in the block, do
3.          d1, d2 <- leave-one-out residuals from (beta_hat, q, beta)
4.          E <- Equation (2) at (N1j / lambda1, N2j / lambda2, cross_corr)
5.          mu_s, ldet_s <- state conditionals for s in {00, 10, 01, 11}
6.          w_s <- Equation (9); P_s <- normalize(exp(w_s - max w))
7.          if retain: accumulate Equation (10) into the effect-mean sums
8.          draw S ~ Categorical(P); draw the state's effect(s)
9.          q1, q2 <- q1, q2 + R[:, j] (beta_new - beta_old) per trait
10.     end for
11. end for
12. accumulate same-sweep quadratics beta1'R beta1, beta1'R beta2, beta2'R beta2
13. lambda_t <- Equation (5) update from the sweep residuals (or hold at 1)
14. pi <- Equation (11) Dirichlet draw from the sweep state counts
15. (s1, s2, s12) <- Equation (12) damped update; apply floors, h2_cap, PD cap
16. return the updated state
```

Blocks stream genome-wide while `pi` and `Sigma` stay global: every sweep
waits for all blocks before step 14, so the genome-wide LD is never
materialized. Dense blocks sweep serially or block-parallel under `ncores`;
the per-sweep `R @ beta` buffer is rebuilt from scratch every 100 sweeps to
clear floating-point drift.

## Numerical constants

**Table 2. Hardcoded sampler constants.**

| Constant | Value | Meaning |
|---|---|---|
| damping | 0.2 | weight of the new moment term in Equation (12); not user-settable |
| `log pi` floor | 1e-300 | mixture log-weights never see an exact zero |
| slab floors | 1e-12 | `s1`, `s2` are clamped above this after Equation (12) |
| PD cap | 0.999 | `\|s12\| <= 0.999 sqrt(s1 s2)` after Equation (12) |
| `R @ beta` resync | 100 sweeps | buffers rebuilt from scratch to clear drift |
| structural-LD probe | ridge 0.05, ≤1024 variants | Cholesky check rejecting structurally indefinite LD |

The 0.2 damping is a fixed stability choice, not a tuning parameter: larger
values adapt faster but noisier across sweeps. Diagnostic thresholds (divergence
ratios, chain-filter rules, stopping tolerances) are documented in
[`diagnostics.md`](diagnostics.md), not here.

## Genetic correlation

The target is the LD-adjusted effect correlation:

**Equation (6). LD-adjusted genetic correlation.**

```text
r_g = beta1' R beta2 /
      sqrt((beta1' R beta1) (beta2' R beta2))
```

The default estimator averages sampled LD-aware quadratic forms. Its denominator
shares some posterior-noise inflation with the numerator, which is useful for
ordinary pairs but can attenuate a weak trait under strongly asymmetric power.
On non-positive-definite int8-quantized blocks a variance quadratic can come out
non-positive; the ratio then returns 0.0 rather than slamming to ±1 through the
floor. `rg_decorrelated=True` instead averages cross-sweep ordered-pair
quadratics:

**Equation (13). Decorrelated cross-sweep covariance.**

```text
cov12 = (quad(sum_t beta1(t), sum_t beta2(t)) - sum_t quad_t) / (n (n - 1))
```

where the sums run over the `n >= 2` thinned retained effect states and
`quad` is the LD-aware quadratic form — the quadratic of the sample sums
minus the same-sweep diagonals, over ordered cross-sweep pairs. Thinning
reduces, but does not prove the absence of, dependence between retained MCMC
states. Treat this as a sensitivity diagnostic only. In the committed
synthetic sweep the default estimator had lower paired realized-rg MAE under
both symmetric and asymmetric power (see
[`rg.md`](rg.md#asymmetric-power-sensitivity) for the numbers), so the
default is the recommended estimator; this option exists for sensitivity
analysis and is incompatible with multichain pooling and adaptive stopping.

`res.h2` reports the mean sampled quadratic `beta_t' R beta_t`, clamped to
`h2_bounds`. Because sampled rather than Rao–Blackwellized effects are used, it
can be mildly upward-biased at low power. The clamp applies to the reported
heritability only: `r_g` is the ratio of the unclamped quadratics in Equation (6),
so tightening `h2_bounds` does not rescale it. (`h2_cap` is different — it is
an in-sampler ceiling on implied per-trait heritability,
`s_t ≤ h2_cap_t / n_causal,t`, so it moves both.) A clamp that binds is reported through
the *implausible fit* warning on panels of at least 1,000 variants; below that
the warning is suppressed, and `res.h2` landing exactly on a bound, or the raw
`(gvar_1, gcov, gvar_2)` in `res.genetic_samples`, is the only signal.
These same-sweep quadratics are posterior genetic-variance and covariance
draws. They are not predictive-R2 draws: the latter requires a cross-product of
independent chains' effect draws, or direct evaluation against an independent
target phenotype.

Cross-trait LDSC (`bipred.ldsc_rg`) is a separate moment estimator and useful
screen. Its ratio can be unstable when either marginal LDSC heritability is near
zero. Estimator choice and sample overlap are covered in [`rg.md`](rg.md).

## Polygenic overlap

The four-state prior yields a MiXeR-style decomposition.

**Equation (7). Polygenic-overlap decomposition.**

```text
pi1             = pi10 + pi11
pi2             = pi01 + pi11
frac_shared     = pi11 / min(pi1, pi2)
rho_beta        = s12 / sqrt(s1 s2)
rg_from_overlap = rho_beta pi11 / sqrt(pi1 pi2)
```

`.mixer["rho_beta"]` uses the posterior-mean Sigma in that formula (ratio of
means). `mixer_iterate_summary()["rho_beta"]["mean"]` averages the same ratio
computed on each retained iterate (mean of ratios). They differ by Jensen's
inequality; neither is a bug.

Ratios such as `frac_shared`, `rho_beta`, and `rg_from_overlap` avoid the literal
causal-count interpretation, but still require calibration. A point-normal
mixture can spread inclusion mass to LD neighbours, and finite-reference
mismatch can add inflation. Noise inflation and univariate calibration are
sensitivity variants; the committed sweep found power-dependent gains and
losses. They do not turn the counts into identified causal-variant totals.

Anchoring counts on two univariate ldpred3 fits (`res.mixer_calibrated(infer1,
infer2)`) keeps the joint shared fraction and `rho_beta` and replaces only
the per-trait polygenicities:

**Equation (14). Univariate-anchored overlap.**

```text
pi11_cal = (pi11 / min(pi1, pi2)) * min(p_univ1, p_univ2)
```

with `p_univ` the univariate `p_est` values. The ratio in parentheses is the
joint `frac_shared` of Equation (7); the absolute counts inherit the
univariate scale instead.

## Prediction

`beta1_est` and `beta2_est` are posterior-mean effects. Borrowing can help when
one trait is weak and the other is well powered with genuine shared signal.
Prediction gains remain an empirical question and require out-of-sample
validation. Bipred does not infer observed out-of-sample R2 from summary
statistics. Nor does it currently retain the per-sweep effect vectors needed
for LDpred3's model-implied cross-chain predictive-R2 estimator. An R-hat over
all chain-pair products would be invalid because overlapping pairs are not
independent.
