# Model and algorithm

bipred extends LDpred3-auto to two traits that share one LD reference. This page
defines the statistical model and estimators; see [`guide.md`](guide.md) for the
API and implementation options.

## Summary-statistic model

For trait `t`, standardized marginal effects follow the usual LDpred working
model. With scalar sample sizes:

**Equation 1. Per-trait summary-statistic model.**

```text
beta_hat_t = R beta_t + epsilon_t
Cov(epsilon_t) = R / N_t
Cov(epsilon_1, epsilon_2) = cross_corr R / sqrt(N_1 N_2)
```

`R` is the LD correlation matrix and `beta_t` is the vector of joint effects.
The LD correlation in the sampling noise is why a coordinate's conditional
variance is `1 / N`, not an assumption that marginal errors are independent.
At variant `j`, bipred uses the following conditional two-trait covariance:

**Equation 2. Per-variant sampling-noise covariance.**

```text
E_j = [[1 / N_1j,                         cross_corr / sqrt(N_1j N_2j)],
       [cross_corr / sqrt(N_1j N_2j),     1 / N_2j]]
```

`cross_corr=0` assumes uncorrelated cross-trait sampling errors. Non-overlapping
GWAS samples are sufficient, but not necessary, for that condition. Equation 2
is the per-coordinate conditional covariance implied by Equation 1 for scalar
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
itself matters.

For an explicit mixture:

**Equation 3. Marginal and shared causal probabilities.**

```text
p1 = pi10 + pi11
p2 = pi01 + pi11
u  = pi11
```

The initial slab covariance is calibrated as:

**Equation 4. Initial slab calibration.**

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

For each sweep and variant, the sampler:

1. removes current LD spillover from the marginal effects;
2. evaluates the four state likelihoods;
3. samples a state and its effect or effects; and
4. updates the persistent `R @ beta` projection.

After a sweep:

- `pi` is drawn from a Dirichlet posterior with symmetric concentration
  `pi_prior`;
- `s1`, `s2`, and `s12` receive a damped moment update, shrunk toward a diagonal
  target controlled by `iw_df`; and
- optional residual-noise factors `lambda_t >= 1` update the effective sample
  sizes.

**Equation 5. Noise-inflation effective sample size.**

```text
N_eff,t = N_t / lambda_t
```

When `cross_corr` is non-zero, these deflated sample sizes also enter the
off-diagonal term in Equation 2. The covariance update is deterministic
conditional on sampled effects; it is not a conditional inverse-Wishart draw.
Consequently, intervals from `mixer_iterate_summary()` summarize retained
iterates but are not Bayesian credible intervals.

## Genetic correlation

The target is the LD-adjusted effect correlation:

**Equation 6. LD-adjusted genetic correlation.**

```text
r_g = beta1' R beta2 /
      sqrt((beta1' R beta1) (beta2' R beta2))
```

The default estimator averages sampled LD-aware quadratic forms. Its denominator
shares some posterior-noise inflation with the numerator, which is useful for
ordinary pairs but can attenuate a weak trait under strongly asymmetric power.
`rg_decorrelated=True` instead averages cross-sweep quadratics, excluding
same-sweep pairs. Thinning reduces, but does not prove the absence of, dependence
between retained MCMC states. Treat this as a sensitivity diagnostic only. In
the committed synthetic sweep the default estimator had lower paired
realized-rg MAE under both symmetric (0.0086 vs 0.0108) and asymmetric
(0.0174 vs 0.0242) power, so the default is the recommended estimator; this
option exists for sensitivity analysis and is incompatible with multichain
pooling and adaptive stopping.

`res.h2` reports the mean sampled quadratic `beta_t' R beta_t`, clamped to
`h2_bounds`. Because sampled rather than Rao–Blackwellized effects are used, it
can be mildly upward-biased at low power. The clamp applies to the reported
heritability only: `r_g` is the ratio of the unclamped quadratics in Equation 6,
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

**Equation 7. Polygenic-overlap decomposition.**

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

## Prediction

`beta1_est` and `beta2_est` are posterior-mean effects. Borrowing can help when
one trait is weak and the other is well powered with genuine shared signal.
Prediction gains remain an empirical question and require out-of-sample
validation. Bipred does not infer observed out-of-sample R2 from summary
statistics. Nor does it currently retain the per-sweep effect vectors needed
for LDpred3's model-implied cross-chain predictive-R2 estimator. An R-hat over
all chain-pair products would be invalid because overlapping pairs are not
independent.
