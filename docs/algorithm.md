# Algorithm and model

bipred extends LDpred3-auto to two traits sharing one LD reference. It uses the
same summary-statistic model as univariate LDpred:

**Equation 1. Per-trait summary-statistic model.**

```text
beta_hat_t = R beta_t + error_t
```

where `R` is the LD correlation matrix. The sampling errors have
`Var(error_tj) = 1/N_t` per trait and cross-trait covariance
`Cov(error_1j, error_2j) = cross_corr / sqrt(N_1 N_2)` from correlated sampling
noise (sample overlap). The bivariate extension models two effect vectors and
their cross-trait covariance. The per-SNP conditional update treats the noise
as independent across variants with these per-trait variances (the usual
LDpred-style diagonal approximation), keeping only the cross-trait correlation
`cross_corr` at each variant.

## Four-state effect model

Each variant belongs to one latent state:

**Table 1. Four-state effect prior.**

| state | meaning | effect prior |
|---|---|---|
| `00` | neither trait causal | `(0, 0)` |
| `10` | trait 1 only | `beta1 ~ N(0, s1)`, `beta2 = 0` |
| `01` | trait 2 only | `beta1 = 0`, `beta2 ~ N(0, s2)` |
| `11` | both traits causal | `(beta1, beta2) ~ N(0, Sigma)` |

`Sigma = [[s1, s12], [s12, s2]]`; `s12` is the effect covariance within the
shared component. The mixture probabilities are
`pi = (pi00, pi10, pi01, pi11)`.

This per-trait state structure is the important design choice. Shared causal
variants are learned through `pi11`; they are not forced. When the data support
little overlap, the shared state can shrink and the two fits largely decouple.

## Coherent initialization

`h2_init`, `p_init`, and `rg_init` are calibrated jointly. The scalar `p_init`
is the initial union probability `P(trait 1 or trait 2 causal)`; by default its
non-null mass is divided equally among `10`, `01`, and `11`. For large
`|rg_init|`, the shorthand increases the initial shared mass just enough to keep
the within-shared covariance positive definite; above the sampler's 0.999
boundary the shared mass saturates at the union probability (an all-shared
start), which keeps the implied moments exact for any `rg_init` in `(-1, 1)`.
Supply `pi_init` to specify the overlap directly.

For an explicit `pi_init`, define marginal causal probabilities as follows:

**Equation 2. Initial marginal and shared causal probabilities.**

```text
p1 = pi10 + pi11
p2 = pi01 + pi11
u  = pi11
```

The slab covariance is then calibrated rather than guessed:

**Equation 3. Initial slab calibration.**

```text
s1 = h2_init_1 / (M * p1)
s2 = h2_init_2 / (M * p2)
rho_beta = rg_init * sqrt(p1 * p2) / u
s12 = rho_beta * sqrt(s1 * s2)
```

This makes the implied genetic moments equal the documented starting values:
`M*p1*s1 = h2_init_1`, `M*p2*s2 = h2_init_2`, and
`M*u*s12 = rg_init*sqrt(h2_init_1*h2_init_2)`. A requested combination is
rejected when it would require `|rho_beta| >= 1`. `sigma_prior_scale` optionally
sets the persistent diagonal shrinkage target separately from this starting
covariance; this separation is required when comparing chains with different
starts but the same prior.

## Sequential multi-chain inference

`ldpred3_auto_bivariate_chains` runs chains one at a time, using deterministic
child seeds. Its default four union-causal starts are log-spaced from `1e-4` to
`0.2`; explicit four-state `pi_inits` are the alternative. Every chain uses
the same covariance-prior scale, derived once from `prior_p_init=0.02` unless
set explicitly.

Each retained sweep records raw genetic quadratics `(gvar_1, gcov, gvar_2)` and
the two noise scales. The driver pools all finite, equal-length traces and
posterior effects with equal weight. A non-finite or unequal chain is a hard
failure: there is no estimate-based chain filtering. Classical basic split-Rhat
is reported for scalar genetic, mixture, covariance, and—when enabled—noise-scale
traces, with explicit degeneracy flags. It does not diagnose variant-level
effects. It is diagnostic metadata, not a convergence decision, and no
`converged` flag is produced.
`rg_decorrelated=True` is unsupported because that estimator requires a
different cross-chain trace contract. Chains remain sequential; `ncores`
parallelises independent blocks within one chain, not chains themselves.

## Sampler and hyperparameter updates

For each sweep and SNP, the sampler:

1. forms the residual marginal estimates after subtracting current LD spillover,
2. evaluates the four bivariate Gaussian state likelihoods,
3. samples the state,
4. draws the effect(s) under that state, and
5. updates `R @ beta` incrementally.

After each sweep, the global mixture and covariance parameters are updated:

- `pi` is drawn from a Dirichlet posterior with prior concentration `pi_prior`
  plus the sweep's state counts,
- `s1`, `s2`, and `s12` receive a damped moment update from the sampled effects,
  shrunk toward a weak diagonal target controlled by `iw_df`, and
- optional `noise_inflation=True` learns per-trait residual noise factors
  `lambda_t >= 1` and fits with effective sample sizes `N_t / lambda_t`.

When `noise_inflation` is combined with `cross_corr`, the deflated sample sizes
also enter the cross-trait noise covariance `E12 = cross_corr / sqrt(N_1 N_2)`,
so the learned excess noise is treated as cross-correlated at `cross_corr`.
Keep this in mind when interpreting `lambda` under sample overlap.

The covariance update is deterministic conditional on the sampled effects; it
is not a conditional inverse-Wishart draw. The shrinkage keeps `Sigma` positive
definite and avoids the older ad hoc heritability pre-pass. `res.sigma` is the
mean of the retained covariance iterates, while `res.pi` is the mean of the
retained Dirichlet draws. `h2_cap` remains available as an expert clamp.

## Genetic Correlation

The target is:

**Equation 4. LD-adjusted genetic correlation.**

```text
r_g = beta1' R beta2 / sqrt((beta1' R beta1) * (beta2' R beta2))
```

The default estimate uses sampled quadratic forms from the joint chain. This is
the recommended estimator for most pairs: the posterior-noise inflation the
sampling adds to each quadratic partially cancels between numerator and
denominator.

`res.h2` reports the denominators of that ratio: the mean sampled LD-adjusted
quadratics `beta_t' R beta_t`, clamped to `h2_bounds`. Because they are formed
from sampled (not Rao-Blackwellized) effects, they include posterior-noise
variance and are mildly upward-biased at low power.

Set `rg_decorrelated=True` for asymmetric-power pairs. It estimates the
cross-trait covariance from effects sampled at different sweeps, removing
the same-sweep noise coupling that inflates the genetic covariance, and recovers
the weak trait's covariance from its posterior mean — the sampled-quadratic
ratio attenuates it through the weak trait's inflated sampled variance.

Cross-trait LDSC (`bipred.ldsc_rg`) is a separate moment-based estimator. It is
fast and useful as a screen or cross-check, but it is noisier when marginal LDSC
heritability estimates are unstable.

## Polygenic Overlap

The four-state mixture gives a MiXeR-style decomposition:

**Equation 5. Polygenic-overlap decomposition.**

```text
pi1 = pi10 + pi11
pi2 = pi01 + pi11
rho_beta = s12 / sqrt(s1 * s2)
r_g ≈ rho_beta * pi11 / sqrt(pi1 * pi2)
```

`res.mixer` reports:

- per-trait polygenicity,
- per-trait causal counts,
- shared causal count,
- shared fraction,
- `rho_beta`, and
- `rg_from_overlap`.

The ratios are the safer quantities. Absolute counts can be inflated because the
point-normal mixture assigns partial causal probability to LD neighbours. The
inflation is usually mild with the current `p_init=0.02` default, but it grows
with power and reference-panel mismatch.

Mitigations:

- use `noise_inflation=True` for finite reference-panel LD,
- use `res.mixer_calibrated(infer1, infer2)` to anchor counts on two univariate
  ldpred3 fits, and
- validate count calibration with simulations when counts are a primary result.

`res.mixer_iterate_summary()` reports empirical means and quantile intervals
across retained `pi` draws and `Sigma` iterates. These are retained-chain
intervals, not Bayesian credible intervals, because `Sigma` is updated by a
damped moment step rather than sampled from its conditional posterior. They also
do not capture bias from LD-reference mismatch. `res.mixer_posterior()` is a
deprecated compatibility alias.

## Prediction

The returned `beta1_est` and `beta2_est` are posterior-mean effects. Cross-trait
borrowing helps most when one trait is weak, the other is strong, and the genetic
correlation is high. With little shared signal, the model should mostly decouple
the traits, but prediction gains still need out-of-sample validation.

## Block representations

`ldpred3_auto_bivariate_blocks` streams dense or compact low-rank LD blocks and
pools global hyperparameters across them. The full genome-wide LD matrix is
never materialized, and dense and `LowRankLD` blocks may be mixed.

For a low-rank factor with `k` variants and rank `r`, bipred uses ldpred3's
effective correlation semantics:

**Equation 6. Effective low-rank LD and persistent score.**

```text
W = diag(row_scales) U
R_eff = W W' + diag(d)
s_t = W' beta_t
R_eff beta_t = W s_t + d * beta_t
```

The sampler keeps `U` compact and maintains one rank-length score `s_t` per
trait. Each SNP update is `O(r)`, a block sweep is `O(k*r)`, and storage is
`O(k*r)` rather than dense `O(k^2)`. In newer ldpred3 representations, `d`
stores diagonal mass discarded by truncation or quantisation; LR8 retains one
global factor scale rather than changing individual-row geometry. Released
row-normalised factors are handled backward-compatibly with `d = 0`. LR8
factors remain int8 and float factors are canonicalised to float32. The
effective LD is the approximation encoded by the factor and diagonal residual,
so its accuracy still depends on the construction rank or retained variance.

With Numba and `ncores>1`, homogeneous dense blocks sharing a dtype and scale,
or homogeneous low-rank factors sharing a factor dtype, enter one fused
block-parallel sweep. Random arrays are generated in sorted genome order before
workers start. Each worker owns one block's effects and persistent projections;
the three integer counts and six floating statistics are then reduced in the
original block order. Consequently seeded `ncores=1` and `ncores>1` fits are
array-identical. Mixed representations or dtypes retain the serial path. The
Numba workers are threads in one process and persist across sweeps, but every
sweep ends at a required barrier before the global mixture, covariance, and
optional noise-scale updates. The slowest block therefore determines the
parallel portion of a sweep; speed-up depends on the number and workload balance
of the blocks and is not expected to scale linearly with `ncores`.

With the default `ld_int8=None`, supplied dense int8 blocks are consumed as-is;
float blocks with at most 1500 variants are stored as
`round(clip(R, -1, 1) * 127)` int8, while larger float blocks remain float32.
The small-block D8 payload uses a quarter of the float32 memory and is
dequantised in the bandwidth-bound inner loop (`corr[i, j] / 127`); its unit
diagonal remains exact. Keeping large dense blocks float32 avoids magnifying the
small entrywise rounding error through poor conditioning. `ld_int8=True`
quantises every dense float block and `False` keeps every dense float input
float32. It does not change `LowRankLD` factors.
