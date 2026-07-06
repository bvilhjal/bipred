# Algorithm and model

bipred extends LDpred3-auto to two traits sharing one LD reference. It uses the
same summary-statistic model as univariate LDpred:

```text
beta_hat_t = R beta_t + error_t
```

where `R` is the LD correlation matrix and `error_t` has variance determined by
the effective sample size. The bivariate extension models two effect vectors and
their cross-trait covariance.

## Four-state effect model

Each variant belongs to one latent state:

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

## Gibbs sampler

For each sweep and SNP, the sampler:

1. forms the residual marginal estimates after subtracting current LD spillover,
2. evaluates the four bivariate Gaussian state likelihoods,
3. samples the state,
4. draws the effect(s) under that state, and
5. updates `R @ beta` incrementally.

After each sweep, the global mixture and covariance parameters are updated:

- `pi` is drawn from a Dirichlet posterior with concentration `pi_prior`,
- `s1`, `s2`, and `s12` are updated from sampled effects,
- the covariance update is shrunk toward a weak diagonal prior controlled by
  `iw_df`, and
- optional `noise_inflation=True` learns per-trait residual noise factors.

The shrinkage keeps `Sigma` positive definite and avoids the older ad hoc
heritability pre-pass. `h2_cap` remains available as an expert clamp.

## Genetic Correlation

The target is:

```text
r_g = beta1' R beta2 / sqrt((beta1' R beta1) * (beta2' R beta2))
```

The default estimate uses sampled quadratic forms from the joint chain. This is
the recommended estimator for most pairs because numerator and denominator are
formed on the same shrinkage scale.

Set `rg_decorrelated=True` for asymmetric-power pairs. It estimates the
cross-trait covariance from effects sampled at different sweeps, reducing
same-sweep noise coupling that can attenuate the weak trait's covariance.

Cross-trait LDSC (`bipred.ldsc_rg`) is a separate moment-based estimator. It is
fast and useful as a screen or cross-check, but it is noisier when marginal LDSC
heritability estimates are unstable.

## Polygenic Overlap

The four-state mixture gives a MiXeR-style decomposition:

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

`res.mixer_posterior()` summarizes posterior uncertainty conditional on the
supplied LD reference. It does not capture bias from LD-reference mismatch.

## Prediction

The returned `beta1_est` and `beta2_est` are posterior-mean effects. Cross-trait
borrowing helps most when one trait is weak, the other is strong, and the genetic
correlation is high. With little shared signal, the model should mostly decouple
the traits, but prediction gains still need out-of-sample validation.

## Dense-block implementation

`ldpred3_auto_bivariate_blocks` streams dense LD blocks and pools global
hyperparameters across them. The full genome-wide LD matrix is never
materialized, but each block must currently be dense. Compact `LowRankLD` blocks
are rejected by design until the bivariate kernel supports that representation.

By default the LD is stored **int8**-quantised (`round(clip(R, -1, 1) * 127)`,
scale `1/127`) — a quarter of the float32 memory, matching ldpred3's default
representation. The sampler dequantises each entry on the fly in the
bandwidth-bound inner loop (`corr[i, j] * scale`); the unit diagonal quantises
exactly (`127/127 == 1`), which the residual update `d = beta_hat − R·beta + beta`
relies on. The quantisation error is negligible (≈`0.5/127 ≈ 0.004` per entry,
diagonal exact). int8 blocks from `ldpred3.compute_ld_blocks(quantize=True)` are
consumed as-is; pass `ld_int8=False` for an exact dense-float32 fit.
