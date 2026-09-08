# Fit-validity and convergence diagnostics

Two different warnings, one scalar diagnostic, one stopping rule. None of them
certifies convergence or correctness; each names a specific failure it can see.
The sampler itself is defined in [`algorithm.md`](algorithm.md); estimator
choice is in [`rg.md`](rg.md).

## Two warnings

A fit can diverge while `h2` and the causal fraction look ordinary. The
motivating failure was a real LDL x CAD fit on a public GWAS: it reported
`h2` 0.64 with a sparse causal fraction, touched no bound, and completed
without a warning — while posterior means reached 3.33 against the 0.030 slab
SD the fit itself inferred, cancelling through near-perfect LD so the
quadratic form stayed plausible. The two warnings below catch different
failures and do not overlap: that runaway tripped only the second.

**Implausible fit.** Fires on a sizeable panel (at least 1,000 variants) when
any of these hold: the fitted causal fraction exceeds one half, a raw sampled
variance reaches the upper `h2_bounds` clamp, is non-positive (degenerate —
this also zeroes reported `rg`), or falls to the lower clamp (reported `h2`
is then a clamped value, while `rg` from the raw quadratics stands). Below
1,000 variants the heuristic carries no information and stays quiet; there,
an `h2` exactly on a bound, or the raw `(gvar_1, gcov, gvar_2)` in
`res.genetic_samples`, is the only signal. Do not interpret the estimates
until LD quality, reference size, block size, and regularization have been
inspected — besides being suspect, such fits are markedly slower, because the
guarded per-variant LD row update then fires for nearly every variant.

**Diverged fit.** Compares the fitted effects against the fitted model after
the run, with three ratios that need no second pass over the LD:

**Table 1. Divergence ratios.**

| Ratio | Threshold | Diverged example | Healthy example |
|---|---|---|---|
| `sum(beta^2)` vs raw genetic variance | 10 | 246 / 92 | 0.31 / 0.17 |
| `max\|beta\|` vs per-causal slab SD | 25 | 110 | 4.3 |
| retained-trace drift, first vs last quarter | 1.25 | 1.60 | ~1.0 |

A posterior mean is a shrunk quantity, so tens of slab SDs is the model
contradicting its own prior; effects that cancel through LD inflate the first
ratio while leaving the quadratic plausible; systematic post-burn-in drift
means the chain never settled. A non-positive raw genetic variance also flags.
Each threshold separates the diverged and repaired LDL x CAD regimes by more
than an order of magnitude. The check is silent below 1,000 variants or 40
retained trace iterations. A flagged fit raises a `RuntimeWarning` naming the
check, and `write_weights` refuses it unless `allow_diverged=True` (CLI:
`--allow-diverged`).

`res.divergence_diagnostics` records the full structure: per-trait ratios,
flags, trace moments and directions, the thresholds above, variant and trace
counts, and whether the check was evaluated at all. These are fit-validity
heuristics, not R-hat, ESS, or a convergence certificate.

## Multi-chain pooling and split-Rhat

The multi-chain driver disperses starts over union-causal fractions, pools
every finite equal-length chain with equal weight, and aborts on a failed or
wrong-length chain (serially, later chains never run). Pooling records
`retained_iterations = n_chains * retained_per_chain` and
`stopped_early=False`; there is no adaptive stopping here.

`basic_split_rhat` is the classical scalar split-Rhat delegated from
`ldpred3.diagnostics`, with degeneracy flags (identical constant chains have
no scale and return NaN; different constants return infinity). It runs over
these retained traces:

`gvar_1`, `gvar_2`, clipped `h2_1`, `h2_2`, `gcov`, `rg` from the *raw*
quadratics, `p`, `pi00`, `pi10`, `pi01`, `pi11`, `sigma1`, `sigma2`,
`sigma12`, clipped `rho_beta` — plus `noise_scale1/2` when noise inflation
is on. Heritabilities enter clipped to `h2_bounds` but `rg` deliberately
uses raw quadratics: clamping the denominator alone would saturate the ratio
at ±1 whenever a bound binds, faking perfect between-chain agreement.

Per-chain divergence diagnostics pool alongside: the pooled
`divergence_diagnostics` reports whether any chain was evaluated or flagged,
how many of each, and keeps every chain's dict, so one diverged chain cannot
vanish into the pool. R-hat itself never drops a chain.

## Adaptive stopping (single chain only)

With `tol > 0`, the running posterior means snapshot every `check_every`
retained sweeps (after two snapshots exist, so the first test compares like
with like). The run stops when the relative RMS change of *both* traits'
means and the change in running `rg` all clear `tol`:

```text
||mean - prev||^2 <= tol^2 ||mean||^2   (per trait)
|rg - prev_rg| <= tol
```

`rg` is included because it converges on its own timescale: the effect
vectors can settle while the genetic covariance still drifts. This is a
schedule- and seed-dependent stabilization heuristic, not convergence
evidence. It is disabled for `rg_decorrelated=True` (which needs the full
thinned schedule) and unsupported by multi-chain inference; `stopped_early`
and `retained_iterations` record what actually ran.
