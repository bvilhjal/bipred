# bipred

**Bivariate (two-trait) LDpred** — a joint LDpred model that fits two GWAS traits
sharing one LD reference at once. From the two sets of summary statistics it
estimates, in a single Gibbs sampler:

- each trait's **SNP heritability**,
- the **genetic correlation** `r_g` between them,
- the per-trait and **shared polygenicity** (a MiXeR-style polygenic-overlap
  summary), and
- posterior-mean effects for **prediction** — a well-powered trait sharpens a
  correlated under-powered one.

bipred is the bivariate half of [LDpred3](https://github.com/bvilhjal/ldpred3),
split into its own package. It builds on ldpred3 for the shared LD handling and
the Numba-accelerated sampler internals, so ldpred3 is a runtime dependency.

## The model

Each variant falls in one of **four** latent states with probabilities
`(π₀₀, π₁₀, π₀₁, π₁₁)`: causal for neither trait, trait 1 only, trait 2 only, or
**both**. A trait-1-causal effect is `N(0, s₁)`, a trait-2-causal one `N(0, s₂)`,
and a *both*-causal pair is drawn from `N(0, Σ)` with `Σ = [[s₁, s₁₂], [s₁₂, s₂]]`
— the off-diagonal `s₁₂` is the genetic covariance and is the only place the two
traits couple. Each Gibbs sweep evaluates the four bivariate-Gaussian
likelihoods of the residual marginal estimate, samples a state, then draws the
effects; `π` and `(s₁, s₂, s₁₂)` are re-estimated every sweep.

This **per-trait** indicator (rather than a single shared one) is what makes the
joint model safe: whether the two traits' causal variants co-occur is *learned*
(`π₁₁`), not assumed. Two genetically correlated traits that share causal
variants let the better-powered one sharpen the other; two traits with disjoint
causal variants drive `π₁₁ → 0`, so the joint fit reduces to the independent
ones and does no harm.

Both GWAS are assumed to use the **same** LD reference (same ancestry). Sample
overlap is passed via `cross_corr` (the cross-trait correlation of the sampling
noise, i.e. the bivariate-LDSC intercept); the default `0` assumes independent
GWAS samples.

## Installation

bipred depends on `ldpred3`. Until both are on PyPI, install ldpred3 from git
first (add the `[fast]` extra for the Numba-accelerated sampler — strongly
recommended):

```bash
pip install "ldpred3[fast] @ git+https://github.com/bvilhjal/ldpred3.git"
pip install "bipred[fast] @ git+https://github.com/bvilhjal/bipred.git"
```

Or, for local development of both side by side:

```bash
pip install -e ../ldpred3"[fast]"
pip install -e ."[fast,test]"
```

NumPy is the only hard dependency; Numba (`[fast]`) accelerates the Gibbs
sampler; msprime (`[sim]`) is only needed for some benchmark scripts.

## Quickstart

```python
import numpy as np
from bipred import ldpred3_auto_bivariate

# corr: a dense LD matrix (m x m); beta_hat1/2: standardized marginal effects for
# the two traits in the same variant order; n1/n2: per-trait GWAS sample sizes.
res = ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n1, n2)

print(res)                       # h2=(...), rg=..., p=..., n_variants=...
res.h2                           # (h2_trait1, h2_trait2)
res.rg                           # genetic correlation
res.beta1_est, res.beta2_est     # posterior-mean effects for prediction
res.mixer                        # MiXeR-style polygenic-overlap summary
```

For a genome-wide run, stream block by block with
`ldpred3_auto_bivariate_blocks(blocks, beta_hat1, beta_hat2, n1, n2, ...)`,
where `blocks` is the `[(R, idx), ...]` list of per-block LD (contiguous `idx`
partitioning `0..m-1`) used throughout ldpred3 — the two traits share it. Both
functions return a `BivariateResult`; see its docstring (and
[`docs/algorithm.md`](docs/algorithm.md)) for every field and option, including
`cross_corr` (sample overlap), the inverse-Wishart `Σ` shrinkage, and the two
`r_g` estimators.

## Genetic correlation and polygenic overlap

The reported `r_g` is an independent cross-check on **bivariate LDSC**
(`ldpred3.ldsc_rg`): under realistic reference-panel LD both are roughly
unbiased and the bivariate sampler is ~2× more precise, because it uses the full
LD likelihood rather than binned LD scores. See [`docs/rg.md`](docs/rg.md) for
the accuracy/timing comparison, choosing the right `r_g` estimator, handling
sample overlap, and the MiXeR-style overlap readout (`res.mixer` /
`res.mixer_calibrated`).

## Documentation

- [`docs/algorithm.md`](docs/algorithm.md) — the model, the four-state Gibbs
  sampler, the two `r_g` estimators, and the polygenic-overlap decomposition.
- [`docs/rg.md`](docs/rg.md) — genetic-correlation accuracy vs bivariate LDSC,
  sample-overlap corrections, and the MiXeR-style overlap parameters.
- `benchmarks/` — reproducible accuracy/timing benchmarks (many need msprime);
  see [`benchmarks/README.md`](benchmarks/README.md).

## License

MIT — see [LICENSE](LICENSE).
