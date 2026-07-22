# bipred

**bipred** is bivariate LDpred for two GWAS traits that share one LD reference.
It jointly estimates:

- SNP heritability for each trait,
- genetic correlation `r_g`,
- posterior-mean effects for prediction, and
- a MiXeR-style polygenic-overlap summary.

The package contains the bivariate pieces split out from
[ldpred3](https://github.com/bvilhjal/ldpred3). It still depends on ldpred3 for
shared LD utilities and sampler internals.

## Installation

Python 3.9-3.14 is supported. Numba is strongly recommended; Python 3.14 uses
Numba 0.66 or newer. Until ldpred3 is published, install the exact revision
tested by bipred, then install bipred:

```bash
python -m pip install "ldpred3[fast] @ git+https://github.com/bvilhjal/ldpred3.git@e0f6171a2635f87d60134b6d76bc8cdb7ab81119"
python -m pip install "bipred[fast] @ git+https://github.com/bvilhjal/bipred.git"
```

The package metadata requires ldpred3 0.2.13 or newer within the 0.2 series,
while the source install pins the exact tested commit because bipred currently
shares private sampler and LDSC helpers with ldpred3. That pin should be updated
deliberately when the seam changes. Blindly following a moving branch would be
exciting in all the wrong ways.

For a Conda environment on Linux, macOS, or Windows:

```bash
conda create -n bipred -c conda-forge python-gil=3.14 numpy numba pip
conda activate bipred
python -m pip install "ldpred3 @ git+https://github.com/bvilhjal/ldpred3.git@e0f6171a2635f87d60134b6d76bc8cdb7ab81119"
python -m pip install "bipred @ git+https://github.com/bvilhjal/bipred.git"
```

`python-gil` deliberately selects standard CPython. Conda may otherwise choose
the separate free-threaded `cp314t` build, which is not the CI compatibility
target.

For local development with sibling checkouts:

```bash
python -m pip install -e "../ldpred3[fast]"
python -m pip install -e ".[fast,test]"
```

`[sim]` installs only the `msprime` simulator. `[bench]` adds `msprime` and
Matplotlib for the self-contained benchmark scripts. The HAPNEST and cached-LD
benchmarks still need their documented external data.

## Quickstart

```python
from bipred import ldpred3_auto_bivariate

# corr: dense LD correlation matrix, shape (m, m)
# beta_hat1/2: standardized marginal effects in the same variant order
# n1/n2: scalar or per-variant effective sample sizes
res = ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n1, n2)

res.h2                       # (h2_trait1, h2_trait2)
res.rg                       # genetic correlation
res.beta1_est, res.beta2_est # posterior-mean effects for PRS scoring
res.mixer                    # polygenic-overlap summary
```

`res.sigma` is the mean of retained covariance iterates.
`res.mixer_iterate_summary()` returns empirical retained-chain intervals;
because the covariance uses a damped moment update rather than a conditional
posterior draw, these are not Bayesian credible intervals.
`res.mixer_posterior()` remains only as a deprecated compatibility alias.

For genome-wide runs, stream dense LD blocks:

```python
from bipred import ldpred3_auto_bivariate_blocks

res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n1, n2,
    burn_in=200, num_iter=200, seed=0,
)
```

`blocks` must be `[(R, idx), ...]` with dense LD matrices and contiguous indices
that partition `0..m-1`. The current bivariate sampler rejects ldpred3's compact
low-rank `LowRankLD` representation; pass dense float or dense int8 blocks.

By default the LD is stored **int8**-quantised (a quarter of the float32 memory;
the sampler dequantises on the fly), matching ldpred3's pipeline default. int8 blocks from
`ldpred3.compute_ld_blocks(quantize=True)` are used as-is; pass `ld_int8=False`
for an exact dense-float32 fit.

## Model In Brief

Each variant has one of four states:

- neither trait causal,
- trait 1 only,
- trait 2 only,
- both traits causal.

The sampler learns the state probabilities `pi` and the two-trait effect
covariance `Sigma`. The shared state drives cross-trait borrowing: a well-powered
trait can improve estimates for a correlated weaker trait. If the data do not
support shared causal variants, the model can drive the shared component down,
but prediction should still be validated out of sample.

Sample overlap is supplied with `cross_corr`, the cross-trait sampling-noise
correlation. The default `0` assumes independent GWAS samples.

## Genetic Correlation

`res.rg` is the recommended genetic-correlation estimate from the joint fit.
For a fast moment-based cross-check, use cross-trait LD Score regression:

```python
from bipred import ldsc_rg
from ldpred3 import ld_scores

ell = ld_scores(blocks)
rgr = ldsc_rg(beta_hat1, beta_hat2, ell, n1, n2)
rgr.rg, rgr.rg_se, rgr.gcov_intercept
```

Use `rg_decorrelated=True` for strongly asymmetric-power pairs, where one trait
is much better powered than the other.

## Polygenic Overlap

`res.mixer` reports per-trait polygenicity, shared causal count, shared fraction,
within-shared effect correlation, and the overlap decomposition of `r_g`.

Read the overlap ratios (`frac_shared`, `rho_beta`, `rg_from_overlap`) as the
most stable outputs. Absolute causal counts are approximate because LD can spread
posterior inclusion mass to nearby correlated variants. For count-sensitive work,
consider `noise_inflation=True` and/or `res.mixer_calibrated(infer1, infer2)`
using two univariate ldpred3 runs.

## Documentation

- [docs/guide.md](docs/guide.md): practical inputs, outputs, options, and pitfalls.
- [docs/algorithm.md](docs/algorithm.md): model and sampler reference.
- [docs/rg.md](docs/rg.md): genetic-correlation and overlap guidance.
- [benchmarks/README.md](benchmarks/README.md): reproducible benchmark scripts.

## License

MIT. See [LICENSE](LICENSE).
