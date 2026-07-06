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

Until both packages are on PyPI, install ldpred3 first:

```bash
pip install "ldpred3[fast] @ git+https://github.com/bvilhjal/ldpred3.git"
pip install "bipred[fast] @ git+https://github.com/bvilhjal/bipred.git"
```

For local development:

```bash
pip install -e ../ldpred3"[fast]"
pip install -e ."[fast,test]"
```

`[fast]` installs Numba support and is strongly recommended. `msprime` is only
needed for benchmark scripts.

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

For genome-wide runs, stream dense LD blocks:

```python
from bipred import ldpred3_auto_bivariate_blocks

res = ldpred3_auto_bivariate_blocks(
    blocks, beta_hat1, beta_hat2, n1, n2,
    burn_in=200, num_iter=200, seed=0,
)
```

`blocks` must be `[(R, idx), ...]` with dense LD matrices and contiguous indices
that partition `0..m-1`. The current bivariate sampler rejects ldpred3 low-rank
`LowRankLD` blocks.

By default the LD is stored **int8**-quantised (a quarter of the float32 memory;
the sampler dequantises on the fly), matching ldpred3's default. int8 blocks from
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
