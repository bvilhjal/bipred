# Overlap correction and regional genetic correlation

**Status:** `bipred.regional_rg` is now a public exploratory API. The
in-sampler estimator of `cross_corr` and the per-region covariance models in
this directory remain research prototypes and are not used by the package.

## Start here

**Table 1. Research documents.**

| Goal | Document |
|---|---|
| derivations, failed approaches, and conclusions | [`NOTES.md`](NOTES.md) |
| genome-wide `cross_corr` experiments | [`RESULTS.md`](RESULTS.md) |
| evidence behind public `regional_rg` caveats | [`RESULTS_REGIONAL.md`](RESULTS_REGIONAL.md) |

## Conclusions

`cross_corr` is the correlation of GWAS sampling noise induced by overlapping
samples. It reflects the phenotypic correlation among shared individuals; it is
not the environmental correlation alone.

- Genome-wide joint estimation was not worth integrating. In these simulations
  it helped below roughly 50,000 variants, while the LDSC intercept caught up at
  larger `m`, where real GWAS normally operate.
- Regional estimates cannot identify overlap locally. Users must supply a
  defensible genome-wide `cross_corr` to the fit before calling `regional_rg`.
  Leaving overlap uncorrected contaminated every simulated region.
- Public `regional_rg` deliberately reads regional quadratics from
  posterior-mean effects. It supports dense, int8, and low-rank LD, but estimates
  remain shrunk toward the genome-wide correlation and are intended mainly for
  ranking and comparison.
- Experimental per-region covariance models did not resolve calibration well
  enough to ship.

## Files and commands

**Table 2. Research artifacts.**

| File | Purpose |
|---|---|
| `estimate_cross_corr_prototype.py` | minimal joint-estimation prototype |
| `bench_cross_corr.py` | genome-wide grids: `main`, `n`, `m`, `ldwide`, `scale` |
| `bench_regional_rg.py` | evidence used to characterize `regional_rg` |
| `bench_hier_sigma.py` | unshipped per-region covariance experiments |
| `bench_*.csv`, `results.csv` | committed raw outputs |

```bash
OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/bench_cross_corr.py all 20
OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/bench_regional_rg.py all 10
OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/bench_hier_sigma.py main 10
```

The scripts are seeded and use Numba where useful, with NumPy fallbacks.

The earliest prototype redraws sampling noise but reuses one causal-effect draw
across its five replicates and reports population SD (`ddof=0`). Its dispersion
is therefore understated. Later benchmarks redraw effects and noise per
replicate; keep the prototype only as a short derivation.

## What remains unshipped

Integrating the joint `cross_corr` update would still require:

1. whitening for dense, int8, and low-rank LD;
2. validation in the production four-state mixture and realistic LD;
3. opt-in API, result metadata, seeded threading behavior, and tests; and
4. evidence that it improves prediction and heritability, not only `r_g`.

The regional readout itself has shipped; its known biases are part of the public
contract rather than solved by the unshipped joint estimator.
