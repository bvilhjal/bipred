# Overlap correction and regional genetic correlation

**Status: research. Nothing here is shipped.** This directory imports almost
nothing from `bipred` (only `ldsc_rg`, to build a competing baseline) and adds no
public API. It exists to decide whether two features are worth building, and to
leave a record adequate for someone else to check the reasoning.

## Start here

| if you want | read |
|---|---|
| the story, with the derivations and the dead ends | **[`NOTES.md`](NOTES.md)** |
| genome-wide numbers | [`RESULTS.md`](RESULTS.md) |
| regional numbers | [`RESULTS_REGIONAL.md`](RESULTS_REGIONAL.md) |

## The question, and the answer

`cross_corr` is the correlation of the GWAS *sampling noise* induced by
overlapping samples. bipred takes it as a fixed user input, usually derived from
the cross-trait LDSC intercept. Can the Gibbs sampler estimate it instead?

Yes — and the useful part of the answer is *where it matters*:

- **Genome-wide: not worth building.** The in-sampler estimator wins below
  m ≈ 50,000 variants, but the LDSC intercept converges as `1/sqrt(m)` and is
  equal or better past that. A real GWAS is comfortably past it.
- **Regionally: a prerequisite.** A region has 10²–10³ variants by construction,
  where the intercept is unusable, and overlap cannot be estimated within a
  region at all. Left uncorrected it manufactures r_g ≈ 0.26 in regions whose
  true r_g is **zero** — at every null locus, since the same spurious covariance
  is added to all of them.
- **A second problem sits behind it.** Regional estimates are also shrunk toward
  the genome-wide correlation, because the sampler carries one effect covariance
  for the whole genome. Correcting `cross_corr` does not touch this. Pooling the
  full per-region covariance fails outright; constraining regions to share a
  scale and differ only in correlation does much better. See `NOTES.md` §8–§10.

A clarification, since the name misleads: `cross_corr` is **not** the
environmental correlation `r_e`. It is a *noise* correlation, reflecting the
phenotypic correlation among shared individuals — genetic and environmental
combined. Summary statistics do not separate them.

## Files

| file | what it is |
|---|---|
| `NOTES.md` | the narrative account, including two errors of ours and what they cost |
| `RESULTS.md`, `RESULTS_REGIONAL.md` | measurements, with limitations |
| `estimate_cross_corr_prototype.py` | the original minimal proof of concept (see caveat below) |
| `bench_cross_corr.py` | genome-wide benchmark; grids `main`, `n`, `m`, `ldwide`, `scale` |
| `bench_regional_rg.py` | regional r_g benchmark; grids `main`, `size` |
| `bench_hier_sigma.py` | per-region covariance models: `global`, `perregion`, `hier`, `rho`, fixed-`nu` sweep |
| `bench_*.csv`, `results.csv` | committed outputs; every figure quoted in the docs re-derives from these |

Each script is runnable standalone and documents its own design in its module
docstring:

```bash
OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/bench_cross_corr.py all 20
OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/bench_regional_rg.py all 10
OPENBLAS_NUM_THREADS=1 python research/cross_corr_estimation/bench_hier_sigma.py main 10
```

Numba-accelerated where it matters, with a pure-NumPy fallback; all seeded.

> **Caveat on `estimate_cross_corr_prototype.py` and `results.csv`.** The
> prototype draws the causal effects *once* and redraws only the sampling noise
> across its five replicates, and reports `np.std` with `ddof=0`. Its ± figures
> therefore understate replicate-to-replicate variability — the same defect
> described in `NOTES.md` §6, which is why that section exists. The *direction*
> of its conclusion is confirmed by the properly replicated benchmarks; its
> dispersions are not. It is kept because it is the shortest readable derivation
> of the method.

## What a production feature would still require

1. **Whitening for every LD representation.** The update needs a per-block
   `L^{-1}`. Dense LD has one; bipred's default int8 and its low-rank factors do
   not. This is the main integration question.
2. **Opt-in**, default off, so the golden-test-guarded kernels and existing
   outputs are untouched when it is not in use — the more so because estimating
   the parameter costs ~1.7× RMSE when there is no overlap to find.
3. The sweep-boundary draw wired into the Numba/threading RNG, a
   `BivariateResult` field, validation, docs.
4. Validation on the **four-state mixture** and realistic LD, neither of which
   these infinitesimal, dense-LD prototypes exercise.
5. For regional inference specifically: resolve the shrinkage of `NOTES.md` §8.
   It is a larger obstacle than the whitening.
