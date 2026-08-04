# bipred

**bipred** jointly fits two GWAS traits against one LD reference. It estimates
SNP heritability, genetic correlation (`r_g`), posterior-mean effects for
prediction, and a MiXeR-style polygenic-overlap summary.

The package contains the bivariate methods split out of
[ldpred3](https://github.com/bvilhjal/ldpred3) and still uses ldpred3 for shared
LD representations and sampler utilities.

## Installation

Python 3.9–3.14 is supported. Numba is strongly recommended. ldpred3 is private
and not on PyPI, so its Git install requires authenticated GitHub read access.
Install the exact revision tested here:

```bash
python -m pip install "ldpred3[fast] @ git+https://github.com/bvilhjal/ldpred3.git@5436dcc8152531000b223c5088f726d588d8a8cd"
python -m pip install "bipred[fast] @ git+https://github.com/bvilhjal/bipred.git"
```

For development with sibling checkouts:

```bash
python -m pip install -e "../ldpred3[fast]"
python -m pip install -e ".[fast,test]"
```

Bumping the pinned ldpred3 (the seam in `bipred/_ldpred3_compat.py` is
private and unversioned): run the weekly `ldpred3-head` CI leg (or trigger it
manually — it runs the suite against ldpred3@master), audit the seam for any
drift it flags, then update `LDPRED3_REV` / `LDPRED3_VERSION` in
`.github/workflows/ci.yml` and the git revision in the install command above.

The `[sim]` extra adds msprime. `[bench]` adds msprime and Matplotlib; benchmarks
that use HAPNEST or cached LD require their separately documented inputs.

## Runnable example

From a checkout:

```bash
python -m examples.minimal
```

The example creates a small synthetic two-trait problem, fits one dense LD
matrix, and prints the main estimates. Its complete source is
[`examples/minimal.py`](examples/minimal.py).

For real data, start with the input contract and blockwise call in the user
guide. Summary statistics must already be ancestry-matched and harmonized.

## Documentation

- [`docs/guide.md`](docs/guide.md): inputs, calls, options, outputs, and pitfalls.
- [`docs/algorithm.md`](docs/algorithm.md): model and estimator theory.
- [`docs/rg.md`](docs/rg.md): genome-wide and regional genetic correlation,
  sample overlap, and polygenic overlap.
- [Benchmark guide](https://github.com/bvilhjal/bipred/blob/main/benchmarks/README.md):
  scripts and reproducibility.
- [Benchmark results](https://github.com/bvilhjal/bipred/blob/main/benchmarks/RESULTS.md):
  the reproducible bipred 0.2.0 snapshot, including limitations and provenance.

## License

MIT. See [`LICENSE`](LICENSE).
