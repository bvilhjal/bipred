# bipred

**bipred** jointly fits two GWAS traits against one LD reference. It estimates
SNP heritability, genetic correlation (`r_g`), posterior-mean effects for
prediction, and a MiXeR-style polygenic-overlap summary.

The package contains the bivariate methods split out of
[ldpred3](https://github.com/bvilhjal/ldpred3) and still uses ldpred3 for shared
LD representations and sampler utilities.

## Installation

Python 3.9–3.14 is supported. Numba is strongly recommended. The current
Bipred development line requires LDpred3 0.5. Neither package is on PyPI, so a
Git install requires authenticated GitHub read access. Install the exact
LDpred3 revision tested by CI; its private interoperability seam is not stable
on the sibling repository's moving default branch:

```bash
python -m pip install "ldpred3[fast] @ git+https://github.com/bvilhjal/ldpred3.git@aba8b55d7c8c083e4d2dd5715e995786bbf14599"
python -m pip install "bipred[fast] @ git+https://github.com/bvilhjal/bipred.git"
```

For development with sibling checkouts:

```bash
python -m pip install -e "../ldpred3[fast]"
python -m pip install -e ".[fast,test]"
```

CI preinstalls that immutable revision, then installs Bipred through its
declared `ldpred3>=0.5.5.dev0,<0.6` dependency contract rather than bypassing
the resolver. The private sampler seam in `bipred/_ldpred3_compat.py` still has
behavioural tests. Archived real-data
benchmarks deliberately retain their immutable LDpred3 0.4.5 provenance in
`benchmarks/real_data_inputs.py`; do not rewrite historical evidence when the
runtime dependency moves.

The `[sim]` extra adds msprime, the default simulation backend for benchmarks
(fastest: 0.80 s per 10,000-sample segment); without it, benchmark scripts fall
back to a bundled Numba coalescent (`benchmarks/_coalescent.py`,
dependency-free, about 2x msprime's per-segment time). `[bench]` adds msprime
and Matplotlib; benchmarks
that use HAPNEST or cached LD require their separately documented inputs.

## Runnable example

From a checkout:

```bash
python -m examples.minimal
```

The example creates a small synthetic two-trait problem, fits one dense LD
matrix, and prints the main estimates. Its complete source is
[`examples/minimal.py`](examples/minimal.py).

For real data, start from an ldpred3 LD cache:

```python
from bipred import prepare_bivariate_sumstats, ldpred3_auto_bivariate_blocks

with prepare_bivariate_sumstats(
        "ld.npz", "t1.tsv", "t2.tsv", n_eff1=N1, n_eff2=N2) as prep:
    res = ldpred3_auto_bivariate_blocks(
        prep.blocks, prep.beta_hat1, prep.beta_hat2,
        prep.n_eff1, prep.n_eff2, seed=0)
    res.write_weights("t1.weights", trait=1, id=prep.id, chrom=prep.chrom,
                      pos=prep.pos, effect_allele=prep.effect_allele,
                      other_allele=prep.other_allele)
```

```bash
python -m bipred --ld-cache ld.npz --sumstats1 t1.tsv --sumstats2 t2.tsv \
    --n-eff1 N1 --n-eff2 N2 --out-weights1 t1.weights --out-weights2 t2.weights
```

The default file has no fit-cohort dosage-scale metadata; score it with
`ldpred3.score_from_weights(..., scaling="target")`. Passing `af=prep.af`
adds an HWE-derived `SD_REF`, which is an explicit approximation rather than an
observed fit-cohort dosage SD. The CLI likewise requires
`--hwe-frozen-scale` to opt into that approximation. The user guide covers QC.
Bipred includes a lightweight LD-consistency
sensitivity screen:

```python
from bipred.qc import ld_consistency_screen

keep = (ld_consistency_screen(blocks, beta1 / se1)
        & ld_consistency_screen(blocks, beta2 / se2))
```

Here `beta/se` is the original GWAS z-score, not the standardized `beta_hat`
passed to the fit. The routine is DENTIST-inspired rather than an implementation
of the full published DENTIST workflow. It exposed serious file/reference
inconsistencies in the committed real-data study; that evidence does not make it
a universal substitute for study-specific QC. See *Quality control before
fitting real data* in [`docs/guide.md`](docs/guide.md).

## Documentation

- [`docs/guide.md`](docs/guide.md): inputs, calls, options, outputs, and pitfalls.
- [`docs/algorithm.md`](docs/algorithm.md): model and estimator theory.
- [`docs/rg.md`](docs/rg.md): genome-wide and regional genetic correlation,
  sample overlap, and polygenic overlap.
- [Benchmark guide](https://github.com/bvilhjal/bipred/blob/main/benchmarks/README.md):
  scripts and reproducibility.
- [Benchmark results](https://github.com/bvilhjal/bipred/blob/main/benchmarks/RESULTS.md):
  results, limitations, and per-section provenance.

## Citing and prior work

bipred has no paper of its own yet; cite the repository and the version you ran
(`bipred.__version__`). The methods it builds on should be cited directly:

- **LDpred / LDpred2** — the summary-statistic Gaussian-mixture model and Gibbs
  sampler this extends to two traits. Vilhjálmsson et al., *AJHG* 97:576–592
  (2015); Privé et al., *Bioinformatics* 36:5424–5431 (2020).
- **Cross-trait LD Score regression** — the moment estimator in
  `bipred.ldsc_rg`. Bulik-Sullivan et al., *Nature Genetics* 47:1236–1241 (2015).
- **MiXeR** — the polygenic-overlap parameterisation behind
  `BivariateResult.mixer`. Frei et al., *Nature Communications* 10:2417 (2019).

The estimators here are reimplementations, not wrappers. The repository's
synthetic benchmarks characterize bipred's behavior; equivalence to the
original cross-trait LDSC and MiXeR implementations has not been validated. See
[benchmark results](https://github.com/bvilhjal/bipred/blob/main/benchmarks/RESULTS.md)
for what has and has not been tested.

## License

MIT. See [`LICENSE`](LICENSE).
