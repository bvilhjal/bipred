# HAPNEST bivariate benchmark

Every other bivariate benchmark in this repo simulates its genome with a
coalescent (`msprime`) or a cached `ld_library.npz`. This one uses
[**HAPNEST**](https://github.com/intervene-EU-H2020/synthetic_data) (Wharrie et
al., *Bioinformatics* 2023) instead: synthetic genotypes **resampled from a real
1000G + HGDP reference**, so they carry real LD, MAF spectra and population
structure that the coalescent smooths over — together with HAPNEST's own
multi-trait phenotypes, whose genetic correlation, heritability and causal
overlap are fixed by its config and therefore **known ground truth**.

`run_bivariate.py` scores bipred against that truth: estimated `rg` (cross-trait
LDSC and the joint fit), per-trait `h²`, the MiXeR-style overlap, and — the real
payoff — **out-of-sample PRS R²** on a held-out test split, bivariate vs a
univariate ldpred3 baseline.

## Ground truth lives in the HAPNEST config

You never have to parse a truth file: the quantities bipred is scored on are the
ones you set in `config_bivariate.yaml` and pass to the script.

| bipred output | HAPNEST field | passed as |
|---|---|---|
| genetic correlation `rg` | `TraitCorr` (flattened `nTrait×nTrait`) | `--rg-true 0.5` |
| per-trait `h²` | `ProportionGeno` | `--h2-true 0.5 0.5` |
| causal overlap (`frac_shared`) | `Pleiotropy` / `causal_list` | `--overlap-true 1.0` |

## Step 1 — generate the dataset with HAPNEST

HAPNEST is a Julia tool distributed as a container; nothing here is needed to run
it. Reference data (`fetch`) is multi-GB, so this can't run in CI — treat it like
the other `msprime` benchmarks: an opt-in local run.

```bash
singularity pull docker://sophiewharrie/intervene-synthetic-data
mkdir -p data/output
cp benchmarks/hapnest/config_bivariate.yaml data/config.yaml   # then edit paths

SIF=intervene-synthetic-data_latest.sif
singularity exec --bind data/:/data/ $SIF init      # writes an example config to diff against
singularity exec --bind data/:/data/ $SIF fetch     # downloads the 1000G+HGDP reference (large)
singularity exec --bind data/:/data/ $SIF generate_geno 4 data/config.yaml
singularity exec --bind data/:/data/ $SIF generate_pheno data/config.yaml
```

This writes a PLINK fileset (`.bed/.bim/.fam`) and a phenotype file under
`data/output/`. **Reconcile `config_bivariate.yaml`'s section names against the
`init`-generated example first** — HAPNEST's field names are stable but the YAML
nesting can shift between versions.

## Step 2 — run the benchmark

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/hapnest/run_bivariate.py \
    --geno data/output/synthetic \
    --pheno data/output/synthetic.pheno \
    --rg-true 0.5 --h2-true 0.5 0.5 --overlap-true 1.0
```

Needs `bipred` and `ldpred3` installed (the univariate PRS baseline uses
`ldpred3_by_blocks`; if `ldpred3` can't provide it the baseline columns are
`nan`). Writes `run_bivariate.csv`. Key flags:

- `--traits 0,1` — which two trait columns of the phenotype file to use.
- `--ref PREFIX` — a separate PLINK panel for the LD reference (e.g. a second,
  smaller HAPNEST sample). Default: hold out `--ref-n` samples of `--geno`.
- `--test-frac 0.2` — fraction held out for out-of-sample PRS.
- `--block-size`, `--burn-in`, `--num-iter`, `--seed`.

## Building a grid

One config = one `(rg, h², overlap)` cell. To sweep `rg` (or overlap), copy the
config, change `TraitCorr` (or `Pleiotropy` / the causal list), regenerate, and
run the script with matching `--rg-true` / `--overlap-true`; append the
`run_bivariate.csv` rows. That mirrors how `rg_architectures.py` sweeps `rg` — the
difference is real resampled LD underneath instead of a coalescent.

## Caveats

- **Not CI-runnable**: the reference download and generation are large and slow.
- **Phenotype file format** is assumed to be `FID IID <trait0> <trait1> …`
  (whitespace-delimited, header auto-detected). Adjust `--traits`, or the
  `read_pheno` reader, if your HAPNEST version differs.
- With both traits on the **same** HAPNEST samples the GWAS have full sample
  overlap; the script sets the joint fit's `cross_corr` from the free LDSC
  cross-trait intercept, exactly as `rg_env_overlap.py` does.
