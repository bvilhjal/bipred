# Python API and command line

Every name below is importable directly from `bipred` (lazy PEP 562 imports,
so `import bipred` stays cheap). Use `help(name)` for signatures. The
statistical definitions live in [`algorithm.md`](algorithm.md), estimator
choice in [`rg.md`](rg.md), and diagnostics in
[`diagnostics.md`](diagnostics.md).

**Table 1. Fitting entry points.**

| Name | Purpose |
|---|---|
| `ldpred3_auto_bivariate` | joint fit on one dense LD matrix |
| `ldpred3_auto_bivariate_blocks` | genome-wide fit streaming LD blocks; pooled global `pi` and `Sigma` |
| `ldpred3_auto_bivariate_chains` | dispersed multi-chain fit with pooling and scalar diagnostics |
| `BivariateResult` | single-fit posterior effects, heritabilities, `rg`, mixture, traces |
| `MultiChainBivariateResult` | pooled `posterior`, `basic_split_rhat`, per-chain summaries |
| `BivariateChainSummary` | one chain's seed, starts, and own estimates |

**Table 2. Preparation and LD handling.**

| Name | Purpose |
|---|---|
| `prepare_bivariate_sumstats` | harmonize two GWAS to an LD cache; context manager owning the mapping |
| `prepare_trait_sumstats` | prepare one GWAS (re-exported from `ldpred3.prepare`) |
| `screen_prepared_trait` | trait-local LD-consistency screen (re-exported from `ldpred3`) |
| `pair_prepared_traits` | intersect two prepared traits into a fittable pair |
| `PreparedTrait`, `PreparedBivariate` | serializable prepared-trait and pair objects |
| `subset_blocks` | retile blocks to a retained index set after dropping variants |

**Table 3. QC helpers.**

| Name | Purpose |
|---|---|
| `ld_consistency_screen` | block LD-consistency mask (lives in `ldpred3.qc`; re-exported) |
| `dentist`, `dentist_statistic` | compatibility alias and underlying statistic for the screen |
| `sd_consistency` | per-variant reported-SD versus genotype-scale check |
| `implied_sample_size` | genotype-scale sample size implied by `beta`, `se`, AF |
| `in_long_range_ld` | Price-2008 long-range-LD mask plus APOE, for sensitivity analysis |

**Table 4. Genetic correlation and overlap.**

| Name | Purpose |
|---|---|
| `ldsc_rg`, `LDSCRgResult` | cross-trait LD Score regression screen |
| `estimate_sample_overlap` | LDSC-intercept overlap sensitivity value |
| `ldsc_chi2_mask` | chi-square row filter for `ldsc_rg` inputs only |
| `regional_rg`, `RegionalRgResult` | region-restricted LD-aware correlations from a fit |

## `BivariateResult` fields

**Table 5. Main result fields.**

| Field | Meaning |
|---|---|
| `beta1_est`, `beta2_est` | posterior-mean standardized effects; score, never `X @ beta` on raw dosages |
| `h2` | SNP heritability pair, clamped to `h2_bounds` for reporting |
| `rg` | genome-wide genetic correlation from raw quadratics |
| `p` | total non-null mixture fraction |
| `pi` | `(pi00, pi10, pi01, pi11)` posterior-mean mixture |
| `sigma` | mean retained 2x2 effect covariance |
| `noise_scale` | learned residual factors; `(1, 1)` when disabled |
| `pi_samples`, `sigma_samples` | retained post-burn-in hyperparameter iterates |
| `genetic_samples` | retained same-sweep `(gvar_1, gcov, gvar_2)` quadratics; heritability and covariance draws, not predictive-R2 draws |
| `noise_scale_samples` | retained per-sweep noise scales; ones when inflation is off |
| `retained_iterations` | post-burn-in sweeps actually retained |
| `stopped_early` | whether single-chain adaptive stopping fired |
| `divergence_diagnostics` | structured fit-validity ratios (see `diagnostics.md`) |
| `burn_in_pi_samples`, `burn_in_genetic_samples` | burn-in traces, only with `trace_burn_in=True`; enter no estimate |

**Table 6. Overlap and weight methods.**

| Method | Meaning |
|---|---|
| `mixer` | MiXeR-style summary from posterior-mean `pi` and `Sigma`; ratio of means |
| `mixer_iterate_summary(level=0.95)` | empirical retained-iterate summaries; mean of per-iterate ratios with percentile intervals — not credible intervals |
| `mixer_calibrated(infer1, infer2)` | overlap with per-trait counts anchored on two univariate ldpred3 `p_est` values; keeps joint `frac_shared` and `rho_beta` |
| `write_weights(path, trait=..., ...)` | one trait's posterior means as an ldpred3 weight file; refuses flagged fits unless `allow_diverged=True` |

## Command line

```bash
python -m bipred --ld-cache ld.npz --sumstats1 t1.tsv --sumstats2 t2.tsv \
    --n-eff1 80000 --n-cases2 12000 --n-controls2 38000 --screen \
    --out-weights1 t1.weights --out-weights2 t2.weights
```

`python -m bipred --help` is the canonical flag list. Groups and defaults:

**Table 7. CLI flags.**

| Flags | Default | Notes |
|---|---|---|
| `--ld-cache`, `--sumstats1/2` | required | one cache plus two GWAS files |
| `--n-eff1/2`, `--n-cases1/2`, `--n-controls1/2` | required per trait | effective N directly, or case/control counts |
| `--column1/2 FIELD=COLUMN` | — | repeatable column overrides per trait |
| `--no-qc`, `--min-n-ratio`, `--min-maf`, `--min-info` | 0.7, 0.01, 0.7 | summary-statistic QC floors |
| `--max-chisq` | unset | expert-only joint-panel deletion; normally unset (see guide) |
| `--keep-duplicates` | off | keep duplicate variants (default drops every occurrence) |
| `--min-af-corr` | unset | minimum aligned GWAS/cache AF correlation |
| `--screen`, `--screen-rounds`, `--screen-window` | off, 4, 1000 | LD-consistency screen and its schedule |
| `--screen-threshold` | 29.72 | chi-square with 1 df at p = 5e-8 |
| `--screen-eigenvalue-floor`, `--screen-seed`, `--screen-ncores` | 1e-3, `--seed`, `--ncores` | screen numerics and threading |
| `--cross-corr` | 0.0 | cross-trait sampling-error correlation in (-1, 1) |
| `--seed`, `--burn-in`, `--num-iter` | 0, 200, 200 | single and multi-chain schedule |
| `--ncores`, `--n-chains`, `--chain-ncores` | 1, 1, 1 | within-chain threads; 1 chain is a single fit, >= 2 disperses |
| `--out-weights1/2` | — | LDpred3 weight files per trait |
| `--hwe-frozen-scale` | off | write cache-AF/HWE `SD_REF`: an approximation, not observed scale |
| `--allow-diverged` | off | write weights despite a fired divergence diagnostic |

Python-only (no CLI flag): `h2_init`, `p_init`, `rg_init`, `pi_init`,
`sigma_prior_scale`, `iw_df`, `noise_inflation`, `ni_damp`, `pi_prior`,
`h2_bounds`, `h2_cap`, `tol`, `check_every`, `rg_decorrelated`,
`sample_every`, `ld_int8`, `trace_burn_in`, `progress`. The multi-chain
driver additionally rejects `tol > 0` and `rg_decorrelated=True`, and
`--n-chains >= 2` requires an even `--num-iter` of at least 4.
