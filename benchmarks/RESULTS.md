# Benchmark record

All artifacts in this record were regenerated with bipred 0.3.5 from clean revision
`5c06ec7`. Tables 2--11 and the `sweep_cost.csv` and `fit_memory.csv` artifacts
come from the single-core `run_all.log`; all ten scripts completed. Table 12
comes from the separate archive run, Tables 13--14 from LDL-CAD, and Tables
15--17 from the QC factorial. Their provenance sidecars pin the same clean
bipred revision, package environment, and input hashes. CSVs are the
authoritative numeric record; tables below are rounded summaries.

**Screen-dependent rows do not reproduce exactly on 0.3.6 or later.** Every
figure here that depends on `ld_consistency_screen` — the retained counts and
`rg` of Sections 9 and 10, and Tables 13--17 — was produced under 0.3.5's
random-split keying. 0.3.6 gives each block its own stream per round so the
screen can be pooled, which changes *which* variants a given seed drops. The
separations these sections report are qualitative (12/12 unscreened arms warned
against 0/12 screened) and are not expected to turn on the exact mask, but the
counts, timings and estimates would move on a regeneration. They are recorded
as 0.3.5 measurements and have not been re-run.

Sections 9 and 10 are the only non-simulated studies. They use public summary
statistics against a UK Biobank LD reference and expose file/reference
mismatches that simulations drawing `beta_hat` from the fitted model cannot.
They are diagnostic case studies, not independent population samples or a
universal QC validation.

Two findings are easy to misread:

- **Table 11 no longer reproduces the old aggregate failure.** Joint-fit MAE
  is at most 0.0242 after removing fit-time LD quantisation, but three
  individual fits still raise divergence warnings. Low mean error is not a
  clean bill of numerical health.
- Timings are machine-specific and were regenerated with every accuracy row in
  the current tables.

[External runs](#external-runs): HAPNEST and GCTB still need inputs or binaries
this host does not have, as declared there.

Simulation backend: msprime (`msprime-v1`). The fallback
bundled Numba coalescent (`benchmarks/_coalescent.py`) is available when
msprime is absent; it draws different events from the same model, so cached
segments are tagged per backend and never mix.

**Table 1. Environment recorded by the current log and provenance sidecars.**

| Component | Value |
|---|---|
| Python | 3.14.6 |
| bipred | 0.3.5 (`5c06ec7`, clean) |
| ldpred3 | 0.4.5 (installed-tree SHA-256 `76e2c2c2dacf...`) |
| NumPy / Numba | 2.4.6 / 0.66.0 |
| Simulator | msprime (`msprime-v1`) |
| Platform | Apple M2 Pro (10 cores), macOS 26.5.2, arm64 |
| Numerical threads | 1 |

All timing runs used one OpenBLAS, OMP, MKL, Numba, and NumExpr thread. Times
are machine-specific. Peak RSS includes simulation, LD construction, reference
panels, and JIT state present in each process.

## Reading the results

The generating parameter is an effect-correlation target. A finite causal draw
under LD generally has a different genetic correlation. Every accuracy column
therefore uses paired error against that replicate's realized

**Equation 1. Realized LD-adjusted genetic correlation.**

```text
r_g = beta1' R beta2 /
      sqrt((beta1' R beta1) (beta2' R beta2)).
```

That distinction matters most for sparse architectures. The previous artifacts
scored against the target and could mislabel Monte Carlo variation as estimator
error.

Other limits are equally important:

- most runs simulate the fitted `beta_hat ~ N(R beta, R / N)` model, so they
  favor a model-based estimator;
- estimates outside `|r_g| <= 1.5` are counted as failures before summaries;
- the environmental-overlap run is an intentionally out-of-model stress test;
- Monte Carlo sizes range from 5 to 20 replicates, not publication-scale
  calibration studies; and
- external HAPNEST and GCTB runs were not available.

## 1. Genetic correlation across architectures

The main sweep has five architectures, six targets, and ten replicates per cell
at 5,000 variants. MAE is computed per replicate against realized genetic
correlation, then averaged over the six targets.

**Table 2. Paired genetic-correlation error by architecture.**

| Architecture | LDSC MAE | LDpred3 MAE | LDSC mean SD | LDpred3 mean SD | Failures, LDSC / LDpred3 |
|---|---:|---:|---:|---:|---:|
| infinitesimal | 0.0289 | 0.0218 | 0.0448 | 0.0239 | 0 / 0 |
| sparse | 0.0914 | 0.0075 | 0.1434 | 0.0990 | 0 / 0 |
| moderate | 0.0583 | 0.0108 | 0.0800 | 0.0459 | 0 / 0 |
| polygenic | 0.0484 | 0.0135 | 0.0802 | 0.0324 | 0 / 0 |
| major locus | 0.0989 | 0.0081 | 0.1748 | 0.1152 | 4 / 0 |
| **All cells** | **0.0652** | **0.0123** | **0.1047** | **0.0633** | **4 / 0** |

Within this likelihood-matched simulation, the joint fit has lower paired MAE
in 26 of 30 cells and lower SD in all 30. It does not win everywhere:
high-correlation shrinkage is visible. In the infinitesimal target-0.95 cell,
realized r_g was 0.9509, while the joint and LDSC means were 0.9001 and 0.9474.
This is evidence for the tested model and geometry, not a universal ranking over
real GWAS. The log records 26 implausible-fit warnings among the 300 joint
fits; Table 2 retains them, so the paired-error ranking does not override the
sampler's h2/polygenicity diagnostics.

**Figure 1. Genetic-correlation estimates across architectures.**

![Genetic correlation across architectures](rg_architectures.png)

## 2. Polygenicity

At very low polygenicity, the realized truth itself becomes highly variable.
The expected causal count includes the script's enforced minimum of one causal
variant.

**Table 3. Recovery as the causal fraction decreases.**

| Causal fraction | Expected / observed causal count | Realized r_g, mean ± SD | LDSC MAE | LDpred3 MAE | Failures, LDSC / LDpred3 |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 500.0 / 492.6 | 0.493 ± 0.030 | 0.0491 | 0.0091 | 0 / 0 |
| 0.01 | 50.0 / 45.2 | 0.424 ± 0.071 | 0.1734 | 0.0109 | 0 / 0 |
| 0.001 | 5.007 / 3.6 | 0.320 ± 0.589 | 0.3159 | 0.0061 | 0 / 0 |
| 0.0001 | 1.107 / 1.0 | 0.600 ± 0.800 | 0.2448 | 0.0150 | 3 / 0 |

At one realized causal variant, genetic correlation is essentially a
sign-dominated quantity. The large SD is not evidence that the target 0.5 was
missed; the estimator must recover the realized draw.

**Figure 2. Genetic-correlation recovery by polygenicity.**

![Genetic correlation by polygenicity](rg_polygenicity.png)

## 3. Estimator comparison

The comparison uses five target points and six replicates at symmetric
`N=50k/50k` and asymmetric `N=50k/10k` power.

**Table 4. Mean paired MAE across the target grid.**

| Estimator | Symmetric MAE | Asymmetric MAE | Mean 5k fit time, symmetric |
|---|---:|---:|---:|
| LDSC | 0.0542 | 0.0608 | 0.033 s |
| two univariate fits, `uni_gv` | 0.0190 | 0.0422 | 0.942 s |
| two univariate fits, `uni_r2` | 0.0184 | 0.0420 | 0.942 s |
| joint default | **0.0086** | **0.0174** | 0.136 s |
| joint cross-sweep sensitivity | 0.0108 | 0.0242 | 0.135 s |

No estimator failed in these cells. The cross-sweep
`rg_decorrelated=True` estimator did not improve on the default, including in
the asymmetric setting. It remains a sensitivity analysis, not a preferred
replacement.

**Table 5. Single-core timing scan.**

| Variants | Blocks | LDSC | Two univariate fits | Joint default | Joint cross-sweep |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 25 | 0.024 s | 0.887 s | 0.132 s | 0.130 s |
| 20,000 | 100 | 0.329 s | 3.341 s | 0.507 s | 0.511 s |
| 50,000 | 250 | 2.033 s | 9.177 s | 1.386 s | 1.444 s |

`uni_gv` and `uni_r2` reuse the same two univariate fits, so their recorded cost
is identical.

**Figure 3. Estimator accuracy and running time.**

![Genetic-correlation estimators](rg_methods.png)

## 4. Scaling

Each size runs in a fresh subprocess and reports one realized draw.

**Table 6. Scaling with variant count.**

| Variants | LDSC time | LDpred3 time | Peak RSS | Realized r_g | LDSC absolute error | LDpred3 absolute error |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 0.027 s | 0.118 s | 0.207 GB | 0.511 | 0.0399 | 0.0172 |
| 10,000 | 0.132 s | 0.256 s | 0.318 GB | 0.519 | 0.0590 | 0.0024 |
| 20,000 | 0.307 s | 0.463 s | 0.328 GB | 0.513 | 0.0332 | 0.0005 |
| 40,000 | 1.300 s | 1.056 s | 0.421 GB | 0.518 | 0.0138 | 0.0016 |
| 80,000 | 5.358 s | 2.053 s | 2.559 GB | 0.519 | 0.0632 | 0.0322 |

LDSC time overtakes the joint fit between 20k and 40k variants. Peak RSS is
not smooth: it rises from 0.421 GB at 40k to 2.559 GB at 80k in this single
subprocess draw. Treat that process-wide peak as machine- and
allocator-dependent, not as a fixed per-variant memory law. This sweep measures
the default dense D32 path, not million-variant LR8 production behavior.

**Figure 4. Running time, memory, and single-draw recovery.**

![Genetic-correlation scaling](rg_scaling.png)

## 5. Polygenic overlap

The overlap sweep fixes per-trait causal fraction at 0.1 and within-shared
effect-correlation target at 0.8.

**Table 7. MiXeR-style overlap sweep.**

| Shared-fraction target | Estimated shared fraction ± SD | Realized r_g | Joint r_g / MAE | Overlap-derived r_g / MAE |
|---:|---:|---:|---:|---:|
| 0.00 | 0.039 ± 0.020 | -0.020 | -0.020 / 0.010 | -0.005 / 0.018 |
| 0.25 | 0.344 ± 0.063 | 0.206 | 0.203 / 0.011 | 0.171 / 0.035 |
| 0.50 | 0.605 ± 0.045 | 0.401 | 0.393 / 0.010 | 0.332 / 0.069 |
| 0.75 | 0.854 ± 0.038 | 0.617 | 0.615 / 0.007 | 0.528 / 0.089 |
| 1.00 | 0.985 ± 0.005 | 0.804 | 0.794 / 0.010 | 0.725 / 0.079 |

The shared fraction is ordered but overestimates intermediate targets, while
`rg_from_overlap` is attenuated. Absolute-count calibration is also
power-dependent: `noise_inflation=True` moves relative polygenicity from
1.174 to 1.062 at `N=20k`, but from 0.974 to 0.808 at `N=2.5k`.
Univariate anchoring shows the same mixed pattern. The log records one
divergence warning in the retained-iterate calibration arm. These are
diagnostics, not guaranteed corrections.

Every sweep above holds the per-trait causal fraction at 0.1, so on its own it
cannot say whether that overestimate is a property of the estimator or of that
one architecture. Varying the causal fraction with the overlap target held fixed
answers it: the bias is a function of polygenicity.

**Table 8. Shared-fraction bias against per-trait polygenicity.** `rho_beta`
target 0.8, `N=50k/20k`, eight replicates per cell.

| Causal fraction | Shared-fraction target | Estimated ± SD | Bias | Relative polygenicity |
|---:|---:|---:|---:|---:|
| 0.01 | 0.25 | 0.283 ± 0.036 | +0.033 | 1.50 |
| 0.01 | 0.50 | 0.571 ± 0.032 | +0.071 | 1.57 |
| 0.01 | 0.75 | 0.799 ± 0.062 | +0.049 | 1.64 |
| 0.03 | 0.25 | 0.272 ± 0.042 | +0.022 | 1.34 |
| 0.03 | 0.50 | 0.535 ± 0.049 | +0.035 | 1.30 |
| 0.03 | 0.75 | 0.769 ± 0.077 | +0.019 | 1.36 |
| 0.10 | 0.25 | 0.305 ± 0.046 | +0.055 | 1.25 |
| 0.10 | 0.50 | 0.617 ± 0.065 | +0.117 | 1.27 |
| 0.10 | 0.75 | 0.843 ± 0.041 | +0.093 | 1.24 |
| 0.30 | 0.25 | 0.430 ± 0.062 | +0.180 | 0.98 |
| 0.30 | 0.50 | 0.780 ± 0.034 | +0.280 | 1.04 |
| 0.30 | 0.75 | 0.964 ± 0.017 | +0.214 | 1.03 |

Mean bias by causal fraction: **+0.051** at 0.01, **+0.025** at 0.03,
**+0.088** at 0.10, **+0.225** at 0.30. The SD column is the spread of a single
replicate, which is what one fit of one dataset actually sees, so compare the
bias against it rather than against a standard error. At 0.03 every cell's bias
is smaller than that spread and would not be visible in practice; at 0.30 it is
2.9 to 12.6 times the spread, and those fits also trip the package's own
implausible-fit warning (fitted causal fraction ≈ 0.5). The minimum sits near
0.03 rather than at the sparse end: at 0.01 only 50 variants per trait are
causal, and both the bias and the count inflation (relative polygenicity 1.50 to
1.64, the worst in the table) grow again. Read a reported `frac_shared` with the
fitted polygenicity beside it.

**Figure 5. MiXeR-style overlap and count diagnostics.**

![MiXeR-style overlap](mixer_overlap.png)

## 6. Sample overlap

In the higher-power idealized run, known `cross_corr` nearly removes the paired
shift introduced by correlated sampling noise.

**Table 9. Paired overlap shift relative to the no-overlap cell.**

| Target | Noise correlation | Shift with `cross_corr=0` | Shift with known `cross_corr` | MAE, unset / set |
|---:|---:|---:|---:|---:|
| 0.0 | 0.2 | 0.0128 ± 0.0017 | 0.0002 ± 0.0021 | 0.0102 / 0.0081 |
| 0.0 | 0.4 | 0.0257 ± 0.0038 | 0.0002 ± 0.0038 | 0.0209 / 0.0086 |
| 0.5 | 0.2 | 0.0119 ± 0.0021 | 0.0009 ± 0.0019 | 0.0090 / 0.0072 |
| 0.5 | 0.4 | 0.0238 ± 0.0037 | 0.0011 ± 0.0038 | 0.0191 / 0.0071 |

At lower power (`N=15k/15k`, eight replicates), Monte Carlo variation dominates
the expected shift and setting the correction is not uniformly closer.

**Table 10. Lower-power sample-overlap MAE.**

| Target | Realized r_g, mean ± SD | LDSC constrained / free | Joint unset / set |
|---:|---:|---:|---:|
| 0.0 | 0.034 ± 0.119 | 0.0481 / 0.0920 | 0.0149 / 0.0187 |
| 0.3 | 0.328 ± 0.096 | 0.0664 / 0.0788 | 0.0134 / 0.0179 |
| 0.6 | 0.615 ± 0.073 | 0.1026 / 0.0663 | 0.0093 / 0.0140 |

The known-correction result does not validate LDSC-intercept inversion. That
mapping remains assumption-dependent; see [`docs/rg.md`](../docs/rg.md).

## 7. Environmental overlap stress test

The individual-genotype stress test uses the same 20,000 people for both traits
and correlates their residual environments.

**Table 11. Paired MAE against realized genetic correlation.**

| Target | Environmental correlation | Realized r_g | LDSC free / constrained | Joint unset / set |
|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | -0.022 | 0.0666 / 0.0353 | 0.0126 / 0.0118 |
| 0.0 | 0.3 | -0.003 | 0.0622 / 0.0331 | 0.0146 / 0.0105 |
| 0.0 | 0.6 | -0.003 | 0.0639 / 0.0297 | 0.0185 / 0.0090 |
| 0.5 | 0.0 | 0.487 | 0.0389 / 0.0620 | 0.0147 / 0.0082 |
| 0.5 | 0.6 | 0.494 | 0.0377 / 0.0633 | 0.0242 / 0.0072 |

The old fit-time `ld_int8` default quantised the caller's float LD and produced
MAE as high as 0.86. Consuming the supplied float32 blocks leaves the current
cell means between 0.0072 and 0.0242; setting the mechanistic `cross_corr`
reduces them further in all five cells. The current log nevertheless records
three individual divergence warnings. This stress test supports removing
in-fit quantisation; it does not establish blanket robustness to environmental
overlap.

## 8. Joint-fit gain on an under-powered trait

`bivariate_demo.py` is the one run whose LD comes from a stored archive rather
than from the suite's own cache, which is why it is not part of `run_all.sh`.
Build the archive first — it is gitignored because it is large — and run the
demo from the directory holding it:

```text
python benchmarks/make_ld_library.py          # writes ./ld_library.npz
python benchmarks/bivariate_demo.py
```

Trait 2 is deliberately under-powered (`N=2,000` against trait 1's `100,000`,
h²=0.5 each, m=6,000, six paired replicates). The same population effects,
GWAS noise and finite reference-panel draw are fitted twice: as the raw sample
correlation matrix and after 5% shrinkage toward the identity. The CSV records
every replicate at full precision.

**Table 12. Trait-2 genetic R² under paired reference-LD regularisation.**

| Reference shrinkage | Architecture | Realized r_g | Alone | Joint | Gain | Estimated r_g | Joint fits with implausibility warning |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0% | shared, target 0.0 | +0.003 | 0.577 | 0.582 | +0.0048 | +0.001 | 5 / 6 |
| 0% | shared, target 0.3 | +0.306 | 0.576 | 0.591 | +0.0143 | +0.220 | 4 / 6 |
| 0% | shared, target 0.6 | +0.606 | 0.575 | 0.577 | +0.0025 | +0.439 | 4 / 6 |
| 0% | shared, target 0.9 | +0.902 | 0.566 | 0.699 | +0.1330 | +0.818 | 6 / 6 |
| 0% | disjoint causal support | -0.015 | 0.584 | 0.582 | -0.0016 | +0.001 | 6 / 6 |
| 5% | shared, target 0.0 | +0.003 | 0.583 | 0.596 | +0.0129 | +0.000 | 0 / 6 |
| 5% | shared, target 0.3 | +0.306 | 0.581 | 0.609 | +0.0281 | +0.295 | 0 / 6 |
| 5% | shared, target 0.6 | +0.606 | 0.577 | 0.660 | +0.0824 | +0.546 | 0 / 6 |
| 5% | shared, target 0.9 | +0.902 | 0.570 | 0.771 | +0.2004 | +0.767 | 0 / 6 |
| 5% | disjoint causal support | -0.015 | 0.586 | 0.583 | -0.0032 | -0.003 | 0 / 6 |

The raw-reference arm is an unstable control: 25 of 30 fits raise the
implausibility warning and three raise a divergence warning. Its apparent gains
are not interpretable as estimator performance. Five-percent shrinkage removes
all warnings in this fixture; its shared-effect gain rises from +0.0129 at
target 0 to +0.2004 at target 0.9. That pattern is conditional on this panel,
regularisation, and simulation rather than a general gain estimate.

The last row has exactly disjoint causal supports, but LD gives it mean realized
genetic correlation -0.015. Its small negative mean gain is likewise not
exactly zero; the arm does not establish “no harm.”

## 9. Real GWAS: LDL x CAD

Every section above simulates `beta_hat ~ N(R beta, R/N)` from the model the
sampler assumes, on well-conditioned coalescent LD. This one does not simulate
anything. It is here because a defect shipped in 0.3.0 that the thirty
architecture cells of Table 2 were structurally incapable of catching: it needs
summary statistics that *disagree* with the LD reference, which a simulation
drawing both from one model cannot produce.

LDL from GLGC 2013 (Willer et al., per-variant N) and CAD from
CARDIoGRAMplusC4D 2015 (Nikpay et al., GCST003116), on the bigsnpr HapMap3
European LD reference (362,320 UK Biobank individuals, 1,054,330 variants, 625
blocks). The consortia are believed to be close to disjoint; the cross-trait
LDSC intercept of +0.02 is consistent with small correlated sampling error but
does not identify cohort overlap. Reproduce with
[`real_ldl_cad.py`](real_ldl_cad.py) (about 9 GB of inputs; 22.8 minutes on the
recorded host); it is
excluded from `run_all.sh` for that reason.

CAD's sample size is calibrated rather than taken as published, which is the
first thing the benchmark does. `bipred.qc.implied_sample_size` returns 92,966
against the reported 162,973, a ratio of 0.570; both genomic control on the SE
column and the pooled `4/(1/n_case + 1/n_ctrl)` formula push in that direction
and neither is separable from the file. Since `n_eff` scales every standardized
effect, the reported figure understated CAD's `h2` by that factor. Every number
below uses the implied value for CAD, for both bipred and LDSC.

Two comparisons provide context without supplying a truth label: cross-trait
LDSC on the identical data, and a rough external LDL-CAD interval of 0.2 to 0.4.
They differ in model, samples, phenotype definition and QC, so neither is a
pass/fail oracle for this fit.

**Table 13. The same analysis at three levels of cleaning.**

| Stage | Variants | LDSC r_g | Joint r_g | h2 LDL | h2 CAD | sum(b^2)/h2 LDL | max abs b LDL | Trace drift | Divergence warning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| harmonised only | 924,254 | +0.2238 | +0.0558 | 0.6732 | 0.1713 | 255.2 | 3.1929 | 1.63 | yes |
| + per-variant QC | 887,361 | +0.1973 | +0.1390 | 0.5688 | 0.0594 | 271.0 | 3.3215 | 1.78 | yes |
| + LD-consistency screen | 845,623 | +0.1851 | **+0.2856** | **0.0882** | **0.0706** | **0.7** | **0.0239** | **0.92** | no |

**Harmonisation alone produces a silently diverged fit.** Stage one reports an
`h2` of 0.67 and a causal fraction of 0.00076, both inside every bound, and
under 0.3.0 returned no divergence warning. It had diverged: posterior means reach
3.19 against the 0.031 per-causal effect SD the fit itself inferred, the sum of
squared effects is 255 times the genetic variance, and that variance is still
climbing at the last iteration. The runaway effects sit on variants in
near-perfect LD and cancel inside the quadratic form, which is precisely why
`h2` looks ordinary while the fit is worthless.

**Per-variant filters repair one trait and not the other.** MAF, imputation
quality, a chi-square cap, allele-frequency concordance and MHC exclusion
remove 4.0% of variants. CAD's cancellation ratio falls from 91.2 to 4.9 --
its `median_info` column was doing the work, with 27.3% of variants below 0.9 --
while LDL's goes from 255 to 271, untouched. LDL's problem is not a property of
any single variant, so no filter that judges variants one at a time can see it.

**The LD-consistency screen resolves the fit diagnostics in this case.** The
DENTIST-inspired screen removes a further 4.7% of the shared variants: the
cancellation ratio drops from 271 to 0.7, the largest effect falls 139-fold to
0.0239, trace drift is 0.92, and the divergence warning does not fire. This is
not the complete published DENTIST procedure, and one case does not calibrate
the screen. The joint `rg` of
+0.2856 and LDSC's +0.1851 happen to fall inside that rough external interval;
that agreement is context, not validation.

Note that LDSC's own numbers move as the data is cleaned -- its LDL `h2` halves
from 0.2308 to 0.1155 -- so the target is not fixed. Both estimators end within
the rough external range; they disagree by 1.9 standard errors. That is a
descriptive comparison between estimators, not evidence that either has passed
or failed.

The `h2` figures for CAD are observed-scale at the study's case fraction, not
liability-scale, and the genomic-control correction on the SE column biases them
downward beyond what the sample-size calibration recovers. The calibration is
also not a reconciliation between the two estimators: LDSC rescales with `n_eff`
exactly as bipred does, so correcting it moved LDSC's CAD `h2` from 0.0687 to
0.1205 at the same time, and bipred's 0.0706 stays at 0.59 of it.

The timed run used four screening rounds and one numerical thread. Table 14
partitions its leaf timings into non-overlapping steps; the 0.129 s remainder is
driver overhead. The three stage fits took 281.707 s (harmonised), 271.404 s
(per-variant QC), and 262.562 s (screened).

**Table 14. Wall time of the real LDL-CAD benchmark.**

| Step | Seconds | Share of total |
|---|---:|---:|
| Source and input checks | 1.164 | 0.09% |
| Load LD reference and harmonise GWAS | 21.504 | 1.57% |
| Shared-data preparation | 0.985 | 0.07% |
| LD-consistency screening, including retile | 526.031 | 38.52% |
| Fit-stage retile and standardisation | 6.623 | 0.48% |
| LD-score calculation across three stages | 280.132 | 20.51% |
| LDSC regression across three stages | 89.018 | 6.52% |
| Bivariate fitting across three stages | 425.280 | 31.14% |
| Fit diagnostics across three stages | 14.619 | 1.07% |
| Write and regression-check outputs | 0.207 | 0.02% |
| Driver overhead | 0.129 | 0.01% |
| **Total** | **1365.693** | **100.00%** |

The exact recorded total is **1365.693 s** (22 min 45.693 s). Screening and
bivariate fitting account for 69.66% of it; LD-score calculation adds 20.51%.

## 10. QC sensitivity: a 24-arm factorial

Section 9 showed improved diagnostics after screening one real pair. This
section applies the same screen under every
combination of three factors:
strict per-variant thresholds, long-range LD exclusion, and the screen.
Reproduce with
[`qc_factorial.py`](qc_factorial.py) (about 126 minutes on this host; not in
`run_all.sh`).

The pairs span the sign range: LDL x CAD (positive, disjoint consortia), height
x LDL (near null, cross-domain), and HDL x TG (strongly negative, complete
sample overlap). All three contain at least one GLGC file; the 24 arms are eight
perturbations of each pair, not 24 independent validations. Every arm records
bipred and cross-trait LDSC on identical variants and sample sizes.

Each arm refits a free LDSC intercept after its filtering choices and passes
that raw, arm-specific value as a `cross_corr` sensitivity value; the saved
values range from -0.3574 to +0.0294. This assumes the whole intercept is
correlated sampling noise, although confounding can also contribute. Thus the
correction varies with the variant set rather than being a fixed factorial
input. All saved values were already inside `(-1, 1)`.

**Table 15. Divergence-warning count by factor, 24 arms.**

| Factor | off | on |
|---|---:|---:|
| strict per-variant thresholds | 6 / 12 | 6 / 12 |
| long-range LD exclusion | 6 / 12 | 6 / 12 |
| **LD-consistency screen** | **12 / 12 triggered** | **0 / 12 triggered** |

The screen separates the divergence-warning outcome in these files.
Equal marginal counts for the other factors mean only that they did not change
that binary diagnostic;
they do not establish that those choices are inert. Among screened arms,
long-range-LD exclusion shifts `rg` by 0.0001--0.0067 for LDL x CAD, about
0.012 for height x LDL, and 0.021--0.023 for HDL x TG. Treat exclusion as an
estimator-specific sensitivity, especially where a major locus could dominate.

**Table 16. No-divergence-warning arm means with the screen.**

| Pair | LDSC r_g | Joint r_g | LDSC h2 (1, 2) | Joint h2 (1, 2) |
|---|---:|---:|---:|---:|
| LDL x CAD | +0.189 | **+0.262** | 0.107, 0.112 | 0.079, 0.067 |
| height x LDL | -0.095 | **-0.040** | 0.264, 0.118 | **0.415**, 0.095 |
| HDL x TG | -0.692 | **-0.534** | 0.126, 0.119 | 0.100, 0.090 |

Across the four screened arms, the joint and LDSC estimates differ by
1.16--1.43 LDSC standard errors for LDL x CAD and 2.93--4.74 for HDL x TG. The
height x LDL joint estimate is nearer zero in this case, but three related pairs
cannot establish a general shrinkage law.

**Heritability disagrees, and not in one direction.** The joint fit is below
LDSC for LDL and CAD and 57% *above* it for height. Its 0.415 can be compared
with rough external context near 0.45; that does not establish that LDSC's
0.264 is biased. An earlier
draft of this record claimed a systematic downward bias in the joint estimate;
height refutes that, and no single mechanism explains both directions.

LDL h2 is 0.079 beside CAD and 0.095 beside height; LDSC moves from 0.107 to
0.118 in the same direction. The source summary file is unchanged, but each
pair has a different variant intersection, filtered set and arm-specific
`cross_corr`. This is sensitivity to the combined pair design and estimator,
not evidence that partner conditioning alone changed heritability.

**Table 17. HDL x TG diagnostics by screening choice.**

| Arm | Variants | Joint r_g | Cancellation | Divergence warning |
|---|---:|---:|---:|:---:|
| no screen | 917,879 | -0.5687 | 246 | yes |
| no screen, -LR | 892,062 | -0.6544 | 256 | yes |
| no screen, strict | 886,649 | -0.6248 | 238 | yes |
| no screen, strict -LR | 861,745 | -0.6393 | 254 | yes |
| **screen** | 840,209 | **-0.5212** | 0.5 | no |
| **screen, -LR** | 816,796 | **-0.5444** | 0.5 | no |
| **screen, strict** | 811,011 | **-0.5256** | 0.5 | no |
| **screen, strict -LR** | 788,382 | **-0.5466** | 0.5 | no |

Cross-trait LDSC on this pair reports -0.64 to -0.73; roughly -0.5 to -0.6 is
useful external context, not ground truth. All four screened estimates fall
inside that rough range;
only one of four unscreened estimates does. That agreement is reassuring but
cannot certify convergence: external ranges are broad, LDSC is not a gold
standard, and fit diagnostics carry different information.

**Within these pairs, the warning follows the unstable trait.** Every
unscreened arm warns and every screened arm does not. On height x LDL, LDL has
cancellation ratios of 150--212 while height retains ratios of 0.3--0.5 and h2
near 0.41. The latter is numerically stable and close to rough external context
near 0.45; neither observation establishes correctness. Because every pair
contains a GLGC file, this design cannot estimate a general
bivariate-versus-univariate tolerance difference.

**Overlap readouts are least stable where the overlap is weakest.** Across the
four screened LDL x CAD arms `frac_shared` ranges 0.33 to 0.59 and `rho_beta`
0.56 to 0.92 while `rg` spans only 0.0067 -- the data constrains the product much
better than the decomposition. On HDL x TG, where the two lipid fractions are
measured in the same individuals, `frac_shared` is 0.94 to 0.95 and
`rg_from_overlap` (-0.53 to -0.56) agrees with `rg` (-0.52 to -0.55): the
decomposition is more stable in this example. It is not a calibration study.

Note also that `mixer_overlap.py` selects causal variants uniformly at random,
which is the exchangeability the model assumes. Real causal variants are not
uniformly distributed, so the simulated bias figures for `frac_shared` do not
transfer to these fits.

## External runs

Not regenerated (inputs unavailable on this host):

- `hapnest/run_bivariate.py`, which needs a HAPNEST dataset (containerized
  Julia tool plus a multi-GB reference download; no container runtime here); and
- standalone `infer_vs_ldsc_sbayes.py`, which needs GCTB (Linux-only binary;
  absent on macOS). Its SBayesS arm can now run on macOS through the R
  SBayesRC package backend — see the script's backend resolver.

The CSV files are the authoritative numeric record. See [`README.md`](README.md)
for commands and artifact names.
