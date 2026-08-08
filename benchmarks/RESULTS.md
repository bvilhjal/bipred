# Benchmarks for bipred 0.3.1

Every self-contained script was re-run end to end for this record, against the
same cached coalescent truths as the 2026-08-03 snapshot (revision `17d6ae2`),
so accuracy, timing and memory all come from one sweep. Re-run with
`bash benchmarks/run_all.sh`; the CSVs are the authoritative numbers.

**Section 9 is new and is the most important thing here.** It is the only
benchmark in this file that does not simulate: real LDL and CAD summary
statistics against a real UK Biobank LD reference. It exists because 0.3.0
shipped a defect that the thirty simulated architecture cells of Table 2 could
not have caught, because every one of them draws `beta_hat` from the model the
sampler assumes. Read it before trusting a joint fit on real data.

Two things moved materially in the simulated record:

- **Table 11, the environmental-overlap stress test, is no longer a failure.**
  Joint-fit MAE was up to 0.86 there and is now at most 0.0242. The cause was
  the fit-time `ld_int8` default, which quantized the caller's dense LD
  internally; it now consumes it as given. That row was measuring in-fit LD
  quantization, not the model — see the table's own note.
- Joint-fit timings roughly halved, from the low-rank `fastmath` and LR8
  widening ports. Unrelated to the default change.

Accuracy on the well-conditioned panels barely moved: joint MAE across all 30
architecture cells went from 0.0122 to 0.0123.

[External runs](#external-runs): `bivariate_demo` is now regenerated from a
committed generator and moved with the rest; HAPNEST and SBayesS still need
inputs this host does not have, as declared there.

Simulation backend: msprime 1.4.2 (default where installed). The fallback
bundled Numba coalescent (`benchmarks/_coalescent.py`) is available when
msprime is absent; it draws different events from the same model, so cached
segments are tagged per backend and never mix.

**Table 1. Recorded environment.**

| Component | Value |
|---|---|
| Python | 3.14.6 |
| bipred | 0.3.1 |
| ldpred3 | 0.4.5 |
| NumPy / Numba | 2.4.6 / 0.66.0 |
| msprime / Matplotlib | 1.4.2 / 3.11.1 |
| Platform | Darwin 25.5.0, arm64 |
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
- external HAPNEST, cached-LD, and GCTB runs were not available.

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
real GWAS.

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
| LDSC | 0.0542 | 0.0608 | 0.028 s |
| two univariate fits, `uni_gv` | 0.0190 | 0.0422 | 0.968 s |
| two univariate fits, `uni_r2` | 0.0184 | 0.0420 | 0.968 s |
| joint default | **0.0086** | **0.0174** | 0.133 s |
| joint cross-sweep sensitivity | 0.0108 | 0.0242 | 0.133 s |

No estimator failed in these cells. The cross-sweep
`rg_decorrelated=True` estimator did not improve on the default, including in
the asymmetric setting. It remains a sensitivity analysis, not a preferred
replacement.

**Table 5. Single-core timing scan.**

| Variants | Blocks | LDSC | Two univariate fits | Joint default | Joint cross-sweep |
|---:|---:|---:|---:|---:|---:|
| 5,000 | 25 | 0.021 s | 0.903 s | 0.129 s | 0.129 s |
| 20,000 | 100 | 0.296 s | 3.350 s | 0.518 s | 0.521 s |
| 50,000 | 250 | 1.839 s | 9.060 s | 1.359 s | 1.375 s |

`uni_gv` and `uni_r2` reuse the same two univariate fits, so their recorded cost
is identical.

**Figure 3. Estimator accuracy and running time.**

![Genetic-correlation estimators](rg_methods.png)

## 4. Scaling

Each size runs in a fresh subprocess and reports one realized draw.

**Table 6. Scaling with variant count.**

| Variants | LDSC time | LDpred3 time | Peak RSS | Realized r_g | LDSC absolute error | LDpred3 absolute error |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 0.025 s | 0.116 s | 0.228 GB | 0.511 | 0.0399 | 0.0172 |
| 10,000 | 0.070 s | 0.235 s | 0.229 GB | 0.519 | 0.0590 | 0.0024 |
| 20,000 | 0.316 s | 0.466 s | 0.359 GB | 0.513 | 0.0332 | 0.0005 |
| 40,000 | 1.256 s | 0.956 s | 0.416 GB | 0.517 | 0.0138 | 0.0016 |
| 80,000 | 4.425 s | 2.064 s | 0.633 GB | 0.519 | 0.0632 | 0.0322 |

Peak RSS grows sublinearly across this range (0.23 GB to 0.63 GB for a 16x
variant count). LDSC time grows faster than the joint fit's and overtakes it
between 20k and 40k variants. An earlier record showed a 2.51 GB spike at 80k;
no run since has reproduced it, so treat single-run memory figures as machine-
and allocator-dependent rather than as a property of the method. This sweep measures the default dense/Q8 path, not million-variant LR8
production behavior.

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
Univariate anchoring shows the same mixed pattern. These are diagnostics, not
guaranteed corrections.

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

**This table previously recorded the package's worst failure mode, and it was
an artifact of in-fit LD quantization rather than of the model.** Through 0.2.1
the joint estimator was unstable here — MAE 0.31 and up to 0.86 in the
`rg_target=0.5` cells — and the instability was read as a genuine limit under
shared environment. It is not. The fit-time `ld_int8` default used to quantize
these float blocks to int8 internally, and re-running the current code with
`ld_int8=True` reproduces the old row byte for byte (0.1733 / 0.0166 / 0.3101 /
0.8343 / 0.8603). Consuming the caller's float32 LD instead leaves every cell
between 0.0072 and 0.0242 — a 36x improvement in the worst one.

The int8 resolution (~4e-3 per LD entry) is evidently enough to destabilize the
fit where correlated environment already stresses the conditioning, even though
it costs nothing measurable on the well-conditioned panels of Tables 2-7.
Supplying the mechanistic `cross_corr` still helps slightly and no longer needs
to rescue anything. A blanket robustness claim is still not made — this is one
simulated stress test — but the specific failure this table documented is
resolved, and the lesson is about quantizing LD inside a fit, not about
environmental correlation.

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
h²=0.5 each, m=6,000, six replicates). The question is whether fitting it
jointly with the strong trait beats fitting it alone.

**Table 12. Trait-2 genetic R² alone and jointly.**

| Architecture | Alone | Joint | Gain | Estimated r_g |
|---|---:|---:|---:|---:|
| shared causal, target r_g 0.0 | 0.583 | 0.596 | +0.013 | +0.00 |
| shared causal, target r_g 0.3 | 0.581 | 0.609 | +0.028 | +0.29 |
| shared causal, target r_g 0.6 | 0.577 | 0.660 | +0.082 | +0.55 |
| shared causal, target r_g 0.9 | 0.570 | 0.771 | +0.200 | +0.77 |
| disjoint causal variants | 0.584 | 0.584 | +0.000 | −0.03 |

The gain rises monotonically with shared signal and is exactly zero when the
causal sets are disjoint — the joint fit does not borrow strength that is not
there.

**The 0.2.1 record of this run said the LD library needed 5% spectral
shrinkage to fit at all. It does not, and that claim was another instance of
Bug 1.** Three runs separate the two causes. Holding the old shrunk library
fixed and restoring the 0.2.1 `ld_int8` default reproduces the 0.2.1 row
exactly (+0.006 / +0.014 / +0.034 / +0.082, `rg_est` −0.02 / +0.23 / +0.49 /
+0.79); the same library under the current default gives +0.011 / +0.018 /
+0.055 / +0.123; and the unshrunk library `make_ld_library.py` actually
produces gives the table above. So in-fit quantization cost this demo roughly
a third of its gain at r_g 0.9, and the committed generator's library — 59 to
96 of 500 eigenvalues below 1e-4 per block, near-singular by any
reading — fits without shrinkage and scores best of the three. The library
that collapsed at 0.2.1 was a different, more degenerate artifact than the one
the committed script builds, which is the reason that script now exists.

## 9. Real GWAS: LDL x CAD

Every section above simulates `beta_hat ~ N(R beta, R/N)` from the model the
sampler assumes, on well-conditioned coalescent LD. This one does not simulate
anything. It is here because a defect shipped in 0.3.0 that the thirty
architecture cells of Table 2 were structurally incapable of catching: it needs
summary statistics that *disagree* with the LD reference, which a simulation
drawing both from one model cannot produce.

LDL from GLGC 2013 (Willer et al., per-variant N) and CAD from
CARDIoGRAMplusC4D 2015 (Nikpay et al., GCST003116, effective N 162,973), on the
bigsnpr HapMap3 European LD reference (362,320 UK Biobank individuals,
1,054,330 variants, 625 blocks). The consortia are close to disjoint, and the
cross-trait LDSC intercept of +0.02 confirms that rather than assuming it.
Reproduce with [`real_ldl_cad.py`](real_ldl_cad.py) (about 9 GB of inputs, 22
minutes); it is excluded from `run_all.sh` for that reason.

Two anchors make this checkable without a simulated truth: cross-trait LDSC on
the identical data, and the published LDL-CAD genetic correlation of roughly
0.2 to 0.4.

**Table 13. The same analysis at three levels of cleaning.**

| Stage | Variants | LDSC r_g | Joint r_g | h2 LDL | sum(b^2)/h2 LDL | max abs b | Trace drift | Warned |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| harmonised only | 924,254 | +0.2238 | +0.0675 | 0.6395 | 258.0 | 3.3286 | 1.60 | yes |
| + per-variant QC | 887,361 | +0.1973 | +0.1244 | 0.5121 | 263.7 | 3.1087 | 1.79 | yes |
| + LD-consistency screen | 845,623 | +0.1851 | **+0.2796** | **0.0875** | **0.6** | **0.0265** | **0.92** | no |

**Harmonisation alone produces a silently diverged fit.** Stage one reports an
`h2` of 0.64 and a causal fraction of 0.00075, both inside every bound, and
under 0.3.0 returned no warning at all. It had diverged: posterior means reach
3.33 against the 0.030 per-causal effect SD the fit itself inferred, the sum of
squared effects is 258 times the genetic variance, and that variance is still
climbing at the last iteration. The runaway effects sit on variants in
near-perfect LD and cancel inside the quadratic form, which is precisely why
`h2` looks ordinary while the fit is worthless.

**Per-variant filters repair one trait and not the other.** MAF, imputation
quality, a chi-square cap, allele-frequency concordance and MHC exclusion
remove 4.0% of variants. CAD's cancellation ratio falls from 96.6 to 3.3 --
its `median_info` column was doing the work, with 27.3% of variants below 0.9 --
while LDL's goes from 258 to 264, untouched. LDL's problem is not a property of
any single variant, so no filter that judges variants one at a time can see it.

**The LD-consistency screen repairs it.** `bipred.qc.dentist` removes a further
4.7% (LDL 30,481, CAD 12,541) and every diagnostic resolves at once: the
cancellation ratio drops from 264 to 0.6, the largest effect falls 117-fold to
0.0265, the variance trace settles, and the divergence warning stops firing.
The joint `rg` of +0.2796 sits inside the published range, as does LDSC's
+0.1851 at 3.5 standard errors from zero.

Note that LDSC's own numbers move as the data is cleaned -- its LDL `h2` halves
from 0.2308 to 0.1155 -- so the target is not fixed. Both estimators end within
the literature; they disagree by 1.8 standard errors, which is a real question
about the two estimators and not evidence that either has failed.

Two caveats on the CAD side. Its SE column is genomic-control corrected, which
deflates z-scores and biases its `h2` downward, and a case/control trait
standardised against an effective N gives an observed-scale `h2` at the study's
case fraction. The reported 0.0401 is therefore conservative and is not a
liability-scale heritability.

## External runs

Not regenerated (inputs unavailable on this host):

- `hapnest/run_bivariate.py`, which needs a HAPNEST dataset (containerized
  Julia tool plus a multi-GB reference download; no container runtime here); and
- standalone `infer_vs_ldsc_sbayes.py`, which needs GCTB (Linux-only binary;
  absent on macOS). Its SBayesS arm can now run on macOS through the R
  SBayesRC package backend — see the script's backend resolver.

The CSV files are the authoritative numeric record. See [`README.md`](README.md)
for commands and artifact names.
