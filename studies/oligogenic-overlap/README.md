# Polygenic overlap when one trait is oligogenic

## Question

Genome-wide genetic correlation averages over a million variants. For a trait
whose heritability sits in one or two loci, that average is dominated by the
~999,900 variants doing nothing. Does bipred's MiXeR-style overlap
parameterisation recover a relationship that `rg` dilutes to noise?

The motivating case is lipoprotein(a): essentially monogenic at *LPA*, a
validated causal risk factor for coronary artery disease, and the target of
phase-3 lowering trials. If `rg` cannot see that, it is the wrong summary for a
whole class of clinically important exposures.

## Design

Twelve trait pairs, one arm each -- **001**: lenient per-variant QC, long-range
LD and the MHC retained, LD-consistency screen on. That is the switch the
[`ldl-cad`](../ldl-cad/) results found to be load-bearing, isolated from the
other two.

Every fit is bivariate against the European UK Biobank HapMap3 LD reference,
threads pinned to one, four screening rounds.

Some pairs are there to calibrate rather than to ask anything. Direct against
total bilirubin (two studies, same biology) and urate against gout (crystals
are the disease mechanism) fix what a large `rho_beta` looks like; three pairs
with no expected shared biology fix the null. Without them, +0.36 for
Lp(a) x CAD is a number with no scale.

## Result

See [`REPORT.pdf`](REPORT.pdf) (source: [`REPORT.tex`](REPORT.tex) and
[`sections/`](sections/)).

`rho_beta` orders monotonically by biological closeness -- +0.99 for
same-biology and causal pairs, down to a -0.11 to -0.05 null band. Lp(a) x CAD
lands at +0.3644 while its `rg` of +0.0481 is indistinguishable from noise and
cross-trait LDSC (-0.0945, se 0.1161) points the wrong way.

A second result fell out of the null pairs: **`frac_shared` is not the
informative statistic.** Unrelated polygenic pairs still share 63-80% of their
causal variants.

## Gaps

- **No interval on `rho_beta`.** The null band is descriptive, not a test.
  Largest gap here.
- **Every oligogenic reading is Lp(a).** Its `n_causal` reproduced across three
  pairings (106, 131, 172), which is reassuring about the estimate but says
  nothing about generality.
- **Sample overlap throughout** -- most exposures are UK Biobank.
- **HapMap3 tags the *LPA* KIV-2 repeat poorly**, so h2 = 0.007 is a floor set
  by the reference.

## Reproducing

Inputs are GWAS Catalog harmonised releases registered in
[`../_lib/traits.toml`](../_lib/traits.toml); none are committed. Read
[`../_lib/datasets.md`](../_lib/datasets.md) before adding a dataset -- two of
the three defects recorded there are invisible until a fit has already started.
