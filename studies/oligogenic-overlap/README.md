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
threads pinned to one, four screening rounds. The chi-square cap of 80 is
applied to the LD Score regression, which needs it, and **not** to the joint
fit -- see the report's *chi-square cap* section for why that distinction turned
out to matter more than anything else in the QC.

Some pairs are there to calibrate rather than to ask anything. Direct against
total bilirubin (two studies, same biology) and urate against gout (crystals
are the disease mechanism) fix what a large `rho_beta` looks like; three pairs
with no expected shared biology fix the null. Without them, +0.52 for
Lp(a) x CAD is a number with no scale.

## Result

See [`REPORT.pdf`](REPORT.pdf) (source: [`REPORT.tex`](REPORT.tex) and
[`sections/`](sections/)).

`rho_beta` orders by biological closeness -- above +0.98 for same-biology and
causal pairs, down to a -0.11 to -0.03 null band. Lp(a) x CAD lands at +0.5170
while its `rg` of +0.0712 is indistinguishable from noise and cross-trait LDSC
(-0.0947, se 0.1238) points the wrong way. It ranks fourth of the ten
admissible pairs, and is the highest-ranked one that is neither the same trait
measured twice nor a mechanism acting on its own disease.

A second result fell out of the null pairs: **`frac_shared` is not the
informative statistic.** Four pairs share 61-75% of their causal variants while
their `rho_beta` spans -0.03 to +0.32.

A third came from the QC. The chi-square cap of 80, inherited from LD Score
regression, was also being applied to the joint fit -- suppressing Lp(a)'s h2
by a factor of six and inflating every trait's estimated polygenicity, by 9%
for Lp(a) and 93% for dbilirubin. Removing it from the fit is what the numbers
above rest on. It also broke one fit: GGT x bilirubin diverges without the cap,
so the filter was load-bearing there even as it destroyed signal elsewhere.

## Gaps

- **No interval on `rho_beta`.** The null band is descriptive, not a test.
  Largest gap here.
- **Every oligogenic reading is Lp(a).** Its `n_causal` reproduced across three
  pairings (90, 94, 96), which is reassuring about the estimate but says
  nothing about generality.
- **Sample overlap throughout** -- most exposures are UK Biobank.
- **HapMap3 tags the *LPA* KIV-2 repeat poorly**, so h2 = 0.045 is still a
  floor set by the reference, not an estimate of Lp(a) heritability.
- **One pair has no admissible fit.** GGT x bilirubin diverges uncapped and is
  excluded from every figure.

## Reproducing

Inputs are GWAS Catalog harmonised releases registered in
[`../_lib/traits.toml`](../_lib/traits.toml); none are committed. Read
[`../_lib/datasets.md`](../_lib/datasets.md) before adding a dataset -- two of
the three defects recorded there are invisible until a fit has already started.
