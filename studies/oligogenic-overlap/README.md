# Polygenic overlap when one trait is oligogenic

## Question

Genome-wide genetic correlation is an average over a million variants. For a
trait whose heritability is concentrated in one or two loci, that average is
dominated by the ~999,900 variants doing nothing. Does bipred's
MiXeR-style overlap parameterisation recover a relationship that `rg` dilutes
to noise?

The motivating case is lipoprotein(a): essentially monogenic at *LPA*, a
validated causal risk factor for coronary artery disease, and the target of
phase-3 lowering trials. If `rg` cannot see that relationship, it is the wrong
summary for an entire class of clinically important exposures.

## Design

One arm, **001** — lenient per-variant QC, long-range LD and the MHC retained,
LD-consistency screen on. That is the single switch the
[`ldl-cad`](../ldl-cad/) study found to be load-bearing, isolated from the
other two so whatever it does is attributable to it.

Every estimate is a bivariate fit against the European UK Biobank HapMap3 LD
reference, threads pinned to one, four screening rounds.

**Anchors and controls are declared in `config.toml` before the fits, not
chosen afterwards.** An overlap statistic without a scale cannot be read: is
`rho_beta` = +0.36 large or small? Two anchors fix the top of the scale from
opposite directions — two measurements of the same biology (direct against
total bilirubin), and two genuinely distinct traits with an undisputed causal
link (urate against gout, where crystals *are* the disease mechanism). Three
controls with no expected shared biology fix the null band.

## Result

See [`REPORT.md`](REPORT.md), generated from `results/`.

The short version: `rho_beta` orders monotonically by biological closeness,
from +0.99 for same-trait and causal pairs down to a -0.11 to -0.05 null
band. Lp(a) x CAD lands at +0.3644 — with the genuinely related pairs — while
its `rg` of +0.0481 is indistinguishable from noise and cross-trait LDSC
(-0.0945, se 0.1161) points the wrong way.

A second finding fell out of the controls: **`frac_shared` is not the
informative statistic.** Unrelated polygenic pairs still share 63-80% of their
causal variants. The direction correlation is what discriminates.

## What would strengthen this

- **Uncertainty on `rho_beta`.** The estimates carry `rg_iterate_sd` but the
  overlap statistics have no interval, so the null band is descriptive rather
  than a test. That is the largest gap.
- **A second oligogenic exposure.** Every oligogenic reading here is Lp(a).
  Its `n_causal` reproduced across three independent pairings (106, 131, 172),
  which is reassuring about the estimate but says nothing about generality.
- **A non-UK-Biobank replication.** Sample overlap is substantial throughout;
  the intercept absorbs it, but a disjoint-cohort pair would test that.
- **A better Lp(a) reference.** HapMap3 tags the *LPA* KIV-2 repeat poorly, so
  h2 = 0.007 is a floor set by the reference, not a property of the trait.

## Reproducing

Inputs are the GWAS Catalog harmonised releases registered in
[`../_lib/traits.toml`](../_lib/traits.toml); none are committed. Rejected
sources and the checks that rejected them are in
[`../_lib/datasets.md`](../_lib/datasets.md) — read that before adding a
dataset, since two of the three defects it records are invisible until after a
fit has already started.
