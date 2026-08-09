# Datasets: what was tried, and what is wrong with it

A source that cannot be used is a result. Each entry below cost between twenty
minutes and an hour to establish, and every one of them is invisible until you
have already parsed the file and started a fit.

Scope: usability against the **European UK Biobank HapMap3 LD reference**
(1,054,330 variants). A dataset rejected here may be perfectly good elsewhere.

## Rejected

### GCST90128518 — gallstone disease (Fairfield 2021)

Two independent defects.

**Publishes MAF under an `effect_allele_frequency` header.** Values cap at
exactly 0.5000, and against the reference:

```
|freq - ref_af| < 0.05      : 72.2%
|(1-freq) - ref_af| < 0.05  : 32.3%
neither                     :  0.0%
```

Every variant matches either `ref_af` or `1-ref_af`, which is the signature of
minor-allele frequency. Left uncorrected, the per-variant AF-concordance check
rejects every variant whose effect allele is the major one — 23% of the file,
for a reason that is an artifact of the header.

**Widespread LD inconsistency.** With the AF defect corrected so the screen
sees the full variant set, it rejects **19.8%** — against 0.0-0.3% for the
clean biomarkers and 1.4% for CAD — and the joint fit then diverges
(cancellation 14.3, past the limit of 10). The file's own metadata records
`hm_coordinate_conversion: lo`, i.e. it was lifted over between builds, a
classic source of position and strand errors that survive harmonisation.

Note the order of events: the header bug was *masking* the LD problem by
discarding a fifth of the data. Fixing the first defect exposed the second.

### GCST90025993 — lipoprotein(a) (Barton 2021)

European-only, which is what the LD reference wants, but exome-based
(`bolt_460K_selfRepWhite.biochemistry_LipoproteinA_v2.WES.stats.gz`). It
matched **170,921 of 1,054,330** reference variants — 16.2%. Too sparse for a
genome-wide bivariate fit. The Sinnott-Armstrong release covers 99.3% and is
used instead, accepting that it is a 4-cohort meta-analysis at 95.9% European.

## Usable with corrections

### Sinnott-Armstrong 2021 biomarker panel (GCST900195xx)

**No allele frequency at any row** — both `effect_allele_frequency` and
`hm_effect_allele_frequency` are `NA` throughout, verified at 3,000,000 rows
deep. The AF-concordance check cannot be applied; substituting the reference AF
makes that term vacuous while leaving the MAF cut, chi2 bound, N filter,
`sd_consistency` and the LD-consistency screen intact.

Consequence worth stating plainly: the `af corr` diagnostic reads **+1.0000 by
construction** for these traits. It is an artifact of the substitution, not
evidence of allele alignment. Given that the same blind spot concealed the
gallstone defect, the screen's drop rate is the diagnostic to read instead.

Ancestry is mixed within the panel. ALP, CRP, cystatin C, direct bilirubin and
SHBG are European-only; GGT, total bilirubin and Lp(a) are ~96% European
meta-analyses.

### GCST90013872 — total bilirubin (Mbatchou 2021)

European-only, 99.7% coverage, but also carries no allele frequency. Its SD
offset is **1.270** against 0.90-1.02 for most traits, which suggests the
effects were not rank-inverse-normalised at source. `standardize_betas`
rescales, so it does not block the fit, but an implausible h2 downstream should
be read in that light first.

## Clean

`GCST90319906` (urate, Cho 2024) and `GCST90428600` (gout, Major 2024) both
publish real allele frequencies, so the AF-concordance check genuinely applies
— the first inputs since CAD for which that is true. Gout's screen drop of
6.2% is above the clean biomarkers and worth watching, though far below the
gallstone file's 19.8%.

## How to check a new source

In order of cost, cheapest first:

1. **Coverage** — matched variants as a fraction of the reference. Below ~90%
   suspect an exome or targeted array.
2. **`af corr`** against the reference. Near +1 is right; near +0.4 means a MAF
   header; near -1 means an inverted convention.
3. **Frequency range** — capping at exactly 0.5 is MAF, whatever the header says.
4. **Effect column** — an `odds_ratio` needs `log()`, and note that
   `read_aligned` negates on allele flip, which is right for a beta and wrong
   for an OR. Use `sign(x)*log|x|`.
5. **Screen drop rate** — the strongest single indicator. Under 1% is clean;
   above ~10% suspect liftover or a mixed imputation panel.
