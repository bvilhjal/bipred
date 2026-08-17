# LDL x CAD: does QC change what a real bivariate fit concludes?

## Question

The simulation benchmarks draw `beta_hat ~ N(R beta, R/N)` from the model the
sampler assumes. That is the right way to measure an estimator against known
truth, and it is structurally incapable of catching a failure that appears only
when the summary statistics disagree with the LD reference — which is exactly
what shipped in 0.3.0, where the first real GWAS bipred was pointed at produced
a silently diverged fit that thirty architecture cells could not detect.

This study asks what three levels of cleaning do to a real bivariate fit:
harmonisation alone, per-variant filters, and those filters followed by the
DENTIST-inspired LD-consistency screen.

## Traits

* **LDL** — GLGC 2013 (Willer et al.), continuous, per-variant N.
* **CAD** — CARDIoGRAMplusC4D 2015 (Nikpay et al.), GCST003116. Case/control,
  fitted at an effective N. Its SE column is genomic-control corrected, so the
  reported h2 is observed-scale at the study's case fraction *and*
  conservative. It is not a liability-scale heritability.

The two consortia are believed close to disjoint. Their cross-trait LDSC
intercept (~+0.02) is consistent with small correlated sampling error, but an
intercept cannot identify cohort overlap by itself.

## Result

See [`REPORT.pdf`](REPORT.pdf) (source: [`REPORT.tex`](REPORT.tex) and
[`sections/`](sections/)).

The screen is the load-bearing step. Cancellation falls from 271 to 0.65 and
the divergence warning clears; only at that point is the joint fit admissible
at all. The `rg` of ~0.27 that emerges is a consequence of the fit becoming
valid, not an improvement to an already-valid estimate — at stages 1 and 2 the
reported `rg` of 0.06 and 0.14 are not estimates of anything.

## A caution this study exists to make

`results/estimates.csv` carries two runs: macOS/arm64 under bipred 0.3.5, and
Windows/x86-64 under 0.3.6. Stages 1 and 2 agree to every recorded digit across
a different OS, architecture, BLAS and an independently rebuilt LD reference.
Stage 3 does not: `rg` moved 0.2856 -> 0.2658.

The cause is benign — 0.3.6 deliberately gave up reproducing 0.3.5's screening
masks for a given seed, so 200 of 845,000 variants differ. But the shape of it
is the reason `studies/` is separate from `benchmarks/`: **a biological
estimate moved because of an implementation refactor**, and under a
regenerate-every-release contract nothing would have flagged it.

## Relationship to benchmarks/

`benchmarks/real_ldl_cad.py` still generates these numbers, and its value as a
regression canary — does a real-shaped GWAS still produce an admissible fit? —
is real and belongs there. What belongs here is the estimate and its
interpretation. Those two roles want opposite handling: the canary should be
regenerated freely, the estimate should not change quietly.

### Reproducing the committed rows

`--chi2-cap` now defaults to `regression`: the chi-square cap of 80 stays on
the LD Score regression rows, which need it, and comes off the joint fit, whose
slab component exists to hold large effects. Six of the nine committed rows
predate that change and carry `fit_chi2_cap=capped`, so reproducing them takes

```bash
python benchmarks/real_ldl_cad.py --chi2-cap both
```

The default invocation now reproduces the three `uncapped` rows instead. Read
`fit_chi2_cap` before comparing any two rows in this file. On LDL the cap costs
7.6% of summed chi-square and on CAD 0.4%, so the LDL x CAD estimate is one of
the least cap-sensitive in the registry — this is a provenance note, not a
retraction. [`oligogenic-overlap`](../oligogenic-overlap/) is where the
distinction changes conclusions.
