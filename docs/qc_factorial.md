# The quality-control factorial

A record of the committed real-data factorial that motivated the
quality-control procedure in the [guide](guide.md#quality-control-before-fitting-real-data).
It describes what was run and what came out; it is not itself a procedure,
and three related trait pairs do not establish general validity.

## What the factorial established

Three related trait pairs spanning the sign range — LDL × CAD (`rg` ≈ +0.27),
height × LDL (≈ 0), and HDL × TG (`rg` ≈ −0.52 to −0.55 across the screened
arms) — were each fitted under all eight
combinations of stricter per-variant thresholds, long-range-LD exclusion, and
the screen. Every pair contains at least one GLGC lipid file, so these 24 arms
are repeated perturbations of three file combinations, not independent
validation across 24 settings. The saved rows come from a clean 0.3.5 run in
which every requested random partition completed. That makes them current for
these files and this reference, not a general validation of the screen.
Two decimals are the resolution these point values support: the companion
LDL × CAD study records its screened estimate moving from 0.2856 to 0.2658 when
0.3.6 reseeded the screen's random partitions, and HDL × TG under a
`cross_corr` overlap correction gives −0.52 (see [`rg.md`](rg.md)).

**Table 1. Divergence warnings across 24 current-screen arms.**

| factor | off | on |
|---|---:|---:|
| strict per-variant thresholds | 6/12 | 6/12 |
| long-range LD exclusion | 6/12 | 6/12 |
| **LD-consistency screen** | **12/12 diverged** | **0/12 diverged** |

In this run, the screen separated the warnings in these file/reference
combinations. The other factors did not change the warning count, but they did
change estimates; Table 1 cannot establish that they "do nothing." Among its
screened fits,
long-range exclusion moved `rg` by about 0.012 for height × LDL, 0.021–0.023
for HDL × TG, and 0.0001–0.0067 for LDL × CAD. Use
[`bipred.qc.in_long_range_ld`](../bipred/qc.py) as an estimator-specific
sensitivity analysis. Exclusion may protect genome-wide moments, while retaining
APOE may matter for prediction; the appropriate choice depends on the target.

## Why diagnostics matter

A diverged fit can still look plausible. On HDL × TG, all four screened
estimates and only one of four unscreened estimates lay in a rough external
range of −0.5 to −0.6 used by the historical study. That uncited context is not
ground truth, and agreement with any external point or interval cannot by
itself certify a fit.

Nor is the failure uniform. On LDL × CAD divergence halved `rg`; on height ×
LDL it shrank it toward zero; on HDL × TG it inflated it. And it can strike one
trait while sparing the other in the same fit — height × LDL diverged at
cancellation 150–212 on the LDL side while height remained in the rough
external range 0.3–0.5, with `h2` 0.41 against rough context around 0.45.

Within this study, warning status tracked the *summary-statistic file*: all
three GLGC lipid files diverged in every pairing, while height and CAD did not.
Three related pairs do not establish that pattern generally.

Since 0.3.1 a fit that trips a divergence diagnostic raises a `RuntimeWarning`
naming the check. Do not interpret `h2`, `rg`, or the overlap readouts until the
data/LD mismatch has been investigated; passing the diagnostic is necessary
evidence, not proof of correctness.
