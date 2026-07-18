# bipred Review — Theory, Documentation, and Implementation

**Date:** 2026-07-18
**Scope:** full review of the bipred package at commit `8953dbf` ("feat: add coherent
bivariate initialization") — statistical theory (`docs/algorithm.md`, `docs/rg.md`),
documentation (`README.md`, `CHANGELOG.md`, `docs/`, `benchmarks/README.md`), and
implementation (`bipred/bivariate.py`, `bipred/ldsc_rg.py`, tests).
**Test suite:** 117 passed, 0 warnings (Python 3.14.6 + Numba, conda env `bipred`),
including the bit-exact float32/int8 golden tests and the LDSC-rg golden.

## Verdict

No critical defects. The math in both docs and both code files is sound; every formula
in the sampler core was re-derived and matches. Two **major** documentation problems
(a wrong bias mechanism in the `rg_decorrelated` docs, and CHANGELOG benchmark numbers
that committed artifacts contradict), one real but far-corner **bug** (negative initial
mixture weights for extreme `rg_init`), and a set of minor/nit documentation and
robustness items.

**Findings: 2 major · 9 minor · 11 nit.**

---

## Major findings

### M1. `rg_decorrelated` mechanism documented with the wrong sign and mechanism

- `docs/algorithm.md:112-114` — "…reducing same-sweep noise coupling that can
  **attenuate** the weak trait's covariance."
- `bipred/bivariate.py:881-886` (code comment) — "…avoids the same-sweep cross-noise
  that **inflates** the genetic covariance and recovers a weak trait's covariance from
  its posterior mean (which the sampled-quadratic ratio attenuates)."

These contradict each other. Derivation: conditionally on the data, the same-sweep
cross-quadratic is `E[β1'Rβ2|D] = pm1'R·pm2 + tr(R·PostCov12)`, and the posterior
cross-covariance has the sign of `s12` (in the both-state draw, `new1`/`new2` share
`z1` via `L21 ∝ s12`, `bivariate.py:415-416`), so the cross-covariance is **inflated**
toward `sign(s12)`, not attenuated. What attenuates the rg ratio for a weak trait is
posterior-noise inflation of the weak trait's sampled variance `gv22` in the
denominator. The code comment has it right; the doc has the mechanism backwards (and
the "same shrinkage scale" phrasing at `algorithm.md:108-110` is similarly loose —
sampled quadratics are noise-*inflated*, not shrunk; the defensible argument is partial
cancellation of that inflation in the ratio).

**Fix:** reword to "…reducing same-sweep noise coupling that inflates the
cross-covariance and inflates the weak trait's sampled variance — the latter
attenuating the rg ratio for under-powered traits."

### M2. CHANGELOG quotes noise-inflation benchmark numbers no committed artifact supports

- `CHANGELOG.md:93-97` — "counts calibrate ~fully (e.g. **n₁ 909→309 vs truth 300 at
  N=200k**)" and "cuts the inflation from **~2.4× to ~1.6× at N=200k**".
- `benchmarks/mixer_overlap.csv` (calibration sweep, 5 rows) only runs
  **N = 1,000–20,000** with `rel_off` ≤ 1.215, `rel_on` ≤ 1.105, and truth
  500 causal per trait (`NCAUSAL = 0.10*M`, `benchmarks/mixer_overlap.py:77`,
  M=5000) — not 300.
- `benchmarks/RESULTS.md:147-149` describes the same calibration as "relative
  polygenicity rising to ~1.2 … back toward 1", consistent with the CSV, not with the
  CHANGELOG.

**Fix:** reword the entry to the committed numbers (inflation rising to ≈1.2 by
N=20k, pulled back to ≈1.0), or commit the N=200k run the numbers came from.

---

## Minor findings

### m1. Negative initial mixture weights for `rg_init ∈ (0.999, 1)` — silent invalid start

`bipred/bivariate.py:129-136`. In the shorthand init,
`shared = max(q/3, |rg|·q/(2·0.999 − |rg|))` exceeds `q` when `|rg_init| > 0.999`, so
`single = (q − shared)/2 < 0` and the returned `pi` has negative entries. Only the
explicit-`pi_init` path validates nonnegativity (`:142-145`); the shorthand path does
not. `p1 = single + shared` stays positive so no error fires; `log(max(pi, 1e-300))`
(`:811`) masks the negative states to ~zero probability on sweep 1, after which the
Dirichlet redraw repairs the mixture — silent, self-correcting, but the start is not a
probability vector, contrary to the coherent-init contract and the exactness claim in
`docs/algorithm.md:67-69`. Reproducer:

```python
_initial_hyperparameters(1000, 0.1, 0.02, 0.9999)
# pi = [0.98, -1.803e-05, -1.803e-05, 0.020036]   # negative entries, no error
```

**Fix:** clamp `shared = min(shared, q)` (then `single = 0`, `rho_beta` stays strictly
< 1 and the sampler's 0.999 PD clamp handles the boundary), or raise `ValueError` for
`|rg_init| ≥ 0.999` mirroring the explicit-`pi_init` infeasibility error at `:163`.
Add a regression test for `|rg_init| > 0.999` with `pi_init=None`.

### m2. `res.h2` is the same-sweep sampled quadratic — inflated by posterior noise, undocumented

`bipred/bivariate.py:878-880` reports `h2_t = clamp(mean(β_t'Rβ_t))` over **sampled**
(not Rao-Blackwellized) effects. Given the data, `E[β'Rβ|D] = pm'R·pm + tr(R·PostCov)`
— upward-biased relative to the posterior-mean quadratic, the same inflation the docs
discuss for the rg denominator (`algorithm.md:112-114`) but never mention for `h2`,
which `BivariateResult` (`:452`) presents plainly as "the pair of SNP heritabilities".
Standard LDpred-family behavior, not a bug — but the estimand should be stated.

**Fix:** one sentence in `algorithm.md` noting `h2` is the mean sampled LD-adjusted
quadratic and is mildly upward-biased at low power.

### m3. `rg` fallback ratio uses unclamped sampled quadratics

`bipred/bivariate.py:895`. `h2` is clamped to `h2_bounds` (`:879-880`), but the
fallback `rg = g12/sqrt(max(g11*g22, 1e-12))` uses raw `g11`/`g22`. int8 quantization
can make a block non-PD (small negative eigenvalues), so `g11·g22` can go ≤ 0, the
`1e-12` floor turns it into `1e-6`, and `rg` slams to ±1 silently.

**Fix:** clamp `g11`/`g22` to `lo` (or a small positive floor) before the ratio,
consistent with the h2 reporting.

### m4. `PackedSymmetricInt8LD` rejection is undocumented

`bipred/bivariate.py:755-759` raises `NotImplementedError` for **both** `LowRankLD`
and `PackedSymmetricInt8LD` (tested at `tests/test_bivariate.py:429-440`), but
`README.md:90-91`, `docs/guide.md:35-36`, and `docs/algorithm.md:172-173` all mention
only `LowRankLD`. A user holding packed-int8 (D8T) blocks from
`ldpred3.pack_symmetric_int8_ld` — plausible, since the same pages tout int8 support —
hits an undocumented rejection.

**Fix:** "…rejects ldpred3's compact representations (`LowRankLD` and packed-int8
`PackedSymmetricInt8LD`); pass dense float or dense int8 blocks."

### m5. Equation 1's error model is too vague to verify

`docs/algorithm.md:8-12` — "`error_t` has variance determined by the effective sample
size." Omits (a) the cross-trait error covariance `cross_corr/√(N1·N2)` that the whole
sample-overlap section depends on, and (b) that the sampler's per-SNP conditional
likelihood uses the diagonal `I/N` approximation (`E11 = 1/nn1`,
`bivariate.py:250-252`) rather than the LD-correlated `R/N` of the full joint model.

**Fix:** state `Cov(error_1j, error_2j) = [[1/N1, cross_corr/√(N1N2)],
[cross_corr/√(N1N2), 1/N2]]` and note the per-SNP conditional approximation.

### m6. `noise_inflation` × `cross_corr` interaction is an undocumented modeling choice

`bipred/bivariate.py:817-818` deflates `n1e = n1/λ1`, and `_bivar_const` (`:252`)
then computes `E12 = cross_corr/√(n1e·n2e)` — holding the noise *correlation* fixed,
which inflates the cross-trait noise *covariance* by `√(λ1λ2)`. The learned excess
(reference-mismatch) noise is thereby implicitly treated as fully cross-correlated at
`cross_corr`; if λ captures LD-reference mismatch, that component need not be
cross-correlated across traits. Only matters when both options are combined.

**Fix:** document it, or compute `E12 = cross_corr/√(n1·n2)` from the undeflated N.

### m7. `benchmarks/README.md` misstates `bivariate_demo.py`'s data shape

`benchmarks/README.md:40-41` says `ld_library.npz` holds "100 blocks × 500×500
correlation matrices"; the script uses `K, NB = 500, 12`
(`benchmarks/bivariate_demo.py:18`) — 12 blocks of 500 variants.

**Fix:** "12 blocks × 500×500".

### m8. `benchmarks/README.md`'s `mixer_overlap.py` row is stale

`benchmarks/README.md:59` lists only "overlap / ρ_β / power sweeps", but the script now
has six — `overlap, rho, power, ldmatch, calibration, unical`
(`benchmarks/mixer_overlap.py:55`, dispatch at `:335-336`) — and `CHANGELOG.md:56-60`
advertises the new `unical` sweep and corrected `ldmatch` framing.

**Fix:** add the `ldmatch` / `calibration` / `unical` sweeps to the table row.

### m9. CHANGELOG "Notes" seam list is incomplete after the int8 change

`CHANGELOG.md:115-119` says bipred "imports `_jit`, `_as_n_vector` and `LowRankLD` from
`ldpred3.ldpred3`, and `_wls` / `_weights` from `ldpred3.ldsc`". The code
(`bipred/bivariate.py:36-55`) additionally imports `PackedSymmetricInt8LD`,
`_check_h2_p`, `_finite_control`, `_integer_at_least`, `_validate_beta_hat`,
`_validate_blocks`, `_validate_boolean_controls`, `_validate_iterations`,
`_validate_seed` from `ldpred3.ldpred3`, and `_Q8` from the private `ldpred3._kernels`.
This matters because `README.md:26-30` tells maintainers to advance the ldpred3 pin
"when the seam changes".

**Fix:** list the full seam, including `ldpred3._kernels._Q8`.

---

## Nits

| # | Location | Issue | Suggested fix |
|---|----------|-------|---------------|
| n1 | `docs/algorithm.md:87` vs `bivariate.py:843-844` | "`pi` drawn from a Dirichlet posterior with concentration `pi_prior`" — actual concentration is `pi_prior + state_counts` | "with prior concentration `pi_prior` plus the sweep's state counts" |
| n2 | `bipred/bivariate.py:84-106` | `_finite_scalar_or_pair` accepts numeric strings (`h2_init="0.1"` passes via 0-d object array), unlike sibling `_finite_pair` (`:66`) which rejects `str/bytes` | add the same `isinstance(value, (str, bytes))` guard; affects `h2_init`, `sigma_prior_scale` |
| n3 | `bipred/ldsc_rg.py:59` | `_as_sample_size` bool guard checks dtype before float conversion, so `[True, 1e4, 1e4]` (or object arrays containing bools) passes; scalar `True` / all-bool arrays are correctly rejected | check for bools after conversion, or inspect `np.asarray(value, dtype=object).flat` |
| n4 | `bipred/ldsc_rg.py:293-294` | `estimate_sample_overlap` on a non-result input raises bare `AttributeError` instead of a validated `ValueError` (docstring does specify `LDSCRgResult`) | validate or leave; cosmetic |
| n5 | `bipred/bivariate.py:500` (via `_mixer_dict`) | `frac_shared = pi11 / max(min(pi1, pi2), 1e-300)` per-iterate guard can yield absurd values in `mixer_iterate_summary` if a Dirichlet draw makes `min(p1,p2)` ≈ 0; unreachable for genome-wide m with `pi_prior ≥ 1`, matters only for toy m | trim/floor per-iterate ratios, or leave |
| n6 | `docs/guide.md:78` vs `bivariate.py:469,907` | `noise_scale` documented as present "if `noise_inflation=True`"; the field is always populated, `(1.0, 1.0)` when off | "always present; `(1.0, 1.0)` when `noise_inflation=False`" |
| n7 | `docs/guide.md:214` vs `bivariate.py:202-208,691-692,796-797` | `sample_every` row omits that it only applies to `rg_decorrelated=True` | append "(only with `rg_decorrelated=True`)" |
| n8 | `docs/rg.md:13` | "`rg_decorrelated` can be slightly high when traits are similarly powered" — no committed benchmark uses `rg_decorrelated`; `RESULTS.md` never evaluates it | soften to a non-quantified caveat or add a benchmark cell |
| n9 | `README.md:94`, `guide.md:39`, `algorithm.md:176`, `CHANGELOG.md:20-21`, `bivariate.py:633` | "matching ldpred3's default" is loose: int8/LR8 is ldpred3's *pipeline* default, but `ldpred3_by_blocks`' explicit `ld_int8` flag still defaults `False` (`ldpred3/ldpred3/ldpred3.py:693`) | "matching ldpred3's pipeline default" |
| n10 | `docs/algorithm.md:108-110` | "numerator and denominator are formed on the same shrinkage scale" — sampled quadratics are noise-inflated, not shrunk; the defensible argument is partial cancellation in the ratio | reword or drop the clause (see M1) |
| n11 | `ldpred3/ldpred3/ldsc.py:101` (cross-repo) | stale comment "`h2`: … (regression slope * M / N)" — with `x = N·ell/M` the slope *is* h2 directly; bipred's `ldsc_rg.py` is consistent with the correct convention | fix the comment in ldpred3 |

---

## Test-suite result and coverage gaps

**Result:** `117 passed in ~10 s`, zero warnings (`pytest tests/`, conda env `bipred`,
Python 3.14.6 + Numba). Includes bit-exact golden characterization tests for the
float32 and int8 paths and the LDSC-rg golden.

**Coverage gaps** (important untested behavior):

- `rg_init ∈ (0.999, 1)` shorthand init — the negative-`pi` path (m1); no test
  exercises `|rg_init| > 0.999` with `pi_init=None`.
- `rg_decorrelated=True` statistical behavior — only a monkeypatched plumbing test.
  Probe (m=2000, rg=0.8, N=100k/5k): decorrelated mean 0.765 vs sampled-quadratic
  0.746 — works and is slightly closer to truth, as designed, but unpinned.
- `cross_corr` with per-variant N (the per-SNP `_bivar_const` branch with `E12 ≠ 0`)
  — untested; probe runs and stays finite.
- `noise_inflation` with per-variant N — untested; probe runs, λ ≥ 1.
- `ldsc_rg(constrain_intercept=…)` — only invalid-value validation is tested; no
  recovery/golden pins the constrained fit (probe: rg=1.137 on the exactly-linear
  fixture, intercept fixed at 0, jackknife SE produced — plausible but unpinned).
- Single-variant fit (m=1, one-SNP block) — untested; probe works (rg=0, h2 at lower
  bound).
- `h2_bounds` clamping of reported h2 — only bound validation is tested, not the
  clamp's effect.
- Bit-identity claim that the hoisted `n_const` path equals the per-SNP path
  (docstring, `bivariate.py:320`) — untested.

---

## Verified correct

### Theory / sampler core (re-derived, algebraically exact)

- **Four-state likelihoods and sampling** (`bivariate.py:250-290, 360-416`): all four
  quadratic forms `q0..q3`, determinants, log-weights, per-state posterior
  means/variances, the both-state posterior `V = (E⁻¹ + Σ⁻¹)⁻¹` with correct Cholesky
  draw (shared `z1` via `L21`), and Rao-Blackwell accumulation — exact for
  `N(d; 0, E + S_slab)`. Positive-definiteness of all four state covariances is
  guaranteed by the 0.999 clamp.
- **Σ damped moment update** (`bivariate.py:853-862`): matches the documented "damped
  moment update shrunk toward a diagonal target"; at stationarity `E[S1] = n1c·s1`,
  so targets are convex combinations of ψ and the moment estimate; "inverse-Wishart-
  style" is an accurate hedge (posterior-mean form, not an IW draw — as the docs
  state).
- **Dirichlet π update** (`bivariate.py:842-844`): correct conjugate form.
- **Coherent initialization** (`bivariate.py:109-168` vs `docs/algorithm.md:51-69`):
  verified algebraically — implied `M·p1·s1 = h2_init_1`, `M·p2·s2 = h2_init_2`,
  implied `rg = rg_init` exactly on the reachable domain; the shared-mass bound
  `shared ≥ |rg|·q/(2·0.999 − |rg|)` is the correct solution of `|ρ_β| ≤ 0.999`;
  rejection at `|ρ_β| ≥ 1` matches the docs. The CHANGELOG's historical account
  ("previously implied h2 = 2/3·h2_init, rg = rg_init/2") verified against the parent
  commit's code.
- **Both rg estimators**: same-sweep `g12/√(g11·g22)` from `β'Rβ` quadratics matches
  Eq. 4; `_decorrelated_cov` (`bivariate.py:211-236`) is exactly the mean over ordered
  pairs a≠b (all-pairs sum minus diagonal, n(n−1) pairs) — the LDpred2 out-of-sample
  trick, correctly implemented.
- **MiXeR decomposition** (`bivariate.py:493-503`): `rg_from_overlap =
  ρ_β·π11/√(π1·π2)` is exactly the model-implied genetic correlation;
  `frac_shared = π11/min(π1,π2)` follows the MiXeR convention; `mixer_calibrated`'s
  anchoring `π11_cal = frac_shared·min(p1,p2)` is internally consistent and bounded;
  interval-honesty of `mixer_iterate_summary` matches the docs ("not Bayesian credible
  intervals").
- **`cross_corr`**: `E12 = cross_corr/√(N1N2)` with `cross_corr =
  N_s·ρ_pheno/√(N1N2)` is the standard Bulik-Sullivan intercept;
  `estimate_sample_overlap`'s inversion, the signed `effective_overlap`, and the
  "pheno_corr=1 gives a *lower* bound on N_shared" claim (`docs/rg.md:86-88`,
  `ldsc_rg.py:295-302`) are all correct.
- **`noise_inflation`** (`bivariate.py:831-839`): target `E[n·resid²] = 1` under
  matched LD is exact when residuals use sampled effects (posterior-predictive
  identity); damping and N-deflation into the likelihood are consistent.
- **Cross-trait LDSC** (`ldsc_rg.py:168-239`): `cross = z1·z2`,
  `xc = √(N1N2)·ell/M` so the slope is ρ_g; marginal h2 slopes consistent with the
  same M; Isserlis/genetic-covariance weights `Var(z1z2) = E[z1²]E[z2²] + E[z1z2]²`
  (correctly no factor 2, unlike univariate `1/(2μ²)`); constrained-intercept WLS
  algebra correct; contiguous delete-block jackknife SE matches the standard delete-1
  form; NaN-conservatism for non-positive h2 behaves as documented.
- **int8 claims** (`docs/algorithm.md:174-181`): `_Q8 = 127`, exact unit diagonal
  (`127·(1/127) == 1.0` in IEEE float64, so the leave-one-out identity is exact),
  ≤0.004 entry error, 4× memory saving; `ldpred3.compute_ld_blocks(quantize=True)`
  blocks consumed as-is; `LowRankLD`/`PackedSymmetricInt8LD` rejected as coded.
- **Burn-in/retained accounting**: `count == num_iter` invariant, `pi_samples`/
  `sig_samples` shapes, `n_saved = (num_iter−1)//sample_every + 1` — no off-by-ones.
- **Validation**: `|cross_corr| < 1` strictly enforced; `dS > 0` via the 0.999 clamp
  and `s1,s2 ≥ 1e-12`; `iw_df > 0` keeps `(ν0 + c11) > 0` when `c11 = 0`. Deliberate
  non-validation of `p_init` when `pi_init` is given is enshrined by a test
  (`tests/test_bivariate.py:225-230`) — intentional, not a gap.
- **`bipred/__init__.py`** lazy exports exactly match both modules' `__all__`.

### Documentation (verified accurate)

- **Quickstarts run as written** (by signature/attribute matching):
  `ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n1, n2[, seed=0])`,
  `ldpred3_auto_bivariate_blocks(..., burn_in=200, num_iter=200, seed=0,
  rg_decorrelated=True, cross_corr=...)`, `ldsc_rg(beta_hat1, beta_hat2, ell, n1, n2)`
  with `rgr.rg/.rg_se/.gcov_intercept`, and `estimate_sample_overlap(rgr, n1, n2,
  pheno_corr=0.4)` all match the real signatures; all referenced result attributes
  exist.
- **Every default in `guide.md` Table 2 matches code** (`bivariate.py:616-624`):
  `ld_int8=True`, `burn_in/num_iter=200`, `h2_init=0.1`, `p_init=0.02`, `rg_init=0.0`,
  `pi_init=None`, `sigma_prior_scale=None`, `cross_corr=0.0`, `rg_decorrelated=False`,
  `noise_inflation=False`, `ni_damp=0.1`, `pi_prior=1.0`, `h2_bounds=(1e-4, 1.0)`,
  `h2_cap=None`, `iw_df=10.0`, `sample_every=5`, `seed=None`.
- **Coherent-initialization docs are exact**: `algorithm.md` Eq. 2–3 match
  `bivariate.py:148-167` line-for-line; `p_init` correctly documented as union-causal;
  `pi_init` semantics match `:137-152`.
- **mixer/iterate-summary docs**: dict keys and formulas match `_mixer_dict`
  (`bivariate.py:493-503`); `mixer_posterior()` is a real deprecated alias emitting
  `DeprecationWarning` (`:575-588`).
- **Benchmark quotes vs `RESULTS.md`**: rg.md's "roughly unbiased, ~1.5–2× more
  precise than LDSC" (SD 0.063 vs 0.120) and the "failed corrected cell at r_g=0,
  r_e=0.6" (0.7402 ± 0.4181) are consistent; `agg()`'s `|rg| > 1.5` exclusion
  confirmed in `rg_env_overlap.py:92-94`; all scripts named in the docs exist.
- **Packaging claims**: Python 3.9–3.14, Numba ≥0.66 on 3.14, `[sim]`/`[bench]`
  extras all match `pyproject.toml`.

---

## Verification caveats

- The ldpred3-side seam was verified against the sibling checkout at `81899b5`, which
  is **newer** than the README-pinned `3444da1`. The pinned revision could not be
  fetched (raw.githubusercontent 404 — likely a private-repo artifact), so whether the
  pinned commit actually contains `PackedSymmetricInt8LD`, `_Q8`, and the validator
  helpers is **unverified**. If it predates them, the README's install pin is broken —
  worth a manual check (`git -C ldpred3 log --oneline 3444da1 -1` once the remote is
  reachable, or fetch the pin and `pip install` it in a clean env).
- Quickstart snippets were verified by signature/attribute matching, not by executing
  the sampler end-to-end.
- Statistical correctness claims ("verified correct" above) are algebraic/numerical,
  backed by the passing golden and recovery tests; no new simulation study was run for
  this review beyond the small probes noted under coverage gaps.

---

## Recommended actions (priority order)

1. **M1** — fix the `rg_decorrelated` mechanism wording in `docs/algorithm.md`
   (and the "same shrinkage scale" clause, n10).
2. **M2** — reconcile `CHANGELOG.md:93-97` with the committed calibration artifacts.
3. **m1** — clamp or reject `shared > q` in `_initial_hyperparameters`; add the
   `|rg_init| > 0.999` regression test.
4. **m3** — clamp `g11`/`g22` before the fallback rg ratio.
5. **m2, m4, m5, m6** — short doc additions (`h2` estimand, `PackedSymmetricInt8LD`
   rejection, Eq. 1 noise covariance, noise_inflation × cross_corr note).
6. **m7–m9** — refresh `benchmarks/README.md` rows and the CHANGELOG seam list.
7. Add coverage for the gaps above, starting with `rg_decorrelated` statistical
   recovery, `cross_corr`/`noise_inflation` with per-variant N, and m=1.
8. Verify the README ldpred3 pin contains the current seam (see caveats).

---

## Resolution (2026-07-18, same day)

All findings above were addressed in the working tree. Test suite after the
changes: **128 passed** (117 pre-existing + 11 new), zero warnings; the
bit-exact golden values are unchanged.

### Code

- **m1** — `_initial_hyperparameters` now saturates the shared mass at the
  union probability (`shared = min(shared, q)`): an all-shared start for
  `|rg_init| > 0.999`, a valid probability vector, implied moments still exact
  (`rho_beta = rg_init`, strictly inside `(-1, 1)`; the sampler's 0.999 PD
  clamp applies from the first covariance update). Covered by
  `test_initial_hyperparameters_extreme_rg_saturates_shared` (0.999, ±0.9999).
- **m3** — the fallback `rg` ratio now uses the clamped reported-`h2` scale
  (`g12 / sqrt(h2_1 * h2_2)`) instead of `max(g11 * g22, 1e-12)`, so non-PD
  int8 blocks can no longer slam `rg` to ±1 through the floor.
- **n2** — `_finite_scalar_or_pair` rejects `str`/`bytes` scalars and elements.
- **n3** — `ldsc_rg._as_sample_size` validates element-wise on an object-array
  view, rejecting mixed bool/string sequences.
- **n4** — `estimate_sample_overlap` requires an `LDSCRgResult` (clear
  `ValueError` instead of `AttributeError`).
- **n5** — intentionally left as-is: cosmetic for toy `m` only.

### Documentation

- **M1** — `docs/algorithm.md` now states the correct mechanism (same-sweep
  coupling *inflates* the genetic covariance; the sampled-quadratic ratio
  attenuates through the weak trait's inflated sampled variance). The "same
  shrinkage scale" clause (n10) was replaced by the cancellation argument.
- **M2** — the CHANGELOG noise-inflation entry now quotes the committed
  `calibration` sweep (N = 1k-20k, inflation ~1.2× off → ~1.0× on,
  λ ≈ 1.1-1.2).
- **m2** — `algorithm.md` states the `res.h2` estimand (mean sampled quadratic,
  clamped to `h2_bounds`, mildly upward-biased at low power).
- **m4** — `PackedSymmetricInt8LD` rejection documented in README, guide,
  algorithm.md, and the `ldpred3_auto_bivariate_blocks` docstring.
- **m5** — Equation 1 now carries the 2×2 noise covariance and the per-SNP
  diagonal approximation.
- **m6** — `algorithm.md` documents that deflated N also scales `E12`.
- **m7, m8** — `benchmarks/README.md` corrected (12 blocks; six sweeps).
- **m9** — CHANGELOG Notes now list the full seam incl. `ldpred3._kernels._Q8`.
- **n1, n6, n7, n8, n9** — Dirichlet concentration wording, `noise_scale`
  always present, `sample_every` scoped to `rg_decorrelated`, softened
  `rg_decorrelated` caveat, "pipeline default" phrasing — all applied.
- **n11** — stale slope comment fixed in `ldpred3/ldpred3/ldsc.py:101`.
- A `### Fixed` entry summarizing the user-visible changes was added to
  `CHANGELOG.md` under `[Unreleased]`.

### Tests added

- `test_initial_hyperparameters_extreme_rg_saturates_shared` (m1 regression)
- string-input validation rows for `h2_init` / `sigma_prior_scale` (n2)
- mixed-bool / string rows for `ldsc_rg` sample sizes (n3)
- `test_estimate_sample_overlap_requires_result_type` (n4)
- `test_rg_decorrelated_recovers_rg_for_asymmetric_power` (statistical gap)
- `test_cross_corr_with_per_variant_n` and
  `test_noise_inflation_with_per_variant_n` (per-variant-N gaps)
- `test_single_variant_fit_is_well_formed` (m=1, both entry points)
- `test_ldsc_rg_constrain_intercept_recovers_rg` (constrained-fit recovery)

### Remaining / not addressed

- **n5** (`frac_shared` per-iterate guard) — left as-is (see above).
- The scalar-N hoisting bit-identity claim (`bivariate.py` docstring) remains
  unpinned by a test.
- `rg_decorrelated` now has a statistical *test* but still no benchmark cell;
  the docs caveat was softened instead (n8).

### ldpred3 pin verification (caveat resolved)

The README-pinned revision `3444da1` is present in the sibling clone and
contains the complete private seam — `LowRankLD`, `PackedSymmetricInt8LD`,
`_as_n_vector`, `_check_h2_p`, `_finite_control`, `_integer_at_least`, `_jit`,
`_validate_beta_hat`, `_validate_blocks`, `_validate_boolean_controls`,
`_validate_iterations`, `_validate_seed` from `ldpred3.ldpred3`; `_Q8` from
`ldpred3._kernels`; `_wls` / `_weights` from `ldpred3.ldsc`. The install pin
is intact.
