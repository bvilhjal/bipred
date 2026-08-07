# Changelog

User-visible changes to **bipred** are recorded here. The project is currently
`0.2.1`.

## [Unreleased]

### Performance

- **Block parallelism silently stopped working after any serial run.** The two
  fused sweep drivers are each jitted twice from one Python function —
  `parallel=True` and `nogil=True` — and Numba keys its on-disk cache on
  (source file, qualname, first line, signature) but *not* on the compilation
  flags. The twins therefore shared one cache entry, and whichever compiled
  first was served to both. Since the cache lives in `__pycache__` beside
  `bivariate.py` and persists, one `ncores=1` run disabled `ncores>1` for every
  later run on that checkout. Measured at m=20,000 / k=500: `ncores=4` ran at
  1.73 ms/sweep from a clean cache and 5.38 — no better than the 5.49 serial
  baseline — from a cache a serial run had touched. The parallel twins now opt
  out of the on-disk cache, which restores 1.77 ms/sweep bit-identically, at
  the cost of one compilation per process when `ncores>1`.
- **The low-rank sweep kernel is compiled with `fastmath`**, mirroring ldpred3's
  scoping (`_kernels.py:1277`): its O(rank) projection dots are the bulk of a
  low-rank sweep and are add-latency-bound, so letting LLVM reassociate and
  vectorise the reduction is most of the available win. Measured at m=20,000,
  k=500, rank 481: **1.83x** on LR8 (14.05 -> 7.68 ms/sweep) and **2.57x** on
  float32 low-rank (12.43 -> 4.84). Results move at the reassociation level
  (`rg` by 2e-16 relative). The dense kernel is deliberately left plain — it
  measured only 1.12x, because its guarded row update fires on a few per cent of
  visits and the sweep is dominated by the four `exp()` calls instead.
- **Per-variant `n_eff` recomputes the sweep's residual-independent scalars only
  when N changes**, rather than once per variant. `_bivar_const` is ~29
  quantities including four logs, and real summary statistics carry long runs of
  identical `n_eff`. It is a pure function of its arguments, so a memo hit is
  bit-identical. Measured 1.16x on a dense per-variant-N fit with runs of 250;
  the constant-N path is unchanged.
- `benchmarks/sweep_cost.py` measures per-sweep cost by representation and core
  count, giving each cell a private Numba cache in a subprocess — without that
  isolation a grid measures its first arm repeatedly, for the reason above.

### Changed

- **`ld_int8` now defaults to `False`: dense blocks are consumed in the
  representation they arrive in.** The previous default quantised float blocks
  of at most 1,500 variants *inside the fit*, allocating a second genome-scale
  int8 payload while the caller's panel was still alive. Measured peak inside
  the call at m=100,000 / k=500: **78.4 MB before, 13.1 MB after** — the extra
  payload is k/2 bytes per variant, so roughly 500 MB at m=1,000,000. This
  follows ldpred3, whose fit-time default is also `False` and which quantises at
  LD-build time, where the float source is private and discardable: prefer
  `ldpred3.compute_ld_blocks(quantize=True)`. `ld_int8=True` and `None` are
  retained for the old behaviour. **This changes results** for callers who
  relied on the default with float blocks, by the int8 quantisation resolution
  (~4e-3 on LD entries) and in the direction of more accuracy.
- `regional_rg` no longer warns when handed float blocks at or below the fit's
  old auto-quantise cutoff. That warning existed because the fit-time default
  quantised them into a private copy; with the default consuming blocks as
  given, the pattern it flagged — the same float blocks to both calls — is now
  the aligned one, and the warning was firing on the correct usage.

### Fixed

- **`rg` no longer saturates against `h2_bounds`.** The reported genetic
  correlation divided the raw genetic covariance by the *clamped* per-trait
  heritabilities, so a binding bound drove `rg` toward ±1 while the underlying
  quadratics were unchanged — on one 2,000-variant fixture a true `rg` of 0.43
  was reported as 1.00 after tightening `h2_bounds`. `rg` is now the ratio of
  the raw quadratics, exactly as `docs/algorithm.md` Equation 6 defines it;
  `h2` is still clamped for reporting. The same correction applies to the
  adaptive-stopping check, the pooled multi-chain `rg`, and the per-draw `rg`
  trace feeding split-Rhat (where saturation also faked between-chain
  agreement). Fits whose `h2` stayed inside its bounds are unaffected.
- **`from bipred import estimate_sample_overlap, ldsc_rg` bound the module, not
  the function**, so the documented snippet in `docs/rg.md` raised
  `TypeError: 'module' object is not callable`. The LDSC module was named
  `ldsc_rg`, colliding with the `ldsc_rg` function it exports, and which one
  `bipred.ldsc_rg` resolved to depended on import order. The module is now
  `bipred/ldsc.py`; the public API is unchanged. Submodules
  (`bipred.bivariate`, `bipred.ldsc`, `bipred.multichain`, `bipred.regional`)
  are now reachable as attributes and appear in `dir(bipred)`, and a name
  collision is asserted against at import. `import bipred.ldsc_rg` no longer
  resolves; the function `bipred.ldsc_rg` and every other public name are
  unchanged.
- A diverged chain raised through `ldpred3_auto_bivariate_chains` now keeps its
  `FloatingPointError` type instead of being flattened into `RuntimeError` by
  the chain wrapper, matching the type `_validated_chain_traces` already raises
  for a non-finite trace.
- A diverged fit now raises `FloatingPointError` instead of returning NaN or a
  floored `h2`. NaN was not self-announcing in the sweep: the log-sum-exp left
  `wmax` at the first state, every state probability became NaN, and each
  variant fell through to the both-causal branch, so `h2`, `rg`, `sigma` and
  both effect vectors came back NaN with no error. The *implausible fit*
  warning is also two-sided now — a floored `h2` was previously silent — and
  reads the raw quadratics, since a clamped `h2` cannot show that a bound was
  reached. The low end distinguishes the two cases: a non-positive sampled
  genetic variance is degenerate and also reports `rg` as 0, while a small but
  strictly positive quadratic under the caller's own floor only means the
  reported `h2` is clamped. The warning is still suppressed below 1,000
  variants.
- The benchmark simulator resolved its backend twice: frozen into
  `SIMULATOR_CACHE_TAG` at import, then re-probed per call. `import msprime` is
  not idempotent — a failed attempt can leave partial state that lets a retry
  succeed — so segments could be simulated by msprime and cached under the
  `numba` tag. The backend is now resolved once per process.
- `rg_decorrelated=True` no longer aborts an otherwise usable fit when the
  cross-sweep genetic-variance estimate is degenerate. A non-positive variance
  (exactly zero in finite samples, or slightly negative — e.g. a sparse, weakly
  powered fit whose retained states share no causal support) now warns and
  reports `rg` as `NaN` rather than raising `ValueError`; only non-finite
  cross-sweep quadratics still raise. This removes a platform-dependent crash:
  the degenerate case is reachable through floating-point ordering differences
  (observed on Windows for a `seed` that is fine on the Linux CI).
- The seam gate `test_private_ldpred3_imports_stay_centralised` now reads each
  module as UTF-8 instead of the platform default, so it no longer raises
  `UnicodeDecodeError` on a cp1252 (Windows) checkout over the Greek letters in
  the LDSC module (`bipred/ldsc.py`, then named `ldsc_rg.py`).
- `regional_rg` rejects a float `regions` array containing non-finite labels
  (`NaN`/`inf`) instead of silently pooling every `NaN`-marked variant into one
  spurious cross-genome region — matching the existing rejection of `None`
  labels in object arrays.

### Changed

- The `ldpred3` dependency is now the range `>=0.4.3,<0.5` rather than
  `==0.4.3`. ldpred3 is on no package index, so the exact specifier sent pip
  looking for a distribution it can never find whenever the installed version
  differed — breaking the README's own sibling-checkout development recipe
  against an ldpred3 tree past the pin. The exact tested revision still lives
  in the README install command and in CI's `LDPRED3_REV` plus its version
  assertion.
- The weekly `ldpred3-head` drift-watch CI leg installs bipred with
  `--no-deps`. It deliberately runs an ldpred3 whose version differs from
  bipred's declared range, so letting pip re-resolve the dependency failed at
  resolution and the leg never reached `pytest` — silently, because it is
  `continue-on-error`.
- Documentation corrections: `docs/rg.md` said a degenerate `rg_decorrelated`
  fit raises (it warns and returns NaN); `docs/guide.md` and `docs/rg.md`
  offered "pass the fit's prepared blocks" as a remedy with no public API
  behind it; `docs/guide.md` Table 2 now marks which options the chains driver
  accepts, rejects, or renames, documents `sample_every`, and gives the chains
  `seed` contract; `ldsc_rg` documents that it is one-step and unfiltered (no
  chi-square cap, so single large-effect loci keep full leverage) and that
  `m_snps` and `ld_scores` must describe the same variant map;
  `RegionalRgResult` no longer claims regions can be aggregated by summing
  (that drops cross-region LD within a block).
- `benchmarks/RESULTS.md` Tables 4–6 were still printing 0.2.0 timing and
  peak-memory numbers against regenerated CSVs, including a 2.51 GB memory
  spike at 80k variants that the current data does not reproduce. Accuracy
  columns were correct and unchanged. A test now re-derives Table 6 from
  `rg_scaling.csv` cell by cell; Tables 4 and 5 were corrected by hand and
  remain unguarded.
- The README gained a citation section pointing at LDpred/LDpred2, cross-trait
  LDSC, and MiXeR.
- Passing `tol > 0` together with `rg_decorrelated=True` to the single-chain
  `ldpred3_auto_bivariate[_blocks]` now emits a `RuntimeWarning` and is
  documented as a no-op (the thinned decorrelated-rg estimator needs the full
  retained schedule). Previously the positive `tol` was silently ignored, while
  the multichain driver already rejects the pairing.

## [0.2.1] - 2026-08-04

Review-hardening release: no estimator changes, no public-API changes.

### Added

- A seam-centralisation gate: `test_private_ldpred3_imports_stay_centralised`
  fails if any underscore-private `ldpred3` import lives outside
  `bipred/_ldpred3_compat.py` (previously uncovered modules such as
  `multichain.py` are now scanned).
- A weekly `ldpred3-head` CI leg (also manually dispatchable) that runs the
  suite against `ldpred3@master` as an informational drift early-warning, plus
  a pin-bump checklist in the README.
- `regional_rg` now warns once when handed a float dense block at or below the
  fit's auto-quantise cutoff — the common call pattern that evaluates different
  LD than a default fit used — naming the three ways to keep the
  representations aligned. Silent for int8, larger float blocks, and
  `LowRankLD`.
- A dependency-free fallback for benchmark simulation: ldpred3's bundled
  Numba coalescent, vendored as `benchmarks/_coalescent.py`, used when
  msprime is absent. msprime stays the default where installed (measured
  0.80 s against 1.64 s for a 10,000-sample, 5 Mb segment), and cached
  segments are tagged per backend so the two never mix.

### Changed

- `rg_decorrelated`'s documentation now leads with "sensitivity diagnostic
  only — do not use for production estimates", carries the measured MAE from
  `RESULTS.md` Table 4, and states the multichain/adaptive-stopping
  incompatibility.
- The environmental-overlap failure mode (joint-fit MAE up to 0.86 in the 0.2.0
  stress test, `RESULTS.md` Table 10) is cross-referenced wherever
  `cross_corr` is documented, with the set-from-external-evidence instruction.
- `test_public_api.py` asserts the exact 13-name public surface instead of
  non-None resolution.
- Benchmark evidence regenerated against ldpred3 0.4.3 with this release.

## [0.2.0] - 2026-08-03

### Added

- Bivariate LDpred for dense and blockwise LD:
  `ldpred3_auto_bivariate`, `ldpred3_auto_bivariate_blocks`, and
  `BivariateResult`.
- Four-state polygenic-overlap summaries, retained-iterate summaries, optional
  residual-noise inflation, and calibration against two univariate ldpred3
  fits.
- Cross-trait LD Score regression (`ldsc_rg`) and the
  `estimate_sample_overlap` helper.
- Deterministic multi-chain fitting with dispersed starts, equal-weight pooling,
  basic split-Rhat diagnostics, and optional chain concurrency through
  `chain_ncores`.
- Regional exploratory genetic correlation (`regional_rg`) from posterior-mean
  effects. Results expose regional covariance, variances, and variant counts;
  known sample-overlap contamination and genome-wide shrinkage are documented.
- Optional single-chain adaptive stopping through `tol` and `check_every`, with
  explicit retained-iteration metadata. This is a stabilization heuristic, not
  a convergence test.

### Changed

- Coherent initialization now makes `h2_init`, `p_init`, and `rg_init` match the
  implied starting moments. `pi_init` exposes overlap directly, and
  `sigma_prior_scale` can keep the shrinkage target fixed across starts.
- The default union-causal start is `p_init=0.02`.
- Dense-LD storage defaults to `ld_int8=None`: supplied int8 remains int8,
  float blocks up to 1,500 variants are quantized, and larger blocks remain
  float32.
- Compact float and LR8 `LowRankLD` blocks use ldpred3's current scalar-scale,
  float32-residual contract without expanding per-variant float64 adapters.
- With Numba, one-core fits fuse each homogeneous block bucket into one native
  call; `ncores>1` uses the parallel twin while preserving seeded results.
- Private ldpred3 helpers are routed through one lazy compatibility module;
  `LowRankLD` is imported from ldpred3's public API.
- The tested ldpred3 dependency revision is
  `5436dcc8152531000b223c5088f726d588d8a8cd` (`ldpred3==0.4.3`).
- Repository benchmark artifacts were regenerated against 0.2.0. They now
  separate generating targets from realized LD-adjusted genetic correlation,
  report paired errors and failures, and directly compare the default and
  cross-sweep sensitivity estimators.

### Removed

- `BivariateResult.mixer_posterior` (deprecated alias of
  `mixer_iterate_summary`) and the unreferenced `LDSCRgResult.rg_ci` property.
- `ldsc_rg`'s local scalar/sample-size validators in favour of the ldpred3
  compatibility seam (`_as_n_vector`, `_finite_control`, `_integer_at_least`).
- The obsolete `allow_legacy_lowrank` fitting and regional options; rebuild old
  row-normalized factors with the pinned ldpred3 version.
- Benchmark scripts and result CSVs are no longer shipped in the sdist; they
  remain in the repository with their README.

### Fixed

- The sdist check now matches the repository-only benchmark policy, and shipped
  documentation links to benchmark material through repository URLs.
- `rg_decorrelated=True` no longer silently substitutes the default estimator
  when its cross-sweep estimate is unavailable.
- Sample-overlap documentation now distinguishes scalar cohort counts from
  effective/sample-varying `N` and states the `noise_inflation` interaction.
- Cross-trait LDSC documents the genome-order requirement for block-jackknife
  standard errors, and `regional_rg` states exactly which supplied LD it
  evaluates and reports undefined ratios for non-positive regional variances.
- Cross-trait LDSC recovers exact signed z statistics from standardized effects
  (`|beta_hat| < 1`); its genotype benchmark supplies exact chi-square input.
- Repository benchmarks use their own lazy-msprime simulator after ldpred3
  removed simulation helpers from its installed package, and tag cached
  segments by simulator schema.
- Within-chain parallel fits restore the caller's Numba thread mask, including
  when a sweep raises.
- Multi-chain fitting rejects adaptive stopping instead of failing later on
  unequal trace lengths; `rg_decorrelated=True` remains unsupported there.
- Input validation now rejects invalid booleans, non-finite controls, malformed
  regional vectors, and non-positive LD scores at public boundaries.
- The fallback `r_g` ratio uses the reported heritability scale, avoiding
  boundary artifacts from non-positive-definite quantized LD.
- Public exports and the private ldpred3 compatibility seam have behavioral
  regression tests.

### Performance

- Per-sweep random and residual buffers are reused.
- Mixed LD panels use fused serial or block-parallel sweeps; regional
  quadratics and LDSC jackknife paths avoid repeated gathers and masks.
- Decorrelated `r_g` stores O(M) online sufficient statistics instead of
  O(retained samples × M) effect traces.
- All-low-rank fits without noise inflation omit two unused genome-length
  residual vectors.

Performance depends on hardware, block balance, LD representation, and current
code. Reproduce the benchmark suite rather than treating development anecdotes
as release guarantees.
