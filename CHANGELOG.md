# Changelog

User-visible changes to **bipred** are recorded here. The project is currently
`0.2.2`.

## [0.2.2] - 2026-08-07

Performance and benchmark-evidence release. The sampler's estimates are
unchanged except where a defect made them wrong; the one deliberate behaviour
change is the `ld_int8` default, which also retires the package's worst
documented failure mode. Every benchmark artifact in `benchmarks/` was
regenerated for this version from a single sweep.

### Performance

- **Block parallelism silently stopped working after any serial run.** The two
  fused sweep drivers are each jitted twice from one Python function —
  `parallel=True` and `nogil=True` — and Numba keys its on-disk cache on
  (source file, qualname, first line, signature) but *not* on the compilation
  flags. The twins therefore shared one cache entry, and whichever compiled
  first was served to both. Since the cache lives in `__pycache__` beside
  `bivariate.py` and persists, one `ncores=1` run disabled `ncores>1` for every
  later run on that checkout. Measured at m=20,000 / k=500 on LR8: from a cache a
  serial run had touched, `ncores=4` ran at 9.50 ms/sweep against a 9.54 serial
  baseline — no scaling whatever — where a clean cache gave 2.76. The parallel
  twins now opt out of the on-disk cache, which restores full scaling (1.16
  ms/sweep against a 1.12 private-cache reference) bit-identically, at the cost
  of one compilation per process when `ncores>1`.
- **The low-rank sweep kernel is compiled with `fastmath`**, mirroring ldpred3's
  scoping (`_kernels.py:1277`): its O(rank) projection dots are the bulk of a
  low-rank sweep and are add-latency-bound, so letting LLVM reassociate and
  vectorise the reduction is most of the available win. Measured at m=20,000,
  k=500, rank 481, serial: **2.34x** on LR8 (9.46 -> 4.03 ms/sweep) and
  **3.79x** on float32 low-rank (9.36 -> 2.47). Results move at the
  reassociation level (`rg` by 2e-16 relative). The dense kernel is deliberately
  left plain: its O(k) row update is guarded on a variant's effect changing, so
  at a realistic causal fraction it fires on about 1% of visits and the sweep is
  dominated by the four `exp()` calls instead.
- **Per-variant `n_eff` recomputes the sweep's residual-independent scalars only
  when N changes**, rather than once per variant. `_bivar_const` is ~29
  quantities including four logs, and real summary statistics carry long runs of
  identical `n_eff`. It is a pure function of its arguments, so a memo hit is
  bit-identical. Measured **1.69x** on a dense per-variant-N fit with runs of 250
  (1.442 -> 0.853 ms/sweep), which brings per-variant `n_eff` to parity with a
  shared scalar N (0.833) — the penalty for varying N is essentially gone. The
  constant-N path is unchanged.
- **int8 low-rank (LR8) factors are widened into a float32 scratch once per
  sweep**, ported from ldpred3 0.4.5. The projection dots read `U[j, c]` for
  every element of every O(rank) dot, for every variant, every sweep, paying an
  int8 sign-extend-and-convert each time; widening once per block amortises it.
  The scratch is one stride per *thread*, so it stays O(k × rank) and int8
  remains the storage format. Measured **4.03 → 2.97 ms/sweep (1.36x)** at rank
  481. Cumulative with `fastmath`, **9.46 → 2.97 (3.19x)**.
  The conversion is exact, so the fit moves only where `fastmath` reassociates
  (1.1e-15 relative), and both drivers widen, so seeded results remain identical
  across `ncores`.

  The rank gate is bipred's own measurement, not ldpred3's: upstream gates at 64
  because rank 32 was a loss for its kernel, whereas here widening won at every
  rank tested (1.07x at 16, 1.16x at 32, 1.31x at 128, 1.43x at 256), so the
  gate is 32.
- `_pinned_numba_threads` now touches Numba's threading layer even at
  `ncores=1`. The low-rank kernel references `_get_thread_id`, and loading it
  from the on-disk cache before that layer exists **segfaults the interpreter**
  — reproduced on numba 0.66 as an exit-139 on a warm cache with a serial-only
  run. This is why ldpred3 pins the mask unconditionally.
- `benchmarks/sweep_cost.py` measures per-sweep cost by representation and core
  count, giving each cell a private Numba cache in a subprocess — without that
  isolation a grid measures its first arm repeatedly, for the reason above.
  `benchmarks/sweep_cost.csv` records the grid at m=20,000 / k=500.

### Changed

- **`ld_int8` now defaults to `False`, which also resolves the package's worst
  documented failure mode.** The environmental-overlap stress test
  (`benchmarks/RESULTS.md` Table 10) recorded joint-fit MAE up to 0.86 under
  strong shared environment, and that was read as a limit of the model. It was
  not: it was in-fit LD quantization. Re-running the current code with
  `ld_int8=True` reproduces the old row byte for byte (0.1733 / 0.0166 / 0.3101
  / 0.8343 / 0.8603); with the float32 default every cell lands between 0.0072
  and 0.0242, a 36x improvement in the worst one. The int8 resolution (~4e-3 per
  LD entry) is enough to destabilize a fit whose conditioning is already
  stressed, while costing nothing measurable on the well-conditioned panels of
  Tables 2-7 (joint MAE across all 30 architecture cells moved 0.0122 to 0.0123).

  Mechanically, dense blocks are now consumed in the representation they arrive
  in. The previous default quantised float blocks
  of at most 1,500 variants *inside the fit*, allocating a second genome-scale
  int8 payload while the caller's panel was still alive. Measured peak inside
  the call at m=100,000 / k=500: **78.4 MB before, 13.1 MB after** — the extra
  payload is `k` bytes per variant -- measured 121 bytes/variant by default
  against 621 with `ld_int8=True` at k=500 -- so roughly 500 MB at m=1,000,000. This
  follows ldpred3, whose fit-time default is also `False` and which quantises at
  LD-build time, where the float source is private and discardable: prefer
  `ldpred3.compute_ld_blocks(quantize=True)`. `ld_int8=True` and `None` are
  retained for the old behaviour. **This changes results** for callers who
  relied on the default with float blocks, by the int8 quantisation resolution
  (~4e-3 on LD entries) and in the direction of more accuracy.
- **The pinned ldpred3 revision moves to `5d86ac9` (0.4.5)** and the dependency
  floor to `>=0.4.5,<0.5`. The seam was audited against it first, as the README
  requires: all eighteen borrowed symbols still resolve. The bump is what makes
  `_get_thread_id`, and therefore the LR8 widening above, available.
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
