# Bipred speed review — 7 September 2026

There is room for another modest sampler speedup and a useful reduction in memory. The strongest combined prototype ran **1.14–1.17× faster** in the main probes: **12–14% less fit time**, with **9–10% less process CPU time**. Separately, LDpred3's cache-identity fix removes repeated compilation of the parallel driver. These are research prototypes; production source is unchanged.

The timings are provisional. A concurrent SMARTpred job was using roughly three cores during the later experiments. Five repetitions, alternating execution order, and process CPU timings help interpret the result, but do not replace a quiet-machine benchmark. Nothing here establishes mixing or calibration of the timed chains.

**Table 1. Main warm-fit measurements.** Each panel contains 100,000 variants in 200 distinct, independently stored blocks of 500 variants; LR8 rank is 32. All arms use one core, 50 burn-in and 150 retained sweeps, identical inputs and seeds. Times are medians of five runs; ± is the unscaled median absolute deviation. “Driver” combines ordered compiled reductions and direct effect-mean accumulation. “Combined” also reuses the conditional means in the likelihood calculation.

| LD | Arm | Input preparation, s | Fit, s | Fit CPU, s | Public call total, s | Fit speedup |
|---|---|---:|---:|---:|---:|---:|
| Dense int8 | Current | 0.090 | 1.538 ± 0.051 | 1.265 | 1.602 | 1.00× |
| Dense int8 | Driver | 0.054 | 1.520 ± 0.010 | 1.175 | 1.574 | 1.01× |
| Dense int8 | Combined | 0.045 | 1.316 ± 0.020 | 1.137 | 1.387 | **1.17×** |
| LR8 | Current | 0.036 | 1.489 ± 0.124 | 1.199 | 1.523 | 1.00× |
| LR8 | Driver | 0.036 | 1.406 ± 0.014 | 1.184 | 1.452 | 1.06× |
| LR8 | Combined | 0.044 | 1.304 ± 0.028 | 1.094 | 1.353 | **1.14×** |

Input preparation is unchanged by the candidates: its variation illustrates the timing noise. Separately calculated medians need not add to the median total. “Fit” includes chain-workspace setup, sweeps and finalization; compilation is warmed first. Fixture construction took 0.089 s for the dense panel and 0.066 s for LR8 and is excluded from the public call. This is synthetic fixture construction, not measured construction of LD from genotypes. The corresponding whole-process peaks were 214 and 171 MiB, including construction, compilation and all candidate arms; they are not comparative fit-memory measurements. Full repetitions and summaries are in the [evidence archive](/Users/au507860/REPOS/bipred/benchmarks/results/20260907-speed-review/summary.json).

## Changes worth implementing

**1. Accumulate the posterior effect means directly and compile the ordered block reduction.** The current [chain driver](/Users/au507860/REPOS/bipred/bipred/bivariate.py:1747) fills two per-sweep vectors, adds them into two running sums, and reduces nine block statistics in Python. The prototype lets the kernels update the running sums directly, clearing discarded burn-in contributions once before the first retained sweep. A small strict-math compiled loop reduces block statistics in the original genomic order. This removes two float64 vectors and repeated passes over them, while retaining the RNG sequence and reduction order.

**Equation 1. Persistent workspace saved.**

\[
\Delta M=2\times8m=16m\ \text{bytes per active chain}.
\]

That is **15.3 MiB per million variants per chain**, or 61 MiB across four simultaneous chains. This is an allocation count, not a measured RSS reduction. All result fields were bit-identical in the tested driver-only cases. Runtime savings from this change alone are modest; the memory reduction is the clearer benefit.

**2. Reuse posterior means to calculate the four state likelihoods.** Both [dense](/Users/au507860/REPOS/bipred/bipred/_bivar_kernels.py:197) and [low-rank](/Users/au507860/REPOS/bipred/bipred/_bivar_kernels.py:367) kernels calculate four Mahalanobis quadratics before calculating the conditional effect means. Gaussian conditioning gives the following equivalent weights relative to the null state.

**Equation 2. Relative Gaussian state weight, derived for this review.** Let \(E\) be the two-trait sampling covariance, \(S_s\) the slab covariance of state \(s\), \(d\) the residual marginal effects, \(g=E^{-1}d\), and \(\mu_s=S_s(E+S_s)^{-1}d\). Then

\[
\log\widetilde w_s
=\log\pi_s-\tfrac12\log\frac{|E+S_s|}{|E|}
+\tfrac12 g^\mathsf T\mu_s,
\qquad \log\widetilde w_{00}=\log\pi_{00}.
\]

Normalizing these weights gives the existing state probabilities. The prototype calculates the already-needed means first and removes the four original quadratic divisions. It retains four exponentials, the conditional variances, state draws, overlap covariance, variable-N handling and hyperparameter updates. This is an algebraic rewrite; it adds no LD approximation and removes no sampling iterations. Floating-point rounding changes, so retain the present calculation as a numerical oracle during implementation.

**3. Reuse LDpred3's separate serial/parallel cache identities.** Bipred's [_jit_parallel_uncached](/Users/au507860/REPOS/bipred/bipred/_bivar_kernels.py:40) disables the parallel cache to prevent collision with the serial twin. LDpred3 already solves this with a [renamed function clone](/Users/au507860/REPOS/ldpred3/ldpred3/_numba.py:19), and `_jit_parallel` is available through bipred's existing compatibility seam. Use that helper and test the minimum supported provider version. Simply enabling caching on the existing shared function would reintroduce the original collision. Numba supports caching parallel functions, subject to its documented invalidation limits. [Numba caching documentation](https://numba.readthedocs.io/en/stable/developer/caching.html).

**Table 2. Parallel-cache experiment.** A tiny dense-int8 fit, 2,000 variants, eight blocks, two burn-in and four retained sweeps. Each row describes the second fresh interpreter using its own previously populated experiment cache. This isolates startup behavior, not genome-scale throughput.

| Implementation and call order | Serial cache hit | Parallel cache hit | Parallel public call, s |
|---|---:|---:|---:|
| Current, serial then parallel | Yes | No | 1.191 |
| Distinct identities, serial then parallel | Yes | Yes | 0.025 |
| Distinct identities, parallel then serial | Yes | Yes | 0.409 |

The parallel-first call also pays runtime initialization. All serial/parallel result hashes matched in both cold and warm launches. This saves compilation when a new process uses `ncores > 1`; it does not accelerate a warmed sweep or the already-cached serial path. Low-rank and mixed cache reuse still need integration coverage before shipping.

## What to investigate next

**Generate Gaussian draws only for selected non-null states.** The current driver generates two normals for every variant on every sweep. In the instrumented 60-sweep LR8 fit (20 burn-in, 40 retained), RNG took 0.133 of 0.485 s (27%); fused kernels plus scattering took 0.316 s (65%), and Python block reduction took 0.018 s (4%). The fitted union causal fraction was about 2.3%. Most normal draws therefore go unused. A bounded normal reservoir or suitable compiled RNG could avoid that work. This candidate is **not implemented or timed** here. Preserve valid independent streams under block and chain parallelism, and compare distributions across seeds because selectively consuming draws changes seeded trajectories. Removing all measured RNG cost would give an upper bound of 1.38× for this profile; normal-only savings would necessarily be smaller.

**Dense fastmath is a secondary candidate.** LDpred3 [uses it for dense sweeps](/Users/au507860/REPOS/ldpred3/ldpred3/_kernels.py:825); bipred currently uses it for low-rank sweeps only. In a separate five-repeat dense experiment, adding fastmath to the combined prototype reduced median process CPU time from 1.139 to 1.078 s, but median wall time increased from 1.335 to 1.399 s. Fastmath alone changed wall time from 1.512 to 1.502 s. Thus there is a small CPU-cost signal, not a demonstrated additional elapsed-time benefit. It permits reassociation and broader floating-point assumptions, so I would defer it until a quiet benchmark and stronger overflow/rejected-state coverage. [Numba performance guidance](https://numba.readthedocs.io/en/stable/user/performance-tips.html).

Bipred already has serial fused dispatch, heterogeneous block buckets, shared LD preparation across chains, bounded LR8 widening scratch and chain-level parallelism. Those LDpred3-style gains are already present. Single-trait preparation and screening also live upstream in LDpred3; optimizing a complete SMARTpred job requires a fresh phase breakdown. The present experiments do not measure Catalog access, real LD construction, screening, or a full production pipeline.

## Numerical evidence and decision boundary

The [validation record](/Users/au507860/REPOS/bipred/benchmarks/results/20260907-speed-review/validation-final.json) contains:

- **42 paired comparisons:** seven candidate arms across six cases covering dense float32/int8, LR32/LR8, mixed blocks, variable N, nonzero sampling-error correlation, noise inflation, the decorrelated sensitivity output, an enabled adaptive-stopping path, burn-in traces and zero burn-in. Cases use 2,000 variants and up to 270 sweeps, crossing the periodic residual resynchronization boundary.
- **24 independent Gaussian-conditioning oracle checks:** eight cases for each likelihood/fastmath variant, including sampling-error correlations of ±0.95. These use the existing generic matrix-conditioning oracle rather than the shared scalar likelihood helper.
- **Eight existing targeted tests passed under each of the combined and fastmath-combined prototypes:** golden results, non-finite result rejection, invalid LD, quantization handling, variable-N noise inflation, overlap correction and callback validation.

Driver-only candidates matched every compared field exactly. Without the optional dense fastmath change, the six small cases had maximum absolute output differences of \(2.8\times10^{-17}\). In the 100,000-variant main probes, retained mixture, covariance and genetic-quadratic traces, h² and r_g were bit-identical; maximum effect-mean differences were \(1.4\times10^{-17}\). Warnings and diagnostic flags agreed. Fastmath introduced additional rounding, with maximum differences across the timed result and diagnostic fields of \(1.8\times10^{-15}\).

These checks support preserving the implemented transition and summaries. They are not a posterior-calibration study: bipred draws mixture probabilities conditionally but updates the slab covariance through damped moments, as the [algorithm documentation](/Users/au507860/REPOS/bipred/docs/algorithm.md:102) states. Fewer sweeps, aggressive LD truncation, replacing draws by expectations, or changing that covariance update would require separate statistical validation.

**Recommendation:** implement direct accumulation and the ordered reduction, the likelihood reuse, and the existing provider's cache helper. Before calling the speedup established, rerun paired benchmarks on an idle machine with realistic block-size/rank variation and multi-chain fits, check numerical behavior in long chains and difficult covariance regimes, and measure fit RSS separately. A lost quiet-machine gain or a change in conditioning-oracle results, warnings, overlap summaries or effective sample size per second would overturn the relevant recommendation. Dense fastmath and selective Gaussian generation remain follow-up experiments.

## Reproduction

The [archive manifest](/Users/au507860/REPOS/bipred/benchmarks/results/20260907-speed-review/manifest.json) records hashes, source snapshots and scope. Bipred: `701ab5b7e2814f565ccff537ac2be284e8baf51f` / `0.3.16.dev0`; LDpred3: `dad2badc424ba7629c54e8ace0cf49dc6aac0a4a` / `0.7.17`. Runtime: Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0, Numba 0.66.0; Apple M2 Pro, 10 cores, 16 GiB RAM, macOS 26.6.2, AC power, Low Power Mode off. The BLAS/LAPACK libraries link to Apple Accelerate; their [linkage and build configuration](/Users/au507860/REPOS/bipred/benchmarks/results/20260907-speed-review/numerical_stack.json) are archived. BLAS/OpenMP/Numba threads were pinned to one except the two-core cache probe. LD payloads are distinct, input hashes are recorded, and each arm reconstructs fresh chain state. LR simulation uses its represented factor-plus-diagonal matrix, avoiding an unreported truncation mismatch.

From the bipred repository root, with the revisions above installed, a representative command is:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  NUMBA_CACHE_DIR=/private/tmp/bipred-speed-review-reproduction-cache \
  /Users/au507860/anaconda3/envs/ldpred3-accelerate/bin/python \
  benchmarks/results/20260907-speed-review/probe.py bench \
  --kind lr8 --modes baseline driver combined --reps 5 \
  --burn 50 --retained 150 --out /private/tmp/bipred-speed-review-reproduction.json
```

`fastmath.py` adds the optional dense fastmath arms. `validate.py` runs the targeted checks; run it from a scratch copy of the archive because it writes results beside the script. `startup.py --out /private/tmp/bipred-startup.json` launches fresh cache-test interpreters sequentially; use a fresh experiment-directory copy for another cold-cache comparison. The probes operate only inside their own interpreters. No production source was edited, committed or pushed.
