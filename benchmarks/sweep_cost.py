"""Per-sweep cost of the bivariate sampler, by LD representation and core count.

The fit's cost is dominated by the per-sweep kernels, so this measures a sweep
directly rather than a whole fit: each cell runs the public driver twice at
different ``num_iter`` and takes the difference, which cancels preparation,
compilation and the fixed post-processing. Reported as milliseconds per sweep.

Two properties of the sampler make a naive timing harness lie about it, and
both are controlled here.

*Numba's on-disk cache is keyed without the compilation flags.* The fused
drivers are jitted twice from one Python function -- ``parallel=True`` and
``nogil=True`` -- so the twins collide in the cache and whichever compiled
first is served to both. A run that shares one cache across core counts
therefore measures the first arm twice. Every cell here gets a private
``NUMBA_CACHE_DIR`` in a subprocess (``--in-process`` opts out, for profiling a
single cell).

*Compilation is not free at ``parallel=True``.* The first call to each cell is
discarded before timing.

Usage::

    python -m benchmarks.sweep_cost                      # the default grid
    python -m benchmarks.sweep_cost --m 50000 --cores 1 2 4 8
    python -m benchmarks.sweep_cost --reps 5 --csv out.csv

Representations: ``dense_i8``, ``dense_f32``, ``lr8_<rank>``, ``lr32_<rank>``,
``mixed``. Ranks are requested, not guaranteed: ``lowrank_ld`` picks the rank
that carries the requested variance, and the achieved rank is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np


def _blocks(kind, m, k, seed=0):
    """Build one LD panel of the requested representation.

    Returns ``(blocks, label)``; the label records the achieved rank, which for
    the low-rank representations is chosen by the eigenspectrum rather than by
    the caller.
    """
    from ldpred3 import lowrank_ld

    nb = m // k
    pos = np.arange(k)
    # AR(1) at rho=0.6 is well conditioned; a low-rank factor of it needs a
    # substantial rank, which is the regime the projection dots dominate.
    dense = (0.6 ** np.abs(pos[:, None] - pos[None, :])).astype(np.float64)

    def idx(b):
        return np.arange(b * k, (b + 1) * k)

    if kind == "dense_f32":
        block = np.ascontiguousarray(dense, dtype=np.float32)
        return [(block, idx(b)) for b in range(nb)], kind
    if kind == "dense_i8":
        block = np.rint(np.clip(dense, -1.0, 1.0) * 127.0).astype(np.int8)
        return [(block, idx(b)) for b in range(nb)], kind
    if kind.startswith(("lr8_", "lr32_")):
        quantize = kind.startswith("lr8_")
        variance = float(kind.split("_", 1)[1])
        factor = lowrank_ld(dense, variance=variance, quantize=quantize)
        return ([(factor, idx(b)) for b in range(nb)],
                f"{kind} (rank {factor.U.shape[1]})")
    if kind == "mixed":
        i8 = np.rint(np.clip(dense, -1.0, 1.0) * 127.0).astype(np.int8)
        lr = lowrank_ld(dense, variance=0.99, quantize=True)
        return ([(i8 if b % 2 else lr, idx(b)) for b in range(nb)],
                f"mixed (rank {lr.U.shape[1]})")
    raise SystemExit(f"unknown representation {kind!r}")


def _sumstats(m, seed=0):
    rng = np.random.default_rng(seed + 1)
    beta_hat1 = rng.normal(scale=0.01, size=m)
    beta_hat2 = 0.6 * beta_hat1 + 0.8 * rng.normal(scale=0.01, size=m)
    return beta_hat1, beta_hat2


def _measure(kind, m, k, cores, reps, short, long_, seed=0):
    """Milliseconds per sweep, by difference of two run lengths."""
    import warnings

    from bipred import ldpred3_auto_bivariate_blocks

    blocks, label = _blocks(kind, m, k, seed=seed)
    beta_hat1, beta_hat2 = _sumstats(m, seed=seed)

    def run(num_iter):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            result = ldpred3_auto_bivariate_blocks(
                blocks, beta_hat1, beta_hat2, 50_000, 50_000,
                burn_in=2, num_iter=num_iter, seed=seed, ncores=cores)
            return time.perf_counter() - start, result

    run(short)                                    # discard: compilation
    per_sweep = []
    for _ in range(reps):
        t_short, _ = run(short)
        t_long, result = run(long_)
        per_sweep.append((t_long - t_short) / (long_ - short) * 1e3)
    return {"representation": label, "m": m, "k": k, "ncores": cores,
            "ms_per_sweep": min(per_sweep),
            "rg": float(result.rg), "h2_1": float(result.h2[0]),
            "h2_2": float(result.h2[1])}


def _cell_subprocess(kind, m, k, cores, reps, short, long_, seed):
    """Run one cell in a fresh interpreter with a private Numba cache."""
    with tempfile.TemporaryDirectory(prefix="bipred-nbc-") as cache_dir:
        env = dict(os.environ, NUMBA_CACHE_DIR=cache_dir)
        proc = subprocess.run(
            [sys.executable, "-m", "benchmarks.sweep_cost", "--in-process",
             "--kind", kind, "--m", str(m), "--k", str(k),
             "--cores", str(cores), "--reps", str(reps),
             "--short", str(short), "--long", str(long_), "--seed", str(seed)],
            env=env, capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"cell {kind} ncores={cores} failed")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=20_000)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--cores", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--short", type=int, default=20)
    parser.add_argument("--long", dest="long_", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kinds", nargs="+",
                        default=["dense_i8", "dense_f32", "lr8_0.99",
                                 "lr32_0.99", "mixed"])
    parser.add_argument("--csv")
    # Internal: run exactly one cell here and print it as JSON.
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--kind")
    args = parser.parse_args(argv)

    if args.in_process:
        row = _measure(args.kind, args.m, args.k, args.cores[0], args.reps,
                       args.short, args.long_, seed=args.seed)
        print(json.dumps(row))
        return 0

    rows = []
    print(f"m={args.m}  k={args.k}  best of {args.reps}  "
          f"(private Numba cache per cell)\n")
    print(f"{'representation':24} {'cores':>5} {'ms/sweep':>10} {'speed-up':>9}")
    print("-" * 52)
    for kind in args.kinds:
        baseline = None
        for cores in args.cores:
            row = _cell_subprocess(kind, args.m, args.k, cores, args.reps,
                                   args.short, args.long_, args.seed)
            baseline = baseline or row["ms_per_sweep"]
            row["speedup_vs_1core"] = baseline / row["ms_per_sweep"]
            rows.append(row)
            print(f"{row['representation']:24} {cores:5d} "
                  f"{row['ms_per_sweep']:10.3f} "
                  f"{row['speedup_vs_1core']:8.2f}x")
        print()

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
