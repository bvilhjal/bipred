"""Generate ``ld_library.npz``, the population-LD block library ``bivariate_demo`` tiles.

``bivariate_demo.py`` reads a library of coalescent **population** correlation
matrices from the working directory and derives its own finite-reference panel
from them, so what is stored here is the population LD, unshrunk. Until this
script existed the library was an undocumented local artifact: it is gitignored
(492 MB of segment cache behind it, and the archive itself is large), so the
demo's committed result could not be reproduced from the repository alone. The
generating parameters now live in code rather than in a prose sentence.

Blocks are independent coalescent segments trimmed to ``K`` SNPs, built with the
same convention as the rest of the suite (:mod:`benchmarks.rg_architectures`):
standardise the genotype panel, take ``Z'Z / n``. Segments reuse the shared
per-segment cache under ``benchmarks/.rg_cache``, so a library that overlaps an
existing benchmark's segments costs only the correlation step.

Run once from the repo root::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/make_ld_library.py

``--blocks``/``--k`` must match what the consumer expects; ``bivariate_demo.py``
reads 12 blocks of 500. The archive embeds the simulator cache tag, and the
consumer hashes the complete archive in its provenance sidecar.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.simulate import (                                    # noqa: E402
    SIMULATOR_CACHE_TAG,
    simulate_genotypes_by_mutation_rate,
)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".rg_cache")

#: Coalescent geometry. N_POP is the number of individuals defining the
#: population correlation; SEG_LEN and MUT_RATE together set how many SNPs a
#: segment yields, which must reach K.
N_POP = 10_000
SEG_LEN = 5_000_000
MUT_RATE = 5e-9
MIN_MAF = 0.02


def _segment_cache_path(b, k):
    """Per-segment cache path, tagged so simulator schemas never mix."""
    name = (f"seg{b + 1}_k{k}_npop{N_POP}_seg{int(SEG_LEN)}"
            f"_mu{MUT_RATE:g}_{SIMULATOR_CACHE_TAG}.npz")
    return os.path.join(CACHE, name)


def _population_ld(b, k):
    """One segment's population correlation matrix, cached on disk.

    Shares the cache and the key format with the rest of the suite, so a
    library built at the suite's own ``K`` costs no new simulation.
    """
    path = _segment_cache_path(b, k)
    if os.path.exists(path):
        return np.load(path)["R"]

    mutation = MUT_RATE
    for _ in range(4):
        genotypes = simulate_genotypes_by_mutation_rate(
            N_POP, SEG_LEN, mut_rate=mutation, min_maf=MIN_MAF, seed=b + 1)
        if genotypes.shape[1] >= k:
            break
        mutation *= 1.6                       # denser segment, same geometry
    if genotypes.shape[1] < k:
        raise SystemExit(
            f"segment {b} yielded {genotypes.shape[1]} SNPs, fewer than k={k}; "
            "raise MUT_RATE or SEG_LEN")

    panel = genotypes[:, :k].astype(np.float64)
    standardised = (panel - panel.mean(0)) / panel.std(0)
    corr = (standardised.T @ standardised) / N_POP
    cholesky = np.linalg.cholesky(corr + 1e-4 * np.eye(k))
    os.makedirs(CACHE, exist_ok=True)
    # savez insists on the .npz suffix, so stage under one and rename.
    staged = path.replace(".npz", f".tmp{os.getpid()}.npz")
    np.savez(staged, R=corr, C=cholesky)
    os.replace(staged, path)
    return corr


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--out", default="ld_library.npz")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    library = np.empty((args.blocks, args.k, args.k), dtype=np.float64)
    for b in range(args.blocks):
        library[b] = _population_ld(b, args.k)
        print(f"  block {b + 1}/{args.blocks}", flush=True)

    # Key "R": the stacked (blocks, k, k) array bivariate_demo.py loads. Keep
    # the backend-derived cache tag beside it so the local archive is not an
    # anonymous matrix payload; the consumer also hashes the complete file.
    np.savez(args.out, R=library,
             simulator_cache_tag=np.array(SIMULATOR_CACHE_TAG))
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}: {args.blocks} blocks of {args.k}x{args.k} "
          f"population LD, {size_mb:.0f} MB, simulator {SIMULATOR_CACHE_TAG} "
          f"({time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
