"""Independent-block coalescent genome shared by the individual-genotype benchmarks.

A genome here is ``NB`` *independent* coalescent segments, each trimmed to
``BLOCK_SIZE`` common SNPs, so the block-diagonal LD is exact: a continuous
chromosome chopped into arbitrary blocks leaks cross-block LD that corrupts h2
inference. Everything that depends only on the genotypes -- the standardized
GWAS genotypes, the reference-panel LD blocks and their LD scores -- is built
once here and memoised, so estimators are compared on the *same* genotypes.

The same construction backs the univariate ``infer_vs_ldsc_sbayes.py``
benchmark in the ldpred3 repository. This module holds only the genome, not
that benchmark's estimator arms, so the two repositories share a described
construction rather than a copied script that drifts.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.simulate import simulate_genotypes_by_mutation_rate      # noqa: E402
from ldpred3 import ld_scores                                            # noqa: E402
from ldpred3.ld import compute_ld_blocks                                 # noqa: E402

N_REF, N_GWAS = 4000, 20000
NB, BLOCK_SIZE = 20, 150
SEG_LEN, MUT_RATE = 0.6e6, 3e-8              # per-segment length / SNP density
M = NB * BLOCK_SIZE

_GENOME = {}


def genome(rep):
    """Genome for one replicate, memoised on ``rep``.

    Returns the standardized GWAS genotypes ``Zg`` and the variant mask
    ``good`` alongside the reference LD blocks ``ld`` and scores ``ell``.
    """
    if rep in _GENOME:
        return _GENOME[rep]
    cols = []
    for b in range(NB):
        mut = MUT_RATE
        for _ in range(4):                   # bump density until the segment fills
            Gb = simulate_genotypes_by_mutation_rate(
                N_REF + N_GWAS, SEG_LEN, mut_rate=mut, min_maf=0.02,
                seed=rep * NB + b + 1)
            if Gb.shape[1] >= BLOCK_SIZE:
                break
            mut *= 1.6
        if Gb.shape[1] < BLOCK_SIZE:
            raise RuntimeError("segment produced too few SNPs; raise MUT_RATE")
        cols.append(Gb[:, :BLOCK_SIZE])
    G = np.concatenate(cols, axis=1)
    m = G.shape[1]
    Gref = G[:N_REF].astype(np.int8)
    Gg = G[N_REF:].astype(float)
    f = Gg.mean(0) / 2.0
    sd = np.sqrt(2 * f * (1 - f))
    good = sd > 0
    Zg = np.where(good, (Gg - 2 * f) / np.where(good, sd, 1.0), 0.0)
    ld = [(R.astype(np.float64), idx) for R, idx in
          compute_ld_blocks(Gref, block_size=BLOCK_SIZE)]
    ell = ld_scores(ld, n_ref=N_REF)
    _GENOME[rep] = dict(m=m, Gref=Gref, Gg=Gg, Zg=Zg, f=f, sd=sd, good=good,
                        ld=ld, ell=ell)
    return _GENOME[rep]
