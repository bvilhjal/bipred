"""Coalescent genotype simulation shared by the repository benchmarks.

Two backends, one model (Hudson coalescent with recombination, infinite-sites
mutations, MAF filter):

- **msprime** (the ``[sim]`` extra): the reference implementation and the
  default where installed — measured fastest at the benchmark shape (0.80 s
  for a 10,000-sample, 5 Mb segment here).
- **bundled Numba coalescent** (``benchmarks/_coalescent.py``, vendored from
  ldpred3's benchmark suite): dependency-free fallback, used when msprime is
  not installed. Measured 2.0x msprime's per-segment time at that shape
  (1.64 s); statistically equivalent outputs (site counts within simulation
  noise). Runs JIT-compiled where Numba is available, interpreted otherwise.

The backends draw different events from the same model and are **not**
bitwise-identical, so cached segments are tagged per backend and never mix:
``SIMULATOR_CACHE_TAG`` resolves at import to the active backend's tag.
"""

from __future__ import annotations

import numpy as np


def _backend():
    """The simulator backend used for new segments on this host."""
    try:
        import msprime  # noqa: F401
        return "msprime"
    except ImportError:
        # Bundled vendored coalescent (JIT where Numba exists, interpreted
        # pure-Python otherwise): the dependency-free fallback.
        return "numba"


#: Cache tag for the active backend. Bump the version suffix when a backend's
#: draws change; the two backends never share cached segments.
SIMULATOR_CACHE_TAG = {"numba": "numba-v1", "msprime": "msprime-v1"}[_backend()]


def simulate_genotypes_by_mutation_rate(n, seq_len, *, recomb_rate=1e-8,
                                        mut_rate=1e-8, Ne=10000, min_maf=0.01,
                                        seed=None):
    """Simulate ordered diploid dosages on one fixed coalescent segment.

    Returns int8 ``(n, sites)`` dosages with allele frequency strictly inside
    ``(min_maf, 1 - min_maf)``, sites in physical order. ``seed=None`` draws a
    fresh random seed (either backend).
    """
    if _backend() == "numba":
        from . import _coalescent
        if seed is None:
            seed = int(np.random.default_rng().integers(1, 2 ** 31 - 1))
        dosages, _pos, allele_frequency = _coalescent.simulate_dosages(
            n, seq_len, recomb_rate=recomb_rate, mut_rate=mut_rate, Ne=Ne,
            seed=seed)
    else:
        try:
            import msprime
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("benchmark simulation needs msprime "
                              "(python -m pip install -e '.[sim]')") from exc

        ms_seed = None if seed is None else int(seed)
        ancestry = msprime.sim_ancestry(
            samples=n, ploidy=2, population_size=Ne,
            recombination_rate=recomb_rate, sequence_length=int(seq_len),
            random_seed=ms_seed)
        mutated = msprime.sim_mutations(
            ancestry, rate=mut_rate, random_seed=ms_seed,
            model=msprime.BinaryMutationModel())
        haplotypes = mutated.genotype_matrix()            # (sites, 2n), 0/1
        dosages = (haplotypes[:, 0::2] + haplotypes[:, 1::2]).T
        allele_frequency = dosages.mean(axis=0) / 2.0

    common = ((allele_frequency > min_maf)
              & (allele_frequency < 1.0 - min_maf))
    return np.ascontiguousarray(dosages[:, common], dtype=np.int8)
