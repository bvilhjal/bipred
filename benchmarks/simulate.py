"""Small msprime helper shared by the repository benchmarks."""

import numpy as np


SIMULATOR_CACHE_TAG = "msprime-v1"


def simulate_genotypes_by_mutation_rate(n, seq_len, *, recomb_rate=1e-8,
                                        mut_rate=1e-8, Ne=10000, min_maf=0.01,
                                        seed=None):
    """Simulate ordered diploid dosages on one fixed coalescent segment."""
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
