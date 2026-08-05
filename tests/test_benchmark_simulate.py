"""The repository benchmark simulator: both backends, and cache separation."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np

import benchmarks.simulate as simulate
from benchmarks.simulate import simulate_genotypes_by_mutation_rate


def test_msprime_backend_returns_filtered_diploid_dosages(monkeypatch):
    # Stub msprime; the helper's call contract is what this pins.
    haplotypes = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ], dtype=np.int8)
    ancestry = object()
    calls = {}

    def sim_ancestry(**kwargs):
        calls["ancestry"] = kwargs
        return ancestry

    def sim_mutations(value, **kwargs):
        calls["mutations"] = (value, kwargs)
        return SimpleNamespace(genotype_matrix=lambda: haplotypes)

    fake_msprime = SimpleNamespace(
        sim_ancestry=sim_ancestry,
        sim_mutations=sim_mutations,
        BinaryMutationModel=lambda: "binary",
    )
    monkeypatch.setitem(sys.modules, "msprime", fake_msprime)

    dosages = simulate_genotypes_by_mutation_rate(
        3, 123.9, recomb_rate=2e-8, mut_rate=3e-8, Ne=12000,
        min_maf=0.2, seed=np.int64(7))

    np.testing.assert_array_equal(dosages, [[2], [1], [0]])
    assert dosages.dtype == np.int8
    assert dosages.flags.c_contiguous
    assert calls["ancestry"] == {
        "samples": 3,
        "ploidy": 2,
        "population_size": 12000,
        "recombination_rate": 2e-8,
        "sequence_length": 123,
        "random_seed": 7,
    }
    value, mutation_kwargs = calls["mutations"]
    assert value is ancestry
    assert mutation_kwargs == {
        "rate": 3e-8,
        "random_seed": 7,
        "model": "binary",
    }


def test_numba_backend_returns_filtered_diploid_dosages(monkeypatch):
    monkeypatch.setattr(simulate, "_backend", lambda: "numba")
    a = simulate_genotypes_by_mutation_rate(
        40, 200_000, mut_rate=1e-7, min_maf=0.05, seed=3)
    b = simulate_genotypes_by_mutation_rate(
        40, 200_000, mut_rate=1e-7, min_maf=0.05, seed=3)
    c = simulate_genotypes_by_mutation_rate(
        40, 200_000, mut_rate=1e-7, min_maf=0.05, seed=4)
    np.testing.assert_array_equal(a, b)          # same seed, same segment
    assert not np.array_equal(a, c)              # different seed, different draw
    assert a.shape[0] == 40 and a.dtype == np.int8 and a.flags.c_contiguous
    af = a.mean(axis=0) / 2.0
    assert af.min() > 0.05 and af.max() < 0.95   # the MAF filter is applied


def test_cache_tag_matches_the_resolved_backend():
    expected = {"numba": "numba-v1", "msprime": "msprime-v1"}[
        simulate._backend()]
    assert simulate.SIMULATOR_CACHE_TAG == expected


def test_architecture_cache_key_names_the_simulator_schema():
    code = """
import os
os.environ["NB"] = "0"
import benchmarks.rg_architectures as benchmark
assert benchmark._segment_cache_path(0).endswith(
    f"_{benchmark.SIMULATOR_CACHE_TAG}.npz"
)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
