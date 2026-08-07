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
    # The backend is resolved once per process, so select it explicitly rather
    # than relying on the stub being visible to a re-probe.
    monkeypatch.setattr(simulate, "_RESOLVED_BACKEND", "msprime")

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


def test_backend_resolution_is_stable_within_a_process():
    """The tag names the simulator that actually produced a cached segment.

    ``import msprime`` is not idempotent -- a failed attempt can leave partial
    state that lets a retry succeed -- so re-probing per call once let segments
    be simulated by msprime and cached under the numba tag. Resolving once
    makes the two answers the same object.
    """
    code = """
import benchmarks.simulate as simulate

first = simulate._backend()
assert simulate.SIMULATOR_CACHE_TAG == simulate._BACKEND_TAGS[first]
# Re-probe many times: a non-idempotent import would flip on a later attempt.
for _ in range(5):
    assert simulate._backend() == first, "backend changed mid-process"
assert simulate.SIMULATOR_CACHE_TAG == simulate._BACKEND_TAGS[simulate._backend()]
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _markdown_table_after(text, caption):
    """Rows of the first markdown table following ``caption``, as cell lists."""
    lines = text[text.index(caption):].splitlines()
    rows = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):        # the header separator
            continue
        rows.append(cells)
    return rows[1:]                                  # drop the header row


def test_results_scaling_table_matches_its_csv():
    """RESULTS.md's prose tables are transcriptions; keep them honest.

    The 0.2.1 regeneration updated benchmarks/rg_scaling.csv but left the
    printed timing and peak-memory columns at their 0.2.0 values, including a
    2.51 GB memory spike the current data does not show.
    """
    import csv
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "benchmarks"
    rows = list(csv.DictReader((root / "rg_scaling.csv").open()))
    table = _markdown_table_after(
        (root / "RESULTS.md").read_text(encoding="utf-8"),
        "**Table 6. Scaling with variant count.**",
    )
    assert len(table) == len(rows)

    def number(cell):
        return float(cell.split()[0].replace(",", ""))

    columns = ["m", "t_ldsc_s", "t_ldpred3_s", "peak_gb", "rg_realized",
               "abs_error_ldsc_realized", "abs_error_ldpred3_realized"]
    for printed, record in zip(table, rows):
        assert len(printed) == len(columns)
        for cell, column in zip(printed, columns):
            # Each cell transcribes its CSV value at the precision it is
            # printed to, so the tolerance is that column's own half-ulp --
            # not one fixed window, which would be far too loose for a cell
            # printed to fewer decimals. Exact halfway values (0.5175 -> 0.517
            # or 0.518) are accepted either way; anything beyond half an ulp is
            # a real desync.
            digits = len(cell.split()[0].partition(".")[2])
            # The 1e-9 slack is floating point, not looseness: an exact tie
            # such as |0.518 - 0.5175| evaluates to 5.000000000000004e-4.
            half_ulp = 0.5 * 10.0 ** -digits * (1.0 + 1e-9)
            assert abs(number(cell) - float(record[column])) <= half_ulp, (
                f"{column}: table says {cell!r}, CSV says {record[column]!r}")


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
