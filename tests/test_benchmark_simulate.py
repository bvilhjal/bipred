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


def test_bivariate_demo_controls_are_literal_and_recorded(tmp_path):
    """No-shrinkage and disjoint mean those things, not rounded facsimiles."""
    import csv

    from benchmarks import bivariate_demo as demo

    rng = np.random.default_rng(7)
    first, second = demo._disjoint_masks(rng, 100_000, p_causal=0.1)
    assert not np.any(first & second)
    assert abs(first.mean() - 0.1) < 0.005
    assert abs(second.mean() - 0.1) < 0.005

    library = np.repeat(np.eye(6)[None, :, :], 2, axis=0)
    _, _, unshrunk, _ = demo._build_panels(
        library, n_ref=64, seed=4)
    _, _, shrunk, _ = demo._build_panels(
        library, n_ref=64, shrinkage=0.1, seed=4)
    for (raw, _), (regularised, _) in zip(unshrunk, shrunk):
        np.testing.assert_allclose(
            regularised, 0.9 * raw + 0.1 * np.eye(raw.shape[0]), rtol=2e-6)

    row = dict.fromkeys(demo.CSV_FIELDS)
    row.update(architecture="disjoint causal", target_rg="", replicate=0,
               realized_rg=-0.00123456789012345,
               reference_shrinkage=0.0, n_shared_causal=0,
               solo_r2=0.123456789012345, joint_r2=0.123456789012346,
               gain=1e-15, rg_est=-0.0123456789012345, joint_p=0.1,
               joint_h2_1=0.4, joint_h2_2=0.5, joint_warned=0,
               joint_warning_count=0, joint_implausible_warnings=0,
               joint_divergence_warnings=0, joint_other_warnings=0)
    path = tmp_path / "bivariate_demo.csv"
    demo._write_csv(path, [row])
    saved = list(csv.DictReader(path.open()))
    assert len(saved) == 1
    assert saved[0]["reference_shrinkage"] == "0.0"
    assert saved[0]["n_shared_causal"] == "0"
    assert float(saved[0]["realized_rg"]) == row["realized_rg"]
    assert float(saved[0]["gain"]) == 1e-15
    assert float(saved[0]["rg_est"]) == row["rg_est"]


def test_bivariate_demo_artifact_is_per_replicate_and_full_precision():
    import csv
    import pathlib

    path = (pathlib.Path(__file__).resolve().parent.parent
            / "benchmarks" / "bivariate_demo.csv")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 60
    assert {row["reference_shrinkage"] for row in rows} == {"0.0", "0.05"}
    assert all(np.isfinite(float(row["realized_rg"])) for row in rows)
    assert all(np.isfinite(float(row["gain"])) for row in rows)
    assert all(np.isfinite(float(row["joint_p"])) for row in rows)
    assert all(np.isfinite(float(row["joint_h2_1"])) for row in rows)
    assert all(np.isfinite(float(row["joint_h2_2"])) for row in rows)
    assert {int(row["joint_warned"]) for row in rows} <= {0, 1}
    disjoint = [row for row in rows if row["architecture"] == "disjoint causal"]
    assert len(disjoint) == 12
    assert all(row["target_rg"] == "" for row in disjoint)
    assert all(int(row["n_shared_causal"]) == 0 for row in disjoint)
    paired = {}
    for row in rows:
        paired.setdefault((row["architecture"], row["replicate"]), []).append(row)
        expected = float(row["joint_r2"]) - float(row["solo_r2"])
        assert abs(float(row["gain"]) - expected) < 1e-14
        warning_count = int(row["joint_warning_count"])
        categories = sum(int(row[column]) for column in (
            "joint_implausible_warnings", "joint_divergence_warnings",
            "joint_other_warnings"))
        assert warning_count == categories
        assert int(row["joint_warned"]) == int(warning_count > 0)
    assert len(paired) == 30
    for arm in paired.values():
        assert len(arm) == 2
        assert {row["reference_shrinkage"] for row in arm} == {"0.0", "0.05"}
        assert len({row["realized_rg"] for row in arm}) == 1
        assert len({row["n_causal_1"] for row in arm}) == 1
        assert len({row["n_causal_2"] for row in arm}) == 1
        assert len({row["n_shared_causal"] for row in arm}) == 1
    assert any(len(row["gain"].partition(".")[2]) > 8 for row in rows)


def test_bivariate_demo_records_clean_source_and_library_hash(tmp_path,
                                                               monkeypatch):
    import json
    from types import SimpleNamespace

    from benchmarks import bivariate_demo as demo

    library = tmp_path / "ld_library.npz"
    np.savez(library, R=np.eye(2)[None, :, :],
             simulator_cache_tag=np.array("fixture-v1"))
    csv_path = tmp_path / "demo.csv"
    args = SimpleNamespace(
        library=library, csv=csv_path, shrinkage=[0.0, 0.05], blocks=1,
        burn_in=2, n1=100, n2=50, n_ref=20, num_iter=3, reps=1)
    monkeypatch.setattr(demo, "_run_arm", lambda _args, _lib, _shrink: [])

    revision = "a" * 40
    dependency_sources = {
        "ldpred3": {"identity": "fixture", "source_tree_sha256": "b" * 64}}
    demo._run(args, source_revision=revision,
              dependency_sources=dependency_sources)
    sidecar = tmp_path / "demo.provenance.json"
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["source_revision"] == revision
    assert metadata["source_clean"] is True
    assert metadata["dependency_sources"] == dependency_sources
    assert len(metadata["inputs_sha256"]["ld_library.npz"]) == 64
    assert metadata["run_controls"]["reference_shrinkage"] == [0.0, 0.05]
    assert metadata["run_controls"]["simulator_cache_tag"] == "fixture-v1"


def test_synthetic_ld_panels_have_distinct_payloads_and_honest_size():
    from benchmarks.fit_memory import _payload_bytes
    from benchmarks.sweep_cost import _add_speedups, _blocks

    blocks, _, _ = _blocks("dense_f32", m=40, k=10)
    payloads = [block for block, _ in blocks]
    assert len({id(block) for block in payloads}) == len(payloads)
    assert not any(np.shares_memory(a, b) for i, a in enumerate(payloads)
                   for b in payloads[i + 1:])
    assert _payload_bytes(blocks) == 40 * 10 * np.dtype(np.float32).itemsize

    twice, _, _ = _blocks("dense_f32", m=80, k=10)
    assert _payload_bytes(twice) == 2 * _payload_bytes(blocks)

    rows = [
        {"ncores": 4, "ms_per_sweep_median": 2.0},
        {"ncores": 1, "ms_per_sweep_median": 6.0},
    ]
    _add_speedups(rows)
    assert [row["speedup_vs_1core"] for row in rows] == [3.0, 1.0]


def test_representation_artifacts_record_distinct_storage_and_scaling():
    import csv
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "benchmarks"
    sweep = list(csv.DictReader((root / "sweep_cost.csv").open()))
    memory = list(csv.DictReader((root / "fit_memory.csv").open()))
    assert len(sweep) == 10 and len(memory) == 8
    assert {row["block_storage"] for row in sweep + memory} == {"distinct"}
    assert all(float(row["ms_per_sweep_median"]) > 0 for row in sweep)
    assert all(float(row["ms_per_sweep_mad"]) >= 0 for row in sweep)
    assert {int(row["timing_reps"]) for row in sweep} == {3}
    assert {int(row["short_sweeps"]) for row in sweep} == {20}
    assert {int(row["long_sweeps"]) for row in sweep} == {80}
    assert all("ms_per_sweep" not in row for row in sweep)

    groups = {}
    for row in memory:
        key = row["representation"], row["ld_int8"]
        groups.setdefault(key, []).append(row)
        assert int(row["blocks"]) == int(row["m"]) // int(row["k"])
    for rows in groups.values():
        assert len(rows) == 2
        rows.sort(key=lambda row: int(row["m"]))
        assert int(rows[1]["m"]) == 2 * int(rows[0]["m"])
        assert abs(float(rows[1]["payload_mb"])
                   - 2 * float(rows[0]["payload_mb"])) < 1e-12
    dense = [row for row in memory
             if row["representation"] == "dense_f32"
             and row["ld_int8"] == "default"]
    assert [float(row["payload_mb"]) for row in dense] == [100.0, 200.0]


def test_factorial_keeps_reported_n_when_quantitative_scale_is_unknown():
    import pytest

    from benchmarks.qc_factorial import _cross_corr_from_ldsc, _sample_size_plan

    reported = np.array([80_000.0, 100_000.0])
    unidentified = {"median": np.nan, "ratio": np.nan, "consistent": False}
    fitted, implied, ratio, note = _sample_size_plan(
        reported, unidentified, binary=False)
    np.testing.assert_array_equal(fitted, reported)
    assert (implied, ratio, note) == ("unidentified", "--", "")

    binary = {"median": 50_000.0, "ratio": 0.5, "consistent": False}
    fitted, implied, ratio, note = _sample_size_plan(
        reported, binary, binary=True)
    np.testing.assert_array_equal(fitted, reported * 0.5)
    assert (implied, ratio) == ("50,000", "0.500")
    assert "MISSPECIFIED" in note

    assert _cross_corr_from_ldsc(SimpleNamespace(gcov_intercept=-0.35)) == -0.35
    for invalid in (-1.0, 1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="Refusing to clip"):
            _cross_corr_from_ldsc(SimpleNamespace(gcov_intercept=invalid))


def test_factorial_frac_shared_spread_uses_the_public_estimand():
    from benchmarks.qc_factorial import _frac_shared_trace

    pi = np.array([
        [0.60, 0.20, 0.10, 0.10],
        [0.70, 0.05, 0.15, 0.10],
    ])
    expected = pi[:, 3] / np.minimum(
        pi[:, 1] + pi[:, 3], pi[:, 2] + pi[:, 3])
    np.testing.assert_allclose(_frac_shared_trace(pi), expected)


def test_factorial_artifact_uses_the_primary_screen_name():
    import csv
    import pathlib

    path = (pathlib.Path(__file__).resolve().parent.parent
            / "benchmarks" / "qc_factorial.csv")
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames
    assert "ld_screen" in fields
    assert "dentist" not in fields
    assert "divergence_warned" in fields
    assert "warned" not in fields
    assert len(rows) == 24
    assert all(row["expected"].startswith("rough context:") for row in rows)
    assert all(-1.0 < float(row["cross_corr"]) < 1.0 for row in rows)


def test_real_data_checksum_manifest_and_validator(tmp_path):
    import hashlib
    import pathlib

    import pytest

    from benchmarks.real_data_inputs import (
        _source_tree_sha256, load_manifest, validate_inputs,
    )

    committed = load_manifest()
    assert set(committed) == {
        "ldref-hm3/ldpred3_ldref_hm3.npz",
        "sumstats/jointGwasMc_LDL.txt.gz",
        "sumstats/cad.add.160614.website.txt",
        "sumstats/jointGwasMc_HDL.txt.gz",
        "sumstats/jointGwasMc_TG.txt.gz",
        "sumstats/GIANT_HEIGHT_2014.txt.gz",
    }
    assert all(len(digest) == 64 for digest in committed.values())
    benchmark_readme = (pathlib.Path(__file__).resolve().parent.parent
                        / "benchmarks" / "README.md").read_text(
                            encoding="utf-8")
    assert "**Table 4. Acquisition record" in benchmark_readme
    assert all(f"`{name}`" in benchmark_readme for name in committed)
    for source_id in (
            "figshare.com/articles/dataset/European_LD_reference_with_blocks_/19213299",
            "jointGwasMc_LDL.txt.gz", "jointGwasMc_HDL.txt.gz",
            "jointGwasMc_TG.txt.gz", "GCST003116",
            "GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz",
            "5d86ac9d97e42c57fa31d84ff093d3bf637dc0e6"):
        assert source_id in benchmark_readme

    data = tmp_path / "input.dat"
    data.write_bytes(b"the exact benchmark input")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest = tmp_path / "inputs.sha256"
    manifest.write_text(f"{digest}  source/input.dat\n", encoding="ascii")
    assert validate_inputs(
        {"source/input.dat": data}, manifest_path=manifest,
        verbose=False) == {"source/input.dat": digest}
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_inputs({"source/input.dat": data}, manifest_path=manifest,
                        verbose=False)

    package = tmp_path / "package"
    package.mkdir()
    source = package / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first = _source_tree_sha256(package)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert _source_tree_sha256(package) != first


def test_manual_benchmark_ldpred3_pin_matches_ci_and_install_docs():
    import pathlib

    from benchmarks.real_data_inputs import LDPRED3_REV, LDPRED3_VERSION

    root = pathlib.Path(__file__).resolve().parent.parent
    ci = (root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert f'LDPRED3_REV: "{LDPRED3_REV}"' in ci
    assert f'LDPRED3_VERSION: "{LDPRED3_VERSION}"' in ci
    assert f"ldpred3.git@{LDPRED3_REV}" in readme


def test_real_data_provenance_requires_clean_source_and_writes_sidecar(tmp_path):
    import json
    import pathlib

    import pytest

    from benchmarks.real_data_inputs import (
        require_clean_source, write_provenance_sidecar,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "bench@example.invalid"],
                   cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Benchmark Test"],
                   cwd=repo, check=True)
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"],
        cwd=repo, check=True)

    revision = require_clean_source(repo)
    assert len(revision) == 40
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean source tree"):
        require_clean_source(repo)

    csv_path = tmp_path / "result.csv"
    csv_path.write_text("value\n1\n", encoding="utf-8")
    sidecar = pathlib.Path(write_provenance_sidecar(
        csv_path, source_revision=revision,
        input_hashes={"source/input.dat": "a" * 64},
        dependency_sources={"ldpred3": {"source_tree_sha256": "b" * 64}},
        run_controls={"rounds": 7}))
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar.name == "result.provenance.json"
    assert metadata["artifact"] == "result.csv"
    assert metadata["source_revision"] == revision
    assert metadata["source_clean"] is True
    assert metadata["schema_version"] == 2
    assert metadata["dependency_sources"] == {
        "ldpred3": {"source_tree_sha256": "b" * 64}}
    assert metadata["inputs_sha256"] == {"source/input.dat": "a" * 64}
    assert metadata["run_controls"] == {"rounds": 7}
    assert set(metadata["packages"]) == {"bipred", "ldpred3", "numpy", "numba"}


def test_run_all_refuses_an_untracked_source_file(tmp_path):
    """HEAD provenance excludes untracked code, not only unstaged diffs."""
    import pathlib
    import shutil

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "benchmarks" / "run_all.sh")
    repo = tmp_path / "repo"
    scripts = repo / "benchmarks"
    scripts.mkdir(parents=True)
    shutil.copy2(source, scripts / "run_all.sh")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "benchmarks/run_all.sh"], cwd=repo,
                   check=True)
    subprocess.run(
        ["git", "-c", "user.name=Benchmark test", "-c",
         "user.email=benchmark@example.invalid", "-c",
         "commit.gpgsign=false", "commit", "-qm", "fixture"],
        cwd=repo, check=True)
    (repo / "untracked.py").write_text("# changes the source tree\n")
    proc = subprocess.run(
        ["bash", "benchmarks/run_all.sh"], cwd=repo,
        capture_output=True, text=True)
    assert proc.returncode == 2
    assert "dirty source tree" in proc.stderr


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


def test_results_bivariate_table_matches_per_replicate_csv():
    import csv
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "benchmarks"
    records = list(csv.DictReader((root / "bivariate_demo.csv").open()))
    table = _markdown_table_after(
        (root / "RESULTS.md").read_text(encoding="utf-8"),
        "**Table 12. Trait-2 genetic R² under paired reference-LD regularisation.**",
    )
    labels = {
        "shared, rg=0.0": "shared, target 0.0",
        "shared, rg=0.3": "shared, target 0.3",
        "shared, rg=0.6": "shared, target 0.6",
        "shared, rg=0.9": "shared, target 0.9",
        "disjoint causal": "disjoint causal support",
    }
    groups = []
    for shrinkage in (0.0, 0.05):
        for source_label in labels:
            group = [row for row in records
                     if float(row["reference_shrinkage"]) == shrinkage
                     and row["architecture"] == source_label]
            assert len(group) == 6
            groups.append((shrinkage, source_label, group))
    assert len(table) == len(groups) == 10

    def mean(group, column):
        return float(np.mean([float(row[column]) for row in group]))

    def close(cell, value):
        numeric = cell.lstrip("+")
        digits = len(numeric.partition(".")[2])
        return abs(float(numeric) - value) <= 0.5 * 10.0 ** -digits * (1 + 1e-9)

    for printed, (shrinkage, source_label, group) in zip(table, groups):
        shrink, architecture, realized, alone, joint, gain, estimate, warned = printed
        assert shrink == f"{shrinkage * 100:g}%"
        assert architecture == labels[source_label]
        assert close(realized, mean(group, "realized_rg"))
        assert close(alone, mean(group, "solo_r2"))
        assert close(joint, mean(group, "joint_r2"))
        assert close(gain, mean(group, "gain"))
        assert close(estimate, mean(group, "rg_est"))
        warned_count = sum(int(int(row["joint_implausible_warnings"]) > 0)
                           for row in group)
        assert warned == f"{warned_count} / {len(group)}"


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


def test_results_environmental_overlap_table_matches_its_csv():
    """Table 10 carries this release's central claim, so transcribe it exactly.

    It asserts that the environmental-overlap failure mode was an artifact of
    in-fit LD quantization, and a hand-edit had silently duplicated the realized
    r_g into the environmental-correlation column of both `rg_target=0.5` rows
    -- the two rows the claim rests on.
    """
    import csv
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "benchmarks"
    rows = list(csv.DictReader((root / "rg_env_overlap.csv").open()))
    table = _markdown_table_after(
        (root / "RESULTS.md").read_text(encoding="utf-8"),
        "**Table 11. Paired MAE against realized genetic correlation.**",
    )
    assert len(table) == len(rows)

    for printed, record in zip(table, rows):
        target, env, realized, ldsc, joint = printed
        ldsc_free, ldsc_con = (c.strip() for c in ldsc.split("/"))
        joint_unset, joint_set = (c.strip() for c in joint.split("/"))
        for cell, column in (
            (target, "rg_target"), (env, "re"), (realized, "rg_realized"),
            (ldsc_free, "ldsc_free_mae_realized"),
            (ldsc_con, "ldsc_con_mae_realized"),
            (joint_unset, "biv_cc0_mae_realized"),
            (joint_set, "biv_cc_mae_realized"),
        ):
            digits = len(cell.partition(".")[2])
            half_ulp = 0.5 * 10.0 ** -digits * (1.0 + 1e-9)
            assert abs(float(cell) - float(record[column])) <= half_ulp, (
                f"{column}: table says {cell!r}, CSV says {record[column]!r}")

    # The claim itself: no cell may exceed the bound the prose states.
    worst = max(float(r["biv_cc0_mae_realized"]) for r in rows)
    assert worst <= 0.0242 + 1e-9, worst


def test_results_polygenicity_table_matches_its_csv():
    """Table 8 is the evidence that the shared-fraction bias is not a constant.

    Its whole point is a comparison between two columns -- the bias and the
    replicate spread beside it -- so a transcription slip in either one would
    invert the reading. The derived columns (bias, and the mean-bias sentence
    below the table) are recomputed here rather than trusted.
    """
    import csv
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "benchmarks"
    rows = [r for r in csv.DictReader((root / "mixer_overlap.csv").open())
            if r["sweep"] == "polygenicity"]
    text = (root / "RESULTS.md").read_text(encoding="utf-8")
    table = _markdown_table_after(
        text, "**Table 8. Shared-fraction bias against per-trait polygenicity.**")
    assert len(table) == len(rows) == 12

    def close(cell, value):
        digits = len(cell.partition(".")[2])
        return abs(float(cell) - value) <= 0.5 * 10.0 ** -digits * (1.0 + 1e-9)

    for printed, record in zip(table, rows):
        fraction, target, estimate, bias, relative = printed
        shown, spread = (c.strip() for c in estimate.split("±"))
        hat = float(record["frac_shared_hat"])
        goal = float(record["frac_shared_target"])
        assert close(fraction, float(record["true_pi1"])), printed
        assert close(target, goal), printed
        assert close(shown, hat), printed
        assert close(spread, float(record["frac_shared_sd"])), printed
        assert close(relative, float(record["rel_poly"])), printed
        # The bias column is derived, so recompute it instead of transcribing.
        assert close(bias, hat - goal), printed

    def biases(fraction):
        return [float(r["frac_shared_hat"]) - float(r["frac_shared_target"])
                for r in rows if abs(float(r["true_pi1"]) - fraction) < 1e-9]

    printed_means = re.search(
        r"Mean bias by causal fraction: (.+?)\.\s", text, re.S).group(1)
    found = dict(zip((0.01, 0.03, 0.10, 0.30),
                     re.findall(r"\*\*([+-][\d.]+)\*\*", printed_means)))
    assert len(found) == 4, printed_means
    for fraction, cell in found.items():
        mean = sum(biases(fraction)) / len(biases(fraction))
        assert close(cell.lstrip("+"), mean), (fraction, cell, mean)

    # The two claims the section is built on, asserted against the CSV.
    assert all(b < float(r["frac_shared_sd"])
               for r, b in zip([r for r in rows
                                if abs(float(r["true_pi1"]) - 0.03) < 1e-9],
                               biases(0.03))), "0.03 bias should be under 1 SD"
    ratios = [b / float(r["frac_shared_sd"])
              for r, b in zip([r for r in rows
                               if abs(float(r["true_pi1"]) - 0.30) < 1e-9],
                              biases(0.30))]
    # The prose quotes this range to one decimal; hold it to that.
    assert (round(min(ratios), 1), round(max(ratios), 1)) == (2.9, 12.6), ratios


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


def test_results_real_data_table_matches_its_csv():
    """Table 13 exactly transcribes its historical three-stage artifact.

    The numerical rows predate the current always-run screen semantics. This
    test protects the saved contrast and divergence-warning label; it does not
    promote the old artifact into current-screen validation.
    """
    import csv
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "benchmarks"
    csv_path = root / "real_ldl_cad.csv"
    if not csv_path.exists():             # inputs are ~9 GB and not committed
        import pytest
        pytest.skip("real_ldl_cad.csv not generated on this host")
    rows = list(csv.DictReader(csv_path.open()))
    table = _markdown_table_after(
        (root / "RESULTS.md").read_text(encoding="utf-8"),
        "**Table 13. The same analysis at three levels of cleaning.**")
    assert len(table) == len(rows) == 3

    def number(cell):
        return float(cell.replace("**", "").replace(",", "").strip())

    columns = [None, "m", "ldsc_rg", "rg", "h2_ldl", "h2_cad",
               "cancellation_ldl", "max_abs_beta_ldl", "trace_drift_ldl", None]
    for printed, record in zip(table, rows):
        assert len(printed) == len(columns)
        for cell, column in zip(printed, columns):
            if column is None:
                continue
            digits = len(cell.replace("**", "").partition(".")[2])
            half_ulp = 0.5 * 10.0 ** -digits * (1.0 + 1e-9)
            assert abs(number(cell) - float(record[column])) <= half_ulp, (
                f"{column}: table {cell!r}, CSV {record[column]!r}")
        # The divergence-warning column is prose in the table and 0/1 in CSV.
        assert printed[-1].strip() == (
            "yes" if record["divergence_warned"] == "1" else "no")

    # Internal diagnostics recorded by this historical artifact, not an
    # published truth-range check.
    assert rows[-1]["divergence_warned"] == "0", (
        "final historical stage should have no divergence warning")
    assert all(r["divergence_warned"] == "1" for r in rows[:-1]), (
        "earlier historical stages should trigger the divergence warning")
    assert float(rows[-1]["cancellation_ldl"]) < 10.0
