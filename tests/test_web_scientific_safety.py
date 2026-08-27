"""Regression tests for Catalog N semantics and scientific release gates."""

import csv
import gzip
import json
import math
from types import SimpleNamespace

import numpy as np


def _catalog_source(path, *, n_column="sample_size", values=(160_000, 140_000)):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "rsid", "effect_allele", "other_allele", "beta",
            "standard_error", n_column,
        ])
        for index, value in enumerate(values, 1):
            writer.writerow([f"rs{index}", "A", "G", 0.01, 0.02, value])
    return path


def _n_values(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [float(row["n"])
                for row in csv.DictReader(fh, delimiter="\t")]


def test_store_lock_heartbeat_and_immediate_reacquire_are_windows_portable(
        tmp_path, monkeypatch):
    from webapp import gwascat

    path = tmp_path / "catalog.lock"
    first = gwascat._StoreLock(path)
    assert first.acquire()
    first._touched = 0.0
    monkeypatch.setattr(
        gwascat.os, "utime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("lock heartbeat must not call os.utime(fd)")))
    first.touch()
    assert path.stat().st_size == 1
    first.release()
    assert path.exists()  # persistent pathname avoids successor-unlink races

    second = gwascat._StoreLock(path)
    assert second.acquire()
    second.release()
    assert path.exists()


def test_catalog_case_control_metadata_is_ancestry_labelled_and_effective():
    from webapp.gwascat import _sample_metadata_fields

    meta = _sample_metadata_fields(
        "25,000 European ancestry cases, 75,000 European ancestry controls")
    assert meta["sample_size_population"] == "European"
    assert meta["sample_size_design"] == "case_control"
    assert meta["n_total_selected"] == 100_000
    assert math.isclose(meta["n_eff"], 75_000.0)


def test_catalog_unknown_ancestry_total_is_not_called_effective_n():
    from webapp.gwascat import _sample_metadata_fields

    meta = _sample_metadata_fields("123,456 mixed ancestry individuals")
    assert meta["n_total_reported"] == 123_456
    assert meta["sample_size_population"] == "unresolved"
    assert "n_eff" not in meta


def test_cached_legacy_unknown_ancestry_n_eff_is_not_reused(tmp_path):
    from webapp import gwascat

    cache = tmp_path / "_meta" / "gwascat" / "GCST000099.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({
        "accession": "GCST000099", "trait": "Mixed trait",
        "sample": "123,456 mixed ancestry individuals",
        "n_eff": 123_456.0,
        "n_basis": "largest N in the catalog's initial sample size text",
    }))
    meta = gwascat._study_metadata("GCST000099", tmp_path)
    assert "n_eff" not in meta
    assert meta["n_total_reported"] == 123_456


def test_catalog_total_n_is_rescaled_to_selected_effective_n(
        tmp_path):
    from webapp import gwascat

    source = _catalog_source(tmp_path / "source.tsv.gz")
    dest = tmp_path / "job.tsv.gz"
    info = gwascat.fetch_filtered(
        str(source), dest, accession="GCST000001",
        root=tmp_path / "data", keep_ids={"rs1", "rs2"},
        fingerprint="a" * 64,
        coverage=lambda: ({"rs1", "rs2"}, {"a" * 64: "EUR"}),
        target_effective_n=75_000, target_total_n=100_000)

    # Median 150k -> selected total 100k -> effective 75k: factor 0.5.
    np.testing.assert_allclose(_n_values(dest), [80_000, 70_000])
    assert info["n_source_kind"] == "reported_total"
    assert info["sample_size_safe_for_effective_n"] is True
    assert math.isclose(info["sample_size_transform"]["factor"], 0.5)
    assert math.isclose(
        np.median(_n_values(dest)),
        info["sample_size_transform"]["target_effective_n"])


def test_catalog_total_n_without_a_target_is_not_marked_effective_safe(
        tmp_path):
    from webapp import gwascat

    source = _catalog_source(tmp_path / "source.tsv.gz", values=(100_000,))
    dest = tmp_path / "job.tsv.gz"
    info = gwascat.fetch_filtered(
        str(source), dest, accession="GCST000002",
        root=tmp_path / "data", keep_ids={"rs1"},
        fingerprint="b" * 64,
        coverage=lambda: ({"rs1"}, {"b" * 64: "EUR"}))
    assert info["has_per_variant_n"] is True
    assert info["sample_size_transform"] is None
    assert info["sample_size_safe_for_effective_n"] is False


def test_unsafe_catalog_total_does_not_replace_explicit_effective_n(
        tmp_path, monkeypatch):
    from webapp import runner

    job = {
        "id": "job", "files": {}, "stages": [], "stage_details": {},
        "options": {
            "catalog1": {}, "catalog_n_user_supplied1": True,
            "n_eff1": 75_000.0, "n_cases1": None, "n_controls1": None,
        },
    }
    monkeypatch.setattr(runner.jobs, "save_job", lambda *_args: None)
    info = {
        "kept": 1, "seen": 1, "effect_from": "beta",
        "sha256": "a" * 64, "normalised_sha256": "b" * 64,
        "has_per_variant_n": True, "per_variant_n_usable_frac": 1.0,
        "n_source_column": "sample_size",
        "n_source_kind": "reported_total",
        "sample_size_transform": None,
        "sample_size_safe_for_effective_n": False,
    }
    runner._Stages(tmp_path, job).finish_acquire_trait(
        1, job["options"]["catalog1"], tmp_path / "trait1.tsv.gz",
        info, "download", {})
    assert job["options"]["n_eff1"] == 75_000.0


def test_catalog_auto_n_requires_explicit_reference_population_agreement():
    from webapp.runner import (
        _catalog_n_requires_explicit, _population_agrees,
        _reference_population,
    )

    assert _reference_population("ukb-eur-hm3") == "European"
    assert _reference_population("AFR-reference") == "African"
    assert _reference_population("east-asian-panel") == "East Asian"
    assert _reference_population("production") is None
    assert _reference_population("eur-afr-combined") is None
    assert _population_agrees("European", "European")
    assert not _population_agrees("European", "African")

    options = {
        "n_eff1": None, "n_cases1": 25_000.0, "n_controls1": 75_000.0,
        "catalog_n_user_supplied1": False,
    }
    total_info = {
        "has_per_variant_n": True,
        "sample_size_safe_for_effective_n": False,
        "n_source_kind": "reported_total",
    }
    assert _catalog_n_requires_explicit(options, 1, total_info, False)
    assert not _catalog_n_requires_explicit(options, 1, total_info, True)

    # A canonical effective-N column is self-contained and does not consume
    # the Catalog's automatic European scalar.
    variant_info = {
        "has_per_variant_n": True,
        "sample_size_safe_for_effective_n": True,
        "n_source_kind": "variant_n",
    }
    assert not _catalog_n_requires_explicit(options, 1, variant_info, False)
    options["catalog_n_user_supplied1"] = True
    assert not _catalog_n_requires_explicit(options, 1, total_info, False)


def test_n_basis_reports_the_scalar_retained_for_an_unsafe_legacy_n():
    from webapp.runner import _n_basis

    options = {
        "catalog1": {
            "has_per_variant_n": True,
            "sample_size_safe_for_effective_n": False,
            "per_variant_n_usable_frac": 1.0,
        },
        "catalog_n_user_supplied1": False,
        "n_eff1": None, "n_cases1": 25_000.0, "n_controls1": 75_000.0,
    }
    assert _n_basis(options, 1) == "4/(1/n_cases + 1/n_controls)"
    options["n_cases1"] = options["n_controls1"] = None
    options["n_eff1"] = 75_000.0
    assert _n_basis(options, 1) == "constant effective N"


def test_fit_release_gate_catches_implausible_warning_and_af_mismatch():
    from webapp.runner import _assess_allele_frequency, _warnings_are_critical

    assert _warnings_are_critical([{
        "message": "Implausible bivariate fit: h2 reached its upper bound",
    }])
    quality = _assess_allele_frequency({"trait1": 0.99, "trait2": 0.2})
    assert quality["status"] == "warning"
    assert quality["critical"] is True
    assert quality["failed_traits"] == ["trait2"]
    assert _warnings_are_critical([{
        "message": quality["summary"], "critical": quality["critical"],
    }])


def test_fit_release_gate_rejects_nonfinite_prediction_vector():
    from webapp.runner import _fit_result_issues

    class Result:
        beta1_est = np.array([0.1, np.nan])
        beta2_est = np.array([0.2, 0.3])

    joint = {
        "h2": [0.2, 0.3], "rg": 0.1, "p": 0.01,
        "pi": [0.8, 0.1, 0.05, 0.05],
        "retained_iterations": 100,
        "noise_scale": [1.0, 1.0],
        "mixer": {"n_causal": [10.0, 20.0], "n_shared": 5.0},
    }
    assert any("beta1_est" in issue
               for issue in _fit_result_issues(Result(), joint))


def _consistent_fit_result():
    pi = [0.7, 0.1, 0.15, 0.05]
    poly = [pi[1] + pi[3], pi[2] + pi[3]]
    shared = 100 * pi[3]
    rho = 0.4
    joint = {
        "h2": [0.2, 0.3], "rg": 0.1, "p": sum(pi[1:]), "pi": pi,
        "retained_iterations": 100, "noise_scale": [1.0, 1.0],
        "mixer": {
            "polygenicity": poly,
            "n_causal": [100 * poly[0], 100 * poly[1]],
            "n_shared": shared,
            "frac_shared": shared / (100 * min(poly)),
            "rho_beta": rho,
            "rg_from_overlap": rho * pi[3] / math.sqrt(poly[0] * poly[1]),
        },
    }
    result = SimpleNamespace(
        beta1_est=np.zeros(100), beta2_est=np.zeros(100))
    return result, joint


def test_fit_release_gate_accepts_a_consistent_mixer_overlap():
    from webapp.runner import _fit_result_issues

    result, joint = _consistent_fit_result()
    assert _fit_result_issues(result, joint) == []


def test_fit_release_gate_rejects_finite_but_impossible_mixer_overlap():
    from webapp.runner import _fit_result_issues

    result, joint = _consistent_fit_result()
    joint["mixer"]["n_shared"] = 30.0
    issues = _fit_result_issues(result, joint)
    assert any("n_shared" in issue for issue in issues)

    result, joint = _consistent_fit_result()
    joint["mixer"]["n_causal"] = [120.0, 20.0]
    issues = _fit_result_issues(result, joint)
    assert any("fitted variant panel" in issue for issue in issues)

    result, joint = _consistent_fit_result()
    joint["mixer"]["rho_beta"] = 1.1
    issues = _fit_result_issues(result, joint)
    assert any("rho_beta" in issue for issue in issues)

    result, joint = _consistent_fit_result()
    joint["mixer"]["frac_shared"] = 0.9
    issues = _fit_result_issues(result, joint)
    assert any("frac_shared is inconsistent" in issue for issue in issues)


def test_exact_n_column_remains_a_variant_n_for_legacy_compatibility(tmp_path):
    from webapp import gwascat

    source = _catalog_source(
        tmp_path / "source.tsv.gz", n_column="n", values=(100_000,))
    info = gwascat.stream_filter(
        str(source), {"rs1"}, tmp_path / "normalised.tsv.gz")
    assert info["n_source_kind"] == "variant_n"
    assert info["has_per_variant_n"] is True


def test_exact_n_column_is_not_case_control_scaled_again(tmp_path):
    from webapp import gwascat

    source = _catalog_source(
        tmp_path / "source.tsv.gz", n_column="n", values=(80_000, 70_000))
    dest = tmp_path / "job.tsv.gz"
    info = gwascat.fetch_filtered(
        str(source), dest, accession="GCST000003",
        root=tmp_path / "data", keep_ids={"rs1", "rs2"},
        fingerprint="c" * 64,
        coverage=lambda: ({"rs1", "rs2"}, {"c" * 64: "EUR"}),
        target_effective_n=75_000, target_total_n=100_000)

    np.testing.assert_allclose(_n_values(dest), [80_000, 70_000])
    assert info["n_source_kind"] == "variant_n"
    assert info["sample_size_transform"] is None
    assert info["sample_size_safe_for_effective_n"] is True
