"""Unit and end-to-end tests for the checkout-only bipred web service."""

import asyncio
import csv
import gzip
import hashlib
import inspect
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

JOB_TIMEOUT = 240.0


@pytest.fixture(scope="module")
def web(tmp_path_factory):
    root = tmp_path_factory.mktemp("webdata")
    os.environ["BIPRED_WEB_DATA"] = str(root)
    os.environ["BIPRED_WEB_CONCURRENCY"] = "1"
    os.environ.pop("BIPRED_WEB_CACHES", None)
    from webapp import caches, demo
    demo.build_demo(caches.demo_cache_dir(root), m=1500, n_samples=600, seed=7)
    # Normal uploads need an explicitly registered real cache.  The fixture
    # reuses demo bytes under a clearly test-only registry key; production code
    # refuses the synthetic ``demo`` key outside /demo.
    os.environ["BIPRED_WEB_CACHES"] = (
        f"TEST={root / 'caches' / 'demo' / 'demo.ld.npz'}")
    from webapp.app import create_app
    with TestClient(create_app()) as client:
        yield client, root
    del os.environ["BIPRED_WEB_DATA"]
    del os.environ["BIPRED_WEB_CONCURRENCY"]
    del os.environ["BIPRED_WEB_CACHES"]


def _wait_for_terminal(root, job_id, timeout=JOB_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with open(root / "jobs" / job_id / "job.json") as fh:
            job = json.load(fh)
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(2)
    raise TimeoutError(f"job {job_id} still {job['status']} after {timeout}s")


def _demo_upload(root, trait):
    return (f"trait{trait}.tsv",
            open(root / "caches" / "demo" / f"trait{trait}.tsv", "rb"),
            "text/tab-separated-values")


def test_save_upload_unlinks_a_partial_file_on_read_error(tmp_path):
    from webapp.app import _save_upload

    class BrokenUpload:
        filename = "private.tsv"

        def __init__(self):
            self.reads = 0

        async def read(self, _size):
            self.reads += 1
            if self.reads == 1:
                return b"private partial contents"
            raise OSError("client stream failed")

    dest = tmp_path / "private.tsv"
    with pytest.raises(OSError, match="client stream failed"):
        asyncio.run(_save_upload(BrokenUpload(), dest, 1024))
    assert not dest.exists()


@pytest.mark.parametrize("failure_type", [OSError, asyncio.CancelledError])
def test_interrupted_submit_removes_the_staging_job(
        web, monkeypatch, failure_type):
    import webapp.app as app_module

    client, root = web
    route = next(
        route for route in client.app.routes
        if getattr(route, "path", None) == "/jobs"
        and "POST" in getattr(route, "methods", set()))
    kwargs = {
        name: getattr(parameter.default, "default", parameter.default)
        for name, parameter in inspect.signature(route.endpoint).parameters.items()
        if name != "request"
    }

    class NamedUpload:
        def __init__(self, filename):
            self.filename = filename

    kwargs.update({
        "request": None,
        "sumstats1": NamedUpload("private1.tsv"),
        "sumstats2": NamedUpload("private2.tsv"),
        "cache_key": "TEST",
    })
    before = {path.name for path in (root / "jobs").iterdir()}

    async def interrupted(_upload, dest, _cap):
        dest.write_bytes(b"private partial contents")
        raise failure_type("upload interrupted")

    monkeypatch.setattr(app_module, "_save_upload", interrupted)
    with pytest.raises(failure_type):
        asyncio.run(route.endpoint(**kwargs))
    assert {path.name for path in (root / "jobs").iterdir()} == before


def test_index_offers_demo_and_form(web):
    client, _ = web
    page = client.get("/")
    assert page.status_code == 200
    assert "Run the synthetic demo" in page.text
    assert 'name="sumstats1"' in page.text
    assert 'name="n_eff1"' in page.text
    assert 'name="screen"' not in page.text
    assert "Mandatory LD-consistency screen" in page.text
    assert "own usable principal LD panel before intersection" in page.text
    assert "screened summary-statistics arrays are retained" in page.text
    assert "can outlive the job" in page.text
    assert "sensitive or private data" in page.text
    submit = next(route for route in client.app.routes
                  if getattr(route, "path", None) == "/jobs"
                  and "POST" in getattr(route, "methods", set()))
    assert "screen" not in {field.name for field in submit.dependant.body_params}
    # Header navigation.
    assert 'href="/catalog"' in page.text
    assert 'aria-current="page"' in page.text


def test_index_carries_preview_assets(web):
    client, _ = web
    page = client.get("/")
    assert page.status_code == 200
    assert "BIPRED_ALIASES" in page.text      # alias map for the JS preview
    assert 'class="card trait-1"' in page.text
    for asset in ("style.css", "job.js", "preview.js"):
        assert client.get(f"/static/{asset}").status_code == 200


def test_job_progress_contract_supports_one_line_per_trait(web):
    from webapp import jobs

    client, root = web
    job = jobs.create_job(
        root, options={"weights": False},
        labels={"trait1": "A", "trait2": "B"}, status="staging")
    job["status"] = "running"
    job["stage"] = "screen"
    job["active_stages"] = ["prepare", "screen"]
    jobs.save_job(root, job)

    page = client.get(f"/jobs/{job['id']}")
    script = client.get("/static/job.js").text
    assert page.status_code == 200
    assert 'id="progress-lines"' in page.text
    assert 'class="progress-lines"' in page.text
    assert ('id="stage-name">Prepare each trait + Run LD-consistency '
            'screen</strong>') in page.text
    assert "stageSchema:" in page.text
    assert "p.traits" in script
    assert "Object.keys(p.traits).sort()" in script
    assert "active_stages" in script
    assert 'join(" + ")' in script
    # Trait-scoped events retain both their identity and useful step text;
    # ``phase`` is stage grouping, not a replacement for the operation label.
    assert 'p.phase === "burn-in" || p.phase === "sampling"' in script
    assert "fitPhase ? p.phase : (p.step || p.phase)" in script
    assert '(p.trait ? where + " — " : "") + renderStep(p)' in script
    assert "progressLines.replaceChildren(...nodes)" in script
    assert "ld_consistency_screen" in script


def test_job_poller_reloads_after_stage_schema_upgrade(web):
    """A page opened while an old queued job waits must adopt the live schema."""
    client, _ = web
    script = client.get("/static/job.js").text

    schema_check = (
        "Number(s.stage_schema || 1) !== Number(cfg.stageSchema || 1)")
    assert schema_check in script
    check_at = script.index(schema_check)
    assert script.index("window.location.reload();", check_at) > check_at
    assert script.index("return;", check_at) > check_at
    assert check_at < script.index("renderStages(s);", check_at)


@pytest.mark.parametrize(
    "value", ["0", "-1", "nan", "inf", "-inf", "not-a-number"])
def test_config_rejects_a_retention_period_that_cannot_delete(value, monkeypatch):
    from webapp.app import _config

    monkeypatch.setenv("BIPRED_WEB_TTL_DAYS", value)
    with pytest.raises(ValueError, match=(
            "BIPRED_WEB_TTL_DAYS must be a finite number greater than zero")):
        _config()


def test_config_accepts_a_positive_finite_retention_period(monkeypatch):
    from webapp.app import _config

    monkeypatch.setenv("BIPRED_WEB_TTL_DAYS", "0.5")
    assert _config()["ttl_days"] == 0.5


@pytest.mark.slow
@pytest.mark.integration
def test_demo_job_end_to_end(web):
    client, root = web
    assert client.get("/demo", follow_redirects=False).status_code == 405
    redirect = client.post("/demo", follow_redirects=False)
    assert redirect.status_code == 303
    job_id = redirect.headers["location"].rsplit("/", 1)[-1]
    job = _wait_for_terminal(root, job_id)
    assert job["status"] == "done", job.get("error")

    with open(root / "jobs" / job_id / "result.json") as fh:
        result = json.load(fh)
    assert -1.0 <= result["joint"]["rg"] <= 1.0
    assert result["joint"]["h2"][0] > 0.0
    assert job["options"]["screen"] is True
    assert result["provenance"]["screen"] is True
    assert result["munge"]["n_kept"] > 0
    assert result["weights"] == ["weights1.tsv", "weights2.tsv"]
    # Posterior uncertainty is reported for the headline estimates.
    assert result["joint"]["rg_sd"] is None or result["joint"]["rg_sd"] >= 0
    # munge.json carries the per-step QC and harmonization logs.
    t1 = result["munge"]["trait1"]
    assert t1["qc"]["n_input"] >= t1["qc"]["n_kept"] > 0
    assert t1["harmonize"]["n_matched"] > 0
    assert t1["n_usable"] > 0
    for trait in ("trait1", "trait2"):
        screen = result["munge"][trait]["ld_consistency_screen"]
        assert screen["n_input"] == screen["n_kept"] + screen["n_dropped"]
    assert "af_corr" in result["munge"]
    assert result["diagnostics"]["valid_for_interpretation"] in (True, False)
    divergence = result["diagnostics"]["divergence"]
    assert divergence["evaluated"] is (
        divergence["variant_count"] >=
        divergence["thresholds"]["minimum_variants"])
    assert set(divergence["traits"]) == {"trait1", "trait2"}
    assert divergence["thresholds"]["effect_energy_ratio"] == 10.0
    # result.json is strict JSON; no NaN/Infinity escapes from a diagnostic.
    json.dumps(result, allow_nan=False)
    assert result["provenance"]["compute"]["logical_cpus"] >= 1
    assert result["provenance"]["resources"]["wall_s"] > 0
    peak_rss = result["provenance"]["resources"]["peak_rss_gb"]
    assert peak_rss is None or peak_rss > 0
    assert result["provenance"]["sample_size"]["trait1"]["median"] > 0
    ldsc = result["ldsc"]
    assert ldsc["m_snps"] == 1500
    assert 0 < ldsc["n_regression_variants"] <= ldsc["m_snps"]
    assert ldsc["score_definition"] == (
        "full-reference-transformed-cache-colsum-r2-including-self")
    assert ldsc["score_algorithm"] == "ldpred3.ld_scores-v1"
    assert ldsc["finite_reference_correction"] == "none"
    assert ldsc["effective_rank"] > 0
    assert len(ldsc["h2_init"]) == 2
    assert result["joint"]["h2_init"] == ldsc["h2_init"]

    page = client.get(f"/jobs/{job_id}/results")
    assert page.status_code == 200
    assert "polygenic overlap" in page.text.lower()
    assert "Fit-stability diagnostics" in page.text
    assert "Full-reference-score LDSC-style" in page.text
    assert "Participation-ratio effective rank" in page.text
    assert "QC and harmonization" in page.text
    assert "After mandatory screen" in page.text
    assert "joint pre-screen drop" in page.text
    assert "Dropped by the LD-consistency screen" not in page.text
    assert "<svg" in page.text and "Model-implied MiXeR overlap" in page.text
    for kind in ("result", "munge", "weights1", "weights2"):
        assert client.get(f"/jobs/{job_id}/download/{kind}").status_code == 200


@pytest.mark.slow
@pytest.mark.integration
def test_upload_job_end_to_end(web):
    client, root = web
    with _demo_upload(root, 1)[1] as f1, _demo_upload(root, 2)[1] as f2:
        response = client.post(
            "/jobs",
            files={"sumstats1": ("t1.tsv", f1, "text/tab-separated-values"),
                   "sumstats2": ("t2.tsv", f2, "text/tab-separated-values")},
            data={"label1": "upload 1", "label2": "upload 2",
                  "n_eff1": "100000", "n_eff2": "100000",
                  "cache_key": "TEST"},
            follow_redirects=False)
    assert response.status_code == 303
    job_id = response.headers["location"].rsplit("/", 1)[-1]
    job = _wait_for_terminal(root, job_id)
    assert job["status"] == "done", job.get("error")
    assert job["options"]["screen"] is True
    with open(root / "jobs" / job_id / "result.json") as fh:
        result = json.load(fh)
    assert -1.0 <= result["joint"]["rg"] <= 1.0


@pytest.mark.slow
@pytest.mark.integration
def test_unreadable_columns_fail_the_job(web):
    client, root = web
    response = client.post(
        "/jobs",
        files={"sumstats1": ("bad.tsv", b"foo\tbar\n1\t2\n", "text/tsv"),
               "sumstats2": _demo_upload(root, 2)},
        data={"n_eff1": "100000", "n_eff2": "100000", "cache_key": "TEST"},
        follow_redirects=False)
    assert response.status_code == 303
    job_id = response.headers["location"].rsplit("/", 1)[-1]
    job = _wait_for_terminal(root, job_id)
    assert job["status"] == "failed"
    assert "column" in job["error"]


def test_incomplete_case_control_rejected(web):
    client, root = web
    response = client.post(
        "/jobs",
        files={"sumstats1": _demo_upload(root, 1),
               "sumstats2": _demo_upload(root, 2)},
        data={"cache_key": "TEST", "label1": "MyTrait",
              "n_cases1": "100", "n_eff2": "100000"})
    assert response.status_code == 400
    assert "both cases and controls" in response.text
    # The upload form is re-rendered inline with entries preserved, not the
    # bare error page.
    assert 'name="sumstats1"' in response.text
    assert 'value="MyTrait"' in response.text
    assert "re-select" in response.text
    assert "Something needs fixing" not in response.text
    assert 'name="screen"' not in response.text
    assert "Mandatory LD-consistency screen" in response.text


def test_public_fit_schedule_rejects_non_diagnostic_chain(web):
    client, root = web
    with _demo_upload(root, 1)[1] as f1, _demo_upload(root, 2)[1] as f2:
        response = client.post(
            "/jobs",
            files={"sumstats1": ("t1.tsv", f1, "text/tab-separated-values"),
                   "sumstats2": ("t2.tsv", f2, "text/tab-separated-values")},
            data={"cache_key": "TEST", "n_eff1": "100000",
                  "n_eff2": "100000", "burn_in": "0", "num_iter": "1"})
    assert response.status_code == 400
    assert "burn-in: must be between 50" in response.text


def test_obsolete_screen_form_field_cannot_disable_mandatory_screen(web):
    client, root = web
    response = client.post(
        "/jobs",
        files={"sumstats1": _demo_upload(root, 1),
               "sumstats2": _demo_upload(root, 2)},
        data={"cache_key": "TEST", "n_cases1": "100",
              "n_eff2": "100000", "screen": ""})
    assert response.status_code == 400
    assert 'name="screen"' not in response.text
    assert "Mandatory LD-consistency screen" in response.text


def test_job_page_shows_cache_aware_stage_definitions(web):
    """The visible stages follow the reusable-trait boundary, not old verbs."""
    from webapp import jobs

    client, root = web
    labels = {"trait1": "First trait", "trait2": "Second trait"}
    upload = jobs.create_job(
        root, options={"weights": False}, labels=labels, status="staging")
    upload_page = client.get(f"/jobs/{upload['id']}")
    assert upload_page.status_code == 200
    upload_labels = [
        "Prepare each trait",
        "Run LD-consistency screen",
        "Combine the two traits",
        "Run LD-score diagnostic",
        "Fit bivariate model",
    ]
    assert all(label in upload_page.text for label in upload_labels)
    assert [upload_page.text.index(label) for label in upload_labels] == sorted(
        upload_page.text.index(label) for label in upload_labels)
    assert "Get Catalog data" not in upload_page.text
    assert "Write prediction weights" not in upload_page.text

    catalog = jobs.create_job(
        root,
        options={"gcst1": "GCST000001", "gcst2": "", "weights": True},
        labels=labels, status="staging")
    catalog_page = client.get(f"/jobs/{catalog['id']}")
    assert catalog_page.status_code == 200
    catalog_labels = [
        "Get Catalog data",
        *upload_labels,
        "Write prediction weights",
    ]
    assert [catalog_page.text.index(label) for label in catalog_labels] == \
        sorted(catalog_page.text.index(label) for label in catalog_labels)
    assert "Prepare both traits independently against one shared LD" in \
        catalog_page.text
    assert "quick univariate LD-score h2/intercept check" in \
        catalog_page.text
    assert "harmonized, and screened trait artifact" in catalog_page.text
    assert "mandatory DENTIST-inspired" in catalog_page.text
    assert "trait-local screen" in catalog_page.text
    assert "as soon as that trait is ready" in catalog_page.text
    assert "Recomputed for every analysis" in catalog_page.text
    assert "prepare summary statistics" not in catalog_page.text.lower()


def test_status_endpoint_keeps_completed_stage_details(web):
    """Cache outcomes remain available after transient progress is cleared."""
    from webapp import jobs

    client, root = web
    job = jobs.create_job(
        root, options={"weights": False},
        labels={"trait1": "First", "trait2": "Second"}, status="staging")
    job["stages"]["screen"] = 1.25
    job["active_stages"] = ["pair"]
    job["stage_details"]["screen"] = {
        "summary": "2 complete post-screen trait artifacts reused.",
        "traits": {
            "trait1": "reused QC'd, LD-aligned, screened data",
            "trait2": "reused QC'd, LD-aligned, screened data",
        },
    }
    job["progress"] = None
    jobs.save_job(root, job)

    payload = client.get(f"/jobs/{job['id']}/status").json()
    assert payload["stage_schema"] == jobs.STAGE_SCHEMA
    assert payload["progress"] is None
    assert payload["active_stages"] == ["pair"]
    assert payload["stage_details"] == job["stage_details"]


def test_legacy_jobs_keep_their_original_visible_stages():
    """Schema-1 and schema-2 jobs retain their original stage semantics."""
    from webapp import jobs
    from webapp.app import TEMPLATES

    legacy = {
        "options": {"gcst1": "GCST000001", "weights": False},
        # Deliberately no stage_schema: historical job.json files lack it.
    }
    definitions = jobs.stage_definitions(legacy)
    assert [item["key"] for item in definitions] == [
        "download", "validate", "harmonize", "ldsc", "fit"]
    assert [item["label"] for item in definitions[:3]] == [
        "Get Catalog data", "Check input columns", "Prepare and combine traits"]
    assert "intersected fitted panel" in definitions[3]["description"]
    assert "precomputed LD scores" not in definitions[3]["description"]
    assert jobs.stage_label("harmonize", 1) == "Prepare and combine traits"
    fallback = TEMPLATES.env.get_template(
        "_stage_ui.html").module.fallback_description("ldsc")
    assert "intersected fitted panel" in str(fallback)

    legacy_upload = {"options": {"weights": True}}
    assert [item["key"] for item in jobs.stage_definitions(legacy_upload)] == [
        "validate", "harmonize", "ldsc", "fit", "weights"]

    schema2 = {
        "stage_schema": 2,
        "options": {"gcst1": "GCST000001", "weights": True},
    }
    definitions = jobs.stage_definitions(schema2)
    assert [item["key"] for item in definitions] == [
        "acquire", "prepare", "pair", "ldsc", "fit", "weights"]
    assert "optional LD-consistency screen" in definitions[2]["description"]
    assert "Catalog preparations can be reused" in definitions[1]["description"]
    assert all(item["key"] != "screen" for item in definitions)
    assert jobs.stage_label("pair", 2) == "Combine the two traits"

    schema3 = {
        "stage_schema": 3,
        "options": {"gcst1": "GCST000001", "weights": False},
    }
    definitions = jobs.stage_definitions(schema3)
    assert [item["key"] for item in definitions] == [
        "acquire", "prepare", "screen", "pair", "ldsc", "fit"]
    assert "Validate and read both inputs" in definitions[1]["description"]
    assert "before storing it" in definitions[2]["description"]


@pytest.mark.slow
@pytest.mark.integration
def test_status_endpoint(web):
    from webapp import jobs

    client, root = web
    assert client.get("/jobs/nope/status").status_code == 404
    redirect = client.post("/demo", follow_redirects=False)
    job_id = redirect.headers["location"].rsplit("/", 1)[-1]
    job = _wait_for_terminal(root, job_id)
    response = client.get(f"/jobs/{job_id}/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == job_id
    assert payload["status"] == job["status"] == "done"
    assert payload["stage"] is None or isinstance(payload["stage"], str)
    assert payload["error"] is None
    assert payload["progress"] is None      # cleared once the stage completes
    assert payload["stage_schema"] == jobs.STAGE_SCHEMA == 4
    assert payload["stage_details"] == job["stage_details"]
    assert payload["stage_details"]["screen"]["mandatory"] is True
    assert set(payload["stage_details"]["screen"]["traits"]) == {
        "trait1", "trait2"}
    for trait in ("trait1", "trait2"):
        screen = payload["munge"][trait]["ld_consistency_screen"]
        assert screen["n_input"] == screen["n_kept"] + screen["n_dropped"]
    assert payload["stage_details"]["pair"]["rerun"] is True
    assert payload["stages"]["fit"] > 0
    assert payload["munge"]["n_kept"] > 0


# --- GWAS Catalog support -------------------------------------------------

def _fake_catalog_file(src_tsv, dest, empty_beta=False):
    """Rewrite a demo trait file into GWAS-Catalog harmonised schema (gz)."""
    rename = {"SNP": "rsid", "CHR": "chromosome", "BP": "base_pair_location",
              "A1": "effect_allele", "A2": "other_allele",
              "EAF": "effect_allele_frequency", "BETA": "beta",
              "SE": "standard_error"}
    with open(src_tsv, newline="") as fh, \
            gzip.open(dest, "wt", newline="") as out:
        reader = csv.DictReader(fh, delimiter="\t")
        writer = csv.DictWriter(
            out, fieldnames=["rsid", "chromosome", "base_pair_location",
                             "effect_allele", "other_allele", "beta",
                             "standard_error", "effect_allele_frequency",
                             "p_value", "n"],
            delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            out_row = {new: row[old] for old, new in rename.items()}
            if empty_beta:
                out_row["beta"] = ""    # a deposit with no usable effects
            out_row["p_value"] = "0.5"
            out_row["n"] = "100000"
            writer.writerow(out_row)
    return dest


def _fake_resolve(accession, root):
    if accession not in _FAKE_URLS:
        raise ValueError(f"{accession}: no such study in the GWAS Catalog")
    return {"accession": accession, "trait": f"Fake trait {accession[-2:]}",
            "title": "", "pmid": "12345", "n_eff": 100000.0,
            "n_basis": "test fixture", "url": _FAKE_URLS[accession],
            "remote_bytes": 0}


_FAKE_URLS = {}


def test_catalog_lookup(web, monkeypatch, tmp_path):
    client, root = web
    _FAKE_URLS["GCST000001"] = str(
        _fake_catalog_file(root / "caches" / "demo" / "trait1.tsv",
                           tmp_path / "cat1.h.tsv.gz"))
    monkeypatch.setattr("webapp.gwascat.resolve", _fake_resolve)
    page = client.get("/catalog/lookup?accession=GCST000001")
    assert page.status_code == 200
    assert page.json()["trait"] == "Fake trait 01"
    assert client.get("/catalog/lookup?accession=bogus").status_code == 404


def test_submit_needs_file_or_accession(web):
    client, _ = web
    response = client.post("/jobs", data={"cache_key": "TEST"})
    assert response.status_code == 400
    assert "file or give a GWAS Catalog accession" in response.text


def test_submit_rejects_file_plus_accession(web):
    client, root = web
    response = client.post(
        "/jobs",
        files={"sumstats1": _demo_upload(root, 1)},
        data={"gcst1": "GCST000001", "cache_key": "TEST"})
    assert response.status_code == 400
    assert "not both" in response.text


def test_catalog_autofill_marker_updates_stale_n(web, monkeypatch):
    client, _ = web

    def resolve(accession, _root):
        return {"accession": accession, "trait": "New catalog trait",
                "title": "", "pmid": "", "n_eff": 222222.0,
                "n_basis": "fixture", "url": "unused", "remote_bytes": 0}

    monkeypatch.setattr("webapp.gwascat.resolve", resolve)
    response = client.post(
        "/jobs", data={"gcst1": "GCST000777", "n_eff1": "111111",
                       "catalog_auto_n1": "1",
                       "label1": "Old catalog trait",
                       "catalog_auto_label1": "1",
                       "cache_key": "TEST"})
    assert response.status_code == 400       # trait 2 intentionally absent
    assert 'name="n_eff1"' in response.text
    assert 'value="222222.0"' in response.text
    assert 'value="New catalog trait"' in response.text


@pytest.mark.slow
@pytest.mark.integration
def test_catalog_job_end_to_end(web, monkeypatch, tmp_path):
    client, root = web
    for trait, acc in ((1, "GCST000001"), (2, "GCST000002")):
        _FAKE_URLS[acc] = str(
            _fake_catalog_file(root / "caches" / "demo"
                               / f"trait{trait}.tsv",
                               tmp_path / f"cat{trait}.h.tsv.gz"))
    monkeypatch.setattr("webapp.gwascat.resolve", _fake_resolve)
    response = client.post(
        "/jobs",
        data={"gcst1": "GCST000001", "gcst2": "GCST000002",
              "cache_key": "TEST"},          # N auto-filled from "metadata"
        follow_redirects=False)
    assert response.status_code == 303, response.text
    job_id = response.headers["location"].rsplit("/", 1)[-1]
    job = _wait_for_terminal(root, job_id)
    assert job["status"] == "done", job.get("error")
    assert "acquire" in job["stages"]
    assert "download" not in job["stages"]
    assert job["labels"]["trait1"] == "Fake trait 01"
    with open(root / "jobs" / job_id / "result.json") as fh:
        result = json.load(fh)
    assert -1.0 <= result["joint"]["rg"] <= 1.0
    catalog = result["provenance"]["catalog"]
    assert catalog["trait1"]["accession"] == "GCST000001"
    assert catalog["trait1"]["kept"] == catalog["trait1"]["seen"] > 0
    assert catalog["trait1"]["has_per_variant_n"] is True
    assert result["provenance"]["sample_size"]["trait1"]["basis"].startswith(
        "per-variant n column")
    # The supervisor records the success in the accession registry on reap.
    from webapp import gwascat
    deadline = time.time() + 15
    while time.time() < deadline:
        if gwascat.accession_registry(root).get("GCST000001", {}).get("works"):
            break
        time.sleep(1)
    entry = gwascat.accession_registry(root)["GCST000001"]
    assert entry["works"] is True
    assert entry["kept"] > 0


@pytest.mark.slow
@pytest.mark.integration
def test_unusable_catalog_deposit_recorded(web, monkeypatch, tmp_path):
    """A deposit that downloads but loses every variant to QC/harmonization
    is recorded as not working, and the /catalog page shows it."""
    from webapp import gwascat
    client, root = web
    _FAKE_URLS["GCST000003"] = str(
        _fake_catalog_file(root / "caches" / "demo" / "trait1.tsv",
                           tmp_path / "cat3.h.tsv.gz", empty_beta=True))
    _FAKE_URLS["GCST000004"] = str(
        _fake_catalog_file(root / "caches" / "demo" / "trait2.tsv",
                           tmp_path / "cat4.h.tsv.gz"))
    monkeypatch.setattr("webapp.gwascat.resolve", _fake_resolve)
    response = client.post(
        "/jobs",
        data={"gcst1": "GCST000003", "gcst2": "GCST000004",
              "cache_key": "TEST"},
        follow_redirects=False)
    assert response.status_code == 303, response.text
    job_id = response.headers["location"].rsplit("/", 1)[-1]
    job = _wait_for_terminal(root, job_id)
    assert job["status"] == "failed"
    assert "GCST000003" in job["error"]
    assert "all GWAS variants were removed" in job["error"]
    # The supervisor records the blamed accession on reap; the other trait's
    # deposit is not blamed for trait 1's failure.
    deadline = time.time() + 15
    while time.time() < deadline:
        if "GCST000003" in gwascat.accession_registry(root):
            break
        time.sleep(1)
    entry = gwascat.accession_registry(root)["GCST000003"]
    assert entry["works"] is False
    assert "all GWAS variants were removed" in entry["reason"]
    assert "GCST000004" not in gwascat.accession_registry(root)
    page = client.get("/catalog")
    assert page.status_code == 200
    assert "GCST000003" in page.text


def test_catalog_page_lists_track_record(web):
    from webapp import gwascat
    client, root = web
    gwascat.record_accession(root, "GCST000001", True,
                             trait="Independent fixture", kept=5)
    page = client.get("/catalog")
    assert page.status_code == 200
    assert "Compatible inputs" in page.text
    assert "GCST000001" in page.text


def test_catalog_summary_counts_cover_merged_tables(web, monkeypatch,
                                                    tmp_path):
    """The stat strip counts the merged tables, not the canonical rows only."""
    import hashlib
    from webapp import gwascat
    client, root = web
    table = tmp_path / "real_gwas_pipeline_catalog.csv"
    table.write_text(
        "trait,status,note,source,accession,pmid,n_eff_value,n_final,total_s,"
        "driver_peak_gb,n_chains,n_chains_kept,infer_burn_in,infer_num_iter,"
        "ncores,ldpred3_version,cohort_id\n"
        "good_trait,ok,,paper,GCST1,1,1000,500,2.0,0.1,8,8,200,200,8,v,c\n")
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    (tmp_path / "real_gwas_pipeline_catalog.manifest.json").write_text(
        json.dumps({"table_sha256": digest, "row_source": {},
                    "settings": {}, "known_limits": []}))
    (tmp_path / "gwas_catalog_traits.toml").write_text("")
    monkeypatch.setenv("BIPRED_WEB_LDPRED3_BENCHMARKS", str(tmp_path))
    gwascat.record_accession(root, "GCST000777", True,
                             trait="Server-only fixture", kept=5)
    page = client.get("/catalog")
    assert page.status_code == 200
    strip = {label: n for n, label in re.findall(
        r"<strong>(\d+)</strong><span>([^<]+)</span>", page.text)}
    # One canonical completed row, plus at least the server-only fixture: the
    # strip must agree with the tables it summarizes (previously canonical
    # counts only, which disagreed as soon as this server observed anything).
    works_n = re.search(r"Compatible inputs \((\d+)\)", page.text).group(1)
    failed_n = re.search(r"Rejected or failed inputs \((\d+)\)",
                         page.text).group(1)
    assert strip["completed inputs"] == works_n
    assert int(works_n) >= 2
    assert strip["rejected or failed inputs"] == failed_n
    assert int(strip["also observed by this server"]) >= 1


def test_failed_resolution_recorded(web, monkeypatch):
    from webapp import gwascat
    client, root = web

    def boom(accession, _root):
        raise ValueError(f"{accession}: no harmonised summary-statistics "
                         "file in the GWAS Catalog")
    monkeypatch.setattr("webapp.gwascat.resolve", boom)
    response = client.post(
        "/jobs", data={"gcst1": "GCST000009", "cache_key": "TEST"})
    assert response.status_code == 400
    entry = gwascat.accession_registry(root)["GCST000009"]
    assert entry["works"] is False
    assert "no harmonised" in entry["reason"]


def test_transient_resolution_error_not_recorded(web, monkeypatch):
    from webapp import gwascat
    client, root = web

    def boom(accession, _root):
        raise ValueError(f"{accession}: catalog lookup failed (timed out)")
    monkeypatch.setattr("webapp.gwascat.resolve", boom)
    response = client.post(
        "/jobs", data={"gcst1": "GCST000010", "cache_key": "TEST"})
    assert response.status_code == 400
    assert "GCST000010" not in gwascat.accession_registry(root)


def test_results_page_tolerates_pre_qc_report_munge(web):
    """Jobs from before the per-step QC report still render, with a note."""
    from webapp import jobs
    client, root = web
    job = jobs.create_job(root, options={"weights": False},
                          labels={"trait1": "A", "trait2": "B"})
    jobs.update_job(root, job["id"], status="done", finished=time.time())
    result = {
        "joint": {"rg": 0.1, "h2": [0.1, 0.2], "p": 0.01, "pi": None,
                  "noise_scale": None, "retained_iterations": 1,
                  "stopped_early": False,
                  "mixer": {"polygenicity": [0.1, 0.2], "n_causal": [10, 20],
                            "n_shared": 5, "frac_shared": 0.5,
                            "rho_beta": 0.3, "rg_from_overlap": 0.1}},
        "ldsc": {"error": "not run"},
        "munge": {"n_cache": 100, "n_joint": 90, "n_kept": 80,
                  "n_screen_drop": 0},
        "weights": [],
        "provenance": {"bipred": "x", "ldpred3": "y", "numpy": "z",
                       "cache_key": "demo", "cache_sha256": "abc",
                       "seed": 0, "burn_in": 1, "num_iter": 1,
                       "screen": False, "stages": {}},
    }
    (root / "jobs" / job["id"] / "result.json").write_text(json.dumps(result))
    page = client.get(f"/jobs/{job['id']}/results")
    assert page.status_code == 200
    assert "predates the per-step QC report" in page.text
    assert "<svg" in page.text              # MiXeR overlap needs no QC counts
    assert "Model-implied MiXeR overlap" in page.text
    assert "Variant processing" not in page.text


def _full_munge_result(n_usable=True, stage_schema=1):
    munge = {"n_cache": 1000, "n_joint": 700, "n_kept": 690,
             "n_screen_drop": 10,
             "trait1": {"qc": {"n_input": 900, "n_kept": 850},
                        "harmonize": {"n_matched": 800, "n_sumstats": 900}},
             "trait2": {"qc": {"n_input": 950, "n_kept": 940},
                        "harmonize": {"n_matched": 910, "n_sumstats": 950}}}
    if n_usable:
        munge["trait1"]["n_usable"] = 780
        munge["trait2"]["n_usable"] = 900
    if stage_schema >= 3:
        munge["trait1"]["ld_consistency_screen"] = {
            "n_input": 780, "n_kept": 750, "n_dropped": 30}
        munge["trait2"]["ld_consistency_screen"] = {
            "n_input": 900, "n_kept": 840, "n_dropped": 60}
        munge["trait1"]["n_usable"] = 750
        munge["trait2"]["n_usable"] = 840
    provenance = {"bipred": "x", "ldpred3": "y", "numpy": "z",
                  "cache_key": "demo", "cache_sha256": "abc",
                  "seed": 0, "burn_in": 1, "num_iter": 1,
                  "screen": True, "stages": {}}
    if stage_schema > 1:
        provenance["stage_schema"] = stage_schema
    return {
        "joint": {"rg": 0.1, "h2": [0.1, 0.2], "p": 0.01, "pi": None,
                  "noise_scale": None, "retained_iterations": 1,
                  "stopped_early": False,
                  "mixer": {"polygenicity": [0.1, 0.2], "n_causal": [10, 20],
                            "n_shared": 5, "frac_shared": 0.5,
                            "rho_beta": 0.3, "rg_from_overlap": 0.1}},
        "ldsc": {"error": "not run"},
        "munge": munge,
        "weights": [],
        "provenance": provenance,
    }


def test_figure_data_from_munge():
    from webapp.app import _figure_data
    figs = _figure_data(_full_munge_result())
    assert figs["joint"] == 690 and figs["screen_drop"] == 10
    assert figs["current_screen"] is False
    assert figs["traits"]["trait1"]["on_reference"] == 780
    assert figs["traits"]["trait1"]["after_screen"] is None
    assert figs["traits"]["trait1"]["usable"] == 780
    assert figs["traits"]["trait1"]["only"] == 90
    assert figs["traits"]["trait2"]["only"] == 210
    # Jobs from before n_usable existed fall back to harmonization matches.
    figs = _figure_data(_full_munge_result(n_usable=False))
    assert figs["traits"]["trait1"]["usable"] == 800
    current = _figure_data(_full_munge_result(stage_schema=3))
    assert current["current_screen"] is True
    assert current["screen_drop"] == 0
    assert current["traits"]["trait1"]["on_reference"] == 780
    assert current["traits"]["trait1"]["after_screen"] == 750
    assert current["traits"]["trait1"]["usable"] == 750
    assert current["traits"]["trait1"]["only"] == 60
    # Jobs predating the per-trait report draw nothing.
    assert _figure_data({"munge": {"n_kept": 5}}) is None


def test_mixer_figure_data_uses_model_implied_overlap():
    from webapp.app import _mixer_figure_data

    partial = _mixer_figure_data(_full_munge_result())
    assert partial == {
        "trait1_total": 10.0, "trait2_total": 20.0, "shared": 5.0,
        "trait1_only": 5.0, "trait2_only": 15.0,
        "fraction_trait1": 0.5, "fraction_trait2": 0.25,
    }
    full = _full_munge_result()
    full["joint"]["mixer"].update(
        {"n_causal": [5, 5], "n_shared": 5})
    assert _mixer_figure_data(full)["trait1_only"] == 0
    empty = _full_munge_result()
    empty["joint"]["mixer"].update(
        {"n_causal": [0, 0], "n_shared": 0})
    assert _mixer_figure_data(empty) is None
    impossible = _full_munge_result()
    impossible["joint"]["mixer"].update(
        {"n_causal": [4, 5], "n_shared": 6})
    assert _mixer_figure_data(impossible) is None


def test_results_page_draws_mixer_overlap_and_qc_attrition(web):
    """MiXeR overlap and observed QC attrition remain distinct figures."""
    from webapp import jobs
    client, root = web
    job = jobs.create_job(root, options={"weights": False},
                          labels={"trait1": "A", "trait2": "B"})
    jobs.update_job(root, job["id"], status="done", finished=time.time())
    result = _full_munge_result(stage_schema=3)
    (root / "jobs" / job["id"] / "munge.json").write_text(
        json.dumps(result["munge"]))
    (root / "jobs" / job["id"] / "result.json").write_text(json.dumps(result))
    page = client.get(f"/jobs/{job['id']}/results")
    assert page.status_code == 200
    assert "<svg" in page.text
    assert "Model-implied MiXeR overlap" in page.text
    assert 'data-overlap-mode="partial"' in page.text
    assert ">5<" in page.text              # MiXeR shared count in the Venn
    assert "Figure 2." in page.text and "Variant-count attrition" in page.text
    assert "After mandatory screen" in page.text
    assert ">750<" in page.text
    assert "joint pre-screen drop" in page.text
    assert "Dropped by the LD-consistency screen" not in page.text

    status_page = client.get(f"/jobs/{job['id']}")
    assert status_page.status_code == 200
    assert "Mandatory trait screens" in status_page.text
    assert 'id="m-screen-input-trait1">780<' in status_page.text
    assert 'id="m-screen-kept-trait1">750<' in status_page.text
    assert 'id="m-screen-dropped-trait1">30<' in status_page.text
    assert "Observed in both GWAS" not in status_page.text


@pytest.mark.parametrize(
    ("totals", "shared", "mode"),
    [([10, 20], 0, "empty"), ([10, 20], 5, "partial"),
     ([5, 10], 5, "complete"), ([5, 5], 5, "identical")],
)
def test_results_page_mixer_overlap_geometries(web, totals, shared, mode):
    from webapp import jobs

    client, root = web
    job = jobs.create_job(
        root, options={"weights": False},
        labels={"trait1": "A", "trait2": "B"})
    jobs.update_job(root, job["id"], status="done", finished=time.time())
    result = _full_munge_result(stage_schema=3)
    result["joint"]["mixer"].update(
        {"n_causal": totals, "n_shared": shared})
    (root / "jobs" / job["id"] / "result.json").write_text(
        json.dumps(result))

    page = client.get(f"/jobs/{job['id']}/results")

    assert page.status_code == 200
    assert f'data-overlap-mode="{mode}"' in page.text
    if mode == "complete":
        assert "shared (smaller set)" in page.text
        assert "trait 2 only" in page.text
    if mode == "identical":
        assert "identical modeled sets" in page.text
        assert "The two modeled sets are identical and fully shared" in page.text


def test_legacy_results_keep_fitted_panel_ldsc_description(web):
    """Old result files must not acquire the new full-reference semantics."""
    from webapp import jobs

    client, root = web
    job = jobs.create_job(root, options={"weights": False},
                          labels={"trait1": "A", "trait2": "B"})
    job.pop("stage_schema")
    jobs.save_job(root, job)
    jobs.update_job(root, job["id"], status="done", finished=time.time())
    result = _full_munge_result()
    result["ldsc"] = {
        "rg": 0.1, "rg_se": 0.02, "gcov": 0.01,
        "gcov_intercept": 0.0, "h2": [0.1, 0.2],
    }
    # Historical result files predate both stage_schema and the reference-LD
    # score metadata introduced with the reusable sidecar.
    assert "stage_schema" not in result["provenance"]
    assert "m_snps" not in result["ldsc"]
    (root / "jobs" / job["id"] / "result.json").write_text(
        json.dumps(result))

    page = client.get(f"/jobs/{job['id']}/results")
    assert page.status_code == 200
    assert "Unfiltered fitted-panel LDSC-style" in page.text
    assert "computed LD scores and M from the intersected" in page.text
    assert "Full-reference-score LDSC-style" not in page.text
    assert "Dropped by the LD-consistency screen" in page.text
    assert "After mandatory screen" not in page.text


def test_current_result_with_ldsc_error_keeps_full_reference_description(web):
    """A failed current diagnostic still reflects the reference-score path."""
    from webapp import jobs

    client, root = web
    job = jobs.create_job(root, options={"weights": False},
                          labels={"trait1": "A", "trait2": "B"})
    jobs.update_job(root, job["id"], status="done", finished=time.time())
    result = _full_munge_result()
    assert result["ldsc"] == {"error": "not run"}
    (root / "jobs" / job["id"] / "result.json").write_text(
        json.dumps(result))

    page = client.get(f"/jobs/{job['id']}/results")
    assert page.status_code == 200
    assert "Full-reference-score LDSC-style" in page.text
    assert "mandatory DENTIST-inspired screen" in page.text
    assert "Unfiltered fitted-panel LDSC-style" not in page.text


def test_results_page_reports_structured_divergence_statistics(web):
    """The numerical reason for quarantine is visible, not buried in prose."""
    from webapp import jobs

    client, root = web
    job = jobs.create_job(
        root, options={"weights": True},
        labels={"trait1": "A", "trait2": "B"})
    jobs.update_job(root, job["id"], status="done", finished=time.time())
    result = _full_munge_result()
    result["munge"].update(
        n_cache=12_000, n_joint=10_500, n_kept=10_000,
        n_screen_drop=500)
    result["munge"].pop("trait1")
    result["munge"].pop("trait2")
    trait = {
        "sum_beta_squared": 2.5,
        "raw_genetic_variance": 0.2,
        "effect_energy_ratio": 12.5,
        "max_abs_posterior_mean": 0.3,
        "per_causal_effect_sd": 0.02,
        "max_effect_slab_sd": 15.0,
        "trace_first_quarter_mean": 0.2,
        "trace_last_quarter_mean": 0.28,
        "trace_last_over_first": 1.4,
        "trace_drift_fold": 1.4,
        "trace_direction": "rising",
        "flags": {"effect_energy_ratio": True,
                  "max_effect_slab_sd": False, "trace_drift": True},
    }
    result["diagnostics"] = {
        "valid_for_interpretation": False,
        "critical": True,
        "warnings": [],
        "weights_withheld": True,
        "divergence": {
            "evaluated": True, "flagged": True,
            "variant_count": 10_000, "largest_block_variants": 15_976,
            "trace_iterations": 200, "trace_evaluated": True,
            "thresholds": {
                "minimum_variants": 1_000,
                "minimum_trace_iterations": 40,
                "effect_energy_ratio": 10.0,
                "max_effect_slab_sd": 25.0,
                "trace_drift_fold": 1.25,
            },
            "traits": {"trait1": trait, "trait2": {
                **trait,
                "sum_beta_squared": 0.1,
                "raw_genetic_variance": -0.2,
                "effect_energy_ratio": None,
                "trace_last_quarter_mean": 0.2,
                "trace_last_over_first": 1.0,
                "trace_drift_fold": 1.0,
                "trace_direction": "flat",
                "flags": {"nonpositive_genetic_variance": True,
                          "effect_energy_ratio": False,
                          "max_effect_slab_sd": False,
                          "trace_drift": False},
            }},
        },
    }
    (root / "jobs" / job["id"] / "result.json").write_text(
        json.dumps(result, allow_nan=False))

    page = client.get(f"/jobs/{job['id']}/results")
    assert page.status_code == 200
    assert "Fit-stability diagnostics" in page.text
    assert "Divergence threshold crossed" in page.text
    assert "12.5 (flagged)" in page.text
    assert "-0.2 (flagged)" in page.text
    assert "≤ 0" in page.text
    assert "1.4×, rising (flagged)" in page.text
    assert "15,976 variants" in page.text
    assert "not R-hat or effective" in page.text


def test_results_page_quarantines_null_diagnostics_without_500(web):
    """Non-finite fit outputs serialized as JSON null remain renderable."""
    from webapp import jobs

    client, root = web
    job = jobs.create_job(root, options={"weights": False},
                          labels={"trait1": "A", "trait2": "B"})
    jobs.update_job(root, job["id"], status="done", finished=time.time())
    result = _full_munge_result()
    result["joint"].update(rg=None, h2=[None, None], p=None)
    result["joint"]["mixer"] = {
        "polygenicity": [None, None], "n_causal": [None, None],
        "n_shared": None, "frac_shared": None,
        "rho_beta": None, "rg_from_overlap": None,
    }
    trait = {
        "sum_beta_squared": None,
        "raw_genetic_variance": None,
        "effect_energy_ratio": None,
        "max_abs_posterior_mean": None,
        "per_causal_effect_sd": None,
        "max_effect_slab_sd": None,
        "trace_first_quarter_mean": None,
        "trace_last_quarter_mean": None,
        "trace_last_over_first": None,
        "trace_drift_fold": None,
        "trace_direction": None,
        # A ratio may overflow, trigger a flag, and then become JSON null.
        "flags": {"effect_energy_ratio": True,
                  "max_effect_slab_sd": False, "trace_drift": False},
    }
    result["diagnostics"] = {
        "valid_for_interpretation": False,
        "critical": True,
        "warnings": [],
        "weights_withheld": True,
        "divergence": {
            "evaluated": True, "flagged": True,
            "variant_count": 10_000, "largest_block_variants": None,
            "trace_iterations": 200, "trace_evaluated": True,
            "thresholds": {
                "minimum_variants": 1_000,
                "minimum_trace_iterations": 40,
                "effect_energy_ratio": 10.0,
                "max_effect_slab_sd": 25.0,
                "trace_drift_fold": 1.25,
            },
            "traits": {"trait1": trait, "trait2": {
                **trait,
                "flags": {"effect_energy_ratio": False,
                          "max_effect_slab_sd": False,
                          "trace_drift": False},
            }},
        },
    }
    (root / "jobs" / job["id"] / "result.json").write_text(
        json.dumps(result, allow_nan=False))

    page = client.get(f"/jobs/{job['id']}/results")
    assert page.status_code == 200
    assert "Fit-stability diagnostics" in page.text
    assert page.text.count("—") >= 20
    assert "not recorded" in page.text


def test_results_page_needs_no_python_stage_helper(web):
    """A template reload must not depend on reloading the Python process."""
    from webapp import jobs
    from webapp.app import TEMPLATES

    client, root = web
    job = jobs.create_job(
        root, options={"weights": False},
        labels={"trait1": "A", "trait2": "B"})
    jobs.update_job(root, job["id"], status="done", finished=time.time())
    result = _full_munge_result()
    result["provenance"]["stages"] = {"download": 1.0, "fit": 2.0}
    (root / "jobs" / job["id"] / "result.json").write_text(
        json.dumps(result))

    assert "stage_label" not in TEMPLATES.env.globals
    page = client.get(f"/jobs/{job['id']}/results")
    assert page.status_code == 200
    assert "Get Catalog data 1.0s" in page.text


def test_accession_registry_semantics(tmp_path):
    from webapp import gwascat
    gwascat.record_accession(tmp_path, "GCST1", False, reason="nope")
    gwascat.record_accession(tmp_path, "GCST1", True, kept=5)     # upgrade
    gwascat.record_accession(tmp_path, "GCST1", False, reason="later")
    entry = gwascat.accession_registry(tmp_path)["GCST1"]
    assert entry["works"] is True               # never downgraded by a failure
    assert entry["kept"] == 5
    assert gwascat.worth_recording("GCST1: no such study in the GWAS Catalog")
    assert gwascat.worth_recording(
        "trait1: all GWAS variants were removed by sumstats QC")
    assert not gwascat.worth_recording("catalog lookup failed (timeout)")


def test_stream_filter_schema_variants(tmp_path):
    """OR-only and z-only deposits become betas; non-reference ids drop out."""
    from webapp import gwascat
    src = tmp_path / "mixed.h.tsv.gz"
    with gzip.open(src, "wt", newline="") as fh:
        fh.write("hm_rsid\thm_effect_allele\thm_other_allele\tstandard_error"
                 "\thm_odds_ratio\thm_z_score\n")
        fh.write("rs1\tA\tG\t0.02\t1.5\t\n")       # OR only -> log(1.5)
        fh.write("rs2\tC\tT\t0.01\t\t2.5\n")        # z only  -> 2.5 * 0.01
        fh.write("rsX\tA\tG\t0.02\t1.5\t\n")        # not in reference
    dest = tmp_path / "out.tsv.gz"
    info = gwascat.stream_filter(str(src), {"rs1", "rs2"}, dest)
    assert (info["seen"], info["kept"]) == (3, 2)
    assert info["effect_from"] == "log(odds_ratio)"
    with gzip.open(dest, "rt") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert abs(float(rows[0]["beta"]) - 0.405465) < 1e-5
    assert abs(float(rows[1]["beta"]) - 0.025) < 1e-9
    with pytest.raises(ValueError, match="no variants overlap"):
        gwascat.stream_filter(str(src), {"rs999"}, tmp_path / "none.tsv.gz")


def test_stream_filter_reports_progress(tmp_path):
    """on_bytes fires with a rising compressed-byte count ending at EOF."""
    from webapp import gwascat
    src = tmp_path / "prog.h.tsv.gz"
    with gzip.open(src, "wt", newline="") as fh:
        fh.write("rsid\teffect_allele\tother_allele\tbeta\tstandard_error\n")
        for i in range(2000):
            fh.write(f"rs{i}\tA\tG\t0.01\t0.02\n")
    seen_calls = []
    dest = tmp_path / "out.tsv.gz"
    gwascat.stream_filter(str(src), {"rs0"}, dest,
                          on_bytes=seen_calls.append)
    assert seen_calls == sorted(seen_calls) and seen_calls
    assert seen_calls[-1] == src.stat().st_size
    # The runner throttles; the callback itself need not.
    assert len(seen_calls) >= 1


def test_stream_filter_reports_usable_per_variant_n(tmp_path):
    from webapp import gwascat
    src = tmp_path / "with-n.h.tsv.gz"
    with gzip.open(src, "wt", newline="") as fh:
        fh.write("rsid\teffect_allele\tother_allele\tbeta\tstandard_error\tn\n")
        fh.write("rs1\tA\tG\t0.01\t0.02\t100000\n")
        fh.write("rs2\tC\tT\t0.02\t0.03\t120000\n")
    info = gwascat.stream_filter(
        str(src), {"rs1", "rs2"}, tmp_path / "with-n.out.tsv.gz")
    assert info["has_per_variant_n"] is True
    assert info["per_variant_n_usable_frac"] == 1.0
    assert len(info["sha256"]) == 64


# --- shared download store --------------------------------------------------

def _store_source(tmp_path, name="src.h.tsv.gz", ids=("rs1", "rs2", "rs3")):
    """A small harmonised-layout file standing in for a catalog deposit."""
    src = tmp_path / name
    with gzip.open(src, "wt", newline="") as fh:
        fh.write("rsid\tchromosome\tbase_pair_location\teffect_allele\t"
                 "other_allele\tbeta\tstandard_error\tp_value\tn\n")
        for i, rsid in enumerate(ids):
            fh.write(f"{rsid}\t1\t{100 + i}\tA\tG\t0.0{i + 1}"
                     f"\t0.02\t0.5\t100000\n")
    return src


def _rows(path):
    with gzip.open(path, "rt") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _never_called():
    raise AssertionError("coverage was computed for a reusable stored copy")


def _legacy_catalog_job(root, remote, *, accession, fingerprint,
                        input_sha256=None):
    """One completed pre-store job with enough provenance for migration."""
    from webapp import gwascat
    job_id = "legacy-completed-job"
    job_dir = root / "jobs" / job_id
    job_dir.mkdir(parents=True)
    name = "trait1.gcst.tsv.gz"
    local = job_dir / name
    info = gwascat.stream_filter(str(remote), {"rs1"}, local)
    compressed_sha = hashlib.sha256(local.read_bytes()).hexdigest()
    catalog = {
        "accession": accession, "url": str(remote),
        "remote_bytes": remote.stat().st_size,
        "kept": info["kept"], "seen": info["seen"],
        "effect_from": info["effect_from"], "sha256": info["sha256"],
        "has_per_variant_n": info["has_per_variant_n"],
        "per_variant_n_usable_frac": info["per_variant_n_usable_frac"],
    }
    job = {
        "id": job_id, "status": "done", "files": {"sumstats1": name},
        "options": {"gcst1": accession, "catalog1": catalog},
    }
    result_catalog = {key: catalog[key] for key in (
        "accession", "kept", "seen", "effect_from", "sha256",
        "has_per_variant_n", "per_variant_n_usable_frac")}
    result = {"provenance": {
        "bipred": "0.3.10.dev0", "cache_sha256": fingerprint,
        "catalog": {"trait1": result_catalog},
        "inputs": {"trait1": {
            "filename": name,
            "sha256": input_sha256 or compressed_sha,
        }},
    }}
    (job_dir / "job.json").write_text(json.dumps(job))
    (job_dir / "result.json").write_text(json.dumps(result))
    return info


def _race_store_fetches(monkeypatch, gwascat, first, second):
    """Start ``second`` only after ``first`` owns the shared-store lock."""
    original_stream = gwascat.stream_filter
    original_acquire = gwascat._StoreLock.acquire
    real_sleep = time.sleep
    owner_started = threading.Event()
    release_owner = threading.Event()
    waiter_contended = threading.Event()
    roles = threading.local()
    calls = []
    calls_lock = threading.Lock()

    def gated_stream(url, *args, **kwargs):
        with calls_lock:
            calls.append(url)
            first_call = len(calls) == 1
        if first_call:
            owner_started.set()
            assert release_owner.wait(5.0), "store owner was never released"
        return original_stream(url, *args, **kwargs)

    def observed_acquire(lock):
        acquired = original_acquire(lock)
        if getattr(roles, "name", None) == "waiter" and not acquired:
            waiter_contended.set()
        return acquired

    def run(role, fetch):
        roles.name = role
        return fetch()

    monkeypatch.setattr(gwascat, "stream_filter", gated_stream)
    monkeypatch.setattr(gwascat._StoreLock, "acquire", observed_acquire)
    # Production polls every two seconds. Preserve the polling path while
    # shortening its clock-free wait for this unit test.
    monkeypatch.setattr(gwascat.time, "sleep", lambda _: real_sleep(0.01))
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(run, "owner", first)
        assert owner_started.wait(5.0), "first request did not start its fetch"
        waiter = pool.submit(run, "waiter", second)
        try:
            assert waiter_contended.wait(5.0), "second request did not contend"
        finally:
            release_owner.set()
        results = owner.result(timeout=5.0), waiter.result(timeout=5.0)
    return results, calls


def test_legacy_job_file_is_adopted_without_network_fetch(tmp_path):
    from webapp import gwascat
    root = tmp_path / "data"
    remote = _store_source(tmp_path)
    accession, fingerprint = "GCST000017", "hash-a"
    original = _legacy_catalog_job(
        root, remote, accession=accession, fingerprint=fingerprint)
    assert gwascat.adopt_legacy_job_file(
        root, accession=accession, url=str(remote),
        fingerprint=fingerprint, keep_ids={"rs1"}, cache_label="ref A",
        remote_bytes=remote.stat().st_size)
    remote_bytes = remote.stat().st_size
    remote.unlink()                    # any fallback download would now fail
    reused = gwascat.fetch_filtered(
        str(remote), tmp_path / "adopted.tsv.gz", accession=accession,
        root=root, keep_ids={"rs1"}, fingerprint=fingerprint,
        remote_bytes=remote_bytes, coverage=_never_called)
    assert reused["reused"] is True
    assert reused["store_origin"] == "legacy job"
    assert reused["normalised_sha256"]
    assert reused["kept"] == original["kept"] == 1


def test_legacy_job_with_wrong_recorded_hash_is_rejected(tmp_path):
    from webapp import gwascat
    root = tmp_path / "data"
    remote = _store_source(tmp_path)
    accession = "GCST000018"
    _legacy_catalog_job(
        root, remote, accession=accession, fingerprint="hash-a",
        input_sha256="0" * 64)
    assert not gwascat.adopt_legacy_job_file(
        root, accession=accession, url=str(remote), fingerprint="hash-a",
        keep_ids={"rs1"}, cache_label="ref A",
        remote_bytes=remote.stat().st_size)
    assert not (root / "catalog" / f"{accession}.tsv.gz").exists()


def test_store_serves_a_second_reference_without_downloading(tmp_path):
    """The point of the store: switching LD reference must not re-fetch."""
    from webapp import gwascat
    src = _store_source(tmp_path)
    root = tmp_path / "data"
    first_ids, second_ids = {"rs1", "rs2"}, {"rs2", "rs3"}
    first = gwascat.fetch_filtered(
        str(src), tmp_path / "first.tsv.gz", accession="GCST000001",
        root=root, keep_ids=first_ids, fingerprint="hash-a",
        coverage=lambda: (first_ids | second_ids,
                          {"hash-a": "ref A", "hash-b": "ref B"}))
    assert first["reused"] is False
    assert (first["seen"], first["kept"]) == (3, 2)
    # Deleting the deposit is the assertion: the same URL is passed again,
    # so any attempt to read it would raise instead of quietly re-fetching.
    src.unlink()
    second = gwascat.fetch_filtered(
        str(src), tmp_path / "second.tsv.gz", accession="GCST000001",
        root=root, keep_ids=second_ids, fingerprint="hash-b",
        coverage=_never_called)
    assert second["reused"] is True
    assert [row["rsid"] for row in _rows(tmp_path / "second.tsv.gz")] == \
        ["rs2", "rs3"]
    # Provenance still describes the remote file, not the stored copy.
    assert second["seen"] == 3 and second["kept"] == 2
    assert second["sha256"] == first["sha256"]
    assert second["effect_from"] == first["effect_from"] == "beta"


def test_store_rebuilds_a_truncated_gzip(tmp_path, monkeypatch):
    """A broken shared copy is discarded and downloaded exactly once more."""
    from webapp import gwascat
    src = _store_source(tmp_path)
    root = tmp_path / "data"
    original = gwascat.stream_filter
    calls = []

    def counted(url, *args, **kwargs):
        calls.append(url)
        return original(url, *args, **kwargs)

    monkeypatch.setattr(gwascat, "stream_filter", counted)
    kwargs = dict(
        accession="GCST000009", root=root, keep_ids={"rs1"},
        fingerprint="hash-a",
        coverage=lambda: ({"rs1"}, {"hash-a": "ref A"}))
    gwascat.fetch_filtered(str(src), tmp_path / "first.gz", **kwargs)
    stored = root / "catalog" / "GCST000009.tsv.gz"
    stored.write_bytes(stored.read_bytes()[:-8])       # remove gzip trailer

    rebuilt = gwascat.fetch_filtered(
        str(src), tmp_path / "rebuilt.gz", **kwargs)

    assert calls == [str(src), str(src)]
    assert rebuilt["reused"] is False
    assert [row["rsid"] for row in _rows(tmp_path / "rebuilt.gz")] == ["rs1"]
    assert not (tmp_path / "rebuilt.gz.part").exists()


def test_concurrent_same_store_fetch_downloads_once(tmp_path, monkeypatch):
    """The waiter reports reuse when the owner supplies its requested build."""
    from webapp import gwascat
    src = _store_source(tmp_path)
    root = tmp_path / "data"

    def fetch(name):
        return lambda: gwascat.fetch_filtered(
            str(src), tmp_path / name, accession="GCST000010", root=root,
            keep_ids={"rs1"}, fingerprint="hash-a",
            coverage=lambda: ({"rs1"}, {"hash-a": "ref A"}))

    (owner, waiter), calls = _race_store_fetches(
        monkeypatch, gwascat, fetch("owner.gz"), fetch("waiter.gz"))

    assert calls == [str(src)]
    assert owner["reused"] is False
    assert waiter["reused"] is True
    assert _rows(tmp_path / "owner.gz") == _rows(tmp_path / "waiter.gz")


def test_store_process_lock_cannot_be_stolen_or_unlinked(tmp_path):
    from webapp import gwascat
    path = tmp_path / "store.lock"
    first = gwascat._StoreLock(path)
    assert first.acquire()
    second = gwascat._StoreLock(path)
    assert not second.acquire()
    first.release()
    assert path.exists()
    assert second.acquire()
    second.release()
    assert path.exists()


def test_concurrent_different_urls_never_reuses_old_content(
        tmp_path, monkeypatch):
    """A waiter for a re-deposited URL downloads after the first owner exits."""
    from webapp import gwascat
    first_src = _store_source(tmp_path, "first.h.tsv.gz", ids=("rs1",))
    second_src = _store_source(tmp_path, "second.h.tsv.gz", ids=("rs9",))
    root = tmp_path / "data"

    def fetch(src, rsid, name):
        return lambda: gwascat.fetch_filtered(
            str(src), tmp_path / name, accession="GCST000011", root=root,
            keep_ids={rsid}, fingerprint="hash-a",
            coverage=lambda: ({rsid}, {"hash-a": "ref A"}))

    (owner, waiter), calls = _race_store_fetches(
        monkeypatch, gwascat,
        fetch(first_src, "rs1", "owner.gz"),
        fetch(second_src, "rs9", "waiter.gz"))

    assert calls == [str(first_src), str(second_src)]
    assert owner["reused"] is False
    assert waiter["reused"] is False
    assert [row["rsid"] for row in _rows(tmp_path / "owner.gz")] == ["rs1"]
    assert [row["rsid"] for row in _rows(tmp_path / "waiter.gz")] == ["rs9"]
    meta = json.loads((root / "catalog" / "GCST000011.json").read_text())
    assert meta["url"] == str(second_src)


def test_normalised_hash_ignores_gzip_mtime(tmp_path, monkeypatch):
    """Prepared-cache identity hashes content, not gzip container bytes."""
    from webapp import gwascat
    src = _store_source(tmp_path)
    root = tmp_path / "data"
    kwargs = dict(
        accession="GCST000012", root=root, keep_ids={"rs1", "rs3"},
        fingerprint="hash-a",
        coverage=lambda: ({"rs1", "rs3"}, {"hash-a": "ref A"}))

    prepared = tmp_path / "prepared.gz"
    monkeypatch.setattr(gwascat.time, "time", lambda: 1_000_000_000.0)
    first = gwascat.fetch_filtered(str(src), prepared, **kwargs)
    first_bytes = prepared.read_bytes()
    monkeypatch.setattr(gwascat.time, "time", lambda: 1_000_000_001.0)
    second = gwascat.fetch_filtered(str(src), prepared, **kwargs)
    second_bytes = prepared.read_bytes()

    assert first_bytes != second_bytes
    with gzip.open(prepared, "rb") as fh:
        expected = hashlib.sha256(fh.read()).hexdigest()
    assert first["normalised_sha256"] == second["normalised_sha256"] == expected
    assert first["sha256"] == second["sha256"]  # remote provenance unchanged


def test_store_output_matches_a_direct_download(tmp_path):
    """Filtering the stored copy reproduces stream_filter's own output."""
    from webapp import gwascat
    src = _store_source(tmp_path)
    keep = {"rs2", "rs3"}
    direct = tmp_path / "direct.tsv.gz"
    straight = gwascat.stream_filter(str(src), keep, direct)
    viastore = tmp_path / "viastore.tsv.gz"
    stored = gwascat.fetch_filtered(
        str(src), viastore, accession="GCST000002", root=tmp_path / "data",
        keep_ids=keep, fingerprint="hash-a",
        coverage=lambda: ({"rs1", "rs2", "rs3"}, {"hash-a": "ref A"}))
    with gzip.open(direct, "rt") as a, gzip.open(viastore, "rt") as b:
        assert a.read() == b.read()
    for field in ("seen", "kept", "sha256", "schema", "effect_from",
                  "has_n", "has_per_variant_n", "per_variant_n_usable_frac"):
        assert stored[field] == straight[field], field


def test_store_rebuilds_for_an_uncovered_reference(tmp_path):
    """A reference the stored union never covered is fetched, not faked."""
    from webapp import gwascat
    src = _store_source(tmp_path)
    root = tmp_path / "data"
    gwascat.fetch_filtered(
        str(src), tmp_path / "a.tsv.gz", accession="GCST000003", root=root,
        keep_ids={"rs1"}, fingerprint="hash-a",
        coverage=lambda: ({"rs1"}, {"hash-a": "ref A"}))
    again = gwascat.fetch_filtered(
        str(src), tmp_path / "c.tsv.gz", accession="GCST000003", root=root,
        keep_ids={"rs3"}, fingerprint="hash-c",
        coverage=lambda: ({"rs1", "rs3"}, {"hash-c": "ref C"}))
    assert again["reused"] is False
    assert [row["rsid"] for row in _rows(tmp_path / "c.tsv.gz")] == ["rs3"]
    build = json.loads((root / "catalog" / "GCST000003.json").read_text())
    assert set(build["covers"]) == {"hash-c"}


def test_store_ignores_a_copy_of_a_different_url(tmp_path):
    """Re-deposition under a new path invalidates the stored copy."""
    from webapp import gwascat
    root = tmp_path / "data"
    first = _store_source(tmp_path, "first.h.tsv.gz")
    gwascat.fetch_filtered(
        str(first), tmp_path / "a.tsv.gz", accession="GCST000004", root=root,
        keep_ids={"rs1"}, fingerprint="hash-a",
        coverage=lambda: ({"rs1"}, {"hash-a": "ref A"}))
    second = _store_source(tmp_path, "second.h.tsv.gz", ids=("rs1", "rs9"))
    redeposited = gwascat.fetch_filtered(
        str(second), tmp_path / "b.tsv.gz", accession="GCST000004", root=root,
        keep_ids={"rs1"}, fingerprint="hash-a",
        coverage=lambda: ({"rs1", "rs9"}, {"hash-a": "ref A"}))
    assert redeposited["reused"] is False
    assert redeposited["seen"] == 2


def test_store_rebuilds_when_same_url_reports_new_size(tmp_path):
    """An in-place Catalog replacement is not hidden by an unchanged URL."""
    from webapp import gwascat
    root = tmp_path / "data"
    src = _store_source(tmp_path, "same-url.h.tsv.gz", ids=("rs1",))
    first_size = src.stat().st_size
    gwascat.fetch_filtered(
        str(src), tmp_path / "first.tsv.gz", accession="GCST000014",
        root=root, keep_ids={"rs1"}, fingerprint="hash-a",
        remote_bytes=first_size,
        coverage=lambda: ({"rs1"}, {"hash-a": "ref A"}))
    _store_source(tmp_path, "same-url.h.tsv.gz", ids=("rs1", "rs2", "rs9"))
    second_size = src.stat().st_size
    assert second_size != first_size
    second = gwascat.fetch_filtered(
        str(src), tmp_path / "second.tsv.gz", accession="GCST000014",
        root=root, keep_ids={"rs1", "rs2"}, fingerprint="hash-a",
        remote_bytes=second_size,
        coverage=lambda: ({"rs1", "rs2"}, {"hash-a": "ref A"}))
    assert second["reused"] is False
    assert second["seen"] == 3


def test_store_rebuilds_when_metadata_is_structurally_invalid(tmp_path):
    """Plausible JSON with missing scientific provenance is never a hit."""
    from webapp import gwascat
    root = tmp_path / "data"
    src = _store_source(tmp_path)
    kwargs = dict(
        accession="GCST000015", root=root, keep_ids={"rs1"},
        fingerprint="hash-a",
        coverage=lambda: ({"rs1"}, {"hash-a": "ref A"}))
    gwascat.fetch_filtered(str(src), tmp_path / "first.tsv.gz", **kwargs)
    meta = root / "catalog" / "GCST000015.json"
    broken = json.loads(meta.read_text())
    broken.pop("effect_from")
    meta.write_text(json.dumps(broken))
    rebuilt = gwascat.fetch_filtered(
        str(src), tmp_path / "second.tsv.gz", **kwargs)
    assert rebuilt["reused"] is False
    assert json.loads(meta.read_text())["effect_from"] == "beta"


def test_store_refuses_a_union_that_misses_the_reference(tmp_path):
    from webapp import gwascat
    src = _store_source(tmp_path)
    with pytest.raises(ValueError, match="does not cover"):
        gwascat.fetch_filtered(
            str(src), tmp_path / "x.tsv.gz", accession="GCST000005",
            root=tmp_path / "data", keep_ids={"rs1", "rs2"},
            fingerprint="hash-a",
            coverage=lambda: ({"rs1"}, {"hash-a": "ref A"}))


def test_store_eviction_spares_recent_copies(tmp_path):
    """The byte budget evicts least-recently-used copies, not live ones."""
    from webapp import gwascat
    root = tmp_path / "data"
    src = _store_source(tmp_path)
    for accession in ("GCST000006", "GCST000007"):
        gwascat.fetch_filtered(
            str(src), tmp_path / f"{accession}.tsv.gz", accession=accession,
            root=root, keep_ids={"rs1"}, fingerprint="hash-a",
            coverage=lambda: ({"rs1"}, {"hash-a": "ref A"}))
    meta = root / "catalog" / "GCST000006.json"
    build = json.loads(meta.read_text())
    build["last_used"] = time.time() - 10 * 86400
    meta.write_text(json.dumps(build))
    assert gwascat.purge_store(root, 1e-9) == ["GCST000006"]
    assert not (root / "catalog" / "GCST000006.tsv.gz").exists()
    assert (root / "catalog" / "GCST000007.tsv.gz").exists()   # used just now
    # A budget of zero disables the cap but still clears abandoned partials.
    stale = root / "catalog" / "GCST000008.tsv.gz.part"
    stale.write_text("half a download")
    os.utime(stale, (0, 0))
    assert gwascat.purge_store(root, 0) == []
    assert not stale.exists()


def test_store_eviction_never_removes_a_locked_generation(tmp_path):
    from webapp import gwascat
    root = tmp_path / "data"
    src = _store_source(tmp_path)
    gwascat.fetch_filtered(
        str(src), tmp_path / "job.tsv.gz", accession="GCST000016",
        root=root, keep_ids={"rs1"}, fingerprint="hash-a",
        coverage=lambda: ({"rs1"}, {"hash-a": "ref A"}))
    meta = root / "catalog" / "GCST000016.json"
    build = json.loads(meta.read_text())
    build["last_used"] = time.time() - 10 * 86400
    meta.write_text(json.dumps(build))
    lock = gwascat._StoreLock(str(meta) + ".lock")
    assert lock.acquire()
    try:
        assert gwascat.purge_store(root, 1e-9) == []
        assert (root / "catalog" / "GCST000016.tsv.gz").exists()
    finally:
        lock.release()
    assert gwascat.purge_store(root, 1e-9) == ["GCST000016"]


def test_coverage_covers_every_registered_reference(tmp_path, monkeypatch):
    """The runner's union must span all registered references, not just the
    one this job asked for — that is what makes a re-run against a different
    reference free."""
    from webapp import caches, demo, gwascat, runner
    first, second = tmp_path / "ref-a", tmp_path / "ref-b"
    demo.build_demo(first, m=400, n_samples=200, seed=3)
    demo.build_demo(second, m=300, n_samples=200, seed=4)
    a, b = first / "demo.ld.npz", second / "demo.ld.npz"
    monkeypatch.setenv("BIPRED_WEB_CACHES", f"A={a};B={b}")
    keep_ids = gwascat.cache_ids(a)
    coverage = runner._coverage_thunk(tmp_path, a, "A", keep_ids,
                                      caches.sha256_cached(a))
    union, covers = coverage()
    # Any real cache this host happens to have registered is covered too; the
    # invariant is that no registered reference is left out.
    assert covers[caches.sha256_cached(a)] == "A"
    assert covers[caches.sha256_cached(b)] == "B"
    assert keep_ids <= union and gwascat.cache_ids(b) <= union
    assert coverage()[0] is union                             # computed once


def test_coverage_skips_an_extra_reference_that_changes_during_id_read(
        tmp_path, monkeypatch):
    """A moving optional reference contributes neither IDs nor a hash claim."""
    from webapp import caches, gwascat, runner

    selected = tmp_path / "selected.ld.npz"
    changing = tmp_path / "changing.ld.npz"
    monkeypatch.setattr(caches, "real_registry", lambda root=None: [
        {"key": "selected", "path": str(selected)},
        {"key": "changing", "path": str(changing)},
    ])
    changed_hashes = iter(("before", "after"))
    hash_calls = []

    def generation(path, root=None):
        path = Path(path)
        hash_calls.append(path)
        if path == selected:
            return "selected-hash"
        return next(changed_hashes)

    monkeypatch.setattr(caches, "sha256_cached", generation)
    monkeypatch.setattr(
        gwascat, "cache_ids",
        lambda path: {"rs-from-changing-reference"})

    coverage = runner._coverage_thunk(
        tmp_path, selected, "selected", {"rs-selected"}, "selected-hash")
    union, covers = coverage()

    assert union == {"rs-selected"}
    assert covers == {"selected-hash": "selected"}
    assert hash_calls == [selected, changing, changing]


def test_runner_acquires_two_catalog_traits_concurrently(tmp_path, monkeypatch):
    """Both independent Catalog transfers start before either one completes."""
    from webapp import gwascat, jobs, runner

    root = tmp_path / "data"
    job = jobs.create_job(
        root,
        options={
            "cache_key": "TEST",
            "gcst1": "GCST000021", "gcst2": "GCST000022",
            "catalog1": {
                "accession": "GCST000021", "url": "fake://trait-1",
                "remote_bytes": 1,
            },
            "catalog2": {
                "accession": "GCST000022", "url": "fake://trait-2",
                "remote_bytes": 1,
            },
        },
        labels={"trait1": "First", "trait2": "Second"})
    barrier = threading.Barrier(2, timeout=5)
    calls = []
    call_lock = threading.Lock()

    monkeypatch.setattr(
        runner.caches, "cache_path", lambda cache_key, data_root: tmp_path / "ld")
    monkeypatch.setattr(
        runner.caches, "sha256_cached", lambda path, data_root=None: "ld-hash")
    monkeypatch.setattr(gwascat, "cache_ids", lambda path: {"rs1"})
    monkeypatch.setattr(
        gwascat, "stored_copy_available", lambda *args, **kwargs: True)

    def meet_then_stop(url, dest, **kwargs):
        with call_lock:
            calls.append((url, threading.get_ident()))
        barrier.wait()
        raise RuntimeError("intentional stop after concurrent acquisition")

    monkeypatch.setattr(gwascat, "fetch_filtered", meet_then_stop)
    with pytest.raises(ValueError, match="summary-statistics preparation failed"):
        runner.run(jobs.job_dir(root, job["id"]), job)

    assert {url for url, _ in calls} == {"fake://trait-1", "fake://trait-2"}
    assert len({thread_id for _, thread_id in calls}) == 2


def test_catalog_acquisition_rejects_an_ld_generation_change(
        tmp_path, monkeypatch):
    """Catalog rows cannot be filtered against IDs from a replaced LD file."""
    from webapp import gwascat, jobs, runner

    root = tmp_path / "data"
    job = jobs.create_job(
        root,
        options={
            "cache_key": "TEST", "gcst1": "GCST000023", "gcst2": "",
            "catalog1": {
                "accession": "GCST000023", "url": "fake://trait-1",
                "remote_bytes": 1,
            },
        },
        labels={"trait1": "First", "trait2": "Second"})
    hashes = iter(("a" * 64, "b" * 64))
    fetched = []
    monkeypatch.setattr(
        runner.caches, "cache_path",
        lambda cache_key, data_root: tmp_path / "ld")
    monkeypatch.setattr(
        runner.caches, "sha256_cached",
        lambda path, data_root=None: next(hashes))
    monkeypatch.setattr(gwascat, "cache_ids", lambda path: {"rs1"})
    monkeypatch.setattr(
        gwascat, "fetch_filtered",
        lambda *args, **kwargs: fetched.append((args, kwargs)))

    with pytest.raises(ValueError, match=(
            "changed while its variant IDs were being read")):
        runner.run(jobs.job_dir(root, job["id"]), job)

    assert fetched == []


@pytest.mark.slow
@pytest.mark.integration
def test_rerunning_an_analysis_reuses_the_download(web, monkeypatch, tmp_path):
    """A second job for the same accessions must not re-fetch the deposits."""
    client, root = web
    monkeypatch.setattr("webapp.gwascat.resolve", _fake_resolve)
    for trait, accession in ((1, "GCST000010"), (2, "GCST000011")):
        _FAKE_URLS[accession] = str(_fake_catalog_file(
            root / "caches" / "demo" / f"trait{trait}.tsv",
            tmp_path / f"rerun{trait}.h.tsv.gz"))

    def submit(*, burn_in, num_iter, cross_corr):
        response = client.post(
            "/jobs", data={"gcst1": "GCST000010", "gcst2": "GCST000011",
                           "cache_key": "TEST", "burn_in": str(burn_in),
                           "num_iter": str(num_iter),
                           "cross_corr": str(cross_corr),
                           # Make this preparation distinct from earlier
                           # module-scoped Catalog jobs with the same rows.
                           "n_eff1": "123457", "n_eff2": "123458"},
            follow_redirects=False)
        assert response.status_code == 303, response.text
        return _wait_for_terminal(
            root, response.headers["location"].rsplit("/", 1)[-1])

    first = submit(burn_in=50, num_iter=40, cross_corr=0)
    assert first["status"] == "done", first.get("error")
    assert set(first["stages"]) >= {
        "acquire", "prepare", "screen", "pair", "fit"}
    assert not {"download", "validate", "harmonize"} & set(first["stages"])
    assert first["options"]["catalog1"]["source"] == "download"
    assert first["options"]["catalog1"]["prepared_key"]
    assert first["stage_details"]["acquire"]["traits"] == {
        "trait1": "downloaded", "trait2": "downloaded"}
    assert first["stage_details"]["prepare"]["published"] is False
    assert all(
        not info["prepared_reused"]
        for info in first["stage_details"]["screen"]["traits"].values())
    assert first["stage_details"]["pair"]["rerun"] is True
    assert (root / "catalog" / "GCST000010.tsv.gz").exists()
    # Deleting the deposits is the assertion: a re-fetch would now fail.
    for accession in ("GCST000010", "GCST000011"):
        os.unlink(_FAKE_URLS[accession])
    second = submit(burn_in=60, num_iter=45, cross_corr=0.1)
    assert second["status"] == "done", second.get("error")
    assert second["stage_details"]["acquire"]["traits"] == {
        "trait1": "reused stored data", "trait2": "reused stored data"}
    assert all(
        info["prepared_reused"]
        for info in second["stage_details"]["screen"]["traits"].values())
    assert second["stage_details"]["pair"]["rerun"] is True
    assert second["stage_details"]["pair"]["n_kept"] > 0
    for trait in (1, 2):
        assert second["options"][f"catalog{trait}"]["source"] == "stored copy"
        assert second["options"][f"catalog{trait}"]["prepared_reused"] is True
        assert second["options"][f"catalog{trait}"]["prepared_key"] == \
            first["options"][f"catalog{trait}"]["prepared_key"]
        # Reuse must not change what the fit sees.
        assert second["options"][f"catalog{trait}"]["kept"] == \
            first["options"][f"catalog{trait}"]["kept"]
        assert second["options"][f"catalog{trait}"]["sha256"] == \
            first["options"][f"catalog{trait}"]["sha256"]
    with open(root / "jobs" / second["id"] / "result.json") as fh:
        provenance = json.load(fh)["provenance"]
    assert provenance["burn_in"] == 60 and provenance["num_iter"] == 45
    assert provenance["cross_corr"] == 0.1
    assert provenance["catalog"]["trait1"]["prepared_reused"] is True
    assert "LD-consistency-screened" in \
        provenance["catalog"]["trait1"]["prepared_scope"]


def test_progress_sink_records_library_events(tmp_path):
    """Library progress events land in job.json for the status endpoint."""
    from webapp import jobs, runner
    job = jobs.create_job(tmp_path, options={}, labels={})
    stage = runner._Stages(tmp_path, job)
    stage.start("fit")
    runner._progress_sink(stage)({"step": "fit", "done": 3, "total": 10,
                                  "unit": "sweep", "phase": "burn-in"})
    saved = jobs.load_job(tmp_path, job["id"])
    assert saved["progress"]["step"] == "fit"
    assert saved["progress"]["done"] == 3
    assert saved["progress"]["phase"] == "burn-in"


def test_acquire_progress_retains_both_trait_counters(tmp_path):
    """One concurrent transfer update must not erase its counterpart."""
    from webapp import jobs, runner

    job = jobs.create_job(tmp_path, options={}, labels={})
    stage = runner._Stages(tmp_path, job)
    stage.start("acquire")
    stage.progress(trait=1, accession="GCST1", bytes=10, total=100,
                   mb_s=1.0)
    stage.progress(trait=2, accession="GCST2", bytes=40, total=200,
                   mb_s=2.0)

    progress = jobs.load_job(tmp_path, job["id"])["progress"]
    assert set(progress["traits"]) == {"trait1", "trait2"}
    assert progress["traits"]["trait1"]["bytes"] == 10
    assert progress["traits"]["trait1"]["total"] == 100
    assert progress["traits"]["trait2"]["bytes"] == 40
    assert progress["traits"]["trait2"]["total"] == 200


def test_prepare_screen_progress_retains_both_trait_states(tmp_path):
    from webapp import jobs, runner

    (tmp_path / "jobs").mkdir()
    job = jobs.create_job(
        tmp_path, options={}, labels={"trait1": "A", "trait2": "B"},
        status="running")
    stage = runner._Stages(tmp_path, job)
    stage.start("prepare")
    stage.progress(trait=2, phase="prepare", step="harmonize trait 2")
    stage.activate("screen")
    stage.progress(trait=1, phase="screen", step="screen trait 1",
                   done=3, total=10, unit="block")

    saved = jobs.load_job(tmp_path, job["id"])
    progress = saved["progress"]
    assert list(sorted(progress["traits"])) == ["trait1", "trait2"]
    assert progress["traits"]["trait1"]["phase"] == "screen"
    assert progress["traits"]["trait2"]["phase"] == "prepare"
    assert saved["active_stages"] == ["prepare", "screen"]


def test_acquire_completion_waits_for_progress_serialization(
        tmp_path, monkeypatch):
    """A completed transfer cannot mutate a job while its peer saves progress."""
    from webapp import jobs, runner

    job = jobs.create_job(
        tmp_path,
        options={
            "catalog1": {"accession": "GCST1"},
            "n_eff1": 1000.0, "n_cases1": None, "n_controls1": None,
        },
        labels={})
    stage = runner._Stages(tmp_path, job)
    stage.start("acquire")

    mutation_started = threading.Event()

    class ObservedFiles(dict):
        def __setitem__(self, key, value):
            mutation_started.set()
            return super().__setitem__(key, value)

    job["files"] = ObservedFiles(job["files"])
    progress_saving = threading.Event()
    release_progress = threading.Event()
    errors = []

    def save(_root, current):
        if threading.current_thread().name == "progress-writer":
            progress_saving.set()
            if not release_progress.wait(5.0):
                raise AssertionError("progress save was not released")
        json.dumps(current)

    monkeypatch.setattr(runner.jobs, "save_job", save)

    def capture(action):
        try:
            action()
        except BaseException as exc:  # surface worker failures in this thread
            errors.append(exc)

    progress = threading.Thread(
        target=lambda: capture(lambda: stage.progress(
            trait=2, accession="GCST2", bytes=10, total=100)),
        name="progress-writer")
    progress.start()
    assert progress_saving.wait(5.0)

    outcomes = {}
    info = {
        "kept": 10, "seen": 20, "effect_from": "beta",
        "sha256": "a" * 64, "normalised_sha256": "b" * 64,
        "has_per_variant_n": True, "per_variant_n_usable_frac": 1.0,
    }
    completion = threading.Thread(
        target=lambda: capture(lambda: stage.finish_acquire_trait(
            1, job["options"]["catalog1"], tmp_path / "trait1.tsv.gz",
            info, "download", outcomes)))
    completion.start()
    assert not mutation_started.wait(0.1)

    release_progress.set()
    progress.join(5.0)
    completion.join(5.0)
    assert not progress.is_alive() and not completion.is_alive()
    assert errors == []
    assert mutation_started.is_set()
    assert job["files"]["sumstats1"] == "trait1.tsv.gz"
    assert job["options"]["n_eff1"] is None
    assert job["stage_details"]["acquire"]["traits"] == {
        "trait1": "downloaded"}


def test_progress_sink_survives_a_failed_status_write():
    """A fit must not die because its status write did; the library lets the
    callback's exception propagate, so the swallowing belongs here."""
    from webapp import runner

    class Failing:
        def progress(self, **fields):
            raise OSError("no space left on device")

    runner._progress_sink(Failing())({"step": "fit", "done": 1, "total": 2})


def test_accession_format_checked(tmp_path):
    from webapp import gwascat
    with pytest.raises(ValueError, match="GCST"):
        gwascat.resolve("not-an-accession", tmp_path)


def test_purge_removes_only_expired_jobs(tmp_path):
    from webapp import jobs
    root = tmp_path
    stale = jobs.create_job(root, options={}, labels={})
    jobs.update_job(root, stale["id"], status="done",
                    finished=time.time() - 8 * 86400)
    fresh = jobs.create_job(root, options={}, labels={})
    jobs.update_job(root, fresh["id"], status="done", finished=time.time())
    running = jobs.create_job(root, options={}, labels={})
    jobs.update_job(root, running["id"], status="running",
                    started=time.time() - 30 * 86400)
    stale_staging = jobs.create_job(
        root, options={}, labels={}, status="staging")
    jobs.update_job(
        root, stale_staging["id"], created=time.time() - 8 * 86400)
    private = jobs.job_dir(root, stale_staging["id"]) / "partial.tsv"
    private.write_bytes(b"private partial contents")
    fresh_staging = jobs.create_job(
        root, options={}, labels={}, status="staging")
    old_queued = jobs.create_job(root, options={}, labels={}, status="queued")
    jobs.update_job(root, old_queued["id"],
                    created=time.time() - 30 * 86400)
    removed = jobs.purge_jobs(root, ttl_days=7)
    assert stale["id"] in removed
    assert stale_staging["id"] in removed
    assert not jobs.job_dir(root, stale_staging["id"]).exists()
    assert fresh["id"] not in removed
    assert fresh_staging["id"] not in removed
    assert old_queued["id"] not in removed      # preserve accepted work
    assert running["id"] not in removed        # never purge a live job


def test_staging_jobs_are_not_launched(tmp_path):
    from webapp import jobs
    from webapp.app import _sweep_once, create_app
    app = create_app()
    app.state.root = tmp_path
    (tmp_path / "jobs").mkdir()
    app.state.config = {"concurrency": 1, "ttl_days": 7}
    app.state.procs = {}
    app.state.last_purge = time.time()
    staged = jobs.create_job(
        tmp_path, options={}, labels={}, status="staging")
    _sweep_once(app)
    assert jobs.load_job(tmp_path, staged["id"])["status"] == "staging"
    assert app.state.procs == {}


def test_supervisor_stops_owned_fit_at_runtime_limit(tmp_path):
    from webapp import jobs
    from webapp.app import _sweep_once, create_app

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    app = create_app()
    app.state.root = tmp_path
    (tmp_path / "jobs").mkdir()
    app.state.config["job_timeout_s"] = 1
    app.state.last_purge = time.time()
    job = jobs.create_job(tmp_path, options={}, labels={}, status="running")
    jobs.update_job(tmp_path, job["id"], started=time.time() - 10)
    proc = FakeProcess()
    app.state.procs = {job["id"]: proc}
    app.state.orphans = {}

    _sweep_once(app)

    saved = jobs.load_job(tmp_path, job["id"])
    assert proc.terminated is True
    assert saved["status"] == "failed"
    assert "runtime limit" in saved["error"]


def test_webapp_defaults_to_one_memory_heavy_fit(monkeypatch):
    from webapp.app import _config

    monkeypatch.delenv("BIPRED_WEB_CONCURRENCY", raising=False)
    assert _config()["concurrency"] == 1


def test_webapp_rejects_invalid_resource_limits(monkeypatch):
    from webapp.app import _config

    for name, value in (("BIPRED_WEB_CONCURRENCY", "0"),
                        ("BIPRED_WEB_QUEUE_MAX", "nope"),
                        ("BIPRED_WEB_JOB_TIMEOUT_HOURS", "inf"),
                        ("BIPRED_WEB_MAX_ROWS", "0"),
                        ("BIPRED_WEB_MAX_EXPANDED_GB", "nan")):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            _config()
        monkeypatch.delenv(name)
    for name in ("BIPRED_WEB_STORE_GB", "BIPRED_WEB_PREPARED_GB"):
        monkeypatch.setenv(name, "0")
        assert _config()["store_gb" if name.endswith("STORE_GB")
                         else "prepared_gb"] == 0
        monkeypatch.delenv(name)


def test_body_limit_rejects_before_route_parsing():
    from fastapi import FastAPI
    from webapp.app import _BodyLimitMiddleware

    small = FastAPI()

    @small.post("/upload")
    async def upload():
        return {"unexpected": True}

    small.add_middleware(_BodyLimitMiddleware, limit=8)
    with TestClient(small) as client:
        response = client.post(
            "/upload", content=b"0123456789",
            headers={"content-type": "application/octet-stream"})
    assert response.status_code == 413


def test_runner_input_work_guard_bounds_rows_and_gzip_expansion(tmp_path):
    from webapp.runner import _input_work_guard

    plain = tmp_path / "sumstats.tsv"
    plain.write_text("id\tbeta\nrs1\t.1\nrs2\t.2\n", encoding="utf-8")
    report = _input_work_guard(
        plain, max_rows=2, max_expanded_bytes=1024)
    assert report["rows"] == 2
    with pytest.raises(ValueError, match="1-row limit"):
        _input_work_guard(
            plain, max_rows=1, max_expanded_bytes=1024)

    compressed = tmp_path / "sumstats.tsv.gz"
    with gzip.open(compressed, "wb") as fh:
        fh.write(b"id\tbeta\n" + b"rs1\t0.1\n" * 20)
    with pytest.raises(ValueError, match="decompressed input"):
        _input_work_guard(
            compressed, max_rows=100, max_expanded_bytes=32)


def test_demo_is_post_only_same_origin_and_queue_bounded(web):
    from webapp import jobs

    client, root = web
    assert client.get("/demo", follow_redirects=False).status_code == 405
    refused = client.post(
        "/demo", headers={"origin": "https://attacker.invalid"},
        follow_redirects=False)
    assert refused.status_code == 403

    old_limit = client.app.state.config["queue_max"]
    client.app.state.config["queue_max"] = 1
    blocker = jobs.create_job(
        root, options={}, labels={}, status="running")
    try:
        full = client.post("/demo", follow_redirects=False)
        assert full.status_code == 503
        assert full.headers["retry-after"] == "30"
    finally:
        jobs.update_job(root, blocker["id"], status="failed",
                        finished=time.time(), error="test cleanup")
        client.app.state.config["queue_max"] = old_limit


def test_restart_reconciles_interrupted_but_not_queued_jobs(tmp_path):
    from webapp import jobs
    (tmp_path / "jobs").mkdir()
    staging = jobs.create_job(
        tmp_path, options={}, labels={}, status="staging")
    partial = jobs.job_dir(tmp_path, staging["id"]) / "partial.tsv"
    partial.write_bytes(b"private partial contents")
    running = jobs.create_job(tmp_path, options={}, labels={}, status="running")
    queued = jobs.create_job(tmp_path, options={}, labels={}, status="queued")
    recovered = jobs.recover_interrupted_jobs(tmp_path)
    assert set(recovered) == {staging["id"], running["id"]}
    assert not jobs.job_dir(tmp_path, staging["id"]).exists()
    assert jobs.load_job(tmp_path, running["id"])["status"] == "failed"
    assert "server restarted" in jobs.load_job(tmp_path, running["id"])["error"]
    assert jobs.load_job(tmp_path, queued["id"])["status"] == "queued"


def test_restart_preserves_live_runner_and_counts_its_slot(tmp_path):
    from webapp import jobs
    from webapp.app import _sweep_once, create_app

    (tmp_path / "jobs").mkdir()
    running = jobs.create_job(
        tmp_path, options={}, labels={}, status="running")
    token = jobs.new_runner_token()
    running = jobs.update_job(
        tmp_path, running["id"], pid=os.getpid(),
        pid_identity=jobs.process_identity(os.getpid()), runner_token=token)
    lease = jobs.ProcessFileLock(
        jobs.runner_lease_path(tmp_path, running["id"], token))
    assert lease.acquire()
    queued = jobs.create_job(
        tmp_path, options={}, labels={}, status="queued")
    try:
        recovered, live = jobs.recover_interrupted_jobs(
            tmp_path, preserve_live=True)
        assert recovered == [] and live == {running["id"]: os.getpid()}

        app = create_app()
        app.state.root = tmp_path
        app.state.config["concurrency"] = 1
        app.state.procs = {}
        app.state.orphans = live
        app.state.last_purge = time.time()
        _sweep_once(app)
    finally:
        lease.release()

    assert jobs.load_job(tmp_path, queued["id"])["status"] == "queued"
    assert app.state.procs == {}


def test_parse_columns():
    from webapp.app import parse_columns
    assert parse_columns("id=RSID, ea=ALLELE1") == {"id": "RSID",
                                                    "ea": "ALLELE1"}
    assert parse_columns("") == {}
    with pytest.raises(ValueError):
        parse_columns("idRSID")


def test_numeric_validation_and_optional_n():
    from webapp.app import (_bounded_int, _cross_corr,
                            _sample_size_options)
    assert _sample_size_options({}, 1) == (None, None, None)
    assert _cross_corr("-0.25") == -0.25
    with pytest.raises(ValueError, match="strictly between"):
        _cross_corr("1")
    with pytest.raises(ValueError, match="between"):
        _bounded_int("100001", 200, "iterations", 1, 100000)


def test_fit_warning_classifier_quarantines_do_not_interpret():
    from webapp.runner import _warnings_are_critical
    assert _warnings_are_critical([
        {"message": "Do not interpret h2 or rg from this fit."}])
    assert not _warnings_are_critical([
        {"message": "posterior weights use an HWE SD approximation"}])
    # New fits do not depend on warning prose for this safety decision.
    assert _warnings_are_critical([], {"flagged": True})
    assert not _warnings_are_critical([], {"flagged": False})


def test_runner_attributes_per_trait_failure_to_accession():
    from webapp.runner import _attribute_to_catalog
    job = {"options": {"catalog1": {"accession": "GCST000001"}},
           "labels": {"trait1": "Height", "trait2": "BMI"}}
    out = _attribute_to_catalog(
        ValueError("trait1: all GWAS variants were removed by sumstats QC"),
        job)
    assert "GCST000001" in str(out) and "Height" in str(out)
    # A joint failure blames nobody.
    assert _attribute_to_catalog(
        ValueError("the two GWAS share fewer than two cache variants"),
        job) is None
    # The blamed trait is an upload — not the server's to record.
    assert _attribute_to_catalog(ValueError("trait2: boom"), job) is None


def test_normal_submit_rejects_demo_cache(web):
    client, root = web
    with _demo_upload(root, 1)[1] as f1, _demo_upload(root, 2)[1] as f2:
        response = client.post(
            "/jobs",
            files={"sumstats1": ("t1.tsv", f1, "text/tsv"),
                   "sumstats2": ("t2.tsv", f2, "text/tsv")},
            data={"n_eff1": "1000", "n_eff2": "1000",
                  "cache_key": "demo"})
    assert response.status_code == 400
    assert "synthetic demo LD reference" in response.text


def test_ldpred3_catalog_evidence_loader(tmp_path, monkeypatch):
    import hashlib
    from webapp import catalog_evidence
    table = tmp_path / "real_gwas_pipeline_catalog.csv"
    table.write_text(
        "trait,status,note,source,accession,pmid,n_eff_value,n_final,total_s,"
        "driver_peak_gb,n_chains,n_chains_kept,infer_burn_in,infer_num_iter,"
        "ncores,ldpred3_version,cohort_id\n"
        "good_trait,ok,,paper,GCST1,1,1000,500,2.0,0.1,8,8,200,200,8,v,c\n"
        "failed_trait,failed,all GWAS variants were removed by sumstats QC,"
        "paper,GCST2,2,,,,,,,,,,,c\n")
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    (tmp_path / "real_gwas_pipeline_catalog.manifest.json").write_text(
        json.dumps({"table_sha256": digest,
                    "row_source": {"good_trait": "catalog-current-profile"},
                    "settings": {}, "known_limits": []}))
    (tmp_path / "gwas_catalog_traits.toml").write_text(
        "[traits.bad]\naccession='GCST3'\ntrait='Bad deposit'\n"
        "usable=false\nunusable_reason='hits-only deposit'\n")
    monkeypatch.setenv("BIPRED_WEB_LDPRED3_BENCHMARKS", str(tmp_path))
    evidence = catalog_evidence.load()
    assert evidence["table_hash_verified"] is True
    assert evidence["counts"] == {"good": 1, "bad": 2,
                                  "preflight_bad": 1, "failed_fit": 1}
    assert evidence["good"][0]["profile"] == "current"


def test_registry_demo_last_and_default(tmp_path):
    from webapp import caches
    reg = caches.registry(tmp_path)
    assert reg[-1]["key"] == "demo"            # real caches precede the demo
    assert caches.default_key(tmp_path) == reg[0]["key"]
    assert caches.cache_path(reg[0]["key"], tmp_path).name.endswith(".npz")


def test_cache_hash_sidecar_invalidates_after_mutation(tmp_path):
    from webapp import caches
    path = tmp_path / "cache.npz"
    path.write_bytes(b"before")
    first = caches.sha256_cached(path)
    time.sleep(0.002)                 # distinct nanosecond mtime on coarse FSes
    path.write_bytes(b"after")
    second = caches.sha256_cached(path)
    assert first != second


def test_cache_hash_sidecar_rejects_atomic_replacement_with_old_mtime(tmp_path):
    from webapp import caches
    path = tmp_path / "cache.npz"
    path.write_bytes(b"before")
    first = caches.sha256_cached(path)
    old = path.stat()
    replacement = tmp_path / "replacement.npz"
    replacement.write_bytes(b"after!")       # same byte count as "before"
    os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
    os.replace(replacement, path)
    second = caches.sha256_cached(path)
    assert first != second


def _small_ld_cache(path):
    from ldpred3 import save_ld_blocks

    ids = np.array(["rs1", "rs2", "rs3"])
    corr = np.array([[1.0, 0.5, 0.0],
                     [0.5, 1.0, 0.25],
                     [0.0, 0.25, 1.0]], dtype=np.float32)
    save_ld_blocks(
        path, [(corr, np.arange(3))], ids,
        counted_allele=np.array(["A", "A", "A"]),
        other_allele=np.array(["G", "G", "G"]),
        chrom=np.array(["1", "1", "1"]), pos=np.arange(1, 4),
        reference_af=np.full(3, 0.3), n_ref=500)
    return ids


def test_full_reference_ld_score_sidecar_roundtrip_and_effective_rank(tmp_path):
    from webapp import caches

    cache = tmp_path / "ld.npz"
    _small_ld_cache(cache)
    scores = np.array([1.0, 2.0, 3.0])
    sidecar = caches.write_ld_score_sidecar(
        cache, scores, source="unit-test reference",
        source_sha256="a" * 64, algorithm="unit-test-v1")
    assert sidecar == caches.ld_score_sidecar_path(cache)

    panel = caches.load_ld_score_panel(cache)
    np.testing.assert_array_equal(panel.scores, scores)
    assert panel.scores.flags.writeable is False
    assert panel.m_snps == 3
    assert panel.score_sum == 6.0 and panel.score_mean == 2.0
    assert panel.effective_rank == 1.5
    assert panel.algorithm == "unit-test-v1"
    assert panel.correction == "none"


def test_ld_score_sidecar_rejects_another_cache_generation(tmp_path):
    from webapp import caches

    cache = tmp_path / "ld.npz"
    _small_ld_cache(cache)
    caches.write_ld_score_sidecar(
        cache, np.ones(3), source="unit-test reference")
    # Replacing the cache changes its exact generation even though M is equal.
    replacement = tmp_path / "replacement.npz"
    _small_ld_cache(replacement)
    with np.load(replacement, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["ids"] = np.array(["rs3", "rs2", "rs1"])
    np.savez_compressed(replacement, **payload)
    os.replace(replacement, cache)
    with pytest.raises(ValueError, match="another LD-cache generation"):
        caches.load_ld_score_panel(cache)


def test_build_ld_scores_from_map_requires_exact_cache_order(tmp_path):
    from webapp import caches

    cache = tmp_path / "ld.npz"
    ids = _small_ld_cache(cache)
    source = tmp_path / "map.csv"
    source.write_text(
        "rsid,ld\n" + "".join(
            f"{variant},{score}\n"
            for variant, score in zip(ids, (1.25, 1.5, 1.25))))
    caches.build_ld_score_sidecar_from_map(cache, source)
    panel = caches.load_ld_score_panel(cache)
    np.testing.assert_allclose(panel.scores, [1.25, 1.5, 1.25])
    assert panel.m_snps == 3

    caches.ld_score_sidecar_path(cache).unlink()
    source.write_text("rsid,ld\nrs2,1\nrs1,1\nrs3,1\n")
    with pytest.raises(ValueError, match="but the LD reference has"):
        caches.build_ld_score_sidecar_from_map(cache, source)
