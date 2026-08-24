"""Unit and end-to-end tests for the checkout-only bipred web service."""

import csv
import gzip
import json
import os
import re
import time

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


def test_index_offers_demo_and_form(web):
    client, _ = web
    page = client.get("/")
    assert page.status_code == 200
    assert "Run the synthetic demo" in page.text
    assert 'name="sumstats1"' in page.text
    assert 'name="n_eff1"' in page.text
    # The experimental screen is an opt-in sensitivity analysis.
    assert not re.search(r'name="screen"[^>]*\bchecked\b', page.text)
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


@pytest.mark.slow
@pytest.mark.integration
def test_demo_job_end_to_end(web):
    client, root = web
    redirect = client.get("/demo", follow_redirects=False)
    assert redirect.status_code == 303
    job_id = redirect.headers["location"].rsplit("/", 1)[-1]
    job = _wait_for_terminal(root, job_id)
    assert job["status"] == "done", job.get("error")

    with open(root / "jobs" / job_id / "result.json") as fh:
        result = json.load(fh)
    assert -1.0 <= result["joint"]["rg"] <= 1.0
    assert result["joint"]["h2"][0] > 0.0
    assert result["munge"]["n_kept"] > 0
    assert result["weights"] == ["weights1.tsv", "weights2.tsv"]
    # Posterior uncertainty is reported for the headline estimates.
    assert result["joint"]["rg_sd"] is None or result["joint"]["rg_sd"] >= 0
    # munge.json carries the per-step QC and harmonization logs.
    t1 = result["munge"]["trait1"]
    assert t1["qc"]["n_input"] >= t1["qc"]["n_kept"] > 0
    assert t1["harmonize"]["n_matched"] > 0
    assert t1["n_usable"] > 0
    assert "af_corr" in result["munge"]
    assert result["diagnostics"]["valid_for_interpretation"] in (True, False)
    assert result["provenance"]["compute"]["logical_cpus"] >= 1
    assert result["provenance"]["resources"]["wall_s"] > 0
    assert result["provenance"]["resources"]["peak_rss_gb"] > 0
    assert result["provenance"]["sample_size"]["trait1"]["median"] > 0

    page = client.get(f"/jobs/{job_id}/results")
    assert page.status_code == 200
    assert "polygenic overlap" in page.text.lower()
    assert "QC and harmonization" in page.text
    assert "<svg" in page.text and "Variant overlap" in page.text
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
                  "cache_key": "TEST", "screen": "1"},
            follow_redirects=False)
    assert response.status_code == 303
    job_id = response.headers["location"].rsplit("/", 1)[-1]
    job = _wait_for_terminal(root, job_id)
    assert job["status"] == "done", job.get("error")
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


@pytest.mark.slow
@pytest.mark.integration
def test_status_endpoint(web):
    client, root = web
    assert client.get("/jobs/nope/status").status_code == 404
    redirect = client.get("/demo", follow_redirects=False)
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
    assert "download" in job["stages"]
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
    assert "<svg" not in page.text         # no figures without per-trait counts


def _full_munge_result(n_usable=True):
    munge = {"n_cache": 1000, "n_joint": 700, "n_kept": 690,
             "n_screen_drop": 10,
             "trait1": {"qc": {"n_input": 900, "n_kept": 850},
                        "harmonize": {"n_matched": 800, "n_sumstats": 900}},
             "trait2": {"qc": {"n_input": 950, "n_kept": 940},
                        "harmonize": {"n_matched": 910, "n_sumstats": 950}}}
    if n_usable:
        munge["trait1"]["n_usable"] = 780
        munge["trait2"]["n_usable"] = 900
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
        "provenance": {"bipred": "x", "ldpred3": "y", "numpy": "z",
                       "cache_key": "demo", "cache_sha256": "abc",
                       "seed": 0, "burn_in": 1, "num_iter": 1,
                       "screen": True, "stages": {}},
    }


def test_figure_data_from_munge():
    from webapp.app import _figure_data
    figs = _figure_data(_full_munge_result())
    assert figs["joint"] == 690 and figs["screen_drop"] == 10
    assert figs["traits"]["trait1"] == {"input": 900, "after_qc": 850,
                                        "usable": 780, "only": 90}
    assert figs["traits"]["trait2"]["only"] == 210
    # Jobs from before n_usable existed fall back to harmonization matches.
    figs = _figure_data(_full_munge_result(n_usable=False))
    assert figs["traits"]["trait1"]["usable"] == 800
    # Jobs predating the per-trait report draw nothing.
    assert _figure_data({"munge": {"n_kept": 5}}) is None


def test_results_page_draws_variant_figures(web):
    """A full munge report renders the Venn and QC-attrition figures."""
    from webapp import jobs
    client, root = web
    job = jobs.create_job(root, options={"weights": False},
                          labels={"trait1": "A", "trait2": "B"})
    jobs.update_job(root, job["id"], status="done", finished=time.time())
    result = _full_munge_result()
    (root / "jobs" / job["id"] / "result.json").write_text(json.dumps(result))
    page = client.get(f"/jobs/{job['id']}/results")
    assert page.status_code == 200
    assert "<svg" in page.text
    assert "Variant overlap" in page.text
    assert ">690<" in page.text            # shared count inside the Venn


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


def test_accession_format_checked(tmp_path):
    from webapp import gwascat
    with pytest.raises(ValueError, match="GCST"):
        gwascat.resolve("not-an-accession", tmp_path)


def test_purge_removes_only_expired_jobs(web):
    from webapp import jobs
    _, root = web
    stale = jobs.create_job(root, options={}, labels={})
    jobs.update_job(root, stale["id"], status="done",
                    finished=time.time() - 8 * 86400)
    fresh = jobs.create_job(root, options={}, labels={})
    jobs.update_job(root, fresh["id"], status="done", finished=time.time())
    running = jobs.create_job(root, options={}, labels={})
    jobs.update_job(root, running["id"], status="running",
                    started=time.time() - 30 * 86400)
    removed = jobs.purge_jobs(root, ttl_days=7)
    assert stale["id"] in removed
    assert fresh["id"] not in removed
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


def test_restart_reconciles_interrupted_but_not_queued_jobs(tmp_path):
    from webapp import jobs
    (tmp_path / "jobs").mkdir()
    running = jobs.create_job(tmp_path, options={}, labels={}, status="running")
    queued = jobs.create_job(tmp_path, options={}, labels={}, status="queued")
    recovered = jobs.recover_interrupted_jobs(tmp_path)
    assert recovered == [running["id"]]
    assert jobs.load_job(tmp_path, running["id"])["status"] == "failed"
    assert "server restarted" in jobs.load_job(tmp_path, running["id"])["error"]
    assert jobs.load_job(tmp_path, queued["id"])["status"] == "queued"


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
