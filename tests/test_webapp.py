"""End-to-end tests for the bipred web service (webapp/).

Marked slow: each job runs the full fit in a subprocess, which pays the
Numba warm-up once per process. Requires the ``web`` extra; skipped when
FastAPI/httpx are absent so the base suite needs no web dependencies.
"""

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

pytestmark = pytest.mark.slow

JOB_TIMEOUT = 240.0


@pytest.fixture(scope="module")
def web(tmp_path_factory):
    root = tmp_path_factory.mktemp("webdata")
    os.environ["BIPRED_WEB_DATA"] = str(root)
    os.environ["BIPRED_WEB_CONCURRENCY"] = "1"
    os.environ.pop("BIPRED_WEB_CACHES", None)
    from webapp import caches, demo
    demo.build_demo(caches.demo_cache_dir(root), m=1500, n_samples=600, seed=7)
    from webapp.app import create_app
    with TestClient(create_app()) as client:
        yield client, root
    del os.environ["BIPRED_WEB_DATA"]
    del os.environ["BIPRED_WEB_CONCURRENCY"]


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
    assert "Demo (synthetic" in page.text
    assert 'name="sumstats1"' in page.text
    assert 'name="n_eff1"' in page.text
    # The LD-consistency screen ships checked.
    assert re.search(r'name="screen"[^>]*\bchecked\b', page.text)


def test_index_carries_preview_assets(web):
    client, _ = web
    page = client.get("/")
    assert page.status_code == 200
    assert "BIPRED_ALIASES" in page.text      # alias map for the JS preview
    assert 'class="card"' in page.text
    for asset in ("style.css", "job.js", "preview.js"):
        assert client.get(f"/static/{asset}").status_code == 200


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
    assert "af_corr" in result["munge"]

    page = client.get(f"/jobs/{job_id}/results")
    assert page.status_code == 200
    assert "Polygenic overlap" in page.text
    assert "QC and harmonization" in page.text
    for kind in ("result", "munge", "weights1", "weights2"):
        assert client.get(f"/jobs/{job_id}/download/{kind}").status_code == 200


def test_upload_job_end_to_end(web):
    client, root = web
    with _demo_upload(root, 1)[1] as f1, _demo_upload(root, 2)[1] as f2:
        response = client.post(
            "/jobs",
            files={"sumstats1": ("t1.tsv", f1, "text/tab-separated-values"),
                   "sumstats2": ("t2.tsv", f2, "text/tab-separated-values")},
            data={"label1": "upload 1", "label2": "upload 2",
                  "n_eff1": "100000", "n_eff2": "100000",
                  "cache_key": "demo"},
            follow_redirects=False)
    assert response.status_code == 303
    job_id = response.headers["location"].rsplit("/", 1)[-1]
    job = _wait_for_terminal(root, job_id)
    assert job["status"] == "done", job.get("error")
    with open(root / "jobs" / job_id / "result.json") as fh:
        result = json.load(fh)
    assert -1.0 <= result["joint"]["rg"] <= 1.0


def test_unreadable_columns_fail_the_job(web):
    client, root = web
    response = client.post(
        "/jobs",
        files={"sumstats1": ("bad.tsv", b"foo\tbar\n1\t2\n", "text/tsv"),
               "sumstats2": _demo_upload(root, 2)},
        data={"n_eff1": "100000", "n_eff2": "100000", "cache_key": "demo"},
        follow_redirects=False)
    assert response.status_code == 303
    job_id = response.headers["location"].rsplit("/", 1)[-1]
    job = _wait_for_terminal(root, job_id)
    assert job["status"] == "failed"
    assert "column" in job["error"]


def test_missing_sample_size_rejected(web):
    client, root = web
    response = client.post(
        "/jobs",
        files={"sumstats1": _demo_upload(root, 1),
               "sumstats2": _demo_upload(root, 2)},
        data={"cache_key": "demo", "label1": "MyTrait"})
    assert response.status_code == 400
    assert "effective N" in response.text
    # The upload form is re-rendered inline with entries preserved, not the
    # bare error page.
    assert 'name="sumstats1"' in response.text
    assert 'value="MyTrait"' in response.text
    assert "re-select" in response.text
    assert "Something needs fixing" not in response.text


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

def _fake_catalog_file(src_tsv, dest):
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
                             "p_value"],
            delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            out_row = {new: row[old] for old, new in rename.items()}
            out_row["p_value"] = "0.5"
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
    response = client.post("/jobs", data={"cache_key": "demo"})
    assert response.status_code == 400
    assert "file or give a GWAS Catalog accession" in response.text


def test_submit_rejects_file_plus_accession(web):
    client, root = web
    response = client.post(
        "/jobs",
        files={"sumstats1": _demo_upload(root, 1)},
        data={"gcst1": "GCST000001", "cache_key": "demo"})
    assert response.status_code == 400
    assert "not both" in response.text


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
              "cache_key": "demo"},          # N auto-filled from "metadata"
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


def test_catalog_page_lists_track_record(web):
    client, _ = web
    page = client.get("/catalog")
    assert page.status_code == 200
    assert "Known to work" in page.text
    assert "GCST000001" in page.text      # recorded by the end-to-end test


def test_failed_resolution_recorded(web, monkeypatch):
    from webapp import gwascat
    client, root = web

    def boom(accession, _root):
        raise ValueError(f"{accession}: no harmonised summary-statistics "
                         "file in the GWAS Catalog")
    monkeypatch.setattr("webapp.gwascat.resolve", boom)
    response = client.post(
        "/jobs", data={"gcst1": "GCST000009", "cache_key": "demo"})
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
        "/jobs", data={"gcst1": "GCST000010", "cache_key": "demo"})
    assert response.status_code == 400
    assert "GCST000010" not in gwascat.accession_registry(root)


def test_accession_registry_semantics(tmp_path):
    from webapp import gwascat
    gwascat.record_accession(tmp_path, "GCST1", False, reason="nope")
    gwascat.record_accession(tmp_path, "GCST1", True, kept=5)     # upgrade
    gwascat.record_accession(tmp_path, "GCST1", False, reason="later")
    entry = gwascat.accession_registry(tmp_path)["GCST1"]
    assert entry["works"] is True               # never downgraded by a failure
    assert entry["kept"] == 5
    assert gwascat.worth_recording("GCST1: no such study in the GWAS Catalog")
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


def test_parse_columns():
    from webapp.app import parse_columns
    assert parse_columns("id=RSID, ea=ALLELE1") == {"id": "RSID",
                                                    "ea": "ALLELE1"}
    assert parse_columns("") == {}
    with pytest.raises(ValueError):
        parse_columns("idRSID")


def test_registry_demo_last_and_default(tmp_path):
    from webapp import caches
    reg = caches.registry(tmp_path)
    assert reg[-1]["key"] == "demo"            # real caches precede the demo
    assert caches.default_key(tmp_path) == reg[0]["key"]
    assert caches.cache_path(reg[0]["key"], tmp_path).name.endswith(".npz")
