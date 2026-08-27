"""Focused contracts for evidence integrity and the browser-facing UI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIN = "aba8b55d7c8c083e4d2dd5715e995786bbf14599"


def _evidence_files(root: Path) -> Path:
    table = root / "real_gwas_pipeline_catalog.csv"
    table.write_text(
        "trait,status,accession,n_eff_value,n_final\n"
        "alpha,ok,GCST1,1000,500\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    (root / "real_gwas_pipeline_catalog.manifest.json").write_text(
        json.dumps({"table_sha256": digest, "row_source": {},
                    "settings": {}, "known_limits": []}),
        encoding="utf-8",
    )
    (root / "gwas_catalog_traits.toml").write_text(
        "[traits.alpha]\naccession='GCST1'\ntrait='Alpha'\n",
        encoding="utf-8",
    )
    return table


def test_catalog_evidence_hash_mismatch_fails_closed(tmp_path, monkeypatch):
    from webapp import catalog_evidence

    table = _evidence_files(tmp_path)
    table.write_text(table.read_text(encoding="utf-8") +
                     "injected,ok,GCST2,2000,600\n", encoding="utf-8")
    monkeypatch.setenv("BIPRED_WEB_LDPRED3_BENCHMARKS", str(tmp_path))

    evidence = catalog_evidence.load()

    assert evidence["available"] is False
    assert evidence["trusted"] is False
    assert evidence["table_hash_verified"] is False
    assert evidence["good"] == []
    assert evidence["bad"] == []
    assert evidence["counts"]["good"] == 0
    assert "quarantined" in evidence["error"]


def test_catalog_evidence_verified_shape(tmp_path, monkeypatch):
    from webapp import catalog_evidence

    _evidence_files(tmp_path)
    monkeypatch.setenv("BIPRED_WEB_LDPRED3_BENCHMARKS", str(tmp_path))

    evidence = catalog_evidence.load()

    assert evidence["available"] is True
    assert evidence["trusted"] is True
    assert evidence["table_hash_verified"] is True
    assert evidence["good"][0]["accession"] == "GCST1"


def test_catalog_evidence_malformed_manifest_fails_closed(tmp_path, monkeypatch):
    from webapp import catalog_evidence

    _evidence_files(tmp_path)
    (tmp_path / "real_gwas_pipeline_catalog.manifest.json").write_text(
        "[]", encoding="utf-8")
    monkeypatch.setenv("BIPRED_WEB_LDPRED3_BENCHMARKS", str(tmp_path))

    evidence = catalog_evidence.load()

    assert evidence["available"] is False
    assert evidence["good"] == [] and evidence["bad"] == []
    assert "JSON object" in evidence["error"]


def test_ldpred3_install_is_immutable_and_consistent():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert f"ldpred3.git@{PIN}" in readme
    assert f'LDPRED3_REV: "{PIN}"' in workflow
    assert "ldpred3.git@master" not in readme
    assert 'LDPRED3_REV: "master"' not in workflow


def test_frontend_integrity_contracts_are_explicit():
    templates = ROOT / "webapp/templates"
    static = ROOT / "webapp/static"
    base = (templates / "base.html").read_text(encoding="utf-8")
    index = (templates / "index.html").read_text(encoding="utf-8")
    job = (templates / "job.html").read_text(encoding="utf-8")
    results = (templates / "results.html").read_text(encoding="utf-8")
    scripts = "\n".join(
        (static / name).read_text(encoding="utf-8")
        for name in ("catalog.js", "form.js", "job.js", "preview.js")
    )

    assert 'href="/demo"' not in base + index
    assert (base + index).count('method="post" action="/demo"') == 2
    assert ".csv,.gz,.bgz" in index
    assert "combined multipart upload limit" in index.lower()
    assert "hasFile !== hasAccession" in scripts
    assert "innerHTML" not in scripts
    assert "AbortController" in scripts
    assert "document.hidden" in scripts
    assert "data-overlap-mode" in results
    assert "mixer_fig.shared" in results
    assert "mix_unc.n_shared.interval" in results
    assert "mixer_uncertainty_basis" in results
    assert "figs.joint" not in results
    assert "Model-implied MiXeR overlap" in results
    assert 'data-overlap-mode="{% if overlap_empty %}empty{% elif overlap_identical %}identical' in results
    assert "identical modeled sets" in results
    assert "no liability-scale conversion" in " ".join(results.split())
    assert results.count('<th scope="col"') >= 14
    assert results.count('<th scope="row"') >= 35
    assert job.count('aria-live="polite"') == 1
    for number in range(1, 9):
        assert f"Table {number}" in results
