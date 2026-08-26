"""Regression tests for GWAS Catalog deposit-generation validators."""

import gzip
import hashlib
import io
import json
import os
import time
from pathlib import Path

import pytest

from webapp import gwascat


class _HeadResponse:
    def __init__(self, headers):
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _GetResponse(io.BytesIO):
    def __init__(self, body, headers):
        super().__init__(body)
        self.headers = headers


def _write_source(path: Path, beta: str = "0.1") -> int:
    """Write equal-length fixtures with deterministic, uncompressed gzip."""
    header = (
        "rsid\tchromosome\tbase_pair_location\teffect_allele\tother_allele"
        "\tbeta\tstandard_error\teffect_allele_frequency\tp_value\tn\n"
    )
    row = f"rs1\t1\t101\tA\tG\t{beta}\t0.01\t0.4\t0.05\t1000\n"
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=0,
                           mtime=0) as dst:
            dst.write((header + row).encode())
    return path.stat().st_size


def _fetch(src, dest, root, **validators):
    return gwascat.fetch_filtered(
        str(src), dest, accession="GCST000099", root=root,
        keep_ids={"rs1"}, fingerprint="ld-hash",
        remote_bytes=src.stat().st_size,
        coverage=lambda: ({"rs1"}, {"ld-hash": "reference"}),
        **validators,
    )


def test_resolve_captures_stable_head_validators(tmp_path, monkeypatch):
    accession = "GCST000099"
    monkeypatch.setattr(
        gwascat, "_harmonised_paths",
        lambda _root: {accession: "GCST000099/file.h.tsv.gz"},
    )
    monkeypatch.setattr(
        gwascat, "_study_metadata",
        lambda _accession, _root: {"accession": accession, "trait": "Trait"},
    )

    def fake_open(request, timeout):
        assert request.get_method() == "HEAD"
        assert timeout == 60
        return _HeadResponse({
            "Content-Length": "123",
            "ETag": '  W/"deposit-2"  ',
            "Last-Modified": "  Wed, 26 Aug 2026 08:00:00 GMT ",
        })

    monkeypatch.setattr(gwascat.urllib.request, "urlopen", fake_open)
    meta = gwascat.resolve(accession, tmp_path)

    assert meta["remote_bytes"] == 123
    assert meta["remote_etag"] == 'W/"deposit-2"'
    assert meta["remote_last_modified"] == (
        "Wed, 26 Aug 2026 08:00:00 GMT")


def test_resolve_omits_unavailable_validators(tmp_path, monkeypatch):
    accession = "GCST000099"
    monkeypatch.setattr(
        gwascat, "_harmonised_paths",
        lambda _root: {accession: "GCST000099/file.h.tsv.gz"},
    )
    monkeypatch.setattr(
        gwascat, "_study_metadata",
        lambda _accession, _root: {"accession": accession, "trait": "Trait"},
    )
    monkeypatch.setattr(
        gwascat.urllib.request, "urlopen",
        lambda _request, timeout: _HeadResponse(
            {"Content-Length": "123", "ETag": "  "}),
    )

    meta = gwascat.resolve(accession, tmp_path)

    assert "remote_etag" not in meta
    assert "remote_last_modified" not in meta


@pytest.mark.parametrize(
    ("field", "first", "second"),
    [
        ("remote_etag", '"deposit-1"', '"deposit-2"'),
        ("remote_last_modified",
         "Tue, 25 Aug 2026 08:00:00 GMT",
         "Wed, 26 Aug 2026 08:00:00 GMT"),
    ],
)
def test_changed_validator_rebuilds_same_url_and_size(
        tmp_path, field, first, second):
    root = tmp_path / "data"
    src = tmp_path / "same-url.h.tsv.gz"
    source_bytes = _write_source(src, "0.1")

    original = _fetch(
        src, tmp_path / "first.tsv.gz", root, **{field: first})
    assert original["reused"] is False

    assert _write_source(src, "0.2") == source_bytes
    replaced = _fetch(
        src, tmp_path / "second.tsv.gz", root, **{field: second})

    assert replaced["reused"] is False
    with gzip.open(tmp_path / "second.tsv.gz", "rt") as result:
        assert "\t0.2\t" in result.read()
    build = json.loads(
        (root / "catalog" / "GCST000099.json").read_text())
    assert build[field] == second


def test_current_validator_rebuilds_legacy_unvalidated_copy(tmp_path):
    root = tmp_path / "data"
    src = tmp_path / "same-url.h.tsv.gz"
    _write_source(src)

    _fetch(src, tmp_path / "first.tsv.gz", root)
    current = _fetch(
        src, tmp_path / "second.tsv.gz", root,
        remote_etag='"deposit-1"')

    assert current["reused"] is False
    build = json.loads(
        (root / "catalog" / "GCST000099.json").read_text())
    assert build["remote_etag"] == '"deposit-1"'


def test_missing_current_validator_uses_compatible_fallback(tmp_path):
    root = tmp_path / "data"
    src = tmp_path / "same-url.h.tsv.gz"
    _write_source(src)

    _fetch(
        src, tmp_path / "first.tsv.gz", root,
        remote_etag='"deposit-1"')
    fallback = _fetch(src, tmp_path / "second.tsv.gz", root)

    assert fallback["reused"] is True


def test_get_is_conditioned_and_matches_the_head_generation(
        tmp_path, monkeypatch):
    source = tmp_path / "remote.h.tsv.gz"
    _write_source(source)
    body = source.read_bytes()
    etag = '"deposit-1"'
    modified = "Wed, 26 Aug 2026 08:00:00 GMT"
    requests = []

    def fake_open(request, timeout):
        requests.append(request)
        assert timeout == 900
        return _GetResponse(body, {
            "ETag": etag, "Last-Modified": modified,
        })

    monkeypatch.setattr(gwascat.urllib.request, "urlopen", fake_open)
    result = gwascat.fetch_filtered(
        "https://example.test/deposit.gz", tmp_path / "result.tsv.gz",
        accession="GCST000100", root=tmp_path / "data",
        keep_ids={"rs1"}, fingerprint="ld-hash",
        remote_bytes=len(body), remote_etag=etag,
        remote_last_modified=modified,
        coverage=lambda: ({"rs1"}, {"ld-hash": "reference"}))

    headers = {key.lower(): value
               for key, value in requests[0].header_items()}
    assert headers["if-match"] == etag
    assert headers["if-unmodified-since"] == modified
    assert result["reused"] is False
    build = json.loads(
        (tmp_path / "data/catalog/GCST000100.json").read_text())
    assert build["remote_etag"] == etag
    assert build["remote_last_modified"] == modified


def test_weak_etag_is_checked_but_not_sent_as_if_match(tmp_path, monkeypatch):
    source = tmp_path / "remote.h.tsv.gz"
    size = _write_source(source)
    etag = 'W/"deposit-1"'
    requests = []

    def fake_open(request, timeout):
        requests.append(request)
        return _GetResponse(source.read_bytes(), {"ETag": etag})

    monkeypatch.setattr(gwascat.urllib.request, "urlopen", fake_open)
    result = gwascat.fetch_filtered(
        "https://example.test/deposit.gz", tmp_path / "result.tsv.gz",
        accession="GCST000104", root=tmp_path / "data",
        keep_ids={"rs1"}, fingerprint="ld-hash", remote_bytes=size,
        remote_etag=etag,
        coverage=lambda: ({"rs1"}, {"ld-hash": "reference"}))

    headers = {key.lower(): value
               for key, value in requests[0].header_items()}
    assert "if-match" not in headers
    assert result["reused"] is False


def test_changed_get_generation_is_retried_before_publication(
        tmp_path, monkeypatch):
    source = tmp_path / "same-name.h.tsv.gz"
    expected_bytes = _write_source(source, "0.1")
    expected_body = source.read_bytes()
    assert _write_source(source, "0.2") == expected_bytes
    changed_body = source.read_bytes()
    bodies = [
        _GetResponse(changed_body, {"ETag": '"deposit-2"'}),
        _GetResponse(expected_body, {"ETag": '"deposit-1"'}),
    ]
    calls = []

    def fake_open(request, timeout):
        calls.append(request)
        return bodies.pop(0)

    monkeypatch.setattr(gwascat.urllib.request, "urlopen", fake_open)
    root = tmp_path / "data"
    result = gwascat.fetch_filtered(
        "https://example.test/deposit.gz", tmp_path / "result.tsv.gz",
        accession="GCST000101", root=root, keep_ids={"rs1"},
        fingerprint="ld-hash", remote_bytes=expected_bytes,
        remote_etag='"deposit-1"',
        coverage=lambda: ({"rs1"}, {"ld-hash": "reference"}))

    assert len(calls) == 2
    assert result["sha256"] == hashlib.sha256(expected_body).hexdigest()
    with gzip.open(tmp_path / "result.tsv.gz", "rt") as output:
        assert "\t0.1\t" in output.read()
    build = json.loads(
        (root / "catalog/GCST000101.json").read_text())
    assert build["remote_etag"] == '"deposit-1"'
    assert not list((root / "catalog").glob("*.build"))


def test_persistent_or_missing_get_validator_is_rejected(
        tmp_path, monkeypatch):
    source = tmp_path / "changed.h.tsv.gz"
    size = _write_source(source, "0.2")
    body = source.read_bytes()
    responses = [
        _GetResponse(body, {}),
        _GetResponse(body, {"ETag": '"deposit-2"'}),
    ]
    monkeypatch.setattr(
        gwascat.urllib.request, "urlopen",
        lambda _request, timeout: responses.pop(0))
    root = tmp_path / "data"

    with pytest.raises(ValueError, match=(
            "changed between validation and download.*retry")):
        gwascat.fetch_filtered(
            "https://example.test/deposit.gz", tmp_path / "result.tsv.gz",
            accession="GCST000102", root=root, keep_ids={"rs1"},
            fingerprint="ld-hash", remote_bytes=size,
            remote_etag='"deposit-1"',
            coverage=lambda: ({"rs1"}, {"ld-hash": "reference"}))

    assert not (root / "catalog/GCST000102.tsv.gz").exists()
    assert not (root / "catalog/GCST000102.json").exists()
    assert not list((root / "catalog").glob("*.build"))
    assert not list((root / "catalog").glob("*.part"))


def test_get_without_head_validators_uses_size_fallback(tmp_path, monkeypatch):
    source = tmp_path / "remote.h.tsv.gz"
    size = _write_source(source)
    requests = []

    def fake_open(request, timeout):
        requests.append(request)
        return _GetResponse(source.read_bytes(), {})

    monkeypatch.setattr(gwascat.urllib.request, "urlopen", fake_open)
    result = gwascat.fetch_filtered(
        "https://example.test/deposit.gz", tmp_path / "result.tsv.gz",
        accession="GCST000103", root=tmp_path / "data",
        keep_ids={"rs1"}, fingerprint="ld-hash", remote_bytes=size,
        coverage=lambda: ({"rs1"}, {"ld-hash": "reference"}))

    headers = {key.lower(): value
               for key, value in requests[0].header_items()}
    assert "if-match" not in headers
    assert "if-unmodified-since" not in headers
    assert result["reused"] is False


def test_purge_removes_stale_builds_and_counts_live_transients(tmp_path):
    root = tmp_path / "data"
    source = tmp_path / "remote.h.tsv.gz"
    _write_source(source)
    _fetch(source, tmp_path / "result.tsv.gz", root)
    base = root / "catalog"
    data = base / "GCST000099.tsv.gz"
    meta = base / "GCST000099.json"
    build = json.loads(meta.read_text())
    build["last_used"] = time.time() - gwascat.EVICT_GRACE - 1
    meta.write_text(json.dumps(build))

    stale = base / ".GCST000200.tsv.gz.dead.build"
    stale.write_bytes(b"stale")
    os.utime(stale, (0, 0))
    live = base / ".GCST000201.tsv.gz.live.build"
    live.write_bytes(b"active build bytes")

    assert gwascat.purge_store(root, 0) == []
    assert not stale.exists()
    assert live.exists()
    # The published copy alone exactly fits. Counting the active build pushes
    # the store over budget and therefore evicts the old, eligible copy.
    budget_gb = data.stat().st_size / 2 ** 30
    assert gwascat.purge_store(root, budget_gb) == ["GCST000099"]
    assert not data.exists()
    assert live.exists()
