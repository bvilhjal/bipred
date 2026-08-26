"""Focused contracts for web LD-cache and LD-score provenance."""

import hashlib
import json
import os

import numpy as np
import pytest

from webapp import caches


def _small_cache(path, *, mmap=False):
    from ldpred3 import save_ld_blocks

    ids = np.array(["rs1", "rs2", "rs3"])
    corr = np.array([
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.25],
        [0.0, 0.25, 1.0],
    ], dtype=np.float32)
    save_ld_blocks(
        path, [(corr, np.arange(3))], ids, mmap=mmap,
        counted_allele=np.array(["A", "A", "A"]),
        other_allele=np.array(["G", "G", "G"]),
        chrom=np.array(["1", "1", "1"]), pos=np.arange(1, 4),
        reference_af=np.full(3, 0.3), n_ref=500)
    return ids


def test_ordinary_cache_hash_remains_raw_file_sha256(tmp_path):
    path = tmp_path / "ordinary.bin"
    path.write_bytes(b"ordinary cache bytes")
    assert caches.sha256_cached(path) == hashlib.sha256(
        path.read_bytes()).hexdigest()


def test_mmap_cache_hash_invalidates_when_payload_changes(tmp_path):
    cache = tmp_path / "mmap.ld.npz"
    _small_cache(cache, mmap=True)
    metadata_sha = hashlib.sha256(cache.read_bytes()).hexdigest()
    first = caches.sha256_cached(cache)

    with np.load(cache, allow_pickle=False) as archive:
        payload_name = str(archive["payload_file"].reshape(-1)[0])
    payload_path = cache.parent / payload_name
    payload = np.load(payload_path, mmap_mode="r+")
    payload[1] = np.float32(payload[1] + 0.125)
    payload.flush()
    del payload
    os.utime(payload_path, None)

    second = caches.sha256_cached(cache)
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == metadata_sha
    assert second != first

    record = json.loads(caches._hash_sidecars(cache, None).__next__().read_text())
    assert record["hash_kind"] == "ldpred3-mmap-generation-sha256-v1"
    assert [member["role"] for member in record["members"]] == [
        "metadata", "payload_file"]


def test_mmap_payload_binding_rejects_path_traversal(tmp_path):
    cache = tmp_path / "traversal.ld.npz"
    np.savez(
        cache, ondisk=np.array([1], dtype=np.int8),
        payload_file=np.array(["../outside.npy"]))
    with pytest.raises(ValueError, match="sibling filename"):
        caches.sha256_cached(cache)


def test_ld_score_definitions_distinguish_cache_from_source_map(tmp_path):
    cache = tmp_path / "ld.npz"
    ids = _small_cache(cache)
    scores = np.array([1.25, 1.5, 1.25])
    source_hash = "a" * 64

    caches.write_ld_score_sidecar(
        cache, scores, source="exact transformed blocks",
        source_sha256=source_hash, algorithm="ldpred3.ld_scores-v1")
    exact = caches.load_ld_score_panel(cache)
    assert exact.definition == caches.LD_SCORE_CACHE_DEFINITION
    assert exact.algorithm == "ldpred3.ld_scores-v1"
    assert exact.source_sha256 == source_hash

    caches.ld_score_sidecar_path(cache).unlink()
    source = tmp_path / "map.csv"
    source.write_text(
        "rsid,ld\n" + "".join(
            f"{variant},{score}\n"
            for variant, score in zip(ids, scores)), encoding="utf-8")
    caches.build_ld_score_sidecar_from_map(cache, source)
    mapped = caches.load_ld_score_panel(cache)
    assert mapped.definition == caches.LD_SCORE_SOURCE_MAP_DEFINITION
    assert mapped.algorithm == "source-map-colsum-r2-v1"
    assert mapped.source_sha256 == hashlib.sha256(
        source.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="unsupported LD-score definition"):
        caches.write_ld_score_sidecar(
            cache, scores, source="unknown", algorithm="unknown-v1",
            definition="ambiguous-colsum-r2")


def test_legacy_ukb_map_sidecar_loads_with_explicit_definition(tmp_path):
    cache = tmp_path / "ld.npz"
    _small_cache(cache)
    scores = np.array([1.25, 1.5, 1.25], dtype=np.float64)
    score_sha = hashlib.sha256(scores.astype("<f8").tobytes()).hexdigest()
    cache_sha = caches.sha256_cached(cache)
    source_sha = "b" * 64
    np.savez_compressed(
        caches.ld_score_sidecar_path(cache),
        schema_version=np.array(1), cache_sha256=np.array(cache_sha),
        m_snps=np.array(3),
        definition=np.array("full-reference-colsum-r2-including-self"),
        source=np.array("UKB reference map"),
        source_sha256=np.array(source_sha),
        algorithm=np.array("source-map-colsum-r2-v1"),
        correction=np.array("none"), score_sha256=np.array(score_sha),
        scores=scores)

    panel = caches.load_ld_score_panel(cache)
    assert panel.definition == caches.LD_SCORE_SOURCE_MAP_DEFINITION
    assert panel.source_sha256 == source_sha
