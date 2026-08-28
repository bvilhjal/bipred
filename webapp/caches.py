"""LD-cache registry for the bipred web service.

The default is the **UK Biobank European HapMap3+** reference when present
(LDpred2's SNP set, 1.44M variants), else HapMap3 (1.05M), both converted by
``ldpred3/benchmarks/convert_bigsnpr_ldref.py``. More real caches can be
registered through the
``BIPRED_WEB_CACHES`` environment variable as ``name=/path/cache.ld.npz``
entries separated by ``;`` — build them once with
``ldpred3.compute_ld_blocks(..., quantize=True)`` + ``ldpred3.save_ld_blocks``
and mount them read-only on the worker host. A synthetic *demo* cache (built
on first use by :mod:`webapp.demo`) is always listed last.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import logging
import os
import threading
import uuid
from pathlib import Path

import numpy as np

from . import jobs

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Conventional locations of converted UKB European caches in this workspace
# (sibling ldpred3 checkout's benchmark work dir). HapMap3+ is LDpred2's
# default SNP set; list it first so it is the form default when both exist.
_LDPRED3_WORK = REPO_ROOT.parent / "ldpred3" / "benchmarks" / ".work"
_REAL_CACHES = (
    ("ukb-eur-hm3plus",
     _LDPRED3_WORK / "ldref-hm3-plus" / "ldpred3_ldref_hm3_plus.npz",
     "UK Biobank European (HapMap3+, 1.44M variants, n=362k)"),
    ("ukb-eur-hm3",
     _LDPRED3_WORK / "ldref-hm3" / "ldpred3_ldref_hm3.npz",
     "UK Biobank European (HapMap3, 1.05M variants, n=362k)"),
)
UKB_EUR_KEY = "ukb-eur-hm3"
UKB_EUR_PATH = _REAL_CACHES[1][1]
UKB_EUR_LABEL = _REAL_CACHES[1][2]

_HASH_CACHE = {}
_HASH_CACHE_LOCK = threading.Lock()

_HASH_RECORD_SCHEMA = 3
_RAW_HASH_KIND = "raw-file-sha256"
_MMAP_HASH_KIND = "ldpred3-mmap-generation-sha256-v1"
_MMAP_PAYLOAD_FIELDS = ("payload_file", "payload_file_i8")

_LD_SCORE_SCHEMA = 2
LD_SCORE_CACHE_DEFINITION = (
    "full-reference-transformed-cache-colsum-r2-including-self")
LD_SCORE_SOURCE_MAP_DEFINITION = (
    "full-reference-source-map-colsum-r2-including-self")
# Compatibility name for callers that compute scores from the exact blocks
# passed to ldpred3. Source-reference-map scores must opt into their distinct
# definition explicitly.
LD_SCORE_DEFINITION = LD_SCORE_CACHE_DEFINITION
_LD_SCORE_DEFINITIONS = frozenset({
    LD_SCORE_CACHE_DEFINITION, LD_SCORE_SOURCE_MAP_DEFINITION,
})
_LEGACY_LD_SCORE_DEFINITION = "full-reference-colsum-r2-including-self"


@dataclass(frozen=True)
class LDScorePanel:
    """Reference-wide LD scores bound to one exact LD-cache generation."""

    scores: np.ndarray
    m_snps: int
    cache_sha256: str
    source: str
    source_sha256: str | None
    algorithm: str
    correction: str
    score_sha256: str
    score_sum: float
    score_mean: float
    effective_rank: float
    definition: str = LD_SCORE_CACHE_DEFINITION


def demo_cache_dir(root: Path | None = None) -> Path:
    root = root or jobs.data_root()
    out = root / "caches" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    return out


def real_registry(root: Path | None = None) -> list[dict]:
    """Real analysis caches, excluding the deliberately synthetic demo."""
    out = []
    for key, path, label in _REAL_CACHES:
        if path.exists():
            out.append({"key": key, "label": label, "path": str(path)})
    for entry in os.environ.get("BIPRED_WEB_CACHES", "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"BIPRED_WEB_CACHES entry {entry!r} needs name=/path/cache.npz")
        name, path = (part.strip() for part in entry.split("=", 1))
        if not name:
            raise ValueError(
                f"BIPRED_WEB_CACHES entry {entry!r}: empty name")
        if not Path(path).exists():
            # A reference that vanishes after startup (an unmounted drive)
            # must not fail the index page or other caches' running jobs;
            # submissions naming this key are rejected at cache_path.
            LOGGER.warning(
                "BIPRED_WEB_CACHES entry %r no longer exists; skipping", entry)
            continue
        out.append({"key": name, "label": name, "path": path})
    return out


def registry(root: Path | None = None) -> list[dict]:
    """All caches as ``{key, label, path}``; the demo is always last."""
    out = real_registry(root)
    out.append({
        "key": "demo",
        "label": "Demo (synthetic, small)",
        "path": str(demo_cache_dir(root) / "demo.ld.npz"),
    })
    return out


def default_key(root: Path | None = None) -> str:
    """Form default: the first real cache when one exists, else the demo."""
    return registry(root)[0]["key"]


def cache_path(key: str, root: Path | None = None) -> Path:
    for entry in registry(root):
        if entry["key"] == key:
            return Path(entry["path"])
    raise KeyError(f"unknown LD cache {key!r}")


def _identity(stat):
    # ``st_ctime`` is not a portable change counter.  In particular, Python
    # 3.14 reports Windows creation time for ``Path.stat()`` while CRT-backed
    # ``fstat()`` can expose a different value for the same open file.  That
    # made an unchanged cache look as if it had been replaced while hashing.
    # Device/inode bind the pathname to the opened file; size and mtime bind
    # in-place payload changes.  The content digest remains the final identity.
    return {name: int(getattr(stat, f"st_{name}")) for name in
            ("dev", "ino", "size", "mtime_ns")}


def _safe_payload_path(cache_path, value, field):
    """Resolve one metadata-bound mmap payload without leaving its directory."""
    if not isinstance(value, str):
        raise ValueError(f"LD cache {field} must be a sibling filename")
    name = value.strip()
    if (not name or name in (".", "..") or os.path.basename(name) != name
            or Path(name).is_absolute()):
        raise ValueError(f"LD cache {field} must be a sibling filename")
    base = Path(cache_path).resolve().parent
    candidate = base / name
    # The lexical basename check rejects ../ traversal. Resolving as well
    # prevents an existing symlink from smuggling a referenced payload out of
    # the cache directory.
    resolved = candidate.resolve(strict=False)
    if resolved.parent != base or resolved == Path(cache_path).resolve():
        raise ValueError(f"LD cache {field} must be a sibling payload file")
    return resolved


def _mmap_payloads(cache_path):
    """Return ``(metadata field, path)`` payloads referenced by an mmap cache."""
    try:
        archive = np.load(cache_path, allow_pickle=False)
    except (OSError, TypeError, ValueError, EOFError):
        return ()                         # an ordinary non-NPZ file
    if not isinstance(archive, np.lib.npyio.NpzFile):
        return ()                         # an ordinary NPY or other numpy file
    with archive:
        if "ondisk" not in archive:
            return ()
        marker = np.asarray(archive["ondisk"])
        if marker.size != 1:
            raise ValueError("LD cache ondisk metadata must be scalar")
        marker = marker.reshape(-1)[0].item()
        if (isinstance(marker, bool)
                or isinstance(marker, (int, np.integer))) \
                and int(marker) in (0, 1):
            is_mmap = bool(marker)
        else:
            raise ValueError("LD cache ondisk metadata must be 0 or 1")
        if not is_mmap:
            return ()
        out = []
        for field in _MMAP_PAYLOAD_FIELDS:
            if field not in archive:
                continue
            value = np.asarray(archive[field])
            if value.size != 1:
                raise ValueError(f"LD cache {field} metadata must be scalar")
            path = _safe_payload_path(
                cache_path, value.reshape(-1)[0].item(), field)
            out.append((field, path))
    if out:
        return tuple(out)
    # Schema-1 mmap caches used one implicit float payload. It is still part of
    # the numerical generation even though no payload_file field names it.
    legacy = Path(str(cache_path) + ".dat.npy").resolve(strict=False)
    if legacy.exists() and legacy.parent == Path(cache_path).resolve().parent:
        return (("legacy_payload_file", legacy),)
    raise ValueError("mmap LD cache has no payload-file binding")


def _generation_members(path):
    """Snapshot a stable metadata file and every numerical mmap payload."""
    path = Path(path).resolve()
    for attempt in (0, 1):
        before = _identity(path.stat())
        payloads = _mmap_payloads(path)
        after = _identity(path.stat())
        if before != after:
            if attempt:
                raise OSError(f"LD cache changed while reading metadata: {path}")
            continue
        members = [("metadata", path), *payloads]
        identities = []
        try:
            for role, member in members:
                identities.append((role, member, _identity(member.stat())))
        except OSError:
            if attempt:
                raise
            continue
        if _identity(path.stat()) == before:
            kind = _MMAP_HASH_KIND if payloads else _RAW_HASH_KIND
            return kind, tuple(identities)
        if attempt:
            raise OSError(f"LD cache changed while reading metadata: {path}")
    raise OSError(f"LD cache changed while reading metadata: {path}")


def _members_record(kind, members):
    return {
        "schema_version": _HASH_RECORD_SCHEMA,
        "hash_kind": kind,
        "path": str(members[0][1]),
        # Retain the historical top-level identity for easy inspection and
        # ordinary-file sidecar compatibility.
        "identity": members[0][2],
        "members": [
            {"role": role, "path": str(path), "identity": identity}
            for role, path, identity in members
        ],
    }


def _record_matches(record, kind, members):
    expected = _members_record(kind, members)
    if kind == _RAW_HASH_KIND and "schema_version" not in record:
        # Sidecars written by bipred 0.3.10/early 0.3.11 contain exactly these
        # fields and already represent the ordinary file's raw SHA-256.
        return (record.get("path") == expected["path"]
                and record.get("identity") == expected["identity"])
    return all(record.get(key) == expected[key] for key in (
        "schema_version", "hash_kind", "path", "identity", "members"))


def _process_hash_key(kind, members):
    return (kind, tuple(
        (role, str(path), *identity.values())
        for role, path, identity in members))


def _hash_open_file(path, expected):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        before = _identity(os.fstat(handle.fileno()))
        if before != expected:
            raise OSError(f"LD cache generation changed while hashing: {path}")
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
        after = _identity(os.fstat(handle.fileno()))
    if before != after or _identity(path.stat()) != expected:
        raise OSError(f"LD cache generation changed while hashing: {path}")
    return digest.hexdigest()


def _hash_generation(kind, members):
    file_hashes = []
    for role, path, identity in members:
        file_hashes.append({
            "role": role,
            "bytes": identity["size"],
            "sha256": _hash_open_file(path, identity),
        })
    for _role, path, identity in members:
        if _identity(path.stat()) != identity:
            raise OSError(f"LD cache generation changed while hashing: {path}")
    if kind == _RAW_HASH_KIND:
        return file_hashes[0]["sha256"]
    manifest = {
        "definition": _MMAP_HASH_KIND,
        "members": file_hashes,
    }
    wire = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(wire).hexdigest()


def _hash_sidecars(path, root):
    yield path.with_name(path.name + ".sha256")
    if root is not None:
        resolved = str(path.resolve()).encode("utf-8")
        name = hashlib.sha256(resolved).hexdigest() + ".json"
        yield Path(root) / "_meta" / "ld-cache-hashes" / name


def sha256_cached(path: Path, root: Path | None = None) -> str:
    """Content hash for an ordinary file or complete mmap-cache generation.

    Ordinary files retain their raw SHA-256. For an ldpred3 mmap cache the
    digest binds the metadata NPZ and every referenced numerical payload.
    Process and persistent caches are keyed by all member identities, so an
    in-place payload mutation cannot reuse a stale generation digest.
    """
    path = Path(path).resolve()
    kind, members = _generation_members(path)
    process_key = _process_hash_key(kind, members)
    with _HASH_CACHE_LOCK:
        value = _HASH_CACHE.get(process_key)
    if value is not None and all(
            _identity(member.stat()) == identity
            for _role, member, identity in members):
        return value
    for sidecar in _hash_sidecars(path, root):
        try:
            record = json.loads(sidecar.read_text())
            value = record.get("sha256", "").lower()
            if (not _record_matches(record, kind, members)
                    or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                    or any(_identity(member.stat()) != identity
                           for _role, member, identity in members)):
                continue
            with _HASH_CACHE_LOCK:
                _HASH_CACHE[process_key] = value
            return value
        except (OSError, ValueError, TypeError, AttributeError):
            continue

    # Hash one open generation, then ensure every pathname still names that
    # same member. A concurrent atomic cache publication is retried once.
    for attempt in (0, 1):
        try:
            value = _hash_generation(kind, members)
            break
        except OSError:
            if attempt:
                raise
            kind, members = _generation_members(path)
            process_key = _process_hash_key(kind, members)
    with _HASH_CACHE_LOCK:
        _HASH_CACHE[process_key] = value
    record = {**_members_record(kind, members), "sha256": value}
    for sidecar in _hash_sidecars(path, root):
        tmp = sidecar.with_name(
            f".{sidecar.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(record, sort_keys=True) + "\n")
            os.replace(tmp, sidecar)
            break
        except OSError:
            pass                            # try the writable data-root copy
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
    return value


def ld_score_sidecar_path(path: Path) -> Path:
    """Conventional precomputed full-reference LD-score sidecar path."""
    path = Path(path)
    return path.with_name(path.name + ".ldscores.npz")


def _ld_score_sidecars(path, root, cache_sha256):
    yield ld_score_sidecar_path(path)
    if root is not None:
        yield Path(root) / "_meta" / "ld-scores" / f"{cache_sha256}.npz"


def _one(archive, name):
    if name not in archive:
        raise ValueError(f"LD-score sidecar is missing {name!r}")
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"LD-score sidecar field {name!r} must be scalar")
    return value.reshape(-1)[0].item()


def _score_sha256(scores):
    wire = np.ascontiguousarray(scores, dtype=np.dtype("<f8"))
    return hashlib.sha256(memoryview(wire).cast("B")).hexdigest()


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value.lower()))


def _score_definition(schema, definition, algorithm):
    """Validate current definitions and migrate the two produced v1 forms."""
    if schema == 1 and definition == _LEGACY_LD_SCORE_DEFINITION:
        if algorithm == "source-map-colsum-r2-v1":
            return LD_SCORE_SOURCE_MAP_DEFINITION
        if algorithm == "ldpred3.ld_scores-v1":
            return LD_SCORE_CACHE_DEFINITION
        raise ValueError(
            "legacy LD-score sidecar has an ambiguous score definition")
    if definition not in _LD_SCORE_DEFINITIONS:
        raise ValueError(
            f"LD-score sidecar has incompatible definition {definition!r}")
    return definition


def _read_ld_score_sidecar(path, *, cache_sha256, n_variants):
    try:
        with np.load(path, allow_pickle=False) as archive:
            schema = int(_one(archive, "schema_version"))
            bound_hash = str(_one(archive, "cache_sha256"))
            m_snps = int(_one(archive, "m_snps"))
            definition = str(_one(archive, "definition"))
            source = str(_one(archive, "source"))
            source_sha256 = (
                str(_one(archive, "source_sha256"))
                if "source_sha256" in archive else None)
            algorithm = str(_one(archive, "algorithm"))
            correction = str(_one(archive, "correction"))
            stored_score_sha256 = str(_one(archive, "score_sha256"))
            if "scores" not in archive:
                raise ValueError("LD-score sidecar is missing 'scores'")
            scores = np.asarray(archive["scores"])
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read LD-score sidecar {path}: {exc}") \
            from None
    if schema not in (1, _LD_SCORE_SCHEMA):
        raise ValueError(
            f"LD-score sidecar {path} has schema {schema}, expected "
            f"{_LD_SCORE_SCHEMA}")
    if bound_hash != cache_sha256:
        raise ValueError(
            f"LD-score sidecar {path} belongs to another LD-cache generation")
    if not source.strip() or not algorithm.strip():
        raise ValueError(
            f"LD-score sidecar {path} must name its source and algorithm")
    try:
        definition = _score_definition(schema, definition, algorithm)
    except ValueError as exc:
        raise ValueError(f"LD-score sidecar {path}: {exc}") from None
    source_sha256 = source_sha256 or None
    if source_sha256 is not None and not _valid_sha256(source_sha256):
        raise ValueError(
            f"LD-score sidecar {path} has an invalid source SHA-256")
    if (definition == LD_SCORE_SOURCE_MAP_DEFINITION
            and source_sha256 is None):
        raise ValueError(
            f"LD-score sidecar {path} source-map scores require a source "
            "SHA-256")
    if correction != "none":
        raise ValueError(
            f"LD-score sidecar {path} has unsupported finite-reference "
            f"correction {correction!r}")
    if scores.dtype != np.float64:
        raise ValueError(
            f"LD-score sidecar {path} must store float64 scores")
    if m_snps != n_variants or scores.shape != (n_variants,):
        raise ValueError(
            f"LD-score sidecar {path} has {scores.size:,} scores and M={m_snps:,}; "
            f"the LD reference has {n_variants:,} variants")
    if not np.all(np.isfinite(scores)) or np.any(scores <= 0.0):
        raise ValueError(
            f"LD-score sidecar {path} must contain finite positive scores")
    actual_score_sha256 = _score_sha256(scores)
    if stored_score_sha256 != actual_score_sha256:
        raise ValueError(
            f"LD-score sidecar {path} has a score-payload hash mismatch")
    score_sum = float(np.sum(scores, dtype=np.float64))
    scores.setflags(write=False)
    return LDScorePanel(
        scores=scores, m_snps=m_snps, cache_sha256=bound_hash,
        source=source, source_sha256=source_sha256,
        algorithm=algorithm, correction=correction,
        score_sha256=actual_score_sha256, score_sum=score_sum,
        score_mean=score_sum / m_snps,
        effective_rank=(m_snps * m_snps) / score_sum,
        definition=definition)


def load_ld_score_panel(path: Path, root: Path | None = None, *,
                        cache_sha256: str | None = None,
                        n_variants: int | None = None) -> LDScorePanel:
    """Load scores for the exact full reference; never calculate job subsets.

    A sidecar next to the immutable LD cache is preferred. A content-addressed
    copy under the writable web-data root supports read-only cache mounts.
    """
    path = Path(path)
    cache_sha256 = cache_sha256 or sha256_cached(path, root)
    if n_variants is None:
        with np.load(path, allow_pickle=False) as archive:
            n_variants = int(np.asarray(archive["ids"]).size)
    candidates = list(_ld_score_sidecars(path, root, cache_sha256))
    for sidecar in candidates:
        if sidecar.exists():
            return _read_ld_score_sidecar(
                sidecar, cache_sha256=cache_sha256,
                n_variants=int(n_variants))
    expected = " or ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "this LD reference has no precomputed full-reference LD scores; "
        f"expected {expected}")


def write_ld_score_sidecar(
        path: Path, scores, *, source: str, root: Path | None = None,
        source_sha256: str | None = None,
        cache_sha256: str | None = None,
        algorithm: str = "external-precomputed-v1",
        definition: str = LD_SCORE_CACHE_DEFINITION) -> Path:
    """Atomically bind precomputed, cache-ordered scores to one LD reference."""
    path = Path(path)
    cache_sha256 = cache_sha256 or sha256_cached(path, root)
    with np.load(path, allow_pickle=False) as archive:
        n_variants = int(np.asarray(archive["ids"]).size)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (n_variants,):
        raise ValueError(
            f"expected {n_variants:,} cache-ordered LD scores, got "
            f"shape {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("LD scores must be finite and positive")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("LD-score source must be a non-empty string")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError("LD-score algorithm must be a non-empty string")
    if definition not in _LD_SCORE_DEFINITIONS:
        raise ValueError(
            f"unsupported LD-score definition {definition!r}")
    if source_sha256 is not None and not _valid_sha256(source_sha256):
        raise ValueError("LD-score source_sha256 must be a SHA-256 digest")
    if (definition == LD_SCORE_SOURCE_MAP_DEFINITION
            and source_sha256 is None):
        raise ValueError("source-map LD scores require source_sha256")
    source, algorithm = source.strip(), algorithm.strip()
    score_sha256 = _score_sha256(values)

    last_error = None
    for sidecar in _ld_score_sidecars(path, root, cache_sha256):
        tmp = sidecar.with_name(
            f".{sidecar.name}.{os.getpid()}.{uuid.uuid4().hex}.part.npz")
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                tmp, schema_version=np.array(_LD_SCORE_SCHEMA),
                cache_sha256=np.array(cache_sha256),
                m_snps=np.array(n_variants),
                definition=np.array(definition),
                source=np.array(source),
                source_sha256=np.array(source_sha256 or ""),
                algorithm=np.array(algorithm), correction=np.array("none"),
                score_sha256=np.array(score_sha256),
                scores=values)
            os.replace(tmp, sidecar)
            return sidecar
        except OSError as exc:
            last_error = exc
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
    raise OSError(f"could not write an LD-score sidecar: {last_error}")


def build_ld_score_sidecar_from_map(
        path: Path, map_path: Path, *, id_column="rsid", score_column="ld",
        root: Path | None = None, source: str | None = None) -> Path:
    """Import already-computed, cache-ordered scores from a reference map."""
    path, map_path = Path(path), Path(map_path)
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"])
    scores = np.empty(ids.size, dtype=np.float64)
    with open(map_path, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or id_column not in rows.fieldnames \
                or score_column not in rows.fieldnames:
            raise ValueError(
                f"{map_path} must contain {id_column!r} and "
                f"{score_column!r} columns")
        count = 0
        for count, row in enumerate(rows, start=1):
            i = count - 1
            if i >= ids.size:
                raise ValueError(
                    f"{map_path} contains more rows than the LD reference")
            if row[id_column] != str(ids[i]):
                raise ValueError(
                    f"{map_path} row {count + 1} has {row[id_column]!r}, "
                    f"but the LD reference has {str(ids[i])!r}")
            try:
                scores[i] = float(row[score_column])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{map_path} row {count + 1} has an invalid LD score") \
                    from None
    if count != ids.size:
        raise ValueError(
            f"{map_path} contains {count:,} rows; the LD reference has "
            f"{ids.size:,} variants")
    return write_ld_score_sidecar(
        path, scores, root=root,
        source=source or f"{map_path.name}:{score_column}",
        source_sha256=sha256_cached(map_path),
        algorithm="source-map-colsum-r2-v1",
        definition=LD_SCORE_SOURCE_MAP_DEFINITION)


def _builtin_ld_score_map(path):
    resolved = Path(path).resolve()
    for _key, cache, _label in _REAL_CACHES:
        if cache.exists() and cache.resolve() == resolved:
            source = cache.parent / "map.csv"
            return source if source.exists() else None
    return None


def load_or_create_ld_score_panel(
        path: Path, root: Path | None = None, *,
        cache_sha256: str | None = None,
        n_variants: int | None = None) -> LDScorePanel:
    """Load scores, importing the built-in reference map once if necessary."""
    try:
        return load_ld_score_panel(
            path, root, cache_sha256=cache_sha256,
            n_variants=n_variants)
    except FileNotFoundError:
        source = _builtin_ld_score_map(path)
        if source is None:
            raise
    build_ld_score_sidecar_from_map(
        path, source, root=root,
        source="Florian Prive UKB reference map column 'ld'")
    return load_ld_score_panel(
        path, root, cache_sha256=cache_sha256,
        n_variants=n_variants)
