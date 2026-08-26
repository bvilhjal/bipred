"""Persistent, semantic cache for one prepared GWAS trait.

The expensive trait-local work -- reading summary statistics, QC,
harmonisation, standardisation and the mandatory trait-local LD-consistency
screen -- depends on the logical input, LD reference, resolved sample-size
semantics and preparation algorithm.  It does not depend on the other trait or
on later pairing and fitting choices.

Callers make that boundary explicit with :func:`semantic_spec`, then pass the
specification and a zero-argument builder to :func:`get_or_build`.  Artifacts
contain only numeric arrays in a compressed NPZ; provenance, warnings and the
checksum live in adjacent JSON.  A content checksum makes the two atomic
renames a logically atomic publication: readers accept neither half of a
partially published pair.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import math
import os
import platform
import re
import sys
import threading
import time
import uuid
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import ldpred3
import numpy as np

import bipred
from bipred.prepare import PreparedTrait


__all__ = [
    "semantic_spec", "key_for", "get_or_build", "purge", "store_dir",
]

STORE_DIRNAME = "prepared"
SPEC_SCHEMA = "bipred-prepared-semantic-v4"
DEFAULT_ALGORITHM_SCHEMA = "prepared-trait-v4"
FORMAT = "bipred-prepared-trait"
FORMAT_VERSION = 4

LOCK_STALE = 900.0
LOCK_TOUCH = 30.0
WAIT_LIMIT = 3600.0
WAIT_POLL = 0.5
PART_STALE = 3600.0
EVICT_GRACE = 3600.0

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPEC_FIELDS = frozenset({
    "spec_schema", "logical_input_sha256", "ld_sha256", "n_semantics",
    "columns", "qc", "screen", "algorithm_schema", "versions",
    "numerical_environment",
})
_QC_FIELDS = frozenset({"enabled", "params"})
_SCREEN_FIELDS = frozenset({"enabled", "params"})
_SCREEN_PARAM_FIELDS = frozenset({
    "rounds", "window", "threshold", "eigenvalue_floor", "seed", "ncores",
    "verbose",
})
_SCREEN_RECORD_FIELDS = frozenset({
    "n_input", "n_kept", "n_dropped", "parameters",
})
_VERSION_FIELDS = frozenset({"bipred", "ldpred3"})
_NUMERICAL_ENVIRONMENT_FIELDS = frozenset({"numpy_version", "backend"})
_NUMERICAL_BACKEND_FIELDS = frozenset({"blas", "lapack"})
_NUMERICAL_COMPONENT_FIELDS = frozenset({
    "implementation", "version", "integer_api", "architecture",
})
_INTEGER_APIS = frozenset({"lp64", "ilp64", "unknown"})
_ARRAY_NAMES = ("indices", "beta_hat", "n_eff", "z", "eaf", "n_cache")
_META_FIELDS = frozenset({
    "format", "format_version", "key", "spec", "npz_sha256", "npz_bytes",
    "created", "last_used", "arrays", "log", "warnings",
})
_WARNING_FIELDS = frozenset({"message", "category", "module"})

DEFAULT_SCREEN_PARAMS = {
    "rounds": 4,
    "window": 1000,
    "threshold": 29.72,
    "eigenvalue_floor": 1e-3,
    "seed": 0,
    "ncores": 1,
    "verbose": False,
}


def _fields(value, expected, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unknown = sorted(actual - set(expected))
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unknown:
            detail.append(f"unknown {unknown}")
        raise ValueError(f"{path} has " + " and ".join(detail))


def _plain(value, path="value"):
    """Convert JSON-like numpy scalars while rejecting ambiguous objects."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            out[key] = _plain(item, f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise ValueError(f"{path} contains unsupported {type(value).__name__}")


def _sha(value, name):
    if not isinstance(value, str) or not _SHA256.fullmatch(value.lower()):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _integer_api(*values, explicit_ilp64=False, known_backend=False):
    if explicit_ilp64:
        return "ilp64"
    text = " ".join(str(value) for value in values if value is not None)
    if re.search(
            r"(?:ILP64|USE_64BITINT\s*=\s*1|SYMBOL_SUFFIX\s*=\s*64_|"
            r"OPENBLAS64|MKL_ILP64)", text, flags=re.IGNORECASE):
        return "ilp64"
    return "lp64" if known_backend else "unknown"


def _normalised_architecture(value=None):
    """Return a portable CPU-family label, never a machine-specific path."""
    name = str(value or platform.machine() or "unknown").strip().lower()
    name = name.replace("-", "_")
    return {
        "amd64": "x86_64",
        "x64": "x86_64",
        "arm64": "aarch64",
    }.get(name, name or "unknown")


def _named_mapping(mapping, name):
    if not isinstance(mapping, dict):
        return {}
    wanted = name.lower().replace("_", " ")
    return next((
        value for key, value in mapping.items()
        if str(key).lower().replace("_", " ") == wanted
        and isinstance(value, dict)
    ), {})


def _backend_family(*values):
    """Classify selected build markers without inspecting their paths."""
    text = " ".join(str(value) for value in values if value is not None)
    if re.search(r"accelerate|veclib", text, flags=re.IGNORECASE):
        return "accelerate"
    if re.search(r"openblas", text, flags=re.IGNORECASE):
        return "openblas"
    if re.search(r"(?:^|[^a-z])mkl(?:[^a-z]|$)", text,
                 flags=re.IGNORECASE):
        return "mkl"
    if re.search(r"flexiblas", text, flags=re.IGNORECASE):
        return "flexiblas"
    if re.search(r"(?:^|[^a-z])blis(?:[^a-z]|$)", text,
                 flags=re.IGNORECASE):
        return "blis"
    if re.search(r"(?:^|[^a-z])atlas(?:[^a-z]|$)", text,
                 flags=re.IGNORECASE):
        return "atlas"
    return None


def _dependency_markers(info):
    """Build fields that describe a library rather than where it was built."""
    if not isinstance(info, dict):
        return []
    return [
        info.get("name"), info.get("libraries"),
        info.get("openblas configuration"),
        info.get("blas configuration"), info.get("configuration"),
        info.get("define_macros"),
    ]


def _compiler_defines(config):
    """Extract only compiler definitions, excluding args that contain paths."""
    definitions = []
    for compiler in _named_mapping(config, "compilers").values():
        if not isinstance(compiler, dict):
            continue
        for field in ("args", "linker args"):
            value = compiler.get(field)
            if value is None:
                continue
            definitions.extend(re.findall(
                r"(?:^|[\s,])(-D[A-Za-z_][A-Za-z0-9_]*"
                r"(?:=[^\s,]+)?)", str(value)))
        macros = compiler.get("define_macros")
        if macros is not None:
            definitions.append(str(macros))
    return definitions


def _modern_architecture(config):
    machine = _named_mapping(config, "machine information")
    host = _named_mapping(machine, "host")
    return _normalised_architecture(host.get("cpu") or host.get("family"))


def _accelerate_version():
    """Use the stable OS release because Accelerate has no library version."""
    version = platform.mac_ver()[0].strip()
    return version or "system"


def _modern_dependency(config, kind):
    dependencies = _named_mapping(config, "build dependencies")
    info = _named_mapping(dependencies, kind)
    if not info:
        return None
    other_kind = "lapack" if kind == "blas" else "blas"
    other = _named_mapping(dependencies, other_kind)
    own_family = _backend_family(*_dependency_markers(info))
    compiler_defines = _compiler_defines(config)
    compiler_family = _backend_family(*compiler_defines)
    other_family = _backend_family(*_dependency_markers(other))
    implementation = own_family or compiler_family or other_family or "unknown"
    if implementation == "accelerate":
        version = _accelerate_version()
    elif own_family == implementation:
        version = str(info.get("version") or "unknown").strip()
    elif other_family == implementation:
        version = str(other.get("version") or "unknown").strip()
    else:
        # A generic ``blas``/``lapack`` package version describes the ABI
        # shim, not the implementation selected underneath it.
        version = "unknown"
    stable_configuration = (
        _dependency_markers(info) + _dependency_markers(other)
        + compiler_defines)
    return {
        "implementation": implementation,
        "version": version or "unknown",
        "integer_api": _integer_api(
            *stable_configuration, known_backend=implementation != "unknown"),
        "architecture": _modern_architecture(config),
    }


def _legacy_dependency(kind):
    get_info = getattr(np.__config__, "get_info", None)
    if not callable(get_info):
        return None
    candidates = (
        f"{kind}_ilp64_opt_info", f"{kind}_opt_info",
        "openblas64__info", "openblas_info", f"{kind}_mkl_info",
        "accelerate_info",
    )
    for key in candidates:
        try:
            info = get_info(key)
        except (AttributeError, TypeError, ValueError):
            continue
        if not isinstance(info, dict) or not info:
            continue
        markers = [key, *_dependency_markers(info)]
        implementation = _backend_family(*markers) or "unknown"
        version = (_accelerate_version() if implementation == "accelerate"
                   else str(info.get("version") or "unknown").strip())
        return {
            "implementation": implementation,
            "version": version or "unknown",
            "integer_api": _integer_api(
                *markers, explicit_ilp64="ilp64" in key,
                known_backend=implementation != "unknown"),
            "architecture": _normalised_architecture(),
        }
    return None


def _runtime_dependency(architecture):
    """Identify a generic BLAS shim from its loaded implementation, if able.

    ``threadpoolctl`` is optional. Its paths and thread counts help it inspect
    the process but are deliberately excluded from the returned identity.
    """
    try:
        np.linalg.eigh(np.eye(1, dtype=np.float64))
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except (ImportError, OSError, RuntimeError):
        return None
    candidates = []
    numpy_root = Path(np.__file__).resolve().parent
    for pool in pools:
        if not isinstance(pool, dict) or pool.get("user_api") != "blas":
            continue
        family = _backend_family(pool.get("internal_api"), pool.get("prefix"))
        if family is None:
            continue
        numpy_owned = False
        try:
            Path(pool.get("filepath", "")).resolve().relative_to(numpy_root)
            numpy_owned = True
        except (OSError, TypeError, ValueError):
            pass
        candidates.append((numpy_owned, family, pool))
    owned = [item for item in candidates if item[0]]
    selected = owned if owned else candidates
    identities = {
        (family, str(pool.get("version") or "unknown").strip() or "unknown",
         _integer_api(
             pool.get("prefix"), Path(pool.get("filepath") or "").name,
             known_backend=True))
        for _, family, pool in selected
    }
    if len(identities) != 1:
        return None
    implementation, version, integer_api = identities.pop()
    return {
        "implementation": implementation,
        "version": version,
        "integer_api": integer_api,
        "architecture": architecture,
    }


def _detected_numerical_backend():
    """Stable NumPy BLAS/LAPACK identity, without paths or thread counts."""
    config = getattr(np.__config__, "CONFIG", None)
    if not isinstance(config, dict):
        try:
            config = np.show_config(mode="dicts")
        except (AttributeError, TypeError, ValueError):
            config = None
    architecture = (_modern_architecture(config)
                    if isinstance(config, dict)
                    else _normalised_architecture())
    unknown = {
        "implementation": "unknown", "version": "unknown",
        "integer_api": "unknown", "architecture": architecture,
    }
    detected = {
        kind: (_modern_dependency(config, kind)
               if isinstance(config, dict) else None)
              or _legacy_dependency(kind) or dict(unknown)
        for kind in ("blas", "lapack")
    }
    runtime = None
    if any(item["implementation"] == "unknown" for item in detected.values()):
        runtime = _runtime_dependency(architecture)
    if runtime is not None:
        for kind, item in detected.items():
            if item["implementation"] == "unknown":
                detected[kind] = dict(runtime)
    return detected


def _validated_numerical_component(component, path):
    _fields(component, _NUMERICAL_COMPONENT_FIELDS, path)
    out = {}
    for name in ("implementation", "version", "integer_api", "architecture"):
        value = component[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}.{name} must be a non-empty string")
        out[name] = value.strip()
    out["implementation"] = out["implementation"].lower()
    out["integer_api"] = out["integer_api"].lower()
    out["architecture"] = _normalised_architecture(out["architecture"])
    if out["integer_api"] not in _INTEGER_APIS:
        raise ValueError(
            f"{path}.integer_api must be one of {sorted(_INTEGER_APIS)}")
    return out


def _validated_numerical_environment(environment):
    _fields(
        environment, _NUMERICAL_ENVIRONMENT_FIELDS,
        "spec.numerical_environment")
    numpy_version = environment["numpy_version"]
    if not isinstance(numpy_version, str) or not numpy_version.strip():
        raise ValueError(
            "spec.numerical_environment.numpy_version must be a non-empty "
            "string")
    backend = environment["backend"]
    _fields(
        backend, _NUMERICAL_BACKEND_FIELDS,
        "spec.numerical_environment.backend")
    return {
        "numpy_version": numpy_version.strip(),
        "backend": {
            kind: _validated_numerical_component(
                backend[kind], f"spec.numerical_environment.backend.{kind}")
            for kind in ("blas", "lapack")
        },
    }


def _screen_integer(params, name, minimum, path, maximum=None):
    value = params[name]
    if (isinstance(value, bool) or not isinstance(value, int)
            or value < minimum or (maximum is not None and value > maximum)):
        bound = (f" in [{minimum}, {maximum}]" if maximum is not None
                 else f" >= {minimum}")
        raise ValueError(f"{path}.{name} must be an integer{bound}")
    return int(value)


def _validated_screen_params(params, path):
    _fields(params, _SCREEN_PARAM_FIELDS, path)
    rounds = _screen_integer(params, "rounds", 1, path)
    window = _screen_integer(params, "window", 50, path)
    seed = _screen_integer(
        params, "seed", 0, path, maximum=np.iinfo(np.uint32).max)
    ncores = _screen_integer(params, "ncores", 1, path)
    for name in ("threshold", "eigenvalue_floor"):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}.{name} must be a finite number")
        try:
            value = float(value)
        except (OverflowError, ValueError):
            raise ValueError(f"{path}.{name} must be a finite number") \
                from None
        if not math.isfinite(value):
            raise ValueError(f"{path}.{name} must be a finite number")
        if name == "threshold" and value <= 0.0:
            raise ValueError(f"{path}.threshold must be > 0")
        if name == "eigenvalue_floor" and not 0.0 <= value < 1.0:
            raise ValueError(
                f"{path}.eigenvalue_floor must be in [0, 1)")
        params[name] = value
    if not isinstance(params["verbose"], bool):
        raise ValueError(f"{path}.verbose must be boolean")
    return {
        "rounds": rounds,
        "window": window,
        "threshold": params["threshold"],
        "eigenvalue_floor": params["eigenvalue_floor"],
        "seed": seed,
        "ncores": ncores,
        "verbose": params["verbose"],
    }


def _validated_spec(spec):
    spec = _plain(spec, "spec")
    _fields(spec, _SPEC_FIELDS, "spec")
    if spec["spec_schema"] != SPEC_SCHEMA:
        raise ValueError(f"spec.spec_schema must be {SPEC_SCHEMA!r}")
    logical = _sha(spec["logical_input_sha256"],
                   "spec.logical_input_sha256")
    ld_sha = _sha(spec["ld_sha256"], "spec.ld_sha256")
    n_semantics = spec["n_semantics"]
    if not isinstance(n_semantics, dict) or not n_semantics:
        raise ValueError("spec.n_semantics must be a non-empty object")
    columns = spec["columns"]
    if not isinstance(columns, dict):
        raise ValueError("spec.columns must be an object")
    qc = spec["qc"]
    _fields(qc, _QC_FIELDS, "spec.qc")
    if not isinstance(qc["enabled"], bool):
        raise ValueError("spec.qc.enabled must be boolean")
    if not isinstance(qc["params"], dict):
        raise ValueError("spec.qc.params must be an object")
    screen = spec["screen"]
    _fields(screen, _SCREEN_FIELDS, "spec.screen")
    if screen["enabled"] is not True:
        raise ValueError("spec.screen.enabled must be true")
    if not isinstance(screen["params"], dict):
        raise ValueError("spec.screen.params must be an object")
    screen_params = _validated_screen_params(
        screen["params"], "spec.screen.params")
    algorithm = spec["algorithm_schema"]
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("spec.algorithm_schema must be a non-empty string")
    versions = spec["versions"]
    _fields(versions, _VERSION_FIELDS, "spec.versions")
    for name in sorted(_VERSION_FIELDS):
        if not isinstance(versions[name], str) or not versions[name]:
            raise ValueError(f"spec.versions.{name} must be a non-empty string")
    numerical_environment = _validated_numerical_environment(
        spec["numerical_environment"])
    return {
        "spec_schema": SPEC_SCHEMA,
        "logical_input_sha256": logical,
        "ld_sha256": ld_sha,
        "n_semantics": n_semantics,
        "columns": columns,
        "qc": {"enabled": qc["enabled"], "params": qc["params"]},
        "screen": {"enabled": True, "params": screen_params},
        "algorithm_schema": algorithm,
        "versions": {
            "bipred": versions["bipred"],
            "ldpred3": versions["ldpred3"],
        },
        "numerical_environment": numerical_environment,
    }


def semantic_spec(*, logical_input_sha256, ld_sha256, n_semantics,
                  columns=None, qc=True, qc_params=None,
                  screen=True, screen_params=None,
                  algorithm_schema=DEFAULT_ALGORITHM_SCHEMA,
                  bipred_version=None, ldpred3_version=None,
                  numpy_version=None, numerical_backend=None):
    """Return the complete canonical identity of trait preparation.

    ``n_semantics`` is the caller's resolved description, for example a
    scalar override or the chosen per-variant-N column and fallback policy.
    The mandatory screen is part of the identity because the stored trait is
    already screened. ``screen_params`` records all fixed controls, including
    the seed. The NumPy version and stable BLAS/LAPACK identity are detected by
    default; explicit values support reproducible tests and migrations. The
    CPU architecture is retained because it can change eigensolver results;
    paths, CPU dispatch flags and thread counts are deliberately absent. Labels,
    counterpart-trait identity and fit options are absent by construction and
    are rejected as unknown top-level fields by :func:`key_for`.
    """
    if not isinstance(qc, bool):
        raise ValueError("qc must be boolean")
    if screen is not True:
        raise ValueError("screen must be true for prepared-trait persistence")
    candidate = {
        "spec_schema": SPEC_SCHEMA,
        "logical_input_sha256": logical_input_sha256,
        "ld_sha256": ld_sha256,
        "n_semantics": {} if n_semantics is None else n_semantics,
        "columns": {} if columns is None else columns,
        "qc": {
            "enabled": qc,
            "params": {} if qc_params is None else qc_params,
        },
        "screen": {
            "enabled": True,
            "params": (dict(DEFAULT_SCREEN_PARAMS) if screen_params is None
                       else screen_params),
        },
        "algorithm_schema": algorithm_schema,
        "versions": {
            "bipred": bipred.__version__ if bipred_version is None
                      else bipred_version,
            "ldpred3": ldpred3.__version__ if ldpred3_version is None
                       else ldpred3_version,
        },
        "numerical_environment": {
            "numpy_version": (np.__version__ if numpy_version is None
                              else numpy_version),
            "backend": (_detected_numerical_backend()
                        if numerical_backend is None else numerical_backend),
        },
    }
    return _validated_spec(candidate)


def _canonical_bytes(spec):
    return json.dumps(spec, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def key_for(spec) -> str:
    """SHA-256 key for a complete semantic specification."""
    return hashlib.sha256(_canonical_bytes(_validated_spec(spec))).hexdigest()


def store_dir(root: Path) -> Path:
    out = Path(root) / STORE_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass(frozen=True)
class _Paths:
    data: Path
    meta: Path
    lock: Path
    used: Path


def _paths(root: Path, key: str) -> _Paths:
    if not _SHA256.fullmatch(key):
        raise ValueError("prepared-store key is not a SHA-256 digest")
    base = store_dir(root)
    return _Paths(
        data=base / f"{key}.npz",
        meta=base / f"{key}.json",
        lock=base / f"{key}.lock",
        used=base / f"{key}.used",
    )


def _stat_identity(stat):
    return stat.st_dev, stat.st_ino


class _Lock:
    """Heartbeated O_EXCL lock whose owner cannot unlink a successor."""

    def __init__(self, path):
        self.path = Path(path)
        self.fd = None
        self.identity = None

    def acquire(self):
        for attempt in (0, 1):
            try:
                self.fd = os.open(
                    str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644)
                stat = os.fstat(self.fd)
                self.identity = _stat_identity(stat)
                os.write(self.fd, f"{os.getpid()}\n".encode("ascii"))
                return True
            except FileExistsError:
                if attempt or not self._steal_if_stale():
                    return False
        return False

    def _steal_if_stale(self):
        try:
            first = self.path.stat()
            if time.time() - first.st_mtime <= LOCK_STALE:
                return False
            second = self.path.stat()
            if (_stat_identity(first) != _stat_identity(second)
                    or first.st_mtime_ns != second.st_mtime_ns):
                return False
            self.path.unlink()
            return True
        except OSError:
            return False

    def touch(self):
        if self.fd is None:
            return
        try:
            os.utime(self.fd, None)
        except OSError:
            pass

    def owned(self):
        if self.fd is None:
            return False
        try:
            return _stat_identity(self.path.stat()) == self.identity
        except OSError:
            return False

    def release(self):
        if self.fd is None:
            return
        try:
            try:
                if _stat_identity(self.path.stat()) == self.identity:
                    self.path.unlink()
            except OSError:
                pass
        finally:
            os.close(self.fd)
            self.fd = None
            self.identity = None


class _Heartbeat:
    def __init__(self, lock):
        self.lock = lock
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        def beat():
            while not self.stop.wait(LOCK_TOUCH):
                self.lock.touch()
        self.thread = threading.Thread(target=beat, daemon=True,
                                       name="bipred-prepared-store-lock")
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        self.thread.join(timeout=min(1.0, LOCK_TOUCH))
        return False


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_screen_log(log, spec, n_rows):
    if log.get("screen") is not True:
        raise ValueError("PreparedTrait.log.screen must be true")
    record = log.get("ld_consistency_screen")
    _fields(
        record, _SCREEN_RECORD_FIELDS,
        "PreparedTrait.log.ld_consistency_screen")
    counts = {}
    for name in ("n_input", "n_kept", "n_dropped"):
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"PreparedTrait.log.ld_consistency_screen.{name} must be a "
                "non-negative integer")
        counts[name] = int(value)
    if counts["n_input"] != counts["n_kept"] + counts["n_dropped"]:
        raise ValueError(
            "PreparedTrait.log.ld_consistency_screen counts are incoherent")
    if counts["n_kept"] != n_rows:
        raise ValueError(
            "PreparedTrait.log.ld_consistency_screen.n_kept does not match "
            "the prepared arrays")
    parameters = _validated_screen_params(
        record["parameters"],
        "PreparedTrait.log.ld_consistency_screen.parameters")
    if parameters != spec["screen"]["params"]:
        raise ValueError(
            "PreparedTrait.log.ld_consistency_screen parameters do not match "
            "the semantic specification")


def _normalise_trait(trait, label, spec):
    if not isinstance(trait, PreparedTrait):
        raise TypeError("prepared-store builder must return PreparedTrait")
    if (isinstance(trait.n_cache, (bool, np.bool_))
            or not isinstance(trait.n_cache, (int, np.integer))):
        raise ValueError("PreparedTrait.n_cache must be an integer")
    n_cache = int(trait.n_cache)
    if n_cache < 0:
        raise ValueError("PreparedTrait.n_cache must be non-negative")
    indices = np.asarray(trait.indices)
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or np.issubdtype(indices.dtype, np.bool_)):
        raise ValueError("PreparedTrait.indices must be a 1D integer array")
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    if len(indices) and (indices[0] < 0 or indices[-1] >= n_cache):
        raise ValueError("PreparedTrait.indices lie outside n_cache")
    if len(indices) > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError("PreparedTrait.indices must be strictly increasing")

    arrays = {}
    for name in ("beta_hat", "n_eff", "z", "eaf"):
        try:
            values = np.ascontiguousarray(getattr(trait, name), dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"PreparedTrait.{name} must be numeric") from None
        if values.ndim != 1 or len(values) != len(indices):
            raise ValueError(
                f"PreparedTrait.{name} must be 1D and aligned with indices")
        arrays[name] = values
    for name in ("beta_hat", "n_eff", "z"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"PreparedTrait.{name} must be finite")
    if np.any(arrays["n_eff"] <= 0):
        raise ValueError("PreparedTrait.n_eff must be positive")
    eaf = arrays["eaf"]
    if np.any(np.isinf(eaf)):
        raise ValueError("PreparedTrait.eaf must be finite or NaN")
    finite_eaf = np.isfinite(eaf)
    if np.any((eaf[finite_eaf] < 0) | (eaf[finite_eaf] > 1)):
        raise ValueError("PreparedTrait.eaf finite values must lie in [0, 1]")
    log = _plain(trait.log, "PreparedTrait.log")
    if not isinstance(log, dict):
        raise ValueError("PreparedTrait.log must be an object")
    log = dict(log)
    _validated_screen_log(log, spec, len(indices))
    log["label"] = label
    return PreparedTrait(
        indices=indices, beta_hat=arrays["beta_hat"],
        n_eff=arrays["n_eff"], z=arrays["z"], eaf=eaf,
        n_cache=n_cache, log=log)


def _warning_records(caught):
    return [{
        "message": str(item.message),
        "category": item.category.__name__,
        "module": item.category.__module__,
    } for item in caught]


def _validated_warnings(records):
    if not isinstance(records, list):
        raise ValueError("metadata warnings must be a list")
    out = []
    for i, record in enumerate(records):
        _fields(record, _WARNING_FIELDS, f"metadata.warnings[{i}]")
        if not all(isinstance(record[name], str) for name in _WARNING_FIELDS):
            raise ValueError("metadata warning fields must be strings")
        out.append(dict(record))
    return out


def _replay(records):
    for record in records:
        category = UserWarning
        module = (builtins if record["module"] == "builtins" else
                  sys.modules.get(record["module"]))
        if module is not None:
            candidate = vars(module).get(record["category"])
            if isinstance(candidate, type) and issubclass(candidate, Warning):
                category = candidate
        warnings.warn(record["message"], category, stacklevel=3)


def _array_manifest(arrays):
    return {
        name: {"dtype": arrays[name].dtype.str,
               "shape": list(arrays[name].shape)}
        for name in _ARRAY_NAMES
    }


def _write_npz(path, trait):
    arrays = {
        "indices": trait.indices,
        "beta_hat": trait.beta_hat,
        "n_eff": trait.n_eff,
        "z": trait.z,
        "eaf": trait.eaf,
        "n_cache": np.asarray(trait.n_cache, dtype=np.int64),
    }
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    return arrays


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def _temp(path):
    return path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part")


def _replace_json(path, payload):
    tmp = _temp(path)
    try:
        _write_json(tmp, payload)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _publish(paths, key, spec, trait, warning_records):
    data_tmp, meta_tmp = _temp(paths.data), _temp(paths.meta)
    now = time.time()
    try:
        arrays = _write_npz(data_tmp, trait)
        checksum = _sha256_file(data_tmp)
        metadata = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "key": key,
            "spec": spec,
            "npz_sha256": checksum,
            "npz_bytes": data_tmp.stat().st_size,
            "created": now,
            "last_used": now,
            "arrays": _array_manifest(arrays),
            "log": trait.log,
            "warnings": warning_records,
        }
        _write_json(meta_tmp, metadata)
        os.replace(data_tmp, paths.data)
        os.replace(meta_tmp, paths.meta)
        return metadata
    finally:
        for tmp in (data_tmp, meta_tmp):
            try:
                tmp.unlink()
            except OSError:
                pass


def _read_checked(paths, key, spec, label):
    metadata = json.loads(paths.meta.read_text(encoding="utf-8"))
    _fields(metadata, _META_FIELDS, "metadata")
    if (not isinstance(metadata["format"], str)
            or metadata["format"] != FORMAT
            or isinstance(metadata["format_version"], bool)
            or not isinstance(metadata["format_version"], int)
            or metadata["format_version"] != FORMAT_VERSION):
        raise ValueError("prepared artifact has an unsupported format")
    stored_spec = _validated_spec(metadata["spec"])
    if (not isinstance(metadata["key"], str) or metadata["key"] != key
            or key_for(stored_spec) != key
            or _canonical_bytes(stored_spec) != _canonical_bytes(spec)):
        raise ValueError("prepared artifact semantic identity does not match")
    if _sha(metadata["npz_sha256"], "metadata.npz_sha256") \
            != metadata["npz_sha256"]:
        raise ValueError("metadata checksum is not canonical lowercase hex")
    if (isinstance(metadata["npz_bytes"], bool)
            or not isinstance(metadata["npz_bytes"], int)
            or metadata["npz_bytes"] < 0):
        raise ValueError("metadata.npz_bytes must be a non-negative integer")
    if paths.data.stat().st_size != metadata["npz_bytes"]:
        raise ValueError("prepared NPZ byte count does not match metadata")
    if _sha256_file(paths.data) != metadata["npz_sha256"]:
        raise ValueError("prepared NPZ checksum does not match metadata")
    for name in ("created", "last_used"):
        value = metadata[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise ValueError(f"metadata.{name} must be finite")
    if not isinstance(metadata["arrays"], dict) \
            or set(metadata["arrays"]) != set(_ARRAY_NAMES):
        raise ValueError("metadata arrays do not match PreparedTrait fields")
    if not isinstance(metadata["log"], dict):
        raise ValueError("metadata.log must be an object")
    warning_records = _validated_warnings(metadata["warnings"])

    with np.load(paths.data, allow_pickle=False) as archive:
        if set(archive.files) != set(_ARRAY_NAMES):
            raise ValueError("prepared NPZ has unexpected or missing arrays")
        arrays = {name: np.array(archive[name], copy=True)
                  for name in _ARRAY_NAMES}
    for name, array in arrays.items():
        manifest = metadata["arrays"].get(name)
        _fields(manifest, {"dtype", "shape"},
                f"metadata.arrays.{name}")
        if manifest["dtype"] != array.dtype.str \
                or manifest["shape"] != list(array.shape):
            raise ValueError(f"prepared array {name} does not match metadata")
    n_cache_array = arrays.pop("n_cache")
    if n_cache_array.shape != () \
            or not np.issubdtype(n_cache_array.dtype, np.integer) \
            or np.issubdtype(n_cache_array.dtype, np.bool_):
        raise ValueError("prepared n_cache must be an integer scalar")
    trait = PreparedTrait(
        n_cache=int(n_cache_array.item()), log=dict(metadata["log"]),
        **arrays)
    return _normalise_trait(trait, label, spec), metadata, warning_records


_LOAD_ERRORS = (
    OSError, ValueError, TypeError, KeyError, EOFError, UnicodeError,
    zipfile.BadZipFile, zipfile.LargeZipFile,
)


def _load(paths, key, spec, label):
    if not paths.data.exists() or not paths.meta.exists():
        return None
    try:
        return _read_checked(paths, key, spec, label)
    except _LOAD_ERRORS:
        return None


def _touch_used(paths):
    try:
        paths.used.touch(exist_ok=True)
    except OSError:
        pass


def _record_use(paths, key):
    """Keep LRU exact even if the metadata lock is briefly unavailable."""
    _touch_used(paths)
    lock = _Lock(paths.lock)
    if not lock.acquire():
        return
    try:
        try:
            metadata = json.loads(paths.meta.read_text(encoding="utf-8"))
            if metadata.get("key") != key:
                return
            metadata["last_used"] = time.time()
            _replace_json(paths.meta, metadata)
        except (OSError, ValueError, TypeError):
            pass
    finally:
        lock.release()


def _discard(paths):
    for path in (paths.data, paths.meta, paths.used):
        try:
            path.unlink()
        except OSError:
            pass


def _built(builder, label, spec):
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trait = builder()
    except BaseException:
        _replay(_warning_records(caught))
        raise
    return _normalise_trait(trait, label, spec), _warning_records(caught)


def get_or_build(root: Path, spec, *, label: str, builder, on_wait=None):
    """Return ``(PreparedTrait, reused)`` for one semantic specification.

    Exactly one same-key caller executes ``builder``.  Warnings emitted by the
    builder are replayed to that caller, persisted, and replayed on every
    reuse.  Corrupt or incomplete artifacts are ignored and rebuilt under the
    same lock.  ``label`` rewrites ``trait.log['label']`` on every return and
    never contributes to the key.
    """
    spec = _validated_spec(spec)
    key = key_for(spec)
    paths = _paths(root, key)
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    if not callable(builder):
        raise TypeError("builder must be callable")
    if on_wait is not None and not callable(on_wait):
        raise TypeError("on_wait must be callable")

    loaded = _load(paths, key, spec, label)
    if loaded is not None:
        trait, _, warning_records = loaded
        _record_use(paths, key)
        _replay(warning_records)
        return trait, True

    lock = _Lock(paths.lock)
    start = time.monotonic()
    while not lock.acquire():
        loaded = _load(paths, key, spec, label)
        if loaded is not None:
            trait, _, warning_records = loaded
            _record_use(paths, key)
            _replay(warning_records)
            return trait, True
        elapsed = time.monotonic() - start
        if elapsed >= WAIT_LIMIT:
            raise TimeoutError("timed out waiting for prepared-trait cache")
        if on_wait is not None:
            on_wait(round(elapsed))
        time.sleep(WAIT_POLL)

    try:
        loaded = _load(paths, key, spec, label)
        if loaded is not None:
            trait, metadata, warning_records = loaded
            metadata["last_used"] = time.time()
            _replace_json(paths.meta, metadata)
            reused = True
        else:
            _discard(paths)
            with _Heartbeat(lock):
                trait, warning_records = _built(builder, label, spec)
                if not lock.owned():
                    raise RuntimeError(
                        "lost the prepared-trait cache lock while building; "
                        "retry the job")
                _publish(paths, key, spec, trait, warning_records)
            reused = False
        _touch_used(paths)
    finally:
        lock.release()
    _replay(warning_records)
    return trait, reused


def _last_used(paths):
    last = 0.0
    try:
        last = max(last, paths.used.stat().st_mtime)
    except OSError:
        pass
    try:
        metadata = json.loads(paths.meta.read_text(encoding="utf-8"))
        value = metadata.get("last_used", 0.0)
        if not isinstance(value, bool) and math.isfinite(float(value)):
            last = max(last, float(value))
    except (OSError, ValueError, TypeError):
        pass
    return last


def _artifact_size(paths):
    total = 0
    for path in (paths.data, paths.meta, paths.used):
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def purge(root: Path, budget_gb: float) -> list[str]:
    """Evict least-recently-used artifacts until ``budget_gb`` is met."""
    if isinstance(budget_gb, bool):
        raise ValueError("budget_gb must be a finite number")
    try:
        budget_gb = float(budget_gb)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("budget_gb must be a finite number") from None
    if not math.isfinite(budget_gb):
        raise ValueError("budget_gb must be a finite number")
    base = store_dir(root)
    now = time.time()
    for part in base.glob("*.part"):
        try:
            if now - part.stat().st_mtime > PART_STALE:
                part.unlink()
        except OSError:
            pass
    if budget_gb <= 0:
        return []

    keys = set()
    for suffix in ("*.npz", "*.json", "*.used"):
        for path in base.glob(suffix):
            key = path.name.split(".", 1)[0]
            if _SHA256.fullmatch(key):
                keys.add(key)
    entries = []
    total = 0
    for key in keys:
        paths = _paths(root, key)
        size = _artifact_size(paths)
        total += size
        entries.append((_last_used(paths), key, paths, size))
    budget = budget_gb * 2 ** 30
    removed = []
    for last, key, paths, size in sorted(entries):
        if total <= budget:
            break
        if now - last <= EVICT_GRACE:
            continue
        lock = _Lock(paths.lock)
        if not lock.acquire():
            continue
        try:
            current_last = _last_used(paths)
            if time.time() - current_last <= EVICT_GRACE:
                continue
            current_size = _artifact_size(paths)
            _discard(paths)
            total -= current_size
            removed.append(key)
        finally:
            lock.release()
    return removed
