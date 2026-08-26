"""Input validation and provenance for manual real-data benchmarks."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata


HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "real_data_inputs.sha256")
REPO_ROOT = os.path.dirname(HERE)
# The ldpred3 the benchmark artifacts were generated against. Moving this
# pin re-bases the record: numbers may shift for ldpred3 reasons as well as
# bipred ones, so regenerate every artifact in the same sweep rather than
# splicing a new pin's rows into an old pin's table.
LDPRED3_REV = "af5d92c7aab6a5b67d15c94ebe28b89e33f5d69d"
LDPRED3_VERSION = "0.6.1"


def load_manifest(path=MANIFEST):
    """Return ``{logical relative name: sha256}`` from the committed manifest."""
    checksums = {}
    with open(path, encoding="ascii") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                digest, name = stripped.split(None, 1)
            except ValueError:
                raise ValueError(f"malformed checksum line {line_no} in {path}") from None
            name = name.lstrip("*")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"invalid SHA-256 on line {line_no} in {path}")
            if name in checksums:
                raise ValueError(f"duplicate checksum name {name!r} in {path}")
            checksums[name] = digest
    return checksums


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    """Return the SHA-256 digest of one file without loading it in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(root):
    """Hash one installed package tree, excluding generated cache files."""
    root = os.path.realpath(os.fspath(root))
    digest = hashlib.sha256()
    found = 0
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in {".git", "__pycache__"})
        for filename in sorted(filenames):
            if filename.endswith((".pyc", ".pyo", ".nbc", ".nbi")):
                continue
            path = os.path.join(directory, filename)
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
            found += 1
    if not found:
        raise RuntimeError(f"cannot fingerprint empty package tree {root}")
    return digest.hexdigest()


def require_ldpred3_source(*, expected_revision=LDPRED3_REV,
                            expected_version=LDPRED3_VERSION):
    """Return exact ldpred3 source identity; reject a dirty or wrong Git pin.

    A VCS checkout must be clean under the imported package directory and at
    the benchmark pin. A non-VCS installation is identified by a hash of its
    complete package tree; a PEP 610 commit is checked when available.
    """
    module = importlib.import_module("ldpred3")
    package_root = os.path.realpath(os.path.dirname(module.__file__))
    version = getattr(module, "__version__", None)
    version = str(version if version is not None else _module_version("ldpred3"))
    if version != expected_version:
        raise RuntimeError(
            f"benchmarks require ldpred3 {expected_version} at "
            f"{expected_revision}; imported version {version} from "
            f"{package_root}")
    record = {
        "expected_revision": expected_revision,
        "location": package_root,
        "source_tree_sha256": _source_tree_sha256(package_root),
        "version": version,
    }

    probe = subprocess.run(
        ["git", "-C", package_root, "rev-parse", "--show-toplevel"],
        check=False, capture_output=True, text=True)
    if probe.returncode == 0:
        git_root = os.path.realpath(probe.stdout.strip())
        relative = os.path.relpath(package_root, git_root)
        tracked = subprocess.run(
            ["git", "-C", git_root, "ls-files", "--", relative],
            check=True, capture_output=True, text=True)
        if tracked.stdout.strip():
            status = subprocess.run(
                ["git", "-C", git_root, "status", "--porcelain",
                 "--untracked-files=all", "--", relative],
                check=True, capture_output=True, text=True)
            if status.stdout.strip():
                raise RuntimeError(
                    "benchmarks require a clean ldpred3 package tree; "
                    f"imported dirty source from {package_root}")
            revision = subprocess.run(
                ["git", "-C", git_root, "rev-parse", "HEAD"], check=True,
                capture_output=True, text=True).stdout.strip()
            if revision != expected_revision:
                raise RuntimeError(
                    f"benchmarks require ldpred3 revision {expected_revision}; "
                    f"imported {revision} from {package_root}")
            record.update(identity="git", revision=revision, source_clean=True)
            return record

    direct_url = None
    try:
        direct_url = metadata.distribution("ldpred3").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        pass
    revision = None
    if direct_url:
        try:
            revision = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        except (TypeError, ValueError):
            revision = None
    if revision is not None and revision != expected_revision:
        raise RuntimeError(
            f"benchmarks require ldpred3 revision {expected_revision}; "
            f"installed package records {revision}")
    record.update(
        identity="vcs-install" if revision else "installed-tree",
        revision=revision,
        source_clean=None,
    )
    return record


def validate_inputs(paths, *, manifest_path=MANIFEST, verbose=True):
    """Validate ``{manifest name: local path}``; return the computed hashes."""
    expected = load_manifest(manifest_path)
    unknown = sorted(set(paths) - set(expected))
    if unknown:
        raise ValueError(f"inputs absent from checksum manifest: {unknown}")
    started = time.perf_counter()
    observed = {}
    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing real-data input {path}")
        observed[name] = sha256_file(path)
        if observed[name] != expected[name]:
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {expected[name]}, "
                f"got {observed[name]}")
    if verbose:
        print(f"verified {len(paths)} external input checksum(s) "
              f"in {time.perf_counter() - started:.1f}s", flush=True)
    return observed


def require_clean_source(repo_root=REPO_ROOT):
    """Return ``HEAD`` after refusing a dirty or untracked source tree."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root, check=True, capture_output=True, text=True)
    if status.stdout.strip():
        raise RuntimeError(
            "real-data benchmarks require a clean source tree; commit or "
            "remove staged, unstaged, and untracked changes first")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        capture_output=True, text=True).stdout.strip()
    if (len(revision) not in (40, 64)
            or any(char not in "0123456789abcdef" for char in revision)):
        raise RuntimeError(f"could not resolve a full source revision: {revision!r}")
    return revision


def _module_version(name):
    try:
        version = getattr(importlib.import_module(name), "__version__", None)
        return str(version if version is not None else metadata.version(name))
    except (metadata.PackageNotFoundError, ModuleNotFoundError):
        return "unavailable"


def write_provenance_sidecar(csv_path, *, source_revision, input_hashes,
                             dependency_sources, run_controls=None):
    """Write ``<stem>.provenance.json`` beside a manual benchmark CSV."""
    csv_path = os.path.abspath(os.fspath(csv_path))
    stem, _ = os.path.splitext(csv_path)
    sidecar = f"{stem}.provenance.json"
    record = {
        "schema_version": 2,
        "artifact": os.path.basename(csv_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": source_revision,
        "source_clean": True,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: _module_version(name)
                     for name in ("bipred", "ldpred3", "numpy", "numba")},
        "dependency_sources": dependency_sources,
        "inputs_sha256": dict(sorted(input_hashes.items())),
        "run_controls": dict(sorted((run_controls or {}).items())),
    }
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return sidecar
