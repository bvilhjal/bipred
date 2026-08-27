"""Read the sibling LDpred3 GWAS-Catalog compatibility evidence.

The web application is deliberately checkout-only, and bipred already depends
on the sibling LDpred3 checkout for its real LD reference.  Reading the
benchmark registry in place keeps this page tied to the canonical, hashed
evidence instead of growing a second hand-maintained phenotype list.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

try:                                      # Python 3.11+
    import tomllib
except ImportError:                       # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARKS = REPO_ROOT.parent / "ldpred3" / "benchmarks"
TABLE = "real_gwas_pipeline_catalog.csv"
MANIFEST = "real_gwas_pipeline_catalog.manifest.json"
REGISTRY = "gwas_catalog_traits.toml"


def _unavailable(base: Path, *, missing=None, error=None, table_sha=None,
                 expected_sha=None) -> dict:
    """Return one stable, fail-closed shape for untrusted evidence."""
    return {
        "available": False,
        "trusted": False,
        "path": str(base),
        "missing": list(missing or []),
        "error": error,
        "good": [],
        "bad": [],
        "counts": {"good": 0, "bad": 0, "preflight_bad": 0,
                   "failed_fit": 0},
        "table_sha256": table_sha,
        "expected_table_sha256": expected_sha,
        "table_hash_verified": False,
        "registry_sha256": None,
        "manifest_settings": {},
        "known_limits": [],
        "current_traits": 0,
        "current_completed": 0,
        "legacy_completed": 0,
        "runtime": {},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value, cast=float):
    if value in (None, ""):
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _benchmarks_dir() -> Path:
    return Path(os.environ.get("BIPRED_WEB_LDPRED3_BENCHMARKS",
                               DEFAULT_BENCHMARKS)).resolve()


def load() -> dict:
    """Return display-ready good/bad accessions and their run provenance."""
    base = _benchmarks_dir()
    paths = {name: base / name for name in (TABLE, MANIFEST, REGISTRY)}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return _unavailable(base, missing=missing)

    try:
        with open(paths[MANIFEST], encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _unavailable(base, error=f"Manifest could not be read: {exc}")
    if not isinstance(manifest, dict):
        return _unavailable(base, error="Manifest must be a JSON object")
    row_source = manifest.get("row_source", {})
    if not isinstance(row_source, dict):
        return _unavailable(
            base, error="Manifest row_source must be a JSON object")

    try:
        table_sha = _sha256(paths[TABLE])
    except OSError as exc:
        return _unavailable(base, error=f"Canonical table could not be read: {exc}")
    expected_sha = manifest.get("table_sha256")
    if not isinstance(expected_sha, str) or table_sha != expected_sha.lower():
        return _unavailable(
            base, table_sha=table_sha, expected_sha=expected_sha,
            error=("Canonical table SHA-256 does not match its manifest; "
                   "canonical rows were quarantined."))

    try:
        with open(paths[REGISTRY], "rb") as fh:
            registry_document = tomllib.load(fh)
            registry = registry_document.get("traits", {})
            if (not isinstance(registry, dict)
                    or any(not isinstance(value, dict)
                           for value in registry.values())):
                raise ValueError("registry traits must be a TOML table")
        with open(paths[TABLE], encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, UnicodeError, ValueError, csv.Error,
            tomllib.TOMLDecodeError) as exc:
        return _unavailable(
            base, table_sha=table_sha, expected_sha=expected_sha,
            error=f"Verified evidence files could not be parsed: {exc}")

    current = {name for name, source in row_source.items()
               if isinstance(source, str)
               and "catalog-current-profile" in source}
    good = []
    failed_fit = []
    for row in rows:
        trait_meta = registry.get(row.get("trait"), {})
        entry = {
            "accession": row.get("accession"),
            "trait": (trait_meta.get("trait") or
                      row.get("trait", "").replace("_", " ")),
            "source": row.get("source"),
            "pmid": row.get("pmid"),
            "n_eff": _number(row.get("n_eff_value")),
            "n_cases": _number(row.get("n_case"), int),
            "n_controls": _number(row.get("n_control"), int),
            "n_final": _number(row.get("n_final"), int),
            "total_s": _number(row.get("total_s")),
            "peak_gb": _number(row.get("driver_peak_gb")),
            "n_chains": _number(row.get("n_chains"), int),
            "n_chains_kept": _number(row.get("n_chains_kept"), int),
            "burn_in": _number(row.get("infer_burn_in"), int),
            "num_iter": _number(row.get("infer_num_iter"), int),
            "ncores": _number(row.get("ncores"), int),
            "ldpred3_version": row.get("ldpred3_version"),
            "cohort_id": row.get("cohort_id"),
            "profile": "current" if row.get("trait") in current else "legacy",
            "reason": row.get("note"),
            "evidence": "completed LDpred3 end-to-end fit",
        }
        if row.get("status") == "ok":
            good.append(entry)
        else:
            entry["evidence"] = "LDpred3 end-to-end fit failed"
            failed_fit.append(entry)

    unusable = []
    for trait in registry.values():
        if trait.get("usable", True):
            continue
        unusable.append({
            "accession": trait.get("accession"),
            "trait": trait.get("trait"),
            "source": trait.get("source"),
            "pmid": trait.get("pmid"),
            "n_eff": trait.get("n_effective"),
            "n_cases": trait.get("n_case"),
            "n_controls": trait.get("n_control"),
            "n_final": trait.get("n_hm3_variants"),
            "reason": trait.get("unusable_reason"),
            "profile": "preflight",
            "evidence": "rejected by LDpred3 input preflight",
        })

    runtime_candidates = []
    for source in row_source.values():
        if (not isinstance(source, str)
                or "catalog-current-profile" not in source):
            continue
        relative = Path(source)
        if relative.parts and relative.parts[0] == "benchmarks":
            relative = Path(*relative.parts[1:])
        runtime_candidates.extend((base / relative).parent.glob(
            "*.runtime.json"))
        break
    runtime_path = sorted(set(runtime_candidates))[0] \
        if runtime_candidates else base / "missing.runtime.json"
    runtime = {}
    if runtime_path.exists():
        try:
            with open(runtime_path, encoding="utf-8") as fh:
                sidecar = json.load(fh)
        except (OSError, UnicodeError, json.JSONDecodeError):
            sidecar = {}
        if sidecar:
            rt = sidecar.get("runtime", {})
            env = sidecar.get("environment", {})
            runtime = {
                "host": "Apple M2 Pro, arm64, 10 CPU cores, 16 GiB RAM",
                "platform": rt.get("platform"),
                "python": (rt.get("python") or "").split(" | ", 1)[0],
                "numpy": rt.get("numpy"), "scipy": rt.get("scipy"),
                "numba": rt.get("numba"), "llvmlite": rt.get("llvmlite"),
                "backend": "Apple Accelerate",
                "numba_threads": rt.get("numba_threads"),
                "sweep_threads": env.get("LDPRED3_SWEEP_NCORES"),
                "outer_workers": "up to 8 (ncores setting 8 or 10)",
                "source_commit": sidecar.get("source", {}).get("git_commit"),
                "source_diff_sha256": sidecar.get("source", {}).get(
                    "git_diff_sha256"),
                "ld_cache_sha256": sidecar.get("inputs", {}).get(
                    "ld_cache", {}).get("sha256"),
            }

    registry_sha = _sha256(paths[REGISTRY])
    return {
        "available": True, "trusted": True, "path": str(base),
        "good": sorted(good, key=lambda e: (e["trait"] or "").lower()),
        "bad": sorted(unusable + failed_fit,
                      key=lambda e: (e["trait"] or "").lower()),
        "counts": {"good": len(good), "bad": len(unusable + failed_fit),
                   "preflight_bad": len(unusable),
                   "failed_fit": len(failed_fit)},
        "table_sha256": table_sha,
        "expected_table_sha256": expected_sha,
        "table_hash_verified": True,
        "registry_sha256": registry_sha,
        "manifest_settings": manifest.get("settings", {}),
        "known_limits": manifest.get("known_limits", []),
        "current_traits": len(current),
        "current_completed": sum(e["profile"] == "current" for e in good),
        "legacy_completed": sum(e["profile"] == "legacy" for e in good),
        "runtime": runtime,
    }
