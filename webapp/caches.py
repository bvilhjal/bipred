"""LD-cache registry for the bipred web service.

The default is the **UK Biobank European HapMap3** reference when it is present
at its conventional workspace location (the bigsnpr Figshare LD reference,
converted by ``ldpred3/benchmarks/convert_bigsnpr_ldref.py``; the real-data
benchmarks use the same file). More real caches can be registered through the
``BIPRED_WEB_CACHES`` environment variable as ``name=/path/cache.ld.npz``
entries separated by ``;`` — build them once with
``ldpred3.compute_ld_blocks(..., quantize=True)`` + ``ldpred3.save_ld_blocks``
and mount them read-only on the worker host. A synthetic *demo* cache (built
on first use by :mod:`webapp.demo`) is always listed last.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from . import jobs

REPO_ROOT = Path(__file__).resolve().parent.parent

# Conventional location of the converted UKB European HapMap3 cache in this
# workspace (sibling ldpred3 checkout's benchmark work dir).
UKB_EUR_KEY = "ukb-eur-hm3"
UKB_EUR_PATH = (REPO_ROOT.parent / "ldpred3" / "benchmarks" / ".work"
                / "ldref-hm3" / "ldpred3_ldref_hm3.npz")
UKB_EUR_LABEL = "UK Biobank European (HapMap3, 1.05M variants, n=362k)"


def demo_cache_dir(root: Path | None = None) -> Path:
    root = root or jobs.data_root()
    out = root / "caches" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    return out


def registry(root: Path | None = None) -> list[dict]:
    """Available caches as ``{key, label, path}`` dicts; real caches first."""
    out = []
    if UKB_EUR_PATH.exists():
        out.append({"key": UKB_EUR_KEY, "label": UKB_EUR_LABEL,
                    "path": str(UKB_EUR_PATH)})
    for entry in os.environ.get("BIPRED_WEB_CACHES", "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"BIPRED_WEB_CACHES entry {entry!r} needs name=/path/cache.npz")
        name, path = (part.strip() for part in entry.split("=", 1))
        if not name or not Path(path).exists():
            raise ValueError(
                f"BIPRED_WEB_CACHES entry {entry!r}: file does not exist")
        out.append({"key": name, "label": name, "path": path})
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


def sha256_cached(path: Path) -> str:
    """Content hash of a cache, memoized in a sidecar (caches are GBs)."""
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists():
        return sidecar.read_text().strip()
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    try:
        sidecar.write_text(value + "\n")
    except OSError:
        pass                                # read-only mount: report anyway
    return value
