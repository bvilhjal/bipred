"""GWAS Catalog fetch support for the bipred web service.

Two responsibilities, mirroring the two moments they are needed:

* :func:`resolve` runs at *submit* time (in the web process): validate the
  accession, find its harmonised file, and read study metadata (trait name,
  sample size) so the form can confirm what will be downloaded. Answers are
  cached under ``<data root>/_meta/gwascat/`` — the harmonised-file index for
  a week and descriptive study metadata indefinitely. File size and HTTP
  validators are refreshed for every submission.

* :func:`stream_filter` runs at *fit* time (in the runner subprocess): stream
  the harmonised file, keep only variants present in a given variant set,
  and write one normalised TSV.GZ that ``ldpred3.sumstats`` reads with no
  overrides. Raw catalog files run to hundreds of MB and are ~90% variants
  no reference contains, so filtering in the stream is what keeps job
  directories small.

* :func:`fetch_filtered` wraps that for the runner so one observed deposit
  generation is downloaded at most once: a shared per-accession copy under
  ``<data root>/catalog/`` is kept filtered to the union of the registered LD
  references, and each job filters that copy locally. Re-running an analysis
  then avoids transferring the deposit body; submit-time HEAD validation still
  checks for an in-place replacement.

The schema handling — the ``hm_``-prefixed 2015-era layout versus the current
one, an effect carried as beta / odds ratio / z-score with per-row fallback —
is adapted from ``ldpred3/benchmarks/gwas_catalog_harvest.py``, which is not
an installed package and so cannot be imported; the adaptations are marked as
such. The sample-size parsing reuses its European case/control regexes.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import stat as stat_module
import time
import urllib.error
import urllib.request
import uuid
import zlib
from array import array
from pathlib import Path

import numpy as np

from .jobs import ProcessFileLock

FTP = "https://ftp.ebi.ac.uk/pub/databases/gwas"
SUMSTATS = f"{FTP}/summary_statistics"
HARMONISED_LIST = f"{SUMSTATS}/harmonised_list.txt"
REST_STUDY = "https://www.ebi.ac.uk/gwas/rest/api/studies/{accession}"

LIST_TTL = 7 * 86400.0          # re-fetch the harmonised index weekly

ACCESSION = re.compile(r"^GCST\d{3,}$")

_REMOTE_VALIDATOR_FIELDS = ("remote_etag", "remote_last_modified")
GENERATION_ATTEMPTS = 2


def _remote_validator(value) -> str | None:
    """Canonical non-empty HTTP validator value, or ``None`` when absent."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _remote_validators(remote_etag=None, remote_last_modified=None) -> dict:
    """Only validators actually supplied by the Catalog response/caller."""
    values = {
        "remote_etag": _remote_validator(remote_etag),
        "remote_last_modified": _remote_validator(remote_last_modified),
    }
    return {name: value for name, value in values.items()
            if value is not None}


class _GenerationMismatch(Exception):
    """The GET response is not the deposit generation resolved by HEAD."""


def _validated_remote_open(url, remote_etag=None,
                           remote_last_modified=None):
    """Open one GET whose response belongs to the resolved generation.

    Strong ETags and modification dates are also sent as HTTP preconditions.
    Response validators remain mandatory when HEAD supplied them: this catches
    servers or intermediaries that ignore a conditional request. A weak ETag
    cannot be used with ``If-Match``, but is still checked explicitly.
    """
    expected = _remote_validators(remote_etag, remote_last_modified)
    request = urllib.request.Request(url)
    etag = expected.get("remote_etag")
    modified = expected.get("remote_last_modified")
    if etag is not None and not etag.lower().startswith("w/") \
            and etag != "*":
        request.add_header("If-Match", etag)
    if modified is not None:
        request.add_header("If-Unmodified-Since", modified)
    try:
        response = urllib.request.urlopen(request, timeout=900)
    except urllib.error.HTTPError as exc:
        if exc.code == 412:
            raise _GenerationMismatch(
                "the server rejected the resolved HTTP validator") from exc
        raise
    observed = _remote_validators(
        response.headers.get("ETag"), response.headers.get("Last-Modified"))
    mismatches = [
        name for name, value in expected.items()
        if observed.get(name) != value
    ]
    if mismatches:
        try:
            response.close()
        finally:
            labels = ", ".join(name.removeprefix("remote_")
                               for name in mismatches)
            raise _GenerationMismatch(
                f"GET response has a missing or changed {labels}")
    return response, observed


# --- normalised output schema and per-file detection (harvester adaptation) ---

OUT_COLS = ("rsid", "chrom", "pos", "effect_allele", "other_allele",
            "beta", "se", "eaf", "pval", "n")
_ALIASES = {
    "rsid": ("hm_rsid", "rsid", "rs_id", "variant_id"),
    "chrom": ("hm_chrom", "chromosome", "chr"),
    "pos": ("hm_pos", "base_pair_location", "position"),
    "effect_allele": ("hm_effect_allele", "effect_allele"),
    "other_allele": ("hm_other_allele", "other_allele"),
    "beta": ("hm_beta", "beta"),
    "se": ("standard_error", "se", "standarderror"),
    "eaf": ("hm_effect_allele_frequency", "effect_allele_frequency", "eaf"),
    "pval": ("p_value", "pvalue", "p"),
    "n": ("n", "sample_size", "n_total"),
}
# GWAS Catalog's standard ``sample_size`` field is the number contributing to
# that variant, not a case/control effective sample size.  Keep the column so
# its relative missingness pattern can be retained, but label its semantics so
# the runner can put it on an ancestry-matched effective-N scale before fitting.
_TOTAL_N_COLUMNS = frozenset(("sample_size", "n_total"))
_ODDS = ("hm_odds_ratio", "odds_ratio")
# A z-score with a standard error IS an effect: beta = z * se, sign included.
_ZSCORE = ("hm_z_score", "z_score", "zscore", "z")
_REQUIRED = ("rsid", "effect_allele", "other_allele", "se")

# European case/control counts in the catalog's free-text sample size
# (same patterns as the harvester).
_CASES = re.compile(r"([\d,]+)\s+European[^,;]*?cases", re.I)
_CONTROLS = re.compile(r"([\d,]+)\s+European[^,;]*?controls", re.I)
_EUR_INDIVIDUALS = re.compile(r"([\d,]+)\s+European[^,;]*?individuals", re.I)
_INDIVIDUALS = re.compile(r"([\d,]+)\s+[^,;]*?individuals", re.I)


def _sample_metadata_fields(sample: str) -> dict:
    """Conservative, ancestry-labelled sample-size facts from study text."""
    out = {}
    cases = _CASES.search(sample)
    controls = _CONTROLS.search(sample)
    if cases and controls:
        out["n_cases"] = int(cases.group(1).replace(",", ""))
        out["n_controls"] = int(controls.group(1).replace(",", ""))
        out["n_total_selected"] = out["n_cases"] + out["n_controls"]
        out["n_eff"] = round(4.0 / (1.0 / out["n_cases"]
                                    + 1.0 / out["n_controls"]), 1)
        out["n_basis"] = ("4/(1/ncase+1/nctrl) from catalog European "
                          "case/control counts")
        out["sample_size_population"] = "European"
        out["sample_size_design"] = "case_control"
        return out

    eur = [int(match.group(1).replace(",", ""))
           for match in _EUR_INDIVIDUALS.finditer(sample)]
    if eur:
        out.update({
            "n_eff": float(sum(eur)),
            "n_total_selected": int(sum(eur)),
            "n_basis": "catalog European initial sample size",
            "sample_size_population": "European",
            "sample_size_design": "individuals",
            "sample_size_components": eur,
        })
        return out

    numbers = [int(match.group(1).replace(",", ""))
               for match in _INDIVIDUALS.finditer(sample)]
    if numbers:
        out.update({
            "n_total_reported": int(max(numbers)),
            "n_basis": ("largest N in the catalog's initial sample-size "
                        "text; not used automatically because ancestry/design "
                        "are unresolved"),
            "sample_size_population": "unresolved",
            "sample_size_design": "unresolved",
        })
    return out


def _resolve_schema(fieldnames):
    """Map normalised names onto whichever harmonised layout this file uses."""
    have = {}
    for field in (fieldnames or []):
        have.setdefault(field.strip().lower(), field.strip())
    mapping = {}
    for target, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in have:
                mapping[target] = have[alias]
                break
    odds = next((have[o] for o in _ODDS if o in have), None)
    zscore = next((have[z] for z in _ZSCORE if z in have), None)
    missing = [c for c in _REQUIRED if c not in mapping]
    if "beta" not in mapping and odds is None and zscore is None:
        missing.append("beta/odds_ratio/z_score")
    if missing:
        raise ValueError(f"unmappable columns {missing}; header={fieldnames}")
    return mapping, odds, zscore


def _is_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _effect_from_row(row, zscore, se_col, odds):
    """Recover a per-row effect when the beta column is empty/NA."""
    if zscore is not None and _is_number(row.get(zscore)) \
            and _is_number(row.get(se_col)):
        return repr(float(row[zscore]) * float(row[se_col]))
    if odds is not None:
        try:
            value = float(row.get(odds, ""))
        except (TypeError, ValueError):
            return ""
        return "" if value <= 0 else repr(math.log(value))
    return ""


class _HashingReader:
    """Wrap a byte stream: hash bytes as read, count them, report progress.

    ``digest`` may be None when only the byte count is wanted.
    """

    def __init__(self, stream, digest, on_bytes=None):
        self._stream, self._digest = stream, digest
        self._on_bytes = on_bytes
        self.total = 0

    def readable(self):
        return True

    def read(self, size=-1):
        data = self._stream.read(size)
        if self._digest is not None:
            self._digest.update(data)
        self.total += len(data)
        if self._on_bytes is not None:
            self._on_bytes(self.total)      # throttling is the caller's job
        return data


# --- catalog metadata (submit-time, web process) ---

def _meta_dir(root: Path) -> Path:
    out = Path(root) / "_meta" / "gwascat"
    out.mkdir(parents=True, exist_ok=True)
    return out


# --- accession track record -------------------------------------------------
#
# One JSON dict keyed by accession, updated only by the web process (submit
# time for resolution failures, job-reap time for download and per-trait
# stage outcomes) so concurrent runner subprocesses never race it. Structural
# failures (no such study, no harmonised file, dead URL, no usable variants
# after QC/harmonization) are recorded; transient network errors are not.

REGISTRY_NAME = "accessions.json"

_STRUCTURAL = ("no such study", "no harmonised", "harmonised file not found",
               "no variants overlap", "unmappable columns",
               "all gwas variants were removed")


def worth_recording(message: str) -> bool:
    """Is this resolve() failure a property of the accession, not the network?"""
    message = str(message).lower()
    return any(s in message for s in _STRUCTURAL)


def accession_registry(root: Path) -> dict:
    path = _meta_dir(root) / REGISTRY_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def record_accession(root: Path, accession: str, works: bool, **fields) -> None:
    """Merge one accession's outcome into the registry (atomic rewrite).

    A later ``works=True`` upgrades an earlier failure (catalogs do fix
    deposits); a failure never erases a recorded success.
    """
    registry = accession_registry(root)
    entry = registry.get(accession, {})
    if entry.get("works") and not works:
        return
    entry.update(fields)
    entry["works"] = bool(works)
    entry["when"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    registry[accession] = entry
    path = _meta_dir(root) / REGISTRY_NAME
    tmp = str(path) + ".part"
    with open(tmp, "w") as fh:
        json.dump(registry, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _harmonised_paths(root: Path) -> dict:
    """{accession: ftp path} for every study with a harmonised file."""
    path = _meta_dir(root) / "harmonised.txt"
    if not path.exists() or time.time() - path.stat().st_mtime > LIST_TTL:
        tmp = str(path) + ".part"
        with urllib.request.urlopen(HARMONISED_LIST, timeout=300) as resp, \
                open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        os.replace(tmp, path)
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            found = re.search(r"/(GCST\d+)/harmonised/(\S+\.h\.tsv\.gz)",
                              line.strip())
            if found:
                out[found.group(1)] = line.strip().lstrip("./")
    return out


def _study_metadata(accession: str, root: Path) -> dict:
    """Trait name and sample size from the GWAS Catalog REST API (cached)."""
    cache = _meta_dir(root) / f"{accession}.json"
    if cache.exists():
        meta = json.loads(cache.read_text())
        # Re-derive fields when semantics improve: an old cache must not keep
        # the former unsafe unknown-ancestry ``n_eff`` interpretation alive.
        for name in ("n_eff", "n_cases", "n_controls", "n_total_selected",
                     "n_total_reported", "n_basis", "sample_size_population",
                     "sample_size_design", "sample_size_components"):
            meta.pop(name, None)
        meta.update(_sample_metadata_fields(meta.get("sample", "")))
        return meta
    url = REST_STUDY.format(accession=accession)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"{accession}: no such study in the GWAS Catalog")
        raise ValueError(f"{accession}: catalog lookup failed ({exc})")
    except (urllib.error.URLError, OSError) as exc:
        raise ValueError(f"{accession}: catalog lookup failed ({exc})")
    sample = (payload.get("initialSampleSize") or "").strip()
    trait = ((payload.get("diseaseTrait") or {}).get("trait")
             or payload.get("reportedTrait") or accession)
    pub = payload.get("publicationInfo") or {}
    meta = {"accession": accession, "trait": trait.strip(),
            "title": (pub.get("title") or "").strip(),
            "pmid": str(pub.get("pubmedId") or ""), "sample": sample}
    meta.update(_sample_metadata_fields(sample))
    tmp = str(cache) + ".part"
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(tmp, cache)
    return meta


def resolve(accession: str, root: Path) -> dict:
    """Everything the form and the runner need about one accession.

    Raises ``ValueError`` with a user-readable message for anything that
    would make the download fail later: bad format, unknown accession, no
    harmonised file, or a dead URL (the catalog's index and FTP tree disagree
    in places, so the path is verified with a HEAD request).
    """
    accession = (accession or "").strip().upper()
    if not ACCESSION.match(accession):
        raise ValueError(f"{accession!r}: expected a GCST accession like "
                         "GCST90446168")
    try:
        path = _harmonised_paths(root).get(accession)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise ValueError(f"{accession}: catalog index lookup failed ({exc})")
    if path is None:
        raise ValueError(f"{accession}: no harmonised summary-statistics "
                         "file in the GWAS Catalog")
    url = f"{SUMSTATS}/{path}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            remote_bytes = int(resp.headers.get("Content-Length", 0) or 0)
            validators = _remote_validators(
                resp.headers.get("ETag"), resp.headers.get("Last-Modified"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"{accession}: harmonised file not found (404)")
        raise ValueError(f"{accession}: harmonised-file check failed ({exc})")
    except (urllib.error.URLError, OSError) as exc:
        raise ValueError(f"{accession}: harmonised-file check failed ({exc})")
    meta = _study_metadata(accession, root)
    meta["url"] = url
    meta["remote_bytes"] = remote_bytes
    meta.update(validators)
    return meta


# --- download (fit-time, runner subprocess) ---

def cache_ids(cache_path: Path) -> set:
    """Variant ids of the LD reference the job will harmonize against."""
    with np.load(str(cache_path), allow_pickle=False) as cache:
        return set(cache["ids"].tolist())


def stream_filter(url_or_path: str, keep_ids: set, dest: Path,
                  on_bytes=None, *, remote_etag=None,
                  remote_last_modified=None) -> dict:
    """Stream one harmonised file into the normalised layout, keeping only
    ``keep_ids`` variants, and return download provenance.

    ``url_or_path`` may also be a local file path, which is how the test
    suite exercises this without network access. ``on_bytes(n)``, if given,
    is called with the running compressed-byte count as the stream is read —
    the caller throttles what it does with it.
    """
    digest = hashlib.sha256()
    tmp = str(dest) + ".part"
    observed_validators = {}
    if os.path.exists(url_or_path):
        raw = open(url_or_path, "rb")
    else:
        raw, observed_validators = _validated_remote_open(
            url_or_path, remote_etag, remote_last_modified)
    seen = kept = n_usable = 0
    with raw:
        stream = _HashingReader(raw, digest, on_bytes)
        with gzip.open(stream, "rt", encoding="utf-8",
                       errors="replace") as src, \
                gzip.open(tmp, "wt", encoding="utf-8", newline="") as dst:
            reader = csv.DictReader(src, delimiter="\t")
            mapping, odds, zscore = _resolve_schema(reader.fieldnames)
            has_beta = "beta" in mapping
            n_source_column = mapping.get("n")
            n_source_kind = (
                "reported_total" if n_source_column is not None
                and n_source_column.strip().lower() in _TOTAL_N_COLUMNS
                else "variant_n" if n_source_column is not None else "none"
            )
            info = {"schema": ("hm_prefixed" if "hm_rsid" in
                               (reader.fieldnames or []) else "unprefixed"),
                    "has_n": "n" in mapping,
                    "n_source_column": n_source_column,
                    "n_source_kind": n_source_kind,
                    "effect_from": ("beta" if has_beta else
                                    "log(odds_ratio)" if odds is not None
                                    else "z * se")}
            writer = csv.DictWriter(dst, fieldnames=OUT_COLS, delimiter="\t",
                                    extrasaction="ignore",
                                    lineterminator="\n")
            writer.writeheader()
            rsid_col = mapping["rsid"]
            se_col = mapping["se"]
            for row in reader:
                seen += 1
                if row.get(rsid_col) not in keep_ids:
                    continue
                out_row = {t_: row.get(src_col, "")
                           for t_, src_col in mapping.items()}
                if has_beta and not _is_number(out_row.get("beta")):
                    # A beta column can exist and be NA on (part of) the
                    # rows; fall back per row, not per file.
                    out_row["beta"] = _effect_from_row(row, zscore, se_col,
                                                       odds)
                elif not has_beta:
                    # Primary route first (OR for an OR file, z for a z
                    # file), the other as a per-row fallback. Never let a
                    # sign-flip on allele swap negate an OR.
                    out_row["beta"] = _effect_from_row(row, None, se_col,
                                                       odds) \
                        or _effect_from_row(row, zscore, se_col, None)
                if _is_number(out_row.get("n")) \
                        and float(out_row["n"]) > 0:
                    n_usable += 1
                writer.writerow(out_row)
                kept += 1
    if kept == 0:
        os.unlink(tmp)
        raise ValueError("no variants overlap the LD reference — wrong "
                         "genome build or a hits-only deposition?")
    os.replace(tmp, dest)
    n_frac = n_usable / kept
    info.update(seen=seen, kept=kept, sha256=digest.hexdigest(),
                source_bytes=stream.total,
                per_variant_n_usable_frac=n_frac,
                has_per_variant_n=bool(info["has_n"] and n_frac >= 0.99))
    info.update(observed_validators)
    return info


# --- shared download store (fit-time, runner subprocess) --------------------
#
# :func:`stream_filter` writes a file that is specific to one LD reference,
# because the filter runs *during* the download. Re-running an analysis
# therefore used to re-fetch hundreds of megabytes already on disk, and a
# re-run against a different reference could not have reused the old file
# even in principle: it no longer contains the variants the new reference
# needs.
#
# So keep one normalised copy per accession under ``<data root>/catalog/``,
# filtered to the *union* of the LD references registered when it was built,
# and have each job filter that copy locally. Re-running with any covered
# reference then costs no network at all. The stored copy is roughly an
# order of magnitude smaller than the raw harmonised file (ten columns, and
# only variants some reference contains), so this trades much less disk than
# caching the raw download would.
#
# Coverage is keyed on the *content* hash of each LD cache file, not its
# registry name, so re-pointing a name at different bytes rebuilds instead of
# silently filtering against a reference the store never covered.

STORE_DIRNAME = "catalog"
WAIT_LIMIT = 3600.0         # give up waiting on another job's download
EVICT_GRACE = 3600.0        # never evict a copy used this recently
PART_STALE = 3600.0         # abandoned partial downloads older than this go


def _write_json(path: Path, payload: dict) -> None:
    tmp = Path(path).with_name(
        f".{Path(path).name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def store_dir(root: Path) -> Path:
    out = Path(root) / STORE_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def _store_paths(root: Path, accession: str) -> tuple:
    """``(data, meta)`` paths for one accession's stored copy."""
    accession = (accession or "").strip().upper()
    if not ACCESSION.match(accession):
        raise ValueError(f"{accession!r}: not a GCST accession")
    base = store_dir(root)
    return base / f"{accession}.tsv.gz", base / f"{accession}.json"


def _load_build(meta_path: Path, fingerprint: str | None = None) -> dict | None:
    """The stored copy's build record, or None when it cannot serve us."""
    try:
        build = json.loads(Path(meta_path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(build, dict):
        return None
    covers = build.get("covers")
    if (not isinstance(build.get("url"), str)
            or not isinstance(covers, dict)
            or not covers
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in covers.items())
            or build.get("schema") not in (
                "hm_prefixed", "unprefixed", "legacy-normalised")
            or build.get("effect_from") not in (
                "beta", "log(odds_ratio)", "z * se")
            or not isinstance(build.get("has_n"), bool)):
        return None
    n_kind = build.get("n_source_kind")
    if n_kind is not None and n_kind not in (
            "reported_total", "variant_n", "none", "unknown"):
        return None
    for name in ("seen", "kept", "bytes"):
        value = build.get(name)
        if (isinstance(value, bool) or not isinstance(value, int)
                or value <= 0):
            return None
    remote_bytes = build.get("remote_bytes")
    if remote_bytes is not None and (
            isinstance(remote_bytes, bool)
            or not isinstance(remote_bytes, int) or remote_bytes < 0):
        return None
    for name in _REMOTE_VALIDATOR_FIELDS:
        value = build.get(name)
        if value is not None and _remote_validator(value) != value:
            return None
    if build["seen"] < build["kept"]:
        return None
    last_used = build.get("last_used")
    if (isinstance(last_used, bool)
            or not isinstance(last_used, (int, float))
            or not math.isfinite(float(last_used))):
        return None
    for name in ("sha256", "normalised_sha256", "store_sha256"):
        value = build.get(name)
        if value is not None and (
                not isinstance(value, str)
                or not re.fullmatch(r"[0-9a-f]{64}", value)):
            return None
    if fingerprint is not None and fingerprint not in covers:
        return None
    return build


def _matching_build(meta_path: Path, url: str, fingerprint: str,
                    remote_bytes: int = 0, *, remote_etag: str | None = None,
                    remote_last_modified: str | None = None) -> dict | None:
    """A build for exactly this deposit and LD-cache content, if present."""
    build = _load_build(meta_path, fingerprint)
    if build is None or build.get("url") != url:
        return None
    if remote_bytes and build.get("remote_bytes") != remote_bytes:
        return None
    # A current validator is positive evidence about this deposit generation:
    # require the stored copy to carry the same value. When the server supplies
    # neither validator, retain the historical URL/size fallback so old jobs and
    # less capable HTTP servers remain usable.
    for name, value in _remote_validators(
            remote_etag, remote_last_modified).items():
        if build.get(name) != value:
            return None
    return build


class _StoreLock(ProcessFileLock):
    """Kernel-backed Catalog lock, automatically released on process exit."""


def _file_identity(path: Path) -> tuple | None:
    """Identity of one published file, stable across an atomic replacement."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wait_for_unlock(lock: _StoreLock, on_wait=None) -> None:
    """Acquire after the current process owner releases its kernel lock."""
    start = time.time()
    while time.time() - start < WAIT_LIMIT:
        time.sleep(2.0)
        if lock.acquire():
            return
        if on_wait is not None:
            on_wait(round(time.time() - start))
    raise TimeoutError("timed out waiting for the shared Catalog download")


def _build_store(url: str, data: Path, meta_path: Path, accession: str,
                 union_ids: set, covers: dict, fingerprint: str,
                 remote_bytes: int = 0,
                 remote_etag: str | None = None,
                 remote_last_modified: str | None = None,
                 on_bytes=None, on_wait=None,
                 rejected_identity=None) -> tuple:
    """Return ``(build, reused)`` after publishing or awaiting the store."""
    lock = _StoreLock(str(meta_path) + ".lock")
    provisional = data.with_name(
        f".{data.name}.{os.getpid()}.{uuid.uuid4().hex}.build")
    while not lock.acquire():
        _wait_for_unlock(lock, on_wait)
    try:
        # The previous owner may have published between our last check and
        # this acquisition. Re-check under the lock before downloading.
        build = _matching_build(
            meta_path, url, fingerprint, remote_bytes,
            remote_etag=remote_etag,
            remote_last_modified=remote_last_modified)
        identity = _file_identity(data)
        if build is not None and identity is not None \
                and identity != rejected_identity:
            build["last_used"] = time.time()
            _write_json(meta_path, build)
            return build, True
        if rejected_identity is not None and identity == rejected_identity:
            _discard(data, meta_path)

        def heartbeat(total, _inner=on_bytes):
            lock.touch()
            if _inner is not None:
                _inner(total)
        for attempt in range(GENERATION_ATTEMPTS):
            try:
                info = stream_filter(
                    url, union_ids, provisional, on_bytes=heartbeat,
                    remote_etag=remote_etag,
                    remote_last_modified=remote_last_modified)
                if remote_bytes and info["source_bytes"] != remote_bytes:
                    raise _GenerationMismatch(
                        f"source size changed from {remote_bytes:,} to "
                        f"{info['source_bytes']:,} bytes")
                break
            except _GenerationMismatch as exc:
                for path in (provisional, Path(str(provisional) + ".part")):
                    try:
                        path.unlink()
                    except OSError:
                        pass
                if attempt + 1 == GENERATION_ATTEMPTS:
                    raise ValueError(
                        f"{accession}: the Catalog deposit changed between "
                        "validation and download; retry the submission "
                        f"({exc})") from exc
        store_sha256 = _file_sha256(provisional)
        size = provisional.stat().st_size
        lock.touch()
        if not lock.owned():
            raise RuntimeError(
                f"{accession}: lost the shared-store lock while downloading; "
                "retry the job")
        build = {"accession": accession, "url": url,
                 "remote_bytes": int(remote_bytes or 0),
                 "built": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                 "last_used": time.time(),
                 "seen": info["seen"], "kept": info["kept"],
                 "sha256": info["sha256"], "schema": info["schema"],
                 "effect_from": info["effect_from"], "has_n": info["has_n"],
                 "n_source_column": info.get("n_source_column"),
                 "n_source_kind": info.get("n_source_kind", "unknown"),
                 "covers": covers, "bytes": size,
                 "store_sha256": store_sha256,
                 "origin": "download"}
        build.update(_remote_validators(remote_etag, remote_last_modified))
        build.update(_remote_validators(
            info.get("remote_etag"), info.get("remote_last_modified")))
        os.replace(provisional, data)
        _write_json(meta_path, build)
        return build, False
    finally:
        for path in (provisional, Path(str(provisional) + ".part")):
            try:
                path.unlink()
            except OSError:
                pass
        lock.release()


def _discard(data: Path, meta_path: Path) -> None:
    for path in (data, meta_path):
        try:
            path.unlink()
        except OSError:
            pass


class _StoreReadError(Exception):
    """A stored file cannot be trusted and records which copy was read."""

    def __init__(self, detail, identity):
        super().__init__(detail)
        self.identity = identity


class _StoreLayoutError(_StoreReadError):
    """The stored copy is not in the layout we wrote; rebuild rather than read."""


class _StoreGzipError(_StoreReadError):
    """The stored copy is not a complete, valid gzip stream."""


HEADER = "\t".join(OUT_COLS)

# Every historical job in this checkout was written by the same canonical
# normaliser introduced with the web service. Restrict migration to those
# recorded producers; a future layout or transform must opt in explicitly.
_LEGACY_NORMALISER_VERSIONS = frozenset(("0.3.9.dev0", "0.3.10.dev0"))


def stored_copy_available(root: Path, *, accession: str, url: str,
                          fingerprint: str, remote_bytes: int = 0,
                          remote_etag: str | None = None,
                          remote_last_modified: str | None = None) -> bool:
    """Whether the shared store already serves this exact source/reference."""
    try:
        data, meta_path = _store_paths(root, accession)
        return (_file_identity(data) is not None
                and _matching_build(
                    meta_path, url, fingerprint, remote_bytes,
                    remote_etag=remote_etag,
                    remote_last_modified=remote_last_modified) is not None)
    except (OSError, ValueError):
        return False


def _legacy_candidate_records(root: Path, accession: str, url: str,
                              fingerprint: str, remote_bytes: int,
                              exclude_job_id: str | None,
                              remote_etag: str | None = None,
                              remote_last_modified: str | None = None):
    """Yield metadata for completed, provenance-compatible legacy files."""
    jobs_dir = Path(root) / "jobs"
    try:
        paths = sorted(jobs_dir.glob("*/job.json"),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True)
    except OSError:
        return
    for job_path in paths:
        try:
            job_dir = job_path.parent
            if job_dir.is_symlink() \
                    or job_dir.resolve().parent != jobs_dir.resolve():
                continue
            job = json.loads(job_path.read_text())
            job_id = job.get("id")
            if job_id != job_dir.name or job.get("status") != "done" \
                    or job_id == exclude_job_id:
                continue
            result = json.loads((job_dir / "result.json").read_text())
            provenance = result.get("provenance") or {}
            if provenance.get("cache_sha256") != fingerprint \
                    or provenance.get("bipred") not in \
                    _LEGACY_NORMALISER_VERSIONS:
                continue
            result_catalog = provenance.get("catalog") or {}
            options = job.get("options") or {}
            files = job.get("files") or {}
        except (OSError, TypeError, ValueError):
            continue

        for trait in (1, 2):
            catalog = options.get(f"catalog{trait}")
            recorded = result_catalog.get(f"trait{trait}")
            if not isinstance(catalog, dict) or not isinstance(recorded, dict):
                continue
            if options.get(f"gcst{trait}") != accession \
                    or catalog.get("accession") != accession \
                    or recorded.get("accession") != accession \
                    or catalog.get("url") != url:
                continue
            old_bytes = catalog.get("remote_bytes") or 0
            if remote_bytes and old_bytes != remote_bytes:
                continue
            if any(catalog.get(name) != value for name, value in
                   _remote_validators(
                       remote_etag, remote_last_modified).items()):
                continue

            name = files.get(f"sumstats{trait}")
            if not isinstance(name, str) or not name or Path(name).name != name:
                continue
            source = job_dir / name
            try:
                if source.is_symlink() or not source.is_file() \
                        or source.resolve().parent != job_dir.resolve():
                    continue
            except OSError:
                continue

            kept = (catalog.get("kept"), recorded.get("kept"))
            seen = (catalog.get("seen"), recorded.get("seen"))
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or int(v) != v or v <= 0
                   for v in kept + seen):
                continue
            if kept[0] != kept[1] or seen[0] != seen[1] \
                    or seen[0] < kept[0]:
                continue
            effects = (catalog.get("effect_from"),
                       recorded.get("effect_from"))
            if effects[0] != effects[1] or effects[0] not in \
                    ("beta", "log(odds_ratio)", "z * se"):
                continue

            raw_values = (catalog.get("sha256"), recorded.get("sha256"))
            if any(value is not None and not isinstance(value, str)
                   for value in raw_values):
                continue
            raw_hashes = [value.lower() for value in raw_values
                          if isinstance(value, str)]
            if any(len(value) != 64
                   or any(c not in "0123456789abcdef" for c in value)
                   for value in raw_hashes) \
                    or len(set(raw_hashes)) > 1:
                continue
            inputs = provenance.get("inputs") or {}
            input_record = inputs.get(f"trait{trait}") or {}
            if not isinstance(input_record, dict):
                continue
            if input_record and input_record.get("filename") not in (None, name):
                continue
            input_hash = input_record.get("sha256")
            if input_hash is not None and (
                    not isinstance(input_hash, str) or len(input_hash) != 64
                    or any(c not in "0123456789abcdef"
                           for c in input_hash.lower())):
                continue
            yield {
                "job_id": job_id, "trait": trait, "source": source,
                "kept": int(kept[0]), "seen": int(seen[0]),
                "effect_from": effects[0],
                "remote_sha256": raw_hashes[0] if raw_hashes else None,
                "input_sha256": (input_hash.lower()
                                  if isinstance(input_hash, str) else None),
                "per_variant_n_usable_frac": catalog.get(
                    "per_variant_n_usable_frac"),
                "has_per_variant_n": catalog.get("has_per_variant_n"),
            }


def _copy_legacy_candidate(candidate: dict, part: Path, keep_ids: set) -> dict:
    """Copy and fully validate one job-local canonical gzip into ``part``."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(candidate["source"]), flags)
    compressed = hashlib.sha256()
    try:
        source_stat = os.fstat(fd)
        if not stat_module.S_ISREG(source_stat.st_mode):
            raise ValueError("legacy source is not a regular file")
        src = os.fdopen(fd, "rb")
        fd = -1
        with src, open(part, "wb") as dst:
            while chunk := src.read(1 << 20):
                compressed.update(chunk)
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    input_sha256 = compressed.hexdigest()
    if candidate["input_sha256"] is not None \
            and input_sha256 != candidate["input_sha256"]:
        raise ValueError("legacy input SHA-256 does not match result.json")

    logical = hashlib.sha256()
    kept = n_usable = 0
    with gzip.open(part, "rt", encoding="utf-8", errors="strict") as src:
        header = src.readline().rstrip("\n")
        if header != HEADER:
            raise ValueError("legacy source has a non-canonical header")
        logical.update((HEADER + "\n").encode("utf-8"))
        for line in src:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(OUT_COLS) or not fields[0] \
                    or fields[0] not in keep_ids:
                raise ValueError("legacy source has an invalid retained row")
            output = line if line.endswith("\n") else line + "\n"
            logical.update(output.encode("utf-8"))
            kept += 1
            if _is_number(fields[-1]) and float(fields[-1]) > 0:
                n_usable += 1
    if kept != candidate["kept"]:
        raise ValueError("legacy row count does not match job provenance")
    n_frac = n_usable / kept
    recorded_frac = candidate["per_variant_n_usable_frac"]
    if recorded_frac is not None and (
            not isinstance(recorded_frac, (int, float))
            or not math.isfinite(recorded_frac)
            or not math.isclose(float(recorded_frac), n_frac,
                                rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError("legacy per-variant N fraction does not match")
    recorded_has_n = candidate["has_per_variant_n"]
    if recorded_has_n is not None:
        if not isinstance(recorded_has_n, bool) \
                or recorded_has_n != bool(n_usable and n_frac >= 0.99):
            raise ValueError("legacy per-variant N status does not match")
    return {
        "input_sha256": input_sha256,
        "normalised_sha256": logical.hexdigest(),
        "has_n": bool(n_usable),
    }


def adopt_legacy_job_file(root: Path, *, accession: str, url: str,
                          fingerprint: str, keep_ids: set,
                          cache_label: str,
                          remote_bytes: int = 0,
                          remote_etag: str | None = None,
                          remote_last_modified: str | None = None,
                          exclude_job_id: str | None = None) -> bool:
    """Best-effort promotion of a compatible completed job into the store.

    Every scientific identity check is strict; any missing or inconsistent
    evidence merely rejects that candidate and leaves the normal network path
    to :func:`fetch_filtered`. Publication shares the accession lock and atomic
    rename protocol used by downloads.
    """
    try:
        data, meta_path = _store_paths(root, accession)
        lock = _StoreLock(str(meta_path) + ".lock")
        if not lock.acquire():
            return False
        try:
            if _matching_build(
                    meta_path, url, fingerprint, remote_bytes,
                    remote_etag=remote_etag,
                    remote_last_modified=remote_last_modified) is not None \
                    and _file_identity(data) is not None:
                return False
            part = data.with_name(
                f".{data.name}.{os.getpid()}.{uuid.uuid4().hex}.legacy")
            for candidate in _legacy_candidate_records(
                    root, accession, url, fingerprint, remote_bytes,
                    exclude_job_id, remote_etag, remote_last_modified):
                try:
                    checked = _copy_legacy_candidate(candidate, part, keep_ids)
                    now = time.time()
                    build = {
                        "accession": accession, "url": url,
                        "remote_bytes": int(remote_bytes or 0),
                        "built": time.strftime("%Y-%m-%d %H:%M:%S",
                                               time.gmtime()),
                        "last_used": now,
                        "seen": candidate["seen"],
                        "kept": candidate["kept"],
                        "sha256": candidate["remote_sha256"],
                        "normalised_sha256": checked["normalised_sha256"],
                        "store_sha256": checked["input_sha256"],
                        "schema": "legacy-normalised",
                        "effect_from": candidate["effect_from"],
                        "has_n": checked["has_n"],
                        # Historical job provenance did not distinguish a
                        # reported total from an effective/unspecified ``n``.
                        "n_source_column": None,
                        "n_source_kind": "unknown",
                        "covers": {fingerprint: cache_label},
                        "bytes": part.stat().st_size,
                        "origin": "legacy job",
                        "legacy": {
                            "job_id": candidate["job_id"],
                            "trait": candidate["trait"],
                            "input_sha256": checked["input_sha256"],
                        },
                    }
                    build.update(_remote_validators(
                        remote_etag, remote_last_modified))
                    lock.touch()
                    if not lock.owned():
                        raise RuntimeError("lost the legacy-adoption lock")
                    os.replace(part, data)
                    _write_json(meta_path, build)
                    return True
                except Exception:
                    try:
                        part.unlink()
                    except OSError:
                        pass
                    continue
            return False
        finally:
            lock.release()
    except Exception:
        return False


def _sample_size_transform(values, target_effective_n=None,
                           target_total_n=None) -> dict | None:
    """A downward-only scale preserving relative per-variant sample size.

    For a case/control study, ``target_effective_n / target_total_n`` first
    converts a cohort total to effective N. If the deposited median is larger
    than the selected ancestry total, the same factor also corrects that
    ancestry level. Values are never scaled upward: missing-cohort variants
    must not be credited with information they did not contain.
    """
    if len(values) == 0 or not _is_number(target_effective_n) \
            or float(target_effective_n) <= 0:
        return None
    observed = np.asarray(values, dtype=np.float64)
    median = float(np.median(observed))
    if not math.isfinite(median) or median <= 0:
        return None
    target = float(target_effective_n)
    total = (float(target_total_n) if _is_number(target_total_n)
             and float(target_total_n) > 0 else target)
    design_factor = min(1.0, target / total)
    ancestry_factor = min(1.0, total / median)
    factor = design_factor * ancestry_factor
    return {
        "method": "downward median rescale",
        "source_median": median,
        "target_effective_n": target,
        "target_total_n": total,
        "factor": factor,
        "applied": bool(factor < 1.0 - 1e-12),
        "upward_scaling_refused": bool(target > median and factor == 1.0),
    }


def _filter_normalised(store: Path, keep_ids: set, dest: Path,
                       on_bytes=None, expected_sha256=None,
                       target_effective_n=None,
                       target_total_n=None) -> tuple:
    """Copy rows; return counts plus logical and compressed SHA-256.

    Rows are written byte-for-byte, so the result is exactly what a download
    filtered to ``keep_ids`` would have produced. Do not be tempted to route
    this through :func:`stream_filter` instead: ``_resolve_schema`` maps the
    *deposit* aliases, and three of our own column names (``chrom``, ``pos``,
    ``pval``) are not among them, so a round trip would silently drop them.
    """
    n_col = OUT_COLS.index("n")
    digest = hashlib.sha256()
    compressed = hashlib.sha256()
    tmp = str(dest) + ".part"
    kept = n_usable = 0
    n_values = array("d")
    identity = None
    finished = False
    try:
        with open(store, "rb") as raw:
            stat = os.fstat(raw.fileno())
            identity = (stat.st_dev, stat.st_ino, stat.st_size,
                        stat.st_mtime_ns)
            counted = _HashingReader(raw, compressed, on_bytes)
            with gzip.open(counted, "rt", encoding="utf-8",
                           errors="replace") as src, \
                    gzip.open(tmp, "wt", encoding="utf-8", newline="") as dst:
                header = src.readline().rstrip("\n")
                if header != HEADER:
                    raise _StoreLayoutError(header, identity)
                output = header + "\n"
                dst.write(output)
                digest.update(output.encode("utf-8"))
                for line in src:
                    fields = line.rstrip("\n").split("\t")
                    if fields[0] not in keep_ids:
                        continue
                    output = line if line.endswith("\n") else line + "\n"
                    dst.write(output)
                    digest.update(output.encode("utf-8"))
                    kept += 1
                    if len(fields) > n_col and _is_number(fields[n_col]) \
                            and float(fields[n_col]) > 0:
                        n_usable += 1
                        n_values.append(float(fields[n_col]))
        store_sha256 = compressed.hexdigest()
        if expected_sha256 is not None and store_sha256 != expected_sha256:
            raise _StoreGzipError(
                "compressed content does not match its recorded SHA-256",
                identity)
        finished = True
    except (gzip.BadGzipFile, EOFError, zlib.error) as exc:
        raise _StoreGzipError(str(exc), identity) from exc
    finally:
        if os.path.exists(tmp) and (not finished or kept == 0):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    if kept == 0:
        raise ValueError("no variants overlap the LD reference — wrong "
                         "genome build or a hits-only deposition?")
    transform = _sample_size_transform(
        n_values, target_effective_n, target_total_n)
    if transform is not None and transform["applied"]:
        scaled = tmp + ".scaled"
        scaled_digest = hashlib.sha256()
        try:
            with gzip.open(tmp, "rt", encoding="utf-8",
                           errors="strict") as src, \
                    gzip.open(scaled, "wt", encoding="utf-8",
                              newline="") as dst:
                header = src.readline()
                dst.write(header)
                scaled_digest.update(header.encode("utf-8"))
                for line in src:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) > n_col and _is_number(fields[n_col]) \
                            and float(fields[n_col]) > 0:
                        fields[n_col] = repr(
                            float(fields[n_col]) * transform["factor"])
                    output = "\t".join(fields) + "\n"
                    dst.write(output)
                    scaled_digest.update(output.encode("utf-8"))
            os.replace(scaled, tmp)
            digest = scaled_digest
        finally:
            try:
                os.unlink(scaled)
            except OSError:
                pass
    os.replace(tmp, dest)
    return kept, n_usable, digest.hexdigest(), store_sha256, transform


def fetch_filtered(url: str, dest: Path, *, accession: str, root: Path,
                   keep_ids: set, fingerprint: str, coverage=None,
                   remote_bytes: int = 0,
                   remote_etag: str | None = None,
                   remote_last_modified: str | None = None,
                   target_effective_n: float | None = None,
                   target_total_n: float | None = None,
                   on_bytes=None, on_filter=None, on_wait=None) -> dict:
    """One accession's variants, filtered to ``keep_ids``, without re-downloading.

    Writes the same normalised TSV.GZ :func:`stream_filter` would have
    written for ``keep_ids``, but reads a stored copy whenever one covers
    this reference. ``coverage()`` is called only when a download is actually
    needed and must return ``(union_ids, covers)``: the variants to keep in
    the stored copy, and ``{cache content hash: label}`` for every reference
    that union covers.

    The returned dict has :func:`stream_filter`'s fields — with ``seen``,
    ``sha256``, ``schema``, ``effect_from`` and ``has_n`` describing the
    *remote* file rather than the stored copy — plus ``normalised_sha256``
    for the decompressed job file, ``reused`` and ``store_kept``.
    """
    data, meta_path = _store_paths(root, accession)
    coverage_value = None
    downloaded = False
    corruption_rebuilds = 0

    def requested_coverage():
        nonlocal coverage_value
        if coverage_value is None:
            union_ids, covers = coverage() if coverage else (
                set(keep_ids), {})
            covers = dict(covers)
            covers.setdefault(fingerprint, "requested LD reference")
            if not keep_ids <= union_ids:
                raise ValueError(
                    "stored-copy variant union does not cover this job's LD "
                    "reference; refusing to filter against it")
            coverage_value = union_ids, covers
        return coverage_value

    while True:
        identity = _file_identity(data)
        build = _matching_build(
            meta_path, url, fingerprint, remote_bytes,
            remote_etag=remote_etag,
            remote_last_modified=remote_last_modified) \
            if identity is not None else None
        if build is None:
            union_ids, covers = requested_coverage()
            _, reused_build = _build_store(
                url, data, meta_path, accession, union_ids, covers,
                fingerprint, remote_bytes=remote_bytes,
                remote_etag=remote_etag,
                remote_last_modified=remote_last_modified,
                on_bytes=on_bytes, on_wait=on_wait,
                rejected_identity=None)
            downloaded |= not reused_build

        # Keep this lock through selection *and* filtering. Otherwise a
        # re-deposited URL can publish new bytes between those operations and
        # make us report old provenance for new content.
        use_lock = _StoreLock(str(meta_path) + ".lock")
        while not use_lock.acquire():
            _wait_for_unlock(use_lock, on_wait)
        retry = False
        try:
            build = _matching_build(
                meta_path, url, fingerprint, remote_bytes,
                remote_etag=remote_etag,
                remote_last_modified=remote_last_modified)
            identity = _file_identity(data)
            if build is None or identity is None:
                retry = True             # another URL won before this lock
            else:
                build["last_used"] = time.time()
                _write_json(meta_path, build)

                def filter_progress(total):
                    use_lock.touch()
                    if on_filter is not None:
                        on_filter(total)
                try:
                    if identity[2] != build["bytes"]:
                        raise _StoreGzipError(
                            "byte count does not match its metadata", identity)
                    (kept, n_usable, normalised_sha256, store_sha256,
                     n_transform) = \
                        _filter_normalised(
                            data, keep_ids, dest, filter_progress,
                            expected_sha256=build.get("store_sha256"),
                            # Only Catalog's explicitly total-N aliases need
                            # conversion.  A canonical ``n`` column is already
                            # the per-variant effective-N contract and must not
                            # be case/control-scaled a second time.
                            target_effective_n=(
                                target_effective_n
                                if build.get("n_source_kind") ==
                                "reported_total" else None),
                            target_total_n=(
                                target_total_n
                                if build.get("n_source_kind") ==
                                "reported_total" else None))
                except _StoreReadError as exc:
                    if not use_lock.owned():
                        retry = True
                    elif corruption_rebuilds:
                        problem = ("an unreadable header" if isinstance(
                            exc, _StoreLayoutError) else
                            "a corrupt, replaced, or truncated gzip stream")
                        raise ValueError(
                            f"{accession}: stored copy has {problem} "
                            f"({exc})") from exc
                    else:
                        # No publisher can replace the copy while we own this
                        # lock. Discard this exact generation and rebuild once.
                        if _file_identity(data) == exc.identity:
                            _discard(data, meta_path)
                        corruption_rebuilds += 1
                        retry = True
                else:
                    if (build.get("store_sha256") is None
                            and use_lock.owned()):
                        # Upgrade a valid store created before checksums were
                        # recorded. The full compressed stream was just read.
                        build["store_sha256"] = store_sha256
                        _write_json(meta_path, build)
                    n_frac = n_usable / kept
                    return {
                        "schema": build["schema"],
                        "effect_from": build["effect_from"],
                        "has_n": build["has_n"], "seen": build["seen"],
                        "n_source_column": build.get("n_source_column"),
                        "n_source_kind": build.get(
                            "n_source_kind", "unknown"),
                        "sha256": build["sha256"], "kept": kept,
                        "normalised_sha256": normalised_sha256,
                        "per_variant_n_usable_frac": n_frac,
                        "has_per_variant_n": bool(
                            build["has_n"] and n_frac >= 0.99),
                        "sample_size_transform": n_transform,
                        "sample_size_safe_for_effective_n": bool(
                            build["has_n"] and n_frac >= 0.99 and (
                                build.get("n_source_kind") == "variant_n"
                                or n_transform is not None)),
                        "reused": not downloaded,
                        "store_kept": build["kept"],
                        "store_origin": build.get("origin", "download"),
                    }
        finally:
            use_lock.release()
        if retry:
            continue


def purge_store(root: Path, budget_gb: float) -> list:
    """Evict least-recently-used stored copies down to a byte budget.

    Copies used within the last hour are never evicted, so a running job
    cannot have the file pulled out from under it. Abandoned partial
    downloads are removed regardless.
    """
    base = store_dir(root)
    now = time.time()
    transient_bytes = 0
    for pattern in ("*.part", "*.legacy", "*.build"):
        for part in base.glob(pattern):
            try:
                if now - part.stat().st_mtime > PART_STALE:
                    part.unlink()
                else:
                    transient_bytes += part.stat().st_size
            except OSError:
                pass
    if not budget_gb or budget_gb <= 0:
        return []
    total, candidates = transient_bytes, []
    for meta_path in sorted(base.glob("GCST*.json")):
        try:
            data, _ = _store_paths(root, meta_path.stem)
        except ValueError:
            continue                        # not one of ours; leave it alone
        build = _load_build(meta_path)
        try:
            size = data.stat().st_size
        except OSError:
            continue
        total += size
        try:
            last_used = float((build or {}).get("last_used") or 0)
        except (TypeError, ValueError):
            last_used = 0.0
        if not math.isfinite(last_used):
            last_used = 0.0
        if build is None or now - last_used > EVICT_GRACE:
            candidates.append((last_used, meta_path.stem, data, meta_path,
                               size))
    budget = budget_gb * 2 ** 30
    removed = []
    for _, accession, data, meta_path, size in sorted(candidates):
        if total <= budget:
            break
        lock = _StoreLock(str(meta_path) + ".lock")
        if not lock.acquire():
            continue
        try:
            current = _load_build(meta_path)
            try:
                current_size = data.stat().st_size
            except OSError:
                current_size = 0
            try:
                current_last = float(
                    (current or {}).get("last_used") or 0)
            except (TypeError, ValueError):
                current_last = 0.0
            if math.isfinite(current_last) \
                    and time.time() - current_last <= EVICT_GRACE:
                continue
            _discard(data, meta_path)
            total -= current_size
            removed.append(accession)
        finally:
            lock.release()
    return removed
