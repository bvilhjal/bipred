"""GWAS Catalog fetch support for the bipred web service.

Two responsibilities, mirroring the two moments they are needed:

* :func:`resolve` runs at *submit* time (in the web process): validate the
  accession, find its harmonised file, and read study metadata (trait name,
  sample size) so the form can confirm what will be downloaded. Answers are
  cached under ``<data root>/_meta/gwascat/`` — the harmonised-file index
  for a week, per-study metadata indefinitely (accessions are immutable).

* :func:`stream_filter` runs at *fit* time (in the runner subprocess): stream
  the harmonised file, keep only variants present in a given variant set,
  and write one normalised TSV.GZ that ``ldpred3.sumstats`` reads with no
  overrides. Raw catalog files run to hundreds of MB and are ~90% variants
  no reference contains, so filtering in the stream is what keeps job
  directories small.

* :func:`fetch_filtered` wraps that for the runner so a file is downloaded
  at most once: a shared per-accession copy under ``<data root>/catalog/``
  is kept filtered to the union of the registered LD references, and each
  job filters that copy locally. Re-running an analysis — including against
  a different registered reference — then touches the network not at all.

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
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

FTP = "https://ftp.ebi.ac.uk/pub/databases/gwas"
SUMSTATS = f"{FTP}/summary_statistics"
HARMONISED_LIST = f"{SUMSTATS}/harmonised_list.txt"
REST_STUDY = "https://www.ebi.ac.uk/gwas/rest/api/studies/{accession}"

LIST_TTL = 7 * 86400.0          # re-fetch the harmonised index weekly

ACCESSION = re.compile(r"^GCST\d{3,}$")

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
        return json.loads(cache.read_text())
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
    cases = _CASES.search(sample)
    controls = _CONTROLS.search(sample)
    if cases and controls:
        meta["n_cases"] = int(cases.group(1).replace(",", ""))
        meta["n_controls"] = int(controls.group(1).replace(",", ""))
        meta["n_eff"] = round(4.0 / (1.0 / meta["n_cases"]
                                     + 1.0 / meta["n_controls"]), 1)
        meta["n_basis"] = ("4/(1/ncase+1/nctrl) from catalog European "
                           "case/control counts")
    else:
        eur = [int(m.group(1).replace(",", ""))
               for m in _EUR_INDIVIDUALS.finditer(sample)]
        if eur:
            meta["n_eff"] = float(sum(eur))
            meta["n_basis"] = "catalog European initial sample size"
        else:
            numbers = [int(m.group(1).replace(",", ""))
                       for m in _INDIVIDUALS.finditer(sample)]
            if numbers:
                meta["n_eff"] = float(max(numbers))
                meta["n_basis"] = ("largest N in the catalog's initial "
                                   "sample size text; check ancestry before "
                                   "trusting it")
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
            remote_bytes = int(resp.headers.get("Content-Length", 0))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"{accession}: harmonised file not found (404)")
        raise ValueError(f"{accession}: harmonised-file check failed ({exc})")
    except (urllib.error.URLError, OSError) as exc:
        raise ValueError(f"{accession}: harmonised-file check failed ({exc})")
    meta = _study_metadata(accession, root)
    meta["url"] = url
    meta["remote_bytes"] = remote_bytes
    return meta


# --- download (fit-time, runner subprocess) ---

def cache_ids(cache_path: Path) -> set:
    """Variant ids of the LD reference the job will harmonize against."""
    with np.load(str(cache_path), allow_pickle=False) as cache:
        return set(cache["ids"].tolist())


def stream_filter(url_or_path: str, keep_ids: set, dest: Path,
                  on_bytes=None) -> dict:
    """Stream one harmonised file into the normalised layout, keeping only
    ``keep_ids`` variants, and return download provenance.

    ``url_or_path`` may also be a local file path, which is how the test
    suite exercises this without network access. ``on_bytes(n)``, if given,
    is called with the running compressed-byte count as the stream is read —
    the caller throttles what it does with it.
    """
    digest = hashlib.sha256()
    tmp = str(dest) + ".part"
    if os.path.exists(url_or_path):
        raw = open(url_or_path, "rb")
    else:
        raw = urllib.request.urlopen(url_or_path, timeout=900)
    seen = kept = n_usable = 0
    with raw:
        stream = _HashingReader(raw, digest, on_bytes)
        with gzip.open(stream, "rt", encoding="utf-8",
                       errors="replace") as src, \
                gzip.open(tmp, "wt", encoding="utf-8", newline="") as dst:
            reader = csv.DictReader(src, delimiter="\t")
            mapping, odds, zscore = _resolve_schema(reader.fieldnames)
            has_beta = "beta" in mapping
            info = {"schema": ("hm_prefixed" if "hm_rsid" in
                               (reader.fieldnames or []) else "unprefixed"),
                    "has_n": "n" in mapping,
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
                per_variant_n_usable_frac=n_frac,
                has_per_variant_n=bool(info["has_n"] and n_frac >= 0.99))
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
LOCK_STALE = 900.0          # a lock unheartbeaten this long is abandoned
LOCK_TOUCH = 30.0           # heartbeat interval while downloading
WAIT_LIMIT = 3600.0         # give up waiting on another job's download
EVICT_GRACE = 3600.0        # never evict a copy used this recently
PART_STALE = 3600.0         # abandoned partial downloads older than this go


def _write_json(path: Path, payload: dict) -> None:
    tmp = str(path) + ".part"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


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
    if fingerprint is not None and fingerprint not in (build.get("covers") or {}):
        return None
    return build


def _record_use(meta_path: Path) -> None:
    """Stamp the copy as in use, which also holds eviction off it."""
    build = _load_build(meta_path)
    if build is None:
        return
    build["last_used"] = time.time()
    try:
        _write_json(Path(meta_path), build)
    except OSError:
        pass                                # bookkeeping never fails a job


class _StoreLock:
    """Best-effort exclusive lock so two jobs do not fetch the same file.

    Ownership is a file created with ``O_EXCL``; the owner touches it while
    downloading, and a lock left unheartbeaten for ``LOCK_STALE`` is treated
    as abandoned by a dead process. Losing the race is never a correctness
    problem — publishing is an atomic rename — so a waiter that gives up
    downloads too rather than failing the job.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.fd = None
        self._touched = 0.0

    def acquire(self) -> bool:
        for attempt in (0, 1):
            try:
                self.fd = os.open(str(self.path),
                                  os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                self._touched = time.time()
                return True
            except FileExistsError:
                if attempt or not self.stale():
                    return False
                self.steal()
        return False

    def stale(self) -> bool:
        try:
            return time.time() - self.path.stat().st_mtime > LOCK_STALE
        except OSError:
            return False                    # gone: the owner released it

    def steal(self) -> None:
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def touch(self) -> None:
        now = time.time()
        if self.fd is None or now - self._touched < LOCK_TOUCH:
            return
        self._touched = now
        try:
            os.utime(self.path, None)
        except OSError:
            pass

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            os.close(self.fd)
        finally:
            self.fd = None
            self.steal()


def _wait_for_build(meta_path: Path, lock: _StoreLock, fingerprint: str,
                    on_wait=None) -> dict | None:
    """Wait for whoever holds the lock to publish a copy that covers us."""
    start = time.time()
    while time.time() - start < WAIT_LIMIT:
        time.sleep(2.0)
        build = _load_build(meta_path, fingerprint)
        if build is not None:
            return build
        if not lock.path.exists():
            return None                     # released without helping us
        if lock.stale():
            lock.steal()
            return None
        if on_wait is not None:
            on_wait(round(time.time() - start))
    return None


def _build_store(url: str, data: Path, meta_path: Path, accession: str,
                 union_ids: set, covers: dict, fingerprint: str,
                 on_bytes=None, on_wait=None) -> dict:
    """Download once into the shared store and record what it covers."""
    lock = _StoreLock(str(meta_path) + ".lock")
    if not lock.acquire():
        build = _wait_for_build(meta_path, lock, fingerprint, on_wait)
        if build is not None and data.exists():
            return build
        lock.acquire()                      # owner died, or built nothing
    try:
        def heartbeat(total, _inner=on_bytes):
            lock.touch()
            if _inner is not None:
                _inner(total)
        info = stream_filter(url, union_ids, data, on_bytes=heartbeat)
        build = {"accession": accession, "url": url,
                 "built": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                 "last_used": time.time(),
                 "seen": info["seen"], "kept": info["kept"],
                 "sha256": info["sha256"], "schema": info["schema"],
                 "effect_from": info["effect_from"], "has_n": info["has_n"],
                 "covers": covers, "bytes": data.stat().st_size}
        _write_json(meta_path, build)
        return build
    except BaseException:
        try:
            os.unlink(str(data) + ".part")
        except OSError:
            pass
        raise
    finally:
        lock.release()


def _discard(data: Path, meta_path: Path) -> None:
    for path in (data, meta_path):
        try:
            path.unlink()
        except OSError:
            pass


class _StoreLayoutError(Exception):
    """The stored copy is not in the layout we wrote; rebuild rather than read."""


HEADER = "\t".join(OUT_COLS)


def _filter_normalised(store: Path, keep_ids: set, dest: Path,
                       on_bytes=None) -> tuple:
    """Copy the stored copy's ``keep_ids`` rows verbatim; return (kept, n_usable).

    Rows are written byte-for-byte, so the result is exactly what a download
    filtered to ``keep_ids`` would have produced. Do not be tempted to route
    this through :func:`stream_filter` instead: ``_resolve_schema`` maps the
    *deposit* aliases, and three of our own column names (``chrom``, ``pos``,
    ``pval``) are not among them, so a round trip would silently drop them.
    """
    n_col = OUT_COLS.index("n")
    tmp = str(dest) + ".part"
    kept = n_usable = 0
    with open(store, "rb") as raw:
        counted = _HashingReader(raw, None, on_bytes)
        with gzip.open(counted, "rt", encoding="utf-8",
                       errors="replace") as src, \
                gzip.open(tmp, "wt", encoding="utf-8", newline="") as dst:
            header = src.readline().rstrip("\n")
            if header != HEADER:
                os.unlink(tmp)
                raise _StoreLayoutError(header)
            dst.write(header + "\n")
            for line in src:
                fields = line.rstrip("\n").split("\t")
                if fields[0] not in keep_ids:
                    continue
                dst.write(line if line.endswith("\n") else line + "\n")
                kept += 1
                if len(fields) > n_col and _is_number(fields[n_col]) \
                        and float(fields[n_col]) > 0:
                    n_usable += 1
    if kept == 0:
        os.unlink(tmp)
        raise ValueError("no variants overlap the LD reference — wrong "
                         "genome build or a hits-only deposition?")
    os.replace(tmp, dest)
    return kept, n_usable


def fetch_filtered(url: str, dest: Path, *, accession: str, root: Path,
                   keep_ids: set, fingerprint: str, coverage=None,
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
    *remote* file rather than the stored copy — plus ``reused`` and
    ``store_kept``.
    """
    data, meta_path = _store_paths(root, accession)
    for attempt in (0, 1):
        build = _load_build(meta_path, fingerprint) if data.exists() else None
        if build is not None and build.get("url") != url:
            # An accession is immutable, but the harmonised file it points at
            # can be re-deposited under a new path. Then the copy is of a file
            # we are no longer being asked for, and reuse would be a lie.
            build = None
        if build is not None:
            _record_use(meta_path)          # holds eviction off before we read
        reused = build is not None and data.exists()
        if not reused:
            union_ids, covers = coverage() if coverage else (set(keep_ids), {})
            covers = dict(covers)
            covers.setdefault(fingerprint, "requested LD reference")
            if not keep_ids <= union_ids:
                raise ValueError(
                    "stored-copy variant union does not cover this job's LD "
                    "reference; refusing to filter against it")
            build = _build_store(url, data, meta_path, accession, union_ids,
                                 covers, fingerprint, on_bytes=on_bytes,
                                 on_wait=on_wait)
        try:
            kept, n_usable = _filter_normalised(data, keep_ids, dest,
                                                on_filter)
            break
        except _StoreLayoutError as exc:
            if attempt:                     # we just wrote it: give up loudly
                raise ValueError(
                    f"{accession}: stored copy has an unreadable header "
                    f"({exc})")
            _discard(data, meta_path)       # an older layout: fetch it again
    n_frac = n_usable / kept
    return {"schema": build["schema"], "effect_from": build["effect_from"],
            "has_n": build["has_n"], "seen": build["seen"],
            "sha256": build["sha256"], "kept": kept,
            "per_variant_n_usable_frac": n_frac,
            "has_per_variant_n": bool(build["has_n"] and n_frac >= 0.99),
            "reused": reused, "store_kept": build["kept"]}


def purge_store(root: Path, budget_gb: float) -> list:
    """Evict least-recently-used stored copies down to a byte budget.

    Copies used within the last hour are never evicted, so a running job
    cannot have the file pulled out from under it. Abandoned partial
    downloads are removed regardless.
    """
    base = store_dir(root)
    now = time.time()
    for part in base.glob("*.tsv.gz.part"):
        try:
            if now - part.stat().st_mtime > PART_STALE:
                part.unlink()
        except OSError:
            pass
    if not budget_gb or budget_gb <= 0:
        return []
    total, candidates = 0, []
    for meta_path in sorted(base.glob("GCST*.json")):
        try:
            data, _ = _store_paths(root, meta_path.stem)
        except ValueError:
            continue                        # not one of ours; leave it alone
        build = _load_build(meta_path)
        try:
            size = data.stat().st_size
        except OSError:
            _discard(data, meta_path)       # orphaned record of a gone copy
            continue
        total += size
        if build is None or now - float(build.get("last_used") or 0) > EVICT_GRACE:
            candidates.append((float((build or {}).get("last_used") or 0),
                               meta_path.stem, data, meta_path, size))
    budget = budget_gb * 2 ** 30
    removed = []
    for _, accession, data, meta_path, size in sorted(candidates):
        if total <= budget:
            break
        for path in (data, meta_path):
            try:
                path.unlink()
            except OSError:
                pass
        total -= size
        removed.append(accession)
    return removed
