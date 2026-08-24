"""GWAS Catalog fetch support for the bipred web service.

Two responsibilities, mirroring the two moments they are needed:

* :func:`resolve` runs at *submit* time (in the web process): validate the
  accession, find its harmonised file, and read study metadata (trait name,
  sample size) so the form can confirm what will be downloaded. Answers are
  cached under ``<data root>/_meta/gwascat/`` — the harmonised-file index
  for a week, per-study metadata indefinitely (accessions are immutable).

* :func:`stream_filter` runs at *fit* time (in the runner subprocess): stream
  the harmonised file, keep only variants present in the job's LD reference,
  and write one normalised TSV.GZ that ``ldpred3.sumstats`` reads with no
  overrides. Raw catalog files run to hundreds of MB and are ~90% variants
  the reference does not contain, so filtering in the stream is what keeps
  job directories small.

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
    """Wrap a byte stream: hash bytes as read, count them, report progress."""

    def __init__(self, stream, digest, on_bytes=None):
        self._stream, self._digest = stream, digest
        self._on_bytes = on_bytes
        self.total = 0

    def readable(self):
        return True

    def read(self, size=-1):
        data = self._stream.read(size)
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
# time for resolution failures, job-reap time for download/fit outcomes) so
# concurrent runner subprocesses never race it. Structural resolution
# failures (no such study, no harmonised file, dead URL) are recorded;
# transient network errors are not.

REGISTRY_NAME = "accessions.json"

_STRUCTURAL = ("no such study", "no harmonised", "harmonised file "
               "unavailable")


def worth_recording(message: str) -> bool:
    """Is this resolve() failure a property of the accession, not the network?"""
    return any(s in str(message) for s in _STRUCTURAL)


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
    path = _harmonised_paths(root).get(accession)
    if path is None:
        raise ValueError(f"{accession}: no harmonised summary-statistics "
                         "file in the GWAS Catalog")
    url = f"{SUMSTATS}/{path}"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            remote_bytes = int(resp.headers.get("Content-Length", 0))
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise ValueError(f"{accession}: harmonised file unavailable ({exc})")
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
    seen = kept = 0
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
                writer.writerow(out_row)
                kept += 1
    if kept == 0:
        os.unlink(tmp)
        raise ValueError("no variants overlap the LD reference — wrong "
                         "genome build or a hits-only deposition?")
    os.replace(tmp, dest)
    info.update(seen=seen, kept=kept, sha256=digest.hexdigest())
    return info
