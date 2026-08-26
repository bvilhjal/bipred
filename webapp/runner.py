"""Fit driver for one web job: ``python -m webapp.runner <job dir>``.

Gets any Catalog inputs, prepares and LD-screens each trait, builds the joint
analysis set, then runs the LD-score diagnostic -> fit -> (optional) weights.
``job.json`` is updated after every stage so the status page shows progress
and durable cache outcomes; ``result.json`` / ``munge.json`` are written on
success. Any unhandled exception fails the job with a user-readable message;
the full traceback lands in ``runner.log``.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import platform
import hashlib
import subprocess
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import caches, jobs


class _Stages:
    """Per-stage progress: current stage + elapsed seconds, in job.json."""

    def __init__(self, root, job):
        self.root, self.job = root, job
        self.t0 = None
        self._lock = threading.Lock()
        self._last_progress = {}

    def start(self, name):
        with self._lock:
            self.job["stage_schema"] = jobs.STAGE_SCHEMA
            self.job.setdefault("stage_details", {})
            self.job["stage"] = name
            self.job["progress"] = None
            self.t0 = time.time()
            self._last_progress = {}
            jobs.save_job(self.root, self.job)

    def done(self, name):
        with self._lock:
            self.job["stages"][name] = round(time.time() - self.t0, 3)
            self.job["progress"] = None
            jobs.save_job(self.root, self.job)

    def progress(self, **fields):
        """Persist throttled progress without losing a concurrent transfer."""
        # Concurrent Catalog transfers have independent counters. Semantic
        # transitions are always persisted; repeated counters are throttled.
        key = (fields.get("trait"), fields.get("step"), fields.get("phase"),
               bool(fields.get("bytes")), bool(fields.get("filtering")),
               bool(fields.get("waiting")),
               bool(fields.get("prepared_waiting")),
               fields.get("prepared_source"))
        now = time.time()
        with self._lock:
            if now - self._last_progress.get(key, 0.0) < 1.0:
                return
            self._last_progress[key] = now
            fields.setdefault("mb_s", None)
            trait = fields.get("trait")
            if (self.job.get("stage") == "acquire"
                    and trait in (1, 2, "1", "2")):
                current = self.job.get("progress")
                if not isinstance(current, dict) or not isinstance(
                        current.get("traits"), dict):
                    current = {"traits": {}}
                current["traits"][f"trait{int(trait)}"] = fields
                self.job["progress"] = current
            else:
                self.job["progress"] = fields
            jobs.save_job(self.root, self.job)

    def detail(self, name, summary, **fields):
        """Persist a completed or partial outcome beyond transient progress."""
        with self._lock:
            details = self.job.setdefault("stage_details", {})
            details[name] = {"summary": summary, **fields}
            jobs.save_job(self.root, self.job)

    def finish_acquire_trait(self, trait, meta, dest, info, source, outcomes):
        """Publish one concurrent transfer's complete state atomically.

        A counterpart transfer may still be serializing progress from another
        thread.  Keep every mutation of the shared job object under the same
        lock as that serialization.
        """
        with self._lock:
            self.job["files"][f"sumstats{trait}"] = dest.name
            meta.update(
                kept=info["kept"], seen=info["seen"],
                effect_from=info["effect_from"], sha256=info["sha256"],
                normalised_sha256=info["normalised_sha256"], source=source,
                has_per_variant_n=info["has_per_variant_n"],
                per_variant_n_usable_frac=info[
                    "per_variant_n_usable_frac"])
            outcomes[f"trait{trait}"] = (
                "downloaded" if source == "download"
                else "reused stored data")
            details = self.job.setdefault("stage_details", {})
            details["acquire"] = {
                "summary": _acquire_summary(outcomes),
                "traits": dict(outcomes),
            }
            options = self.job["options"]
            # Catalog metadata is an advisory scalar fallback. When the file
            # has an almost-complete positive per-variant N column, preserve it
            # unless the submitter explicitly overrode N.
            if info["has_per_variant_n"] and not options.get(
                    f"catalog_n_user_supplied{trait}"):
                options[f"n_eff{trait}"] = None
                options[f"n_cases{trait}"] = None
                options[f"n_controls{trait}"] = None
            jobs.save_job(self.root, self.job)


def _quantity(count, singular, plural=None):
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {noun}"


def _acquire_summary(outcomes):
    downloaded = sum(value == "downloaded" for value in outcomes.values())
    reused = sum(value == "reused stored data" for value in outcomes.values())
    clauses = []
    if downloaded:
        clauses.append(
            f"{_quantity(downloaded, 'Catalog file')} downloaded and stored")
    if reused:
        clauses.append(f"{_quantity(reused, 'stored Catalog file')} reused")
    return "; ".join(clauses) + "."


def _to_float_list(values):
    return [float(v) for v in values]


_DEFAULT_H2_INIT = (0.1, 0.1)
_PREPARED_SCOPE = "post-QC, LD-aligned, trait-local LD-consistency-screened"


def _screen_parameters():
    """Canonical mandatory-screen settings, independent of sampler seed."""
    return {
        "rounds": 4, "window": 1000, "threshold": 29.72,
        "eigenvalue_floor": 1e-3, "seed": 0,
        "ncores": 1, "verbose": False,
    }


def _required_ld_score_rows(cache, root, *, cache_sha256, n_variants,
                            cache_indices, fitted_shape):
    """Load the selected reference's scores and gather exact fitted rows.

    Reference integrity is deliberately outside the optional LDSC-regression
    boundary: a missing, corrupt, or misaligned panel cannot silently change
    the sampler initialization.
    """
    import numpy as np

    panel = caches.load_or_create_ld_score_panel(
        cache, root, cache_sha256=cache_sha256,
        n_variants=int(n_variants))
    indices = np.asarray(cache_indices)
    if (indices.ndim != 1 or indices.shape != tuple(fitted_shape)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.issubdtype(indices.dtype, np.bool_)
            or (indices.size and (indices[0] < 0
                                  or indices[-1] >= panel.m_snps))
            or (indices.size > 1 and np.any(np.diff(indices) <= 0))):
        raise ValueError("paired full-reference row indices are invalid")
    return panel, panel.scores[indices]


def _ldsc_panel_fields(panel, n_regression_variants):
    """Result/provenance fields that remain valid if regression fails."""
    return {
        "m_snps": int(panel.m_snps),
        "n_regression_variants": int(n_regression_variants),
        "score_definition": panel.definition,
        "score_source": panel.source,
        "score_source_sha256": panel.source_sha256,
        "score_sha256": panel.score_sha256,
        "score_algorithm": panel.algorithm,
        "finite_reference_correction": panel.correction,
        "mean_ld_score": float(panel.score_mean),
        "sum_ld_scores": float(panel.score_sum),
        "effective_rank": float(panel.effective_rank),
        "scope": "full-reference LD scores; fitted-panel regression rows",
    }


def _ldsc_h2_start(values):
    """Convert two LDSC estimates into valid, provenance-labelled starts."""
    values = tuple(values)
    if len(values) != 2:
        raise ValueError("LDSC must return exactly two h2 estimates")
    starts, sources = [], []
    for value in values:
        estimate = float(value)
        if math.isfinite(estimate):
            starts.append(min(max(estimate, 1e-3), 1.0))
            sources.append("ldsc")
        else:
            starts.append(0.1)
            sources.append("default_nonfinite_ldsc")
    return tuple(starts), sources


def _run_ldsc_regression(prep, ell, panel):
    """Run LDSC after panel validation, with deterministic fit-only fallback."""
    import numpy as np
    from bipred.ldsc import ldsc_rg

    panel_fields = _ldsc_panel_fields(panel, len(ell))
    try:
        result = ldsc_rg(
            prep.beta_hat1, prep.beta_hat2, ell,
            prep.n_eff1, prep.n_eff2, m_snps=panel.m_snps)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        h2_init = _DEFAULT_H2_INIT
        return h2_init, {
            **panel_fields,
            "error": str(exc),
            "h2_init": _to_float_list(h2_init),
            "h2_init_source": ["default_regression_failure"] * 2,
        }

    h2_init, h2_sources = _ldsc_h2_start(result.h2)
    return h2_init, {
        "rg": float(result.rg), "rg_se": float(result.rg_se),
        "gcov": float(result.gcov),
        "gcov_intercept": float(result.gcov_intercept),
        "h2": _to_float_list(result.h2),
        "h2_init": _to_float_list(h2_init),
        "h2_init_source": h2_sources,
        **panel_fields,
    }


def _json_safe(value):
    """Recursively map non-finite floats to None (NaN is not valid JSON)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_stable_ld_cache(cache, root, *, expected_sha256):
    """Load exactly the LD generation selected when this job began."""
    from ldpred3.interop import prepare_ld_cache

    before = caches.sha256_cached(cache, root)
    if before != expected_sha256:
        raise ValueError(
            "selected LD reference changed before it was loaded; retry the "
            "analysis")
    prepared = None
    try:
        prepared = prepare_ld_cache(str(cache))
        after = caches.sha256_cached(cache, root)
        if after != expected_sha256:
            raise ValueError(
                "selected LD reference changed while it was being loaded; "
                "retry the analysis")
        return prepared, expected_sha256
    except BaseException:
        if prepared is not None:
            prepared.close()
        raise


def _ids_sha256(values):
    """Order-sensitive digest of the fitted variant identifiers."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cpu_model():
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _total_memory_gb():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") *
                     os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 2)
    except (AttributeError, OSError, ValueError):
        return None


def _usage():
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak = float(usage.ru_maxrss)
        if sys.platform == "darwin":           # bytes on macOS, KiB on Linux
            peak /= 2 ** 30
        else:
            peak /= 2 ** 20
        return usage.ru_utime, usage.ru_stime, peak
    except (ImportError, AttributeError):       # pragma: no cover - Windows
        return 0.0, 0.0, None


def _threadpools():
    try:
        from threadpoolctl import threadpool_info
        return [{key: pool.get(key) for key in
                 ("user_api", "internal_api", "prefix", "version",
                  "num_threads")}
                for pool in threadpool_info()]
    except ImportError:
        return []


def _git_state():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent,
            text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent.parent, text=True, timeout=5).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _warning_rows(stage, caught):
    return [{"stage": stage, "category": item.category.__name__,
             "message": str(item.message)} for item in caught]


def _warnings_are_critical(rows, divergence=None):
    """Whether estimates must be quarantined.

    The structured divergence flag is authoritative when available; warning
    text remains as the compatibility path for results from older fit code.
    """
    return bool(divergence and divergence.get("flagged")) or any(
        "do not interpret" in item["message"].lower()
        or "appears to have diverged" in item["message"].lower()
        for item in rows)


def _attribute_to_catalog(exc, job):
    """Name the catalog accession when a stage error blames one trait.

    Post-download stages see plain files, so a per-trait failure carries the
    runner's trait marker (``trait1`` / ``trait 2``, or the staged
    ``traitN.gcst.tsv.gz`` file name inside an ldpred3 message), not the
    accession.  Translate the marker here, mirroring the download stage's
    ``label (accession): …`` format, so the web process's track-record sweep
    can attribute the outcome.  Joint failures blame nobody and return None,
    as does a failure naming a trait that was an upload.
    """
    message = str(exc)
    compact = message.lower().replace(" ", "")
    marked = [trait for trait in (1, 2) if f"trait{trait}" in compact]
    if len(marked) != 1:
        return None
    meta = job.get("options", {}).get(f"catalog{marked[0]}")
    if not meta:
        return None
    return ValueError(f"{job['labels'][f'trait{marked[0]}']} "
                      f"({meta['accession']}): {message}")


def _n_summary(values, basis):
    import numpy as np
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return {"basis": basis, "n_variants": int(len(values)),
            "min": float(values.min()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "max": float(values.max()) if len(values) else None}


def _validated_cache_indices(prep, prepared_ld):
    """Prove fitted rows map exactly to the immutable full reference."""
    import numpy as np

    indices = prep.cache_indices
    if indices is None:
        raise ValueError("paired inputs lack full-reference row indices")
    indices = np.asarray(indices)
    n_cache = int(len(prepared_ld.variant_ids))
    if (indices.ndim != 1 or indices.shape != prep.beta_hat1.shape
            or not np.issubdtype(indices.dtype, np.integer)
            or np.issubdtype(indices.dtype, np.bool_)
            or (indices.size and (indices[0] < 0 or indices[-1] >= n_cache))
            or (indices.size > 1 and np.any(np.diff(indices) <= 0))):
        raise ValueError("paired full-reference row indices are invalid")
    if not np.array_equal(
            np.asarray(prepared_ld.variant_ids)[indices], np.asarray(prep.id)):
        raise ValueError("paired rows are not aligned to the LD-score reference")
    return indices.astype(np.int64, copy=False)


def _n_semantics(options, trait):
    """Canonical sample-size transform used by one prepared-trait key."""
    n_eff = options.get(f"n_eff{trait}")
    n_cases = options.get(f"n_cases{trait}")
    n_controls = options.get(f"n_controls{trait}")
    if n_eff is not None:
        return {"mode": "scalar", "value": float(n_eff)}
    if n_cases is not None or n_controls is not None:
        if n_cases is None or n_controls is None:
            raise ValueError(
                f"trait{trait}: n_cases and n_controls must be given together")
        from ldpred3 import n_eff_case_control
        return {
            "mode": "scalar",
            "value": float(n_eff_case_control(
                float(n_cases), float(n_controls))),
        }
    return {"mode": "per_variant"}


def _progress_sink(stage):
    """Turn library progress events into job-status updates.

    ``bipred._progress`` deliberately lets a callback's exception propagate,
    which is the right default for a library. Here it must not cost an
    otherwise healthy fit: a status write that fails is dropped, and the
    stage carries on unreported rather than dying. ``_Stages.progress``
    throttles to 1 Hz, so a per-sweep or per-block event stream costs one
    clock read until the second is up.
    """
    def sink(event):
        try:
            stage.progress(**event)
        except OSError:
            pass
    return sink


def _coverage_thunk(root, cache, cache_key, keep_ids, fingerprint):
    """What the shared catalog store should cover, computed only if needed.

    The job's own LD reference plus every other registered real one, so that
    re-running the analysis against a different reference re-filters the
    stored copy instead of downloading the file again. Loading the other
    references costs a second or two, hence the thunk: a job that reuses a
    stored copy never pays it.
    """
    computed = {}
    guard = threading.Lock()

    def coverage():
        if not computed:
            with guard:
                if not computed:
                    from . import gwascat
                    union = set(keep_ids)
                    covers = {fingerprint: cache_key}
                    for entry in caches.real_registry(root):
                        path = Path(entry["path"])
                        try:
                            other = caches.sha256_cached(path, root)
                            if other in covers:
                                continue
                            other_ids = gwascat.cache_ids(path)
                            if caches.sha256_cached(path, root) != other:
                                continue
                        except (OSError, ValueError):
                            # An unreadable extra reference is not fatal.
                            continue
                        union |= other_ids
                        covers[other] = entry["key"]
                    computed["value"] = (union, covers)
        return computed["value"]

    return coverage


def run(job_dir: Path, job: dict) -> None:
    wall0 = time.perf_counter()
    user0, system0, _ = _usage()
    started_utc = datetime.now(timezone.utc).isoformat()
    captured_warnings = []
    root = job_dir.parent.parent
    opt = job["options"]
    # This is a method invariant in stage schema 3, not a user-tunable
    # sensitivity option. It also upgrades a queued schema-2 job safely.
    opt["screen"] = True
    cache = caches.cache_path(opt["cache_key"], root)
    stage = _Stages(root, job)
    expected_ld_sha256 = caches.sha256_cached(cache, root)

    if opt.get("gcst1") or opt.get("gcst2"):
        stage.start("acquire")
        from . import gwascat
        keep_ids = gwascat.cache_ids(cache)
        if caches.sha256_cached(cache, root) != expected_ld_sha256:
            raise ValueError(
                "selected LD reference changed while its variant IDs were "
                "being read; retry the analysis")
        fingerprint = expected_ld_sha256
        coverage = _coverage_thunk(root, cache, opt["cache_key"], keep_ids,
                                   fingerprint)
        acquire_outcomes = {}

        def acquire_one(trait, meta):
            if not gwascat.stored_copy_available(
                    root, accession=meta["accession"], url=meta["url"],
                    fingerprint=fingerprint,
                    remote_bytes=meta.get("remote_bytes") or 0,
                    remote_etag=meta.get("remote_etag"),
                    remote_last_modified=meta.get("remote_last_modified")):
                gwascat.adopt_legacy_job_file(
                    root, accession=meta["accession"], url=meta["url"],
                    fingerprint=fingerprint, keep_ids=keep_ids,
                    cache_label=opt["cache_key"],
                    remote_bytes=meta.get("remote_bytes") or 0,
                    remote_etag=meta.get("remote_etag"),
                    remote_last_modified=meta.get("remote_last_modified"),
                    exclude_job_id=job["id"])
            dest = job_dir / f"trait{trait}.gcst.tsv.gz"
            t0 = time.time()
            transferred = {"network": False}

            def on_bytes(n, meta=meta, trait=trait, t0=t0):
                transferred["network"] = True
                elapsed = time.time() - t0
                stage.progress(
                    trait=trait, accession=meta["accession"], bytes=n,
                    total=meta.get("remote_bytes") or 0,
                    mb_s=round(n / elapsed / 1048576, 1) if elapsed else None)

            def on_filter(n, meta=meta, trait=trait):
                stage.progress(trait=trait, accession=meta["accession"],
                               filtering=n,
                               source=("download" if transferred["network"]
                                       else "stored copy"))

            def on_wait(seconds, meta=meta, trait=trait):
                stage.progress(trait=trait, accession=meta["accession"],
                               waiting=seconds)
            try:
                info = gwascat.fetch_filtered(
                    meta["url"], dest, accession=meta["accession"], root=root,
                    keep_ids=keep_ids, fingerprint=fingerprint,
                    coverage=coverage,
                    remote_bytes=meta.get("remote_bytes") or 0,
                    remote_etag=meta.get("remote_etag"),
                    remote_last_modified=meta.get("remote_last_modified"),
                    on_bytes=on_bytes,
                    on_filter=on_filter, on_wait=on_wait)
            except Exception as exc:
                raise ValueError(
                    f"{job['labels'][f'trait{trait}']} "
                    f"({meta['accession']}): summary-statistics preparation "
                    f"failed: {exc}")
            source = ("legacy stored copy" if info["reused"] and
                      info.get("store_origin") == "legacy job" else
                      "stored copy" if info["reused"] else "download")
            return info, source, dest

        catalog_traits = [
            (trait, opt[f"catalog{trait}"])
            for trait in (1, 2) if opt.get(f"catalog{trait}")
        ]
        # The two Catalog inputs are independent I/O. A hard cap of two avoids
        # serial multi-gigabyte transfers without creating an unbounded client.
        with ThreadPoolExecutor(max_workers=len(catalog_traits)) as pool:
            pending = {
                pool.submit(acquire_one, trait, meta): (trait, meta)
                for trait, meta in catalog_traits
            }
            for future in as_completed(pending):
                trait, meta = pending[future]
                info, source, dest = future.result()
                stage.finish_acquire_trait(
                    trait, meta, dest, info, source, acquire_outcomes)
        jobs.save_job(root, job)
        stage.done("acquire")
        # The reference-sized ID sets and completed Future results are useful
        # only while acquiring files. Do not carry hundreds of MiB of Python
        # strings into LD loading and fitting.
        pending.clear()
        del (acquire_one, coverage, keep_ids, pending, catalog_traits,
             future, info, source, dest)

    ss1 = job_dir / job["files"]["sumstats1"]
    ss2 = job_dir / job["files"]["sumstats2"]

    stage.start("prepare")
    from bipred import (pair_prepared_traits, prepare_trait_sumstats,
                        screen_prepared_trait)
    from ldpred3.sumstats import detect_columns
    from . import prepared_store
    prepared_ld = None
    prep = None
    traits = []
    prepared_info = {}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for trait, path, overrides, label in (
                    (1, ss1, opt["columns1"], job["labels"]["trait1"]),
                    (2, ss2, opt["columns2"], job["labels"]["trait2"])):
                try:
                    detect_columns(str(path), **overrides)
                except Exception as exc:
                    raise ValueError(
                        f"trait{trait} ({label}): cannot interpret columns: "
                        f"{exc}")
            stage.progress(step="Load LD reference")
            prepared_ld, ld_sha256 = _load_stable_ld_cache(
                cache, root, expected_sha256=expected_ld_sha256)
            input_sha256 = {
                "trait1": _sha256(ss1), "trait2": _sha256(ss2),
            }
            stage.detail(
                "prepare",
                "Validated both input schemas and loaded the selected LD "
                "reference; no unscreened trait artifact was published.",
                traits={"trait1": "columns validated",
                        "trait2": "columns validated"},
                ld_sha256=ld_sha256, published=False)
    except Exception as exc:
        if prepared_ld is not None:
            prepared_ld.close()
        # Let the track record blame the right catalog accession when one
        # trait fails; preparation failures never belong to its counterpart.
        attributed = _attribute_to_catalog(exc, job)
        if attributed is not None:
            raise attributed from exc
        raise
    captured_warnings.extend(_warning_rows("prepare", caught))
    stage.done("prepare")

    # The mandatory screen is trait-local and deliberately occurs after all
    # reusable QC/alignment work but before the traits are intersected. Thus a
    # trait is judged against every usable neighbour it carries on this exact
    # reference, not against a counterpart-dependent missingness pattern.
    stage.start("screen")
    screen_params = _screen_parameters()
    screen_outcomes = {}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for trait, path in ((1, ss1), (2, ss2)):
                key = f"trait{trait}"
                catalog = opt.get(f"catalog{trait}")
                accession = catalog.get("accession") if catalog else None
                logical_sha256 = (
                    catalog.get("normalised_sha256") if catalog
                    else input_sha256[key])
                if not logical_sha256:
                    raise ValueError(
                        f"{key}: input lacks a logical content hash; "
                        "refusing unsafe prepared-data reuse")
                spec = prepared_store.semantic_spec(
                    logical_input_sha256=logical_sha256,
                    ld_sha256=ld_sha256,
                    n_semantics=_n_semantics(opt, trait),
                    columns=opt[f"columns{trait}"], qc=True, qc_params={},
                    screen=True, screen_params=screen_params)

                def build(path=path, trait=trait, key=key,
                          accession=accession):
                    stage.progress(
                        trait=trait, accession=accession,
                        step=f"Prepare and LD-screen trait {trait}")
                    unscreened = prepare_trait_sumstats(
                        prepared_ld, str(path),
                        n_eff=opt[f"n_eff{trait}"],
                        n_cases=opt[f"n_cases{trait}"],
                        n_controls=opt[f"n_controls{trait}"],
                        columns=opt[f"columns{trait}"], label=key,
                        progress=None)
                    return screen_prepared_trait(
                        prepared_ld, unscreened, **screen_params,
                        progress=_progress_sink(stage))

                def on_wait(seconds, trait=trait, accession=accession):
                    stage.progress(
                        trait=trait, accession=accession,
                        prepared_waiting=seconds)

                screened, reused = prepared_store.get_or_build(
                    root, spec, label=key, builder=build, on_wait=on_wait)
                prepared_key = prepared_store.key_for(spec)
                record = screened.log["ld_consistency_screen"]
                info = {
                    "n_input": int(record["n_input"]),
                    "n_kept": int(record["n_kept"]),
                    "n_dropped": int(record["n_dropped"]),
                    "prepared_key": prepared_key,
                    "prepared_reused": bool(reused),
                    "prepared_scope": _PREPARED_SCOPE,
                }
                screen_outcomes[key] = info
                prepared_info[key] = {
                    "logical_sha256": logical_sha256,
                    "prepared_key": prepared_key,
                    "prepared_reused": bool(reused),
                    "prepared_scope": _PREPARED_SCOPE,
                    "prepared_numerical_environment":
                        spec["numerical_environment"],
                }
                if catalog is not None:
                    catalog.update(
                        prepared_key=prepared_key,
                        prepared_reused=bool(reused),
                        prepared_scope=_PREPARED_SCOPE)
                stage.progress(
                    trait=trait, accession=accession,
                    prepared_source=("stored screened trait" if reused else
                                     "screened and stored"))
                traits.append(screened)
                summary = "; ".join(
                    f"trait {key[-1]}: {value['n_kept']:,} of "
                    f"{value['n_input']:,} kept"
                    for key, value in screen_outcomes.items())
                stage.detail(
                    "screen", summary + ".", traits=screen_outcomes,
                    mandatory=True, parameters=screen_params,
                    prepared_scope=_PREPARED_SCOPE)
            jobs.save_job(root, job)
        captured_warnings.extend(_warning_rows("screen", caught))
    except Exception as exc:
        if prepared_ld is not None:
            prepared_ld.close()
            prepared_ld = None
        attributed = _attribute_to_catalog(exc, job)
        if attributed is not None:
            raise attributed from exc
        raise
    stage.done("screen")

    stage.start("pair")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stage.progress(step="Build shared variant panel")
            source_blocks = prepared_ld.blocks
            is_mmap = getattr(source_blocks, "close", None) is not None
            # Ordinary compressed NPZ references expand to several GiB. Consume
            # them block by block so the full source and full pair subset never
            # coexist; mmap sources stay non-destructive so their views retain a
            # live, OS-shareable mapping owner.
            # Destructive ownership is safe only for LDpred3's exact ordinary
            # list representation. Custom owners keep the established,
            # non-destructive path even when they are not mmap-backed.
            consume_ld_cache = type(source_blocks) is list
            prep = pair_prepared_traits(
                prepared_ld, traits[0], traits[1],
                screen=False, progress=_progress_sink(stage),
                consume_ld_cache=consume_ld_cache)
            prep.log.update({
                "screen": True,
                "screen_params": dict(screen_params),
                "screen_scope": "trait-local before pairing",
                "trait_screen": dict(screen_outcomes),
            })
            cache_indices = _validated_cache_indices(prep, prepared_ld)

            # The destructive ordinary path has already emptied source_blocks
            # one block at a time. A non-destructive owner can now drop its
            # source views; mmap pair views retain the live mapping owner.
            if not consume_ld_cache and prep.blocks is not source_blocks:
                source_blocks.clear()
            if hasattr(prepared_ld, "_bipred_variant_table"):
                delattr(prepared_ld, "_bipred_variant_table")
            if is_mmap:
                prep._ld_owner = prepared_ld
            else:
                prepared_ld.close()
            prepared_ld = None
            traits.clear()
    except Exception as exc:
        if prepared_ld is not None:
            prepared_ld.close()
        # Let the track record blame the right catalog accession when the
        # failure is one trait's; joint failures stay unattributed.
        attributed = _attribute_to_catalog(exc, job)
        if attributed is not None:
            raise attributed from exc
        raise
    captured_warnings.extend(_warning_rows("pair", caught))
    try:
        # Top-level scalar counts drive the status page; the nested per-trait
        # QC and harmonization logs feed the results page's per-step report.
        munge = {key: int(value) for key, value in prep.log.items()
                 if isinstance(value, numbers.Integral)
                 and not isinstance(value, bool)}
        munge["screen"] = bool(prep.log.get("screen"))
        munge["af_corr"] = _json_safe(prep.log.get("af_corr") or {})
        for trait in ("trait1", "trait2"):
            tlog = prep.log.get(trait) or {}
            screen_log = tlog.get("ld_consistency_screen") or {}
            n_usable = screen_log.get("n_kept")
            munge[trait] = {
                "qc_enabled": bool(tlog.get("qc_enabled")),
                "qc": _json_safe(tlog.get("qc") or {}),
                "harmonize": _json_safe(tlog.get("harmonize") or {}),
                "ld_consistency_screen": _json_safe(screen_log),
                "prepared": _json_safe(prepared_info[trait]),
                # Post-screen variants usable before the cross-trait
                # intersection. Pre-screen indices are not persisted.
                "n_usable": int(n_usable) if n_usable is not None else None,
            }
        (job_dir / "munge.json").write_text(json.dumps(munge, indent=1))
        pair_summary = (
            f"Combined the screened traits: {munge['n_kept']:,} shared "
            "variants kept for fitting"
        )
        stage.detail(
            "pair", pair_summary + ".", rerun=True,
            n_joint=munge["n_joint"], n_kept=munge["n_kept"],
            inputs="post-screen traits",
            ld_reference_storage=("memory mapped" if is_mmap else
                                  "ordinary compressed NPZ"),
            ld_subset_mode=("shared mmap views/copies" if is_mmap else
                            "destructive block-wise low-peak subset"))
        stage.done("pair")

        stage.start("ldsc")
        h2_init = _DEFAULT_H2_INIT
        report = _progress_sink(stage)
        report({"step": "Load precomputed full-reference LD scores",
                "done": 0, "total": 2, "unit": "step"})
        try:
            # Scores are an immutable property of the full reference, not of
            # this pair's QC/DENTIST subset. Only gather regression rows.
            panel, ell = _required_ld_score_rows(
                cache, root, cache_sha256=ld_sha256,
                n_variants=int(prep.log["n_cache"]),
                cache_indices=cache_indices,
                fitted_shape=prep.beta_hat1.shape)
        except Exception as exc:
            stage.detail(
                "ldsc",
                "The selected LD reference has no valid aligned "
                "full-reference LD-score panel; model fitting was not "
                "started.",
                unavailable=True, reference_valid=False,
                required_reference_component=True, error=str(exc))
            raise ValueError(
                "selected LD reference has no valid aligned full-reference "
                f"LD-score panel: {exc}") from exc

        report({"step": "LDSC regression", "done": 1, "total": 2,
                "unit": "step"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            h2_init, ldsc_out = _run_ldsc_regression(prep, ell, panel)
        captured_warnings.extend(_warning_rows("ldsc", caught))
        h2_sources = ldsc_out["h2_init_source"]
        if "error" in ldsc_out:
            stage.detail(
                "ldsc",
                f"Validated full-reference LD scores (M={panel.m_snps:,}); "
                "the data-dependent regression was unavailable, so model "
                "fitting used the deterministic default h2 start "
                "(0.1, 0.1).",
                unavailable=True, reference_valid=True,
                regression_unavailable=True, error=ldsc_out["error"],
                m_snps=int(panel.m_snps),
                n_regression_variants=int(len(ell)),
                h2_init=list(h2_init), h2_init_source=h2_sources)
        else:
            fallback = [str(i + 1) for i, source in enumerate(h2_sources)
                        if source != "ldsc"]
            summary = (
                f"Reused full-reference LD scores (M={panel.m_snps:,}) "
                f"and initialized h2 at {h2_init[0]:.3g}, "
                f"{h2_init[1]:.3g}"
            )
            if fallback:
                summary += ("; the deterministic default replaced a "
                            "non-finite LDSC estimate for trait "
                            + ", ".join(fallback))
            stage.detail(
                "ldsc", summary + ".", unavailable=False,
                reference_valid=True, regression_unavailable=False,
                m_snps=int(panel.m_snps),
                n_regression_variants=int(len(ell)),
                h2_init=list(h2_init), h2_init_source=h2_sources)
        del ell, panel
        stage.done("ldsc")
        prep.cache_indices = None
        del cache_indices

        stage.start("fit")
        from bipred import ldpred3_auto_bivariate_blocks
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = ldpred3_auto_bivariate_blocks(
                prep.blocks, prep.beta_hat1, prep.beta_hat2,
                prep.n_eff1, prep.n_eff2,
                cross_corr=opt.get("cross_corr", 0.0),
                h2_init=h2_init,
                seed=opt["seed"], burn_in=opt["burn_in"],
                num_iter=opt["num_iter"], ncores=1,
                progress=_progress_sink(stage))
        captured_warnings.extend(_warning_rows("fit", caught))
        mixer = {key: (_to_float_list(value) if isinstance(value, tuple)
                       else float(value))
                 for key, value in res.mixer.items()}
        joint = {
            "h2": _to_float_list(res.h2),
            "h2_init": _to_float_list(h2_init),
            "rg": float(res.rg),
            "p": float(res.p),
            "pi": _to_float_list(res.pi) if res.pi is not None else None,
            "mixer": mixer,
            "noise_scale": (_to_float_list(res.noise_scale)
                            if res.noise_scale is not None else None),
            "retained_iterations": res.retained_iterations,
            "stopped_early": bool(res.stopped_early),
        }
        divergence = _json_safe(
            getattr(res, "divergence_diagnostics", None))
        # Uncertainty for the headline numbers: posterior SD across the
        # retained (gvar1, gcov, gvar2) sweeps, when the chain kept them.
        if res.genetic_samples is not None and len(res.genetic_samples):
            import numpy as np
            g = np.asarray(res.genetic_samples, dtype=float)
            ok = np.isfinite(g).all(axis=1) & (g[:, 0] > 0) & (g[:, 2] > 0)
            if ok.any():
                g = g[ok]
                joint["h2_sd"] = [float(g[:, 0].std()), float(g[:, 2].std())]
                joint["rg_sd"] = float(
                    (g[:, 1] / np.sqrt(g[:, 0] * g[:, 2])).std())
        fit_summary = (
            f"Completed with {int(res.retained_iterations):,} retained "
            "iterations"
        )
        if res.stopped_early:
            fit_summary += " (stopped early after meeting the stopping rule)"
        if divergence is not None:
            if not divergence["evaluated"]:
                fit_summary += "; divergence thresholds not evaluated on this small panel"
            elif divergence["flagged"]:
                fit_summary += "; divergence guard flagged the fit"
            else:
                fit_summary += "; no divergence threshold crossed"
        stage.detail("fit", fit_summary + ".")
        stage.done("fit")

        critical = _warnings_are_critical(captured_warnings, divergence)
        written = []
        if opt["weights"] and not critical:
            stage.start("weights")
            common = dict(id=prep.id, chrom=prep.chrom, pos=prep.pos,
                          effect_allele=prep.effect_allele,
                          other_allele=prep.other_allele)
            report = _progress_sink(stage)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for trait in (1, 2):
                    report({"step": f"writing weights for trait {trait}",
                            "done": trait - 1, "total": 2, "unit": "step"})
                    name = f"weights{trait}.tsv"
                    res.write_weights(job_dir / name, trait=trait, **common)
                    written.append(name)
            captured_warnings.extend(_warning_rows("weights", caught))
            stage.detail(
                "weights", "Two prediction-weight files written.",
                skipped=False)
            stage.done("weights")
        elif opt["weights"]:
            stage.start("weights")
            stage.detail(
                "weights",
                "Skipped because a critical fit warning makes weights unsafe "
                "to interpret.", skipped=True)
            stage.done("weights")
    finally:
        prep.close()

    import bipred
    import ldpred3
    import numpy
    user1, system1, peak_gb = _usage()

    def n_basis(trait):
        meta = opt.get(f"catalog{trait}") or {}
        if meta.get("has_per_variant_n") and not opt.get(
                f"catalog_n_user_supplied{trait}"):
            return ("per-variant n column in the harmonised GWAS Catalog "
                    f"file ({meta.get('per_variant_n_usable_frac', 0):.1%} "
                    "usable among retained rows)")
        if opt.get(f"n_eff{trait}") is not None:
            suffix = " (explicit user override)" if opt.get(
                f"catalog_n_user_supplied{trait}") else ""
            return f"constant effective N{suffix}"
        if opt.get(f"n_cases{trait}") is not None:
            return "4/(1/n_cases + 1/n_controls)"
        return "per-variant N column detected in the uploaded file"

    n_info = {
        "trait1": _n_summary(prep.n_eff1, n_basis(1)),
        "trait2": _n_summary(prep.n_eff2, n_basis(2)),
    }
    diagnostics = {
        "valid_for_interpretation": not critical,
        "critical": critical,
        "warnings": captured_warnings,
        "weights_withheld": bool(opt["weights"] and critical),
        "divergence": divergence,
    }
    thread_vars = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                   "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    result = {
        "joint": joint,
        "ldsc": ldsc_out,
        "munge": munge,
        "weights": written,
        "diagnostics": diagnostics,
        "provenance": {
            "bipred": bipred.__version__,
            "ldpred3": ldpred3.__version__,
            "numpy": numpy.__version__,
            "python": platform.python_version(),
            "cache_key": opt["cache_key"],
            "cache_sha256": ld_sha256,
            "seed": opt["seed"],
            "burn_in": opt["burn_in"],
            "num_iter": opt["num_iter"],
            "cross_corr": opt.get("cross_corr", 0.0),
            "screen": True,
            "screen_parameters": dict(screen_params),
            "screen_scope": "trait-local before pairing",
            "fitted_variant_ids_sha256": _ids_sha256(prep.id),
            "sample_size": n_info,
            "inputs": {
                f"trait{trait}": {
                    "filename": job["files"][f"sumstats{trait}"],
                    "sha256": input_sha256[f"trait{trait}"],
                    "logical_sha256": prepared_info[f"trait{trait}"][
                        "logical_sha256"],
                    "prepared_key": prepared_info[f"trait{trait}"][
                        "prepared_key"],
                    "prepared_reused": prepared_info[f"trait{trait}"][
                        "prepared_reused"],
                    "prepared_scope": prepared_info[f"trait{trait}"][
                        "prepared_scope"],
                    "prepared_numerical_environment": prepared_info[
                        f"trait{trait}"]["prepared_numerical_environment"],
                    "column_overrides": opt.get(f"columns{trait}", {}),
                } for trait in (1, 2)
            },
            "compute": {
                "cpu_model": _cpu_model(),
                "logical_cpus": os.cpu_count(),
                "memory_gb": _total_memory_gb(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python_executable": sys.executable,
                "thread_limits": {key: os.environ.get(key)
                                  for key in thread_vars},
                "threadpools": _threadpools(),
                "runner_processes": 1,
                "sampler_ncores": 1,
                "ld_reference_storage": (
                    "memory mapped" if is_mmap else "ordinary compressed NPZ"),
                "ld_subset_mode": (
                    "shared mmap views/copies" if is_mmap else
                    "destructive block-wise low-peak subset"),
            },
            "resources": {
                "wall_s": round(time.perf_counter() - wall0, 3),
                "user_cpu_s": round(user1 - user0, 3),
                "system_cpu_s": round(system1 - system0, 3),
                "peak_rss_gb": round(peak_gb, 3) if peak_gb is not None else None,
            },
            "source": _git_state(),
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "stage_schema": job.get("stage_schema", jobs.STAGE_SCHEMA),
            "stages": job["stages"],
            "stage_details": job.get("stage_details", {}),
        },
    }
    catalog = {f"trait{t}": {k: opt[f"catalog{t}"].get(k) for k in
                             ("accession", "trait", "pmid", "n_basis",
                              "kept", "seen", "effect_from", "sha256",
                              "normalised_sha256", "source",
                              "remote_etag", "remote_last_modified",
                              "prepared_key", "prepared_reused",
                              "prepared_scope",
                              "has_per_variant_n",
                              "per_variant_n_usable_frac")}
               for t in (1, 2) if opt.get(f"catalog{t}")}
    if catalog:
        result["provenance"]["catalog"] = catalog
    (job_dir / "result.json").write_text(
        json.dumps(_json_safe(result), indent=1, allow_nan=False))


def main(argv=None) -> int:
    job_dir = Path((argv or sys.argv[1:])[0]).resolve()
    root = job_dir.parent.parent
    job = jobs.load_job(root, job_dir.name)
    if job is None:
        print(f"no job.json in {job_dir}", file=sys.stderr)
        return 2
    job["status"] = "running"
    job["started"] = time.time()
    job["pid"] = os.getpid()
    jobs.save_job(root, job)
    try:
        run(job_dir, job)
    except Exception as exc:
        traceback.print_exc()
        jobs.update_job(root, job["id"], status="failed",
                        finished=time.time(), error=str(exc), progress=None)
        return 1
    jobs.update_job(root, job["id"], status="done", stage=None,
                    finished=time.time())
    return 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())
