"""Fit driver for one web job: ``python -m webapp.runner <job dir>``.

Gets any Catalog inputs, prepares and LD-screens each trait, builds the joint
analysis set, then runs the LD-score diagnostic -> fit -> (optional) weights.
``job.json`` is updated after every stage so the status page shows progress
and durable cache outcomes; ``result.json`` / ``munge.json`` are written on
success. Any unhandled exception fails the job with a user-readable message;
the full traceback lands in ``runner.log``.
"""

from __future__ import annotations

import gzip
import json
import math
import numbers
import os
import platform
import hashlib
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from . import caches, jobs


def _write_json_atomic(path: Path, text: str) -> None:
    """Publish JSON by unique-temp rename so readers never see a partial file.

    The runner can be killed at any instant (the watchdog's ``os._exit(124)``,
    supervisor termination), and the web process parses these files while
    polling job state; a truncated write would surface as a JSON error on a
    job that actually finished.
    """
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        with open(tmp, "x", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


class _Stages:
    """Per-stage progress: current stage + elapsed seconds, in job.json."""

    def __init__(self, root, job):
        self.root, self.job = root, job
        self.t0 = None
        self._lock = threading.Lock()
        self._last_progress = {}
        self._started = {}
        self._prepared_traits = set()

    def _done_locked(self, name):
        started = self._started.pop(name, self.t0)
        if started is None:
            raise RuntimeError(f"stage {name!r} was not started")
        self.job["stages"][name] = round(time.time() - started, 3)
        active = [item for item in self.job.get("active_stages", [])
                  if item != name]
        self.job["active_stages"] = active
        if self.job.get("stage") == name and active:
            self.job["stage"] = active[-1]
        if not active:
            self.job["progress"] = None

    def start(self, name):
        with self._lock:
            self.job["stage_schema"] = jobs.STAGE_SCHEMA
            self.job.setdefault("stage_details", {})
            self.job["stage"] = name
            self.job["active_stages"] = [name]
            self.job["progress"] = None
            self.t0 = time.time()
            self._started[name] = self.t0
            self._last_progress = {}
            jobs.save_job(self.root, self.job)

    def activate(self, name):
        """Start an overlapping stage without completing its predecessor."""
        with self._lock:
            if name not in self._started:
                self._started[name] = time.time()
            active = self.job.setdefault("active_stages", [])
            if name not in active:
                active.append(name)
            self.job["stage"] = name
            jobs.save_job(self.root, self.job)

    def done(self, name):
        with self._lock:
            self._done_locked(name)
            jobs.save_job(self.root, self.job)

    def finish_prepare_trait(self, trait, info):
        """Publish one trait's preparation and close the stage after both."""
        key = f"trait{int(trait)}"
        with self._lock:
            if key in self._prepared_traits:
                return
            self._prepared_traits.add(key)
            details = self.job.setdefault("stage_details", {})
            previous = details.get("prepare") or {}
            traits = dict(previous.get("traits") or {})
            traits[key] = info
            ordered = {name: traits[name] for name in ("trait1", "trait2")
                       if name in traits}
            summaries = []
            for name, value in ordered.items():
                status = value.get("qc_status", "complete")
                text = (f"trait {name[-1]}: {value['n_usable']:,} aligned "
                        f"({status})")
                if value.get("warnings"):
                    text += f" — {value['warnings'][0]}"
                summaries.append(text)
            details["prepare"] = {
                **previous,
                "summary": "; ".join(summaries) + ".",
                "traits": ordered,
                "parallel": True,
            }
            if len(self._prepared_traits) == 2:
                self._done_locked("prepare")
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
            if trait in (1, 2, "1", "2"):
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
                    "per_variant_n_usable_frac"],
                n_source_column=info.get("n_source_column"),
                n_source_kind=info.get("n_source_kind", "unknown"),
                sample_size_transform=info.get("sample_size_transform"),
                sample_size_safe_for_effective_n=info.get(
                    "sample_size_safe_for_effective_n", True),
                reference_population=info.get("reference_population"),
                sample_size_population_agreement=info.get(
                    "sample_size_population_agreement"))
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
            if (info["has_per_variant_n"]
                    and info.get("sample_size_safe_for_effective_n", True)
                    and not options.get(
                        f"catalog_n_user_supplied{trait}")):
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
_QC_LDSC_PARAMETERS = {
    "chi2_cap": 80.0,
    "chi2_n_scale": 0.001,
    "max_jackknife_blocks": 20,
    "n_iter": 2,
    "intercept": "free",
}
_QC_WARNING_THRESHOLDS = {
    # Absolute LDSC warning thresholds are not meaningful on tiny panels.
    "minimum_warning_variants": 10_000,
    "h2_interval": [0.0, 1.0],
    "intercept_interval": [0.8, 1.2],
    "minimum_mean_chi2": 0.9,
    "maximum_ratio": 0.5,
    "minimum_qc_retained_fraction": 0.5,
    "minimum_reference_match_fraction": 0.5,
    "maximum_screen_drop_fraction": 0.01,
    "minimum_screen_warning_variants": 1_000,
    "minimum_af_corr": 0.85,
}


def _screen_parameters():
    """Canonical mandatory-screen settings, independent of sampler seed."""
    return {
        "rounds": 4, "window": 1000, "threshold": 29.72,
        "eigenvalue_floor": 1e-3, "seed": 0,
        "ncores": 1, "verbose": False,
    }


def _ldsc_qc_identity(panel):
    """Semantic identity of the reference-wide scores used before DENTIST."""
    return {
        "m_snps": int(panel.m_snps),
        "score_sha256": panel.score_sha256.lower(),
        "definition": panel.definition.strip(),
        "source": panel.source.strip(),
        "source_sha256": (panel.source_sha256.lower()
                          if panel.source_sha256 is not None else None),
        "algorithm": panel.algorithm.strip(),
        "correction": panel.correction.strip(),
        "parameters": dict(_QC_LDSC_PARAMETERS),
    }


def _run_trait_ldsc_qc(trait, panel):
    """Fast, free-intercept univariate LDSC on aligned pre-screen rows."""
    import numpy as np
    from ldpred3 import ldsc_h2

    identity = _ldsc_qc_identity(panel)
    with np.errstate(over="ignore"):
        chisq = np.square(trait.z)
    threshold = np.maximum(
        _QC_LDSC_PARAMETERS["chi2_cap"],
        _QC_LDSC_PARAMETERS["chi2_n_scale"] * trait.n_eff)
    keep = np.isfinite(chisq) & (chisq <= threshold)
    n_input = int(len(trait))
    n_regression = int(np.count_nonzero(keep))
    diagnostic = {
        "identity": identity,
        "status": "unavailable",
        "n_aligned_variants": n_input,
        "n_regression_variants": n_regression,
        "n_chi2_excluded": n_input - n_regression,
        "h2": None, "h2_se": None,
        "intercept": None, "intercept_se": None,
        "mean_chi2": None, "ratio": None,
        "used_for_filtering": False,
        "used_for_h2_init": False,
    }
    if n_regression < 2:
        diagnostic["error"] = (
            "fewer than two aligned variants passed the LDSC-only "
            "chi-square cap")
        return diagnostic
    indices = trait.indices[keep]
    try:
        result = ldsc_h2(
            chisq[keep], panel.scores[indices],
            trait.n_eff[keep], m_snps=panel.m_snps,
            n_blocks=min(
                _QC_LDSC_PARAMETERS["max_jackknife_blocks"], n_regression),
            n_iter=_QC_LDSC_PARAMETERS["n_iter"])
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        diagnostic["error"] = str(exc)
        return diagnostic
    if not np.all(np.isfinite(
            [result.h2, result.intercept, result.mean_chisq])):
        diagnostic["error"] = (
            "univariate LDSC returned a non-finite h2, intercept, or mean "
            "chi-square estimate")
        return diagnostic
    diagnostic.update({
        "status": "available",
        "h2": float(result.h2),
        "h2_se": float(result.h2_se),
        "intercept": float(result.intercept),
        "intercept_se": float(result.intercept_se),
        "mean_chi2": float(result.mean_chisq),
        "ratio": float(result.ratio),
        "intercept_minus_one": float(result.intercept - 1.0),
        "flags": {
            "h2_nonpositive": bool(result.h2 <= 0.0),
            "h2_above_one": bool(result.h2 > 1.0),
            "intercept_nonpositive": bool(result.intercept <= 0.0),
        },
    })
    return _json_safe(diagnostic)


def _assess_trait_quality(trait, ldsc, *, screen=None):
    """Flag gross attrition or implausible univariate LDSC diagnostics.

    These are transparent triage heuristics, not additional variant filters.
    Structural input failures still raise at the preparation/reference
    boundaries; a noisy or unusual LDSC estimate warns and remains available
    for inspection rather than vetoing a potentially legitimate phenotype.
    """
    qc = trait.log.get("qc") or {}
    harmonize = trait.log.get("harmonize") or {}
    n_input = int(qc.get("n_input") or 0)
    n_qc = int(qc.get("n_kept") or 0)
    n_sumstats = int(harmonize.get("n_sumstats") or n_qc)
    n_usable = int(screen["n_input"]) if screen is not None else int(len(trait))
    qc_fraction = n_qc / n_input if n_input else None
    match_fraction = n_usable / n_sumstats if n_sumstats else None
    issues = []
    if (qc_fraction is not None and qc_fraction <
            _QC_WARNING_THRESHOLDS["minimum_qc_retained_fraction"]):
        issues.append(
            f"sum-statistics QC retained only {qc_fraction:.1%} of input rows")
    if (match_fraction is not None and match_fraction <
            _QC_WARNING_THRESHOLDS["minimum_reference_match_fraction"]):
        issues.append(
            f"only {match_fraction:.1%} of post-QC rows aligned to the LD "
            "reference")

    n_regression = int(ldsc.get("n_regression_variants") or 0)
    ldsc_evaluated = (
        ldsc.get("status") == "available"
        and n_regression >= _QC_WARNING_THRESHOLDS["minimum_warning_variants"])
    if ldsc.get("status") != "available":
        issues.append(
            "the pre-DENTIST univariate LDSC check was unavailable: "
            + str(ldsc.get("error") or "unknown regression failure"))
    elif ldsc_evaluated:
        h2 = ldsc.get("h2")
        intercept = ldsc.get("intercept")
        mean_chi2 = ldsc.get("mean_chi2")
        ratio = ldsc.get("ratio")
        lo, hi = _QC_WARNING_THRESHOLDS["h2_interval"]
        if h2 is None or not lo < h2 <= hi:
            issues.append(
                f"univariate LDSC h2={h2!r} lies outside ({lo:g}, {hi:g}]")
        lo, hi = _QC_WARNING_THRESHOLDS["intercept_interval"]
        if intercept is None or not lo <= intercept <= hi:
            issues.append(
                f"univariate LDSC intercept={intercept!r} lies outside "
                f"[{lo:g}, {hi:g}]")
        if (mean_chi2 is None or mean_chi2 <
                _QC_WARNING_THRESHOLDS["minimum_mean_chi2"]):
            issues.append(
                f"mean chi-square={mean_chi2!r} is below "
                f"{_QC_WARNING_THRESHOLDS['minimum_mean_chi2']:g}")
        if (ratio is not None and ratio >
                _QC_WARNING_THRESHOLDS["maximum_ratio"]):
            issues.append(
                f"LDSC attenuation ratio={ratio:.3g} exceeds "
                f"{_QC_WARNING_THRESHOLDS['maximum_ratio']:g}")

    screen_fraction = None
    if screen is not None:
        n_screen_input = int(screen["n_input"])
        screen_fraction = (
            int(screen["n_dropped"]) / n_screen_input
            if n_screen_input else None)
        if (n_screen_input >=
                _QC_WARNING_THRESHOLDS["minimum_screen_warning_variants"]
                and screen_fraction >
                _QC_WARNING_THRESHOLDS["maximum_screen_drop_fraction"]):
            issues.append(
                f"the LD-consistency screen dropped {screen_fraction:.1%} "
                "of aligned variants")
    return {
        "status": "warning" if issues else (
            "pass" if ldsc_evaluated else "not evaluated at genome scale"),
        "warnings": issues,
        "n_input": n_input,
        "n_after_qc": n_qc,
        "n_usable": n_usable,
        "qc_retained_fraction": qc_fraction,
        "reference_match_fraction": match_fraction,
        "screen_drop_fraction": screen_fraction,
        "ldsc_thresholds_evaluated": bool(ldsc_evaluated),
        "thresholds": dict(_QC_WARNING_THRESHOLDS),
    }


def _assess_allele_frequency(af_corr) -> dict:
    """Classify GWAS/reference allele-frequency agreement for both traits."""
    threshold = float(_QC_WARNING_THRESHOLDS["minimum_af_corr"])
    values = {}
    failed = []
    unavailable = []
    for trait in ("trait1", "trait2"):
        value = (af_corr or {}).get(trait)
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            value = float("nan")
        values[trait] = value if math.isfinite(value) else None
        if not math.isfinite(value):
            unavailable.append(trait)
        elif value < threshold:
            failed.append(trait)
    if failed:
        status = "warning"
        summary = (
            "Allele-frequency mismatch: " + ", ".join(
                f"{trait} correlation={values[trait]:.3f}" for trait in failed)
            + f" is below the {threshold:g} safety threshold; weights are "
              "withheld because allele inversion or reference-population "
              "mismatch is plausible.")
    elif unavailable:
        status = "not_evaluated"
        summary = (
            "Allele-frequency agreement was not evaluable for "
            + ", ".join(unavailable)
            + "; fewer than 10 varying finite EAF/reference-AF pairs may be "
              "available.")
    else:
        status = "pass"
        summary = (
            f"Both GWAS EAF correlations met the {threshold:g} reference-AF "
            "threshold.")
    return {
        "status": status,
        "critical": bool(failed),
        "correlations": values,
        "failed_traits": failed,
        "unavailable_traits": unavailable,
        "threshold": threshold,
        "summary": summary,
    }


def _screen_parallelism():
    """Whether this process may run two DENTIST eigensolvers concurrently."""
    from bipred._ldpred3_compat import _blas_pool_safe, _blas_runtime_info

    threads, nested_safe = _blas_runtime_info()
    safe = bool(_blas_pool_safe(True))
    if safe:
        reason = "single-threaded reentrant BLAS confirmed"
    elif threads is None:
        reason = "BLAS reentrancy could not be confirmed"
    elif threads != 1:
        reason = f"loaded BLAS uses {threads} threads"
    else:
        reason = "OpenMP-layer BLAS is unsafe for concurrent eigensolvers"
    return safe, reason, {
        "concurrent": safe, "blas_threads": threads,
        "blas_reentrant": nested_safe, "reason": reason,
    }


def _required_ld_score_rows(cache, root, *, cache_sha256, n_variants,
                            cache_indices, fitted_shape, panel=None):
    """Load the selected reference's scores and gather exact fitted rows.

    Reference integrity is deliberately outside the optional LDSC-regression
    boundary: a missing, corrupt, or misaligned panel cannot silently change
    the sampler initialization.
    """
    import numpy as np

    if panel is None:
        panel = caches.load_or_create_ld_score_panel(
            cache, root, cache_sha256=cache_sha256,
            n_variants=int(n_variants))
    elif (panel.cache_sha256 != cache_sha256
          or int(panel.m_snps) != int(n_variants)):
        raise ValueError(
            "preloaded LD-score panel does not match the selected LD "
            "reference generation")
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


def _input_work_guard(path, *, max_rows, max_expanded_bytes) -> dict:
    """Bound decompressed input work before a parser allocates full columns."""
    path = Path(path)
    max_rows = int(max_rows)
    max_expanded_bytes = int(max_expanded_bytes)
    if max_rows <= 0 or max_expanded_bytes <= 0:
        raise ValueError("input work limits must be positive")
    opener = gzip.open if path.name.lower().endswith((".gz", ".bgz")) else open
    expanded = newlines = 0
    final_byte = b""
    try:
        with opener(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                expanded += len(chunk)
                if expanded > max_expanded_bytes:
                    raise ValueError(
                        f"{path.name}: decompressed input exceeds the "
                        f"{max_expanded_bytes / 1024 ** 3:.3g} GiB limit")
                newlines += chunk.count(b"\n")
                final_byte = chunk[-1:]
                # One line is the header; a final unterminated data row is
                # checked after EOF.
                if max(newlines - 1, 0) > max_rows:
                    raise ValueError(
                        f"{path.name}: input exceeds the {max_rows:,}-row "
                        "limit")
    except ValueError:
        raise
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError(f"{path.name}: cannot inspect compressed input: {exc}") \
            from exc
    lines = newlines + int(expanded > 0 and final_byte != b"\n")
    rows = max(lines - 1, 0)
    if rows > max_rows:
        raise ValueError(
            f"{path.name}: input exceeds the {max_rows:,}-row limit")
    return {"rows": rows, "expanded_bytes": expanded,
            "max_rows": max_rows,
            "max_expanded_bytes": max_expanded_bytes}


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
        repo = Path(__file__).parent.parent
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            text=True, timeout=5).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo,
            text=True, timeout=5)
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=repo, timeout=5)
        untracked = [line[3:] for line in status.splitlines()
                     if line.startswith("?? ")]
        return {
            "commit": commit,
            "dirty": bool(status.strip()),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_files": untracked,
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None,
                "tracked_diff_sha256": None, "untracked_files": None}


def _distribution_source(name: str) -> dict:
    """Installed-distribution identity, including PEP 610 VCS metadata."""
    try:
        distribution = importlib_metadata.distribution(name)
    except importlib_metadata.PackageNotFoundError:
        return {"version": None, "direct_url": None}
    direct_url = None
    try:
        text = distribution.read_text("direct_url.json")
        if text:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                direct_url = parsed
    except (OSError, ValueError):
        pass
    return {"version": distribution.version, "direct_url": direct_url}


def _warning_rows(stage, caught):
    return [{"stage": stage, "category": item.category.__name__,
             "message": str(item.message)} for item in caught]


def _warnings_are_critical(rows, divergence=None):
    """Whether estimates must be quarantined.

    The structured divergence flag is authoritative when available; warning
    text remains as the compatibility path for results from older fit code.
    """
    critical_phrases = (
        "do not interpret",
        "appears to have diverged",
        "implausible bivariate fit",
        "invalid bivariate fit result",
    )
    return bool(divergence and divergence.get("flagged")) or any(
        bool(item.get("critical"))
        or any(phrase in item.get("message", "").lower()
               for phrase in critical_phrases)
        for item in rows)


def _fit_result_issues(result, joint) -> list:
    """Structural validity checks that must pass before releasing weights."""
    import numpy as np

    issues = []
    h2 = np.asarray(joint.get("h2"), dtype=float)
    if h2.shape != (2,) or not np.all(np.isfinite(h2)) \
            or np.any(h2 <= 0.0) or np.any(h2 > 1.0):
        issues.append("h2 must contain two finite values in (0, 1]")
    rg = joint.get("rg")
    if not _is_finite_number(rg) or not -1.0 <= float(rg) <= 1.0:
        issues.append("rg must be finite and lie in [-1, 1]")
    p = joint.get("p")
    if not _is_finite_number(p) or not 0.0 <= float(p) <= 1.0:
        issues.append("causal fraction p must be finite and lie in [0, 1]")
    pi = joint.get("pi")
    if pi is not None:
        pi = np.asarray(pi, dtype=float)
        if (pi.shape != (4,) or not np.all(np.isfinite(pi))
                or np.any(pi < 0.0) or np.any(pi > 1.0)
                or not math.isclose(float(pi.sum()), 1.0,
                                    rel_tol=1e-6, abs_tol=1e-6)):
            issues.append(
                "the four-state overlap mixture pi must be finite, bounded, "
                "and sum to one")
    if (not isinstance(joint.get("retained_iterations"), numbers.Integral)
            or isinstance(joint.get("retained_iterations"), bool)
            or int(joint["retained_iterations"]) <= 0):
        issues.append("no post-burn-in fit iterations were retained")
    beta_lengths = []
    for name in ("beta1_est", "beta2_est"):
        if not hasattr(result, name):
            continue
        values = np.asarray(getattr(result, name), dtype=float)
        beta_lengths.append(int(values.size) if values.ndim == 1 else None)
        if values.ndim != 1 or values.size == 0 or not np.all(
                np.isfinite(values)):
            issues.append(f"{name} contains no finite prediction vector")
    fitted_m = None
    if len(beta_lengths) == 2:
        if (None not in beta_lengths and beta_lengths[0] > 0
                and beta_lengths[0] == beta_lengths[1]):
            fitted_m = beta_lengths[0]
        else:
            issues.append("trait prediction vectors have different lengths")
    noise = joint.get("noise_scale")
    if noise is not None:
        noise = np.asarray(noise, dtype=float)
        if noise.shape != (2,) or not np.all(np.isfinite(noise)) \
                or np.any(noise <= 0.0):
            issues.append("noise_scale must contain two finite positive values")
    mixer = joint.get("mixer") or {}
    for name, value in mixer.items():
        values = np.asarray(value, dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            issues.append(f"MiXeR overlap field {name} is non-finite")
    strict_mixer = joint.get("pi") is not None or fitted_m is not None
    if strict_mixer:
        required = ("polygenicity", "n_causal", "n_shared", "frac_shared",
                    "rho_beta", "rg_from_overlap")
        missing = [name for name in required if name not in mixer]
        if missing:
            issues.append("MiXeR overlap summary lacks " + ", ".join(missing))
        else:
            poly = np.asarray(mixer["polygenicity"], dtype=float)
            counts = np.asarray(mixer["n_causal"], dtype=float)
            shared = float(mixer["n_shared"])
            frac = float(mixer["frac_shared"])
            rho = float(mixer["rho_beta"])
            overlap_rg = float(mixer["rg_from_overlap"])
            if poly.shape != (2,) or np.any(poly < 0.0) \
                    or np.any(poly > 1.0):
                issues.append(
                    "MiXeR polygenicity must contain two values in [0, 1]")
            if counts.shape != (2,) or np.any(counts < 0.0):
                issues.append(
                    "MiXeR n_causal must contain two non-negative totals")
            elif shared < 0.0 or shared > float(counts.min()) + 1e-6:
                issues.append(
                    "MiXeR n_shared must lie between zero and both n_causal "
                    "totals")
            if not 0.0 <= frac <= 1.0:
                issues.append("MiXeR frac_shared must lie in [0, 1]")
            if not -1.0 <= rho <= 1.0:
                issues.append("MiXeR rho_beta must lie in [-1, 1]")
            if not -1.0 <= overlap_rg <= 1.0:
                issues.append("MiXeR rg_from_overlap must lie in [-1, 1]")
            if counts.shape == (2,) and fitted_m is not None:
                count_tol = max(1e-6, fitted_m * 1e-6)
                if np.any(counts > fitted_m + count_tol):
                    issues.append(
                        "MiXeR n_causal exceeds the fitted variant panel")
                if poly.shape == (2,) and not np.allclose(
                        counts, poly * fitted_m, rtol=1e-6,
                        atol=count_tol):
                    issues.append(
                        "MiXeR n_causal is inconsistent with polygenicity "
                        "and fitted panel size")
                minimum = float(counts.min())
                expected_frac = shared / minimum if minimum > 0.0 else 0.0
                if not math.isclose(frac, expected_frac, rel_tol=1e-6,
                                    abs_tol=1e-8):
                    issues.append(
                        "MiXeR frac_shared is inconsistent with its counts")
                union_fraction = float(poly.sum() - shared / fitted_m)
                if _is_finite_number(p) and not math.isclose(
                        float(p), union_fraction, rel_tol=1e-6,
                        abs_tol=1e-8):
                    issues.append(
                        "MiXeR overlap is inconsistent with causal fraction p")
                if (pi is not None and pi.shape == (4,)
                        and poly.shape == (2,)):
                    expected_poly = np.array(
                        [pi[1] + pi[3], pi[2] + pi[3]])
                    if (not np.allclose(poly, expected_poly, rtol=1e-6,
                                        atol=1e-8)
                            or not math.isclose(
                                shared / fitted_m, float(pi[3]),
                                rel_tol=1e-6, abs_tol=1e-8)):
                        issues.append(
                            "MiXeR overlap is inconsistent with mixture pi")
    return issues


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


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


def _n_basis(options, trait) -> str:
    """Auditable description of the sample-size values actually fitted."""
    meta = options.get(f"catalog{trait}") or {}
    per_variant_safe = bool(
        meta.get("has_per_variant_n")
        and meta.get("sample_size_safe_for_effective_n", True))
    if per_variant_safe and not options.get(
            f"catalog_n_user_supplied{trait}"):
        transform = meta.get("sample_size_transform")
        if transform is not None:
            return (
                "per-variant Catalog sample-size pattern rescaled to "
                f"ancestry-matched effective N (factor "
                f"{transform.get('factor', 1.0):.6g}; source median "
                f"{transform.get('source_median', 0):,.0f}; target "
                f"{transform.get('target_effective_n', 0):,.0f})")
        return ("per-variant n column in the harmonised GWAS Catalog "
                f"file ({meta.get('per_variant_n_usable_frac', 0):.1%} "
                "usable among retained rows)")
    if options.get(f"n_eff{trait}") is not None:
        suffix = " (explicit user override)" if options.get(
            f"catalog_n_user_supplied{trait}") else ""
        return f"constant effective N{suffix}"
    if options.get(f"n_cases{trait}") is not None:
        suffix = " (explicit user override)" if options.get(
            f"catalog_n_user_supplied{trait}") else ""
        return f"4/(1/n_cases + 1/n_controls){suffix}"
    return "per-variant N column detected in the uploaded file"


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


_REFERENCE_POPULATION_TOKENS = {
    "eur": "European", "europe": "European", "european": "European",
    "afr": "African", "africa": "African", "african": "African",
    "eas": "East Asian", "eastasia": "East Asian",
    "sas": "South Asian", "southasia": "South Asian",
}


def _reference_population(cache_key) -> str | None:
    """Population explicitly encoded in an LD-reference registry key.

    The registry currently stores only key/label/path.  Until it has a
    first-class ancestry field, accept automatic Catalog N only for keys that
    say what population they represent; an arbitrary name is not evidence.
    """
    key = str(cache_key or "").strip().lower()
    if not key:
        return None
    compact = re.sub(r"[^a-z0-9]+", "-", key).strip("-")
    tokens = set(compact.split("-"))
    candidates = {
        population for token, population in _REFERENCE_POPULATION_TOKENS.items()
        if token in tokens
    }
    if {"east", "asian"} <= tokens:
        candidates.add("East Asian")
    if {"south", "asian"} <= tokens:
        candidates.add("South Asian")
    return candidates.pop() if len(candidates) == 1 else None


def _population_agrees(catalog_population, reference_population) -> bool:
    """Whether Catalog sample metadata and the LD reference explicitly agree."""
    if not isinstance(catalog_population, str) \
            or not isinstance(reference_population, str):
        return False
    return catalog_population.strip().casefold() == \
        reference_population.strip().casefold()


def _catalog_n_requires_explicit(options, trait, info,
                                 population_agreement) -> bool:
    """Whether using this Catalog file would otherwise assume ancestry."""
    if options.get(f"catalog_n_user_supplied{trait}"):
        return False
    auto_scalar_retained = (
        not info.get("has_per_variant_n")
        or not info.get("sample_size_safe_for_effective_n", True)
    ) and any(options.get(name) is not None for name in (
        f"n_eff{trait}", f"n_cases{trait}", f"n_controls{trait}"))
    source_requires_interpretation = (
        info.get("n_source_kind") in ("reported_total", "unknown")
        and not info.get("sample_size_safe_for_effective_n", True))
    return bool(not population_agreement and (
        auto_scalar_retained or source_requires_interpretation))


class _TraitPipelineCancelled(RuntimeError):
    """Internal cooperative stop after the counterpart trait has failed."""


def _progress_sink(stage, *, trait=None, phase=None, cancel=None):
    """Turn library progress events into job-status updates.

    ``bipred._progress`` deliberately lets a callback's exception propagate,
    which is the right default for a library. Here it must not cost an
    otherwise healthy fit: a status write that fails is dropped, and the
    stage carries on unreported rather than dying. ``_Stages.progress``
    throttles to 1 Hz, so a per-sweep or per-block event stream costs one
    clock read until the second is up.
    """
    def sink(event):
        if cancel is not None and cancel.is_set():
            raise _TraitPipelineCancelled(
                f"trait{trait} stopped because its counterpart failed")
        fields = dict(event)
        if trait is not None:
            fields["trait"] = trait
        if phase is not None:
            fields["phase"] = phase
        try:
            stage.progress(**fields)
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
    # This is a method invariant in stage schema 4, not a user-tunable
    # sensitivity option. It also upgrades a queued schema-2 job safely.
    opt["screen"] = True
    cache = caches.cache_path(opt["cache_key"], root)
    reference_population = _reference_population(opt["cache_key"])
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
            n_user_supplied = bool(
                opt.get(f"catalog_n_user_supplied{trait}"))
            population_agreement = _population_agrees(
                meta.get("sample_size_population"), reference_population)
            target_effective_n = (
                meta.get("n_eff")
                if not n_user_supplied and population_agreement else None)
            target_total_n = (
                meta.get("n_total_selected")
                if not n_user_supplied and population_agreement else None)
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
                    target_effective_n=target_effective_n,
                    target_total_n=target_total_n,
                    on_bytes=on_bytes,
                    on_filter=on_filter, on_wait=on_wait)
            except Exception as exc:
                raise ValueError(
                    f"{job['labels'][f'trait{trait}']} "
                    f"({meta['accession']}): summary-statistics preparation "
                    f"failed: {exc}")
            info["reference_population"] = reference_population
            info["sample_size_population_agreement"] = population_agreement
            if _catalog_n_requires_explicit(
                    opt, trait, info, population_agreement):
                catalog_population = meta.get("sample_size_population")
                raise ValueError(
                    f"{job['labels'][f'trait{trait}']} "
                    f"({meta['accession']}): automatic Catalog sample-size "
                    f"metadata identify {catalog_population or 'no explicit'} "
                    "ancestry, while the selected LD-reference key identifies "
                    f"{reference_population or 'no explicit'} ancestry; "
                    "supply N_eff or case/control counts explicitly")
            if (info.get("n_source_kind") in (
                        "reported_total", "unknown")
                    and not info.get(
                        "sample_size_safe_for_effective_n", True)
                    and not n_user_supplied
                    and opt.get(f"n_eff{trait}") is None
                    and opt.get(f"n_cases{trait}") is None):
                raise ValueError(
                    f"{job['labels'][f'trait{trait}']} "
                    f"({meta['accession']}): the deposited "
                    f"{info.get('n_source_column') or 'sample-size'} column "
                    "is a reported total, but the Catalog metadata does not "
                    "identify an ancestry-matched effective N; supply N_eff "
                    "or case/control counts explicitly")
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
    from bipred.prepare import _cache_variant_table
    from ldpred3.harmonize import _variant_indices
    from ldpred3.sumstats import detect_columns
    from . import prepared_store
    prepared_ld = None
    prep = None
    traits = {}
    prepared_info = {}
    input_sha256 = {}
    screen_params = _screen_parameters()
    screen_outcomes = {}
    warning_rows = {}
    cancel_traits = threading.Event()
    screen_lock = threading.Lock()
    try:
        stage.progress(step="Load shared LD reference and LD scores",
                       phase="prepare")
        prepared_ld, ld_sha256 = _load_stable_ld_cache(
            cache, root, expected_sha256=expected_ld_sha256)
        ld_score_panel = caches.load_or_create_ld_score_panel(
            cache, root, cache_sha256=ld_sha256,
            n_variants=int(len(prepared_ld.variant_ids)))
        ldsc_qc_identity = _ldsc_qc_identity(ld_score_panel)
        # Avoid two cold workers transiently building the reference-sized ID
        # dictionary. The table and its index are immutable thereafter.
        _variant_indices(_cache_variant_table(prepared_ld))
        parallel_screen, screen_reason, screen_execution = \
            _screen_parallelism()
        stage.detail(
            "prepare",
            "Loaded one shared LD reference and its precomputed LD scores; "
            "both trait pipelines are starting independently.",
            traits={}, ld_sha256=ld_sha256, published=False,
            parallel=True, shared_ld_reference=True,
            pre_dentist_ldsc=True)

        def trait_pipeline(trait, path):
            key = f"trait{trait}"
            label = job["labels"][key]
            catalog = opt.get(f"catalog{trait}")
            accession = catalog.get("accession") if catalog else None
            built_here = {"value": False}

            def checkpoint():
                if cancel_traits.is_set():
                    raise _TraitPipelineCancelled(
                        f"{key} stopped because its counterpart failed")

            checkpoint()
            stage.progress(
                trait=trait, accession=accession, phase="prepare",
                step=f"Validate input work budget and hash trait {trait}")
            input_work = _input_work_guard(
                path,
                max_rows=opt.get("max_input_rows", 50_000_000),
                max_expanded_bytes=opt.get(
                    "max_input_expanded_bytes", 16 * 1024 ** 3),
            )
            try:
                detect_columns(str(path), **opt[f"columns{trait}"])
            except Exception as exc:
                raise ValueError(
                    f"{key} ({label}): cannot interpret columns: {exc}") \
                    from exc
            physical_sha256 = _sha256(path)
            logical_sha256 = (
                catalog.get("normalised_sha256") if catalog
                else physical_sha256)
            if not logical_sha256:
                raise ValueError(
                    f"{key}: input lacks a logical content hash; refusing "
                    "unsafe prepared-data reuse")
            spec = prepared_store.semantic_spec(
                logical_input_sha256=logical_sha256,
                ld_sha256=ld_sha256,
                n_semantics=_n_semantics(opt, trait),
                columns=opt[f"columns{trait}"], qc=True, qc_params={},
                screen=True, screen_params=screen_params,
                pre_dentist_ldsc=ldsc_qc_identity)

            def build():
                built_here["value"] = True
                checkpoint()
                stage.progress(
                    trait=trait, accession=accession, phase="prepare",
                    step=f"QC, harmonize, and run univariate LDSC for trait "
                         f"{trait}")
                unscreened = prepare_trait_sumstats(
                    prepared_ld, str(path),
                    n_eff=opt[f"n_eff{trait}"],
                    n_cases=opt[f"n_cases{trait}"],
                    n_controls=opt[f"n_controls{trait}"],
                    columns=opt[f"columns{trait}"], label=key,
                    progress=None)
                checkpoint()
                ldsc_qc = _run_trait_ldsc_qc(
                    unscreened, ld_score_panel)
                pre_quality = _assess_trait_quality(unscreened, ldsc_qc)
                unscreened.log["pre_dentist_ldsc"] = ldsc_qc
                unscreened.log["qc_assessment"] = pre_quality
                stage.finish_prepare_trait(trait, {
                    "n_usable": int(len(unscreened)),
                    "qc_status": pre_quality["status"],
                    "warnings": list(pre_quality["warnings"]),
                    "ldsc_h2": ldsc_qc.get("h2"),
                    "ldsc_intercept": ldsc_qc.get("intercept"),
                    "input_rows": input_work["rows"],
                    "input_expanded_bytes": input_work["expanded_bytes"],
                })

                stage.activate("screen")
                if parallel_screen:
                    acquired = False
                else:
                    acquired = screen_lock.acquire(blocking=False)
                    if not acquired:
                        stage.progress(
                            trait=trait, accession=accession, phase="screen",
                            screen_waiting=True, reason=screen_reason)
                        while not screen_lock.acquire(timeout=0.25):
                            checkpoint()
                        acquired = True
                try:
                    checkpoint()
                    stage.progress(
                        trait=trait, accession=accession, phase="screen",
                        step=f"LD-consistency screen for trait {trait}")
                    screened = screen_prepared_trait(
                        prepared_ld, unscreened, **screen_params,
                        progress=_progress_sink(
                            stage, trait=trait, phase="screen",
                            cancel=cancel_traits))
                finally:
                    if acquired:
                        screen_lock.release()
                record = screened.log["ld_consistency_screen"]
                final_quality = _assess_trait_quality(
                    screened, ldsc_qc, screen=record)
                screened.log["qc_assessment"] = final_quality
                return screened

            def on_wait(seconds):
                checkpoint()
                stage.progress(
                    trait=trait, accession=accession, phase="prepare",
                    prepared_waiting=seconds)

            # The prepared store captures builder warnings for persistence;
            # this outer thread-local buffer receives the replay exactly once
            # for both a new artifact and a cache hit.
            with prepared_store._warning_capture() as caught:
                warnings.simplefilter("always")
                screened, reused = prepared_store.get_or_build(
                    root, spec, label=key, builder=build, on_wait=on_wait)
            ldsc_qc = screened.log["pre_dentist_ldsc"]
            quality = _assess_trait_quality(
                screened, ldsc_qc,
                screen=screened.log["ld_consistency_screen"])
            # Recompute policy flags from validated primary counts/estimates;
            # cached advisory text is never trusted as scientific input.
            screened.log["qc_assessment"] = quality
            if not built_here["value"]:
                stage.finish_prepare_trait(trait, {
                    "n_usable": int(ldsc_qc["n_aligned_variants"]),
                    "qc_status": quality["status"],
                    "warnings": list(quality["warnings"]),
                    "ldsc_h2": ldsc_qc.get("h2"),
                    "ldsc_intercept": ldsc_qc.get("intercept"),
                    "input_rows": input_work["rows"],
                    "input_expanded_bytes": input_work["expanded_bytes"],
                    "reused": True,
                })
                stage.activate("screen")

            prepared_key = prepared_store.key_for(spec)
            record = screened.log["ld_consistency_screen"]
            info = {
                "n_input": int(record["n_input"]),
                "n_kept": int(record["n_kept"]),
                "n_dropped": int(record["n_dropped"]),
                "qc_status": quality.get("status"),
                "qc_warnings": list(quality.get("warnings") or []),
                "prepared_key": prepared_key,
                "prepared_reused": bool(reused),
                "prepared_scope": _PREPARED_SCOPE,
            }
            stage.progress(
                trait=trait, accession=accession, phase="screen",
                prepared_source=("stored screened trait" if reused else
                                 "screened and stored"))
            runtime_warnings = _warning_rows(f"QC trait {trait}", caught)
            runtime_warnings.extend({
                "stage": f"QC trait {trait}", "category": "QCWarning",
                "message": f"{key} summary-statistics QC: {message}",
            } for message in quality["warnings"])
            return {
                "trait": screened,
                "input_sha256": physical_sha256,
                "screen": info,
                "prepared": {
                    "logical_sha256": logical_sha256,
                    "prepared_key": prepared_key,
                    "prepared_reused": bool(reused),
                    "prepared_scope": _PREPARED_SCOPE,
                    "prepared_numerical_environment":
                        spec["numerical_environment"],
                    "ld_score_qc_identity": ldsc_qc_identity,
                    "input_work": input_work,
                },
                "catalog_update": ({
                    "prepared_key": prepared_key,
                    "prepared_reused": bool(reused),
                    "prepared_scope": _PREPARED_SCOPE,
                } if catalog is not None else None),
                "warnings": runtime_warnings,
            }

        pool = ThreadPoolExecutor(max_workers=2)
        pending = {}
        catalog_updates = {}
        failure = None
        try:
            for trait, path in ((1, ss1), (2, ss2)):
                pending[pool.submit(trait_pipeline, trait, path)] = trait
            for future in as_completed(pending):
                trait = pending[future]
                try:
                    outcome = future.result()
                except BaseException as exc:
                    failure = exc
                    cancel_traits.set()
                    for other in pending:
                        if other is not future:
                            other.cancel()
                    break
                key = f"trait{trait}"
                traits[key] = outcome["trait"]
                input_sha256[key] = outcome["input_sha256"]
                screen_outcomes[key] = outcome["screen"]
                prepared_info[key] = outcome["prepared"]
                warning_rows[key] = outcome["warnings"]
                catalog_update = outcome["catalog_update"]
                if catalog_update is not None:
                    # Workers still serialize live progress through ``job``.
                    # Defer changes to its nested options until every worker
                    # has joined, so JSON serialization never races a dict
                    # mutation in the coordinator thread.
                    catalog_updates[key] = catalog_update
                ordered = {
                    name: screen_outcomes[name]
                    for name in ("trait1", "trait2")
                    if name in screen_outcomes
                }
                summary = "; ".join(
                    f"trait {name[-1]}: {value['n_kept']:,} of "
                    f"{value['n_input']:,} kept ({value['qc_status']})"
                    for name, value in ordered.items())
                stage.detail(
                    "screen", summary + ".", traits=ordered,
                    mandatory=True, parameters=screen_params,
                    prepared_scope=_PREPARED_SCOPE,
                    execution=screen_execution)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if failure is not None:
            raise failure
        for key in ("trait1", "trait2"):
            captured_warnings.extend(warning_rows.get(key, []))
            if key in catalog_updates:
                opt[f"catalog{key[-1]}"].update(catalog_updates[key])
        jobs.save_job(root, job)
    except BaseException as exc:
        if prepared_ld is not None:
            prepared_ld.close()
            prepared_ld = None
        if isinstance(exc, Exception):
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
                prepared_ld, traits["trait1"], traits["trait2"],
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
    af_quality = _assess_allele_frequency(prep.log.get("af_corr") or {})
    if af_quality["status"] != "pass":
        captured_warnings.append({
            "stage": "pair",
            "category": "QCWarning",
            "message": af_quality["summary"],
            "critical": af_quality["critical"],
        })
    try:
        # Top-level scalar counts drive the status page; the nested per-trait
        # QC and harmonization logs feed the results page's per-step report.
        munge = {key: int(value) for key, value in prep.log.items()
                 if isinstance(value, numbers.Integral)
                 and not isinstance(value, bool)}
        munge["screen"] = bool(prep.log.get("screen"))
        munge["af_corr"] = _json_safe(prep.log.get("af_corr") or {})
        munge["af_quality"] = _json_safe(af_quality)
        for trait in ("trait1", "trait2"):
            tlog = prep.log.get(trait) or {}
            screen_log = tlog.get("ld_consistency_screen") or {}
            n_usable = screen_log.get("n_kept")
            munge[trait] = {
                "qc_enabled": bool(tlog.get("qc_enabled")),
                "qc": _json_safe(tlog.get("qc") or {}),
                "harmonize": _json_safe(tlog.get("harmonize") or {}),
                "pre_dentist_ldsc": _json_safe(
                    tlog.get("pre_dentist_ldsc") or {}),
                "qc_assessment": _json_safe(
                    tlog.get("qc_assessment") or {}),
                "ld_consistency_screen": _json_safe(screen_log),
                "prepared": _json_safe(prepared_info[trait]),
                # Post-screen variants usable before the cross-trait
                # intersection. Pre-screen indices are not persisted.
                "n_usable": int(n_usable) if n_usable is not None else None,
            }
        _write_json_atomic(job_dir / "munge.json", json.dumps(munge, indent=1))
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
                fitted_shape=prep.beta_hat1.shape,
                panel=ld_score_panel)
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
        ld_score_panel = None
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
        try:
            joint["mixer_uncertainty"] = _json_safe(
                res.mixer_iterate_summary())
            joint["mixer_uncertainty_basis"] = (
                "empirical variability across retained hyperparameter "
                "iterates; not a convergence certificate or a confidence/"
                "credible interval")
        except (AttributeError, TypeError, ValueError, FloatingPointError):
            # Older compatible ldpred3/bipred seams may not retain the traces.
            # The point estimate remains available and provenance states that
            # no iterate summary was produced.
            joint["mixer_uncertainty"] = None
            joint["mixer_uncertainty_basis"] = (
                "unavailable: retained hyperparameter iterates were not "
                "exposed by this fit")
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

        for issue in _fit_result_issues(res, joint):
            captured_warnings.append({
                "stage": "fit",
                "category": "RuntimeWarning",
                "message": f"Invalid bivariate fit result: {issue}.",
                "critical": True,
            })
        critical = _warnings_are_critical(captured_warnings, divergence)
        # Persist the quarantine verdict on the job itself: the supervisor's
        # catalog track record reads job state, not result.json, and a
        # critically flagged fit must not be recorded as a working accession.
        job["valid_for_interpretation"] = not critical
        jobs.save_job(root, job)
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

    n_info = {
        "trait1": _n_summary(prep.n_eff1, _n_basis(opt, 1)),
        "trait2": _n_summary(prep.n_eff2, _n_basis(opt, 2)),
    }
    diagnostics = {
        "valid_for_interpretation": not critical,
        "critical": critical,
        "warnings": captured_warnings,
        "weights_withheld": bool(opt["weights"] and critical),
        "divergence": divergence,
        "allele_frequency": _json_safe(af_quality),
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
            "installed_distributions": {
                "bipred": _distribution_source("bipred"),
                "ldpred3": _distribution_source("ldpred3"),
            },
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
            "trait_pipeline_workers": 2,
            "screen_execution": screen_execution,
            "pre_dentist_ldsc": ldsc_qc_identity,
            "pre_dentist_ldsc_scope": (
                "post-QC, LD-aligned, before trait-local DENTIST"),
            "pre_dentist_ldsc_chi2_cap_scope": "regression only",
            "qc_warning_thresholds": dict(_QC_WARNING_THRESHOLDS),
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
                    "input_work": prepared_info[f"trait{trait}"][
                        "input_work"],
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
                              "sample", "sample_size_population",
                              "sample_size_design", "n_eff", "n_cases",
                              "n_controls", "n_total_selected",
                              "kept", "seen", "effect_from", "sha256",
                              "normalised_sha256", "source",
                              "remote_etag", "remote_last_modified",
                              "prepared_key", "prepared_reused",
                              "prepared_scope",
                              "n_source_column", "n_source_kind",
                              "sample_size_transform",
                              "sample_size_safe_for_effective_n",
                              "reference_population",
                              "sample_size_population_agreement",
                              "has_per_variant_n",
                              "per_variant_n_usable_frac")}
               for t in (1, 2) if opt.get(f"catalog{t}")}
    if catalog:
        result["provenance"]["catalog"] = catalog
    _write_json_atomic(job_dir / "result.json",
                       json.dumps(_json_safe(result), indent=1, allow_nan=False))


def main(argv=None) -> int:
    job_dir = Path((argv or sys.argv[1:])[0]).resolve()
    root = job_dir.parent.parent
    job = jobs.load_job(root, job_dir.name)
    if job is None:
        print(f"no job.json in {job_dir}", file=sys.stderr)
        return 2
    if job.get("status") != "launching":
        # The supervisor owns the lifecycle claim. If it already failed this
        # job (handshake grace or runtime limit expired during interpreter
        # startup), running anyway would resurrect a state the user was told
        # is terminal.
        print(f"job {job_dir.name} is not launching (status "
              f"{job.get('status')!r}); refusing to run", file=sys.stderr)
        return 2
    job["status"] = "running"
    # Keep the supervisor's launch-claim ``started``: the lease wrapper's
    # watchdog already anchored its deadline to it, and resetting it here
    # would give the supervisor a later timeout than the watchdog's hard kill.
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
