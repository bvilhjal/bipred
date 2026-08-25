"""Fit driver for one web job: ``python -m webapp.runner <job dir>``.

Runs validate -> harmonize -> ldsc -> fit -> (optional) weights, updating
``job.json`` after every stage so the status page shows progress, and writes
``result.json`` / ``munge.json`` on success. Any unhandled exception fails the
job with a user-readable message; the full traceback lands in ``runner.log``.
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
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

from . import caches, jobs


class _Stages:
    """Per-stage progress: current stage + elapsed seconds, in job.json."""

    def __init__(self, root, job):
        self.root, self.job = root, job
        self.t0 = None

    def start(self, name):
        self.job["stage"] = name
        self.t0 = time.time()
        jobs.save_job(self.root, self.job)

    def done(self, name):
        self.job["stages"][name] = round(time.time() - self.t0, 3)
        self.job["progress"] = None
        jobs.save_job(self.root, self.job)

    def progress(self, **fields):
        """Intra-stage progress (e.g. download bytes), throttled to 1 Hz."""
        now = time.time()
        if now - getattr(self, "_last_progress", 0.0) < 1.0:
            return
        self._last_progress = now
        fields.setdefault("mb_s", None)
        self.job["progress"] = fields
        jobs.save_job(self.root, self.job)


def _to_float_list(values):
    return [float(v) for v in values]


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


def _warnings_are_critical(rows):
    return any("do not interpret" in item["message"].lower()
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

    def coverage():
        if not computed:
            from . import gwascat
            union = set(keep_ids)
            covers = {fingerprint: cache_key}
            for entry in caches.real_registry(root):
                path = Path(entry["path"])
                try:
                    other = caches.sha256_cached(path)
                    if other in covers:
                        continue
                    union |= gwascat.cache_ids(path)
                except (OSError, ValueError):
                    continue    # an unreadable extra reference is not fatal
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
    cache = caches.cache_path(opt["cache_key"], root)
    stage = _Stages(root, job)

    if opt.get("gcst1") or opt.get("gcst2"):
        stage.start("download")
        from . import gwascat
        keep_ids = gwascat.cache_ids(cache)
        fingerprint = caches.sha256_cached(cache)
        coverage = _coverage_thunk(root, cache, opt["cache_key"], keep_ids,
                                   fingerprint)
        for trait in (1, 2):
            meta = opt.get(f"catalog{trait}")
            if not meta:
                continue
            dest = job_dir / f"trait{trait}.gcst.tsv.gz"
            t0 = time.time()

            def on_bytes(n, meta=meta, trait=trait, t0=t0):
                elapsed = time.time() - t0
                stage.progress(
                    trait=trait, accession=meta["accession"], bytes=n,
                    total=meta.get("remote_bytes") or 0,
                    mb_s=round(n / elapsed / 1048576, 1) if elapsed else None)

            def on_filter(n, meta=meta, trait=trait):
                stage.progress(trait=trait, accession=meta["accession"],
                               filtering=n)

            def on_wait(seconds, meta=meta, trait=trait):
                stage.progress(trait=trait, accession=meta["accession"],
                               waiting=seconds)
            try:
                info = gwascat.fetch_filtered(
                    meta["url"], dest, accession=meta["accession"], root=root,
                    keep_ids=keep_ids, fingerprint=fingerprint,
                    coverage=coverage, on_bytes=on_bytes,
                    on_filter=on_filter, on_wait=on_wait)
            except Exception as exc:
                raise ValueError(
                    f"{job['labels'][f'trait{trait}']} "
                    f"({meta['accession']}): download failed: {exc}")
            job["files"][f"sumstats{trait}"] = dest.name
            meta.update(kept=info["kept"], seen=info["seen"],
                        effect_from=info["effect_from"],
                        sha256=info["sha256"],
                        source="stored copy" if info["reused"] else "download",
                        has_per_variant_n=info["has_per_variant_n"],
                        per_variant_n_usable_frac=info[
                            "per_variant_n_usable_frac"])
            # Catalog metadata is an advisory scalar fallback.  When the
            # deposited file has an almost-complete positive per-variant N
            # column, preserve it unless the submitter explicitly overrode N.
            if info["has_per_variant_n"] and not opt.get(
                    f"catalog_n_user_supplied{trait}"):
                opt[f"n_eff{trait}"] = None
                opt[f"n_cases{trait}"] = None
                opt[f"n_controls{trait}"] = None
        jobs.save_job(root, job)
        stage.done("download")

    ss1 = job_dir / job["files"]["sumstats1"]
    ss2 = job_dir / job["files"]["sumstats2"]

    stage.start("validate")
    from ldpred3.sumstats import detect_columns
    for path, overrides, label in (
            (ss1, opt["columns1"], job["labels"]["trait1"]),
            (ss2, opt["columns2"], job["labels"]["trait2"])):
        try:
            detect_columns(str(path), **overrides)
        except Exception as exc:
            raise ValueError(f"{label}: cannot interpret columns: {exc}")
    stage.done("validate")

    stage.start("harmonize")
    from bipred import prepare_bivariate_sumstats
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prep = prepare_bivariate_sumstats(
                str(cache), str(ss1), str(ss2),
                n_eff1=opt["n_eff1"], n_eff2=opt["n_eff2"],
                n_cases1=opt["n_cases1"], n_controls1=opt["n_controls1"],
                n_cases2=opt["n_cases2"], n_controls2=opt["n_controls2"],
                columns1=opt["columns1"], columns2=opt["columns2"],
                screen=opt["screen"], screen_rounds=4, screen_window=1000,
                screen_threshold=29.72, screen_eigenvalue_floor=1e-3,
                screen_seed=opt["seed"], screen_ncores=1,
                progress=_progress_sink(stage))
    except Exception as exc:
        # Let the track record blame the right catalog accession when the
        # failure is one trait's; joint failures stay unattributed.
        attributed = _attribute_to_catalog(exc, job)
        if attributed is not None:
            raise attributed from exc
        raise
    captured_warnings.extend(_warning_rows("harmonize", caught))
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
            n_usable = tlog.get("n_matched")
            munge[trait] = {
                "qc_enabled": bool(tlog.get("qc_enabled")),
                "qc": _json_safe(tlog.get("qc") or {}),
                "harmonize": _json_safe(tlog.get("harmonize") or {}),
                # Variants usable for fitting: QC-passed, on the reference,
                # with finite effect/SE/N (the ok mask in _align_one).
                "n_usable": int(n_usable) if n_usable is not None else None,
            }
        (job_dir / "munge.json").write_text(json.dumps(munge, indent=1))
        stage.done("harmonize")

        stage.start("ldsc")
        from bipred.ldsc import ldsc_rg
        from ldpred3 import ld_scores
        try:
            report = _progress_sink(stage)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # Neither call reports from inside, so name the two steps.
                report({"step": "LD scores", "done": 0, "total": 2,
                        "unit": "step"})
                ell = ld_scores(prep.blocks)
                report({"step": "LDSC regression", "done": 1, "total": 2,
                        "unit": "step"})
                r = ldsc_rg(prep.beta_hat1, prep.beta_hat2, ell,
                            prep.n_eff1, prep.n_eff2)
            captured_warnings.extend(_warning_rows("ldsc", caught))
            ldsc_out = {"rg": float(r.rg), "rg_se": float(r.rg_se),
                        "gcov": float(r.gcov),
                        "gcov_intercept": float(r.gcov_intercept),
                        "h2": _to_float_list(r.h2),
                        "scope": "unfiltered fitted-panel moment diagnostic"}
        except Exception as exc:                # LDSC is a bonus; fit decides
            ldsc_out = {"error": str(exc)}
        stage.done("ldsc")

        stage.start("fit")
        from bipred import ldpred3_auto_bivariate_blocks
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = ldpred3_auto_bivariate_blocks(
                prep.blocks, prep.beta_hat1, prep.beta_hat2,
                prep.n_eff1, prep.n_eff2,
                cross_corr=opt.get("cross_corr", 0.0),
                seed=opt["seed"], burn_in=opt["burn_in"],
                num_iter=opt["num_iter"], ncores=1,
                progress=_progress_sink(stage))
        captured_warnings.extend(_warning_rows("fit", caught))
        mixer = {key: (_to_float_list(value) if isinstance(value, tuple)
                       else float(value))
                 for key, value in res.mixer.items()}
        joint = {
            "h2": _to_float_list(res.h2),
            "rg": float(res.rg),
            "p": float(res.p),
            "pi": _to_float_list(res.pi) if res.pi is not None else None,
            "mixer": mixer,
            "noise_scale": (_to_float_list(res.noise_scale)
                            if res.noise_scale is not None else None),
            "retained_iterations": res.retained_iterations,
            "stopped_early": bool(res.stopped_early),
        }
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
        stage.done("fit")

        critical = _warnings_are_critical(captured_warnings)
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
            "cache_sha256": caches.sha256_cached(cache),
            "seed": opt["seed"],
            "burn_in": opt["burn_in"],
            "num_iter": opt["num_iter"],
            "cross_corr": opt.get("cross_corr", 0.0),
            "screen": opt["screen"],
            "screen_parameters": {
                "rounds": 4, "window": 1000, "threshold": 29.72,
                "eigenvalue_floor": 1e-3, "seed": opt["seed"],
                "ncores": 1,
            },
            "fitted_variant_ids_sha256": _ids_sha256(prep.id),
            "sample_size": n_info,
            "inputs": {
                f"trait{trait}": {
                    "filename": job["files"][f"sumstats{trait}"],
                    "sha256": _sha256(job_dir / job["files"][
                        f"sumstats{trait}"]),
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
            "stages": job["stages"],
        },
    }
    catalog = {f"trait{t}": {k: opt[f"catalog{t}"].get(k) for k in
                             ("accession", "trait", "pmid", "n_basis",
                              "kept", "seen", "effect_from", "sha256",
                              "has_per_variant_n",
                              "per_variant_n_usable_frac")}
               for t in (1, 2) if opt.get(f"catalog{t}")}
    if catalog:
        result["provenance"]["catalog"] = catalog
    (job_dir / "result.json").write_text(
        json.dumps(_json_safe(result), indent=1))


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
                        finished=time.time(), error=str(exc))
        return 1
    jobs.update_job(root, job["id"], status="done", stage=None,
                    finished=time.time())
    return 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())
