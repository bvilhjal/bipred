"""FastAPI app for the bipred web service.

Layout: upload -> queued job -> subprocess fit (``webapp.runner``) -> results.
The supervisor below is a deliberate no-Redis choice: a small asyncio loop
launches runner subprocesses up to a concurrency cap, reaps dead ones, and
purges expired jobs. It fits the one-VM deployment this service targets; swap
in a real queue if it ever outgrows that.

Run from a checkout: ``python -m webapp`` (see webapp/README.md).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Read-only reuse of ldpred3's column-alias table, so the browser-side header
# preview and the runner's detect_columns can never drift apart.
from ldpred3.sumstats import _ALIASES as SUMSTATS_ALIASES

from . import caches, catalog_evidence, demo, gwascat, jobs

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
TEMPLATES.env.filters["format_datetime"] = (
    lambda fmt, ts: time.strftime(fmt, time.localtime(ts)))


def _f3(value):
    """Three-decimal formatting that tolerates None/NaN (e.g. failed LDSC)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{out:.3f}" if math.isfinite(out) else "—"


TEMPLATES.env.filters["f3"] = _f3

# One numerical thread per runner: deterministic numerics, and N concurrent
# jobs never oversubscribe the host (same pins as the benchmark suite).
_THREAD_PINS = {
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}

_DOWNLOADS = {
    "result": "result.json",
    "munge": "munge.json",
    "weights1": "weights1.tsv",
    "weights2": "weights2.tsv",
}

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _config() -> dict:
    return {
        "concurrency": int(os.environ.get("BIPRED_WEB_CONCURRENCY", "2")),
        "max_upload": int(os.environ.get("BIPRED_WEB_MAX_UPLOAD_MB", "500"))
        * 1024 * 1024,
        "ttl_days": float(os.environ.get("BIPRED_WEB_TTL_DAYS", "7")),
    }


def _float_or_none(value, name):
    value = (value or "").strip()
    if not value:
        return None
    try:
        out = float(value)
    except ValueError:
        raise ValueError(f"{name}: expected a number, got {value!r}")
    if not math.isfinite(out) or not (out > 0):
        raise ValueError(f"{name}: must be positive, got {value!r}")
    return out


def _bounded_int(value, default, name, minimum, maximum):
    value = (value or "").strip()
    if not value:
        return default
    try:
        out = int(value)
    except ValueError:
        raise ValueError(f"{name}: expected an integer, got {value!r}")
    if not minimum <= out <= maximum:
        raise ValueError(
            f"{name}: must be between {minimum:,} and {maximum:,}")
    return out


def _cross_corr(value):
    value = (value or "").strip()
    if not value:
        return 0.0
    try:
        out = float(value)
    except ValueError:
        raise ValueError(f"cross_corr: expected a number, got {value!r}")
    if not math.isfinite(out) or not -1.0 < out < 1.0:
        raise ValueError("cross_corr: must be finite and strictly between -1 and 1")
    return out


def parse_columns(text):
    """Parse ``FIELD=COLUMN`` pairs (comma/space separated), as in the CLI."""
    out = {}
    for item in re.split(r"[,\s]+", (text or "").strip()):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"column override {item!r} needs FIELD=COLUMN")
        field, column = (part.strip() for part in item.split("=", 1))
        if not field or not column:
            raise ValueError(f"column override {item!r} needs FIELD=COLUMN")
        out[field] = column
    return out


def _sample_size_options(form, trait):
    """Effective N, case/control counts, or no scalar (use an N column)."""
    n_eff = _float_or_none(form.get(f"n_eff{trait}"), f"n_eff{trait}")
    n_cases = _float_or_none(form.get(f"n_cases{trait}"), f"n_cases{trait}")
    n_controls = _float_or_none(
        form.get(f"n_controls{trait}"), f"n_controls{trait}")
    if n_eff is not None and (n_cases is not None or n_controls is not None):
        raise ValueError(
            f"trait {trait}: give either effective N or cases/controls")
    if n_eff is None and ((n_cases is None) != (n_controls is None)):
        raise ValueError(
            f"trait {trait}: give both cases and controls, or leave both blank")
    return n_eff, n_cases, n_controls


async def _save_upload(upload: UploadFile, dest: Path, cap: int) -> None:
    n = 0
    with open(dest, "wb") as fh:
        while True:
            chunk = await upload.read(1 << 20)
            if not chunk:
                break
            n += len(chunk)
            if n > cap:
                fh.close()
                dest.unlink(missing_ok=True)
                raise ValueError(
                    f"{upload.filename}: over the {cap // (1024 * 1024)} MB "
                    "per-file limit")
            fh.write(chunk)
    if n == 0:
        raise ValueError(f"{upload.filename or dest.name}: empty file")


def _index_context(app, error=None, form=None) -> dict:
    """Template context for the upload form (initial render and re-render)."""
    real_caches = caches.real_registry(app.state.root)
    return {"caches": real_caches,
            "default_key": real_caches[0]["key"] if real_caches else "",
            "has_real_cache": bool(real_caches),
            "max_mb": app.state.config["max_upload"] // (1024 * 1024),
            "ttl_days": app.state.config["ttl_days"],
            "aliases": SUMSTATS_ALIASES,
            "error": error,
            "form": form}


def _read_munge(root: Path, job_id: str) -> dict | None:
    path = jobs.job_dir(root, job_id) / "munge.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _figure_data(result: dict) -> dict | None:
    """Counts for the results-page variant figures (Venn + QC attrition).

    Returns None when the munge report lacks per-trait counts (jobs that
    predate the per-step report), so older result pages render as before.
    ``usable`` prefers the post-QC finite-effect count (``n_usable``) and
    falls back to the harmonization match count for jobs from before that
    key existed.
    """
    munge = result.get("munge") or {}
    kept = munge.get("n_kept")
    traits = {}
    for key in ("trait1", "trait2"):
        tlog = munge.get(key) or {}
        qc = tlog.get("qc") or {}
        harmonize = tlog.get("harmonize") or {}
        usable = tlog.get("n_usable")
        if usable is None:
            usable = harmonize.get("n_matched")
        n_input = qc.get("n_input") or harmonize.get("n_sumstats")
        if usable is None or not n_input or not kept:
            return None
        after_qc = qc.get("n_kept")
        traits[key] = {
            "input": int(n_input),
            "after_qc": int(after_qc) if after_qc is not None else int(n_input),
            "usable": int(usable),
            "only": max(int(usable) - int(kept), 0),
        }
    return {"traits": traits, "joint": int(kept),
            "screen_drop": int(munge.get("n_screen_drop") or 0)}


def _shown_stages(job) -> list:
    """Stages worth displaying; download/weights only exist when requested."""
    opt = job.get("options", {})
    return [s for s in jobs.STAGE_ORDER
            if (s != "weights" or opt.get("weights"))
            and (s != "download" or opt.get("gcst1") or opt.get("gcst2"))]


def _launch(app, job) -> None:
    root = app.state.root
    job_dir = jobs.job_dir(root, job["id"])
    # Persist the claim before Popen.  Without this transition a second sweep
    # (or a future multi-worker supervisor) can launch the same queued job.
    claimed = jobs.update_job(root, job["id"], status="launching",
                              started=time.time())
    if claimed is None:
        return
    env = dict(os.environ)
    env.update(_THREAD_PINS)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    log = open(job_dir / "runner.log", "ab")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "webapp.runner", str(job_dir)],
            stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env=env)
    except Exception as exc:
        jobs.update_job(root, job["id"], status="failed",
                        finished=time.time(), error=f"could not start fit: {exc}")
        raise
    finally:
        log.close()
    app.state.procs[job["id"]] = proc


def _record_catalog_outcome(root, job) -> None:
    """Fold a finished job's catalog accessions into the track-record registry.

    Called once per job, when the supervisor reaps its runner process: a
    *done* job marks its accessions as working; a *failed* job marks only
    those its error message blames (the runner prefixes per-trait failures —
    download, or a harmonization failure naming one trait — with the
    accession).
    """
    opt = job.get("options", {})
    error = job.get("error") or ""
    for trait in (1, 2):
        meta = opt.get(f"catalog{trait}")
        if not meta:
            continue
        accession = meta.get("accession")
        base = {"trait": meta.get("trait"), "pmid": meta.get("pmid"),
                "n_eff": meta.get("n_eff"), "n_cases": meta.get("n_cases"),
                "n_controls": meta.get("n_controls")}
        if job["status"] == "done":
            gwascat.record_accession(root, accession, True,
                                     kept=meta.get("kept"),
                                     seen=meta.get("seen"), **base)
        elif accession and accession in error and gwascat.worth_recording(error):
            gwascat.record_accession(root, accession, False,
                                     reason=error[:300], **base)


def _sweep_once(app) -> None:
    root = app.state.root
    for job_id, proc in list(app.state.procs.items()):
        rc = proc.poll()
        if rc is None:
            continue
        del app.state.procs[job_id]
        job = jobs.load_job(root, job_id)
        if job is not None and job["status"] in ("launching", "running"):
            # The runner sets done/failed itself before exiting; reaching this
            # branch means it died without writing (OOM kill, segfault).
            jobs.update_job(root, job_id, status="failed",
                            finished=time.time(),
                            error=f"fit process exited unexpectedly (rc={rc}); "
                                  "see runner.log")
            job["status"] = "failed"
        if job is not None and job["status"] in ("done", "failed"):
            try:
                _record_catalog_outcome(root, job)
            except Exception:
                pass                    # bookkeeping must never kill a sweep
    slots = app.state.config["concurrency"] - len(app.state.procs)
    queued = [j for j in jobs.list_jobs(root) if j["status"] == "queued"]
    for job in sorted(queued, key=lambda j: j["created"]):
        if slots <= 0:
            break
        _launch(app, job)
        slots -= 1
    if time.time() - app.state.last_purge > 3600:
        app.state.last_purge = time.time()
        jobs.purge_jobs(root, app.state.config["ttl_days"])


async def _supervisor(app):
    while True:
        await asyncio.sleep(2)
        try:
            _sweep_once(app)
        except Exception:
            pass                            # never let the supervisor die


@asynccontextmanager
async def _lifespan(app):
    jobs.recover_interrupted_jobs(app.state.root)
    demo.ensure_demo(app.state.root)
    task = asyncio.create_task(_supervisor(app))
    try:
        yield
    finally:
        task.cancel()
        for proc in list(app.state.procs.values()):
            proc.terminate()
        for proc in list(app.state.procs.values()):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def create_app() -> FastAPI:
    app = FastAPI(title="bipred", lifespan=_lifespan)
    app.state.root = jobs.data_root()
    app.state.config = _config()
    app.state.procs = {}
    app.state.last_purge = time.time()
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return TEMPLATES.TemplateResponse(
            request, "index.html", _index_context(app))

    @app.post("/jobs")
    async def submit(
            request: Request,
            sumstats1: UploadFile | None = File(None),
            sumstats2: UploadFile | None = File(None),
            label1: str = Form("Trait 1"), label2: str = Form("Trait 2"),
            n_eff1: str = Form(""), n_eff2: str = Form(""),
            n_cases1: str = Form(""), n_controls1: str = Form(""),
            n_cases2: str = Form(""), n_controls2: str = Form(""),
            cache_key: str = Form(""),
            seed: str = Form("0"), burn_in: str = Form("200"),
            num_iter: str = Form("200"),
            cross_corr: str = Form("0"),
            columns1: str = Form(""), columns2: str = Form(""),
            gcst1: str = Form(""), gcst2: str = Form(""),
            catalog_auto_n1: str = Form(""),
            catalog_auto_n2: str = Form(""),
            catalog_auto_label1: str = Form(""),
            catalog_auto_label2: str = Form(""),
            screen: str = Form(""), weights: str = Form("")):
        # Raw values kept for re-rendering the form when validation fails.
        form = {"label1": label1, "label2": label2,
                "n_eff1": n_eff1, "n_eff2": n_eff2,
                "n_cases1": n_cases1, "n_controls1": n_controls1,
                "n_cases2": n_cases2, "n_controls2": n_controls2,
                "cache_key": cache_key, "seed": seed, "burn_in": burn_in,
                "num_iter": num_iter, "cross_corr": cross_corr,
                "columns1": columns1,
                "columns2": columns2, "gcst1": gcst1, "gcst2": gcst2,
                "catalog_auto_n1": catalog_auto_n1,
                "catalog_auto_n2": catalog_auto_n2,
                "catalog_auto_label1": catalog_auto_label1,
                "catalog_auto_label2": catalog_auto_label2,
                "screen": screen, "weights": weights}
        try:
            if not cache_key:
                raise ValueError("No real LD reference is configured on this server")
            if cache_key == "demo":
                raise ValueError(
                    "The synthetic demo LD reference cannot be used with uploads; "
                    "use Run demo instead")
            options = {"cache_key": cache_key}
            labels = {}
            for trait, upload, accession in (
                    (1, sumstats1, gcst1), (2, sumstats2, gcst2)):
                has_file = upload is not None and bool(upload.filename)
                accession = accession.strip().upper()
                if has_file and accession:
                    raise ValueError(
                        f"trait {trait}: give either a file or a GCST "
                        "accession, not both")
                if not has_file and not accession:
                    raise ValueError(
                        f"trait {trait}: upload a file or give a GWAS "
                        "Catalog accession (GCST…)")
                if accession:
                    n_fields_present = any(
                        form[f"n_{k}{trait}"].strip()
                        for k in ("eff", "cases", "controls"))
                    n_was_autofilled = form.get(
                        f"catalog_auto_n{trait}") == "1"
                    n_user_supplied = n_fields_present and not n_was_autofilled
                    try:
                        meta = await asyncio.to_thread(
                            gwascat.resolve, accession, app.state.root)
                    except ValueError as exc:
                        if gwascat.worth_recording(exc):
                            gwascat.record_accession(
                                app.state.root, accession, False,
                                reason=str(exc)[:300])
                        raise
                    options[f"gcst{trait}"] = accession
                    options[f"catalog{trait}"] = meta
                    default_label = f"Trait {trait}"
                    if (not form[f"label{trait}"].strip()
                            or form[f"label{trait}"].strip() == default_label
                            or form.get(f"catalog_auto_label{trait}") == "1"):
                        form[f"label{trait}"] = meta["trait"]
                        form[f"catalog_auto_label{trait}"] = "1"
                    # Fill the sample size from catalog metadata only when
                    # the user left every N field for this trait empty.
                    if not n_fields_present or n_was_autofilled:
                        for key in ("eff", "cases", "controls"):
                            form[f"n_{key}{trait}"] = ""
                        if meta.get("n_cases") and meta.get("n_controls"):
                            form[f"n_cases{trait}"] = str(meta["n_cases"])
                            form[f"n_controls{trait}"] = str(meta["n_controls"])
                        elif meta.get("n_eff"):
                            form[f"n_eff{trait}"] = str(meta["n_eff"])
                        form[f"catalog_auto_n{trait}"] = "1"
                    options[f"catalog_n_user_supplied{trait}"] = n_user_supplied
                elif form.get(f"catalog_auto_n{trait}") == "1":
                    # The user switched from a looked-up accession to a file;
                    # stale advisory Catalog N must not override its N column.
                    for key in ("eff", "cases", "controls"):
                        form[f"n_{key}{trait}"] = ""
                    form[f"catalog_auto_n{trait}"] = ""
                if not accession and form.get(
                        f"catalog_auto_label{trait}") == "1":
                    form[f"label{trait}"] = f"Trait {trait}"
                    form[f"catalog_auto_label{trait}"] = ""
                labels[f"trait{trait}"] = form[f"label{trait}"].strip() \
                    or f"Trait {trait}"
            ne1, nc1, nco1 = _sample_size_options(form, 1)
            ne2, nc2, nco2 = _sample_size_options(form, 2)
            options.update({
                "n_eff1": ne1, "n_cases1": nc1, "n_controls1": nco1,
                "n_eff2": ne2, "n_cases2": nc2, "n_controls2": nco2,
                "seed": _bounded_int(seed, 0, "seed", 0, 2 ** 32 - 1),
                "burn_in": _bounded_int(
                    burn_in, 200, "burn-in", 0, 100_000),
                "num_iter": _bounded_int(
                    num_iter, 200, "iterations", 1, 100_000),
                "cross_corr": _cross_corr(cross_corr),
                "columns1": parse_columns(columns1),
                "columns2": parse_columns(columns2),
                "screen": bool(screen),
                "weights": bool(weights),
            })
            caches.cache_path(cache_key, app.state.root)
        except (ValueError, KeyError) as exc:
            return TEMPLATES.TemplateResponse(
                request, "index.html",
                _index_context(app, error=str(exc), form=form),
                status_code=400)

        job = jobs.create_job(
            app.state.root, options=options, labels=labels, status="staging")
        job_dir = jobs.job_dir(app.state.root, job["id"])
        try:
            for upload, key, prefix in (
                    (sumstats1, "sumstats1", "trait1"),
                    (sumstats2, "sumstats2", "trait2")):
                if upload is None or not upload.filename:
                    continue            # catalog trait: the runner downloads
                name = _SAFE_NAME.sub("_", upload.filename or "sumstats.tsv")
                dest = job_dir / f"{prefix}_{name}"
                await _save_upload(upload, dest, app.state.config["max_upload"])
                job["files"][key] = dest.name
        except ValueError as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            return TEMPLATES.TemplateResponse(
                request, "index.html",
                _index_context(app, error=str(exc), form=form),
                status_code=400)
        job["status"] = "queued"
        jobs.save_job(app.state.root, job)
        return RedirectResponse(f"/jobs/{job['id']}", status_code=303)

    @app.get("/catalog", response_class=HTMLResponse)
    def catalog_track_record(request: Request):
        """Canonical LDpred3 evidence plus accessions observed by this app."""
        evidence = catalog_evidence.load()
        registry = gwascat.accession_registry(app.state.root)
        good = {e["accession"]: dict(e) for e in evidence.get("good", [])}
        bad = {e["accession"]: dict(e) for e in evidence.get("bad", [])}
        for accession, local in registry.items():
            canonical_good = accession in good
            entry = good.get(accession) or bad.get(accession) or {
                "accession": accession, "profile": "server",
                "evidence": "observed by this bipred server"}
            entry.update({key: value for key, value in local.items()
                          if value is not None})
            entry["server_observed"] = True
            if local.get("works"):
                if not canonical_good:
                    entry["evidence"] = "completed by this bipred server"
                    entry["profile"] = "server"
                    entry.pop("reason", None)
                bad.pop(accession, None)
                good[accession] = entry
            elif accession not in good:
                bad[accession] = entry
        works = sorted(good.values(), key=lambda e: (e.get("trait") or "").lower())
        failed = sorted(bad.values(), key=lambda e: (e.get("trait") or "").lower())
        server_observed = sum(1 for e in works + failed
                              if e.get("server_observed"))
        return TEMPLATES.TemplateResponse(
            request, "catalog.html", {"works": works, "failed": failed,
                                      "evidence": evidence,
                                      "server_observed": server_observed})

    @app.get("/catalog/lookup")
    def catalog_lookup(accession: str = ""):
        """Metadata for one accession, for the form's live preview."""
        try:
            meta = gwascat.resolve(accession, app.state.root)
        except ValueError as exc:
            status = 404 if gwascat.worth_recording(exc) else 503
            return JSONResponse({"error": str(exc)}, status_code=status)
        return {key: meta.get(key) for key in
                ("accession", "trait", "title", "pmid", "n_eff",
                 "n_cases", "n_controls", "n_basis", "remote_bytes")}

    @app.get("/demo")
    def demo_job(request: Request):
        meta = demo.ensure_demo(app.state.root)
        truth = demo.demo_meta(app.state.root)
        options = {
            "cache_key": "demo",
            "n_eff1": truth["n_eff1"], "n_cases1": None, "n_controls1": None,
            "n_eff2": truth["n_eff2"], "n_cases2": None, "n_controls2": None,
            "seed": 0, "burn_in": 200, "num_iter": 200,
            "cross_corr": 0.0,
            "columns1": {}, "columns2": {},
            "screen": False, "weights": True,
        }
        job = jobs.create_job(
            app.state.root, options=options,
            labels={"trait1": "Demo trait 1", "trait2": "Demo trait 2"},
            status="staging")
        job_dir = jobs.job_dir(app.state.root, job["id"])
        for trait in (1, 2):
            shutil.copy(meta / f"trait{trait}.tsv",
                        job_dir / f"trait{trait}.tsv")
            job["files"][f"sumstats{trait}"] = f"trait{trait}.tsv"
        job["status"] = "queued"
        jobs.save_job(app.state.root, job)
        return RedirectResponse(f"/jobs/{job['id']}", status_code=303)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_status(request: Request, job_id: str):
        job = jobs.load_job(app.state.root, job_id)
        if job is None:
            return TEMPLATES.TemplateResponse(
                request, "error.html", {"message": "unknown job"},
                status_code=404)
        return TEMPLATES.TemplateResponse(
            request, "job.html",
            {"job": job, "munge": _read_munge(app.state.root, job_id),
             "stages": _shown_stages(job),
             "running": job["status"] in ("queued", "launching", "running")})

    @app.get("/jobs/{job_id}/status")
    def job_status_json(job_id: str):
        """Machine-readable job state; polled by the job page's live view."""
        job = jobs.load_job(app.state.root, job_id)
        if job is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        return {"id": job_id, "status": job["status"], "stage": job["stage"],
                "stages": job["stages"], "error": job["error"],
                "progress": job.get("progress"),
                "munge": _read_munge(app.state.root, job_id)}

    @app.get("/jobs/{job_id}/results", response_class=HTMLResponse)
    def results(request: Request, job_id: str):
        job = jobs.load_job(app.state.root, job_id)
        if job is None:
            return TEMPLATES.TemplateResponse(
                request, "error.html", {"message": "unknown job"},
                status_code=404)
        result_path = jobs.job_dir(app.state.root, job_id) / "result.json"
        if not result_path.exists():
            return RedirectResponse(f"/jobs/{job_id}", status_code=303)
        result = json.loads(result_path.read_text())
        return TEMPLATES.TemplateResponse(
            request, "results.html", {"job": job, "res": result,
                                      "figs": _figure_data(result)})

    @app.get("/jobs/{job_id}/download/{kind}")
    def download(job_id: str, kind: str):
        if jobs.load_job(app.state.root, job_id) is None:
            return RedirectResponse("/", status_code=303)
        name = _DOWNLOADS.get(kind)
        if name is None:
            return RedirectResponse(f"/jobs/{job_id}", status_code=303)
        path = jobs.job_dir(app.state.root, job_id) / name
        if not path.exists():
            return RedirectResponse(f"/jobs/{job_id}", status_code=303)
        return FileResponse(path, filename=name)

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run("webapp.app:app", host=os.environ.get("BIPRED_WEB_HOST",
                                                      "127.0.0.1"),
                port=int(os.environ.get("BIPRED_WEB_PORT", "8000")))


if __name__ == "__main__":                      # pragma: no cover
    main()
