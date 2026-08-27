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
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Read-only reuse of ldpred3's column-alias table, so the browser-side header
# preview and the runner's detect_columns can never drift apart.
from ldpred3.sumstats import _ALIASES as SUMSTATS_ALIASES

from . import caches, catalog_evidence, demo, gwascat, jobs, prepared_store

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
TEMPLATES.env.filters["format_datetime"] = (
    lambda fmt, ts: time.strftime(fmt, time.localtime(ts)))


def _number(value, spec=".3g"):
    """Format one finite number; JSON null and non-finite values become an em dash."""
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return "—"
    if not math.isfinite(out):
        return "—"
    try:
        return format(out, spec)
    except (TypeError, ValueError):
        return "—"


def _f3(value):
    """Three-decimal compatibility formatter used throughout results pages."""
    return _number(value, ".3f")


TEMPLATES.env.filters["f3"] = _f3
TEMPLATES.env.filters["number"] = _number
LOGGER = logging.getLogger(__name__)

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

# Fewer retained draws cannot exercise the fit's trace diagnostic, and a
# vanishing burn-in is not a defensible public-web default.  These are minimum
# safety rails, not a claim that a single 40-draw chain proves convergence.
_MIN_WEB_BURN_IN = 50
_MIN_WEB_NUM_ITER = 40
_REQUEST_OVERHEAD = 1 << 20


class _RequestTooLarge(Exception):
    """Raised while consuming a chunked request that crosses the body cap."""


class _BodyLimitMiddleware:
    """Reject oversized multipart bodies before Starlette spools all of them."""

    def __init__(self, app, *, limit: int):
        self.app = app
        self.limit = int(limit)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = 0
        if declared > self.limit:
            await JSONResponse(
                {"error": "request body exceeds the server upload limit"},
                status_code=413,
            )(scope, receive, send)
            return

        seen = 0
        response_started = False

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.limit:
                    raise _RequestTooLarge
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestTooLarge:
            if response_started:  # pragma: no cover - handlers consume first
                raise
            await JSONResponse(
                {"error": "request body exceeds the server upload limit"},
                status_code=413,
            )(scope, receive, send)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer greater than zero") from None
    if value <= 0:
        raise ValueError(f"{name} must be an integer greater than zero")
    return value


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number greater than zero") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return value


def _nonnegative_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite non-negative number") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def _config() -> dict:
    concurrency = _positive_int_env("BIPRED_WEB_CONCURRENCY", 1)
    max_upload_mb = _positive_int_env("BIPRED_WEB_MAX_UPLOAD_MB", 500)
    max_upload = max_upload_mb * 1024 * 1024
    queue_max = _positive_int_env(
        "BIPRED_WEB_QUEUE_MAX", max(8, 4 * concurrency))
    timeout_hours = _positive_float_env("BIPRED_WEB_JOB_TIMEOUT_HOURS", 6)
    max_expanded_gb = _positive_float_env(
        "BIPRED_WEB_MAX_EXPANDED_GB", 16)
    return {
        "concurrency": concurrency,
        "queue_max": queue_max,
        "max_upload": max_upload,
        # Treat the configured value as the combined file budget.  This lets
        # the ASGI layer reject a declared oversized multipart body before the
        # form parser spools it; the same value is also a per-file backstop.
        "max_request": max_upload + _REQUEST_OVERHEAD,
        "max_input_rows": _positive_int_env(
            "BIPRED_WEB_MAX_ROWS", 50_000_000),
        "max_input_expanded_bytes": int(max_expanded_gb * 1024 ** 3),
        "job_timeout_s": timeout_hours * 3600.0,
        "ttl_days": _positive_float_env("BIPRED_WEB_TTL_DAYS", 7),
        "store_gb": _nonnegative_float_env("BIPRED_WEB_STORE_GB", 20),
        "prepared_gb": _nonnegative_float_env("BIPRED_WEB_PREPARED_GB", 20),
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
    try:
        with open(dest, "wb") as fh:
            while True:
                chunk = await upload.read(1 << 20)
                if not chunk:
                    break
                n += len(chunk)
                if n > cap:
                    raise ValueError(
                        f"{upload.filename}: over the "
                        f"{cap // (1024 * 1024)} MB per-file limit")
                fh.write(chunk)
        if n == 0:
            raise ValueError(f"{upload.filename or dest.name}: empty file")
    except BaseException:
        # CancelledError is a BaseException.  Remove private partial data, but
        # re-raise cancellation, shutdown signals, and I/O failures unchanged.
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _index_context(app, error=None, form=None) -> dict:
    """Template context for the upload form (initial render and re-render)."""
    real_caches = caches.real_registry(app.state.root)
    evidence = catalog_evidence.load()
    counts = evidence.get("counts", {}) if evidence.get("available") else {}
    return {"caches": real_caches,
            "default_key": real_caches[0]["key"] if real_caches else "",
            "has_real_cache": bool(real_caches),
            "max_mb": app.state.config["max_upload"] // (1024 * 1024),
            "ttl_days": app.state.config["ttl_days"],
            "min_burn_in": _MIN_WEB_BURN_IN,
            "min_num_iter": _MIN_WEB_NUM_ITER,
            "evidence_available": bool(evidence.get("available")),
            "evidence_completed": counts.get("good"),
            "evidence_rejected": counts.get("bad"),
            "evidence_total": (
                (counts.get("good", 0) + counts.get("bad", 0))
                if counts else None),
            "aliases": SUMSTATS_ALIASES,
            "error": error,
            "form": form}


def _read_munge(root: Path, job_id: str) -> dict | None:
    path = jobs.job_dir(root, job_id) / "munge.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _figure_data(result: dict, stage_schema: int | None = None) -> dict | None:
    """Counts for the results-page QC attrition figure.

    Returns None when the munge report lacks per-trait counts (jobs that
    predate the per-step report), so older result pages render as before.
    Schema-3 circles use each trait's post-screen count. Earlier jobs retain
    their historical post-QC/reference-aligned definition of ``usable``.
    """
    munge = result.get("munge") or {}
    kept = munge.get("n_kept")
    if stage_schema is None:
        stage_schema = int(
            (result.get("provenance") or {}).get("stage_schema") or 1)
    current = int(stage_schema) >= 3
    traits = {}
    for key in ("trait1", "trait2"):
        tlog = munge.get(key) or {}
        qc = tlog.get("qc") or {}
        harmonize = tlog.get("harmonize") or {}
        screen = tlog.get("ld_consistency_screen") or {}
        if current:
            on_reference = screen.get("n_input")
            after_screen = screen.get("n_kept")
        else:
            on_reference = tlog.get("n_usable")
            if on_reference is None:
                on_reference = harmonize.get("n_matched")
            after_screen = None
        usable = after_screen if current else on_reference
        n_input = qc.get("n_input") or harmonize.get("n_sumstats")
        if usable is None or on_reference is None or not n_input or not kept:
            return None
        after_qc = qc.get("n_kept")
        traits[key] = {
            "input": int(n_input),
            "after_qc": int(after_qc) if after_qc is not None else int(n_input),
            "on_reference": int(on_reference),
            "after_screen": (
                int(after_screen) if after_screen is not None else None),
            "usable": int(usable),
            "only": max(int(usable) - int(kept), 0),
        }
    return {
        "traits": traits,
        "joint": int(kept),
        "current_screen": current,
        # Schema 3 has two trait-local screens, not one joint drop.
        "screen_drop": (
            0 if current else int(munge.get("n_screen_drop") or 0)),
    }


def _mixer_figure_data(result: dict) -> dict | None:
    """Return internally consistent MiXeR overlap regions for the results UI."""
    mixer = ((result.get("joint") or {}).get("mixer") or {})
    totals = mixer.get("n_causal")
    if not isinstance(totals, (list, tuple)) or len(totals) != 2:
        return None
    try:
        trait1, trait2 = (float(totals[0]), float(totals[1]))
        shared = float(mixer.get("n_shared"))
    except (TypeError, ValueError, OverflowError):
        return None
    if (not all(math.isfinite(value) for value in (trait1, trait2, shared))
            or min(trait1, trait2) <= 0
            or shared < 0
            or shared > min(trait1, trait2) + 1e-9):
        return None
    return {
        "trait1_total": trait1,
        "trait2_total": trait2,
        "shared": shared,
        "trait1_only": max(trait1 - shared, 0.0),
        "trait2_only": max(trait2 - shared, 0.0),
        "fraction_trait1": shared / trait1 if trait1 else 0.0,
        "fraction_trait2": shared / trait2 if trait2 else 0.0,
    }


def _active_job_count(root: Path) -> int:
    return sum(job.get("status") in ("staging", "queued", "launching", "running")
               for job in jobs.list_jobs(root))


def _reserve_staging_job(app, *, options: dict, labels: dict) -> dict | None:
    """Atomically reserve one queue slot across async and sync endpoints."""
    with app.state.admission_lock:
        if _active_job_count(app.state.root) >= app.state.config["queue_max"]:
            return None
        return jobs.create_job(
            app.state.root, options=options, labels=labels, status="staging")


def _same_origin(request: Request) -> bool:
    """Reject browser cross-origin state changes while allowing non-browser clients."""
    origin = request.headers.get("origin")
    if not origin:
        return True
    given = urlsplit(origin)
    expected = request.url
    return (given.scheme.lower(), given.netloc.lower()) == (
        expected.scheme.lower(), expected.netloc.lower())


def _terminate_process(proc: subprocess.Popen) -> None:
    """Bounded termination for a runner owned by this supervisor."""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


def _runtime_limit(job: dict | None, app) -> float:
    value = (job or {}).get("runtime_limit_s")
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0):
        return float(app.state.config.get("job_timeout_s", 6 * 3600.0))
    return float(value)


def _job_timed_out(job: dict | None, app, now: float) -> bool:
    started = (job or {}).get("started")
    return (not isinstance(started, bool)
            and isinstance(started, (int, float))
            and now - float(started) > _runtime_limit(job, app))


def _runner_exit_error(job: dict | None, app, now: float, fallback: str) -> str:
    if _job_timed_out(job, app, now):
        hours = _runtime_limit(job, app) / 3600.0
        return (f"fit exceeded the configured runtime limit ({hours:.3g} h); "
                "see runner.log")
    return fallback


def _shown_stages(job) -> list:
    """User-facing stage definitions for a current or historical job."""
    return jobs.stage_definitions(job)


def _launch(app, job) -> None:
    root = app.state.root
    job_dir = jobs.job_dir(root, job["id"])
    # Persist the claim before Popen.  Without this transition a second sweep
    # (or a future multi-worker supervisor) can launch the same queued job.
    token = jobs.new_runner_token()
    claimed = jobs.update_job(
        root, job["id"], status="launching", started=time.time(), pid=None,
        pid_identity=None, runner_token=token,
        runtime_limit_s=app.state.config.get("job_timeout_s", 6 * 3600.0))
    if claimed is None:
        return
    env = dict(os.environ)
    env.update(_THREAD_PINS)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    log = open(job_dir / "runner.log", "ab")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "webapp.jobs", "run-runner", str(job_dir)],
            stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env=env)
    except Exception as exc:
        jobs.update_job(root, job["id"], status="failed",
                        finished=time.time(), error=f"could not start fit: {exc}")
        raise
    finally:
        log.close()
    app.state.procs[job["id"]] = proc
    # The lease wrapper repeats this before importing runner, ensuring runner's
    # in-memory job retains the creation identity across every stage update.
    jobs.update_job(root, job["id"], pid=proc.pid,
                    pid_identity=jobs.process_identity(proc.pid))


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


def _adopt_untracked_runners(app, now: float) -> None:
    """Discover a runner that crossed a web-process crash/launch boundary."""
    root = app.state.root
    tracked = set(app.state.procs) | set(app.state.orphans)
    for snapshot in jobs.list_jobs(root):
        if (snapshot.get("status") not in ("launching", "running")
                or snapshot["id"] in tracked):
            continue
        if (jobs.runner_is_verified(root, snapshot)
                or jobs.runner_handshake_pending(snapshot, now)):
            pid = snapshot.get("pid")
            app.state.orphans[snapshot["id"]] = (
                pid if isinstance(pid, int) and not isinstance(pid, bool)
                else 0)
            continue
        # A lease-free runner cannot write again.  Re-read so a completion
        # published after list_jobs() is never overwritten as failed.
        current = jobs.load_job(root, snapshot["id"])
        if current is not None and current.get("status") in (
                "launching", "running"):
            error = _runner_exit_error(
                current, app, now,
                "fit process has no live runner lease; see runner.log")
            jobs.fail_active_job(
                root, snapshot["id"], error=error, finished=now)


def _sweep_once(app) -> None:
    root = app.state.root
    now = time.time()
    if not hasattr(app.state, "orphans"):
        app.state.orphans = {}

    _adopt_untracked_runners(app, now)

    # A runner may survive an abrupt web-server restart. Its unguessable
    # kernel lease and process-creation identity jointly prove ownership.
    for job_id, recorded_pid in list(app.state.orphans.items()):
        job = jobs.load_job(root, job_id)
        if job is None:
            del app.state.orphans[job_id]
            continue
        verified = jobs.runner_is_verified(root, job)
        if job.get("status") in ("done", "failed"):
            if verified:
                # Runner writes its terminal state immediately before exit;
                # retain the slot until the lifetime lease is actually gone.
                continue
            del app.state.orphans[job_id]
            try:
                _record_catalog_outcome(root, job)
            except Exception:
                pass
            continue
        if verified:
            app.state.orphans[job_id] = job.get("pid") or recorded_pid
            continue
        if jobs.runner_handshake_pending(job, now):
            continue
        del app.state.orphans[job_id]
        current = jobs.load_job(root, job_id)
        if current is not None and current.get("status") in (
                "launching", "running"):
            error = _runner_exit_error(
                current, app, now,
                "fit process exited after server restart; see runner.log")
            current = jobs.fail_active_job(
                root, job_id, error=error, finished=now)
        if current is not None and current.get("status") in ("done", "failed"):
            try:
                _record_catalog_outcome(root, current)
            except Exception:
                pass

    for job_id, proc in list(app.state.procs.items()):
        rc = proc.poll()
        job = jobs.load_job(root, job_id)
        if rc is None and _job_timed_out(job, app, now):
            _terminate_process(proc)
            rc = proc.poll()
            error = _runner_exit_error(job, app, now, "fit timed out")
            job = jobs.fail_active_job(
                root, job_id, error=error, finished=now)
        if rc is None:
            continue
        del app.state.procs[job_id]
        job = jobs.load_job(root, job_id)
        if job is not None and job.get("status") in ("launching", "running"):
            # The runner sets done/failed itself before exiting; reaching this
            # branch means it died without publishing a terminal state.
            fallback = ("fit exceeded the configured runtime limit; see "
                        "runner.log" if rc == 124 else
                        f"fit process exited unexpectedly (rc={rc}); see "
                        "runner.log")
            error = _runner_exit_error(job, app, now, fallback)
            job = jobs.fail_active_job(
                root, job_id, error=error, finished=now)
        if job is not None and job.get("status") in ("done", "failed"):
            try:
                _record_catalog_outcome(root, job)
            except Exception:
                pass                    # bookkeeping must never kill a sweep
    slots = (app.state.config["concurrency"] - len(app.state.procs)
             - len(app.state.orphans))
    queued = [j for j in jobs.list_jobs(root) if j["status"] == "queued"]
    for job in sorted(queued, key=lambda j: j["created"]):
        if slots <= 0:
            break
        _launch(app, job)
        slots -= 1
    if time.time() - app.state.last_purge > 3600:
        app.state.last_purge = time.time()
        jobs.purge_jobs(root, app.state.config["ttl_days"])
        # Downloaded catalog files outlive the jobs that fetched them — that
        # is the point — so they get their own byte budget instead.
        gwascat.purge_store(root, app.state.config["store_gb"])
        prepared_store.purge(root, app.state.config["prepared_gb"])


async def _supervisor(app):
    while True:
        await asyncio.sleep(2)
        try:
            _sweep_once(app)
        except Exception:
            # Keep serving, but leave evidence instead of turning a wedged
            # scheduler into a charming little mystery.
            LOGGER.exception("bipred web supervisor sweep failed")


@asynccontextmanager
async def _lifespan(app):
    _, app.state.orphans = jobs.recover_interrupted_jobs(
        app.state.root, preserve_live=True)
    demo.ensure_demo(app.state.root)
    task = asyncio.create_task(_supervisor(app))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        for job_id, proc in list(app.state.procs.items()):
            _terminate_process(proc)
            jobs.fail_active_job(
                app.state.root, job_id,
                error="web server shut down while the fit was running",
                finished=time.time())
        app.state.procs.clear()


def create_app() -> FastAPI:
    app = FastAPI(title="bipred", lifespan=_lifespan)
    app.state.root = jobs.data_root()
    app.state.config = _config()
    app.state.procs = {}
    app.state.orphans = {}
    app.state.admission_lock = threading.Lock()
    app.state.last_purge = time.time()
    app.add_middleware(
        _BodyLimitMiddleware, limit=app.state.config["max_request"])
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
            weights: str = Form("")):
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
                "weights": weights}
        if _active_job_count(app.state.root) >= app.state.config["queue_max"]:
            return TEMPLATES.TemplateResponse(
                request, "index.html",
                _index_context(
                    app,
                    error=("The server queue is full; wait for a running job "
                           "to finish, then submit again."),
                    form=form,
                ),
                status_code=503,
                headers={"Retry-After": "30"},
            )
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
                    burn_in, 200, "burn-in", _MIN_WEB_BURN_IN, 100_000),
                "num_iter": _bounded_int(
                    num_iter, 200, "iterations", _MIN_WEB_NUM_ITER, 100_000),
                "cross_corr": _cross_corr(cross_corr),
                "columns1": parse_columns(columns1),
                "columns2": parse_columns(columns2),
                "screen": True,
                "weights": bool(weights),
                "max_input_rows": app.state.config["max_input_rows"],
                "max_input_expanded_bytes": app.state.config[
                    "max_input_expanded_bytes"],
            })
            caches.cache_path(cache_key, app.state.root)
        except (ValueError, KeyError) as exc:
            return TEMPLATES.TemplateResponse(
                request, "index.html",
                _index_context(app, error=str(exc), form=form),
                status_code=400)

        job = _reserve_staging_job(app, options=options, labels=labels)
        if job is None:
            return TEMPLATES.TemplateResponse(
                request, "index.html",
                _index_context(
                    app,
                    error=("The server queue filled while this submission "
                           "was being validated; try again shortly."),
                    form=form,
                ),
                status_code=503,
                headers={"Retry-After": "30"},
            )
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
            # Publishing the queued state is the atomic handoff to the
            # supervisor.  Until this replace succeeds, every file is private
            # staging data and is removed if the request fails or is cancelled.
            job["status"] = "queued"
            jobs.save_job(app.state.root, job)
        except ValueError as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            return TEMPLATES.TemplateResponse(
                request, "index.html",
                _index_context(app, error=str(exc), form=form),
                status_code=400)
        except BaseException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
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
                 "n_cases", "n_controls", "n_total_selected",
                 "n_total_reported", "n_basis", "sample_size_population",
                 "sample_size_design", "remote_bytes")}

    @app.post("/demo")
    def demo_job(request: Request):
        if not _same_origin(request):
            return TEMPLATES.TemplateResponse(
                request, "error.html", {"message": "cross-origin request refused"},
                status_code=403)
        if _active_job_count(app.state.root) >= app.state.config["queue_max"]:
            return TEMPLATES.TemplateResponse(
                request, "error.html",
                {"message": "The server queue is full; try the demo again later."},
                status_code=503, headers={"Retry-After": "30"})
        meta = demo.ensure_demo(app.state.root)
        truth = demo.demo_meta(app.state.root)
        options = {
            "cache_key": "demo",
            "n_eff1": truth["n_eff1"], "n_cases1": None, "n_controls1": None,
            "n_eff2": truth["n_eff2"], "n_cases2": None, "n_controls2": None,
            "seed": 0, "burn_in": 200, "num_iter": 200,
            "cross_corr": 0.0,
            "columns1": {}, "columns2": {},
            "screen": True, "weights": True,
            "max_input_rows": app.state.config["max_input_rows"],
            "max_input_expanded_bytes": app.state.config[
                "max_input_expanded_bytes"],
        }
        job = _reserve_staging_job(
            app, options=options,
            labels={"trait1": "Demo trait 1", "trait2": "Demo trait 2"})
        if job is None:
            return TEMPLATES.TemplateResponse(
                request, "error.html",
                {"message": "The server queue is full; try again later."},
                status_code=503, headers={"Retry-After": "30"})
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
                "active_stages": job.get("active_stages", []),
                "stage_schema": job.get("stage_schema", 1),
                "stages": job["stages"], "error": job["error"],
                "stage_details": job.get("stage_details", {}),
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
                                      "figs": _figure_data(
                                          result, job.get("stage_schema", 1)),
                                      "mixer_fig": _mixer_figure_data(result)})

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
