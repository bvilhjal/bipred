"""Job store for the bipred web service.

One directory per job under ``<data root>/jobs/<id>/``; ``job.json`` inside it
is the single source of truth. Writes are atomic (tmp file + ``os.replace``),
so the web process can read job state at any moment without locking. The
fit itself runs in a subprocess (``webapp.runner``) that updates the same
file, which keeps server restarts survivable: a job whose runner died is
detected by the supervisor instead of hanging forever.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import time
from pathlib import Path

STATES = ("queued", "running", "done", "failed")
STAGE_ORDER = ("download", "validate", "harmonize", "ldsc", "fit", "weights")

# Files a job directory starts with; the runner adds result.json, munge.json,
# runner.log and optionally weights*.tsv.
JOB_JSON = "job.json"


def data_root() -> Path:
    """Root for all mutable web-service state (env ``BIPRED_WEB_DATA``)."""
    root = Path(os.environ.get("BIPRED_WEB_DATA", "webapp_data")).resolve()
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    return root


def _now() -> float:
    return time.time()


def create_job(root: Path, *, options: dict, labels: dict) -> dict:
    """Create a queued job with a fresh unguessable id and return it."""
    job_id = secrets.token_urlsafe(12)
    job_dir = root / "jobs" / job_id
    job_dir.mkdir(parents=True)
    job = {
        "id": job_id,
        "status": "queued",
        "stage": None,
        "stages": {},
        "created": _now(),
        "started": None,
        "finished": None,
        "error": None,
        "options": options,
        "labels": labels,
        "files": {},
        "pid": None,
    }
    save_job(root, job)
    return job


def job_dir(root: Path, job_id: str) -> Path:
    # Ids come from secrets.token_urlsafe; still refuse anything that could
    # escape the jobs directory before it touches the filesystem.
    if not job_id or any(c not in "-_0123456789abcdefghijklmnopqrstuvwxyz"
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in job_id):
        raise ValueError(f"invalid job id {job_id!r}")
    return root / "jobs" / job_id


def load_job(root: Path, job_id: str) -> dict | None:
    path = job_dir(root, job_id) / JOB_JSON
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def save_job(root: Path, job: dict) -> None:
    path = job_dir(root, job["id"]) / JOB_JSON
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(job, fh, indent=1)
    os.replace(tmp, path)


def update_job(root: Path, job_id: str, **fields) -> dict | None:
    job = load_job(root, job_id)
    if job is None:
        return None
    job.update(fields)
    save_job(root, job)
    return job


def list_jobs(root: Path) -> list[dict]:
    out = []
    base = root / "jobs"
    if not base.exists():
        return out
    for entry in sorted(base.iterdir()):
        job = load_job(root, entry.name)
        if job is not None:
            out.append(job)
    return out


def purge_jobs(root: Path, ttl_days: float) -> list[str]:
    """Delete finished/failed jobs older than ``ttl_days``; returns their ids."""
    if ttl_days <= 0:
        return []
    cutoff = _now() - ttl_days * 86400.0
    removed = []
    for job in list_jobs(root):
        if job["status"] in ("queued", "running"):
            continue
        if (job["finished"] or job["created"]) < cutoff:
            shutil.rmtree(job_dir(root, job["id"]), ignore_errors=True)
            removed.append(job["id"])
    return removed
