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

STATES = ("staging", "queued", "launching", "running", "done", "failed")

# Version 3 makes the mandatory LD-consistency screen a visible stage. Keep
# both earlier schemas so completed jobs retain the workflow they actually ran.
STAGE_SCHEMA = 3
STAGE_ORDER = (
    "acquire", "prepare", "screen", "pair", "ldsc", "fit", "weights")
STAGE_DEFINITIONS = {
    "acquire": {
        "label": "Get Catalog data",
        "description": (
            "Reuse stored Catalog files, or download and store missing files."
        ),
    },
    "prepare": {
        "label": "Prepare each trait",
        "description": (
            "Validate and read both inputs and the selected LD reference."
        ),
    },
    "screen": {
        "label": "Run LD-consistency screen",
        "description": (
            "Reuse a fully QC'd, harmonized, and screened trait artifact, or "
            "run QC, LD alignment, and the mandatory DENTIST-inspired "
            "trait-local screen before storing it."
        ),
    },
    "pair": {
        "label": "Combine the two traits",
        "description": (
            "Intersect the screened traits, check allele frequencies, and "
            "subset LD. Recomputed for every analysis."
        ),
    },
    "ldsc": {
        "label": "Run LD-score diagnostic",
        "description": (
            "Reuse the selected reference's precomputed LD scores, regress "
            "the paired GWAS rows, and initialize the sampler h2 values."
        ),
    },
    "fit": {
        "label": "Fit bivariate model",
        "description": "Estimate trait-specific and shared genetic architecture.",
    },
    "weights": {
        "label": "Write prediction weights",
        "description": "Create one SNP-weight file for each trait.",
    },
}

SCHEMA2_STAGE_ORDER = (
    "acquire", "prepare", "pair", "ldsc", "fit", "weights")
SCHEMA2_STAGE_DEFINITIONS = {
    "acquire": STAGE_DEFINITIONS["acquire"],
    "prepare": {
        "label": "Prepare each trait",
        "description": (
            "Check columns, run QC, align alleles, and retain variants in the "
            "selected LD reference. Catalog preparations can be reused."
        ),
    },
    "pair": {
        "label": "Combine the two traits",
        "description": (
            "Intersect prepared traits, check allele frequencies, subset LD, "
            "and run the optional LD-consistency screen when enabled. "
            "Recomputed for every analysis."
        ),
    },
    "ldsc": STAGE_DEFINITIONS["ldsc"],
    "fit": STAGE_DEFINITIONS["fit"],
    "weights": STAGE_DEFINITIONS["weights"],
}

LEGACY_STAGE_ORDER = (
    "download", "validate", "harmonize", "ldsc", "fit", "weights")
LEGACY_STAGE_DEFINITIONS = {
    "download": {
        "label": "Get Catalog data",
        "description": "Retrieve and filter Catalog summary statistics.",
    },
    "validate": {
        "label": "Check input columns",
        "description": "Recognize the required summary-statistics fields.",
    },
    "harmonize": {
        "label": "Prepare and combine traits",
        "description": (
            "QC and align both traits, then build their joint LD panel."
        ),
    },
    "ldsc": {
        "label": "Run LD-score diagnostic",
        "description": (
            "Compute LD scores and M on the intersected fitted panel, then "
            "run the optional moment diagnostic."
        ),
    },
    "fit": STAGE_DEFINITIONS["fit"],
    "weights": STAGE_DEFINITIONS["weights"],
}


def stage_definitions(job: dict) -> list[dict]:
    """Return the visible stages for a current or historical job."""
    schema = int(job.get("stage_schema") or 1)
    if schema >= STAGE_SCHEMA:
        order, definitions = STAGE_ORDER, STAGE_DEFINITIONS
    elif schema >= 2:
        order, definitions = SCHEMA2_STAGE_ORDER, SCHEMA2_STAGE_DEFINITIONS
    else:
        order, definitions = LEGACY_STAGE_ORDER, LEGACY_STAGE_DEFINITIONS
    options = job.get("options") or {}
    return [
        {"key": key, **definitions[key]}
        for key in order
        if (key != "weights" or options.get("weights"))
        and (key not in ("acquire", "download")
             or options.get("gcst1") or options.get("gcst2"))
    ]


def stage_label(key: str, schema: int | None = None) -> str:
    """Human label for a persisted stage key."""
    value = int(schema or 1)
    if value >= STAGE_SCHEMA:
        definitions = STAGE_DEFINITIONS
    elif value >= 2:
        definitions = SCHEMA2_STAGE_DEFINITIONS
    else:
        definitions = LEGACY_STAGE_DEFINITIONS
    fallback = STAGE_DEFINITIONS.get(
        key, SCHEMA2_STAGE_DEFINITIONS.get(
            key, LEGACY_STAGE_DEFINITIONS.get(key, {})))
    return definitions.get(key, fallback).get("label", key)

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


def create_job(root: Path, *, options: dict, labels: dict,
               status: str = "queued") -> dict:
    """Create a job with a fresh unguessable id and return it.

    Upload handlers create ``staging`` jobs and expose them to the supervisor
    only after every input has been durably written.
    """
    if status not in STATES:
        raise ValueError(f"unknown job status {status!r}")
    job_id = secrets.token_urlsafe(12)
    job_dir = root / "jobs" / job_id
    job_dir.mkdir(parents=True)
    job = {
        "id": job_id,
        "status": status,
        "stage": None,
        "stage_schema": STAGE_SCHEMA,
        "stages": {},
        "stage_details": {},
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
    """Delete stale staging and expired terminal jobs; return their ids."""
    if ttl_days <= 0:
        return []
    cutoff = _now() - ttl_days * 86400.0
    removed = []
    for job in list_jobs(root):
        if job["status"] in ("queued", "launching", "running"):
            continue
        expired_at = (job["created"] if job["status"] == "staging"
                      else job["finished"] or job["created"])
        if expired_at < cutoff:
            shutil.rmtree(job_dir(root, job["id"]), ignore_errors=True)
            removed.append(job["id"])
    return removed


def recover_interrupted_jobs(root: Path) -> list[str]:
    """Reconcile jobs whose owning web/runner process vanished on restart.

    Queued jobs are preserved for the new supervisor.  Unpublished staging
    directories are deleted; launching and running jobs become failed.
    """
    recovered = []
    for job in list_jobs(root):
        if job["status"] not in ("staging", "launching", "running"):
            continue
        previous = job["status"]
        if previous == "staging":
            # No redirect is returned until a job becomes queued, so an
            # interrupted staging directory has no user-visible job to retain.
            # It may contain a private, partially copied upload.
            shutil.rmtree(job_dir(root, job["id"]), ignore_errors=True)
            recovered.append(job["id"])
            continue
        update_job(root, job["id"], status="failed", stage=None,
                   finished=_now(),
                   error=(f"server restarted while job was {previous}; "
                          "submit it again"))
        recovered.append(job["id"])
    return recovered
