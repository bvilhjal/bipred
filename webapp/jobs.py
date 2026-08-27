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
import logging
import os
import secrets
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

STATES = ("staging", "queued", "launching", "running", "done", "failed")

# Version 4 permits preparation and screening to overlap across the two
# independent trait pipelines and adds pre-DENTIST univariate LDSC QC. Keep
# every earlier schema so completed jobs retain the workflow they actually ran.
STAGE_SCHEMA = 4
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
            "Prepare both traits independently against one shared LD "
            "reference, including QC, harmonization, and a quick univariate "
            "LD-score h2/intercept check."
        ),
    },
    "screen": {
        "label": "Run LD-consistency screen",
        "description": (
            "Reuse a fully QC'd, harmonized, and screened trait artifact, or "
            "run the mandatory DENTIST-inspired trait-local screen as soon "
            "as that trait is ready, then store the completed artifact."
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

SCHEMA3_STAGE_ORDER = STAGE_ORDER
SCHEMA3_STAGE_DEFINITIONS = {
    "acquire": STAGE_DEFINITIONS["acquire"],
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
    "pair": STAGE_DEFINITIONS["pair"],
    "ldsc": STAGE_DEFINITIONS["ldsc"],
    "fit": STAGE_DEFINITIONS["fit"],
    "weights": STAGE_DEFINITIONS["weights"],
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
    elif schema >= 3:
        order, definitions = SCHEMA3_STAGE_ORDER, SCHEMA3_STAGE_DEFINITIONS
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
    elif value >= 3:
        definitions = SCHEMA3_STAGE_DEFINITIONS
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
RUNNER_LEASE_PREFIX = "runner-"
RUNNER_HANDSHAKE_GRACE = 30.0

_IO_ATTEMPTS = 5
_IO_RETRY_SECONDS = 0.01
_WINDOWS_SHARING_ERRORS = frozenset({5, 32, 33})

logger = logging.getLogger(__name__)


class ProcessFileLock:
    """Dependency-free inter-process lock held by an open file descriptor.

    POSIX ``flock`` and Windows CRT byte-range locks are released by the OS
    when their process exits.  The pathname is deliberately persistent:
    unlinking a pathname lock creates an unavoidable check/delete race in
    which an old owner can remove a successor's newly-created lock.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.fd = None

    @staticmethod
    def _lock(fd):
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(fd):
        if os.name == "nt":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)

    def acquire(self) -> bool:
        if self.fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            self._lock(fd)
        except OSError:
            os.close(fd)
            return False
        self.fd = fd
        return True

    def owned(self) -> bool:
        if self.fd is None:
            return False
        try:
            os.fstat(self.fd)
        except OSError:
            return False
        return True

    def touch(self) -> None:
        """Compatibility no-op: kernel ownership needs no heartbeat."""

    def release(self) -> None:
        if self.fd is None:
            return
        fd = self.fd
        self.fd = None
        try:
            self._unlock(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def __enter__(self):
        if not self.acquire():
            raise BlockingIOError(f"lock is already held: {self.path}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False


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
        "active_stages": [],
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
        "pid_identity": None,
        "runner_token": None,
        "runtime_limit_s": None,
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


def _retry_delay(attempt: int) -> None:
    time.sleep(_IO_RETRY_SECONDS * (attempt + 1))


def _transient_io_error(exc: OSError) -> bool:
    return (isinstance(exc, PermissionError)
            or getattr(exc, "winerror", None) in _WINDOWS_SHARING_ERRORS)


def _read_job_json(path: Path):
    """Read one state file, tolerating brief Windows sharing/replace races."""
    for attempt in range(_IO_ATTEMPTS):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Atomic writers cannot expose partial JSON, but an older web or
            # runner process may still be using the pre-v5 fixed temporary
            # file during a rolling restart.  Retry briefly before reporting
            # persistent corruption to the caller.
            if attempt + 1 == _IO_ATTEMPTS:
                raise
        except OSError as exc:
            if (not _transient_io_error(exc)
                    or attempt + 1 == _IO_ATTEMPTS):
                raise
        _retry_delay(attempt)
    raise AssertionError("unreachable")


def load_job(root: Path, job_id: str) -> dict | None:
    path = job_dir(root, job_id) / JOB_JSON
    job = _read_job_json(path)
    if job is None:
        return None
    if not isinstance(job, dict):
        raise ValueError(f"job state {path} must be a JSON object")
    if job.get("id") != job_id:
        raise ValueError(f"job state {path} has a mismatched id")
    return job


def save_job(root: Path, job: dict) -> None:
    path = job_dir(root, job["id"]) / JOB_JSON
    wire = json.dumps(job, indent=1) + "\n"
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        with open(tmp, "x", encoding="utf-8") as fh:
            fh.write(wire)
        for attempt in range(_IO_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return
            except OSError as exc:
                if (not _transient_io_error(exc)
                        or attempt + 1 == _IO_ATTEMPTS):
                    raise
                _retry_delay(attempt)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


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
        try:
            if not entry.is_dir():
                logger.warning("Skipping stray job-store entry %s", entry)
                continue
            job = load_job(root, entry.name)
            if job is not None:
                status = job.get("status")
                created = job.get("created")
                if status not in STATES:
                    raise ValueError(f"unknown job status {status!r}")
                if (isinstance(created, bool)
                        or not isinstance(created, (int, float))):
                    raise ValueError("job created time is missing or invalid")
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            logger.warning("Skipping unreadable job-store entry %s: %s",
                           entry, exc)
            continue
        if job is not None:
            out.append(job)
        else:
            logger.warning("Skipping job-store directory without %s: %s",
                           JOB_JSON, entry)
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


def _valid_pid(pid) -> bool:
    return (not isinstance(pid, bool) and isinstance(pid, int) and pid > 0)


def _windows_process_info(pid):
    """Return ``(alive, creation identity)`` without signalling a PID."""
    if not _valid_pid(pid) or pid > 0xFFFFFFFF:
        return False, None
    import ctypes
    from ctypes import wintypes

    class _FileTime(ctypes.Structure):
        _fields_ = (("low", wintypes.DWORD), ("high", wintypes.DWORD))

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = (wintypes.HANDLE,
                              ctypes.POINTER(wintypes.DWORD))
    get_exit_code.restype = wintypes.BOOL
    get_times = kernel32.GetProcessTimes
    get_times.argtypes = (wintypes.HANDLE, ctypes.POINTER(_FileTime),
                          ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime),
                          ctypes.POINTER(_FileTime))
    get_times.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = open_process(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    if not handle:
        # A protected process can be alive but is never a runner owned by this
        # user, so it deliberately has no verifiable creation identity.
        return ctypes.get_last_error() == 5, None
    try:
        exit_code = wintypes.DWORD()
        if (not get_exit_code(handle, ctypes.byref(exit_code))
                or exit_code.value != 259):            # STILL_ACTIVE
            return False, None
        created = _FileTime()
        exited, kernel, user = _FileTime(), _FileTime(), _FileTime()
        if not get_times(handle, ctypes.byref(created), ctypes.byref(exited),
                         ctypes.byref(kernel), ctypes.byref(user)):
            return True, None
        ticks = (created.high << 32) | created.low
        return True, f"windows-filetime:{ticks}"
    finally:
        close_handle(handle)


def _procfs_process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = text.rfind(")")
        fields = text[close + 2:].split()
        start_ticks = fields[19]                 # proc(5) field 22
        try:
            boot = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii").strip()
        except OSError:
            boot = str(Path("/proc/1").stat().st_ctime_ns)
        return f"procfs:{boot}:{start_ticks}"
    except (OSError, IndexError, ValueError):
        return None


def process_identity(pid) -> str | None:
    """Stable creation identity for a live PID where the OS exposes one."""
    if not _valid_pid(pid):
        return None
    if os.name == "nt":
        return _windows_process_info(pid)[1]
    return _procfs_process_identity(pid)


def pid_is_alive(pid, identity: str | None = None) -> bool:
    """Return whether a PID is live and, when supplied, the same process."""
    if not _valid_pid(pid):
        return False
    current = process_identity(pid)
    if identity is not None:
        return current == identity
    if current is not None:
        return True
    if os.name == "nt":
        return _windows_process_info(pid)[0]
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError, ValueError):
        return False
    return True


def new_runner_token() -> str:
    return secrets.token_urlsafe(18)


def runner_lease_path(root: Path, job_id: str, token: str) -> Path:
    if (not isinstance(token, str) or not token
            or any(c not in "-_0123456789abcdefghijklmnopqrstuvwxyz"
                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in token)):
        raise ValueError("invalid runner lease token")
    return job_dir(root, job_id) / f"{RUNNER_LEASE_PREFIX}{token}.lock"


def runner_lease_held(root: Path, job: dict) -> bool:
    try:
        path = runner_lease_path(root, job["id"], job.get("runner_token"))
    except (KeyError, TypeError, ValueError):
        return False
    probe = ProcessFileLock(path)
    if not probe.acquire():
        return True
    probe.release()
    return False


def runner_is_verified(root: Path, job: dict) -> bool:
    """Bind active state to both an OS lease and the original process."""
    pid = job.get("pid")
    identity = job.get("pid_identity")
    token = job.get("runner_token")
    lease_only = identity == f"lease-only:{token}"
    if (not isinstance(identity, str) or not identity
            or not runner_lease_held(root, job)):
        return False
    if not (pid_is_alive(pid) if lease_only else pid_is_alive(pid, identity)):
        return False
    # Close the small lease-exit/PID-exit observation window.
    return runner_lease_held(root, job)


def runner_handshake_pending(job: dict, now: float | None = None) -> bool:
    """A newly launched wrapper gets a short window to publish its lease."""
    if job.get("status") != "launching" or not job.get("runner_token"):
        return False
    started = job.get("started")
    if isinstance(started, bool) or not isinstance(started, (int, float)):
        return False
    elapsed = (time.time() if now is None else now) - float(started)
    return -1.0 <= elapsed <= RUNNER_HANDSHAKE_GRACE


def fail_active_job(root: Path, job_id: str, *, error: str,
                    finished: float | None = None) -> dict | None:
    """Fail only freshly-read active state after its runner has exited."""
    job = load_job(root, job_id)
    if job is None or job.get("status") not in ("launching", "running"):
        return job
    job.update(status="failed", stage=None,
               finished=_now() if finished is None else finished,
               error=error)
    save_job(root, job)
    return job


def recover_interrupted_jobs(
        root: Path, *, preserve_live: bool = False):
    """Reconcile jobs whose owning web/runner process vanished on restart.

    Queued jobs are preserved for the new supervisor.  Unpublished staging
    directories are deleted; launching and running jobs become failed.  A
    restart-aware supervisor can pass ``preserve_live=True`` to retain jobs
    whose recorded runner PID is still alive and receive ``(recovered,
    live_by_job_id)``.  The default return remains the historical list.
    """
    recovered = []
    live = {}
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
        pid = job.get("pid")
        if preserve_live and (runner_is_verified(root, job)
                              or runner_handshake_pending(job)):
            live[job["id"]] = pid if _valid_pid(pid) else 0
            continue
        # Re-read after observing lease/process exit.  The runner may have
        # published ``done`` between list_jobs() and that observation.
        current = load_job(root, job["id"])
        if current is not None and current.get("status") in (
                "launching", "running"):
            fail_active_job(
                root, job["id"],
                error=(f"server restarted while job was {previous}; "
                       "submit it again"),
            )
            recovered.append(job["id"])
    if preserve_live:
        return recovered, live
    return recovered


def _runner_watchdog(stop: threading.Event, root: Path, job_id: str) -> None:
    job = load_job(root, job_id)
    if job is None:
        return
    started = job.get("started")
    limit = job.get("runtime_limit_s")
    if (isinstance(started, bool) or not isinstance(started, (int, float))
            or isinstance(limit, bool) or not isinstance(limit, (int, float))
            or limit <= 0):
        return
    remaining = max(0.0, float(started) + float(limit) - time.time())
    if not stop.wait(remaining):
        # The supervisor converts exit 124 into the persisted timeout reason.
        # os._exit is intentional: it stops native code that cannot observe a
        # Python cancellation flag, while the lease proves which process died.
        os._exit(124)


def run_runner_with_lease(job_dir_path) -> int:
    """Run ``webapp.runner`` while holding its kernel-backed lifetime lease."""
    target = Path(job_dir_path).resolve()
    root = target.parent.parent
    job = load_job(root, target.name)
    if job is None:
        print(f"no job.json in {target}", file=sys.stderr)
        return 2
    try:
        lease_path = runner_lease_path(root, job["id"], job["runner_token"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"invalid runner lease: {exc}", file=sys.stderr)
        return 2
    lease = ProcessFileLock(lease_path)
    if not lease.acquire():
        print(f"runner lease is already held for {job['id']}", file=sys.stderr)
        return 3
    stop = threading.Event()
    watchdog = threading.Thread(
        target=_runner_watchdog, args=(stop, root, job["id"]), daemon=True,
        name="bipred-runner-timeout")
    try:
        identity = process_identity(os.getpid()) or (
            f"lease-only:{job['runner_token']}")
        update_job(root, job["id"], pid=os.getpid(), pid_identity=identity)
        watchdog.start()
        from . import runner

        return runner.main([str(target)])
    finally:
        stop.set()
        if watchdog.is_alive():
            watchdog.join(timeout=1.0)
        lease.release()


def _main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "run-runner":
        return run_runner_with_lease(args[1])
    print("usage: python -m webapp.jobs run-runner JOB_DIR", file=sys.stderr)
    return 2


if __name__ == "__main__":                    # pragma: no cover
    raise SystemExit(_main())
