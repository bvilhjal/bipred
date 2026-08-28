"""Focused cross-platform regressions for web-service persistence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from webapp import caches, jobs, prepared_store


def _stat(*, ctime_ns):
    return SimpleNamespace(
        st_dev=4, st_ino=21, st_size=128, st_mtime_ns=987654321,
        st_ctime_ns=ctime_ns,
    )


def test_cache_identity_ignores_nonportable_ctime_semantics():
    """Path.stat and fstat creation/change times need not agree on Windows."""
    assert caches._identity(_stat(ctime_ns=1)) == caches._identity(
        _stat(ctime_ns=999))
    assert set(caches._identity(_stat(ctime_ns=1))) == {
        "dev", "ino", "size", "mtime_ns",
    }


def test_cache_hash_still_detects_same_path_payload_mutation(tmp_path):
    path = tmp_path / "cache.bin"
    path.write_bytes(b"first generation")
    first = caches.sha256_cached(path)
    path.write_bytes(b"second-generation")
    os.utime(path, None)

    second = caches.sha256_cached(path)

    assert first != second
    assert second == hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepared_lock_is_persistent_and_reacquires_without_unlink(
        tmp_path, monkeypatch):
    lock = prepared_store._Lock(tmp_path / "prepared.lock")
    assert lock.acquire()
    unlinks = []

    def checked_unlink(path, *args, **kwargs):
        unlinks.append(path)
        raise AssertionError("process locks must never unlink their pathname")

    monkeypatch.setattr(Path, "unlink", checked_unlink)
    lock.release()

    assert unlinks == []
    assert lock.path.exists()
    successor = prepared_store._Lock(lock.path)
    assert successor.acquire()
    successor.release()
    assert lock.path.exists()


def test_prepared_lock_cannot_be_stolen_from_a_live_process(tmp_path):
    first = prepared_store._Lock(tmp_path / "prepared.lock")
    assert first.acquire()
    successor = prepared_store._Lock(first.path)
    assert not successor.acquire()

    first.release()
    assert successor.acquire()
    successor.release()
    assert successor.path.exists()


def test_prepared_kernel_lock_needs_no_heartbeat(tmp_path):
    lock = prepared_store._Lock(tmp_path / "prepared.lock")
    assert lock.acquire()
    try:
        descriptor = lock.fd
        lock.touch()
        assert lock.fd == descriptor
        assert lock.owned()
    finally:
        lock.release()
    assert lock.path.exists()


def test_process_lock_is_released_by_os_when_owner_exits(tmp_path):
    path = tmp_path / "process.lock"
    code = (
        "import sys,time; from webapp.jobs import ProcessFileLock; "
        "lock=ProcessFileLock(sys.argv[1]); assert lock.acquire(); "
        "print('locked', flush=True); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        cwd=Path(__file__).resolve().parent.parent,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "locked"
        probe = jobs.ProcessFileLock(path)
        assert not probe.acquire()
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    probe = jobs.ProcessFileLock(path)
    assert probe.acquire()
    probe.release()


def test_concurrent_job_writers_use_independent_temp_files(
        tmp_path, monkeypatch):
    (tmp_path / "jobs").mkdir()
    initial = jobs.create_job(tmp_path, options={}, labels={})
    original_replace = os.replace
    barrier = threading.Barrier(8)
    barrier_sources = set()
    barrier_lock = threading.Lock()
    errors = []

    def synchronized_replace(source, destination):
        with barrier_lock:
            first_attempt = source not in barrier_sources
            barrier_sources.add(source)
        if Path(destination).name == jobs.JOB_JSON and first_attempt:
            barrier.wait(timeout=5)
        return original_replace(source, destination)

    monkeypatch.setattr(jobs.os, "replace", synchronized_replace)

    def write(index):
        job = dict(initial)
        job["error"] = f"writer-{index}"
        try:
            jobs.save_job(tmp_path, job)
        except Exception as exc:  # collected for an assertion in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(index,))
               for index in range(barrier.parties)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert jobs.load_job(tmp_path, initial["id"])["error"].startswith(
        "writer-")
    assert not list(jobs.job_dir(tmp_path, initial["id"]).glob("*.part"))


def test_job_replace_retries_windows_sharing_violation(
        tmp_path, monkeypatch):
    (tmp_path / "jobs").mkdir()
    job = jobs.create_job(tmp_path, options={}, labels={})
    real_replace = os.replace
    calls = []

    def flaky_replace(source, destination):
        calls.append((source, destination))
        if len(calls) < 3:
            raise PermissionError("file is briefly open by a reader")
        return real_replace(source, destination)

    monkeypatch.setattr(jobs.os, "replace", flaky_replace)
    monkeypatch.setattr(jobs, "_retry_delay", lambda attempt: None)
    job["status"] = "running"
    jobs.save_job(tmp_path, job)

    assert len(calls) == 3
    assert jobs.load_job(tmp_path, job["id"])["status"] == "running"


def test_list_jobs_skips_bad_entries_and_logs_each_diagnostic(
        tmp_path, monkeypatch, caplog):
    (tmp_path / "jobs").mkdir()
    valid = jobs.create_job(tmp_path, options={}, labels={})
    corrupt = tmp_path / "jobs" / "corrupt"
    corrupt.mkdir()
    (corrupt / jobs.JOB_JSON).write_text("{broken", encoding="utf-8")
    missing = tmp_path / "jobs" / "missing"
    missing.mkdir()
    (tmp_path / "jobs" / "stray.txt").write_text("not a job")
    mismatch = tmp_path / "jobs" / "mismatch"
    mismatch.mkdir()
    (mismatch / jobs.JOB_JSON).write_text(
        json.dumps({"id": "someone-else"}), encoding="utf-8")
    monkeypatch.setattr(jobs, "_retry_delay", lambda attempt: None)

    with caplog.at_level("WARNING", logger=jobs.__name__):
        found = jobs.list_jobs(tmp_path)

    assert [job["id"] for job in found] == [valid["id"]]
    messages = [record.getMessage() for record in caplog.records]
    assert any("corrupt" in message and "unreadable" in message
               for message in messages)
    assert any("missing" in message and "without job.json" in message
               for message in messages)
    assert any("stray.txt" in message and "stray" in message
               for message in messages)
    assert any("mismatch" in message and "mismatched id" in message
               for message in messages)


def test_restart_can_preserve_and_report_a_live_runner(tmp_path):
    (tmp_path / "jobs").mkdir()
    live_job = jobs.create_job(
        tmp_path, options={}, labels={}, status="running")
    token = jobs.new_runner_token()
    live_job = jobs.update_job(
        tmp_path, live_job["id"], pid=os.getpid(),
        # Production falls back to a lease-only identity where the OS exposes
        # no creation identity (macOS); mirror that or verification fails.
        pid_identity=(jobs.process_identity(os.getpid())
                      or f"lease-only:{token}"),
        runner_token=token)
    lease = jobs.ProcessFileLock(
        jobs.runner_lease_path(tmp_path, live_job["id"], token))
    assert lease.acquire()
    dead_job = jobs.create_job(
        tmp_path, options={}, labels={}, status="launching")
    jobs.update_job(tmp_path, dead_job["id"], pid=4_000_000_000)
    queued = jobs.create_job(
        tmp_path, options={}, labels={}, status="queued")

    try:
        recovered, live = jobs.recover_interrupted_jobs(
            tmp_path, preserve_live=True)
    finally:
        lease.release()

    assert jobs.pid_is_alive(os.getpid())
    assert not jobs.pid_is_alive(None)
    assert live == {live_job["id"]: os.getpid()}
    assert recovered == [dead_job["id"]]
    assert jobs.load_job(tmp_path, live_job["id"])["status"] == "running"
    assert jobs.load_job(tmp_path, dead_job["id"])["status"] == "failed"
    assert jobs.load_job(tmp_path, queued["id"])["status"] == "queued"


def test_recovery_does_not_overwrite_completion_after_dead_pid_observation(
        tmp_path, monkeypatch):
    (tmp_path / "jobs").mkdir()
    running = jobs.create_job(
        tmp_path, options={}, labels={}, status="running")

    def completes_while_checked(root, snapshot):
        jobs.update_job(root, snapshot["id"], status="done", finished=time.time())
        return False

    monkeypatch.setattr(jobs, "runner_is_verified", completes_while_checked)
    recovered, live = jobs.recover_interrupted_jobs(
        tmp_path, preserve_live=True)

    assert recovered == []
    assert live == {}
    assert jobs.load_job(tmp_path, running["id"])["status"] == "done"


def test_recent_launch_handshake_is_counted_before_pid_publication(tmp_path):
    (tmp_path / "jobs").mkdir()
    launching = jobs.create_job(
        tmp_path, options={}, labels={}, status="launching")
    jobs.update_job(
        tmp_path, launching["id"], started=time.time(),
        runner_token=jobs.new_runner_token())

    recovered, live = jobs.recover_interrupted_jobs(
        tmp_path, preserve_live=True)

    assert recovered == []
    assert live == {launching["id"]: 0}
    assert jobs.load_job(tmp_path, launching["id"])["status"] == "launching"


@pytest.mark.parametrize("pid", [True, False, 0, -1, "123"])
def test_pid_liveness_rejects_nonpositive_and_noninteger_values(pid):
    assert jobs.pid_is_alive(pid) is False
