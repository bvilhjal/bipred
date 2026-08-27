"""Focused scheduler/restart resource-control contracts."""

from __future__ import annotations

import os
import threading
import time

from fastapi.testclient import TestClient

from webapp import app as app_module
from webapp import jobs
from webapp.app import _reserve_staging_job, _sweep_once, create_app


def _app(tmp_path, *, concurrency=1, queue_max=8):
    app = create_app()
    app.state.root = tmp_path
    (tmp_path / "jobs").mkdir(exist_ok=True)
    app.state.config["concurrency"] = concurrency
    app.state.config["queue_max"] = queue_max
    app.state.procs = {}
    app.state.orphans = {}
    app.state.last_purge = time.time()
    return app


def _leased_job(root, *, status="running"):
    job = jobs.create_job(root, options={}, labels={}, status=status)
    token = jobs.new_runner_token()
    job = jobs.update_job(
        root, job["id"], pid=os.getpid(),
        pid_identity=jobs.process_identity(os.getpid()), runner_token=token,
        started=time.time(), runtime_limit_s=3600.0)
    lease = jobs.ProcessFileLock(
        jobs.runner_lease_path(root, job["id"], token))
    assert lease.acquire()
    return job, lease


def test_final_queue_reservation_is_atomic(tmp_path):
    app = _app(tmp_path, queue_max=1)
    barrier = threading.Barrier(8)
    results = []

    def reserve(index):
        barrier.wait(timeout=5)
        results.append(_reserve_staging_job(
            app, options={"index": index}, labels={}))

    threads = [threading.Thread(target=reserve, args=(index,))
               for index in range(barrier.parties)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert sum(result is not None for result in results) == 1
    assert len(jobs.list_jobs(tmp_path)) == 1


def test_simultaneous_upload_and_demo_share_one_queue_reservation(
        tmp_path, monkeypatch):
    app = _app(tmp_path, queue_max=1)
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    for trait in (1, 2):
        (demo_dir / f"trait{trait}.tsv").write_text(
            "rsid\tbeta\tse\tn\nrs1\t0.1\t0.01\t1000\n",
            encoding="utf-8")
    monkeypatch.setattr(app_module.demo, "ensure_demo", lambda root: demo_dir)
    monkeypatch.setattr(
        app_module.demo, "demo_meta",
        lambda root: {"n_eff1": 1000, "n_eff2": 1000})

    with TestClient(app) as client:
        barrier = threading.Barrier(2)

        def gated_demo(root):
            barrier.wait(timeout=5)
            return demo_dir

        def gated_cache(key, root):
            barrier.wait(timeout=5)
            return tmp_path / "real-cache.npz"

        monkeypatch.setattr(app_module.demo, "ensure_demo", gated_demo)
        monkeypatch.setattr(app_module.caches, "cache_path", gated_cache)
        responses = []

        def submit_upload():
            responses.append(client.post(
                "/jobs",
                data={
                    "cache_key": "real", "n_eff1": "1000",
                    "n_eff2": "1000", "burn_in": "50",
                    "num_iter": "40", "seed": "0", "cross_corr": "0",
                },
                files={
                    "sumstats1": ("trait1.tsv", b"x\n"),
                    "sumstats2": ("trait2.tsv", b"x\n"),
                }, follow_redirects=False))

        def submit_demo():
            responses.append(client.post("/demo", follow_redirects=False))

        threads = [threading.Thread(target=submit_upload),
                   threading.Thread(target=submit_demo)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(response.status_code for response in responses) == [303, 503]
    assert len(jobs.list_jobs(tmp_path)) == 1


def test_sweep_adopts_untracked_verified_runner_before_launching(tmp_path):
    app = _app(tmp_path, concurrency=1)
    running, lease = _leased_job(tmp_path)
    queued = jobs.create_job(tmp_path, options={}, labels={}, status="queued")
    try:
        _sweep_once(app)
    finally:
        lease.release()

    assert app.state.orphans == {running["id"]: os.getpid()}
    assert jobs.load_job(tmp_path, queued["id"])["status"] == "queued"
    assert app.state.procs == {}


def test_sweep_counts_launch_handshake_before_pid_publication(tmp_path):
    app = _app(tmp_path, concurrency=1)
    launching = jobs.create_job(
        tmp_path, options={}, labels={}, status="launching")
    jobs.update_job(
        tmp_path, launching["id"], started=time.time(),
        runner_token=jobs.new_runner_token(), runtime_limit_s=3600.0)
    queued = jobs.create_job(tmp_path, options={}, labels={}, status="queued")

    _sweep_once(app)

    assert app.state.orphans == {launching["id"]: 0}
    assert jobs.load_job(tmp_path, queued["id"])["status"] == "queued"


def test_orphan_exit_reloads_terminal_state_before_failing(
        tmp_path, monkeypatch):
    app = _app(tmp_path)
    running = jobs.create_job(
        tmp_path, options={}, labels={}, status="running")
    app.state.orphans = {running["id"]: os.getpid()}

    def completes_during_lease_check(root, snapshot):
        jobs.update_job(
            root, snapshot["id"], status="done", finished=time.time())
        return False

    monkeypatch.setattr(
        jobs, "runner_is_verified", completes_during_lease_check)
    _sweep_once(app)

    assert jobs.load_job(tmp_path, running["id"])["status"] == "done"
    assert app.state.orphans == {}


def test_exited_orphan_uses_persisted_runtime_limit(tmp_path):
    app = _app(tmp_path)
    running = jobs.create_job(
        tmp_path, options={}, labels={}, status="running")
    jobs.update_job(
        tmp_path, running["id"], started=time.time() - 10,
        runtime_limit_s=1.0, runner_token=jobs.new_runner_token(),
        pid=4_000_000_000, pid_identity="gone")
    app.state.orphans = {running["id"]: 4_000_000_000}

    _sweep_once(app)

    saved = jobs.load_job(tmp_path, running["id"])
    assert saved["status"] == "failed"
    assert "runtime limit" in saved["error"]


def test_runner_watchdog_exits_at_persisted_deadline(tmp_path, monkeypatch):
    (tmp_path / "jobs").mkdir()
    running = jobs.create_job(
        tmp_path, options={}, labels={}, status="running")
    jobs.update_job(
        tmp_path, running["id"], started=time.time() - 10,
        runtime_limit_s=1.0)
    exit_codes = []
    monkeypatch.setattr(jobs.os, "_exit", exit_codes.append)

    jobs._runner_watchdog(threading.Event(), tmp_path, running["id"])

    assert exit_codes == [124]
