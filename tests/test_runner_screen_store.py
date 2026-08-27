"""Web-runner persistence starts only after mandatory trait-local screening."""

import hashlib
import json
import shutil
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import bipred
from ldpred3 import save_ld_blocks
from webapp import caches, jobs, prepared_store, runner


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path, m=12):
    positions = np.arange(m)
    corr = np.ascontiguousarray(
        0.5 ** np.abs(positions[:, None] - positions[None, :]),
        dtype=np.float32)
    ids = np.array([f"rs{i}" for i in positions], dtype=object)
    a1 = np.array(["A"] * m, dtype=object)
    a2 = np.array(["G"] * m, dtype=object)
    cache = tmp_path / "ld.npz"
    save_ld_blocks(
        cache, [(corr, positions)], ids,
        counted_allele=a1, other_allele=a2,
        chrom=np.array(["1"] * m, dtype=object), pos=positions + 1,
        reference_af=np.linspace(0.1, 0.4, m), n_ref=500, ridge=0.0)
    caches.write_ld_score_sidecar(
        cache, np.sum(np.asarray(corr, dtype=float) ** 2, axis=0),
        source="test exact cache", algorithm="test-colsum-r2-v1")

    paths = []
    for trait in (1, 2):
        path = tmp_path / f"source{trait}.tsv"
        z = np.linspace(-1.0, 1.0, m) + 0.1 * trait
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("SNP\tCHR\tBP\tA1\tA2\tEAF\tBETA\tSE\tN\n")
            for i in range(m):
                handle.write(
                    f"{ids[i]}\t1\t{i + 1}\tA\tG\t{0.1 + 0.3*i/(m-1):.6f}"
                    f"\t{z[i] / 100:.8g}\t0.01\t10000\n")
        paths.append(path)
    return cache, paths


def _job(root, sources, *, suffix, catalog_first=False):
    options = {
        "cache_key": "TEST",
        "n_eff1": 10_000.0, "n_cases1": None, "n_controls1": None,
        "n_eff2": 10_000.0, "n_cases2": None, "n_controls2": None,
        "seed": 987, "burn_in": 1, "num_iter": 1,
        "cross_corr": 0.0,
        "columns1": {}, "columns2": {},
        "screen": True, "weights": False,
    }
    if catalog_first:
        options["catalog1"] = {
            "accession": "GCSTTEST", "normalised_sha256": _sha256(sources[0]),
            "trait": "Catalog trait", "pmid": None, "n_basis": "test",
            "has_per_variant_n": False,
        }
    job = jobs.create_job(
        root, options=options,
        labels={"trait1": f"First {suffix}", "trait2": f"Second {suffix}"},
        status="running")
    job_dir = jobs.job_dir(root, job["id"])
    for trait, source in enumerate(sources, start=1):
        name = f"trait{trait}.tsv"
        shutil.copyfile(source, job_dir / name)
        job["files"][f"sumstats{trait}"] = name
    jobs.save_job(root, job)
    return job_dir, job


def _fake_fit():
    return SimpleNamespace(
        h2=(0.2, 0.3), rg=0.25, p=0.1, pi=None,
        mixer={"polygenicity": (0.1, 0.2)}, noise_scale=None,
        retained_iterations=1, stopped_early=False, genetic_samples=None,
        divergence_diagnostics={"evaluated": False, "flagged": False})


def test_stable_ld_load_closes_a_changed_generation(monkeypatch, tmp_path):
    closed = []
    prepared = SimpleNamespace(close=lambda: closed.append(True))
    hashes = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        runner.caches, "sha256_cached",
        lambda _path, _root=None: next(hashes))
    monkeypatch.setattr(
        "ldpred3.interop.prepare_ld_cache", lambda _path: prepared)

    with pytest.raises(ValueError, match="changed while it was being loaded"):
        runner._load_stable_ld_cache(
            tmp_path / "ld.npz", tmp_path, expected_sha256="a" * 64)

    assert closed == [True]


def test_stable_ld_load_rejects_a_generation_changed_since_selection(
        monkeypatch, tmp_path):
    loaded = []
    monkeypatch.setattr(
        runner.caches, "sha256_cached",
        lambda _path, _root=None: "b" * 64)
    monkeypatch.setattr(
        "ldpred3.interop.prepare_ld_cache",
        lambda _path: loaded.append(True))

    with pytest.raises(ValueError, match="changed before it was loaded"):
        runner._load_stable_ld_cache(
            tmp_path / "ld.npz", tmp_path, expected_sha256="a" * 64)

    assert loaded == []


def test_upload_and_catalog_traits_store_only_post_screen_artifacts(
        monkeypatch, tmp_path):
    root = tmp_path / "webdata"
    (root / "jobs").mkdir(parents=True)
    cache, sources = _inputs(tmp_path)
    monkeypatch.setenv("BIPRED_WEB_DATA", str(root))
    monkeypatch.setenv("BIPRED_WEB_CACHES", f"TEST={cache}")
    expected_ld_sha = caches.sha256_cached(cache, root)
    ld_hash_calls = []

    def stable_ld_hash(path, data_root=None):
        ld_hash_calls.append((path, data_root))
        return expected_ld_sha

    monkeypatch.setattr(runner.caches, "sha256_cached", stable_ld_hash)

    screen_calls = []

    def fake_screen(_blocks, _indices, z, **kwargs):
        screen_calls.append(dict(kwargs))
        keep = np.ones(len(z), dtype=bool)
        if kwargs["progress_label"].endswith("trait1"):
            keep[0] = False
        else:
            keep[1:3] = False
        return keep

    monkeypatch.setattr(
        "bipred.qc._ld_consistency_screen_selected", fake_screen)
    pair_calls = []
    real_pair = bipred.pair_prepared_traits

    def pair(*args, **kwargs):
        pair_calls.append(dict(kwargs))
        return real_pair(*args, **kwargs)

    monkeypatch.setattr(bipred, "pair_prepared_traits", pair)
    fit_calls = []

    def fit(*_args, **kwargs):
        fit_calls.append(dict(kwargs))
        return _fake_fit()

    monkeypatch.setattr(bipred, "ldpred3_auto_bivariate_blocks", fit)
    monkeypatch.setattr(
        runner, "_run_ldsc_regression",
        lambda prep, ell, panel: (
            (0.2, 0.3),
            {"h2_init": [0.2, 0.3], "h2_init_source": ["ldsc", "ldsc"],
             "h2": [0.2, 0.3], "rg": 0.2, "rg_se": 0.1,
             "gcov": 0.01, "gcov_intercept": 0.0}))

    first_dir, first_job = _job(
        root, sources, suffix="run", catalog_first=True)
    runner.run(first_dir, first_job)
    first = json.loads((first_dir / "result.json").read_text())

    assert len(screen_calls) == 2
    assert all(call["seed"] == 0 for call in screen_calls)
    assert fit_calls[0]["seed"] == 987
    assert fit_calls[0]["h2_init"] == (0.2, 0.3)
    assert pair_calls == [{
        "screen": False,
        "progress": pair_calls[0]["progress"],
        "consume_ld_cache": True,
    }]
    assert first["munge"]["trait1"]["ld_consistency_screen"][
        "n_dropped"] == 1
    assert first["munge"]["trait2"]["ld_consistency_screen"][
        "n_dropped"] == 2
    assert first["munge"]["trait1"]["n_usable"] == 11
    assert first["munge"]["trait2"]["n_usable"] == 10
    assert first["munge"]["n_joint"] == first["munge"]["n_kept"] == 9
    assert first["munge"]["n_screen_drop"] == 0
    pair_detail = first["provenance"]["stage_details"]["pair"]
    assert pair_detail["n_joint"] == 9
    assert "n_joint_before_screen" not in pair_detail
    assert "n_screen_drop" not in pair_detail

    params = first["provenance"]["screen_parameters"]
    assert params == runner._screen_parameters()
    assert params["seed"] == 0 != first["provenance"]["seed"]
    for trait in ("trait1", "trait2"):
        provenance = first["provenance"]["inputs"][trait]
        assert provenance["prepared_reused"] is False
        assert provenance["prepared_scope"] == runner._PREPARED_SCOPE
        assert provenance["prepared_key"]
        assert provenance["logical_sha256"] == provenance["sha256"]
    assert first_job["options"]["catalog1"]["prepared_scope"] == (
        runner._PREPARED_SCOPE)

    metadata = [
        json.loads(path.read_text())
        for path in prepared_store.store_dir(root).glob("*.json")
    ]
    assert len(metadata) == 2
    assert all(item["spec"]["screen"] == {
        "enabled": True, "params": runner._screen_parameters()
    } for item in metadata)
    assert all(item["log"]["screen"] is True for item in metadata)

    second_dir, second_job = _job(
        root, sources, suffix="reuse", catalog_first=True)
    runner.run(second_dir, second_job)
    second = json.loads((second_dir / "result.json").read_text())

    # Each run selects one expected generation, then hashes immediately before
    # and after loading. Provenance reuses that proven generation.
    assert len(ld_hash_calls) == 6
    assert first["provenance"]["cache_sha256"] == expected_ld_sha
    assert second["provenance"]["cache_sha256"] == expected_ld_sha
    assert len(screen_calls) == 2  # both builders were skipped on reuse
    assert all(second["provenance"]["inputs"][trait]["prepared_reused"]
               for trait in ("trait1", "trait2"))
    assert [second["provenance"]["inputs"][trait]["prepared_key"]
            for trait in ("trait1", "trait2")] == [
        first["provenance"]["inputs"][trait]["prepared_key"]
        for trait in ("trait1", "trait2")]
    assert second_job["options"]["catalog1"]["prepared_reused"] is True


def test_parallel_trait_pipelines_join_in_input_order(monkeypatch, tmp_path):
    """Preparation has no cross-trait barrier; pairing has a strict join."""
    root = tmp_path / "webdata"
    (root / "jobs").mkdir(parents=True)
    cache, sources = _inputs(tmp_path)
    monkeypatch.setenv("BIPRED_WEB_DATA", str(root))
    monkeypatch.setenv("BIPRED_WEB_CACHES", f"TEST={cache}")
    monkeypatch.setattr(
        runner, "_screen_parallelism",
        lambda: (True, "test-safe", {
            "concurrent": True, "blas_threads": 1,
            "blas_reentrant": True, "reason": "test-safe",
        }))

    prepare_barrier = threading.Barrier(2)
    trait1_screen_entered = threading.Event()
    screen_barrier = threading.Barrier(2)
    trait2_screen_done = threading.Event()
    both_screen_done = {"trait1": False, "trait2": False}
    real_prepare = bipred.prepare_trait_sumstats
    real_screen = bipred.screen_prepared_trait

    def prepare(*args, **kwargs):
        label = kwargs["label"]
        prepare_barrier.wait(timeout=5)
        if label == "trait2":
            assert trait1_screen_entered.wait(5)
        return real_prepare(*args, **kwargs)

    def screen(*args, **kwargs):
        label = args[1].log["label"]
        if label == "trait1":
            trait1_screen_entered.set()
        screen_barrier.wait(timeout=5)
        if label == "trait1":
            assert trait2_screen_done.wait(5)
            out = real_screen(*args, **kwargs)
        else:
            out = real_screen(*args, **kwargs)
            trait2_screen_done.set()
        both_screen_done[label] = True
        return out

    monkeypatch.setattr(bipred, "prepare_trait_sumstats", prepare)
    monkeypatch.setattr(bipred, "screen_prepared_trait", screen)
    real_pair = bipred.pair_prepared_traits
    pair_order = []

    def pair(cache_owner, trait1, trait2, **kwargs):
        assert all(both_screen_done.values())
        pair_order.append((trait1.log["label"], trait2.log["label"]))
        return real_pair(cache_owner, trait1, trait2, **kwargs)

    monkeypatch.setattr(bipred, "pair_prepared_traits", pair)
    monkeypatch.setattr(
        bipred, "ldpred3_auto_bivariate_blocks",
        lambda *_args, **_kwargs: _fake_fit())
    monkeypatch.setattr(
        runner, "_run_ldsc_regression",
        lambda prep, ell, panel: (
            (0.2, 0.3),
            {"h2_init": [0.2, 0.3], "h2_init_source": ["ldsc", "ldsc"],
             "h2": [0.2, 0.3], "rg": 0.2, "rg_se": 0.1,
             "gcov": 0.01, "gcov_intercept": 0.0}))

    job_dir, job = _job(root, sources, suffix="parallel")
    runner.run(job_dir, job)

    assert pair_order == [("trait1", "trait2")]
    assert trait1_screen_entered.is_set() and trait2_screen_done.is_set()
    assert job["active_stages"] == []
    assert job["stage_details"]["prepare"]["parallel"] is True
    assert job["stage_details"]["screen"]["execution"]["concurrent"] is True


def test_unsafe_blas_serializes_only_the_screen_calls(monkeypatch, tmp_path):
    root = tmp_path / "webdata"
    (root / "jobs").mkdir(parents=True)
    cache, sources = _inputs(tmp_path)
    monkeypatch.setenv("BIPRED_WEB_DATA", str(root))
    monkeypatch.setenv("BIPRED_WEB_CACHES", f"TEST={cache}")
    execution = {
        "concurrent": False, "blas_threads": 1,
        "blas_reentrant": False, "reason": "unsafe test BLAS",
    }
    monkeypatch.setattr(
        runner, "_screen_parallelism",
        lambda: (False, "unsafe test BLAS", execution))

    guard = threading.Lock()
    release_first = threading.Event()
    state = {"active": 0, "maximum": 0, "started": 0}
    real_screen = bipred.screen_prepared_trait

    def screen(*args, **kwargs):
        with guard:
            state["started"] += 1
            ordinal = state["started"]
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
            if ordinal == 2:
                release_first.set()
        if ordinal == 1:
            release_first.wait(0.3)
        try:
            return real_screen(*args, **kwargs)
        finally:
            with guard:
                state["active"] -= 1

    monkeypatch.setattr(bipred, "screen_prepared_trait", screen)
    monkeypatch.setattr(
        bipred, "ldpred3_auto_bivariate_blocks",
        lambda *_args, **_kwargs: _fake_fit())
    monkeypatch.setattr(
        runner, "_run_ldsc_regression",
        lambda prep, ell, panel: (
            (0.2, 0.3),
            {"h2_init": [0.2, 0.3], "h2_init_source": ["ldsc", "ldsc"],
             "h2": [0.2, 0.3], "rg": 0.2, "rg_se": 0.1,
             "gcov": 0.01, "gcov_intercept": 0.0}))

    job_dir, job = _job(root, sources, suffix="serial-screen")
    runner.run(job_dir, job)

    assert state["started"] == 2
    assert state["maximum"] == 1
    assert job["stage_details"]["prepare"]["parallel"] is True
    assert job["stage_details"]["screen"]["execution"] == execution


def test_trait_failure_joins_peer_before_shared_ld_close(monkeypatch, tmp_path):
    root = tmp_path / "webdata"
    (root / "jobs").mkdir(parents=True)
    cache, sources = _inputs(tmp_path)
    monkeypatch.setenv("BIPRED_WEB_DATA", str(root))
    monkeypatch.setenv("BIPRED_WEB_CACHES", f"TEST={cache}")
    prepare_barrier = threading.Barrier(2)
    peer_exited = threading.Event()
    close_calls = []
    real_load = runner._load_stable_ld_cache

    def load(*args, **kwargs):
        owner, digest = real_load(*args, **kwargs)
        real_close = owner.close

        def close():
            assert peer_exited.is_set()
            close_calls.append(True)
            return real_close()

        owner.close = close
        return owner, digest

    monkeypatch.setattr(runner, "_load_stable_ld_cache", load)
    real_prepare = bipred.prepare_trait_sumstats

    def prepare(*args, **kwargs):
        label = kwargs["label"]
        prepare_barrier.wait(timeout=5)
        if label == "trait1":
            raise ValueError("trait1 synthetic preparation failure")
        try:
            # Keep the peer inside its worker until the coordinator has seen
            # trait 1 fail and requested cooperative cancellation.
            time.sleep(0.15)
            return real_prepare(*args, **kwargs)
        finally:
            peer_exited.set()

    monkeypatch.setattr(bipred, "prepare_trait_sumstats", prepare)
    monkeypatch.setattr(
        bipred, "pair_prepared_traits",
        lambda *_args, **_kwargs: pytest.fail("pairing started after failure"))
    job_dir, job = _job(root, sources, suffix="failure-join")

    with pytest.raises(ValueError, match="synthetic preparation failure"):
        runner.run(job_dir, job)

    assert peer_exited.is_set()
    assert close_calls == [True]
    assert not list(prepared_store.store_dir(root).glob("*.part"))
    assert not list(prepared_store.store_dir(root).glob("*.lock"))


def test_structured_nonpositive_variance_withholds_weights(
        monkeypatch, tmp_path):
    root = tmp_path / "webdata"
    (root / "jobs").mkdir(parents=True)
    cache, sources = _inputs(tmp_path)
    monkeypatch.setenv("BIPRED_WEB_DATA", str(root))
    monkeypatch.setenv("BIPRED_WEB_CACHES", f"TEST={cache}")
    monkeypatch.setattr(
        "bipred.qc._ld_consistency_screen_selected",
        lambda _blocks, _indices, z, **_kwargs: np.ones(
            len(z), dtype=bool))

    diagnostic = bipred.bivariate._fit_divergence_statistics(
        beta1=np.zeros(1000), beta2=np.zeros(1000),
        raw_h2=(-0.01, 0.2), sigma_diag=(1.0, 1.0),
        genetic_samples=None, m=1000)
    assert diagnostic["traits"]["trait1"]["flags"][
        "nonpositive_genetic_variance"] is True

    fit = _fake_fit()
    fit.divergence_diagnostics = diagnostic

    def must_not_write(*_args, **_kwargs):
        raise AssertionError("unsafe weights must not be written")

    fit.write_weights = must_not_write
    monkeypatch.setattr(
        bipred, "ldpred3_auto_bivariate_blocks",
        lambda *_args, **_kwargs: fit)
    monkeypatch.setattr(
        runner, "_run_ldsc_regression",
        lambda prep, ell, panel: (
            (0.2, 0.3),
            {"h2_init": [0.2, 0.3], "h2_init_source": ["ldsc", "ldsc"],
             "h2": [0.2, 0.3], "rg": 0.2, "rg_se": 0.1,
             "gcov": 0.01, "gcov_intercept": 0.0}))

    job_dir, job = _job(root, sources, suffix="unsafe-weights")
    job["options"]["weights"] = True
    jobs.save_job(root, job)
    runner.run(job_dir, job)

    result = json.loads((job_dir / "result.json").read_text())
    assert result["weights"] == []
    assert result["diagnostics"]["critical"] is True
    assert result["diagnostics"]["valid_for_interpretation"] is False
    assert result["diagnostics"]["weights_withheld"] is True
    assert result["diagnostics"]["warnings"] == []
    assert result["diagnostics"]["divergence"]["traits"]["trait1"][
        "flags"]["nonpositive_genetic_variance"] is True
    assert not (job_dir / "weights1.tsv").exists()
    assert not (job_dir / "weights2.tsv").exists()
