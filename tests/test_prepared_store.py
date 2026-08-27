"""Scientific-identity and concurrency tests for prepared-trait persistence."""

import gzip
import hashlib
import json
import os
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import numpy as np
import pytest

from bipred.prepare import PreparedTrait
from webapp import prepared_store


_SCREEN_PARAMS = {
    "rounds": 4,
    "window": 1000,
    "threshold": 29.72,
    "eigenvalue_floor": 1e-3,
    "seed": 17,
    "ncores": 1,
    "verbose": False,
}
_NUMERICAL_BACKEND = {
    "blas": {
        "implementation": "openblas",
        "version": "0.3.23",
        "integer_api": "lp64",
        "architecture": "x86_64",
    },
    "lapack": {
        "implementation": "openblas",
        "version": "0.3.23",
        "integer_api": "lp64",
        "architecture": "x86_64",
    },
}
_LDSC_IDENTITY = {
    "m_snps": 5,
    "score_sha256": "e" * 64,
    "definition": "full-reference-test",
    "source": "test scores",
    "source_sha256": None,
    "algorithm": "test-v1",
    "correction": "none",
    "parameters": {"chi2_cap": 80.0, "intercept": "free"},
}


def _spec(logical="a", ld="b", **overrides):
    options = {
        "logical_input_sha256": logical if len(logical) == 64 else logical * 64,
        "ld_sha256": ld if len(ld) == 64 else ld * 64,
        "n_semantics": {"mode": "scalar", "value": 1000.0},
        "columns": {"beta": "BETA", "se": "SE"},
        "qc": True,
        "qc_params": {"min_maf": 0.01},
        "screen": True,
        "screen_params": dict(_SCREEN_PARAMS),
        "numpy_version": "1.26.4-test",
        "numerical_backend": _NUMERICAL_BACKEND,
    }
    options.update(overrides)
    return prepared_store.semantic_spec(**options)


def _trait(beta=0.1, label="built"):
    return PreparedTrait(
        indices=np.array([1, 3], dtype=np.int64),
        beta_hat=np.array([beta, 0.2]),
        n_eff=np.array([1000.0, 1200.0]),
        z=np.array([2.0, 3.0]),
        eaf=np.array([0.2, np.nan]),
        n_cache=5,
        log={
            "label": label,
            "n_matched": 2,
            "screen": True,
            "ld_consistency_screen": {
                "n_input": 3,
                "n_kept": 2,
                "n_dropped": 1,
                "parameters": dict(_SCREEN_PARAMS),
            },
        },
    )


def test_prepared_store_roundtrip_relabels_and_replays_warnings(tmp_path):
    spec = _spec()
    calls = []

    def build():
        calls.append(1)
        warnings.warn("preparation warning", RuntimeWarning)
        return _trait()

    with pytest.warns(RuntimeWarning, match="preparation warning"):
        first, first_reused = prepared_store.get_or_build(
            tmp_path, spec, label="trait1", builder=build)
    with pytest.warns(RuntimeWarning, match="preparation warning"):
        second, second_reused = prepared_store.get_or_build(
            tmp_path, spec, label="trait2",
            builder=lambda: pytest.fail("reusable artifact was rebuilt"))

    assert calls == [1]
    assert (first_reused, second_reused) == (False, True)
    assert first.log["label"] == "trait1"
    assert second.log["label"] == "trait2"
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_allclose(first.beta_hat, second.beta_hat)

    key = prepared_store.key_for(spec)
    base = tmp_path / prepared_store.STORE_DIRNAME
    metadata = json.loads((base / f"{key}.json").read_text())
    assert metadata["spec"] == spec and "label" not in metadata["spec"]
    assert len(metadata["npz_sha256"]) == 64
    with np.load(base / f"{key}.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "indices", "beta_hat", "n_eff", "z", "eaf", "n_cache"}


def test_concurrent_builders_keep_warning_provenance_thread_local(tmp_path):
    """Overlapping Python 3.10 warning contexts must not steal each other."""
    barrier = threading.Barrier(2)
    before_hook = warnings.showwarning
    before_filters = list(warnings.filters)

    def run(name, logical, beta):
        def build():
            barrier.wait(timeout=5)
            warnings.warn(f"warning-{name}", RuntimeWarning)
            return _trait(beta=beta, label=name)

        return prepared_store.get_or_build(
            tmp_path, _spec(logical=logical), label=name, builder=build)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(run, "trait1", "c", 0.11)
            second = pool.submit(run, "trait2", "d", 0.12)
            assert first.result()[1] is False
            assert second.result()[1] is False

    assert {str(item.message) for item in caught} == {
        "warning-trait1", "warning-trait2"}
    assert warnings.showwarning is before_hook
    assert warnings.filters == before_filters
    metadata = [
        json.loads(path.read_text())
        for path in (tmp_path / prepared_store.STORE_DIRNAME).glob("*.json")
    ]
    by_logical = {
        item["spec"]["logical_input_sha256"]: item["warnings"]
        for item in metadata
    }
    assert by_logical["c" * 64][0]["message"] == "warning-trait1"
    assert by_logical["d" * 64][0]["message"] == "warning-trait2"


def test_prepared_semantic_key_tracks_every_preparation_input():
    payload = b"rsid\tbeta\nrs1\t0.1\n"
    gzip_a = gzip.compress(payload, mtime=1)
    gzip_b = gzip.compress(payload, mtime=2)
    assert gzip_a != gzip_b
    logical_a = hashlib.sha256(gzip.decompress(gzip_a)).hexdigest()
    logical_b = hashlib.sha256(gzip.decompress(gzip_b)).hexdigest()
    assert logical_a == logical_b

    base = _spec(logical=logical_a)
    assert prepared_store.key_for(base) == prepared_store.key_for(
        _spec(logical=logical_b))
    alternatives = [
        _spec(logical="c"),
        _spec(ld="d"),
        _spec(n_semantics={"mode": "per_variant"}),
        _spec(columns={"beta": "effect", "se": "SE"}),
        _spec(qc=False),
        _spec(qc_params={"min_maf": 0.02}),
        _spec(screen_params={**_SCREEN_PARAMS, "seed": 18}),
        _spec(screen_params={**_SCREEN_PARAMS, "threshold": 20.0}),
        _spec(algorithm_schema="prepared-trait-v5"),
        _spec(bipred_version="different-bipred"),
        _spec(ldpred3_version="different-ldpred3"),
        _spec(numpy_version="2.0.0-test"),
        _spec(numerical_backend={
            **_NUMERICAL_BACKEND,
            "blas": {**_NUMERICAL_BACKEND["blas"],
                     "implementation": "mkl"},
        }),
        _spec(numerical_backend={
            **_NUMERICAL_BACKEND,
            "lapack": {**_NUMERICAL_BACKEND["lapack"],
                       "version": "0.3.24"},
        }),
        _spec(numerical_backend={
            **_NUMERICAL_BACKEND,
            "blas": {**_NUMERICAL_BACKEND["blas"],
                     "architecture": "aarch64"},
        }),
    ]
    assert all(prepared_store.key_for(item) != prepared_store.key_for(base)
               for item in alternatives)

    with pytest.raises(ValueError, match="screen must be true"):
        _spec(screen=False)
    with pytest.raises(ValueError, match="unknown"):
        prepared_store.key_for({**base, "pairing": {}})
    missing = dict(base)
    missing.pop("screen")
    with pytest.raises(ValueError, match="missing"):
        prepared_store.key_for(missing)

    with_ldsc = _spec(
        logical=logical_a, pre_dentist_ldsc=_LDSC_IDENTITY)
    changed_scores = _spec(
        logical=logical_a,
        pre_dentist_ldsc={
            **_LDSC_IDENTITY, "score_sha256": "f" * 64,
        })
    assert prepared_store.key_for(with_ldsc) != \
        prepared_store.key_for(changed_scores)


def test_numerical_environment_is_strict_stable_and_inspectable():
    spec = _spec()
    environment = spec["numerical_environment"]
    assert environment == {
        "numpy_version": "1.26.4-test",
        "backend": _NUMERICAL_BACKEND,
    }

    with pytest.raises(ValueError, match="unknown"):
        _spec(numerical_backend={
            **_NUMERICAL_BACKEND,
            "blas": {
                **_NUMERICAL_BACKEND["blas"],
                "filepath": "/machine/specific/libblas.so",
            },
        })
    with pytest.raises(ValueError, match="unknown"):
        _spec(numerical_backend={
            **_NUMERICAL_BACKEND,
            "lapack": {
                **_NUMERICAL_BACKEND["lapack"], "num_threads": 8,
            },
        })
    with pytest.raises(ValueError, match="missing"):
        _spec(numerical_backend={"blas": _NUMERICAL_BACKEND["blas"]})


def test_detected_backend_omits_build_paths_and_thread_counts():
    config = {
        "Build Dependencies": {
            "blas": {
                "name": "OpenBLAS64", "version": "0.3.23",
                "include directory": "/build/one/include",
                "lib directory": "/build/one/lib",
                "openblas configuration": (
                    "USE_64BITINT=1 DYNAMIC_ARCH=1 MAX_THREADS=64"),
            },
        },
        "Machine Information": {
            "host": {"cpu": "amd64", "system": "linux"},
        },
    }
    identity = prepared_store._modern_dependency(config, "blas")
    assert identity == {
        "implementation": "openblas",
        "version": "0.3.23",
        "integer_api": "ilp64",
        "architecture": "x86_64",
    }
    assert set(identity) == {
        "implementation", "version", "integer_api", "architecture"}

    detected = prepared_store.semantic_spec(
        logical_input_sha256="a" * 64,
        ld_sha256="b" * 64,
        n_semantics={"mode": "scalar", "value": 1000.0},
        screen_params=_SCREEN_PARAMS)["numerical_environment"]
    assert detected["numpy_version"] == np.__version__
    assert set(detected["backend"]) == {"blas", "lapack"}
    assert all(set(component) == {
        "implementation", "version", "integer_api", "architecture",
    } for component in detected["backend"].values())


@pytest.mark.parametrize(("marker", "name", "version", "implementation"), [
    ("-DACCELERATE_NEW_LAPACK", "blas", "3.9.0", "accelerate"),
    ("-DHAVE_CBLAS", "OpenBLAS64_", "0.3.27", "openblas"),
    ("-DMKL_ILP64", "mkl_rt", "2025.1", "mkl"),
])
def test_modern_backend_family_is_normalised_without_build_paths(
        monkeypatch, marker, name, version, implementation):
    monkeypatch.setattr(prepared_store.platform, "mac_ver",
                        lambda: ("15.6", ("", "", ""), ""))
    config = {
        "Compilers": {
            "c": {
                "args": (
                    f"-I/private/build/ldpred3-accelerate/include {marker}"),
            },
        },
        "Machine Information": {
            "host": {"cpu": "arm64", "system": "darwin"},
        },
        "Build Dependencies": {
            "blas": {"name": name, "version": version},
            "lapack": {"name": name, "version": version},
        },
    }
    expected_version = "15.6" if implementation == "accelerate" else version
    for kind in ("blas", "lapack"):
        identity = prepared_store._modern_dependency(config, kind)
        assert identity == {
            "implementation": implementation,
            "version": expected_version,
            "integer_api": (
                "ilp64" if marker in {
                    "-DMKL_ILP64", "-DACCELERATE_LAPACK_ILP64"
                } or "64" in name else "lp64"),
            "architecture": "aarch64",
        }
        assert all("/" not in value for value in identity.values())


@pytest.mark.parametrize(("key", "libraries", "implementation"), [
    ("blas_opt_info", ["Accelerate"], "accelerate"),
    ("blas_opt_info", ["openblas"], "openblas"),
    ("blas_mkl_info", ["mkl_rt"], "mkl"),
])
def test_legacy_backend_family_is_normalised(
        monkeypatch, key, libraries, implementation):
    def get_info(candidate):
        if candidate == key:
            return {"libraries": libraries, "version": "test-version"}
        return {}

    monkeypatch.setattr(prepared_store.np.__config__, "get_info", get_info,
                        raising=False)
    monkeypatch.setattr(prepared_store.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(prepared_store.platform, "mac_ver",
                        lambda: ("15.6", ("", "", ""), ""))
    identity = prepared_store._legacy_dependency("blas")
    assert identity == {
        "implementation": implementation,
        "version": ("15.6" if implementation == "accelerate"
                    else "test-version"),
        "integer_api": "lp64",
        "architecture": "x86_64",
    }


def test_runtime_backend_uses_library_name_but_never_environment_path(
        monkeypatch):
    module = ModuleType("threadpoolctl")
    module.threadpool_info = lambda: [{
        "user_api": "blas", "internal_api": "openblas",
        "prefix": "libopenblas", "version": "0.3.27",
        "num_threads": 64,
        "filepath": "/tmp/mkl_ilp64_environment/lib/libopenblas.so",
    }]
    monkeypatch.setitem(sys.modules, "threadpoolctl", module)
    monkeypatch.setattr(prepared_store.np.linalg, "eigh",
                        lambda value: (value, value))
    identity = prepared_store._runtime_dependency("x86_64")
    assert identity == {
        "implementation": "openblas",
        "version": "0.3.27",
        "integer_api": "lp64",
        "architecture": "x86_64",
    }
    assert set(identity).isdisjoint({"filepath", "num_threads"})


@pytest.mark.parametrize("mutation", [
    lambda log: log.update(screen=False),
    lambda log: log.pop("ld_consistency_screen"),
    lambda log: log["ld_consistency_screen"].update(n_input=4),
    lambda log: log["ld_consistency_screen"].update(n_kept=1, n_dropped=2),
    lambda log: log["ld_consistency_screen"]["parameters"].update(seed=18),
])
def test_prepared_store_refuses_unscreened_or_mismatched_traits(
        tmp_path, mutation):
    trait = _trait()
    mutation(trait.log)
    with pytest.raises(ValueError, match="PreparedTrait.log"):
        prepared_store.get_or_build(
            tmp_path, _spec(), label="trait1", builder=lambda: trait)

    key = prepared_store.key_for(_spec())
    base = tmp_path / prepared_store.STORE_DIRNAME
    assert not (base / f"{key}.npz").exists()
    assert not (base / f"{key}.json").exists()


def test_prepared_store_same_key_runs_one_concurrent_builder(
        tmp_path, monkeypatch):
    spec = _spec()
    started = threading.Event()
    release = threading.Event()
    waiter_waiting = threading.Event()
    calls = []
    calls_lock = threading.Lock()

    def build():
        with calls_lock:
            calls.append(1)
        started.set()
        assert release.wait(5.0), "test did not release the prepared builder"
        return _trait()

    monkeypatch.setattr(prepared_store, "WAIT_POLL", 0.01)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(
            prepared_store.get_or_build, tmp_path, spec,
            label="trait1", builder=build)
        assert started.wait(5.0), "prepared builder did not start"
        waiter = pool.submit(
            prepared_store.get_or_build, tmp_path, spec,
            label="trait2", builder=build,
            on_wait=lambda _: waiter_waiting.set())
        try:
            assert waiter_waiting.wait(5.0), "second caller did not wait"
        finally:
            release.set()
        owner_result = owner.result(timeout=5.0)
        waiter_result = waiter.result(timeout=5.0)

    assert calls == [1]
    assert owner_result[1] is False
    assert waiter_result[1] is True
    assert waiter_result[0].log["label"] == "trait2"


def test_prepared_store_rebuilds_a_corrupt_npz(tmp_path):
    spec = _spec()
    calls = []

    def build():
        calls.append(1)
        return _trait(beta=0.1 * len(calls))

    first, reused = prepared_store.get_or_build(
        tmp_path, spec, label="trait1", builder=build)
    assert reused is False and first.beta_hat[0] == pytest.approx(0.1)
    key = prepared_store.key_for(spec)
    data = tmp_path / prepared_store.STORE_DIRNAME / f"{key}.npz"
    data.write_bytes(data.read_bytes()[:20])

    rebuilt, reused = prepared_store.get_or_build(
        tmp_path, spec, label="trait1", builder=build)

    assert calls == [1, 1]
    assert reused is False
    assert rebuilt.beta_hat[0] == pytest.approx(0.2)
    assert not list(data.parent.glob("*.part"))


def test_prepared_store_rebuilds_a_mismatched_screen_log(tmp_path):
    spec = _spec()
    prepared_store.get_or_build(
        tmp_path, spec, label="trait1", builder=_trait)
    key = prepared_store.key_for(spec)
    base = tmp_path / prepared_store.STORE_DIRNAME
    metadata_path = base / f"{key}.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["log"]["ld_consistency_screen"]["parameters"]["seed"] = 18
    metadata_path.write_text(json.dumps(metadata))
    calls = []

    def rebuild():
        calls.append(1)
        return _trait(beta=0.3)

    trait, reused = prepared_store.get_or_build(
        tmp_path, spec, label="trait2", builder=rebuild)

    assert calls == [1] and reused is False
    assert trait.beta_hat[0] == pytest.approx(0.3)
    assert trait.log["ld_consistency_screen"]["parameters"] == _SCREEN_PARAMS


def test_prepared_store_purge_is_least_recently_used(tmp_path, monkeypatch):
    old_spec, new_spec = _spec(logical="c"), _spec(logical="d")
    for label, spec in (("old", old_spec), ("new", new_spec)):
        prepared_store.get_or_build(
            tmp_path, spec, label=label, builder=_trait)
    base = tmp_path / prepared_store.STORE_DIRNAME
    old_key, new_key = (prepared_store.key_for(old_spec),
                        prepared_store.key_for(new_spec))
    for key, when in ((old_key, 1.0), (new_key, 2.0)):
        meta_path = base / f"{key}.json"
        metadata = json.loads(meta_path.read_text())
        metadata["last_used"] = when
        meta_path.write_text(json.dumps(metadata))
        os.utime(base / f"{key}.used", (when, when))

    monkeypatch.setattr(prepared_store, "EVICT_GRACE", 0.0)
    new_size = sum(
        path.stat().st_size for path in (
            base / f"{new_key}.npz", base / f"{new_key}.json",
            base / f"{new_key}.used"))
    removed = prepared_store.purge(
        tmp_path, budget_gb=(new_size * 1.01) / 2 ** 30)

    assert removed == [old_key]
    assert not (base / f"{old_key}.npz").exists()
    assert (base / f"{new_key}.npz").exists()


@pytest.mark.parametrize("trait", [
    PreparedTrait(
        np.array([1, 1]), np.array([0.1, 0.2]), np.ones(2), np.ones(2),
        np.ones(2) * 0.2, 5, {}),
    PreparedTrait(
        np.array([1]), np.array([0.1, 0.2]), np.ones(1), np.ones(1),
        np.ones(1) * 0.2, 5, {}),
    PreparedTrait(
        np.array([1]), np.array([0.1]), np.array([0.0]), np.ones(1),
        np.ones(1) * 0.2, 5, {}),
])
def test_prepared_store_rejects_invalid_sparse_arrays(tmp_path, trait):
    with pytest.raises(ValueError):
        prepared_store.get_or_build(
            tmp_path, _spec(), label="trait1", builder=lambda: trait)
