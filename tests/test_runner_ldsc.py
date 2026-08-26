"""Focused contracts for the web runner's LDSC-based initialization."""

from types import SimpleNamespace

import numpy as np
import pytest

from webapp import runner


def _panel(scores=(1.0, 2.0, 3.0)):
    values = np.asarray(scores, dtype=np.float64)
    total = float(values.sum())
    m = len(values)
    return SimpleNamespace(
        scores=values, m_snps=m,
        definition="full-reference-colsum-r2-including-self",
        source="test", source_sha256="source-hash",
        score_sha256="score-hash", algorithm="test-v1",
        correction="none", score_mean=total / m, score_sum=total,
        effective_rank=m * m / total)


def _prep():
    return SimpleNamespace(
        beta_hat1=np.array([0.01, 0.02]),
        beta_hat2=np.array([0.02, 0.01]),
        n_eff1=np.array([100_000.0, 100_000.0]),
        n_eff2=np.array([90_000.0, 90_000.0]))


@pytest.mark.parametrize("failure", [
    FileNotFoundError("missing score panel"),
    ValueError("score-payload hash mismatch"),
])
def test_required_score_panel_errors_propagate(monkeypatch, failure):
    """Missing/corrupt reference data must not become an h2=0.1 fit."""
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(
        runner.caches, "load_or_create_ld_score_panel", fail)
    with pytest.raises(type(failure), match=str(failure)):
        runner._required_ld_score_rows(
            "cache", "root", cache_sha256="cache-hash", n_variants=3,
            cache_indices=np.array([0, 2]), fitted_shape=(2,))


def test_required_score_panel_gathers_only_aligned_rows(monkeypatch):
    panel = _panel()
    monkeypatch.setattr(
        runner.caches, "load_or_create_ld_score_panel",
        lambda *_args, **_kwargs: panel)

    loaded, rows = runner._required_ld_score_rows(
        "cache", "root", cache_sha256="cache-hash", n_variants=3,
        cache_indices=np.array([0, 2]), fitted_shape=(2,))
    assert loaded is panel
    np.testing.assert_array_equal(rows, [1.0, 3.0])

    with pytest.raises(ValueError, match="row indices are invalid"):
        runner._required_ld_score_rows(
            "cache", "root", cache_sha256="cache-hash", n_variants=3,
            cache_indices=np.array([0, 3]), fitted_shape=(2,))


def test_regression_failure_uses_recorded_deterministic_default(monkeypatch):
    """Only a failure after panel validation may use the fixed fallback."""
    import bipred.ldsc as ldsc_module

    def fail(*_args, **_kwargs):
        raise ValueError("data-dependent regression is singular")

    monkeypatch.setattr(ldsc_module, "ldsc_rg", fail)
    h2_init, diagnostic = runner._run_ldsc_regression(
        _prep(), np.array([1.0, 3.0]), _panel())

    assert h2_init == (0.1, 0.1)
    assert diagnostic["h2_init"] == [0.1, 0.1]
    assert diagnostic["h2_init_source"] == [
        "default_regression_failure", "default_regression_failure"]
    assert diagnostic["error"] == "data-dependent regression is singular"
    assert diagnostic["m_snps"] == 3
    assert diagnostic["n_regression_variants"] == 2


def test_unexpected_regression_bug_is_not_silently_downgraded(monkeypatch):
    import bipred.ldsc as ldsc_module

    def fail(*_args, **_kwargs):
        raise RuntimeError("programming error")

    monkeypatch.setattr(ldsc_module, "ldsc_rg", fail)
    with pytest.raises(RuntimeError, match="programming error"):
        runner._run_ldsc_regression(
            _prep(), np.array([1.0, 3.0]), _panel())


def test_nonfinite_ldsc_h2_falls_back_trait_wise(monkeypatch):
    import bipred.ldsc as ldsc_module

    result = SimpleNamespace(
        rg=0.2, rg_se=0.03, gcov=0.01, gcov_intercept=0.0,
        h2=(np.nan, 1.7))
    monkeypatch.setattr(
        ldsc_module, "ldsc_rg", lambda *_args, **_kwargs: result)

    h2_init, diagnostic = runner._run_ldsc_regression(
        _prep(), np.array([1.0, 3.0]), _panel())
    assert h2_init == (0.1, 1.0)
    assert diagnostic["h2_init_source"] == [
        "default_nonfinite_ldsc", "ldsc"]
    assert np.isnan(diagnostic["h2"][0])
