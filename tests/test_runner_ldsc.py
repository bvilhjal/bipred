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


class _TraitQC(SimpleNamespace):
    def __len__(self):
        return len(self.indices)


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


def test_pre_dentist_ldsc_uses_exact_z_full_scores_and_original_m(monkeypatch):
    panel = _panel(scores=(1.0, 2.0, 3.0, 4.0))
    panel.cache_sha256 = "c" * 64
    trait = _TraitQC(
        indices=np.array([0, 2, 3]),
        beta_hat=np.array([0.01, 0.02, 0.03]),
        z=np.array([2.0, 11.0, 3.0]),
        n_eff=np.array([100_000.0, 100_000.0, 100_000.0]),
        log={})
    seen = {}

    def fake_ldsc(chisq, scores, n_eff, **kwargs):
        seen.update(chisq=chisq.copy(), scores=scores.copy(),
                    n_eff=n_eff.copy(), kwargs=kwargs)
        return SimpleNamespace(
            h2=0.25, h2_se=0.03, intercept=1.04,
            intercept_se=0.02, mean_chisq=1.3, ratio=0.13)

    monkeypatch.setattr("ldpred3.ldsc_h2", fake_ldsc)
    out = runner._run_trait_ldsc_qc(trait, panel)

    # z²=121 exceeds max(80, .001N)=100 and is excluded from this regression
    # only. Scores remain rows of the full-reference vector and M remains 4.
    np.testing.assert_array_equal(seen["chisq"], [4.0, 9.0])
    np.testing.assert_array_equal(seen["scores"], [1.0, 4.0])
    np.testing.assert_array_equal(seen["n_eff"], [100_000.0, 100_000.0])
    assert seen["kwargs"]["m_snps"] == 4
    assert out["n_aligned_variants"] == 3
    assert out["n_regression_variants"] == 2
    assert out["n_chi2_excluded"] == 1
    assert out["used_for_filtering"] is False
    assert out["used_for_h2_init"] is False
    np.testing.assert_array_equal(trait.z, [2.0, 11.0, 3.0])
    assert len(trait.indices) == 3


def test_pre_dentist_nonfinite_regression_is_advisory(monkeypatch):
    panel = _panel(scores=(1.0, 2.0, 3.0))
    trait = _TraitQC(
        indices=np.array([0, 1, 2]),
        beta_hat=np.array([0.01, 0.02, 0.03]),
        z=np.array([1.0, 2.0, 3.0]),
        n_eff=np.full(3, 100_000.0), log={})
    monkeypatch.setattr(
        "ldpred3.ldsc_h2",
        lambda *_args, **_kwargs: SimpleNamespace(
            h2=np.nan, h2_se=np.nan, intercept=1.0,
            intercept_se=0.1, mean_chisq=2.0, ratio=0.0))

    out = runner._run_trait_ldsc_qc(trait, panel)

    assert out["status"] == "unavailable"
    assert "non-finite" in out["error"]
    assert out["h2"] is None and out["intercept"] is None


def test_qc_assessment_warns_but_does_not_turn_heuristics_into_filters():
    trait = SimpleNamespace(
        __len__=lambda self: 15_000,
        log={
            "qc": {"n_input": 100_000, "n_kept": 40_000},
            "harmonize": {"n_sumstats": 40_000},
        })
    # Special methods are looked up on the class, not an instance attribute.
    trait = type("Trait", (), {
        "__len__": lambda self: 15_000,
        "log": trait.log,
    })()
    ldsc = {
        "status": "available", "n_regression_variants": 15_000,
        "h2": -0.1, "intercept": 1.4, "mean_chi2": 1.2,
        "ratio": 0.7,
    }
    assessment = runner._assess_trait_quality(trait, ldsc)

    assert assessment["status"] == "warning"
    assert len(assessment["warnings"]) == 5
    assert assessment["n_usable"] == 15_000
    assert assessment["ldsc_thresholds_evaluated"] is True
