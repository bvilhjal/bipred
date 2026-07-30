"""Adaptive stopping and the implausible-fit diagnostic."""

import warnings

import numpy as np
import pytest

from bipred.bivariate import (_DIAGNOSTIC_MIN_VARIANTS,
                              ldpred3_auto_bivariate_blocks)


def panel(nb, k, n_ind, seed=0):
    """Blocks from a reference of n_ind samples. n_ind/k sets the conditioning."""
    rng = np.random.default_rng(seed)
    blocks = []
    idx = 0
    for _ in range(nb):
        X = rng.standard_normal((k, n_ind)).astype(np.float32)
        X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        R = (X @ X.T) / n_ind
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        blocks.append((np.ascontiguousarray(R, np.float32), np.arange(idx, idx + k)))
        idx += k
    return blocks, idx


def sumstats(m, n_causal, n_eff, seed=5, shared=True):
    rng = np.random.default_rng(seed)
    causal = rng.choice(m, n_causal, replace=False)
    bt1 = np.zeros(m)
    bt1[causal] = rng.standard_normal(n_causal) * np.sqrt(0.3 / n_causal)
    bt2 = bt1.copy() if shared else np.zeros(m)
    if not shared:
        other = rng.choice(np.setdiff1d(np.arange(m), causal), n_causal, replace=False)
        bt2[other] = rng.standard_normal(n_causal) * np.sqrt(0.3 / n_causal)
    return (bt1 + rng.standard_normal(m) / np.sqrt(n_eff),
            bt2 + rng.standard_normal(m) / np.sqrt(n_eff))


@pytest.fixture(scope="module")
def well_conditioned():
    blocks, m = panel(4, 300, 6000)
    b1, b2 = sumstats(m, 60, 2e5)
    return blocks, m, b1, b2


class TestAdaptiveStopping:

    def test_disabled_by_default(self, well_conditioned):
        blocks, m, b1, b2 = well_conditioned
        r = ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5,
                                          num_iter=60, burn_in=20, seed=3)
        assert r.retained_iterations == 60
        assert r.stopped_early is False

    def test_stops_early_when_enabled(self, well_conditioned):
        blocks, m, b1, b2 = well_conditioned
        r = ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5,
                                          num_iter=400, burn_in=20,
                                          tol=1e-2, check_every=25, seed=3)
        assert r.stopped_early is True
        assert 0 < r.retained_iterations < 400

    def test_early_stop_tracks_the_full_run(self, well_conditioned):
        """A stopped fit must agree with the full one on the headline outputs."""
        blocks, m, b1, b2 = well_conditioned
        kw = dict(num_iter=400, burn_in=20, seed=3)
        full = ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5, **kw)
        early = ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5,
                                              tol=1e-3, check_every=25, **kw)
        assert early.retained_iterations < full.retained_iterations
        assert abs(early.rg - full.rg) < 0.02
        for a, b in zip(early.h2, full.h2):
            assert abs(a - b) < 0.02
        assert np.corrcoef(early.beta1_est, full.beta1_est)[0, 1] > 0.99
        assert np.corrcoef(early.beta2_est, full.beta2_est)[0, 1] > 0.99

    def test_tighter_tolerance_keeps_more_iterations(self, well_conditioned):
        blocks, m, b1, b2 = well_conditioned
        kw = dict(num_iter=400, burn_in=20, check_every=25, seed=3)
        loose = ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5, tol=1e-1, **kw)
        tight = ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5, tol=1e-4, **kw)
        assert loose.retained_iterations <= tight.retained_iterations

    def test_sample_arrays_match_the_retained_count(self, well_conditioned):
        """Every per-iterate array must be truncated to what was actually kept."""
        blocks, m, b1, b2 = well_conditioned
        r = ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5, num_iter=400,
                                          burn_in=20, tol=1e-2, check_every=25, seed=3)
        n = r.retained_iterations
        assert r.pi_samples.shape[0] == n
        assert r.sigma_samples.shape[0] == n
        assert r.genetic_samples.shape[0] == n
        assert r.noise_scale_samples.shape[0] == n

    def test_decorrelated_rg_runs_the_full_schedule(self, well_conditioned):
        """Thinned effect samples are what that estimator averages over, so
        stopping early would quietly shrink its pair count."""
        blocks, m, b1, b2 = well_conditioned
        r = ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5, num_iter=60,
                                          burn_in=20, tol=1e-1, check_every=10,
                                          rg_decorrelated=True, seed=3)
        assert r.stopped_early is False
        assert r.retained_iterations == 60

    @pytest.mark.parametrize("bad", [-1.0, float("nan")])
    def test_rejects_invalid_tol(self, well_conditioned, bad):
        blocks, m, b1, b2 = well_conditioned
        with pytest.raises(ValueError):
            ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5, num_iter=20,
                                          burn_in=5, tol=bad, seed=3)

    @pytest.mark.parametrize("bad", [0, -5])
    def test_rejects_invalid_check_every(self, well_conditioned, bad):
        blocks, m, b1, b2 = well_conditioned
        with pytest.raises(ValueError):
            ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5, num_iter=20,
                                          burn_in=5, tol=1e-2, check_every=bad, seed=3)


class TestImplausibleFitDiagnostic:

    def test_quiet_on_a_well_conditioned_fit(self):
        blocks, m = panel(4, 300, 6000)
        assert m >= _DIAGNOSTIC_MIN_VARIANTS
        b1, b2 = sumstats(m, 60, 2e5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5,
                                          num_iter=80, burn_in=30, seed=3)

    def test_warns_on_a_degenerate_reference(self):
        """n_ref barely above the block size inflates h2 and the causal
        fraction, which is both statistically wrong and markedly slower."""
        blocks, m = panel(4, 300, 400)
        b1, b2 = sumstats(m, 60, 2e5)
        with pytest.warns(RuntimeWarning, match="Implausible bivariate fit"):
            ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5,
                                          num_iter=80, burn_in=30, seed=3)

    def test_quiet_below_the_variant_threshold(self):
        """Small panels carry no information for the heuristic, so no noise."""
        blocks, m = panel(2, 60, 80)
        assert m < _DIAGNOSTIC_MIN_VARIANTS
        b1, b2 = sumstats(m, 10, 2e5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ldpred3_auto_bivariate_blocks(blocks, b1, b2, 2e5, 2e5,
                                          num_iter=40, burn_in=15, seed=3)
