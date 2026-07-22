"""Unit tests for deterministic bivariate multi-chain aggregation."""

from types import SimpleNamespace

import numpy as np
import pytest

import bipred.multichain as multichain
from bipred.bivariate import ldpred3_auto_bivariate_blocks


def _result(m, retained, chain, *, genetic=None):
    draw = np.arange(retained, dtype=float)
    pi_samples = np.column_stack(
        (
            0.70 - 0.001 * draw,
            0.10 + 0.0002 * draw,
            0.10 + 0.0003 * draw,
            0.10 + 0.0005 * draw,
        )
    )
    sigma_samples = np.column_stack(
        (
            0.20 + 0.001 * draw + 0.01 * chain,
            0.30 + 0.001 * draw + 0.01 * chain,
            0.02 + 0.0001 * draw,
        )
    )
    if genetic is None:
        genetic = np.column_stack(
            (
                0.10 + 0.002 * draw + 0.01 * chain,
                0.03 + 0.001 * draw,
                0.20 + 0.002 * draw + 0.01 * chain,
            )
        )
    noise_samples = np.column_stack(
        (
            1.0 + 0.001 * draw + 0.01 * chain,
            1.1 + 0.001 * draw + 0.01 * chain,
        )
    )
    pi = pi_samples.mean(axis=0)
    sigma_mean = sigma_samples.mean(axis=0)
    return SimpleNamespace(
        beta1_est=np.full(m, chain + 1.0),
        beta2_est=np.full(m, 2.0 * (chain + 1.0)),
        h2=(0.9, 0.8),
        rg=-0.9,
        p=float(pi[1:].sum()),
        sigma=np.array(
            [
                [sigma_mean[0], sigma_mean[2]],
                [sigma_mean[2], sigma_mean[1]],
            ]
        ),
        pi=pi,
        pi_samples=pi_samples,
        sigma_samples=sigma_samples,
        genetic_samples=np.asarray(genetic, dtype=float),
        noise_scale=(1.5, 1.6),
        noise_scale_samples=noise_samples,
    )


def _fake_runner(calls):
    def run(blocks, beta_hat1, beta_hat2, n_eff1, n_eff2, **kwargs):
        chain = len(calls)
        calls.append(kwargs)
        return _result(len(beta_hat1), kwargs["num_iter"], chain)

    return run


def _inputs(m=5):
    blocks = [(np.eye(m), np.arange(m))]
    beta1 = np.linspace(0.01, 0.05, m)
    beta2 = -beta1
    return blocks, beta1, beta2


def test_default_starts_seeds_shared_prior_and_equal_pool(monkeypatch):
    calls = []
    monkeypatch.setattr(
        multichain, "ldpred3_auto_bivariate_blocks", _fake_runner(calls)
    )
    blocks, beta1, beta2 = _inputs()
    result = multichain.ldpred3_auto_bivariate_chains(
        blocks, beta1, beta2, 10_000, 12_000, num_iter=4, seed=17
    )

    expected_p = np.exp(np.linspace(np.log(1e-4), np.log(0.2), 4))
    np.testing.assert_allclose([call["p_init"] for call in calls], expected_p)
    assert all(call["pi_init"] is None for call in calls)
    assert len(set(call["seed"] for call in calls)) == 4
    np.testing.assert_array_equal(
        result.chain_seeds, [call["seed"] for call in calls]
    )
    np.testing.assert_allclose(result.p_inits, expected_p)
    np.testing.assert_allclose(
        result.pi_inits,
        [summary.pi_init for summary in result.chain_summaries],
    )

    _, prior1, prior2, _ = multichain._initial_hyperparameters(
        beta1.size, (0.1, 0.1), 0.02, 0.0
    )
    assert result.sigma_prior_scale == pytest.approx((prior1, prior2))
    assert all(
        call["sigma_prior_scale"] == pytest.approx((prior1, prior2))
        for call in calls
    )
    np.testing.assert_allclose(result.posterior.beta1_est, 2.5)
    np.testing.assert_allclose(result.posterior.beta2_est, 5.0)
    assert result.posterior.pi_samples.shape == (16, 4)
    assert result.posterior.genetic_samples.shape == (16, 3)
    assert result.posterior.noise_scale_samples.shape == (16, 2)
    assert len(result.chain_summaries) == 4
    assert result.n_chains == 4
    assert result.retained_per_chain == 4
    assert isinstance(
        result.basic_split_rhat, multichain.BivariateBasicSplitRHat
    )
    assert "gvar_1" in result.basic_split_rhat.rhat
    assert "gvar_2" in result.basic_split_rhat.rhat
    assert "h2_1" in result.basic_split_rhat.rhat
    assert "h2_2" in result.basic_split_rhat.rhat
    assert "gcov" in result.basic_split_rhat.rhat
    assert "noise_scale1" not in result.basic_split_rhat.rhat
    assert "noise_scale2" not in result.basic_split_rhat.rhat
    assert not hasattr(result, "converged")
    assert not hasattr(result.basic_split_rhat, "converged")

    calls_again = []
    monkeypatch.setattr(
        multichain,
        "ldpred3_auto_bivariate_blocks",
        _fake_runner(calls_again),
    )
    repeated = multichain.ldpred3_auto_bivariate_chains(
        blocks, beta1, beta2, 10_000, 12_000, num_iter=4, seed=17
    )
    np.testing.assert_array_equal(result.chain_seeds, repeated.chain_seeds)

    calls_with_noise = []
    monkeypatch.setattr(
        multichain,
        "ldpred3_auto_bivariate_blocks",
        _fake_runner(calls_with_noise),
    )
    with_noise = multichain.ldpred3_auto_bivariate_chains(
        blocks,
        beta1,
        beta2,
        10_000,
        12_000,
        num_iter=4,
        seed=17,
        noise_inflation=True,
    )
    assert "noise_scale1" in with_noise.basic_split_rhat.rhat
    assert "noise_scale2" in with_noise.basic_split_rhat.rhat


def test_explicit_pi_starts_are_the_alternative(monkeypatch):
    calls = []
    monkeypatch.setattr(
        multichain, "ldpred3_auto_bivariate_blocks", _fake_runner(calls)
    )
    blocks, beta1, beta2 = _inputs()
    pi_inits = np.array(
        [
            [0.90, 0.04, 0.03, 0.03],
            [0.80, 0.08, 0.07, 0.05],
            [0.70, 0.12, 0.10, 0.08],
            [0.60, 0.15, 0.13, 0.12],
        ]
    )
    result = multichain.ldpred3_auto_bivariate_chains(
        blocks,
        beta1,
        beta2,
        10_000,
        12_000,
        pi_inits=pi_inits,
        sigma_prior_scale=(0.7, 0.8),
        num_iter=4,
    )
    for call, expected in zip(calls, pi_inits):
        np.testing.assert_allclose(call["pi_init"], expected)
        assert call["p_init"] == pytest.approx(1.0 - expected[0])
        assert call["sigma_prior_scale"] == (0.7, 0.8)
    np.testing.assert_allclose(result.p_inits, 1.0 - pi_inits[:, 0])
    np.testing.assert_allclose(result.pi_inits, pi_inits)
    assert result.sigma_prior_scale == (0.7, 0.8)

    with pytest.raises(ValueError, match="either p_init_range or pi_inits"):
        multichain.ldpred3_auto_bivariate_chains(
            blocks,
            beta1,
            beta2,
            10_000,
            12_000,
            p_init_range=(0.01, 0.1),
            pi_inits=pi_inits,
            num_iter=4,
        )


def test_h2_and_rg_use_pooled_raw_genetic_traces(monkeypatch):
    calls = []
    genetic_values = [
        (0.10, 0.02, 0.40),
        (0.20, 0.04, 0.50),
        (0.30, 0.06, 0.60),
        (0.40, 0.08, 0.70),
    ]

    def run(blocks, beta_hat1, beta_hat2, n_eff1, n_eff2, **kwargs):
        chain = len(calls)
        calls.append(kwargs)
        genetic = np.tile(genetic_values[chain], (kwargs["num_iter"], 1))
        return _result(len(beta_hat1), kwargs["num_iter"], chain, genetic=genetic)

    monkeypatch.setattr(multichain, "ldpred3_auto_bivariate_blocks", run)
    blocks, beta1, beta2 = _inputs()
    result = multichain.ldpred3_auto_bivariate_chains(
        blocks, beta1, beta2, 10_000, 12_000, num_iter=4
    )

    expected_h2 = (0.25, 0.55)
    expected_rg = 0.05 / np.sqrt(expected_h2[0] * expected_h2[1])
    assert result.posterior.h2 == pytest.approx(expected_h2)
    assert result.posterior.rg == pytest.approx(expected_rg)
    assert result.posterior.h2 != pytest.approx((0.9, 0.8))
    assert result.posterior.rg != pytest.approx(-0.9)


def test_real_sampler_matches_manual_seeded_pooling():
    m = 6
    blocks = [(np.eye(m), np.arange(m))]
    beta1 = np.array([0.030, -0.020, 0.010, 0.025, -0.015, 0.005])
    beta2 = np.array([0.020, -0.010, 0.015, 0.030, -0.005, -0.010])
    sampler_kwargs = {
        "ld_int8": False,
        "h2_init": (0.1, 0.1),
        "h2_bounds": (1e-4, 1.0),
        "burn_in": 2,
        "num_iter": 4,
    }
    result = multichain.ldpred3_auto_bivariate_chains(
        blocks,
        beta1,
        beta2,
        5_000,
        6_000,
        n_chains=2,
        seed=91,
        **sampler_kwargs,
    )

    manual = []
    for chain_seed, p_init in zip(result.chain_seeds, result.p_inits):
        manual.append(
            ldpred3_auto_bivariate_blocks(
                blocks,
                beta1,
                beta2,
                5_000,
                6_000,
                p_init=float(p_init),
                pi_init=None,
                sigma_prior_scale=result.sigma_prior_scale,
                seed=int(chain_seed),
                rg_decorrelated=False,
                **sampler_kwargs,
            )
        )

    beta1_sum = np.zeros(m)
    beta2_sum = np.zeros(m)
    for chain in manual:
        beta1_sum += chain.beta1_est
        beta2_sum += chain.beta2_est
        assert chain.genetic_samples.shape == (4, 3)
        np.testing.assert_array_equal(
            chain.noise_scale_samples, np.ones((4, 2))
        )
    np.testing.assert_array_equal(
        result.posterior.beta1_est, beta1_sum / len(manual)
    )
    np.testing.assert_array_equal(
        result.posterior.beta2_est, beta2_sum / len(manual)
    )

    pooled_pi = np.concatenate([chain.pi_samples for chain in manual])
    pooled_sigma = np.concatenate([chain.sigma_samples for chain in manual])
    pooled_genetic = np.concatenate(
        [chain.genetic_samples for chain in manual]
    )
    pooled_noise = np.concatenate(
        [chain.noise_scale_samples for chain in manual]
    )
    np.testing.assert_array_equal(result.posterior.pi_samples, pooled_pi)
    np.testing.assert_array_equal(result.posterior.sigma_samples, pooled_sigma)
    np.testing.assert_array_equal(
        result.posterior.genetic_samples, pooled_genetic
    )
    np.testing.assert_array_equal(
        result.posterior.noise_scale_samples, pooled_noise
    )

    genetic_mean = pooled_genetic.mean(axis=0)
    expected_h2 = (
        float(np.clip(genetic_mean[0], 1e-4, 1.0)),
        float(np.clip(genetic_mean[2], 1e-4, 1.0)),
    )
    expected_rg = float(
        np.clip(
            genetic_mean[1] / np.sqrt(expected_h2[0] * expected_h2[1]),
            -1.0,
            1.0,
        )
    )
    assert result.posterior.h2 == expected_h2
    assert result.posterior.rg == expected_rg
    np.testing.assert_array_equal(result.posterior.pi, pooled_pi.mean(axis=0))
    np.testing.assert_array_equal(
        result.posterior.noise_scale, pooled_noise.mean(axis=0)
    )


def test_collapsed_finite_chain_is_not_filtered(monkeypatch):
    calls = []

    def run(blocks, beta_hat1, beta_hat2, n_eff1, n_eff2, **kwargs):
        chain = len(calls)
        calls.append(kwargs)
        result = _result(len(beta_hat1), kwargs["num_iter"], chain)
        if chain == 0:
            result.beta1_est.fill(0.0)
            result.beta2_est.fill(0.0)
            result.genetic_samples.fill(0.0)
        return result

    monkeypatch.setattr(multichain, "ldpred3_auto_bivariate_blocks", run)
    blocks, beta1, beta2 = _inputs()
    result = multichain.ldpred3_auto_bivariate_chains(
        blocks, beta1, beta2, 10_000, 12_000, num_iter=4
    )
    np.testing.assert_allclose(result.posterior.beta1_est, 2.25)
    np.testing.assert_allclose(result.posterior.beta2_est, 4.5)
    assert result.posterior.pi_samples.shape[0] == 4 * 4
    assert len(result.chain_summaries) == 4


def test_nonfinite_or_unequal_chain_aborts_the_fit(monkeypatch):
    calls = []

    def nonfinite(blocks, beta_hat1, beta_hat2, n_eff1, n_eff2, **kwargs):
        chain = len(calls)
        calls.append(kwargs)
        result = _result(len(beta_hat1), kwargs["num_iter"], chain)
        if chain == 2:
            result.genetic_samples[0, 0] = np.nan
        return result

    monkeypatch.setattr(
        multichain, "ldpred3_auto_bivariate_blocks", nonfinite
    )
    blocks, beta1, beta2 = _inputs()
    with pytest.raises(
        FloatingPointError, match=r"chain 2 .*non-finite genetic_samples"
    ):
        multichain.ldpred3_auto_bivariate_chains(
            blocks, beta1, beta2, 10_000, 12_000, num_iter=4
        )
    assert len(calls) == 3

    def unequal(blocks, beta_hat1, beta_hat2, n_eff1, n_eff2, **kwargs):
        result = _result(len(beta_hat1), kwargs["num_iter"], 0)
        result.genetic_samples = result.genetic_samples[:-1]
        return result

    monkeypatch.setattr(multichain, "ldpred3_auto_bivariate_blocks", unequal)
    with pytest.raises(RuntimeError, match="genetic_samples with shape"):
        multichain.ldpred3_auto_bivariate_chains(
            blocks, beta1, beta2, 10_000, 12_000, num_iter=4
        )


def test_sampler_failure_reports_chain_and_seed(monkeypatch):
    calls = []

    def fail(blocks, beta_hat1, beta_hat2, n_eff1, n_eff2, **kwargs):
        calls.append(kwargs)
        raise ArithmeticError("deliberate numerical failure")

    monkeypatch.setattr(multichain, "ldpred3_auto_bivariate_blocks", fail)
    blocks, beta1, beta2 = _inputs()
    expected_seed = int(multichain._deterministic_chain_seeds(19, 4)[0])
    with pytest.raises(
        RuntimeError,
        match=rf"chain 0 \(seed {expected_seed}\) failed: deliberate",
    ) as caught:
        multichain.ldpred3_auto_bivariate_chains(
            blocks, beta1, beta2, 10_000, 12_000, num_iter=4, seed=19
        )
    assert isinstance(caught.value.__cause__, ArithmeticError)
    assert len(calls) == 1


@pytest.mark.parametrize("num_iter", [2, 5])
def test_split_rhat_requires_even_retained_length_at_least_four(num_iter):
    blocks, beta1, beta2 = _inputs()
    with pytest.raises(ValueError, match="num_iter"):
        multichain.ldpred3_auto_bivariate_chains(
            blocks,
            beta1,
            beta2,
            10_000,
            12_000,
            num_iter=num_iter,
        )


def test_decorrelated_rg_is_explicitly_unsupported():
    blocks, beta1, beta2 = _inputs()
    with pytest.raises(ValueError, match="rg_decorrelated=True"):
        multichain.ldpred3_auto_bivariate_chains(
            blocks,
            beta1,
            beta2,
            10_000,
            12_000,
            rg_decorrelated=True,
            num_iter=4,
        )


def test_basic_split_rhat_formula_and_degeneracy_metadata():
    diagnostic = multichain._basic_split_rhat(
        {
            "moving": np.array([[0.0, 2.0, 0.0, 2.0],
                                [1.0, 3.0, 1.0, 3.0]]),
            "flat": np.ones((2, 4)),
            "stuck_apart": np.array([[1.0, 1.0, 1.0, 1.0],
                                      [2.0, 2.0, 2.0, 2.0]]),
        }
    )
    assert diagnostic.rhat["moving"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert not diagnostic.degenerate["moving"]
    assert np.isnan(diagnostic.rhat["flat"])
    assert diagnostic.degenerate["flat"]
    assert np.isposinf(diagnostic.rhat["stuck_apart"])
    assert diagnostic.degenerate["stuck_apart"]
    assert diagnostic.n_chains == 2
    assert diagnostic.half_length == 2
    assert not hasattr(diagnostic, "converged")
