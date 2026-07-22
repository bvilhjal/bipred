"""Sequential multi-chain inference for the bivariate LDpred3 sampler.

Every finite, equal-length chain contributes equally to the posterior.  Basic
split-Rhat is returned as a diagnostic only; this module makes no convergence
claim and never filters chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bivariate import (
    BivariateResult,
    _finite_pair,
    _finite_scalar_or_pair,
    _initial_hyperparameters,
    _integer_at_least,
    _validate_seed,
    ldpred3_auto_bivariate_blocks,
)

__all__ = [
    "BivariateBasicSplitRHat",
    "BivariateChainSummary",
    "MultiChainBivariateResult",
    "ldpred3_auto_bivariate_chains",
]


@dataclass
class BivariateBasicSplitRHat:
    """Classical basic split-Rhat for the named scalar traces."""

    rhat: dict[str, float]
    degenerate: dict[str, bool]
    n_chains: int
    half_length: int


@dataclass
class BivariateChainSummary:
    """Starting point and posterior summary for one retained chain."""

    seed: int
    p_init: float
    pi_init: np.ndarray = field(repr=False)
    h2: tuple
    rg: float
    p: float
    pi: np.ndarray = field(repr=False)
    sigma: np.ndarray = field(repr=False)
    noise_scale: tuple


@dataclass
class MultiChainBivariateResult:
    """Equal-weight pooled posterior and explicit chain diagnostics."""

    posterior: BivariateResult
    basic_split_rhat: BivariateBasicSplitRHat
    chain_summaries: tuple[BivariateChainSummary, ...]
    chain_seeds: np.ndarray = field(repr=False)
    p_inits: np.ndarray = field(repr=False)
    pi_inits: np.ndarray = field(repr=False)
    sigma_prior_scale: tuple
    n_chains: int
    retained_per_chain: int


def _deterministic_chain_seeds(seed, n_chains):
    """Spawn reproducible uint32 seeds and deterministically repair collisions."""
    sequence = np.random.SeedSequence(seed)
    children = sequence.spawn(n_chains)
    seeds = np.empty(n_chains, dtype=np.uint32)
    used = set()
    modulus = int(np.iinfo(np.uint32).max) + 1
    for index, child in enumerate(children):
        candidate = int(child.generate_state(1, dtype=np.uint32)[0])
        while candidate in used:
            candidate = (candidate + 1) % modulus
        used.add(candidate)
        seeds[index] = candidate
    return seeds


def _basic_split_rhat(traces):
    """Return classical basic split-Rhat for equal-length scalar traces."""
    if not traces:
        raise ValueError("traces must contain at least one named metric")
    shapes = {np.asarray(values).shape for values in traces.values()}
    if len(shapes) != 1:
        raise ValueError("all diagnostic traces must have the same shape")
    shape = shapes.pop()
    if len(shape) != 2 or shape[0] < 2 or shape[1] < 4 or shape[1] % 2:
        raise ValueError(
            "diagnostic traces must have shape (n_chains >= 2, even draws >= 4)"
        )
    n_chains, n_draws = shape
    half = n_draws // 2
    rhat = {}
    degenerate = {}
    for name, values in traces.items():
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(
                f"diagnostic trace {name!r} contains non-finite values"
            )
        split = np.concatenate((values[:, :half], values[:, half:]), axis=0)
        within = float(np.mean(np.var(split, axis=1, ddof=1)))
        split_means = np.mean(split, axis=1)
        between = float(half * np.var(split_means, ddof=1))
        variance_hat = ((half - 1) / half * within + between / half)
        is_degenerate = within <= 0.0
        degenerate[name] = bool(is_degenerate)
        if is_degenerate:
            # Identical constant split chains contain no scale information;
            # different constants have zero within-chain variance but positive
            # between-chain variance and therefore infinite disagreement.
            rhat[name] = float("inf") if between > 0.0 else float("nan")
        else:
            rhat[name] = float(np.sqrt(max(variance_hat, 0.0) / within))
    return BivariateBasicSplitRHat(
        rhat=rhat,
        degenerate=degenerate,
        n_chains=n_chains,
        half_length=half,
    )


def _validated_chain_traces(result, m, retained, chain_index, seed):
    """Validate one complete chain without silently discarding it."""
    label = f"chain {chain_index} (seed {seed})"
    arrays = {
        "beta1_est": (result.beta1_est, (m,)),
        "beta2_est": (result.beta2_est, (m,)),
        "pi": (result.pi, (4,)),
        "sigma": (result.sigma, (2, 2)),
        "pi_samples": (result.pi_samples, (retained, 4)),
        "sigma_samples": (result.sigma_samples, (retained, 3)),
        "genetic_samples": (
            getattr(result, "genetic_samples", None),
            (retained, 3),
        ),
        "noise_scale_samples": (
            getattr(result, "noise_scale_samples", None),
            (retained, 2),
        ),
    }
    converted = {}
    for name, (value, expected_shape) in arrays.items():
        array = np.asarray(value)
        if array.shape != expected_shape:
            raise RuntimeError(
                f"{label} returned {name} with shape {array.shape}; "
                f"expected {expected_shape}"
            )
        if not np.issubdtype(array.dtype, np.number) or not np.all(
            np.isfinite(array)
        ):
            raise FloatingPointError(f"{label} returned non-finite {name}")
        converted[name] = np.asarray(array, dtype=float)

    scalars = np.asarray(
        [
            result.h2[0],
            result.h2[1],
            result.rg,
            result.p,
            result.noise_scale[0],
            result.noise_scale[1],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(scalars)):
        raise FloatingPointError(f"{label} returned a non-finite summary")
    return converted


def _diagnostic_traces(
    genetic, pi, sigma, noise_scale, h2_bounds, noise_inflation
):
    """Build the small named scalar traces used by basic split-Rhat."""
    lo, hi = h2_bounds
    h21 = np.clip(genetic[:, :, 0], lo, hi)
    h22 = np.clip(genetic[:, :, 2], lo, hi)
    rg = np.clip(genetic[:, :, 1] / np.sqrt(h21 * h22), -1.0, 1.0)
    rho_beta = np.clip(
        sigma[:, :, 2] / np.sqrt(sigma[:, :, 0] * sigma[:, :, 1]),
        -1.0,
        1.0,
    )
    traces = {
        "gvar_1": genetic[:, :, 0],
        "gvar_2": genetic[:, :, 2],
        "h2_1": h21,
        "h2_2": h22,
        "gcov": genetic[:, :, 1],
        "rg": rg,
        "p": pi[:, :, 1:].sum(axis=2),
        "pi00": pi[:, :, 0],
        "pi10": pi[:, :, 1],
        "pi01": pi[:, :, 2],
        "pi11": pi[:, :, 3],
        "sigma1": sigma[:, :, 0],
        "sigma2": sigma[:, :, 1],
        "sigma12": sigma[:, :, 2],
        "rho_beta": rho_beta,
    }
    if noise_inflation:
        traces["noise_scale1"] = noise_scale[:, :, 0]
        traces["noise_scale2"] = noise_scale[:, :, 1]
    return traces


def ldpred3_auto_bivariate_chains(
    blocks,
    beta_hat1,
    beta_hat2,
    n_eff1,
    n_eff2,
    *,
    n_chains=4,
    p_init_range=None,
    pi_inits=None,
    prior_p_init=0.02,
    sigma_prior_scale=None,
    seed=0,
    **bivariate_kwargs,
):
    """Run deterministic bivariate chains sequentially and pool every chain.

    By default, initial union-causal probabilities are log-spaced from 1e-4 to
    0.2.  Explicit pi_inits, with one four-state row per chain, are an
    alternative.  The covariance prior scale is shared by all chains and is
    derived once from prior_p_init unless supplied explicitly.

    num_iter must be even and at least four.  Basic split-Rhat is diagnostic
    metadata only: high values never remove a chain or change the posterior.
    """
    n_chains = _integer_at_least("n_chains", n_chains, 2)
    if n_chains > int(np.iinfo(np.uint32).max) + 1:
        raise ValueError("n_chains exceeds the number of distinct uint32 seeds")
    seed = _validate_seed(seed)
    if seed is None:
        raise ValueError(
            "seed must be an integer for deterministic bivariate chains"
        )

    bivariate_kwargs = dict(bivariate_kwargs)
    for reserved in ("p_init", "pi_init", "seed", "sigma_prior_scale"):
        if reserved in bivariate_kwargs:
            raise ValueError(f"{reserved} is reserved for the chain driver")
    rg_decorrelated = bivariate_kwargs.pop("rg_decorrelated", False)
    if not isinstance(rg_decorrelated, (bool, np.bool_)):
        raise ValueError("rg_decorrelated must be True or False")
    if rg_decorrelated:
        raise ValueError(
            "rg_decorrelated=True is not supported by multi-chain inference"
        )

    retained = _integer_at_least(
        "num_iter", bivariate_kwargs.get("num_iter", 200), 4
    )
    if retained % 2:
        raise ValueError("num_iter must be even for basic split-Rhat")
    bivariate_kwargs["num_iter"] = retained
    noise_inflation = bivariate_kwargs.get("noise_inflation", False)
    if not isinstance(noise_inflation, (bool, np.bool_)):
        raise ValueError("noise_inflation must be True or False")

    bh1 = np.asarray(beta_hat1)
    bh2 = np.asarray(beta_hat2)
    if bh1.ndim != 1 or bh2.ndim != 1 or bh1.size == 0:
        raise ValueError("beta_hat1 and beta_hat2 must be nonempty vectors")
    if bh1.shape != bh2.shape:
        raise ValueError("beta_hat1 and beta_hat2 must have the same length")
    m = bh1.size
    try:
        blocks = list(blocks)
    except TypeError:
        raise ValueError("blocks must be a sequence of (LD, index) pairs") from None

    h2_init = bivariate_kwargs.get("h2_init", 0.1)
    rg_init = bivariate_kwargs.get("rg_init", 0.0)
    h2_init = _finite_scalar_or_pair("h2_init", h2_init)
    bivariate_kwargs["h2_init"] = h2_init
    h2_bounds = _finite_pair(
        "h2_bounds", bivariate_kwargs.get("h2_bounds", (1e-4, 1.0))
    )
    if not (
        0.0 < h2_bounds[0] <= min(h2_init)
        and max(h2_init) <= h2_bounds[1]
    ):
        raise ValueError("h2_bounds must contain both positive h2_init values")
    bivariate_kwargs["h2_bounds"] = h2_bounds

    explicit_pi = pi_inits is not None
    if explicit_pi:
        if p_init_range is not None:
            raise ValueError("pass either p_init_range or pi_inits, not both")
        try:
            raw_pi = np.asarray(pi_inits, dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "pi_inits must have shape (n_chains, 4)"
            ) from None
        if raw_pi.shape != (n_chains, 4):
            raise ValueError("pi_inits must have shape (n_chains, 4)")
        starts = []
        for row in raw_pi:
            starts.append(
                _initial_hyperparameters(
                    m, h2_init, 0.02, rg_init, pi_init=row
                )[0]
            )
        start_pi = np.asarray(starts)
        p_starts = 1.0 - start_pi[:, 0]
    else:
        if p_init_range is None:
            p_init_range = (1e-4, 0.2)
        p_lo, p_hi = _finite_pair("p_init_range", p_init_range)
        if not 0.0 < p_lo <= p_hi <= 1.0:
            raise ValueError("p_init_range must satisfy 0 < low <= high <= 1")
        p_starts = np.exp(
            np.linspace(np.log(p_lo), np.log(p_hi), n_chains)
        )
        start_pi = np.asarray(
            [
                _initial_hyperparameters(
                    m, h2_init, float(p_start), rg_init
                )[0]
                for p_start in p_starts
            ]
        )

    if sigma_prior_scale is None:
        _, prior_s1, prior_s2, _ = _initial_hyperparameters(
            m, h2_init, prior_p_init, rg_init
        )
        shared_prior_scale = (prior_s1, prior_s2)
    else:
        shared_prior_scale = _finite_scalar_or_pair(
            "sigma_prior_scale", sigma_prior_scale
        )

    chain_seeds = _deterministic_chain_seeds(seed, n_chains)
    beta1_sum = np.zeros(m)
    beta2_sum = np.zeros(m)
    pi_traces = []
    sigma_traces = []
    genetic_traces = []
    noise_traces = []
    summaries = []

    for index, (chain_seed, p_start, pi_start) in enumerate(
        zip(chain_seeds, p_starts, start_pi)
    ):
        try:
            chain_result = ldpred3_auto_bivariate_blocks(
                blocks,
                beta_hat1,
                beta_hat2,
                n_eff1,
                n_eff2,
                p_init=float(p_start),
                pi_init=pi_start if explicit_pi else None,
                sigma_prior_scale=shared_prior_scale,
                seed=int(chain_seed),
                rg_decorrelated=False,
                **bivariate_kwargs,
            )
        except Exception as error:
            raise RuntimeError(
                f"chain {index} (seed {int(chain_seed)}) failed: {error}"
            ) from error
        trace = _validated_chain_traces(
            chain_result, m, retained, index, int(chain_seed)
        )
        beta1_sum += trace["beta1_est"]
        beta2_sum += trace["beta2_est"]
        pi_traces.append(trace["pi_samples"])
        sigma_traces.append(trace["sigma_samples"])
        genetic_traces.append(trace["genetic_samples"])
        noise_traces.append(trace["noise_scale_samples"])
        summaries.append(
            BivariateChainSummary(
                seed=int(chain_seed),
                p_init=float(p_start),
                pi_init=pi_start.copy(),
                h2=tuple(float(x) for x in chain_result.h2),
                rg=float(chain_result.rg),
                p=float(chain_result.p),
                pi=trace["pi"].copy(),
                sigma=trace["sigma"].copy(),
                noise_scale=tuple(float(x) for x in chain_result.noise_scale),
            )
        )

    pi_by_chain = np.stack(pi_traces)
    sigma_by_chain = np.stack(sigma_traces)
    genetic_by_chain = np.stack(genetic_traces)
    noise_by_chain = np.stack(noise_traces)
    diagnostic = _basic_split_rhat(
        _diagnostic_traces(
            genetic_by_chain,
            pi_by_chain,
            sigma_by_chain,
            noise_by_chain,
            h2_bounds,
            bool(noise_inflation),
        )
    )

    pooled_pi = pi_by_chain.reshape(-1, 4)
    pooled_sigma = sigma_by_chain.reshape(-1, 3)
    pooled_genetic = genetic_by_chain.reshape(-1, 3)
    pooled_noise = noise_by_chain.reshape(-1, 2)
    pi_mean = pooled_pi.mean(axis=0)
    sigma_mean = pooled_sigma.mean(axis=0)
    genetic_mean = pooled_genetic.mean(axis=0)
    noise_mean = pooled_noise.mean(axis=0)
    h21 = float(np.clip(genetic_mean[0], *h2_bounds))
    h22 = float(np.clip(genetic_mean[2], *h2_bounds))
    rg = float(np.clip(genetic_mean[1] / np.sqrt(h21 * h22), -1.0, 1.0))

    posterior = BivariateResult(
        beta1_est=beta1_sum / n_chains,
        beta2_est=beta2_sum / n_chains,
        h2=(h21, h22),
        rg=rg,
        p=float(pi_mean[1:].sum()),
        sigma=np.array(
            [
                [sigma_mean[0], sigma_mean[2]],
                [sigma_mean[2], sigma_mean[1]],
            ]
        ),
        pi=pi_mean,
        pi_samples=pooled_pi,
        sigma_samples=pooled_sigma,
        noise_scale=(float(noise_mean[0]), float(noise_mean[1])),
        genetic_samples=pooled_genetic,
        noise_scale_samples=pooled_noise,
    )

    return MultiChainBivariateResult(
        posterior=posterior,
        basic_split_rhat=diagnostic,
        chain_summaries=tuple(summaries),
        chain_seeds=chain_seeds,
        p_inits=np.asarray(p_starts, dtype=float).copy(),
        pi_inits=start_pi.copy(),
        sigma_prior_scale=shared_prior_scale,
        n_chains=n_chains,
        retained_per_chain=retained,
    )
