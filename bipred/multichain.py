"""Deterministic multi-chain inference for the bivariate LDpred3 sampler.

Every finite, equal-length chain contributes equally to the posterior.  Basic
split-Rhat is returned as a diagnostic only; this module makes no convergence
claim and never filters chains.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bivariate import (
    _BivariateStart,
    _bivariate_options_from_kwargs,
    BivariateResult,
    _finite_pair,
    _initial_hyperparameters,
    _integer_at_least,
    _ldpred3_auto_bivariate_prepared,
    _prepare_bivariate_inputs,
    _rg_from_quadratics,
    _rg_from_quadratics_array,
    _validate_seed,
    _validate_sigma_prior_scale,
)
from ldpred3.diagnostics import (
    NamedBasicSplitRHat,
    basic_split_rhat,
    deterministic_chain_seeds,
)

__all__ = [
    "BivariateBasicSplitRHat",
    "BivariateChainSummary",
    "MultiChainBivariateResult",
    "ldpred3_auto_bivariate_chains",
]


# The public diagnostic type and the seed helper are LDpred3's, shared
# verbatim with GWFM's per-variant PIP diagnostic: ``deterministic_chain_seeds``
# was character-identical here, and ``basic_split_rhat`` (upstreamed from this
# module, delegated below) carries the same split/degeneracy semantics this
# package defined. The aliases keep bipred's internal call sites and tests
# stable while one implementation serves both samplers.
BivariateBasicSplitRHat = NamedBasicSplitRHat
_deterministic_chain_seeds = deterministic_chain_seeds


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
    """Equal-weight pooled posterior and explicit chain diagnostics.

    ``posterior.retained_iterations`` counts all pooled draws, while
    ``retained_per_chain`` records the fixed trace length of each chain.
    """

    posterior: BivariateResult
    basic_split_rhat: BivariateBasicSplitRHat
    chain_summaries: tuple[BivariateChainSummary, ...]
    chain_seeds: np.ndarray = field(repr=False)
    p_inits: np.ndarray = field(repr=False)
    pi_inits: np.ndarray = field(repr=False)
    sigma_prior_scale: tuple
    n_chains: int
    retained_per_chain: int


def _accumulate_chains(chain_args, chain_results, m, retained, beta1_sum,
                       beta2_sum, pi_traces, sigma_traces, genetic_traces,
                       noise_traces, summaries):
    """Validate and pool chain results in chain order.

    ``chain_results`` is consumed lazily. In the serial path that means a chain
    is only fitted when this loop asks for it, so a chain that fails validation
    aborts the fit without paying for the chains after it.
    """
    for (index, (chain_seed, p_start, pi_start)), chain_result in zip(
        chain_args, chain_results
    ):
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


def _basic_split_rhat(traces):
    """Delegate to :func:`ldpred3.diagnostics.basic_split_rhat`.

    The implementation upstreamed from this module keeps bipred's semantics
    verbatim -- predeclared equal halves, NaN for identical constant split
    chains, infinity for differing ones -- and is shared with GWFM's
    per-variant diagnostic. Retained as a named function so the historical
    call sites and tests read the same.
    """
    return basic_split_rhat(traces)


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
    # rg is the ratio of the *raw* quadratics, matching what the fit reports.
    # Dividing the raw covariance by the clamped variances saturates rg at +/-1
    # whenever a bound binds, which would also fake perfect between-chain
    # agreement in the split-Rhat below.
    rg = _rg_from_quadratics_array(genetic[:, :, 1], genetic[:, :, 0],
                                   genetic[:, :, 2])
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
    chain_ncores=1,
    **bivariate_kwargs,
):
    """Run deterministic bivariate chains and pool every chain equally.

    By default, initial union-causal probabilities are log-spaced from 1e-4 to
    0.2.  Explicit pi_inits, with one four-state row per chain, are an
    alternative.  The covariance prior scale is shared by all chains and is
    derived once from prior_p_init unless supplied explicitly.

    ``num_iter`` must be even and at least four. Adaptive stopping (``tol > 0``)
    is not supported because split-Rhat and equal pooling require equal-length
    traces. Basic split-Rhat is diagnostic metadata only: high values never
    remove a chain or change the posterior.

    ``chain_ncores > 1`` runs that many chains concurrently in threads. With
    Numba, the per-block sweep kernels release the GIL; the pure-Python fallback
    does not provide the same scaling. Chains share canonical LD payloads
    read-only and touch no other common mutable state, and each keeps its own
    deterministic seed, so a threaded run is bit-identical to a serial one.

    chain_ncores > 1 cannot be combined with the per-chain block threading of
    ncores > 1 because nested parallelism would oversubscribe the machine. Pick
    one axis. For n_chains >= the core count, chain_ncores is usually the better
    one, since it has no per-sweep synchronisation at all.
    """
    chain_ncores = _integer_at_least("chain_ncores", chain_ncores, 1)
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
    options = _bivariate_options_from_kwargs(
        bivariate_kwargs, caller="ldpred3_auto_bivariate_chains"
    )
    if options.rg_decorrelated:
        raise ValueError(
            "rg_decorrelated=True is not supported by multi-chain inference"
        )
    if options.tol > 0.0:
        raise ValueError(
            "tol > 0 is not supported by multi-chain inference because "
            "equal-length traces are required for pooling and split-Rhat"
        )
    if chain_ncores > 1 and options.ncores > 1:
        raise ValueError(
            "chain_ncores > 1 cannot be combined with ncores > 1: the two "
            "would create nested parallelism and oversubscribe the machine. "
            "Choose either chain-level threading (chain_ncores) or "
            "within-chain block threading (ncores)."
        )

    retained = options.num_iter
    if retained < 4:
        raise ValueError("num_iter must be an integer >= 4")
    if retained % 2:
        raise ValueError("num_iter must be even for basic split-Rhat")
    noise_inflation = options.noise_inflation
    h2_bounds = options.h2_bounds
    prepared = _prepare_bivariate_inputs(
        blocks, beta_hat1, beta_hat2, n_eff1, n_eff2, options
    )
    m = prepared.m

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
        initial_hypers = []
        for row in raw_pi:
            initial_hypers.append(
                _initial_hyperparameters(
                    m, options.h2_init, 0.02, options.rg_init, pi_init=row
                )
            )
        start_pi = np.asarray([initial[0] for initial in initial_hypers])
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
        initial_hypers = [
            _initial_hyperparameters(
                m, options.h2_init, float(p_start), options.rg_init
            )
            for p_start in p_starts
        ]
        start_pi = np.asarray([initial[0] for initial in initial_hypers])

    if sigma_prior_scale is None:
        _, prior_s1, prior_s2, _ = _initial_hyperparameters(
            m, options.h2_init, prior_p_init, options.rg_init
        )
        shared_prior_scale = (prior_s1, prior_s2)
    else:
        shared_prior_scale = _validate_sigma_prior_scale(sigma_prior_scale)

    chain_seeds = _deterministic_chain_seeds(seed, n_chains)
    chain_starts = tuple(
        _BivariateStart(
            pi=initial[0].copy(),
            s1=initial[1],
            s2=initial[2],
            s12=initial[3],
            psi1=shared_prior_scale[0],
            psi2=shared_prior_scale[1],
            seed=int(chain_seed),
        )
        for initial, chain_seed in zip(initial_hypers, chain_seeds)
    )
    beta1_sum = np.zeros(m)
    beta2_sum = np.zeros(m)
    pi_traces = []
    sigma_traces = []
    genetic_traces = []
    noise_traces = []
    summaries = []

    def _run_one(index, chain_seed, p_start, pi_start):
        """Fit one chain from shared inputs and chain-local mutable buffers."""
        try:
            return _ldpred3_auto_bivariate_prepared(
                prepared, options, chain_starts[index]
            )
        except FloatingPointError as error:
            # A diverged chain surfaces as FloatingPointError from the fit, and
            # _validated_chain_traces raises the same type for a non-finite
            # trace. Keep that type at this boundary instead of flattening the
            # arithmetic failure into a generic RuntimeError.
            raise FloatingPointError(
                f"chain {index} (seed {int(chain_seed)}) failed: {error}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                f"chain {index} (seed {int(chain_seed)}) failed: {error}"
            ) from error

    chain_args = list(enumerate(zip(chain_seeds, p_starts, start_pi)))
    pool = None
    if chain_ncores > 1 and n_chains > 1:
        # Threads, not processes: the sweep kernels release the GIL and the LD
        # blocks are read shared rather than pickled per worker. Keep at most
        # one future per worker: a Future retains its completed BivariateResult,
        # including two genome-length posterior vectors. Submitting every chain
        # at once therefore made completed-result memory O(n_chains * m).
        # Results are still consumed in chain order, so pooling remains
        # bit-identical to a serial run.
        from concurrent.futures import ThreadPoolExecutor

        workers = min(chain_ncores, n_chains)
        pool = ThreadPoolExecutor(max_workers=workers)

        def _bounded_results():
            pending = {}
            next_submit = 0
            for _ in range(workers):
                index, rest = chain_args[next_submit]
                pending[index] = pool.submit(_run_one, index, *rest)
                next_submit += 1
            for index in range(n_chains):
                future = pending.pop(index)
                result = future.result()
                try:
                    yield result
                finally:
                    # Do not retain the just-pooled genome-wide vectors in this
                    # suspended generator frame.
                    del result
                # Refill only after the caller has pooled and released the
                # yielded result. During pooling, the current result plus the
                # remaining futures therefore total at most ``workers``.
                if next_submit < n_chains:
                    new_index, rest = chain_args[next_submit]
                    pending[new_index] = pool.submit(
                        _run_one, new_index, *rest)
                    next_submit += 1

        chain_results = _bounded_results()
    else:
        # Lazy on purpose: a chain is only fitted when the loop below asks for
        # it, so a chain that fails validation aborts the fit without paying for
        # the chains after it. Materialising these eagerly would silently drop
        # that fail-fast behaviour.
        chain_results = (_run_one(index, *rest) for index, rest in chain_args)

    try:
        _accumulate_chains(
            chain_args, chain_results, m, retained, beta1_sum, beta2_sum,
            pi_traces, sigma_traces, genetic_traces, noise_traces, summaries)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

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
    # Raw quadratics, as in the single-chain fit: h2 is clamped for reporting,
    # but clamping the rg denominator alone would saturate the ratio.
    rg = _rg_from_quadratics(genetic_mean[1], genetic_mean[0], genetic_mean[2])
    beta1_sum /= n_chains
    beta2_sum /= n_chains

    posterior = BivariateResult(
        beta1_est=beta1_sum,
        beta2_est=beta2_sum,
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
        retained_iterations=int(pooled_pi.shape[0]),
        stopped_early=False,
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
