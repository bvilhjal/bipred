"""
Bivariate LDpred3-auto: jointly fit two traits that share an LD reference.

Each variant falls in one of **four** latent states with probabilities
``(pi00, pi10, pi01, pi11)``: causal for neither trait, trait 1 only, trait 2
only, or **both**. A trait-1-causal effect is ``N(0, s1)``, a trait-2-causal one
``N(0, s2)``, and a *both*-causal pair is drawn from ``N(0, Sigma)`` with
``Sigma = [[s1, s12], [s12, s2]]`` -- the off-diagonal ``s12`` is the genetic
covariance and is the only place the traits couple. The Gibbs step evaluates the
four bivariate-Gaussian likelihoods of the residual estimate, samples the state,
then draws the effects; ``pi`` and ``(s1, s2, s12)`` are re-estimated each sweep.

This **per-trait** indicator (rather than a single shared one) is what makes the
joint model adaptive: whether the two traits' causal variants co-occur is
*learned* (``pi11``), not assumed. Shared, genetically correlated traits can
borrow strength through the ``both`` component; disjoint traits can drive
``pi11 -> 0`` so the fits largely decouple.

Both GWAS are assumed to use the **same** LD reference (same ancestry). Sample
overlap can be passed via ``cross_corr`` (the cross-trait correlation of the
sampling errors); the default 0 assumes uncorrelated cross-trait sampling errors.
A cross-trait LDSC intercept can reflect overlap but also correlated confounding,
so it is not automatically interchangeable with this correlation.

A distinct failure mode sits with this parameter: a strong **environmental**
correlation between the traits (shared non-genetic effects) can dominate the
fit — the 0.2.0 environmental-overlap stress test measured joint-fit MAE up to
0.86 in that regime (``benchmarks/RESULTS.md``, Table 10). If the traits
plausibly share environmental correlation, set ``cross_corr`` from external
evidence rather than leaving it at the default 0.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
from ldpred3 import LowRankLD

# Keep the pinned private ldpred3 seam in one compatibility module; LowRankLD
# itself is public and imported directly above.
from ._ldpred3_compat import (
    HAVE_NUMBA,
    _Q8,
    _as_n_vector,
    _check_h2_p,
    _finite_control,
    _get_thread_id,
    _integer_at_least,
    _jit,
    _jit_fastmath_nogil,
    _jit_nogil,
    _set_threads,
    _validate_beta_hat,
    _validate_blocks,
    _validate_boolean_controls,
    _validate_iterations,
    _validate_seed,
    prange,
)

# int8 LD quantisation scale (127): correlations in [-1, 1] are stored as
# ``round(R * 127)`` int8 -- a quarter of the float32 memory -- and the sampler
# reads ``R[i, j] * (1 / 127)``. Imported from ldpred3 so bipred's encoding stays
# locked to the blocks ``ldpred3.compute_ld_blocks(quantize=True)`` produces.

__all__ = ["BivariateResult", "ldpred3_auto_bivariate",
           "ldpred3_auto_bivariate_blocks"]

DAMP = 0.2          # damping factor for the variance-component updates
_INIT_RHO_MAX = 0.999
_AUTO_INT8_MAX_BLOCK = 1500
# Thresholds for the implausible-fit diagnostic. A genome-wide fit has m in the
# tens of thousands upward and a causal fraction well below a half; below that
# many variants the heuristic carries no information, so it stays quiet.
_DIAGNOSTIC_MIN_VARIANTS = 1000
_DIAGNOSTIC_MAX_CAUSAL_FRACTION = 0.5
_DENSE = 0
_LOWRANK = 1


def _jit_parallel_uncached(func):
    """``_jit_parallel`` without Numba's on-disk cache.

    Each fused sweep driver below is jitted **twice** from one Python
    function -- once ``parallel=True`` for ``ncores > 1`` and once ``nogil=True``
    for the serial path. Numba keys its on-disk cache on (source file,
    qualname, first line, signature) and *not* on the compilation flags, so the
    two twins share a single cache entry and whichever compiled first is served
    to both. The default cache lives in ``__pycache__`` beside this file and
    persists, so one ``ncores=1`` run would otherwise disable block parallelism
    for every later run on that checkout, permanently and silently.

    Measured (m=20,000, k=500, 40 int8 blocks): ``ncores=4`` runs at 1.73
    ms/sweep from a clean cache but 5.38 ms/sweep -- no better than the 5.49
    serial baseline -- from a cache a prior serial run had touched. Opting the
    parallel twins out of the cache restores 1.77 ms/sweep, bit-identically.

    Only the parallel twins opt out. The serial path stays cached, so the
    default single-core run is unaffected; ``ncores > 1`` pays one compilation
    per process, against a fit that runs for minutes at genome scale.
    """
    if not HAVE_NUMBA:
        return func
    from numba import njit
    # Matches ldpred3's _jit_parallel exactly but for ``cache``.
    return njit(cache=False, parallel=True)(func)


@dataclass(frozen=True)
class _BivariateOptions:
    """Validated controls shared by every chain in one fit."""

    ld_int8: Optional[bool]
    h2_init: tuple
    rg_init: float
    cross_corr: float
    burn_in: int
    num_iter: int
    h2_bounds: tuple
    h2_cap: Optional[tuple]
    iw_df: float
    rg_decorrelated: bool
    noise_inflation: bool
    ni_damp: float
    pi_prior: float
    sample_every: int
    ncores: int
    tol: float
    check_every: int


@dataclass(frozen=True)
class _PreparedBivariateInputs:
    """Canonical read-only inputs reusable across independent chains."""

    beta_hat1: np.ndarray
    beta_hat2: np.ndarray
    n_eff1: np.ndarray
    n_eff2: np.ndarray
    blocks: tuple
    m: int
    n_const: bool


@dataclass(frozen=True)
class _BivariateStart:
    """Validated chain-specific initial state and shared prior target."""

    pi: np.ndarray
    s1: float
    s2: float
    s12: float
    psi1: float
    psi2: float
    seed: Optional[int]


def _finite_pair(name, value):
    """Return a pair of finite floats, rejecting scalar/bool surrogates."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain exactly two finite numbers")
    try:
        values = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must contain exactly two finite numbers") from None
    if (len(values) != 2
            or any(isinstance(x, (bool, np.bool_)) for x in values)):
        raise ValueError(f"{name} must contain exactly two finite numbers")
    try:
        pair = tuple(float(x) for x in values)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must contain exactly two finite numbers") from None
    if not all(np.isfinite(x) for x in pair):
        raise ValueError(f"{name} must contain exactly two finite numbers")
    return pair


def _finite_scalar_or_pair(name, value):
    """Return a positive finite pair, expanding a scalar to both traits."""
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{name} must be a positive finite scalar or pair")
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive finite scalar or pair") from None
    if any(isinstance(x, (bool, np.bool_, str, bytes)) for x in raw.flat):
        raise ValueError(f"{name} must be a positive finite scalar or pair")
    try:
        arr = raw.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive finite scalar or pair") from None
    if arr.ndim == 0:
        pair = (float(arr), float(arr))
    elif arr.shape == (2,):
        pair = (float(arr[0]), float(arr[1]))
    else:
        raise ValueError(f"{name} must be a positive finite scalar or pair")
    if not all(np.isfinite(x) and x > 0.0 for x in pair):
        raise ValueError(f"{name} must be a positive finite scalar or pair")
    return pair


def _initial_hyperparameters(m, h2_init, p_init, rg_init, pi_init=None):
    """Build a four-state start whose implied h2 and genetic rg are exact.

    ``p_init`` is the union probability P(trait 1 or trait 2 causal). Its
    shorthand start divides the non-null mass equally unless a larger shared
    component is required to represent ``rg_init`` with a valid within-shared
    effect correlation. ``pi_init`` exposes the otherwise-unidentified overlap
    degree of freedom directly.
    """
    h21, h22 = _finite_scalar_or_pair("h2_init", h2_init)
    rg_init = _finite_control("rg_init", rg_init)
    if not -1.0 < rg_init < 1.0:
        raise ValueError("rg_init must be in (-1, 1)")

    if pi_init is None:
        _check_h2_p(p=p_init)
        q = float(p_init)
        # Preserve the historical equal split at modest |rg|. For large |rg|,
        # increase the shared mass just enough that the within-shared effect
        # correlation remains at or below the sampler's safe 0.999 boundary.
        shared = q / 3.0
        if rg_init != 0.0:
            shared = max(
                shared,
                abs(rg_init) * q / (2.0 * _INIT_RHO_MAX - abs(rg_init)),
            )
        # |rg_init| above the 0.999 boundary would require more shared mass
        # than the union probability. Saturate at an all-shared start rather
        # than producing negative single-trait mass: with single == 0 the
        # implied rg equals rho_beta = rg_init, so the implied moments stay
        # exact for every rg_init in (-1, 1).
        shared = min(shared, q)
        single = (q - shared) / 2.0
        pi = np.array([1.0 - q, single, single, shared], dtype=float)
    else:
        try:
            pi = np.asarray(pi_init, dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("pi_init must contain four finite probabilities") from None
        if (pi.shape != (4,) or not np.all(np.isfinite(pi))
                or np.any(pi < 0.0) or not np.isclose(pi.sum(), 1.0,
                                                     rtol=0.0, atol=1e-7)):
            raise ValueError("pi_init must contain four nonnegative probabilities summing to 1")
        pi = pi / pi.sum()

    p1 = float(pi[1] + pi[3])
    p2 = float(pi[2] + pi[3])
    shared = float(pi[3])
    if p1 <= 0.0 or p2 <= 0.0:
        raise ValueError("pi_init must give each trait positive causal probability")

    s1 = h21 / (float(m) * p1)
    s2 = h22 / (float(m) * p2)
    if rg_init == 0.0:
        rho_beta = 0.0
    else:
        if shared <= 0.0:
            raise ValueError("nonzero rg_init requires positive shared pi_init mass")
        rho_beta = rg_init * np.sqrt(p1 * p2) / shared
        if abs(rho_beta) >= 1.0:
            raise ValueError(
                "pi_init cannot represent rg_init: the implied within-shared "
                "effect correlation lies outside (-1, 1)"
            )
    s12 = float(rho_beta * np.sqrt(s1 * s2))
    return pi, float(s1), float(s2), s12


def _apply_R_rows(fblocks, V):
    """Right-multiply each row of ``V`` (n, m) by the block-diagonal LD ``R``
    (rows are ``R @ v`` since ``R`` is symmetric), block by block. Dense int8
    blocks carry a dequantisation scale. Low-rank blocks use the same globally
    scaled factor and diagonal residual as the Gibbs sweep, so diagnostics never
    silently evaluate a different effective LD matrix."""
    out = np.zeros_like(V)
    for kind, data, start, k, aux, residual, _score1, _score2 in fblocks:
        sl = slice(start, start + k)
        if kind == _LOWRANK:
            W = data.astype(V.dtype) * aux
            out[:, sl] = ((V[:, sl] @ W) @ W.T
                          + V[:, sl] * residual.astype(V.dtype))
        else:
            out[:, sl] = (V[:, sl] @ data.astype(V.dtype)) * aux
    return out


def _prepare_block(R, ld_int8):
    """Return ``(block, scale)`` for one dense LD block.

    Blocks that are already int8 (built by
    ``ldpred3.compute_ld_blocks(quantize=True)``) are kept int8 as-is, without a
    copy. The default ``ld_int8=False`` likewise keeps a float block as it was
    given: ``np.ascontiguousarray`` on an already-C-contiguous float32 array
    returns a view, so no second payload is built.

    ``True`` quantises every float block and ``None`` quantises those with at
    most 1500 variants -- both allocate a fresh int8 array per block *inside the
    fit*, while the caller's panel is still alive. That measured 78.4 MB of peak
    against 13.1 MB at m=100,000 with 500-variant blocks, and the extra payload
    is k/2 bytes per variant, so ~500 MB at m=1,000,000. Quantise when the LD is
    built (``compute_ld_blocks(quantize=True)``), where the float source is
    private and discardable, rather than here.

    The paired ``scale`` (``1/127`` for int8, ``1.0`` for float32) is what the
    sampler multiplies each LD entry by to dequantise on the fly."""
    arr = np.asarray(R)
    if arr.dtype == np.int8:
        return np.ascontiguousarray(arr), 1.0 / _Q8
    use_int8 = (ld_int8 is True or
                (ld_int8 is None and arr.shape[0] <= _AUTO_INT8_MAX_BLOCK))
    if use_int8:
        q = np.rint(np.clip(np.ascontiguousarray(arr, np.float32), -1.0, 1.0) * _Q8)
        return q.astype(np.int8), 1.0 / _Q8
    return np.ascontiguousarray(arr, dtype=np.float32), 1.0


def _prepare_bivariate_lowrank_block(R):
    """Return the current compact-LD payload without expanding per-row state.

    ``_validate_blocks`` has already checked the public ``LowRankLD`` contract:
    one global factor scale and a float32 diagonal residual. Keeping those
    native representations avoids two float64 length-m copies per prepared
    panel.
    """
    raw = np.asarray(R.U)
    dtype = np.int8 if raw.dtype == np.int8 else np.float32
    return (
        np.ascontiguousarray(raw, dtype=dtype),
        float(R.scale),
        np.ascontiguousarray(R.residual_diag, dtype=np.float32),
    )


def _validate_bivariate_options(*, ld_int8, h2_init,
                                rg_init, cross_corr, burn_in, num_iter,
                                h2_bounds, h2_cap, iw_df, rg_decorrelated,
                                noise_inflation, ni_damp, pi_prior,
                                sample_every, ncores, tol, check_every):
    """Validate and canonicalise controls shared by one or more chains."""
    h2_init = _finite_scalar_or_pair("h2_init", h2_init)
    rg_init = _finite_control("rg_init", rg_init)
    if not -1.0 < rg_init < 1.0:
        raise ValueError("rg_init must be in (-1, 1)")
    cross_corr = _finite_control("cross_corr", cross_corr)
    if not -1.0 < cross_corr < 1.0:
        raise ValueError("cross_corr must be in (-1, 1)")
    burn_in, num_iter = _validate_iterations(burn_in, num_iter)
    if ld_int8 is not None:
        _validate_boolean_controls(ld_int8=ld_int8)
        ld_int8 = bool(ld_int8)
    _validate_boolean_controls(rg_decorrelated=rg_decorrelated,
                               noise_inflation=noise_inflation)
    iw_df = _finite_control("iw_df", iw_df)
    if iw_df <= 0.0:
        raise ValueError("iw_df must be positive")
    ni_damp = _finite_control("ni_damp", ni_damp)
    if not 0.0 < ni_damp <= 1.0:
        raise ValueError("ni_damp must be in (0, 1]")
    pi_prior = _finite_control("pi_prior", pi_prior)
    if pi_prior <= 0.0:
        raise ValueError(
            "pi_prior must be positive (an improper <=0 concentration can "
            "collapse the mixture)"
        )
    tol = _finite_control("tol", tol, lower=0.0)
    check_every = _integer_at_least("check_every", check_every, 1)
    sample_every = _integer_at_least("sample_every", sample_every, 1)
    if rg_decorrelated and num_iter <= sample_every:
        raise ValueError(
            "rg_decorrelated=True requires at least two retained effect samples; "
            "num_iter must be greater than sample_every"
        )
    ncores = _integer_at_least("ncores", ncores, 1)

    lo, hi = _finite_pair("h2_bounds", h2_bounds)
    if not (0.0 < lo <= min(h2_init) and max(h2_init) <= hi):
        raise ValueError("h2_bounds must contain both positive h2_init values")
    h2_bounds = (lo, hi)
    if h2_cap is not None:
        h2_cap = _finite_pair("h2_cap", h2_cap)
        if h2_cap[0] <= 0.0 or h2_cap[1] <= 0.0:
            raise ValueError("h2_cap values must be positive")

    return _BivariateOptions(
        ld_int8=ld_int8,
        h2_init=h2_init,
        rg_init=rg_init,
        cross_corr=cross_corr,
        burn_in=burn_in,
        num_iter=num_iter,
        h2_bounds=h2_bounds,
        h2_cap=h2_cap,
        iw_df=iw_df,
        rg_decorrelated=bool(rg_decorrelated),
        noise_inflation=bool(noise_inflation),
        ni_damp=ni_damp,
        pi_prior=pi_prior,
        sample_every=sample_every,
        ncores=ncores,
        tol=tol,
        check_every=check_every,
    )


_BIVARIATE_OPTION_DEFAULTS = {
    "ld_int8": False,
    "h2_init": 0.1,
    "rg_init": 0.0,
    "cross_corr": 0.0,
    "burn_in": 200,
    "num_iter": 200,
    "h2_bounds": (1e-4, 1.0),
    "h2_cap": None,
    "iw_df": 10.0,
    "rg_decorrelated": False,
    "noise_inflation": False,
    "ni_damp": 0.1,
    "pi_prior": 1.0,
    "sample_every": 5,
    "ncores": 1,
    "tol": 0.0,
    "check_every": 50,
}


def _bivariate_options_from_kwargs(kwargs, *, caller):
    """Validate a public ``**kwargs`` mapping without starting a chain."""
    unknown = sorted(set(kwargs) - set(_BIVARIATE_OPTION_DEFAULTS))
    if unknown:
        raise TypeError(
            f"{caller}() got an unexpected keyword argument {unknown[0]!r}"
        )
    values = dict(_BIVARIATE_OPTION_DEFAULTS)
    values.update(kwargs)
    return _validate_bivariate_options(**values)


def _readonly_view(value):
    """Return a non-writeable view without changing the caller's array flags."""
    view = np.asarray(value).view()
    view.setflags(write=False)
    return view


def _prepare_bivariate_inputs(blocks, beta_hat1, beta_hat2, n_eff1, n_eff2,
                              options):
    """Validate and canonicalise immutable data shared across chains once."""
    bh1 = np.ascontiguousarray(
        _validate_beta_hat(beta_hat1), dtype=np.float64
    )
    bh2 = np.ascontiguousarray(
        _validate_beta_hat(beta_hat2), dtype=np.float64
    )
    m = bh1.shape[0]
    if m == 0:
        raise ValueError("beta_hat vectors must contain at least one variant")
    if bh2.shape[0] != m:
        raise ValueError("beta_hat1 and beta_hat2 must have the same length")
    n1 = np.ascontiguousarray(_as_n_vector(n_eff1, m), dtype=np.float64)
    n2 = np.ascontiguousarray(_as_n_vector(n_eff2, m), dtype=np.float64)
    n_const = bool(n1.min() == n1.max() and n2.min() == n2.max())

    validated_blocks = _validate_blocks(blocks, m, contiguous=True)
    prepared_blocks = []
    for R, idx in sorted(validated_blocks, key=lambda bi: int(bi[1][0])):
        start = int(idx[0])
        size = int(idx.shape[0])
        if isinstance(R, LowRankLD):
            U, scale, residual = _prepare_bivariate_lowrank_block(R)
            prepared_blocks.append(
                (
                    _LOWRANK,
                    _readonly_view(U),
                    start,
                    size,
                    scale,
                    _readonly_view(residual),
                )
            )
        else:
            Rq, scale = _prepare_block(R, options.ld_int8)
            prepared_blocks.append(
                (_DENSE, _readonly_view(Rq), start, size, scale, None)
            )

    return _PreparedBivariateInputs(
        beta_hat1=_readonly_view(bh1),
        beta_hat2=_readonly_view(bh2),
        n_eff1=_readonly_view(n1),
        n_eff2=_readonly_view(n2),
        blocks=tuple(prepared_blocks),
        m=m,
        n_const=n_const,
    )


def _validate_sigma_prior_scale(value):
    """Canonicalise a shared covariance-prior scale once."""
    if value is None:
        return None
    return _finite_scalar_or_pair("sigma_prior_scale", value)


def _prepare_bivariate_start(m, options, *, p_init, pi_init,
                             sigma_prior_scale, seed):
    """Build one validated chain start from canonical shared controls."""
    seed = _validate_seed(seed)
    pi, s1, s2, s12 = _initial_hyperparameters(
        m, options.h2_init, p_init, options.rg_init, pi_init=pi_init
    )
    if sigma_prior_scale is None:
        psi1, psi2 = s1, s2
    else:
        psi1, psi2 = sigma_prior_scale
    return _BivariateStart(
        pi=pi,
        s1=s1,
        s2=s2,
        s12=s12,
        psi1=float(psi1),
        psi2=float(psi2),
        seed=seed,
    )


def _instantiate_chain_blocks(prepared):
    """Add fresh mutable low-rank projection buffers for one chain."""
    blocks = []
    for kind, data, start, size, aux, residual in prepared.blocks:
        if kind == _LOWRANK:
            rank = int(data.shape[1])
            blocks.append(
                (
                    kind,
                    data,
                    start,
                    size,
                    aux,
                    residual,
                    np.zeros(rank),
                    np.zeros(rank),
                )
            )
        else:
            blocks.append(
                (kind, data, start, size, aux, residual, None, None)
            )
    return blocks


def _bivar_converged(avg1, avg2, prev1, prev2, count, rg, prev_rg, tol):
    """Adaptive-stopping test on the running bivariate posterior.

    Mirrors ldpred3's univariate ``_rms_converged`` -- relative RMS change of the
    running posterior mean -- but requires *both* traits to pass, and adds the
    running ``rg``. rg is included because it is a headline output of a bivariate
    fit and converges on its own timescale: the two effect vectors can settle
    while the genetic covariance is still drifting, so a betas-only test would
    stop too early and quietly degrade the quantity most callers came for.

    ``prev1`` / ``prev2`` hold the previous snapshot and are updated in place.
    Returns ``(converged, rg)``.
    """
    ok = True
    for avg, prev in ((avg1, prev1), (avg2, prev2)):
        mean = avg / count
        delta = mean - prev
        num = float(delta @ delta)
        den = float(mean @ mean)
        prev[:] = mean
        if num > tol * tol * den:
            ok = False
    if prev_rg is not None and abs(rg - prev_rg) > tol:
        ok = False
    return ok, rg


def _check_fit_is_finite(quadratics, beta1, beta2):
    """Reject a fit whose estimates are not finite.

    A Gibbs chain on a non-PSD LD reference can diverge to +/-inf and then to
    NaN. NaN is not self-announcing here: the sweep's log-sum-exp leaves
    ``wmax = w0`` (every ``w > wmax`` test is False for NaN), all four state
    probabilities become NaN, all three ``u < p`` tests are False, and the
    variant silently falls through to the both-causal branch. Without this
    check the fit returns NaN ``h2``, ``rg``, ``sigma`` and effect vectors with
    no error, and the h2 clamp below would additionally launder a diverged
    (large negative) genetic variance into an ordinary-looking low-h2 result.
    """
    if not np.all(np.isfinite(quadratics)):
        raise FloatingPointError(
            "the bivariate fit produced non-finite genetic quadratic forms "
            f"{tuple(float(q) for q in quadratics)}; the sampler diverged, "
            "which usually means the LD reference is not positive "
            "semi-definite. Regularise it (ldpred3.shrink_ld_blocks), use "
            "smaller blocks, or use a larger reference panel."
        )
    if not (np.all(np.isfinite(beta1)) and np.all(np.isfinite(beta2))):
        raise FloatingPointError(
            "the bivariate fit produced non-finite posterior-mean effects; "
            "the sampler diverged. Regularise the LD reference "
            "(ldpred3.shrink_ld_blocks), use smaller blocks, or use a larger "
            "reference panel."
        )


def _warn_if_implausible_fit(raw_h2, p, h2_bounds, m):
    """Flag a fit that is simultaneously statistically suspect and slow.

    A poorly conditioned LD reference inflates h2, which inflates the fitted
    causal fraction, which makes the guarded per-variant LD row update fire for
    nearly every variant instead of a small minority. That adds a real slowdown
    on top of a wrong answer, so it is worth saying out loud rather than leaving
    the caller to wonder why a run was both slow and implausible.

    ``raw_h2`` is the pair of *unclamped* sampled quadratics, so a bound that
    binds is visible here; the reported ``h2`` has already been clamped into
    range and could not reveal it. The test is two-sided, with the low end
    split in two because the two cases differ in kind: a **non-positive**
    sampled genetic variance (possible on a non-PSD int8-quantised block) is
    degenerate and also zeroes ``rg``, whereas a small but strictly positive
    quadratic under the caller's own ``lo`` only means the reported ``h2`` is
    a clamped value -- ``rg`` is unaffected, and on a genuinely
    low-heritability trait that is not evidence of anything wrong.
    """
    # Only meaningful at scale. On a handful of variants a large causal fraction
    # or an h2 at its bound is ordinary -- there is not enough data for either to
    # be evidence of anything -- and warning there would be pure noise for the
    # small synthetic panels used in tests and demos.
    if m < _DIAGNOSTIC_MIN_VARIANTS:
        return
    lo, hi = h2_bounds
    reasons = []
    if any(v >= hi * (1.0 - 1e-6) for v in raw_h2):
        reasons.append("h2 reached its upper bound %g" % hi)
    if any(v <= 0.0 for v in raw_h2):
        # The exact condition under which _rg_from_quadratics returns 0.0.
        reasons.append(
            "a sampled genetic variance is non-positive, which also reports "
            "rg as 0")
    elif any(v <= lo * (1.0 + 1e-6) for v in raw_h2):
        # Strictly positive but under the caller's floor: the reported h2 is a
        # clamped value. rg is computed from the raw quadratics, so it stands.
        reasons.append(
            "h2 fell to its lower bound %g, so the reported h2 is clamped "
            "(rg is unaffected)" % lo)
    if p > _DIAGNOSTIC_MAX_CAUSAL_FRACTION:
        reasons.append("the fitted causal fraction is %.2f" % p)
    if reasons:
        warnings.warn(
            "Implausible bivariate fit: " + " and ".join(reasons) + ". This "
            "usually means the LD reference is too small or too weakly "
            "regularised for its block size, which distorts h2 and the causal "
            "fraction. Besides being statistically suspect, it makes the "
            "sampler markedly slower, because the per-variant LD row update "
            "then fires for almost every variant. Consider a larger LD "
            "reference panel, smaller blocks, or shrinking the LD blocks "
            "(ldpred3.shrink_ld_blocks).",
            RuntimeWarning,
            stacklevel=3,
        )


@dataclass
class _DecorrelatedAccumulator:
    """O(m) sufficient statistics for cross-sweep genetic quadratics."""

    sum1: np.ndarray
    sum2: np.ndarray
    diagonal: np.ndarray
    count: int = 0


def _decorrelated_accumulator(enabled, m):
    """Allocate only the two effect sums needed by decorrelated ``rg``."""
    if not enabled:
        return None
    return _DecorrelatedAccumulator(
        np.zeros(m, dtype=np.float64),
        np.zeros(m, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )


def _accumulate_decorrelated(accumulator, beta1, beta2, quadratics):
    """Add one effect pair and its already-computed LD quadratics."""
    accumulator.sum1 += beta1
    accumulator.sum2 += beta2
    accumulator.diagonal += quadratics
    accumulator.count += 1


def _decorrelated_cov(fblocks, accumulator):
    """Genetic (co)variances from online cross-sweep sufficient statistics.

    For each quadratic, the ordered-pair sum is the quadratic of the sample sum
    minus the sum of same-sweep quadratics. Thus two length-m float64 sums and
    three scalar accumulators exactly replace the former two
    ``(n_saved, m)`` effect traces. Reusing the sweep's same-sample quadratics
    also avoids any extra LD multiplication while sampling. Returns ``None``
    with fewer than two samples.
    """
    n = accumulator.count
    if n < 2:
        return None
    sums = np.vstack([accumulator.sum1, accumulator.sum2])
    RS1, RS2 = _apply_R_rows(fblocks, sums)
    all11 = float(sums[0] @ RS1)
    all12 = float(sums[0] @ RS2)
    all22 = float(sums[1] @ RS2)
    d11, d12, d22 = accumulator.diagonal
    npairs = n * (n - 1)
    return (all12 - d12) / npairs, (all11 - d11) / npairs, (all22 - d22) / npairs


@contextmanager
def _pinned_numba_threads(ncores):
    """Temporarily pin Numba's caller-local thread mask.

    ``get_num_threads`` is called even at ``ncores == 1``, where there is no
    mask to pin, because it also forces Numba's threading layer to load. The
    low-rank sweep references ``_get_thread_id``, and loading that kernel from
    the on-disk cache before the threading layer exists segfaults the
    interpreter (reproduced on numba 0.66: a warm cache plus a serial-only run
    exits 139). Touching the layer here costs one call per fit and keeps the
    serial kernel cacheable.
    """
    if not HAVE_NUMBA:
        yield
        return
    from numba import get_num_threads, set_num_threads

    previous = get_num_threads()
    if not (ncores and int(ncores) > 1):
        yield
        return
    _set_threads(ncores)
    try:
        yield
    finally:
        set_num_threads(previous)


def _bivar_const(nn1, nn2, s1, s2, s12, cross_corr):
    """Per-sweep scalars that don't depend on the residual ``(d1, d2)``.

    With a shared (scalar) N these are identical for every SNP in a sweep, so
    they are hoisted out of the per-SNP loop (see :func:`_bivar_one_sweep`). The
    per-variant-N path calls this once per SNP with that SNP's ``nn1``/``nn2``,
    giving bit-identical results to the inlined computation. Returns the noise
    covariance ``E`` / its inverse / state determinants (+ logs), the two 1D
    posterior variances and the both-state posterior covariance ``V`` and its
    Cholesky ``(L11, L21, L22)``.
    """
    E11 = 1.0 / nn1
    E22 = 1.0 / nn2
    E12 = cross_corr / np.sqrt(nn1 * nn2)
    det0 = E11 * E22 - E12 * E12
    Ei11 = E22 / det0
    Ei22 = E11 / det0
    Ei12 = -E12 / det0
    ldet0 = np.log(det0)
    a11 = E11 + s1
    det1 = a11 * E22 - E12 * E12
    ldet1 = np.log(det1)
    a22 = E22 + s2
    det2 = E11 * a22 - E12 * E12
    ldet2 = np.log(det2)
    b11 = E11 + s1
    b22 = E22 + s2
    b12 = E12 + s12
    det3 = b11 * b22 - b12 * b12
    ldet3 = np.log(det3)
    prec1 = Ei11 + 1.0 / s1
    sv1 = np.sqrt(1.0 / prec1)
    prec2 = Ei22 + 1.0 / s2
    sv2 = np.sqrt(1.0 / prec2)
    dS = s1 * s2 - s12 * s12
    Si11 = s2 / dS
    Si22 = s1 / dS
    Si12 = -s12 / dS
    P11 = Ei11 + Si11
    P12 = Ei12 + Si12
    P22 = Ei22 + Si22
    dP = P11 * P22 - P12 * P12
    V11 = P22 / dP
    V22 = P11 / dP
    V12 = -P12 / dP
    L11 = np.sqrt(V11)
    L21 = V12 / L11
    t = V22 - L21 * L21
    L22 = np.sqrt(t) if t > 0.0 else 0.0
    return (E11, E22, E12, det0, ldet0, a11, det1, ldet1, a22, det2, ldet2,
            b11, b22, b12, det3, ldet3, Ei11, Ei22, Ei12, prec1, sv1, prec2, sv2,
            V11, V22, V12, L11, L21, L22)


_bivar_const = _jit(_bivar_const)


def _bivar_one_sweep(corr, bh1, bh2, n1, n2, curr1, curr2, rb1, rb2,
                     rbsum1, rbsum2, unif, z1, z2,
                     lpi00, lpi10, lpi01, lpi11, s1, s2, s12, cross_corr,
                     scale, n_const, resync):
    """One Gibbs sweep of the 4-state model over a block; mutates in place.

    ``corr`` may be dense ``float32`` (``scale == 1.0``) or **int8**-quantised
    (``scale == 1/127``): each LD entry is read as ``corr[i, j] * scale``, so the
    int8 form keeps the block at a quarter of the memory and is dequantised on the
    fly in the (bandwidth-bound) inner loop -- the same trick as ldpred3's dense
    kernels. The unit diagonal quantises exactly (``127/127 == 1``), which the
    residual update ``d = bh - R@beta + beta`` relies on.

    States: 0 = null, 1 = trait-1 only, 2 = trait-2 only, 3 = both. Returns
    ``(c10, c01, c11, sum1sq, sum2sq, sum12, gv11, gv12, gv22)``: per-state counts
    and effect (co)moments for the hyper-parameter update, and the
    (co)heritability quadratics ``beta_t' R beta_u``. ``rbsum1/2`` accumulate the
    Rao-Blackwellised effects ``sum_state P(state) E[beta | state]``.

    When ``n_const`` (shared scalar N) the residual-independent scalars are
    computed once via :func:`_bivar_const` instead of per SNP -- the four state
    determinants, their logs, the noise-covariance inverse, the Sigma inverse and
    the both-state posterior covariance + Cholesky are identical for every SNP in
    a sweep, so this drops four ``log``s and a dozen divisions/roots per SNP. The
    arithmetic is unchanged, so the output is bit-identical to the per-SNP path.
    """
    k = bh1.shape[0]
    if resync:                                   # rebuild R@beta to clear drift
        for i in range(k):
            rb1[i] = 0.0
            rb2[i] = 0.0
        for j in range(k):
            b1 = curr1[j]
            b2 = curr2[j]
            if b1 != 0.0 or b2 != 0.0:
                cj = corr[j]
                for i in range(k):
                    cji = cj[i] * scale
                    rb1[i] += cji * b1
                    rb2[i] += cji * b2

    # Prime the residual-independent scalars from the first variant. With a
    # shared scalar N that is the whole computation for the sweep; with
    # per-variant N it also primes the memo in the loop below, which
    # recomputes only when N actually *changes* rather than once per SNP.
    # Real summary statistics carry long runs of identical n_eff, and
    # _bivar_const is ~29 quantities including four logs. It is a pure
    # function of its arguments, so reusing a hit is bit-identical.
    (E11, E22, E12, det0, ldet0, a11, det1, ldet1, a22, det2, ldet2,
     b11, b22, b12, det3, ldet3, Ei11, Ei22, Ei12, prec1, sv1, prec2, sv2,
     V11, V22, V12, L11, L21, L22) = _bivar_const(
        n1[0], n2[0], s1, s2, s12, cross_corr)
    last_n1 = n1[0]
    last_n2 = n2[0]

    c10 = 0
    c01 = 0
    c11 = 0
    sum1sq = 0.0
    sum2sq = 0.0
    sum12 = 0.0
    for j in range(k):
        b1 = curr1[j]
        b2 = curr2[j]
        d1 = bh1[j] - rb1[j] + b1                 # residual marginal estimates
        d2 = bh2[j] - rb2[j] + b2
        if not n_const and (n1[j] != last_n1 or n2[j] != last_n2):
            (E11, E22, E12, det0, ldet0, a11, det1, ldet1, a22, det2, ldet2,
             b11, b22, b12, det3, ldet3, Ei11, Ei22, Ei12, prec1, sv1, prec2, sv2,
             V11, V22, V12, L11, L21, L22) = _bivar_const(
                 n1[j], n2[j], s1, s2, s12, cross_corr)
            last_n1 = n1[j]
            last_n2 = n2[j]

        # log N(d; 0, E + Slab_state) for each of the 4 states (drop 2*pi const).
        q0 = (E22 * d1 * d1 - 2.0 * E12 * d1 * d2 + E11 * d2 * d2) / det0
        w0 = lpi00 - 0.5 * ldet0 - 0.5 * q0
        q1 = (E22 * d1 * d1 - 2.0 * E12 * d1 * d2 + a11 * d2 * d2) / det1
        w1 = lpi10 - 0.5 * ldet1 - 0.5 * q1
        q2 = (a22 * d1 * d1 - 2.0 * E12 * d1 * d2 + E11 * d2 * d2) / det2
        w2 = lpi01 - 0.5 * ldet2 - 0.5 * q2
        q3 = (b22 * d1 * d1 - 2.0 * b12 * d1 * d2 + b11 * d2 * d2) / det3
        w3 = lpi11 - 0.5 * ldet3 - 0.5 * q3

        wmax = w0
        if w1 > wmax:
            wmax = w1
        if w2 > wmax:
            wmax = w2
        if w3 > wmax:
            wmax = w3
        e0 = np.exp(w0 - wmax)
        e1 = np.exp(w1 - wmax)
        e2 = np.exp(w2 - wmax)
        e3 = np.exp(w3 - wmax)
        tot = e0 + e1 + e2 + e3
        p0 = e0 / tot
        p1 = e1 / tot
        p2 = e2 / tot
        p3 = e3 / tot

        # posterior effect means under each non-null state.
        m1_1 = (Ei11 * d1 + Ei12 * d2) / prec1    # state 1 (trait-1 only)
        m2_2 = (Ei22 * d2 + Ei12 * d1) / prec2    # state 2 (trait-2 only)
        g1 = Ei11 * d1 + Ei12 * d2                # state 3 (both)
        g2 = Ei12 * d1 + Ei22 * d2
        m1_3 = V11 * g1 + V12 * g2
        m2_3 = V12 * g1 + V22 * g2

        # Rao-Blackwell estimate: E[beta_t] = sum_state P(state) E[beta_t|state].
        rbsum1[j] += p1 * m1_1 + p3 * m1_3
        rbsum2[j] += p2 * m2_2 + p3 * m2_3

        # sample a state from (p0, p1, p2, p3).
        u = unif[j]
        if u < p0:
            new1 = 0.0
            new2 = 0.0
        elif u < p0 + p1:
            new1 = m1_1 + sv1 * z1[j]
            new2 = 0.0
            c10 += 1
            sum1sq += new1 * new1
        elif u < p0 + p1 + p2:
            new1 = 0.0
            new2 = m2_2 + sv2 * z2[j]
            c01 += 1
            sum2sq += new2 * new2
        else:
            new1 = m1_3 + L11 * z1[j]
            new2 = m2_3 + L21 * z1[j] + L22 * z2[j]
            c11 += 1
            sum1sq += new1 * new1
            sum2sq += new2 * new2
            sum12 += new1 * new2

        dlt1 = new1 - b1
        dlt2 = new2 - b2
        if dlt1 != 0.0 or dlt2 != 0.0:
            cj = corr[j]
            for i in range(k):
                cij = cj[i] * scale
                rb1[i] += cij * dlt1
                rb2[i] += cij * dlt2
            curr1[j] = new1
            curr2[j] = new2

    gv11 = 0.0
    gv12 = 0.0
    gv22 = 0.0
    for i in range(k):
        gv11 += curr1[i] * rb1[i]
        gv12 += curr1[i] * rb2[i]
        gv22 += curr2[i] * rb2[i]
    return c10, c01, c11, sum1sq, sum2sq, sum12, gv11, gv12, gv22


_bivar_one_sweep_jit = _jit_nogil(_bivar_one_sweep)


def _bivar_one_sweep_lowrank(
        U, factor_scale, residual, bh1, bh2, n1, n2, curr1, curr2, proj1, proj2,
        rb1, rb2, rbsum1, rbsum2, unif, z1, z2,
        lpi00, lpi10, lpi01, lpi11, s1, s2, s12, cross_corr,
        n_const, resync, write_rb):
    """Sweep over ``R = W W.T + diag(residual)`` without materialising it.

    ``W = factor_scale * U`` and ``proj1/2 = W.T @ beta1/2``. Current ldpred3
    factors preserve this global scale and carry the missing unit-diagonal mass
    in ``residual``. A SNP update costs O(rank). ``rb1/2`` are written only when
    the caller's noise-inflation update needs the final ``R @ beta`` vectors.
    """
    k = bh1.shape[0]
    rank = U.shape[1]
    if resync:                                   # rebuild W.T@beta to clear drift
        for c in range(rank):
            proj1[c] = 0.0
            proj2[c] = 0.0
        for j in range(k):
            b1 = curr1[j]
            b2 = curr2[j]
            if b1 != 0.0 or b2 != 0.0:
                fb1 = factor_scale * b1
                fb2 = factor_scale * b2
                for c in range(rank):
                    ujc = U[j, c]
                    proj1[c] += ujc * fb1
                    proj2[c] += ujc * fb2

    # Prime the residual-independent scalars from the first variant. With a
    # shared scalar N that is the whole computation for the sweep; with
    # per-variant N it also primes the memo in the loop below, which
    # recomputes only when N actually *changes* rather than once per SNP.
    # Real summary statistics carry long runs of identical n_eff, and
    # _bivar_const is ~29 quantities including four logs. It is a pure
    # function of its arguments, so reusing a hit is bit-identical.
    (E11, E22, E12, det0, ldet0, a11, det1, ldet1, a22, det2, ldet2,
     b11, b22, b12, det3, ldet3, Ei11, Ei22, Ei12, prec1, sv1, prec2, sv2,
     V11, V22, V12, L11, L21, L22) = _bivar_const(
        n1[0], n2[0], s1, s2, s12, cross_corr)
    last_n1 = n1[0]
    last_n2 = n2[0]

    c10 = 0
    c01 = 0
    c11 = 0
    sum1sq = 0.0
    sum2sq = 0.0
    sum12 = 0.0
    for j in range(k):
        b1 = curr1[j]
        b2 = curr2[j]
        rbj1 = 0.0
        rbj2 = 0.0
        for c in range(rank):
            ujc = U[j, c]
            rbj1 += ujc * proj1[c]
            rbj2 += ujc * proj2[c]
        rbj1 *= factor_scale
        rbj2 *= factor_scale
        rbj1 += residual[j] * b1
        rbj2 += residual[j] * b2
        d1 = bh1[j] - rbj1 + b1                 # diag(R) == 1
        d2 = bh2[j] - rbj2 + b2
        if not n_const and (n1[j] != last_n1 or n2[j] != last_n2):
            (E11, E22, E12, det0, ldet0, a11, det1, ldet1, a22, det2, ldet2,
             b11, b22, b12, det3, ldet3, Ei11, Ei22, Ei12, prec1, sv1, prec2, sv2,
             V11, V22, V12, L11, L21, L22) = _bivar_const(
                 n1[j], n2[j], s1, s2, s12, cross_corr)
            last_n1 = n1[j]
            last_n2 = n2[j]

        # log N(d; 0, E + Slab_state) for each of the 4 states (drop 2*pi const).
        q0 = (E22 * d1 * d1 - 2.0 * E12 * d1 * d2 + E11 * d2 * d2) / det0
        w0 = lpi00 - 0.5 * ldet0 - 0.5 * q0
        q1 = (E22 * d1 * d1 - 2.0 * E12 * d1 * d2 + a11 * d2 * d2) / det1
        w1 = lpi10 - 0.5 * ldet1 - 0.5 * q1
        q2 = (a22 * d1 * d1 - 2.0 * E12 * d1 * d2 + E11 * d2 * d2) / det2
        w2 = lpi01 - 0.5 * ldet2 - 0.5 * q2
        q3 = (b22 * d1 * d1 - 2.0 * b12 * d1 * d2 + b11 * d2 * d2) / det3
        w3 = lpi11 - 0.5 * ldet3 - 0.5 * q3

        wmax = w0
        if w1 > wmax:
            wmax = w1
        if w2 > wmax:
            wmax = w2
        if w3 > wmax:
            wmax = w3
        e0 = np.exp(w0 - wmax)
        e1 = np.exp(w1 - wmax)
        e2 = np.exp(w2 - wmax)
        e3 = np.exp(w3 - wmax)
        tot = e0 + e1 + e2 + e3
        p0 = e0 / tot
        p1 = e1 / tot
        p2 = e2 / tot
        p3 = e3 / tot

        # posterior effect means under each non-null state.
        m1_1 = (Ei11 * d1 + Ei12 * d2) / prec1
        m2_2 = (Ei22 * d2 + Ei12 * d1) / prec2
        g1 = Ei11 * d1 + Ei12 * d2
        g2 = Ei12 * d1 + Ei22 * d2
        m1_3 = V11 * g1 + V12 * g2
        m2_3 = V12 * g1 + V22 * g2

        rbsum1[j] += p1 * m1_1 + p3 * m1_3
        rbsum2[j] += p2 * m2_2 + p3 * m2_3

        u = unif[j]
        if u < p0:
            new1 = 0.0
            new2 = 0.0
        elif u < p0 + p1:
            new1 = m1_1 + sv1 * z1[j]
            new2 = 0.0
            c10 += 1
            sum1sq += new1 * new1
        elif u < p0 + p1 + p2:
            new1 = 0.0
            new2 = m2_2 + sv2 * z2[j]
            c01 += 1
            sum2sq += new2 * new2
        else:
            new1 = m1_3 + L11 * z1[j]
            new2 = m2_3 + L21 * z1[j] + L22 * z2[j]
            c11 += 1
            sum1sq += new1 * new1
            sum2sq += new2 * new2
            sum12 += new1 * new2

        dlt1 = new1 - b1
        dlt2 = new2 - b2
        if dlt1 != 0.0 or dlt2 != 0.0:
            fd1 = factor_scale * dlt1
            fd2 = factor_scale * dlt2
            for c in range(rank):
                ujc = U[j, c]
                proj1[c] += ujc * fd1
                proj2[c] += ujc * fd2
            curr1[j] = new1
            curr2[j] = new2

    if write_rb:
        for j in range(k):
            r1j = 0.0
            r2j = 0.0
            for c in range(rank):
                ujc = U[j, c]
                r1j += ujc * proj1[c]
                r2j += ujc * proj2[c]
            rb1[j] = factor_scale * r1j + residual[j] * curr1[j]
            rb2[j] = factor_scale * r2j + residual[j] * curr2[j]

    gv11 = 0.0
    gv12 = 0.0
    gv22 = 0.0
    for c in range(rank):
        gv11 += proj1[c] * proj1[c]
        gv12 += proj1[c] * proj2[c]
        gv22 += proj2[c] * proj2[c]
    for j in range(k):
        gv11 += residual[j] * curr1[j] * curr1[j]
        gv12 += residual[j] * curr1[j] * curr2[j]
        gv22 += residual[j] * curr2[j] * curr2[j]
    return c10, c01, c11, sum1sq, sum2sq, sum12, gv11, gv12, gv22


# fastmath here and NOT on the dense kernel, mirroring ldpred3's scoping
# (_kernels.py:1277). The O(rank) projection dots are ~90% of a low-rank
# sweep and are add-latency-bound, so letting LLVM reassociate and vectorise
# the reduction measured 1.76x end-to-end on an all-LR8 fit (1.26x/1.38x/
# 1.99x at rank 32/64/170). The dense kernel measured only 1.12x -- its
# guarded row update fires on ~6% of visits, so the sweep is dominated by the
# four exp() calls rather than by anything reassociable -- and is left plain.
# fastmath also asserts no NaN/Inf: the factor and residual are validated
# finite by LowRankLD, and _check_fit_is_finite catches a diverged chain at
# the end of the fit. Results move ~1e-16 relative.
_bivar_one_sweep_lowrank_jit = _jit_fastmath_nogil(_bivar_one_sweep_lowrank)


def _bivar_dense_sweep_all(
        blocks, starts, sizes, bh1, bh2, n1, n2, curr1, curr2, rb1, rb2,
        rbsum1, rbsum2, unif, z1, z2,
        lpi00, lpi10, lpi01, lpi11, s1, s2, s12, cross_corr,
        scale, n_const, resync, counts, stats):
    """Sweep homogeneous independent dense blocks under block-level prange."""
    for bb in prange(len(blocks)):
        b = np.int64(bb)
        start = starts[b]
        stop = start + sizes[b]
        sl = slice(start, stop)
        (a10, a01, a11, s1sq, s2sq, s12s,
         g11, g12, g22) = _bivar_one_sweep_jit(
            blocks[b], bh1[sl], bh2[sl], n1[sl], n2[sl], curr1[sl],
            curr2[sl], rb1[sl], rb2[sl], rbsum1[sl], rbsum2[sl],
            unif[sl], z1[sl], z2[sl], lpi00, lpi10, lpi01, lpi11,
            s1, s2, s12, cross_corr, scale, n_const, resync)
        counts[b, 0] = a10
        counts[b, 1] = a01
        counts[b, 2] = a11
        stats[b, 0] = s1sq
        stats[b, 1] = s2sq
        stats[b, 2] = s12s
        stats[b, 3] = g11
        stats[b, 4] = g12
        stats[b, 5] = g22


_bivar_dense_sweep_all_par_jit = _jit_parallel_uncached(_bivar_dense_sweep_all)
_bivar_dense_sweep_all_jit = _jit_nogil(_bivar_dense_sweep_all)


def _dequantise_lr8_factor(U, out):
    """Widen an int8 block factor into a float32 scratch buffer, exactly.

    Every int8 value is representable in float32, and ``float32 * float64``
    promotes exactly as ``int8 * float64`` does, so the arithmetic the sweep
    performs is unchanged element for element. ``factor_scale`` is deliberately
    *not* folded in: that would round every element once here, where keeping it
    a kernel argument costs one scalar multiply per variant instead.
    """
    flat_in = U.ravel()
    flat_out = out.ravel()
    for i in range(flat_in.shape[0]):
        flat_out[i] = flat_in[i]


_dequantise_lr8_factor_jit = _jit_nogil(_dequantise_lr8_factor)

#: Rank below which widening is not worth its own O(k x rank) streaming pass.
#: Measured for *this* sweep rather than inherited from ldpred3, and the answer
#: differs: ldpred3 gates at 64 because rank 32 measured a small loss for its
#: kernel, whereas here widening won at every rank tested (k=500, 20 blocks,
#: serial, off/on ms per sweep): rank 16 0.611/0.572 = 1.07x, 32 0.725/0.623 =
#: 1.16x, 64 0.959/0.812 = 1.18x, 128 1.393/1.065 = 1.31x, 256 2.250/1.577 =
#: 1.43x. The gate sits at 32 -- the smallest rank whose win is clearly outside
#: the noise -- rather than at 16, where the margin is thin and a sweep that
#: small costs little either way.
_LR8_DEQUANTISE_MIN_RANK = 32


def _lr8_dequant_scratch(payloads, sizes, ncores):
    """Per-thread float32 scratch for widening int8 low-rank factors.

    Returns ``(scratch, stride, min_rank)``. ``min_rank`` is 0 -- which disables
    the branch inside the kernel -- unless Numba is present and some block in
    this bucket is an int8 factor of at least ``_LR8_DEQUANTISE_MIN_RANK``. One
    stride per thread, not per block: the widened factor lives only for the
    duration of one block's sweep, so the cost is the largest block times the
    thread count rather than the genome.
    """
    if not HAVE_NUMBA:
        return np.empty(0, dtype=np.float32), 0, 0
    stride = 0
    for i in range(len(payloads)):
        U = payloads[i]
        if np.asarray(U).dtype == np.int8 and U.shape[1] >= _LR8_DEQUANTISE_MIN_RANK:
            stride = max(stride, int(sizes[i]) * int(U.shape[1]))
    if stride == 0:
        return np.empty(0, dtype=np.float32), 0, 0
    from numba import get_num_threads
    threads = get_num_threads() if ncores > 1 else 1
    return (np.empty(threads * stride, dtype=np.float32), stride,
            _LR8_DEQUANTISE_MIN_RANK)


def _bivar_lowrank_sweep_all(
        factors, factor_scales, residuals, proj1s, proj2s, starts, sizes,
        bh1, bh2, n1, n2, curr1, curr2, rb1, rb2, rbsum1, rbsum2,
        unif, z1, z2, lpi00, lpi10, lpi01, lpi11,
        s1, s2, s12, cross_corr, n_const, resync, write_rb, counts, stats,
        dequant_scratch, dequant_stride, dequant_min_rank):
    """Sweep homogeneous independent low-rank blocks under block-level prange.

    A qualifying int8 factor is widened into this thread's stride of
    ``dequant_scratch`` once per sweep and swept through the kernel's float32
    specialisation. The scratch is one stride per *thread*, not per block, so it
    stays O(k x rank) and int8 remains the storage format. ``dequant_min_rank``
    of 0 disables the branch entirely, and the call is then the one it always
    was.
    """
    for bb in prange(len(factors)):
        b = np.int64(bb)
        start = starts[b]
        stop = start + sizes[b]
        sl = slice(start, stop)
        rank = factors[b].shape[1]
        if 0 < dequant_min_rank <= rank:
            # Block-level branch: the gate is loop-invariant, so the widening
            # is paid once per block per sweep and amortised over the block's
            # k x rank projection dots -- of which the bivariate sweep runs two,
            # one per trait, off each loaded element.
            rows = factors[b].shape[0]
            base = _get_thread_id() * dequant_stride
            widened = dequant_scratch[base:base + rows * rank].reshape(
                rows, rank)
            _dequantise_lr8_factor_jit(factors[b], widened)
            (a10, a01, a11, s1sq, s2sq, s12s,
             g11, g12, g22) = _bivar_one_sweep_lowrank_jit(
                widened, factor_scales[b], residuals[b], bh1[sl], bh2[sl],
                n1[sl], n2[sl], curr1[sl], curr2[sl], proj1s[b], proj2s[b],
                rb1[sl], rb2[sl], rbsum1[sl], rbsum2[sl], unif[sl], z1[sl],
                z2[sl], lpi00, lpi10, lpi01, lpi11, s1, s2, s12,
                cross_corr, n_const, resync, write_rb)
        else:
            (a10, a01, a11, s1sq, s2sq, s12s,
             g11, g12, g22) = _bivar_one_sweep_lowrank_jit(
                factors[b], factor_scales[b], residuals[b], bh1[sl], bh2[sl],
                n1[sl], n2[sl], curr1[sl], curr2[sl], proj1s[b], proj2s[b],
                rb1[sl], rb2[sl], rbsum1[sl], rbsum2[sl], unif[sl], z1[sl],
                z2[sl], lpi00, lpi10, lpi01, lpi11, s1, s2, s12,
                cross_corr, n_const, resync, write_rb)
        counts[b, 0] = a10
        counts[b, 1] = a01
        counts[b, 2] = a11
        stats[b, 0] = s1sq
        stats[b, 1] = s2sq
        stats[b, 2] = s12s
        stats[b, 3] = g11
        stats[b, 4] = g12
        stats[b, 5] = g22


_bivar_lowrank_sweep_all_par_jit = _jit_parallel_uncached(_bivar_lowrank_sweep_all)
_bivar_lowrank_sweep_all_jit = _jit_nogil(_bivar_lowrank_sweep_all)


def _rg_from_quadratics(g12, g1, g2):
    """Clipped genetic correlation from LD-aware quadratic forms.

    Returns 0.0 when either variance is non-positive (possible on non-PD
    int8-quantised blocks) rather than slamming ``rg`` to +/-1 through the
    floor. The single source for the sampled-quadratic ``rg`` ratio used by
    the driver and the multi-chain pooling.
    """
    if g1 <= 0.0 or g2 <= 0.0:
        return 0.0
    return float(min(max(g12 / np.sqrt(g1 * g2), -1.0), 1.0))


def _rg_from_quadratics_array(g12, g1, g2):
    """Elementwise :func:`_rg_from_quadratics` for whole traces.

    Same convention, including the 0.0 for a non-positive variance, so a
    per-draw diagnostic trace cannot disagree with the scalar the fit reports.
    """
    g12, g1, g2 = np.asarray(g12), np.asarray(g1), np.asarray(g2)
    # Mirror the scalar guard exactly, including its NaN behaviour: ``nan <= 0``
    # is False there, so a NaN variance falls through and yields NaN rather than
    # being silently reported as an rg of 0. Writing this as ``g1 > 0`` instead
    # would classify NaN as invalid and diverge from the scalar.
    valid = ~((g1 <= 0.0) | (g2 <= 0.0))
    # Evaluate only where the denominator is defined; np.sqrt of a non-positive
    # variance would otherwise warn and seed NaN into the split-Rhat inputs.
    out = np.zeros(np.broadcast(g12, g1, g2).shape, dtype=float)
    np.divide(g12, np.sqrt(g1 * g2, where=valid, out=np.ones_like(out)),
              out=out, where=valid)
    return np.clip(out, -1.0, 1.0)


@dataclass
class BivariateResult:
    """Output of :func:`ldpred3_auto_bivariate`.

    ``beta1_est`` / ``beta2_est`` are the posterior-mean (standardized) effects
    for the two traits, ``h2`` the pair of SNP heritabilities, ``rg`` the
    estimated genetic correlation, ``p`` the causal fraction, ``sigma`` the
    learned 2x2 effect covariance, and ``pi`` the four-state mixture
    ``(pi00, pi10, pi01, pi11)`` = neither / trait-1-only / trait-2-only / both
    causal. ``sigma``, ``pi``, and ``noise_scale`` are means over the retained
    stochastic hyperparameter iterates. ``genetic_samples`` retains raw
    ``(gvar1, gcov, gvar2)`` quadratics and ``noise_scale_samples`` retains the
    two noise scales at every post-burn-in sweep. ``retained_iterations`` is the
    number of rows retained in those traces; ``stopped_early`` reports whether
    adaptive stopping shortened a single-chain run. See :attr:`mixer` for the
    MiXeR-style polygenic-overlap summary.
    """

    beta1_est: np.ndarray
    beta2_est: np.ndarray
    h2: tuple
    rg: float
    p: float
    sigma: np.ndarray
    pi: Optional[np.ndarray] = None
    pi_samples: Optional[np.ndarray] = None       # (n_kept, 4) conditional mixture draws
    sigma_samples: Optional[np.ndarray] = None    # (n_kept, 3) damped covariance iterates
    noise_scale: Optional[tuple] = None           # learned (lambda1, lambda2); (1,1) if off
    genetic_samples: Optional[np.ndarray] = None  # (n_kept, 3) raw (gvar1, gcov, gvar2)
    noise_scale_samples: Optional[np.ndarray] = None  # (n_kept, 2); ones if inflation off
    retained_iterations: Optional[int] = None     # post-burn-in sweeps actually kept
    stopped_early: bool = False                   # True when adaptive stopping fired

    @property
    def mixer(self):
        """MiXeR-style polygenic-overlap parameters (Frei et al. 2019).

        Returns ``polygenicity``, ``n_causal``, ``n_shared``, ``frac_shared``,
        ``rho_beta`` and ``rg_from_overlap`` over the fitted variants. The ratios
        are usually more stable than absolute counts; counts can be inflated by
        LD-spreading and reference-panel mismatch. Use
        :meth:`mixer_iterate_summary` for empirical variability across retained
        hyperparameter iterates and :meth:`mixer_calibrated` to anchor counts on
        two univariate ldpred3 fits.
        """
        if self.pi is None:
            raise ValueError("pi not available on this result")
        pi00, pi10, pi01, pi11 = (float(x) for x in self.pi)
        return self._mixer_dict(len(self.beta1_est), pi10 + pi11, pi01 + pi11,
                                pi11, self._rho_beta())

    def _rho_beta(self):
        s1, s2 = self.sigma[0, 0], self.sigma[1, 1]
        return float(self.sigma[0, 1] / np.sqrt(max(s1 * s2, 1e-300)))

    @staticmethod
    def _mixer_dict(m, pi1, pi2, pi11, rho_beta):
        denom = np.sqrt(max(pi1 * pi2, 1e-300))
        return {
            "polygenicity": (pi1, pi2),
            "n_causal": (pi1 * m, pi2 * m),
            "n_shared": pi11 * m,
            "frac_shared": pi11 / max(min(pi1, pi2), 1e-300),
            "rho_beta": rho_beta,
            "rg_from_overlap": float(rho_beta * pi11 / denom),
        }

    def _mixer_iterate_summary(self, level, interval_key):
        if self.pi_samples is None or self.sigma_samples is None:
            raise ValueError("hyperparameter iterates not available on this result")
        level = _finite_control("level", level)
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        pi_samples = np.asarray(self.pi_samples, dtype=float)
        sigma_samples = np.asarray(self.sigma_samples, dtype=float)
        if (pi_samples.ndim != 2 or pi_samples.shape[1:] != (4,)
                or sigma_samples.ndim != 2 or sigma_samples.shape[1:] != (3,)
                or len(pi_samples) != len(sigma_samples)):
            raise ValueError(
                "pi_samples and sigma_samples must have matching shapes (n, 4) "
                "and (n, 3)")
        if len(pi_samples) == 0:
            raise ValueError("no post-burn-in hyperparameter iterates were retained")
        if not (np.all(np.isfinite(pi_samples))
                and np.all(np.isfinite(sigma_samples))):
            raise ValueError("hyperparameter iterates must contain only finite values")
        m = len(self.beta1_est)
        lo_q = (1.0 - level) / 2.0 * 100.0
        hi_q = (1.0 + level) / 2.0 * 100.0
        cols = {"n1": [], "n2": [], "n_shared": [], "frac_shared": [],
                "rho_beta": [], "rg_from_overlap": []}
        for (_p00, p10, p01, p11), (s1, s2, s12) in zip(pi_samples,
                                                        sigma_samples):
            rho_beta = float(s12 / np.sqrt(max(s1 * s2, 1e-300)))
            d = self._mixer_dict(m, p10 + p11, p01 + p11, p11, rho_beta)
            cols["n1"].append(d["n_causal"][0])
            cols["n2"].append(d["n_causal"][1])
            cols["n_shared"].append(d["n_shared"])
            cols["frac_shared"].append(d["frac_shared"])
            cols["rho_beta"].append(d["rho_beta"])
            cols["rg_from_overlap"].append(d["rg_from_overlap"])

        def summ(a):
            a = np.asarray(a, dtype=float)
            return {"mean": float(a.mean()), "sd": float(a.std()),
                    interval_key: (float(np.percentile(a, lo_q)),
                                   float(np.percentile(a, hi_q)))}
        n1, n2 = summ(cols["n1"]), summ(cols["n2"])
        return {
            "n_causal": (n1, n2),
            "polygenicity": ({**n1, "mean": n1["mean"] / m,
                              "sd": n1["sd"] / m,
                              interval_key: (n1[interval_key][0] / m,
                                             n1[interval_key][1] / m)},
                             {**n2, "mean": n2["mean"] / m,
                              "sd": n2["sd"] / m,
                              interval_key: (n2[interval_key][0] / m,
                                             n2[interval_key][1] / m)}),
            "n_shared": summ(cols["n_shared"]),
            "frac_shared": summ(cols["frac_shared"]),
            "rho_beta": summ(cols["rho_beta"]),
            "rg_from_overlap": summ(cols["rg_from_overlap"]),
            "level": level,
        }

    def mixer_iterate_summary(self, level=0.95):
        """Empirical summaries of MiXeR quantities across retained iterates.

        ``pi`` is sampled from its conditional Dirichlet distribution, whereas
        ``Sigma`` is a deterministic damped moment update driven by stochastic
        state/effect draws. Consequently, the returned central ``interval`` is
        an empirical range of the hybrid algorithm's retained iterates, **not**
        a Bayesian credible interval and not a frequentist confidence interval.
        It also does not represent LD-reference-mismatch uncertainty.
        """
        return self._mixer_iterate_summary(level, "interval")

    def mixer_calibrated(self, infer1, infer2):
        """:attr:`mixer` with counts anchored on two univariate fits.

        ``infer1`` and ``infer2`` may be ldpred3 ``InferResult`` objects or
        floats. Their ``p_est`` values replace the joint per-trait polygenicities;
        the joint shared fraction and ``rho_beta`` are kept.
        """
        if self.pi is None:
            raise ValueError("pi not available on this result")
        p1 = _finite_control("infer1 polygenicity", getattr(infer1, "p_est", infer1))
        p2 = _finite_control("infer2 polygenicity", getattr(infer2, "p_est", infer2))
        if not 0.0 <= p1 <= 1.0 or not 0.0 <= p2 <= 1.0:
            raise ValueError("calibrated polygenicities must be in [0, 1]")
        pi10, pi11, pi01 = float(self.pi[1]), float(self.pi[3]), float(self.pi[2])
        pj1, pj2 = pi10 + pi11, pi01 + pi11
        frac_shared = pi11 / max(min(pj1, pj2), 1e-300)   # reliable joint ratio
        pi11_cal = frac_shared * min(p1, p2)              # shared count, calib. scale
        return self._mixer_dict(len(self.beta1_est), p1, p2, pi11_cal,
                                self._rho_beta())

    def __repr__(self):
        return (f"BivariateResult(h2=({self.h2[0]:.3f}, {self.h2[1]:.3f}), "
                f"rg={self.rg:+.3f}, p={self.p:.4g}, "
                f"n_variants={len(self.beta1_est)})")


def ldpred3_auto_bivariate_blocks(blocks, beta_hat1, beta_hat2, n_eff1, n_eff2, *,
                                  ld_int8=False,
                                  h2_init=0.1, p_init=0.02, rg_init=0.0,
                                  pi_init=None, sigma_prior_scale=None,
                                  cross_corr=0.0, burn_in=200, num_iter=200,
                                  h2_bounds=(1e-4, 1.0), h2_cap=None,
                                  iw_df=10.0, rg_decorrelated=False,
                                  noise_inflation=False, ni_damp=0.1,
                                  pi_prior=1.0, sample_every=5, ncores=1,
                                  tol=0.0, check_every=50, seed=None):
    """Genome-wide bivariate LDpred3-auto over dense or low-rank LD blocks.

    ``blocks`` is ``[(R, idx), ...]`` with contiguous ``idx`` arrays partitioning
    ``0..m-1``. The two traits share the same LD. Effects are updated block by
    block while ``pi`` and ``Sigma`` are pooled globally, so the genome-wide LD is
    never materialised. Blocks may be dense or ldpred3 ``LowRankLD`` objects;
    mixed representations are supported. Low-rank blocks retain their compact
    factor (including LR8 int8 factors) and diagonal residual.

    Dense blocks are consumed in the representation they arrive in: supplied
    int8 blocks stay int8 and float blocks stay float32, both without a copy.
    Quantise when the LD is *built*, with
    ``ldpred3.compute_ld_blocks(quantize=True)``, rather than in the fit --
    quantising here allocates a second genome-scale payload while the caller's
    panel is still alive (78.4 MB of peak against 13.1 MB at m=100,000, k=500).
    ``ld_int8=True`` and ``None`` are retained for that older behaviour.
    This option does not alter ``LowRankLD`` factors.

    Parameters
    ----------
    blocks : list of (ndarray or LowRankLD, ndarray)
        Per-block LD ``(R, idx)`` partitioning ``0..m-1``. Dense ``R`` may be
        float or int8-quantised; compact float and LR8 ``LowRankLD`` are also
        supported.
    beta_hat1, beta_hat2 : array_like (m,)
        Standardized marginal effects for the two traits (same variant order).
    n_eff1, n_eff2 : float or array_like
        Per-trait GWAS sample sizes.
    ld_int8 : bool or None, default False
        Dense-LD storage policy. ``False`` consumes every block in the
        representation it was given, with no copy -- the memory-cheapest option,
        and the one that keeps this call's LD identical to what
        :func:`~bipred.regional_rg` will evaluate. ``True`` quantises every float
        block and ``None`` quantises those of at most 1500 variants; both build a
        fresh int8 array per block inside the fit, so prefer quantising at LD
        build time (``ldpred3.compute_ld_blocks(quantize=True)``). Supplied int8
        blocks stay int8 under all three settings, and none of them alters
        ``LowRankLD`` factors.
    h2_init : float or pair
        Initial per-trait heritability. A scalar applies to both traits.
    p_init : float, default 0.02
        Initial union causal fraction, ``P(trait 1 or trait 2 causal)``. Used by
        the symmetric shorthand when ``pi_init`` is omitted.
    rg_init : float, default 0
        Initial genetic correlation.
    pi_init : length-4 array, optional
        Explicit initial ``(pi00, pi10, pi01, pi11)`` mixture. This exposes the
        overlap degree of freedom that ``p_init`` alone cannot determine. The
        slab covariance is calibrated so the supplied ``h2_init`` and
        ``rg_init`` are the implied genetic moments exactly.
    sigma_prior_scale : float or pair, optional
        Persistent diagonal shrinkage target for the per-causal effect
        covariance. A scalar applies to both traits. By default it equals the
        coherently calibrated initial slab variances; set it explicitly when
        varying starts across chains so the chains retain the same prior.
    cross_corr : float, default 0.0
        Cross-trait correlation of the sampling errors; sample overlap is one
        possible cause. Must lie in ``(-1, 1)``; 0 assumes uncorrelated errors.
    burn_in, num_iter : int
        Burn-in and sampling sweeps.
    h2_bounds : (float, float)
        Clamp range for the per-trait heritabilities.
    h2_cap : (float, float), optional
        Optional hard ceilings on implied per-trait heritability.
    iw_df : float, default 10
        Shrinkage strength on the effect covariance ``Sigma``. Larger values pull
        more strongly toward independent traits.
    rg_decorrelated : bool, default False
        **Sensitivity diagnostic only — do not use for production estimates.**
        Estimator based on effects sampled at different sweeps, kept for
        strongly asymmetric-power pairs. The committed 0.2.0 synthetic sweep
        measured the **default** estimator more accurate in both power regimes
        (RESULTS.md Table 4: 0.0084 vs 0.0110 symmetric, 0.0174 vs 0.0240
        asymmetric), and this option is incompatible with multichain pooling
        and adaptive stopping. Requires ``num_iter > sample_every``. If the
        cross-sweep quadratics are non-finite it raises; if a variance is
        non-positive (a degenerate, undefined decorrelated rg -- e.g. a sparse,
        weakly powered fit) it warns and reports ``rg`` as ``NaN`` rather than
        aborting the otherwise usable fit.
    noise_inflation : bool, default False
        Learn per-trait residual noise factors ``lambda_t >= 1`` and fit with
        effective sample size ``N_t / lambda_t``. Useful for finite reference-panel
        LD when absolute overlap counts are inflated by mismatch.
    ni_damp : float, default 0.1
        Damping for the per-sweep ``lambda`` update (only used with
        ``noise_inflation``); smaller is more stable, larger adapts faster.
    sample_every : int, default 5
        Thinning for the retained effect states used by the decorrelated ``rg``.
    pi_prior : float, default 1.0
        Symmetric Dirichlet concentration for the four-state mixture prior.
    ncores : int, default 1
        Prepared dense blocks are bucketed by dtype and dequantisation scale;
        low-rank blocks are bucketed by dtype and keep a scalar scale per block.
        Each bucket is swept by one fused call. At one core this is serial;
        larger values parallelise blocks within each bucket, while buckets
        remain sequential. SNP updates stay sequential within each block, and
        seeded results match ``ncores=1``. These are persistent threads, not
        subprocesses, but each sweep waits for every block before updating
        global parameters; imbalanced blocks can therefore limit speed-up.
    tol : float, default 0
        Optional stabilization threshold for the running posterior means and
        genetic correlation. A positive value may stop retained sampling early;
        this is a computational heuristic, not a convergence diagnostic. It is
        ignored when ``rg_decorrelated=True`` (which needs the full thinned
        schedule); passing both here emits a warning, whereas
        :func:`ldpred3_auto_bivariate_chains` rejects the pairing.
    check_every : int, default 50
        Retained sweeps between stabilization checks when ``tol > 0``.
    seed : int or None

    Returns
    -------
    BivariateResult
    """
    options = _validate_bivariate_options(
        ld_int8=ld_int8,
        h2_init=h2_init,
        rg_init=rg_init,
        cross_corr=cross_corr,
        burn_in=burn_in,
        num_iter=num_iter,
        h2_bounds=h2_bounds,
        h2_cap=h2_cap,
        iw_df=iw_df,
        rg_decorrelated=rg_decorrelated,
        noise_inflation=noise_inflation,
        ni_damp=ni_damp,
        pi_prior=pi_prior,
        sample_every=sample_every,
        ncores=ncores,
        tol=tol,
        check_every=check_every,
    )
    prepared = _prepare_bivariate_inputs(
        blocks, beta_hat1, beta_hat2, n_eff1, n_eff2, options
    )
    sigma_prior_scale = _validate_sigma_prior_scale(sigma_prior_scale)
    start = _prepare_bivariate_start(
        prepared.m,
        options,
        p_init=p_init,
        pi_init=pi_init,
        sigma_prior_scale=sigma_prior_scale,
        seed=seed,
    )
    return _ldpred3_auto_bivariate_prepared(prepared, options, start)


def _ldpred3_auto_bivariate_prepared(prepared, options, start):
    """Run one chain while preserving the caller's Numba thread mask."""
    with _pinned_numba_threads(options.ncores):
        return _ldpred3_auto_bivariate_prepared_inner(prepared, options, start)


def _ldpred3_auto_bivariate_prepared_inner(prepared, options, start):
    """Run one chain from canonical shared data and fresh mutable workspaces."""
    bh1 = prepared.beta_hat1
    bh2 = prepared.beta_hat2
    n1 = prepared.n_eff1
    n2 = prepared.n_eff2
    m = prepared.m
    n_const = prepared.n_const
    fblocks = _instantiate_chain_blocks(prepared)

    burn_in = options.burn_in
    num_iter = options.num_iter
    lo, hi = options.h2_bounds
    h2_cap = options.h2_cap
    iw_df = options.iw_df
    rg_decorrelated = options.rg_decorrelated
    noise_inflation = options.noise_inflation
    ni_damp = options.ni_damp
    pi_prior = options.pi_prior
    sample_every = options.sample_every
    ncores = options.ncores
    tol = options.tol
    check_every = options.check_every
    cross_corr = options.cross_corr

    # Thinned decorrelated-rg needs the full retained schedule, so the adaptive-
    # stopping gate below is disabled when rg_decorrelated=True. Warn rather than
    # silently ignore a positive tol; the multichain entry point rejects this
    # pairing outright, so this branch only ever fires on the single-chain path.
    if tol > 0.0 and rg_decorrelated:
        warnings.warn(
            "tol is ignored when rg_decorrelated=True: the thinned "
            "decorrelated-rg estimator requires the full retained schedule, so "
            "no early stopping is performed",
            RuntimeWarning,
            stacklevel=2,
        )

    # A typed.List needs one element type, so dense blocks are bucketed by dtype
    # and dequantisation scale; low-rank blocks are bucketed by dtype and carry
    # scalar factor scales separately. Each bucket gets one fused native call.
    # At ncores=1 the same source is compiled without parallel=True: prange
    # becomes range and avoids one Python->Numba crossing per block without
    # starting a parallel runtime.
    sweep_groups = []
    block_counts = block_stats = None
    if HAVE_NUMBA and fblocks:
        buckets = {}
        for pos, (kind, data, _start, _k, aux, _res, _p1, _p2) in enumerate(
                fblocks):
            key = (kind, np.asarray(data).dtype.str,
                   float(aux) if kind == _DENSE else 0.0)
            buckets.setdefault(key, []).append(pos)

        from numba.typed import List as NumbaList

        block_counts = np.empty((len(fblocks), 3), dtype=np.int64)
        block_stats = np.empty((len(fblocks), 6), dtype=np.float64)
        for (kind, _dtype, scale), positions in buckets.items():
            payloads = NumbaList()
            for pos in positions:
                payloads.append(fblocks[pos][1])
            group = {
                "kind": kind,
                "scale": scale,
                "payloads": payloads,
                "index": np.asarray(positions, dtype=np.int64),
                "starts": np.asarray(
                    [fblocks[p][2] for p in positions], dtype=np.int64
                ),
                "sizes": np.asarray(
                    [fblocks[p][3] for p in positions], dtype=np.int64
                ),
                "counts": np.empty((len(positions), 3), dtype=np.int64),
                "stats": np.empty((len(positions), 6), dtype=np.float64),
            }
            if kind == _LOWRANK:
                residual_list = NumbaList()
                proj1_list = NumbaList()
                proj2_list = NumbaList()
                factor_scales = np.empty(len(positions), dtype=np.float64)
                for i, pos in enumerate(positions):
                    _k2, _d, _s, _n, aux, residual, proj1, proj2 = fblocks[pos]
                    factor_scales[i] = aux
                    residual_list.append(residual)
                    proj1_list.append(proj1)
                    proj2_list.append(proj2)
                group["factor_scales"] = factor_scales
                group["residuals"] = residual_list
                group["proj1s"] = proj1_list
                group["proj2s"] = proj2_list
                # One scratch per bucket, allocated once per fit rather than
                # once per sweep; sized by this bucket's largest qualifying
                # block and by the thread count that will sweep it.
                (group["dequant_scratch"], group["dequant_stride"],
                 group["dequant_min_rank"]) = _lr8_dequant_scratch(
                     payloads, group["sizes"], ncores)
            sweep_groups.append(group)
    dense_sweep_all = (
        _bivar_dense_sweep_all_par_jit if ncores > 1
        else _bivar_dense_sweep_all_jit
    )
    lowrank_sweep_all = (
        _bivar_lowrank_sweep_all_par_jit if ncores > 1
        else _bivar_lowrank_sweep_all_jit
    )

    pi = start.pi.copy()
    s1, s2, s12 = start.s1, start.s2, start.s12
    psi1, psi2 = start.psi1, start.psi2

    # (Co)variance-component regularisation. The effect covariance Sigma is
    # updated each sweep by shrinking toward a weak inverse-Wishart prior (MTGSAM
    # / Sorensen-Gianola): per-trait slab scales (psi1, psi2) on the diagonal,
    # zero off-diagonal, with iw_df pseudo-counts. This replaces the old
    # scheme (a univariate-auto h2 ceiling + a hard 0.999 PD cap): the univariate
    # anchor under-estimates h2 on noisy dense LD -> shrinks the rg denominator ->
    # inflated rg, while the diagonal prior here keeps Sigma positive-definite and
    # the off-diagonal from riding the PD boundary. A caller may still pass
    # ``h2_cap`` to additionally clamp the implied per-trait h2 (expert override).
    nu0 = float(iw_df)

    rng = np.random.default_rng(start.seed)
    curr1 = np.zeros(m); curr2 = np.zeros(m)
    # Dense sweeps carry R@beta incrementally. Compact sweeps carry only their
    # rank-length projections and write R@beta solely for noise inflation, so an
    # all-low-rank default fit needs no genome-length residual buffers.
    needs_rb = noise_inflation or any(b[0] == _DENSE for b in fblocks)
    rb1 = np.zeros(m) if needs_rb else np.empty(0)
    rb2 = np.zeros(m) if needs_rb else np.empty(0)
    avg1 = np.zeros(m); avg2 = np.zeros(m)
    count = 0
    gv_acc = np.zeros(3)
    # Adaptive stopping keeps a snapshot of the previous posterior mean. Only
    # allocate it when the feature is on, so the default path's memory is
    # unchanged.
    prev1 = prev2 = None
    prev_rg = None
    if tol > 0.0:
        prev1 = np.zeros(m); prev2 = np.zeros(m)
    # Retained post-burn-in hyperparameter iterates. pi is a conditional
    # Dirichlet draw; Sigma is a damped moment update driven by stochastic effect
    # and state draws, not a conditional posterior draw.
    pi_samples = np.zeros((num_iter, 4))
    sig_samples = np.zeros((num_iter, 3))
    genetic_samples = np.zeros((num_iter, 3))
    noise_scale_samples = np.zeros((num_iter, 2))
    decorrelated = _decorrelated_accumulator(rg_decorrelated, m)

    # ``pi`` and the slab covariance were calibrated together above: their
    # implied marginal h2 and genetic rg equal the documented starting values.
    # Per-trait noise-inflation factors (LDSC-intercept analog); 1 = off.
    lam1 = lam2 = 1.0

    # Per-sweep working buffers are allocated once and refilled in place. The
    # RNG is drawn with ``out=``, so the stream (count and order of draws) is
    # exactly what fresh ``rng.random(m)`` / ``rng.standard_normal(m)`` calls
    # produced -- the results stay bit-identical, without churning five
    # length-m float64 arrays every sweep.
    unif = np.empty(m); z1 = np.empty(m); z2 = np.empty(m)
    rbs1 = np.zeros(m); rbs2 = np.zeros(m)
    if noise_inflation:
        n1e = np.empty(m); n2e = np.empty(m)
    else:                                     # no deflation: read n1/n2 directly
        n1e, n2e = n1, n2

    for it in range(burn_in + num_iter):
        resync = (it % 100 == 0)
        rng.random(out=unif)
        rng.standard_normal(out=z1)
        rng.standard_normal(out=z2)
        rbs1.fill(0.0); rbs2.fill(0.0)
        lpi = np.log(np.maximum(pi, 1e-300))
        c10 = c01 = c11 = 0
        S1 = S2 = S12 = 0.0
        gv11 = gv12 = gv22 = 0.0
        # Effective per-variant N deflated by the learned noise inflation. A scalar
        # lambda preserves the constant-N fast path (n_const unchanged).
        if noise_inflation:
            np.divide(n1, lam1, out=n1e)
            np.divide(n2, lam2, out=n2e)
        for group in sweep_groups:
            if group["kind"] == _DENSE:
                dense_sweep_all(
                    group["payloads"], group["starts"], group["sizes"],
                    bh1, bh2, n1e, n2e, curr1, curr2, rb1, rb2, rbs1, rbs2,
                    unif, z1, z2, float(lpi[0]), float(lpi[1]), float(lpi[2]),
                    float(lpi[3]), float(s1), float(s2), float(s12),
                    float(cross_corr), float(group["scale"]), n_const, resync,
                    group["counts"], group["stats"])
            else:
                lowrank_sweep_all(
                    group["payloads"], group["factor_scales"], group["residuals"],
                    group["proj1s"], group["proj2s"], group["starts"],
                    group["sizes"], bh1, bh2, n1e, n2e, curr1, curr2, rb1,
                    rb2, rbs1, rbs2, unif, z1, z2, float(lpi[0]),
                    float(lpi[1]), float(lpi[2]), float(lpi[3]), float(s1),
                    float(s2), float(s12), float(cross_corr), n_const, resync,
                    bool(noise_inflation), group["counts"], group["stats"],
                    group["dequant_scratch"], group["dequant_stride"],
                    group["dequant_min_rank"])
            # Scatter back to genome-block order so the reduction below is
            # unchanged whether one bucket or several were used.
            block_counts[group["index"]] = group["counts"]
            block_stats[group["index"]] = group["stats"]

        if sweep_groups:
            # Match the serial driver's exact block and floating reduction order.
            for b in range(len(fblocks)):
                c10 += int(block_counts[b, 0])
                c01 += int(block_counts[b, 1])
                c11 += int(block_counts[b, 2])
                S1 += float(block_stats[b, 0])
                S2 += float(block_stats[b, 1])
                S12 += float(block_stats[b, 2])
                gv11 += float(block_stats[b, 3])
                gv12 += float(block_stats[b, 4])
                gv22 += float(block_stats[b, 5])
        else:
            for kind, data, start, k, aux, residual, proj1, proj2 in fblocks:
                sl = slice(start, start + k)
                if kind == _LOWRANK:
                    result = _bivar_one_sweep_lowrank_jit(
                        data, aux, residual, bh1[sl], bh2[sl], n1e[sl], n2e[sl],
                        curr1[sl], curr2[sl], proj1, proj2, rb1[sl], rb2[sl],
                        rbs1[sl], rbs2[sl], unif[sl], z1[sl], z2[sl],
                        float(lpi[0]), float(lpi[1]), float(lpi[2]), float(lpi[3]),
                        float(s1), float(s2), float(s12), float(cross_corr),
                        n_const, resync, bool(noise_inflation))
                else:
                    result = _bivar_one_sweep_jit(
                        data, bh1[sl], bh2[sl], n1e[sl], n2e[sl],
                        curr1[sl], curr2[sl], rb1[sl], rb2[sl], rbs1[sl], rbs2[sl],
                        unif[sl], z1[sl], z2[sl], float(lpi[0]), float(lpi[1]),
                        float(lpi[2]), float(lpi[3]), float(s1), float(s2),
                        float(s12), float(cross_corr), float(aux), n_const, resync)
                a10, a01, a11, s1sq, s2sq, s12s, g11, g12, g22 = result
                c10 += a10; c01 += a01; c11 += a11
                S1 += s1sq; S2 += s2sq; S12 += s12s
                gv11 += g11; gv12 += g12; gv22 += g22

        if noise_inflation:
            # Update lambda_t from the residual mean-chi2. rb1/rb2 hold R@beta
            # after the sweep, so b_hat - R@beta is the residual; under matched LD
            # it is pure sampling noise (mean n*resid^2 ~ 1) and inflated otherwise.
            # r1/r2 are fresh temporaries, so the (n * r) * r product is formed
            # in place -- same association order as ``n1 * r1 * r1``, one fewer
            # length-m allocation each.
            r1 = bh1 - rb1; r2 = bh2 - rb2
            t1 = n1 * r1; t1 *= r1
            t2 = n2 * r2; t2 *= r2
            lh1 = max(float(np.mean(t1)), 1.0)
            lh2 = max(float(np.mean(t2)), 1.0)
            lam1 = (1.0 - ni_damp) * lam1 + ni_damp * lh1
            lam2 = (1.0 - ni_damp) * lam2 + ni_damp * lh2

        # --- global hyper-parameter updates ---
        c00 = m - c10 - c01 - c11
        pi = rng.dirichlet([pi_prior + c00, pi_prior + c10,
                            pi_prior + c01, pi_prior + c11])
        n1c = c10 + c11
        n2c = c01 + c11
        # Inverse-Wishart-style shrinkage of (s1, s2, s12) toward the weak
        # diagonal prior (psi1/psi2, nu0 pseudo-counts, zero prior covariance).
        # Marginal variances pool all trait-causal variants; the covariance uses
        # the both-causal pairs and is pulled toward 0 by the prior (no genetic
        # covariance a priori), which keeps s12 off the PD boundary. Damped for
        # cross-sweep stability.
        s1 = (1.0 - DAMP) * s1 + DAMP * (nu0 * psi1 + S1) / (nu0 + n1c)
        s2 = (1.0 - DAMP) * s2 + DAMP * (nu0 * psi2 + S2) / (nu0 + n2c)
        s12 = (1.0 - DAMP) * s12 + DAMP * (S12 / (nu0 + c11))
        s1 = max(s1, 1e-12)
        s2 = max(s2, 1e-12)
        if h2_cap is not None:                       # optional expert hard cap
            s1 = min(s1, h2_cap[0] / max(n1c, 1))
            s2 = min(s2, h2_cap[1] / max(n2c, 1))
        mab = 0.999 * np.sqrt(s1 * s2)               # PD safety (rarely binds)
        s12 = min(max(s12, -mab), mab)

        if it >= burn_in:
            avg1 += rbs1; avg2 += rbs2
            gv_acc += (gv11, gv12, gv22)
            pi_samples[count] = pi
            sig_samples[count] = (s1, s2, s12)
            genetic_samples[count] = (gv11, gv12, gv22)
            noise_scale_samples[count] = (lam1, lam2)
            if (rg_decorrelated and (it - burn_in) % sample_every == 0):
                _accumulate_decorrelated(
                    decorrelated, curr1, curr2, (gv11, gv12, gv22)
                )
            count += 1

            # Adaptive stopping. Only after two snapshots exist, so the first
            # check compares like with like rather than against zeros. Effect
            # samples for the decorrelated-rg estimator are thinned, so stopping
            # early there would shrink the pair count that estimator needs;
            # leave it to run the full schedule.
            if (tol > 0.0 and not rg_decorrelated
                    and count % check_every == 0 and count > check_every):
                g11c, g12c, g22c = gv_acc / count
                rg_now = _rg_from_quadratics(g12c, g11c, g22c)
                done, prev_rg = _bivar_converged(avg1, avg2, prev1, prev2,
                                                 count, rg_now, prev_rg, tol)
                if done:
                    break
            elif tol > 0.0 and count % check_every == 0:
                # Prime the snapshot without testing against it.
                g11c, g12c, g22c = gv_acc / count
                prev_rg = _rg_from_quadratics(g12c, g11c, g22c)
                prev1[:] = avg1 / count
                prev2[:] = avg2 / count

    # num_iter >= 1 is validated at the public boundary, so count cannot be 0.
    # With tol > 0 the loop may stop early, so count <= num_iter is the invariant.
    if not 0 < count <= num_iter:                 # defensive internal invariant
        raise RuntimeError("internal error: retained-iteration count mismatch")
    g11, g12, g22 = gv_acc / count
    _check_fit_is_finite((g11, g12, g22), avg1, avg2)
    h2_1 = min(max(g11, lo), hi)
    h2_2 = min(max(g22, lo), hi)
    # rg from effect samples with approximately independent noise (drawn at
    # different sweeps):
    # the decorrelated genetic covariance over the decorrelated predictor
    # variances. This avoids the same-sweep cross-noise that inflates the genetic
    # covariance and estimates a weak trait's covariance from its posterior mean
    # (which the sampled-quadratic ratio attenuates).
    if rg_decorrelated:
        cov = _decorrelated_cov(fblocks, decorrelated)
        if cov is None:  # defensive: the public validator guarantees >=2 samples
            raise RuntimeError(
                "internal error: decorrelated rg retained fewer than two effect "
                "samples"
            )
        num, v1, v2 = cov
        if not np.all(np.isfinite((num, v1, v2))):
            raise ValueError(
                "rg_decorrelated=True produced non-finite cross-sweep quadratic "
                "forms"
            )
        if v1 <= 0.0 or v2 <= 0.0:
            # A degenerate cross-sweep genetic-variance estimate (exactly 0 in
            # finite samples, or slightly negative) leaves the decorrelated rg
            # undefined -- e.g. a sparse, weakly powered fit whose retained
            # states share no causal support. That is not a broken run, so warn
            # and report rg as NaN rather than aborting an otherwise usable fit.
            # (Non-finite quadratics above still raise: those signal a broken
            # computation, not a degenerate-but-valid fit.)
            warnings.warn(
                "rg_decorrelated=True produced non-positive cross-sweep genetic "
                "variance (degenerate decorrelated denominator); reporting rg as "
                "NaN for this fit",
                RuntimeWarning,
                stacklevel=2,
            )
            rg = float("nan")
        else:
            rg = _rg_from_quadratics(num, v1, v2)
    else:
        # Ratio of the *raw* quadratics, exactly as docs/algorithm.md Equation 6
        # defines it. Dividing the raw numerator by the h2_bounds-clamped
        # variances is not a correlation: whenever a bound binds it drives rg
        # toward +/-1 while the true ratio is unchanged (a tightened h2_bounds
        # reported a true rg of 0.43 as 1.00). The non-positive quadratics that
        # motivated the clamped form -- possible on non-PD int8 blocks -- are
        # already handled by _rg_from_quadratics' own guard.
        rg = _rg_from_quadratics(g12, g11, g22)
    # Summarise both hyperparameters over exactly the same retained iterates.
    pi_mean = pi_samples[:count].mean(axis=0)
    s1_mean, s2_mean, s12_mean = sig_samples[:count].mean(axis=0)
    noise_mean = noise_scale_samples[:count].mean(axis=0)
    # Pass the raw quadratics, not the clamped h2: a clamped value can no
    # longer show that a bound was reached.
    _warn_if_implausible_fit((float(g11), float(g22)),
                             float(pi_mean[1] + pi_mean[2] + pi_mean[3]),
                             (lo, hi), m)
    return BivariateResult(beta1_est=avg1 / count, beta2_est=avg2 / count,
                           h2=(float(h2_1), float(h2_2)), rg=rg,
                           p=float(pi_mean[1] + pi_mean[2] + pi_mean[3]),
                           sigma=np.array([[s1_mean, s12_mean],
                                           [s12_mean, s2_mean]]),
                           pi=pi_mean,
                           pi_samples=pi_samples[:count].copy(),
                           sigma_samples=sig_samples[:count].copy(),
                           noise_scale=(float(noise_mean[0]),
                                        float(noise_mean[1])),
                           genetic_samples=genetic_samples[:count].copy(),
                           noise_scale_samples=noise_scale_samples[:count].copy(),
                           retained_iterations=int(count),
                           stopped_early=bool(count < num_iter))


def ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n_eff1, n_eff2, **kwargs):
    """Bivariate LDpred3-auto on a single dense or low-rank LD block.

    Convenience wrapper over :func:`ldpred3_auto_bivariate_blocks` for one block
    (or a block-diagonal genome packed into one matrix). See that function and
    :class:`BivariateResult` for the parameters and output. ``corr`` may be a
    dense matrix or an ldpred3 ``LowRankLD`` object. Dense matrices use
    size-aware automatic storage by default; pass ``ld_int8=True`` to quantise
    all dense float blocks or ``ld_int8=False`` to keep them float32.
    """
    # Derive the logical LD size from the effect vector. The block validator then
    # checks that ``corr`` is exactly square with this shape before quantisation.
    m = _validate_beta_hat(beta_hat1).shape[0]
    return ldpred3_auto_bivariate_blocks([(corr, np.arange(m))], beta_hat1,
                                         beta_hat2, n_eff1, n_eff2, **kwargs)
