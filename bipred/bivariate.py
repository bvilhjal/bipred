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
sampling noise, i.e. the bivariate-LDSC intercept); the default 0 assumes
independent GWAS samples.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

# bipred builds on ldpred3's shared internals: the Numba JIT shim (``_jit``), the
# per-variant-N helper (``_as_n_vector``) and the compact-LD sentinel
# (``LowRankLD``). They are re-exported from ``ldpred3.ldpred3`` -- the internal
# seam ldpred3 keeps stable for its own infer / annot / finemap modules.
from ldpred3.ldpred3 import (
    LowRankLD,
    _as_n_vector,
    _check_h2_p,
    _finite_control,
    _integer_at_least,
    _jit,
    _validate_beta_hat,
    _validate_blocks,
    _validate_boolean_controls,
    _validate_iterations,
    _validate_seed,
)

# int8 LD quantisation scale (127): correlations in [-1, 1] are stored as
# ``round(R * 127)`` int8 -- a quarter of the float32 memory -- and the sampler
# reads ``R[i, j] * (1 / 127)``. Imported from ldpred3 so bipred's encoding stays
# locked to the blocks ``ldpred3.compute_ld_blocks(quantize=True)`` produces.
from ldpred3._kernels import _Q8

__all__ = ["BivariateResult", "ldpred3_auto_bivariate",
           "ldpred3_auto_bivariate_blocks"]

DAMP = 0.2          # damping factor for the variance-component updates
_INIT_RHO_MAX = 0.999


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
    (rows are ``R @ v`` since ``R`` is symmetric), block by block. ``R`` may be
    int8-quantised, so each block carries a dequantisation ``scale`` (``1/127``
    for int8, ``1.0`` for float32) applied after the (off-hot-path) matmul."""
    out = np.zeros_like(V)
    for R, start, k, scale in fblocks:
        sl = slice(start, start + k)
        out[:, sl] = (V[:, sl] @ R.astype(V.dtype)) * scale
    return out


def _prepare_block(R, ld_int8):
    """Return ``(block, scale)`` for one dense LD block.

    Blocks that are already int8 (built by
    ``ldpred3.compute_ld_blocks(quantize=True)``) are kept int8 as-is. Otherwise,
    when ``ld_int8`` (the default) a float block is quantised to int8
    (``round(clip(R, -1, 1) * 127)`` -- a quarter of the float32 memory, matching
    ldpred3's representation); with ``ld_int8=False`` it is kept dense float32.
    The paired ``scale`` (``1/127`` for int8, ``1.0`` for float32) is what the
    sampler multiplies each LD entry by to dequantise on the fly."""
    arr = np.asarray(R)
    if arr.dtype == np.int8:
        return np.ascontiguousarray(arr), 1.0 / _Q8
    if ld_int8:
        q = np.rint(np.clip(np.ascontiguousarray(arr, np.float32), -1.0, 1.0) * _Q8)
        return q.astype(np.int8), 1.0 / _Q8
    return np.ascontiguousarray(arr, dtype=np.float32), 1.0


def _effect_sample_buffers(enabled, num_iter, sample_every, m):
    """Allocate decorrelated-rg effect traces only when explicitly enabled."""
    if not enabled:
        return None, None
    n_saved = (num_iter - 1) // sample_every + 1
    return (np.zeros((n_saved, m), dtype=np.float32),
            np.zeros((n_saved, m), dtype=np.float32))


def _decorrelated_cov(fblocks, samp1, samp2):
    """Genetic (co)variances from effect samples with **independent** noise.

    Uses effect vectors from *different* post-burn-in sweeps (thinned, so their
    sampling noise is ~independent) -- the LDpred2-auto out-of-sample trick
    applied to two traits -- so the same-sweep cross-noise that inflates the
    genetic covariance is removed and the covariance of an under-powered trait is
    recovered from its posterior mean rather than its thresholded samples. For
    each of ``b1' R b2`` / ``b1' R b1`` / ``b2' R b2`` returns the mean over
    ordered pairs ``a != b`` (all-pairs sum minus the ``a == b`` diagonal):
    ``(gcov, var1, var2)``. Returns ``None`` when there are too few samples."""
    n = samp1.shape[0]
    if n < 2:
        return None
    S1 = samp1.sum(0, keepdims=True)
    S2 = samp2.sum(0, keepdims=True)
    RS = _apply_R_rows(fblocks, np.vstack([S1, S2]))
    RS1, RS2 = RS[0], RS[1]
    all11 = float(S1[0] @ RS1); all12 = float(S1[0] @ RS2); all22 = float(S2[0] @ RS2)
    Rs1 = _apply_R_rows(fblocks, samp1)
    Rs2 = _apply_R_rows(fblocks, samp2)
    d11 = float(np.einsum("ij,ij->", samp1, Rs1))
    d12 = float(np.einsum("ij,ij->", samp1, Rs2))
    d22 = float(np.einsum("ij,ij->", samp2, Rs2))
    npairs = n * (n - 1)
    return (all12 - d12) / npairs, (all11 - d11) / npairs, (all22 - d22) / npairs


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

    if n_const:                                  # hoist the per-sweep constants
        (E11, E22, E12, det0, ldet0, a11, det1, ldet1, a22, det2, ldet2,
         b11, b22, b12, det3, ldet3, Ei11, Ei22, Ei12, prec1, sv1, prec2, sv2,
         V11, V22, V12, L11, L21, L22) = _bivar_const(
             n1[0], n2[0], s1, s2, s12, cross_corr)

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
        if not n_const:                           # per-variant N: recompute here
            (E11, E22, E12, det0, ldet0, a11, det1, ldet1, a22, det2, ldet2,
             b11, b22, b12, det3, ldet3, Ei11, Ei22, Ei12, prec1, sv1, prec2, sv2,
             V11, V22, V12, L11, L21, L22) = _bivar_const(
                 n1[j], n2[j], s1, s2, s12, cross_corr)

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


_bivar_one_sweep_jit = _jit(_bivar_one_sweep)


@dataclass
class BivariateResult:
    """Output of :func:`ldpred3_auto_bivariate`.

    ``beta1_est`` / ``beta2_est`` are the posterior-mean (standardized) effects
    for the two traits, ``h2`` the pair of SNP heritabilities, ``rg`` the
    estimated genetic correlation, ``p`` the causal fraction, ``sigma`` the
    learned 2x2 effect covariance, and ``pi`` the four-state mixture
    ``(pi00, pi10, pi01, pi11)`` = neither / trait-1-only / trait-2-only / both
    causal. ``sigma`` and ``pi`` are both means over the retained stochastic
    hyperparameter iterates. See :attr:`mixer` for the MiXeR-style
    polygenic-overlap summary.
    """

    beta1_est: np.ndarray
    beta2_est: np.ndarray
    h2: tuple
    rg: float
    p: float
    sigma: np.ndarray
    pi: np.ndarray = None
    pi_samples: np.ndarray = None       # (n_kept, 4) conditional mixture draws
    sigma_samples: np.ndarray = None    # (n_kept, 3) damped covariance iterates
    noise_scale: tuple = None           # learned (lambda1, lambda2); (1,1) if off

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

    def mixer_posterior(self, level=0.95):
        """Deprecated alias for :meth:`mixer_iterate_summary`.

        The historical ``ci`` fields are retained for compatibility, but they
        contain empirical central iterate intervals, not credible intervals.
        """
        warnings.warn(
            "mixer_posterior() is deprecated because Sigma is not sampled from "
            "a conditional posterior; use mixer_iterate_summary() for empirical "
            "hyperparameter-iterate intervals",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._mixer_iterate_summary(level, "ci")

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
                                  ld_int8=True,
                                  h2_init=0.1, p_init=0.02, rg_init=0.0,
                                  pi_init=None, sigma_prior_scale=None,
                                  cross_corr=0.0, burn_in=200, num_iter=200,
                                  h2_bounds=(1e-4, 1.0), h2_cap=None,
                                  iw_df=10.0, rg_decorrelated=False,
                                  noise_inflation=False, ni_damp=0.1,
                                  pi_prior=1.0, sample_every=5, seed=None):
    """Genome-wide bivariate LDpred3-auto over dense LD blocks.

    ``blocks`` is ``[(R, idx), ...]`` with contiguous ``idx`` arrays partitioning
    ``0..m-1``. The two traits share the same LD. Effects are updated block by
    block while ``pi`` and ``Sigma`` are pooled globally, so the genome-wide LD is
    never materialised. The compact ldpred3 ``LowRankLD`` representation is
    not supported; pass dense float or dense int8 blocks.

    By default the LD is stored **int8**-quantised (a quarter of the float32
    memory; the sampler dequantises on the fly), matching ldpred3's
    pipeline default representation. int8 blocks from
    ``ldpred3.compute_ld_blocks(quantize=True)``
    are consumed as-is; pass ``ld_int8=False`` to keep dense float32.

    Parameters
    ----------
    blocks : list of (ndarray, ndarray)
        Dense per-block LD ``(R, idx)`` partitioning ``0..m-1``. ``R`` may be
        float32/float64 or int8-quantised (``round(R * 127)``).
    beta_hat1, beta_hat2 : array_like (m,)
        Standardized marginal effects for the two traits (same variant order).
    n_eff1, n_eff2 : float or array_like
        Per-trait GWAS sample sizes.
    ld_int8 : bool, default True
        Store the LD int8-quantised (``round(clip(R, -1, 1) * 127)``) -- a quarter
        of the float32 memory, dequantised in the sampler's inner loop. The
        quantisation error is negligible (the diagonal stays exactly 1); set
        ``False`` for an exact dense-float32 fit. Blocks already int8 stay int8
        regardless of this flag.
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
        Cross-trait correlation of the sampling noise (sample overlap); must lie
        in ``(-1, 1)``. 0 assumes independent GWAS samples.
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
        Estimate ``rg`` from effects sampled at different sweeps. Prefer for
        asymmetric-power pairs.
    noise_inflation : bool, default False
        Learn per-trait residual noise factors ``lambda_t >= 1`` and fit with
        effective sample size ``N_t / lambda_t``. Useful for finite reference-panel
        LD when absolute overlap counts are inflated by mismatch.
    ni_damp : float, default 0.1
        Damping for the per-sweep ``lambda`` update (only used with
        ``noise_inflation``); smaller is more stable, larger adapts faster.
    sample_every : int, default 5
        Thinning for the retained effect samples used by the decorrelated ``rg``.
    pi_prior : float, default 1.0
        Symmetric Dirichlet concentration for the four-state mixture prior.
    seed : int or None

    Returns
    -------
    BivariateResult
    """
    h2_init = _finite_scalar_or_pair("h2_init", h2_init)
    rg_init = _finite_control("rg_init", rg_init)
    if not -1.0 < rg_init < 1.0:
        raise ValueError("rg_init must be in (-1, 1)")
    cross_corr = _finite_control("cross_corr", cross_corr)
    if not -1.0 < cross_corr < 1.0:
        raise ValueError("cross_corr must be in (-1, 1)")
    burn_in, num_iter = _validate_iterations(burn_in, num_iter)
    _validate_boolean_controls(
        ld_int8=ld_int8,
        rg_decorrelated=rg_decorrelated,
        noise_inflation=noise_inflation,
    )
    iw_df = _finite_control("iw_df", iw_df)
    if iw_df <= 0.0:
        raise ValueError("iw_df must be positive")
    ni_damp = _finite_control("ni_damp", ni_damp)
    if not 0.0 < ni_damp <= 1.0:
        raise ValueError("ni_damp must be in (0, 1]")
    pi_prior = _finite_control("pi_prior", pi_prior)
    if pi_prior <= 0.0:
        raise ValueError("pi_prior must be positive (an improper <=0 "
                         "concentration can collapse the mixture)")
    sample_every = _integer_at_least("sample_every", sample_every, 1)
    seed = _validate_seed(seed)

    lo, hi = _finite_pair("h2_bounds", h2_bounds)
    if not (0.0 < lo <= min(h2_init) and max(h2_init) <= hi):
        raise ValueError(
            "h2_bounds must contain both positive h2_init values"
        )
    h2_bounds = (lo, hi)
    if h2_cap is not None:
        h2_cap = _finite_pair("h2_cap", h2_cap)
        if h2_cap[0] <= 0.0 or h2_cap[1] <= 0.0:
            raise ValueError("h2_cap values must be positive")

    bh1 = np.ascontiguousarray(_validate_beta_hat(beta_hat1), dtype=np.float64)
    bh2 = np.ascontiguousarray(_validate_beta_hat(beta_hat2), dtype=np.float64)
    m = bh1.shape[0]
    if m == 0:
        raise ValueError("beta_hat vectors must contain at least one variant")
    if bh2.shape[0] != m:
        raise ValueError("beta_hat1 and beta_hat2 must have the same length")
    n1 = _as_n_vector(n_eff1, m)
    n2 = _as_n_vector(n_eff2, m)
    # Shared (scalar) N -> the noise-covariance / determinant / posterior scalars
    # are identical for every SNP each sweep, so the kernel hoists them out of the
    # per-SNP loop. Per-variant N falls back to the exact per-SNP computation.
    n_const = bool(n1.min() == n1.max() and n2.min() == n2.max())

    blocks = _validate_blocks(blocks, m, contiguous=True)
    fblocks = []
    for R, idx in sorted(blocks, key=lambda bi: int(bi[1][0])):
        if isinstance(R, LowRankLD):
            raise NotImplementedError(
                "bivariate LDpred3 needs dense LD blocks, not a "
                f"{type(R).__name__}; it does not support the compact "
                "LD representation")
        Rq, scale = _prepare_block(R, ld_int8)
        fblocks.append((Rq, int(idx[0]), int(idx.shape[0]), scale))
    pi, s1, s2, s12 = _initial_hyperparameters(
        m, h2_init, p_init, rg_init, pi_init=pi_init,
    )
    if sigma_prior_scale is None:
        psi1, psi2 = s1, s2
    else:
        psi1, psi2 = _finite_scalar_or_pair(
            "sigma_prior_scale", sigma_prior_scale,
        )

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

    rng = np.random.default_rng(seed)
    curr1 = np.zeros(m); curr2 = np.zeros(m)
    rb1 = np.zeros(m); rb2 = np.zeros(m)
    avg1 = np.zeros(m); avg2 = np.zeros(m)
    count = 0
    gv_acc = np.zeros(3)
    # Retained post-burn-in hyperparameter iterates. pi is a conditional
    # Dirichlet draw; Sigma is a damped moment update driven by stochastic effect
    # and state draws, not a conditional posterior draw.
    pi_samples = np.zeros((num_iter, 4))
    sig_samples = np.zeros((num_iter, 3))
    # These two O(num_iter * m) buffers are needed only by the optional
    # decorrelated-rg estimator. Default runs should not pay their memory cost.
    samp1, samp2 = _effect_sample_buffers(
        rg_decorrelated, num_iter, sample_every, m)
    n_saved = 0

    # ``pi`` and the slab covariance were calibrated together above: their
    # implied marginal h2 and genetic rg equal the documented starting values.
    # Per-trait noise-inflation factors (LDSC-intercept analog); 1 = off.
    lam1 = lam2 = 1.0

    for it in range(burn_in + num_iter):
        resync = (it % 100 == 0)
        unif = rng.random(m)
        z1 = rng.standard_normal(m)
        z2 = rng.standard_normal(m)
        rbs1 = np.zeros(m); rbs2 = np.zeros(m)
        lpi = np.log(np.maximum(pi, 1e-300))
        c10 = c01 = c11 = 0
        S1 = S2 = S12 = 0.0
        gv11 = gv12 = gv22 = 0.0
        # Effective per-variant N deflated by the learned noise inflation. A scalar
        # lambda preserves the constant-N fast path (n_const unchanged).
        n1e = n1 / lam1 if noise_inflation else n1
        n2e = n2 / lam2 if noise_inflation else n2
        for R, start, k, scale in fblocks:
            sl = slice(start, start + k)
            a10, a01, a11, s1sq, s2sq, s12s, g11, g12, g22 = _bivar_one_sweep_jit(
                R, bh1[sl], bh2[sl], n1e[sl], n2e[sl], curr1[sl], curr2[sl],
                rb1[sl], rb2[sl], rbs1[sl], rbs2[sl], unif[sl], z1[sl], z2[sl],
                float(lpi[0]), float(lpi[1]), float(lpi[2]), float(lpi[3]),
                float(s1), float(s2), float(s12), float(cross_corr),
                float(scale), n_const, resync)
            c10 += a10; c01 += a01; c11 += a11
            S1 += s1sq; S2 += s2sq; S12 += s12s
            gv11 += g11; gv12 += g12; gv22 += g22

        if noise_inflation:
            # Update lambda_t from the residual mean-chi2. rb1/rb2 hold R@beta
            # after the sweep, so b_hat - R@beta is the residual; under matched LD
            # it is pure sampling noise (mean n*resid^2 ~ 1) and inflated otherwise.
            r1 = bh1 - rb1; r2 = bh2 - rb2
            lh1 = max(float(np.mean(n1 * r1 * r1)), 1.0)
            lh2 = max(float(np.mean(n2 * r2 * r2)), 1.0)
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
            if (rg_decorrelated and (it - burn_in) % sample_every == 0):
                samp1[n_saved] = curr1
                samp2[n_saved] = curr2
                n_saved += 1
            count += 1

    # num_iter >= 1 is validated at the public boundary, so count cannot be 0.
    if count != num_iter:                         # defensive internal invariant
        raise RuntimeError("internal error: retained-iteration count mismatch")
    g11, g12, g22 = gv_acc / count
    h2_1 = min(max(g11, lo), hi)
    h2_2 = min(max(g22, lo), hi)
    # rg from effect samples with independent noise (drawn at different sweeps):
    # the decorrelated genetic covariance over the decorrelated predictor
    # variances. This avoids the same-sweep cross-noise that inflates the genetic
    # covariance and recovers a weak trait's covariance from its posterior mean
    # (which the sampled-quadratic ratio attenuates). Falls back to the
    # sampled-quadratic ratio when disabled or too few samples were retained.
    rg = None
    if rg_decorrelated:
        cov = _decorrelated_cov(fblocks, samp1[:n_saved], samp2[:n_saved])
        if cov is not None:
            num, v1, v2 = cov
            if v1 > 0.0 and v2 > 0.0:
                rg = float(min(max(num / np.sqrt(v1 * v2), -1.0), 1.0))
    if rg is None:
        # Use the reported (clamped) h2 scale for the denominator: raw sampled
        # quadratics can go non-positive on non-PD (int8-quantised) blocks,
        # which would slam rg to +/-1 through the floor.
        rg = float(min(max(g12 / np.sqrt(h2_1 * h2_2), -1.0), 1.0))
    # Summarise both hyperparameters over exactly the same retained iterates.
    pi_mean = pi_samples.mean(axis=0)
    s1_mean, s2_mean, s12_mean = sig_samples.mean(axis=0)
    return BivariateResult(beta1_est=avg1 / count, beta2_est=avg2 / count,
                           h2=(float(h2_1), float(h2_2)), rg=rg,
                           p=float(pi_mean[1] + pi_mean[2] + pi_mean[3]),
                           sigma=np.array([[s1_mean, s12_mean],
                                           [s12_mean, s2_mean]]),
                           pi=pi_mean,
                           pi_samples=pi_samples[:count].copy(),
                           sigma_samples=sig_samples[:count].copy(),
                           noise_scale=(float(lam1), float(lam2)))


def ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n_eff1, n_eff2, **kwargs):
    """Bivariate LDpred3-auto on a single dense LD matrix.

    Convenience wrapper over :func:`ldpred3_auto_bivariate_blocks` for one block
    (or a block-diagonal genome packed into one matrix). See that function and
    :class:`BivariateResult` for the parameters and output. By default the matrix
    is stored int8-quantised; pass ``ld_int8=False`` for an exact float32 fit.
    """
    # Derive the logical LD size from the effect vector. The block validator then
    # checks that ``corr`` is exactly square with this shape before quantisation.
    m = _validate_beta_hat(beta_hat1).shape[0]
    return ldpred3_auto_bivariate_blocks([(corr, np.arange(m))], beta_hat1,
                                         beta_hat2, n_eff1, n_eff2, **kwargs)
