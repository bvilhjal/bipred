"""Numba kernels of the bivariate sampler: per-block sweeps and fused drivers.

Moved out of :mod:`bipred.bivariate` so the driver module holds validation,
preparation, the chain loop and the result type, while this module holds only
code that is compiled. Every public-by-convention name here is re-exported by
:mod:`bipred.bivariate`, whose call sites and tests keep addressing them there.
"""

from __future__ import annotations

import numpy as np

from ._ldpred3_compat import (
    _get_thread_id,
    _jit,
    _jit_fastmath_nogil,
    _jit_nogil,
    _jit_parallel,
    prange,
)

__all__ = [
    "_bivar_block_reduce",
    "_bivar_block_reduce_jit",
    "_bivar_const",
    "_bivar_dense_sweep_all",
    "_bivar_dense_sweep_all_jit",
    "_bivar_dense_sweep_all_par_jit",
    "_bivar_lowrank_sweep_all",
    "_bivar_lowrank_sweep_all_jit",
    "_bivar_lowrank_sweep_all_par_jit",
    "_bivar_one_sweep",
    "_bivar_one_sweep_jit",
    "_bivar_one_sweep_lowrank",
    "_bivar_one_sweep_lowrank_jit",
    "_dequantise_lr8_factor",
    "_dequantise_lr8_factor_jit",
]


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

        # posterior effect means under each non-null state.
        m1_1 = (Ei11 * d1 + Ei12 * d2) / prec1    # state 1 (trait-1 only)
        m2_2 = (Ei22 * d2 + Ei12 * d1) / prec2    # state 2 (trait-2 only)
        g1 = Ei11 * d1 + Ei12 * d2                # state 3 (both)
        g2 = Ei12 * d1 + Ei22 * d2
        m1_3 = V11 * g1 + V12 * g2
        m2_3 = V12 * g1 + V22 * g2

        # State weights relative to the null state (Gaussian conditioning):
        # log w_s = log pi_s - (ldet_s - ldet0)/2 + g' mu_s / 2, with
        # g = E^-1 d and mu_s the state-s posterior mean computed above. The
        # null state's -0.5 * (ldet0 + d' E^-1 d) is common to all four states
        # and cancels in the wmax normalisation below. Algebraically equal to
        # the four Mahalanobis quadratics, which remain the test oracle.
        w0 = lpi00
        w1 = lpi10 - 0.5 * (ldet1 - ldet0) + 0.5 * g1 * m1_1
        w2 = lpi01 - 0.5 * (ldet2 - ldet0) + 0.5 * g2 * m2_2
        w3 = lpi11 - 0.5 * (ldet3 - ldet0) + 0.5 * (g1 * m1_3 + g2 * m2_3)

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

        # posterior effect means under each non-null state.
        m1_1 = (Ei11 * d1 + Ei12 * d2) / prec1
        m2_2 = (Ei22 * d2 + Ei12 * d1) / prec2
        g1 = Ei11 * d1 + Ei12 * d2
        g2 = Ei12 * d1 + Ei22 * d2
        m1_3 = V11 * g1 + V12 * g2
        m2_3 = V12 * g1 + V22 * g2

        # State weights relative to the null state (Gaussian conditioning):
        # log w_s = log pi_s - (ldet_s - ldet0)/2 + g' mu_s / 2, with
        # g = E^-1 d and mu_s the state-s posterior mean computed above. The
        # null state's -0.5 * (ldet0 + d' E^-1 d) is common to all four states
        # and cancels in the wmax normalisation below. Algebraically equal to
        # the four Mahalanobis quadratics, which remain the test oracle.
        w0 = lpi00
        w1 = lpi10 - 0.5 * (ldet1 - ldet0) + 0.5 * g1 * m1_1
        w2 = lpi01 - 0.5 * (ldet2 - ldet0) + 0.5 * g2 * m2_2
        w3 = lpi11 - 0.5 * (ldet3 - ldet0) + 0.5 * (g1 * m1_3 + g2 * m2_3)

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


# The serial and parallel twins of each fused driver must not share one Numba
# cache identity: Numba keys its on-disk cache on (module, qualname, first
# line) and ignores the compile flags, so two dispatchers over a single
# function object collide and whichever compiles first is served to both
# (measured: 5.38 vs 1.73 ms/sweep when a serial run had warmed the cache).
# ldpred3's _jit_parallel compiles a clone under a "__par" qualname, giving the
# parallel twin its own cache entry -- so a fresh ncores>1 process no longer
# recompiles, without reintroducing the collision.
_bivar_dense_sweep_all_par_jit = _jit_parallel(_bivar_dense_sweep_all)


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


_bivar_lowrank_sweep_all_par_jit = _jit_parallel(_bivar_lowrank_sweep_all)


_bivar_lowrank_sweep_all_jit = _jit_nogil(_bivar_lowrank_sweep_all)


def _bivar_block_reduce(counts, stats):
    """Reduce per-block sweep statistics in genomic block order.

    ``counts`` (nblocks, 3) int64 and ``stats`` (nblocks, 6) float64 arrive
    already scattered back to genomic block order by the driver. Summation runs
    left to right in that order -- exactly the serial driver's per-block loop --
    so replacing the Python loop with this compiled one changes no result bits.
    Strict math (no fastmath): the summation order is the contract.
    """
    c10 = c01 = c11 = 0
    S1 = S2 = S12 = gv11 = gv12 = gv22 = 0.0
    for b in range(len(counts)):
        c10 += int(counts[b, 0])
        c01 += int(counts[b, 1])
        c11 += int(counts[b, 2])
        S1 += float(stats[b, 0])
        S2 += float(stats[b, 1])
        S12 += float(stats[b, 2])
        gv11 += float(stats[b, 3])
        gv12 += float(stats[b, 4])
        gv22 += float(stats[b, 5])
    return c10, c01, c11, S1, S2, S12, gv11, gv12, gv22


_bivar_block_reduce_jit = _jit_nogil(_bivar_block_reduce)
