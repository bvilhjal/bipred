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
joint model safe: whether the two traits' causal variants co-occur is *learned*
(``pi11``), not assumed. Two traits that share causal variants and are
genetically correlated let the better-powered one sharpen the other (via the
``both`` component); two traits with disjoint causal variants drive ``pi11 -> 0``
so the joint fit reduces to the independent ones and does no harm.

Both GWAS are assumed to use the **same** LD reference (same ancestry). Sample
overlap can be passed via ``cross_corr`` (the cross-trait correlation of the
sampling noise, i.e. the bivariate-LDSC intercept); the default 0 assumes
independent GWAS samples. NumPy only (optional Numba).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# bipred builds on ldpred3's shared internals: the Numba JIT shim (``_jit``), the
# per-variant-N helper (``_as_n_vector``) and the compact-LD sentinel
# (``LowRankLD``). They are re-exported from ``ldpred3.ldpred3`` -- the internal
# seam ldpred3 keeps stable for its own infer / annot / finemap modules.
from ldpred3.ldpred3 import _jit, _as_n_vector, LowRankLD

__all__ = ["BivariateResult", "ldpred3_auto_bivariate",
           "ldpred3_auto_bivariate_blocks"]

DAMP = 0.2          # damping factor for the variance-component updates


def _apply_R_rows(fblocks, V):
    """Right-multiply each row of ``V`` (n, m) by the block-diagonal LD ``R``
    (rows are ``R @ v`` since ``R`` is symmetric), block by block."""
    out = np.zeros_like(V)
    for R, start, k in fblocks:
        sl = slice(start, start + k)
        out[:, sl] = V[:, sl] @ R.astype(V.dtype)
    return out


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
                     n_const, resync):
    """One Gibbs sweep of the 4-state model over a block; mutates in place.

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
                    rb1[i] += cj[i] * b1
                    rb2[i] += cj[i] * b2

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
                cij = cj[i]
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
    learned 2x2 effect covariance, and ``pi`` the posterior-mean four-state
    mixture ``(pi00, pi10, pi01, pi11)`` = neither / trait-1-only / trait-2-only /
    both causal. See :attr:`mixer` for the MiXeR-style polygenic-overlap summary.
    """

    beta1_est: np.ndarray
    beta2_est: np.ndarray
    h2: tuple
    rg: float
    p: float
    sigma: np.ndarray
    pi: np.ndarray = None
    pi_samples: np.ndarray = None       # (n_kept, 4) post-burn-in mixture draws
    sigma_samples: np.ndarray = None    # (n_kept, 3) post-burn-in (s1, s2, s12) draws
    noise_scale: tuple = None           # learned (lambda1, lambda2); (1,1) if off

    @property
    def mixer(self):
        """MiXeR-style polygenic-overlap parameters (Frei et al. 2019).

        The four-state mixture *is* a bivariate causal mixture model, so it yields
        the same quantities MiXeR reports. Returns a dict with, over the ``m``
        fitted variants:

        * ``polygenicity`` -- causal fraction per trait ``(pi1, pi2)`` where
          ``pi1 = pi10 + pi11``.
        * ``n_causal`` -- causal counts ``(pi1*m, pi2*m)``.
        * ``n_shared`` -- shared causal count ``pi11*m``; ``frac_shared`` is
          ``pi11 / min(pi1, pi2)`` (fraction of the less-polygenic trait's causal
          variants shared with the other).
        * ``rho_beta`` -- correlation of effect sizes **within the shared
          component**, ``s12 / sqrt(s1 s2)``.
        * ``rg_from_overlap`` -- the MiXeR decomposition
          ``rho_beta * pi11 / sqrt(pi1 pi2)`` (genetic correlation = shared
          fraction x within-shared effect correlation); a consistency check
          against ``rg``.

        Caveat: the point-normal mixture does **not calibrate absolute
        polygenicity** — the four-state posterior counts a causal variant's LD
        neighbours as (partially) causal too, so ``n_causal`` / ``n_shared`` are
        biased by an architecture- and power-dependent factor (benchmarked from
        ~0.3x with clustered causal variants to ~2.5x when they are spread across
        LD blocks and N is large; see ``benchmarks/mixer_overlap.py``). The
        dominant term is **LD-reference mismatch** (the bias grows with N and
        collapses when the fit uses the exact in-sample LD); ``r_g`` and the
        overlap fraction are *ratios* and cancel it, so read ``polygenicity`` /
        ``n_*`` as *relative*. To put the counts on a calibrated scale, anchor them
        with two univariate ``ldpred3_auto_infer`` runs via :meth:`mixer_calibrated`
        (the univariate polygenicity is far less LD-mismatch-sensitive). A dedicated
        causal-mixture likelihood (MiXeR) is what calibrates the absolute counts
        from scratch. Needs a well-powered pair (large ``N*h2/m``) to be meaningful.

        For the **posterior distribution** of these quantities (a credible interval,
        not just this point estimate), see :meth:`mixer_posterior`, which maps the
        retained Gibbs draws of ``(pi, Sigma)`` through the same decomposition. That
        interval is calibrated and covers the truth when the LD reference matches
        the GWAS sample; the absolute-count bias discussed above is LD-reference
        mismatch and is *not* reflected in the interval.
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

    def mixer_posterior(self, level=0.95):
        """Posterior **distribution** of the overlap counts, from the retained
        Gibbs samples of ``(pi, Sigma)`` -- i.e. the posterior overlap counts
        given the prior and the data, with a credible interval rather than only
        the :attr:`mixer` point estimate.

        Each retained sweep is a draw ``(pi, Sigma)`` from the joint posterior, so
        mapping every draw through the same MiXeR decomposition as :attr:`mixer`
        (``n_causal``, ``n_shared``, ``frac_shared``, ``rho_beta``,
        ``rg_from_overlap``) and summarising gives the posterior mean and a central
        ``level`` credible interval for each quantity.

        Returns a dict mirroring :attr:`mixer`, but each entry is
        ``{"mean": float, "ci": (lo, hi), "sd": float}`` (``n_causal`` /
        ``polygenicity`` are 2-tuples of such dicts, one per trait).

        **What the interval means (and does not).** It is the posterior spread
        *conditional on the supplied LD reference* -- Monte-Carlo / sampling
        uncertainty. It is **not** a bound on LD-reference-mismatch bias: when the
        fit LD matches the GWAS sample the absolute counts are well calibrated and
        this interval covers the truth, but under a mismatched reference panel the
        absolute counts inflate (growing with N) and the interval is tight around
        the *wrong* value. The ratios (``frac_shared``, ``rho_beta``,
        ``rg_from_overlap``) stay reliable either way. See :attr:`mixer` for the
        bias discussion and use matched / larger / QC'd LD for calibrated absolute
        counts.
        """
        if self.pi_samples is None or self.sigma_samples is None:
            raise ValueError("posterior samples not available on this result")
        if len(self.pi_samples) == 0:
            raise ValueError("no post-burn-in samples were retained")
        m = len(self.beta1_est)
        lo_q = (1.0 - level) / 2.0 * 100.0
        hi_q = (1.0 + level) / 2.0 * 100.0
        cols = {"n1": [], "n2": [], "n_shared": [], "frac_shared": [],
                "rho_beta": [], "rg_from_overlap": []}
        for (_p00, p10, p01, p11), (s1, s2, s12) in zip(self.pi_samples,
                                                        self.sigma_samples):
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
                    "ci": (float(np.percentile(a, lo_q)),
                           float(np.percentile(a, hi_q)))}
        n1, n2 = summ(cols["n1"]), summ(cols["n2"])
        return {
            "n_causal": (n1, n2),
            "polygenicity": ({**n1, "mean": n1["mean"] / m,
                              "sd": n1["sd"] / m,
                              "ci": (n1["ci"][0] / m, n1["ci"][1] / m)},
                             {**n2, "mean": n2["mean"] / m,
                              "sd": n2["sd"] / m,
                              "ci": (n2["ci"][0] / m, n2["ci"][1] / m)}),
            "n_shared": summ(cols["n_shared"]),
            "frac_shared": summ(cols["frac_shared"]),
            "rho_beta": summ(cols["rho_beta"]),
            "rg_from_overlap": summ(cols["rg_from_overlap"]),
            "level": level,
        }

    def mixer_calibrated(self, infer1, infer2):
        """:attr:`mixer` with the **absolute counts calibrated** by two univariate
        ``ldpred3_auto_infer`` runs.

        The joint fit's per-trait polygenicity is inflated mainly by LD-reference
        mismatch, and the four-state sampler is ~2x more sensitive to it than a
        univariate fit. This keeps the joint fit's reliable *ratios* — the shared
        fraction ``frac_shared`` and the within-shared effect correlation
        ``rho_beta`` — but replaces ``(pi1, pi2)`` with the univariate learned
        polygenicities (well-calibrated when the LD matches), and rebuilds
        ``n_causal`` / ``n_shared`` / ``rg_from_overlap`` on that scale. This is the
        recommended readout for **absolute** overlap counts.

        ``infer1`` / ``infer2`` are the trait-1 / trait-2 :class:`InferResult`
        objects (their ``p_est`` is used); floats are also accepted.
        """
        if self.pi is None:
            raise ValueError("pi not available on this result")
        p1 = float(getattr(infer1, "p_est", infer1))
        p2 = float(getattr(infer2, "p_est", infer2))
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
                                  h2_init=0.1, p_init=0.1, rg_init=0.0,
                                  cross_corr=0.0, burn_in=200, num_iter=200,
                                  h2_bounds=(1e-4, 1.0), h2_cap=None,
                                  iw_df=10.0, rg_decorrelated=False,
                                  noise_inflation=False, ni_damp=0.1,
                                  sample_every=5, seed=None):
    """Genome-wide (streaming) bivariate LDpred3-auto.

    ``blocks`` is the ``[(R, idx), ...]`` list (contiguous ``idx`` partitioning
    ``0..m-1``) used elsewhere; the two traits' summary statistics share it. The
    effect sweeps run one block at a time while ``pi`` and ``Sigma`` are pooled
    globally, so the genome-wide LD is never materialised.

    Parameters
    ----------
    blocks : list of (ndarray, ndarray)
        Per-block LD ``(R, idx)`` partitioning ``0..m-1``.
    beta_hat1, beta_hat2 : array_like (m,)
        Standardized marginal effects for the two traits (same variant order).
    n_eff1, n_eff2 : float or array_like
        Per-trait GWAS sample sizes.
    h2_init, p_init, rg_init : float
        Initial heritability, causal fraction and genetic correlation.
    cross_corr : float, default 0.0
        Cross-trait correlation of the sampling noise (sample overlap); must lie
        in ``(-1, 1)``. 0 assumes independent GWAS samples.
    burn_in, num_iter : int
        Burn-in and sampling sweeps.
    h2_bounds : (float, float)
        Clamp range for the per-trait heritabilities.
    h2_cap : (float, float), optional
        Optional hard ceilings on the implied per-trait h2 (expert override). By
        default (``None``) the (co)variance components are regularised by the
        inverse-Wishart-style prior instead (see ``iw_df``); pass known
        heritabilities only if you want to additionally clamp them.
    iw_df : float, default 10
        Strength (pseudo-count / prior degrees of freedom) of the inverse-Wishart
        shrinkage on the effect-covariance ``Sigma``: the (co)variance components
        are shrunk toward a weak diagonal prior (zero prior genetic covariance),
        which keeps ``Sigma`` positive-definite and the covariance off the PD
        boundary. Larger = more shrinkage toward independent traits.
    rg_decorrelated : bool, default False
        How ``rg`` is estimated. ``False`` (default) uses the same-sweep
        quadratic ratio ``E[b1'Rb2]/sqrt(E[b1'Rb1]E[b2'Rb2])`` -- accurate and
        tight when the two traits are **similarly powered**. ``True`` estimates it
        from effect vectors sampled at *different* sweeps (independent noise),
        which recovers an **under-powered** trait's covariance that the same-sweep
        ratio attenuates -- prefer it for **asymmetric-power** pairs (e.g.
        boosting a weak trait with a strong correlated one), at the cost of a
        small over-estimate when the traits are balanced.
    noise_inflation : bool, default False
        Learn a per-trait **noise-inflation** factor ``lambda_t >= 1`` (an
        LDSC-intercept analog) from the residual misfit and fit with an
        *effective* sample size ``N_t / lambda_t``. The residual ``b_hat - R@beta``
        is pure sampling noise (mean-chi2 ~ 1) when the LD reference matches the
        GWAS sample, but is inflated under **LD-reference mismatch**; the learned
        ``lambda`` makes the sampler correspondingly less confident so it stops
        reading that misfit as extra polygenicity. This targets the **absolute
        polygenic-overlap counts** (:attr:`BivariateResult.mixer`), whose bias is
        the mismatch-driven inflation that grows with ``N``. It removes that
        N-growing component while leaving ``h2`` and ``rg`` essentially unchanged
        (validated in ``benchmarks/mixer_overlap.py``): on well-conditioned LD the
        counts are deflated ~all the way back to the truth; on realistic coalescent
        LD it substantially reduces the inflation but a scalar ``lambda`` cannot
        absorb structured mismatch and dense-causal LD-spreading entirely. It is
        ~a no-op under matched LD (``lambda ~ 1``). Recommended when fitting on a
        finite **reference panel** (the usual case); left off by default so the
        estimator is unchanged unless requested. The learned factors are returned
        in ``BivariateResult.noise_scale``.
    ni_damp : float, default 0.1
        Damping for the per-sweep ``lambda`` update (only used with
        ``noise_inflation``); smaller is more stable, larger adapts faster.
    sample_every : int, default 5
        Thinning for the retained effect samples used by the decorrelated ``rg``.
    seed : int or None

    Returns
    -------
    BivariateResult
    """
    if not -1.0 < cross_corr < 1.0:
        raise ValueError("cross_corr must be in (-1, 1)")
    bh1 = np.ascontiguousarray(beta_hat1, dtype=np.float64)
    bh2 = np.ascontiguousarray(beta_hat2, dtype=np.float64)
    m = bh1.shape[0]
    if bh2.shape[0] != m:
        raise ValueError("beta_hat1 and beta_hat2 must have the same length")
    n1 = _as_n_vector(n_eff1, m)
    n2 = _as_n_vector(n_eff2, m)
    # Shared (scalar) N -> the noise-covariance / determinant / posterior scalars
    # are identical for every SNP each sweep, so the kernel hoists them out of the
    # per-SNP loop. Per-variant N falls back to the exact per-SNP computation.
    n_const = bool(n1.min() == n1.max() and n2.min() == n2.max())

    fblocks = []
    for R, idx in sorted(blocks, key=lambda bi: int(np.asarray(bi[1])[0])):
        if isinstance(R, LowRankLD):
            raise NotImplementedError(
                "bivariate LDpred3 needs dense LD blocks, not a "
                f"{type(R).__name__}; it does not support the compact "
                "(low-rank) LD representation")
        idx = np.asarray(idx)
        if not np.array_equal(idx, np.arange(idx[0], idx[0] + idx.shape[0])):
            raise ValueError("each block must use contiguous indices")
        fblocks.append((np.ascontiguousarray(R, dtype=np.float32),
                        int(idx[0]), int(idx.shape[0])))
    starts = [s for _, s, _ in fblocks]
    ends = [s + k for _, s, k in fblocks]
    if (sum(k for _, _, k in fblocks) != m or starts[0] != 0
            or starts[1:] != ends[:-1] or ends[-1] != m):
        raise ValueError("blocks must partition 0..m-1 exactly once")

    lo, hi = h2_bounds
    M = float(m)

    # (Co)variance-component regularisation. The effect covariance Sigma is
    # updated each sweep by shrinking toward a weak inverse-Wishart prior (MTGSAM
    # / Sorensen-Gianola): scale PSI0 (the expected per-variant slab) on the
    # diagonal, zero off-diagonal, with iw_df pseudo-counts. This replaces the old
    # scheme (a univariate-auto h2 ceiling + a hard 0.999 PD cap): the univariate
    # anchor under-estimates h2 on noisy dense LD -> shrinks the rg denominator ->
    # inflated rg, while the diagonal prior here keeps Sigma positive-definite and
    # the off-diagonal from riding the PD boundary. A caller may still pass
    # ``h2_cap`` to additionally clamp the implied per-trait h2 (expert override).
    PSI0 = float(h2_init) / max(p_init * M, 1.0)
    nu0 = float(iw_df)

    rng = np.random.default_rng(seed)
    curr1 = np.zeros(m); curr2 = np.zeros(m)
    rb1 = np.zeros(m); rb2 = np.zeros(m)
    avg1 = np.zeros(m); avg2 = np.zeros(m)
    count = 0
    gv_acc = np.zeros(3)
    pi_acc = np.zeros(4)          # posterior-mean 4-state mixture (MiXeR overlap)
    # Retained post-burn-in posterior samples of the mixture pi and the effect
    # covariance (s1, s2, s12) -- the raw material for the posterior distribution
    # (credible intervals) of the overlap counts (see BivariateResult.mixer_posterior).
    pi_samples = np.zeros((num_iter, 4))
    sig_samples = np.zeros((num_iter, 3))
    # Thinned post-burn-in effect samples for the decorrelated rg estimate.
    sample_every = max(int(sample_every), 1)
    max_saved = num_iter // sample_every + 1
    samp1 = np.zeros((max_saved, m), dtype=np.float32)
    samp2 = np.zeros((max_saved, m), dtype=np.float32)
    n_saved = 0

    # state probabilities (pi00, pi10, pi01, pi11) and slab variances.
    pi = np.array([1.0 - p_init, p_init / 3.0, p_init / 3.0, p_init / 3.0])
    s1 = s2 = float(h2_init) / max(p_init * M, 1.0)
    s12 = float(rg_init) * s1
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
        for R, start, k in fblocks:
            sl = slice(start, start + k)
            a10, a01, a11, s1sq, s2sq, s12s, g11, g12, g22 = _bivar_one_sweep_jit(
                R, bh1[sl], bh2[sl], n1e[sl], n2e[sl], curr1[sl], curr2[sl],
                rb1[sl], rb2[sl], rbs1[sl], rbs2[sl], unif[sl], z1[sl], z2[sl],
                float(lpi[0]), float(lpi[1]), float(lpi[2]), float(lpi[3]),
                float(s1), float(s2), float(s12), float(cross_corr),
                n_const, resync)
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
        pi = rng.dirichlet([1.0 + c00, 1.0 + c10, 1.0 + c01, 1.0 + c11])
        n1c = c10 + c11
        n2c = c01 + c11
        # Inverse-Wishart-style shrinkage of (s1, s2, s12) toward the weak
        # diagonal prior (scale PSI0, nu0 pseudo-counts, zero prior covariance).
        # Marginal variances pool all trait-causal variants; the covariance uses
        # the both-causal pairs and is pulled toward 0 by the prior (no genetic
        # covariance a priori), which keeps s12 off the PD boundary. Damped for
        # cross-sweep stability.
        s1 = (1.0 - DAMP) * s1 + DAMP * (nu0 * PSI0 + S1) / (nu0 + n1c)
        s2 = (1.0 - DAMP) * s2 + DAMP * (nu0 * PSI0 + S2) / (nu0 + n2c)
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
            pi_acc += pi
            pi_samples[count] = pi
            sig_samples[count] = (s1, s2, s12)
            if (it - burn_in) % sample_every == 0:
                samp1[n_saved] = curr1
                samp2[n_saved] = curr2
                n_saved += 1
            count += 1

    count = max(count, 1)
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
        rg = float(min(max(g12 / np.sqrt(max(g11 * g22, 1e-12)), -1.0), 1.0))
    pi_mean = pi_acc / count                     # posterior-mean 4-state mixture
    return BivariateResult(beta1_est=avg1 / count, beta2_est=avg2 / count,
                           h2=(float(h2_1), float(h2_2)), rg=rg,
                           p=float(pi_mean[1] + pi_mean[2] + pi_mean[3]),
                           sigma=np.array([[s1, s12], [s12, s2]]),
                           pi=pi_mean,
                           pi_samples=pi_samples[:count].copy(),
                           sigma_samples=sig_samples[:count].copy(),
                           noise_scale=(float(lam1), float(lam2)))


def ldpred3_auto_bivariate(corr, beta_hat1, beta_hat2, n_eff1, n_eff2, **kwargs):
    """Bivariate LDpred3-auto on a single dense LD matrix.

    Convenience wrapper over :func:`ldpred3_auto_bivariate_blocks` for one block
    (or a block-diagonal genome packed into one matrix). See that function and
    :class:`BivariateResult` for the parameters and output.
    """
    corr = np.ascontiguousarray(corr, dtype=np.float32)
    m = corr.shape[0]
    return ldpred3_auto_bivariate_blocks([(corr, np.arange(m))], beta_hat1,
                                         beta_hat2, n_eff1, n_eff2, **kwargs)
