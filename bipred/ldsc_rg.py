"""Cross-trait (bivariate) LD Score regression for genetic correlation.

The two-trait counterpart of ldpred3's univariate ``ldsc_h2``. Fitting

    E[z1_j z2_j] = intercept + (sqrt(N1 N2) * rho_g / M) * ell_j

(with ``z_t = sqrt(N_t) beta_hat_t``) recovers the genetic covariance ``rho_g``
from the slope; the intercept measures sample-overlap confounding (~0 for
independent GWAS). The genetic correlation is ``r_g = rho_g / sqrt(h2_1 h2_2)``
with the marginal heritabilities from univariate LD Score regression.

This is the fast, moment-based cross-check on the bivariate-LDpred joint fit
(:func:`bipred.ldpred3_auto_bivariate`). It reuses ldpred3's univariate LDSC
machinery: LD scores come from ``ldpred3.ld_scores`` and the weighted-least-
squares / regression-weight helpers (``_wls`` / ``_weights``) are imported from
``ldpred3.ldsc`` so the two implementations stay a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Shared univariate-LDSC internals (also used by ldpred3's own ``ldsc_h2``).
from ldpred3.ldsc import _wls, _weights

__all__ = ["ldsc_rg", "LDSCRgResult", "estimate_sample_overlap"]


def _fit_slope(y, x, ell_w, n_iter, constrain):
    """Iterated WLS of y on x with LDSC heteroscedasticity/overcounting weights."""
    pred = np.ones_like(y)
    slope = intercept = 0.0
    for _ in range(n_iter + 1):
        slope, intercept = _wls(x, y, _weights(pred, ell_w), constrain)
        pred = np.maximum(intercept + slope * x, 1.0)
    return slope, intercept


@dataclass
class LDSCRgResult:
    """Output of :func:`ldsc_rg`."""

    rg: float                   # genetic correlation (can fall outside [-1,1] when noisy)
    rg_se: float                # block-jackknife standard error
    gcov: float                 # genetic covariance (cross-trait slope)
    gcov_intercept: float       # ~0 without sample overlap
    h2: tuple                   # (h2_1, h2_2) marginal heritabilities

    @property
    def rg_ci(self):
        return (self.rg - 1.96 * self.rg_se, self.rg + 1.96 * self.rg_se)

    def __repr__(self):
        return (f"LDSCRgResult(rg={self.rg:+.3f} ± {self.rg_se:.3f}, "
                f"gcov={self.gcov:+.4f}, h2=({self.h2[0]:.3f}, {self.h2[1]:.3f}))")


def ldsc_rg(beta_hat1, beta_hat2, ld_scores, n_eff1, n_eff2, *, m_snps=None,
            n_blocks=200, n_iter=2, constrain_intercept=None):
    """Genetic correlation by cross-trait LD Score regression.

    Fits ``E[z1_j z2_j] = intercept + (sqrt(N1 N2) rho_g / M) ell_j`` (with
    ``z_t = sqrt(N_t) beta_hat_t``), giving the genetic covariance ``rho_g`` from
    the slope; the intercept measures sample-overlap confounding (~0 for
    independent GWAS). The genetic correlation is
    ``r_g = rho_g / sqrt(h2_1 h2_2)`` with the marginal heritabilities from
    univariate LD Score regression. Standard errors are by block jackknife.

    Parameters
    ----------
    beta_hat1, beta_hat2 : array_like (m,)
        Standardized marginal effects for the two traits (same variant order).
    ld_scores : array_like (m,)
        LD scores from ``ldpred3.ld_scores``.
    n_eff1, n_eff2 : float or array_like
        Per-trait GWAS sample sizes.
    constrain_intercept : float, optional
        Fix the cross-trait intercept (e.g. ``0.0`` for non-overlapping samples).

    Returns
    -------
    LDSCRgResult
        ``rg`` is undefined when a marginal heritability estimate is
        non-positive (low power / heavy noise); it is then large and not
        meaningful, as with the reference LDSC. Check ``.h2`` in that case.
    """
    b1 = np.asarray(beta_hat1, dtype=float)
    b2 = np.asarray(beta_hat2, dtype=float)
    ell = np.asarray(ld_scores, dtype=float)
    m = b1.shape[0]
    N1 = np.full(m, float(n_eff1)) if np.ndim(n_eff1) == 0 else np.asarray(n_eff1, float)
    N2 = np.full(m, float(n_eff2)) if np.ndim(n_eff2) == 0 else np.asarray(n_eff2, float)
    M = float(m_snps if m_snps is not None else m)

    chi1 = N1 * b1 * b1
    chi2 = N2 * b2 * b2
    cross = np.sqrt(N1 * N2) * b1 * b2
    x1 = N1 * ell / M
    x2 = N2 * ell / M
    xc = np.sqrt(N1 * N2) * ell / M
    ell_w = np.maximum(ell, 1.0)

    def fit(sel):
        h1, i1 = _fit_slope(chi1[sel], x1[sel], ell_w[sel], n_iter, None)
        h2, i2 = _fit_slope(chi2[sel], x2[sel], ell_w[sel], n_iter, None)
        pred1 = np.maximum(i1 + h1 * x1[sel], 1.0)
        pred2 = np.maximum(i2 + h2 * x2[sel], 1.0)
        # cross-trait regression weight ~ 1 / (ell * var(z1 z2)), var ~ pred1*pred2.
        w = 1.0 / (ell_w[sel] * np.maximum(pred1 * pred2, 1e-6))
        gcov, ic = _wls(xc[sel], cross[sel], w, constrain_intercept)
        return h1, h2, gcov, ic

    full = np.ones(m, dtype=bool)
    h1, h2, gcov, ic = fit(full)
    denom = np.sqrt(max(h1 * h2, 1e-12))
    rg = gcov / denom

    nb = int(min(n_blocks, m))
    splits = np.array_split(np.arange(m), nb)
    rg_jk = np.empty(nb)
    for b in range(nb):
        keep = full.copy()
        keep[splits[b]] = False
        hb1, hb2, gb, _ = fit(keep)
        rg_jk[b] = gb / np.sqrt(max(hb1 * hb2, 1e-12))
    rg_se = float(np.sqrt((nb - 1) / nb * np.sum((rg_jk - rg_jk.mean()) ** 2)))

    return LDSCRgResult(rg=float(rg), rg_se=rg_se, gcov=float(gcov),
                        gcov_intercept=float(ic), h2=(float(h1), float(h2)))


def estimate_sample_overlap(rg_result, n_eff1, n_eff2, pheno_corr=1.0):
    """Approximate shared-sample count from the cross-trait LDSC intercept.

    Overlapping GWAS samples make the two studies' sampling noise correlated,
    which shows up as the **cross-trait intercept** of :func:`ldsc_rg` — and,
    unlike the slope, it is (in expectation) independent of the genetic
    correlation, so it isolates the overlap:

        ``gcov_intercept ≈ N_shared · ρ_pheno / sqrt(N_eff1 · N_eff2)``

    where ``ρ_pheno`` is the phenotypic correlation among the shared individuals.
    Given ``ρ_pheno`` this inverts for ``N_shared``; with it unknown, pass
    ``pheno_corr=1.0`` and read the result as the **effective** overlap
    ``N_shared · ρ_pheno`` (an upper bound on ``N_shared`` when ``|ρ_pheno| ≤ 1``).

    Parameters
    ----------
    rg_result : LDSCRgResult
        Output of :func:`ldsc_rg` (fit with a *free* intercept, the default).
    n_eff1, n_eff2 : float
        Per-trait effective GWAS sample sizes.
    pheno_corr : float, default 1.0
        Phenotypic correlation among the shared samples (genetic + environmental).

    Returns
    -------
    dict
        ``overlap_corr`` (the intercept, = ``N_shared ρ_pheno / sqrt(N1 N2)``),
        ``n_shared`` (estimate, clipped at 0), and ``overlap_frac`` (``n_shared``
        as a fraction of ``min(N1, N2)``).

    Notes
    -----
    The intercept is a genome-wide extrapolation, so a reliable absolute estimate
    needs many SNPs spanning a wide LD-score range (real GWAS scale); on a small
    panel it is noisy. Its sign/magnitude relative to zero is the robust signal
    for *detecting* overlap.
    """
    if pheno_corr == 0:
        raise ValueError("pheno_corr must be non-zero to solve for N_shared")
    overlap_corr = float(rg_result.gcov_intercept)
    n_shared = overlap_corr * float(np.sqrt(n_eff1 * n_eff2)) / pheno_corr
    n_shared = max(0.0, n_shared)
    return {"overlap_corr": overlap_corr, "n_shared": n_shared,
            "overlap_frac": n_shared / min(n_eff1, n_eff2)}
