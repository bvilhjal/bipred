"""Cross-trait (bivariate) LD Score regression for genetic correlation.

The two-trait counterpart of ldpred3's univariate ``ldsc_h2``. Fitting

    E[z1_j z2_j] = intercept + (sqrt(N1 N2) * rho_g / M) * ell_j

(with ``z_t = sqrt(N_t) beta_hat_t``) recovers the genetic covariance ``rho_g``
from the slope; the intercept captures correlated sampling error from sample
overlap as well as correlated confounding. The genetic correlation is
``r_g = rho_g / sqrt(h2_1 h2_2)``
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


def _as_finite_vector(value, name):
    """Return a nonempty, one-dimensional finite float array."""
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a one-dimensional numeric array") from None
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _as_sample_size(value, name, m):
    """Return a positive finite scalar sample size expanded to length m, or a vector."""
    if isinstance(value, (bool, np.bool_, str, bytes)):
        raise ValueError(f"{name} must be a positive finite scalar or length-m vector")
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite scalar or length-m vector") \
            from None
    if any(isinstance(x, (bool, np.bool_, str, bytes)) for x in raw.flat):
        raise ValueError(f"{name} must be a positive finite scalar or length-m vector")
    try:
        value = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite scalar or length-m vector") \
            from None
    if value.ndim == 0:
        value = np.full(m, float(value))
    elif value.shape != (m,):
        raise ValueError(f"{name} must be a scalar or length-m vector")
    if not np.all(np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError(f"{name} must contain only positive finite values")
    return value


def _as_finite_scalar(value, name, *, positive=False):
    """Validate and return a finite scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite scalar") from None
    if value.ndim != 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    value = float(value)
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _as_int(value, name, minimum):
    """Validate an integer control parameter without accepting booleans."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or int(value) < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass
class LDSCRgResult:
    """Output of :func:`ldsc_rg`."""

    rg: float                   # genetic correlation (can fall outside [-1,1] when noisy)
    rg_se: float                # block-jackknife standard error
    gcov: float                 # genetic covariance (cross-trait slope)
    gcov_intercept: float       # sample overlap and/or correlated confounding
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
    the slope; the intercept captures sample overlap and correlated confounding.
    The genetic correlation is
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
    n_blocks : int, default 200
        Number of contiguous delete-a-block jackknife blocks.
    n_iter : int, default 2
        Number of regression-weight update iterations.

    Returns
    -------
    LDSCRgResult
        ``rg`` and ``rg_se`` are NaN when either full-data marginal heritability
        is non-positive. For scientific conservatism, ``rg_se`` is NaN if any
        delete-block replicate has a non-positive marginal heritability or a
        singular fit, or if the jackknife has fewer than two blocks.
    """
    b1 = _as_finite_vector(beta_hat1, "beta_hat1")
    b2 = _as_finite_vector(beta_hat2, "beta_hat2")
    ell = _as_finite_vector(ld_scores, "ld_scores")
    m = b1.shape[0]
    if b2.shape != (m,) or ell.shape != (m,):
        raise ValueError("beta_hat1, beta_hat2, and ld_scores must have equal length")
    N1 = _as_sample_size(n_eff1, "n_eff1", m)
    N2 = _as_sample_size(n_eff2, "n_eff2", m)
    M = float(m) if m_snps is None else _as_finite_scalar(
        m_snps, "m_snps", positive=True)
    n_blocks = _as_int(n_blocks, "n_blocks", 1)
    n_iter = _as_int(n_iter, "n_iter", 0)
    if constrain_intercept is not None:
        constrain_intercept = _as_finite_scalar(
            constrain_intercept, "constrain_intercept")

    chi1 = N1 * b1 * b1
    chi2 = N2 * b2 * b2
    sqrt_n1n2 = np.sqrt(N1) * np.sqrt(N2)
    cross = sqrt_n1n2 * b1 * b2
    x1 = N1 * ell / M
    x2 = N2 * ell / M
    xc = sqrt_n1n2 * ell / M
    ell_w = np.maximum(ell, 1.0)

    def fit(sel):
        h1, i1 = _fit_slope(chi1[sel], x1[sel], ell_w[sel], n_iter, None)
        h2, i2 = _fit_slope(chi2[sel], x2[sel], ell_w[sel], n_iter, None)
        pred1 = np.maximum(i1 + h1 * x1[sel], 1.0)
        pred2 = np.maximum(i2 + h2 * x2[sel], 1.0)
        # For approximately bivariate-normal z scores,
        # Var(z1*z2) = E[z1^2] E[z2^2] + E[z1*z2]^2. This is the
        # Gencov.weights formula in the reference LDSC implementation; ell_w
        # supplies its LD-overcounting factor.
        pred_cross = np.full_like(
            cross[sel], 0.0 if constrain_intercept is None else constrain_intercept)
        for _ in range(n_iter + 1):
            variance = pred1 * pred2 + pred_cross * pred_cross
            w = 1.0 / (ell_w[sel] * np.maximum(variance, 1e-6))
            gcov, ic = _wls(xc[sel], cross[sel], w, constrain_intercept)
            pred_cross = ic + gcov * xc[sel]
        return h1, h2, gcov, ic

    full = np.ones(m, dtype=bool)
    try:
        h1, h2, gcov, ic = fit(full)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "LDSC regression is singular; LD scores must vary when intercepts "
            "are estimated") from exc

    if h1 <= 0.0 or h2 <= 0.0:
        rg = rg_se = float("nan")
        return LDSCRgResult(rg=rg, rg_se=rg_se, gcov=float(gcov),
                            gcov_intercept=float(ic), h2=(float(h1), float(h2)))

    rg = gcov / np.sqrt(h1 * h2)

    nb = int(min(n_blocks, m))
    rg_jk = []
    jackknife_valid = nb >= 2
    if nb >= 2:
        splits = np.array_split(np.arange(m), nb)
        for split in splits:
            keep = full.copy()
            keep[split] = False
            try:
                hb1, hb2, gb, _ = fit(keep)
            except np.linalg.LinAlgError:
                jackknife_valid = False
                break
            if hb1 <= 0.0 or hb2 <= 0.0:
                jackknife_valid = False
                break
            value = gb / np.sqrt(hb1 * hb2)
            if not np.isfinite(value):
                jackknife_valid = False
                break
            rg_jk.append(value)
    if not jackknife_valid:
        rg_se = float("nan")
    else:
        rg_jk = np.asarray(rg_jk)
        rg_se = float(np.sqrt(
            (nb - 1) / nb * np.sum((rg_jk - rg_jk.mean()) ** 2)))

    return LDSCRgResult(rg=float(rg), rg_se=rg_se, gcov=float(gcov),
                        gcov_intercept=float(ic), h2=(float(h1), float(h2)))


def estimate_sample_overlap(rg_result, n_eff1, n_eff2, pheno_corr=1.0):
    """Approximate shared-sample count from the cross-trait LDSC intercept.

    Overlapping GWAS samples make the two studies' sampling noise correlated.
    Under the strong assumption that the entire cross-trait intercept is caused
    by overlap,

        ``gcov_intercept ≈ N_shared · ρ_pheno / sqrt(N_eff1 · N_eff2)``

    where ``ρ_pheno`` is the phenotypic correlation among the shared individuals.
    Correlated population stratification or other confounding can also contribute
    to the intercept, so the intercept does not identify overlap by itself.

    The returned ``effective_overlap`` is the signed quantity
    ``N_shared * ρ_pheno`` under the overlap-only assumption. If the correlation
    is unknown but nonnegative, using ``pheno_corr=1.0`` gives a lower bound on
    ``N_shared`` (not an upper bound), provided the intercept really is entirely
    due to overlap.

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
        ``overlap_corr`` (the cross-trait intercept, retained for compatibility),
        ``effective_overlap`` (the signed intercept times ``sqrt(N1 N2)``),
        ``n_shared_raw`` (the overlap-only inversion), ``n_shared`` (the same
        estimate clipped at zero), and ``overlap_frac`` (``n_shared`` as a
        fraction of ``min(N1, N2)``).

    Notes
    -----
    The intercept is a genome-wide extrapolation, so a reliable absolute estimate
    needs many SNPs spanning a wide LD-score range (real GWAS scale); on a small
    panel it is noisy. Its sign and magnitude can signal correlated sampling
    error or confounding, but cannot specifically identify sample overlap.
    """
    n1 = _as_finite_scalar(n_eff1, "n_eff1", positive=True)
    n2 = _as_finite_scalar(n_eff2, "n_eff2", positive=True)
    rho = _as_finite_scalar(pheno_corr, "pheno_corr")
    if rho < -1.0 or rho > 1.0:
        raise ValueError("pheno_corr must lie in [-1, 1]")
    if rho == 0.0:
        raise ValueError("pheno_corr must be non-zero to solve for N_shared")
    if not isinstance(rg_result, LDSCRgResult):
        raise ValueError("rg_result must be an LDSCRgResult returned by ldsc_rg")
    overlap_corr = _as_finite_scalar(
        rg_result.gcov_intercept, "rg_result.gcov_intercept")
    effective_overlap = overlap_corr * float(np.sqrt(n1) * np.sqrt(n2))
    n_shared_raw = effective_overlap / rho
    n_shared = max(0.0, n_shared_raw)
    return {"overlap_corr": overlap_corr,
            "effective_overlap": effective_overlap,
            "n_shared_raw": n_shared_raw,
            "n_shared": n_shared,
            "overlap_frac": n_shared / min(n1, n2)}
