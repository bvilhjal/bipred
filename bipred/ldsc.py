"""Cross-trait (bivariate) LD Score regression for genetic correlation.

The two-trait counterpart of ldpred3's univariate ``ldsc_h2``. Fitting

    E[z1_j z2_j] = intercept + (sqrt(N1 N2) * rho_g / M) * ell_j

For standardized effects returned by ``ldpred3.standardize_betas``, the exact
signed relation is
``z_t = sqrt(N_t) beta_hat_t / sqrt(1 - beta_hat_t**2)``. The slope recovers the
genetic covariance ``rho_g``; the intercept captures correlated sampling error
from sample overlap as well as correlated confounding. The genetic correlation
is ``r_g = rho_g / sqrt(h2_1 h2_2)`` with the marginal heritabilities from
univariate LD Score regression.

This is the fast, moment-based cross-check on the bivariate-LDpred joint fit
(:func:`bipred.ldpred3_auto_bivariate`). It reuses ldpred3's univariate LDSC
machinery: LD scores come from ``ldpred3.ld_scores`` and the weighted-least-
squares / regression-weight helpers (``_wls`` / ``_weights``) are imported from
``ldpred3.ldsc`` so the two implementations stay a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from ._ldpred3_compat import (
    _as_n_vector,
    _finite_control,
    _integer_at_least,
    _weights,
    _wls,
)

__all__ = ["ldsc_rg", "LDSCRgResult", "estimate_sample_overlap"]


def _require_slope_information(x, constrain):
    """Raise when a slope is not identified by the selected design."""
    if x.size == 0:
        raise np.linalg.LinAlgError("LDSC regression has no observations")
    if constrain is None:
        if x.size < 2 or not np.any(x != x[0]):
            raise np.linalg.LinAlgError(
                "LDSC slope and intercept require variation in LD score times N")
    elif not np.any(x != 0.0):
        raise np.linalg.LinAlgError(
            "constrained-intercept LDSC requires a nonzero slope predictor")


def _fit_slope(y, x, ell_w, n_iter, constrain):
    """Iterated WLS of y on x with LDSC heteroscedasticity/overcounting weights."""
    _require_slope_information(x, constrain)
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
    """Strict ``n_eff``: reject strings/booleans outright, then the seam's check.

    ldpred3's ``_as_n_vector`` coerces numeric strings and booleans; bipred
    rejects them at the boundary, so the strict pre-check stays here and the
    scalar-or-length-m mechanics are delegated to the seam.
    """
    raw = np.asarray(value, dtype=object)
    if any(isinstance(x, (bool, np.bool_, str, bytes)) for x in raw.flat):
        raise ValueError(f"{name} must be a positive finite scalar or length-m vector")
    return _as_n_vector(value, m)


def _z_from_standardized(beta_std, n_eff, name):
    """Recover exact signed z scores from LDpred3-standardized effects."""
    if np.any(np.abs(beta_std) >= 1.0):
        raise ValueError(
            f"{name} must contain standardized effects with absolute value < 1")
    beta2 = beta_std * beta_std
    np.subtract(1.0, beta2, out=beta2)
    np.sqrt(beta2, out=beta2)
    z = np.sqrt(n_eff)
    z *= beta_std
    z /= beta2
    return z


def _as_finite_scalar(value, name, *, positive=False):
    """Validate and return a finite scalar.

    Used where strict positivity is required; bounds-free checks use
    ldpred3's ``_finite_control`` directly.
    """
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


@dataclass
class LDSCRgResult:
    """Output of :func:`ldsc_rg`."""

    rg: float                   # genetic correlation (can fall outside [-1,1] when noisy)
    rg_se: float                # block-jackknife standard error
    gcov: float                 # genetic covariance (cross-trait slope)
    gcov_intercept: float       # sample overlap and/or correlated confounding
    h2: tuple                   # (h2_1, h2_2) marginal heritabilities

    def __repr__(self):
        return (f"LDSCRgResult(rg={self.rg:+.3f} ± {self.rg_se:.3f}, "
                f"gcov={self.gcov:+.4f}, h2=({self.h2[0]:.3f}, {self.h2[1]:.3f}))")


def ldsc_rg(beta_hat1, beta_hat2, ld_scores, n_eff1, n_eff2, *, m_snps=None,
            n_blocks=200, n_iter=2, constrain_intercept=None):
    """Genetic correlation by cross-trait LD Score regression.

    Fits ``E[z1_j z2_j] = intercept + (sqrt(N1 N2) rho_g / M) ell_j``. For
    standardized effects returned by ``ldpred3.standardize_betas``, this uses the
    exact signed conversion
    ``z_t = sqrt(N_t) beta_hat_t / sqrt(1 - beta_hat_t**2)``. The slope gives the
    genetic covariance ``rho_g``; the intercept captures sample overlap and
    correlated confounding. The genetic correlation is
    ``r_g = rho_g / sqrt(h2_1 h2_2)`` with marginal heritabilities from
    univariate LD Score regression. Standard errors are by block jackknife.

    This is a **one-step, unfiltered** LDSC: every supplied variant enters the
    regression and there is no chi-square cap. The regression weights are built
    from the *fitted* means, so a variant whose observed chi-square far exceeds
    the LDSC line is not down-weighted and keeps near-full leverage. A handful
    of large-effect variants can therefore pull both the genetic covariance and
    the marginal heritabilities upward. The reference LDSC implementation
    excludes ``chi2 > max(0.001 N, 80)`` for this reason. Exclude long-range-LD
    regions (MHC, APOE) and cap extreme chi-square yourself before calling this
    when a screen needs to be robust to individual loci.

    All per-variant inputs must be aligned to the same variants in genomic
    order. This function receives no genomic coordinates, so it cannot verify
    or restore that order. A common row permutation leaves the point estimates
    unchanged but can change ``rg_se`` because jackknife blocks are contiguous
    rows.

    Parameters
    ----------
    beta_hat1, beta_hat2 : array_like (m,)
        Standardized marginal effects for the two traits, with absolute values
        strictly below one.
    ld_scores : array_like (m,)
        Strictly positive LD scores from ``ldpred3.ld_scores``.
    n_eff1, n_eff2 : float or array_like
        Per-trait GWAS sample sizes.
    m_snps : float, optional
        Number of variants over which the heritabilities and genetic covariance
        are defined. Defaults to the number of supplied summary-statistic rows.
        When those rows are a subset of a larger reference variant map, pass the
        full reference-map count so the LDSC slope uses the intended estimand.

        ``m_snps`` and ``ld_scores`` must describe the **same** variant map. The
        regression is ``x = N * ell / M``, whose model only holds when ``ell``
        sums r^2 over all ``M`` reference variants. ``ldpred3.ld_scores(blocks)``
        sums over exactly the blocks it is given, so building blocks from the
        summary-statistic subset and then passing the full reference count
        inflates both slopes. Either compute ``ell`` over the full reference
        blocks and subset the rows afterwards, or leave ``m_snps`` at its
        default so ``M`` and ``ell`` stay consistent.
    constrain_intercept : float, optional
        Fix the cross-trait intercept (e.g. ``0.0`` for non-overlapping samples).
    n_blocks : int, default 200
        Number of contiguous delete-a-block jackknife blocks in genomic order.
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
    if np.any(ell <= 0.0):
        raise ValueError("ld_scores must contain only positive values")
    N1 = _as_sample_size(n_eff1, "n_eff1", m)
    N2 = _as_sample_size(n_eff2, "n_eff2", m)
    M = float(m) if m_snps is None else _as_finite_scalar(
        m_snps, "m_snps", positive=True)
    n_blocks = _integer_at_least("n_blocks", n_blocks, 1)
    n_iter = _integer_at_least("n_iter", n_iter, 0)
    if constrain_intercept is not None:
        constrain_intercept = _finite_control(
            "constrain_intercept", constrain_intercept)

    z1 = _z_from_standardized(b1, N1, "beta_hat1")
    chi1 = z1 * z1
    z2 = _z_from_standardized(b2, N2, "beta_hat2")
    chi2 = z2 * z2
    cross = z1 * z2
    del z1, z2
    sqrt_n1n2 = np.sqrt(N1) * np.sqrt(N2)
    x1 = N1 * ell / M
    x2 = N2 * ell / M
    xc = sqrt_n1n2 * ell / M
    ell_w = np.maximum(ell, 1.0)

    def fit(sel):
        # Gather each selected column exactly once. The jackknife calls this nb
        # times, so repeating ``x1[sel]`` / ``ell_w[sel]`` inline would allocate
        # and copy several extra length-m arrays per delete-a-block replicate.
        ell_s = ell_w[sel]
        x1s, x2s, xcs, cross_s = x1[sel], x2[sel], xc[sel], cross[sel]
        h1, i1 = _fit_slope(chi1[sel], x1s, ell_s, n_iter, None)
        h2, i2 = _fit_slope(chi2[sel], x2s, ell_s, n_iter, None)
        _require_slope_information(xcs, constrain_intercept)
        pred1 = np.maximum(i1 + h1 * x1s, 1.0)
        pred2 = np.maximum(i2 + h2 * x2s, 1.0)
        # For approximately bivariate-normal z scores,
        # Var(z1*z2) = E[z1^2] E[z2^2] + E[z1*z2]^2. This is the
        # Gencov.weights formula in the reference LDSC implementation; ell_w
        # supplies its LD-overcounting factor.
        pred_cross = np.full_like(
            cross_s, 0.0 if constrain_intercept is None else constrain_intercept)
        for _ in range(n_iter + 1):
            variance = pred1 * pred2 + pred_cross * pred_cross
            w = 1.0 / (ell_s * np.maximum(variance, 1e-6))
            gcov, ic = _wls(xcs, cross_s, w, constrain_intercept)
            pred_cross = ic + gcov * xcs
        return h1, h2, gcov, ic

    full = np.arange(m)
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
        # ``array_split`` yields contiguous ranges, so delete-a-block is two
        # contiguous slices of the index vector rather than a length-m boolean
        # mask rebuilt (and re-scanned by every gather) for each of nb blocks.
        splits = np.array_split(full, nb)
        for split in splits:
            start = int(split[0])
            stop = int(split[-1]) + 1
            keep = np.concatenate((full[:start], full[stop:]))
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
    This scalar-effective-N inversion is an approximation, not a literal
    shared-person identity for case-control effective sizes, meta-analyses, or
    SNP-varying sample sizes.

    **The direct output is ``overlap_corr``**, the raw intercept. When
    ``cross_corr_valid`` is true it lies inside the joint fit's numeric domain
    and can be used as a ``cross_corr`` sensitivity value if treating the whole
    intercept as sampling-error correlation is scientifically defensible. The
    flag does not establish that assumption. On GLGC HDL x TG — the same
    individuals measured for both lipids — the intercept is −0.352, and fitting
    with ``cross_corr=0`` gave ``rg`` −0.90 against −0.52 with the sensitivity
    correction. The historical study used −0.5 to −0.6 as rough external
    context, not as ground truth. Neither fit warned; correlated sampling error
    can bias an otherwise stable fit.

    ``N_shared`` is a different and much weaker claim, because the intercept
    only identifies the *product* ``N_shared * ρ_pheno``. Splitting it needs
    ``ρ_pheno`` supplied from outside, and the default of 1.0 is a placeholder
    rather than a sensible guess for a negatively correlated pair.

    The returned ``effective_overlap`` is the signed quantity
    ``N_shared * ρ_pheno`` under the overlap-only assumption. If the correlation
    is unknown but nonnegative, using ``pheno_corr=1.0`` gives a lower bound on
    ``N_shared`` (not an upper bound), provided the intercept really is entirely
    due to overlap.

    When the intercept's sign disagrees with ``pheno_corr`` the inversion has no
    solution: a shared-sample count cannot be negative. That is a statement
    about ``ρ_pheno``, not about the overlap — most often the traits are
    negatively correlated and the default ``pheno_corr=1.0`` has the wrong sign.
    ``n_shared`` and ``overlap_frac`` are then ``nan`` and a warning is raised,
    because reporting zero would assert *no overlap* for data that may share
    every individual. They are likewise ``nan`` when the inversion exceeds the
    smaller cohort: a study cannot share more people than it contains. Earlier
    versions returned physically impossible counts in that case.

    A free LDSC intercept is not constrained to be a correlation. Sampling
    noise can therefore yield an estimate outside ``(-1, 1)``, whereas the
    joint fit requires ``cross_corr`` in that open interval. Such an intercept
    remains available as ``overlap_corr`` for diagnosis, but a warning is raised
    and ``cross_corr_valid`` is false; do not pass it to the joint fit unchanged.

    Parameters
    ----------
    rg_result : LDSCRgResult
        Output of :func:`ldsc_rg` (fit with a *free* intercept, the default).
    n_eff1, n_eff2 : float
        Scalar per-trait effective GWAS sample sizes used for the approximation.
    pheno_corr : float, default 1.0
        Phenotypic correlation among the shared samples (genetic + environmental).

    Returns
    -------
    dict
        ``overlap_corr`` (the raw cross-trait intercept),
        ``cross_corr_valid`` (whether that intercept lies in the joint fit's
        required open interval, not whether it is scientifically identified as
        sampling error), ``effective_overlap`` (the signed intercept times
        ``sqrt(N1 N2)``), ``n_shared_raw`` (the overlap-only inversion, which
        may be negative or too large), ``n_shared`` (that estimate, or ``nan``
        when it is physically impossible), ``overlap_frac`` (``n_shared`` over
        ``min(N1, N2)``, or ``nan``), ``sign_consistent`` (whether the intercept
        and ``pheno_corr`` agree in sign), and ``physically_consistent``
        (whether the overlap-only inversion obeys the sign, count, and
        correlation bounds).

    Notes
    -----
    The intercept is a genome-wide extrapolation, so a reliable absolute estimate
    needs many SNPs spanning a wide LD-score range (real GWAS scale); on a small
    panel it is noisy. Its sign and magnitude can signal correlated sampling
    error or confounding, but cannot specifically identify sample overlap.
    """
    n1 = _as_finite_scalar(n_eff1, "n_eff1", positive=True)
    n2 = _as_finite_scalar(n_eff2, "n_eff2", positive=True)
    rho = _finite_control("pheno_corr", pheno_corr)
    if rho < -1.0 or rho > 1.0:
        raise ValueError("pheno_corr must lie in [-1, 1]")
    if rho == 0.0:
        raise ValueError("pheno_corr must be non-zero to solve for N_shared")
    if not isinstance(rg_result, LDSCRgResult):
        raise ValueError("rg_result must be an LDSCRgResult returned by ldsc_rg")
    overlap_corr = _finite_control(
        "rg_result.gcov_intercept", rg_result.gcov_intercept)
    effective_overlap = overlap_corr * float(np.sqrt(n1) * np.sqrt(n2))
    n_shared_raw = effective_overlap / rho
    sign_consistent = n_shared_raw >= 0.0
    max_shared = min(n1, n2)
    count_consistent = sign_consistent and n_shared_raw <= max_shared
    cross_corr_valid = -1.0 < overlap_corr < 1.0
    physically_consistent = (
        count_consistent and -1.0 <= overlap_corr <= 1.0
    )
    if count_consistent:
        n_shared = n_shared_raw
        overlap_frac = n_shared / max_shared
    else:
        n_shared = overlap_frac = float("nan")
    if not sign_consistent:
        # No solution: a shared-sample count cannot be negative. Reporting 0
        # here would claim "no overlap" for a pair that may share every
        # individual -- which is what happened on GLGC HDL x TG, an intercept
        # of -0.352 between two lipids measured in the same people.
        warnings.warn(
            f"Cross-trait intercept {overlap_corr:+.4f} has the opposite sign "
            f"to pheno_corr {rho:+.4g}, so N_shared is not identified and is "
            "reported as nan rather than zero. The usual cause is a negatively "
            "correlated trait pair left on the default pheno_corr=1.0; supply "
            "the phenotypic correlation among the shared samples to invert it. "
            "overlap_corr remains available independently of pheno_corr.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif not count_consistent:
        warnings.warn(
            f"The overlap-only inversion is N_shared={n_shared_raw:.6g}, "
            f"which exceeds the smaller cohort ({max_shared:.6g}). The "
            "assumed pheno_corr and overlap-only model are physically "
            "inconsistent, so n_shared and overlap_frac are reported as nan.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not cross_corr_valid:
        warnings.warn(
            f"Cross-trait intercept {overlap_corr:+.6g} is outside the "
            "joint fit's valid cross_corr interval (-1, 1). It is retained as "
            "overlap_corr for diagnosis but must not be passed as cross_corr "
            "unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {"overlap_corr": overlap_corr,
            "cross_corr_valid": cross_corr_valid,
            "effective_overlap": effective_overlap,
            "n_shared_raw": n_shared_raw,
            "n_shared": n_shared,
            "overlap_frac": overlap_frac,
            "sign_consistent": sign_consistent,
            "physically_consistent": physically_consistent}
