"""What bipred still owns of summary-statistic QC.

The LD-consistency screen used to live here, and its own docstring asked why:
"given the package otherwise refuses to touch summary statistics". It is a
single-trait check -- one GWAS against the LD reference it will be fitted
with -- so it now lives in :mod:`ldpred3.qc` beside the other DENTIST
schedule, and the univariate pipeline no longer reaches through the two-trait
package to run it. The names below are re-exported unchanged, so
``from bipred.qc import ld_consistency_screen`` keeps working.

What remains here is genuinely bipred's: the long-range-LD locus list used as
a sensitivity analysis for joint fits, and two file-level diagnostics
(:func:`sd_consistency`, :func:`implied_sample_size`) that predate LDpred3's
equivalents.
"""

from __future__ import annotations

import numpy as np

from ._ldpred3_compat import _finite_control, _validate_boolean_controls

# The screen, re-exported from its new home. Kept as a name-for-name alias
# rather than a wrapper so a caller cannot tell the difference, and so the
# private ``_ld_consistency_screen_selected`` that ``prepare`` uses resolves
# to the same object.
from ldpred3.qc import (                                            # noqa: F401
    DEFAULT_LOO_EIGENVALUE_FLOOR,
    DEFAULT_LOO_RIDGE,
    DEFAULT_MIN_NEIGHBOR_R,
    DEFAULT_PRIVATE_Z_RATIO,
    DEFAULT_ROUNDS,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    MIN_WINDOW,
    _confirmed_drops,
    _dentist_statistic,
    _ld_consistency_screen_selected,
    _precision_loo,
    _window_ld,
    dentist,
    dentist_statistic,
    ld_consistency_screen,
)
from ldpred3.qc import (                                            # noqa: F401
    DEFAULT_SPLIT_HALF_EIGENVALUE_FLOOR as DEFAULT_EIGENVALUE_FLOOR,
)
from ldpred3.qc import DEFAULT_DROP_FRACTION_WARN                   # noqa: F401
from ldpred3.qc import DEFAULT_UNTESTED_FRACTION_WARN               # noqa: F401

__all__ = ["ld_consistency_screen", "dentist", "dentist_statistic",
           "in_long_range_ld",
           "sd_consistency", "implied_sample_size",
           "LONG_RANGE_LD_HG19", "APOE_HG19"]

#: Long-range LD regions, GRCh37/hg19, as ``(chrom, start, end, label)``.
#:
#: The 24 regions of Price et al. 2008 (*Am J Hum Genet* 83:132-135): inversions
#: and other segments where correlation extends far beyond the usual few hundred
#: kilobases, so a block-diagonal or windowed LD approximation is weakest there.
#: Use the mask for estimator-specific sensitivity analysis, not as a universal
#: exclusion rule.
#: The MHC is only the most famous of them -- on real LDL x CAD data the full
#: list removed 6,461 variants against the MHC's 2,159.
#:
#: APOE is appended separately because it is *not* in Price et al. and is not a
#: long-range LD region in the same sense; it is here because it dominates lipid
#: genetics and a single locus of that effect size distorts a genome-wide fit.
LONG_RANGE_LD_HG19 = (
    ("1", 48_000_000, 52_000_000, "1p13"),
    ("2", 86_000_000, 100_500_000, "2p11 / 2q11"),
    ("2", 134_500_000, 138_000_000, "2q21"),
    ("2", 183_000_000, 190_000_000, "2q31"),
    ("3", 47_500_000, 50_000_000, "3p21"),
    ("3", 83_500_000, 87_000_000, "3p12"),
    ("3", 89_000_000, 97_500_000, "3q11"),
    ("5", 44_500_000, 50_500_000, "5p12 / 5q11"),
    ("5", 98_000_000, 100_500_000, "5q21"),
    ("5", 129_000_000, 132_000_000, "5q31"),
    ("5", 135_500_000, 138_500_000, "5q31.2"),
    ("6", 25_000_000, 35_000_000, "MHC"),
    ("6", 57_000_000, 64_000_000, "6p11 / 6q11"),
    ("6", 140_000_000, 142_500_000, "6q24"),
    ("7", 55_000_000, 66_000_000, "7p11 / 7q11"),
    ("8", 7_000_000, 13_000_000, "8p23 inversion"),
    ("8", 43_000_000, 50_000_000, "8p11 / 8q11"),
    ("8", 112_000_000, 115_000_000, "8q23"),
    ("10", 37_000_000, 43_000_000, "10p11 / 10q11"),
    ("11", 46_000_000, 57_000_000, "11p11 / 11q11"),
    ("11", 87_500_000, 90_500_000, "11q14"),
    ("12", 33_000_000, 40_000_000, "12p11 / 12q11"),
    ("12", 109_500_000, 112_000_000, "12q24"),
    ("20", 32_000_000, 34_500_000, "20p11 / 20q11"),
)
#: APOE, hg19. Optional sensitivity locus; inclusion depends on the target.
APOE_HG19 = ("19", 44_912_079, 45_912_079, "APOE")



def in_long_range_ld(chrom, pos, *, include_apoe=True, regions=None):
    """Mask of variants inside a long-range LD region (hg19).

    These regions are where a block-diagonal or windowed LD approximation is
    least accurate, so exclusion is a useful sensitivity analysis for
    genome-wide ``rg`` or ``h2``. It is not a universal repair: in the
    historical three-pair study, screened estimates moved by amounts ranging
    from 0.0001 to about 0.023 after exclusion.

    Exclusion also changes the prediction target. APOE is a strong lipid locus,
    so dropping it can discard predictive signal even when it stabilises a
    genome-wide moment. Since
    :class:`~bipred.bivariate.BivariateResult` carries both ``rg`` and
    ``beta1_est``/``beta2_est``, the right answer differs by what you are going
    to use, and the two uses may need two fits.

    Parameters
    ----------
    chrom : array_like of str
        Chromosome labels, without a ``chr`` prefix ("6", not "chr6").
    pos : array_like of int
        Base-pair positions on GRCh37/hg19. Passing another build silently
        mis-selects, because the coordinates are not validated against one.
    include_apoe : bool, default True
        Append :data:`APOE_HG19`, which is not part of Price et al.
    regions : sequence of (chrom, start, end, label), optional
        Override the region list entirely.

    Returns
    -------
    ndarray of bool
        ``True`` where the variant falls inside a listed region.
    """
    _validate_boolean_controls(include_apoe=include_apoe)
    chrom = np.asarray(chrom).astype(str)
    pos = np.asarray(pos, dtype=np.int64)
    if chrom.shape != pos.shape:
        raise ValueError("chrom and pos must have the same shape")
    if regions is None:
        regions = list(LONG_RANGE_LD_HG19)
        if include_apoe:
            regions = regions + [APOE_HG19]
    inside = np.zeros(chrom.shape, dtype=bool)
    for name, start, end, _label in regions:
        inside |= (chrom == str(name)) & (pos >= start) & (pos <= end)
    return inside


def _sd_from_sumstats(beta, se, n_eff, *, binary, normalise):
    """Genotype SD implied by the summary statistics, optionally rescaled.

    ``normalise`` divides by the 99th percentile against the ``sqrt(0.5)`` an
    ``af=0.5`` variant implies. LDpred2 does this only for quantitative traits,
    on the reasoning that a binary trait's effective sample size fixes the
    scale. That reasoning fails whenever the effective N is wrong -- genomic
    control inflates every standard error, and the pooled
    ``4/(1/n_case + 1/n_ctrl)`` formula overstates N for a meta-analysis of
    cohorts with differing case/control ratios. On CARDIoGRAMplusC4D CAD both
    apply, and the unnormalised ratio has median 0.755 rather than ~1, so a
    threshold calibrated on a well-specified trait removes 83% of the genome.
    Normalising makes the check scale-free and comparable across traits.
    """
    scale = 2.0 if binary else 1.0
    sd = scale / np.sqrt(np.asarray(n_eff, dtype=np.float64)
                         * np.asarray(se, dtype=np.float64) ** 2
                         + np.asarray(beta, dtype=np.float64) ** 2)
    if normalise:
        sd = sd / np.quantile(sd, 0.99) * np.sqrt(0.5)
    return sd


def sd_consistency(beta, se, n_eff, af, *, binary=False, lower=0.5, upper=0.1,
                   normalise=True):
    """LDpred2's SD check, with both trait types put on a common scale.

    Compares the genotype SD implied by ``beta``/``se``/``n_eff`` against the
    one implied by the reference allele frequency. It detects disagreement among
    those columns, but does not identify which input is wrong. In particular,
    quantitative-trait phenotype scale and absolute N are confounded unless the
    scale is supplied externally.

    The defaults follow the usual LDpred2-style check, but thresholds remain
    study- and reference-specific. The historical factorial changed warning
    separation under one UK Biobank reference; it does not identify a universal
    optimum or validate the current screen semantics. What this check cannot
    see is neighbourhood-level disagreement; compare
    :func:`ld_consistency_screen` for that diagnostic.

    Apply a MAF filter first. ``sd_ref`` is unstable at low frequency, and the
    normalisation is a quantile of whatever variants survive, so filtering
    first improves it.

    Returns
    -------
    (ndarray of bool, float)
        The keep mask, and the *unnormalised* median of
        ``sd_ss / sd_ref``. That second value is a diagnostic in its own
        right. For binary or externally standardised traits, a large departure
        can diagnose inconsistent N/SE scaling. For an otherwise unscaled
        quantitative trait it is only a relative scale diagnostic; see
        :func:`implied_sample_size`.
    """
    _validate_boolean_controls(binary=binary, normalise=normalise)
    beta = np.asarray(beta, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    af = np.asarray(af, dtype=np.float64)
    if (beta.ndim != 1 or se.ndim != 1 or af.ndim != 1
            or beta.size == 0 or se.shape != beta.shape or af.shape != beta.shape):
        raise ValueError("beta, se and af must be non-empty equal-length vectors")
    n_eff = np.asarray(n_eff, dtype=np.float64)
    if n_eff.ndim == 0:
        n_eff = np.full(beta.shape, float(n_eff))
    elif n_eff.shape != beta.shape:
        raise ValueError("n_eff must be a scalar or a vector matching beta")
    if not np.all(np.isfinite(beta)):
        raise ValueError("beta contains non-finite values; filter them first")
    if not np.all(np.isfinite(se)) or np.any(se <= 0):
        raise ValueError("se must contain finite positive values")
    if not np.all(np.isfinite(n_eff)) or np.any(n_eff <= 0):
        raise ValueError("n_eff must contain finite positive values")
    if not np.all(np.isfinite(af)) or np.any((af < 0) | (af > 1)):
        raise ValueError("af must contain finite values in [0, 1]")
    lower = _finite_control("lower", lower)
    upper = _finite_control("upper", upper)
    if not 0 < lower <= 1:
        raise ValueError("lower must be finite and in (0, 1]")
    if upper < 0:
        raise ValueError("upper must be finite and non-negative")
    sd_ref = np.sqrt(2.0 * af * (1.0 - af))
    raw = _sd_from_sumstats(beta, se, n_eff, binary=binary, normalise=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = float(np.nanmedian(np.where(sd_ref > 0, raw / sd_ref, np.nan)))
    sd_ss = _sd_from_sumstats(beta, se, n_eff, binary=binary,
                              normalise=normalise)
    bad = ((sd_ss < lower * sd_ref) | (sd_ss > sd_ref + upper)
           | (sd_ss < 0.1) | (sd_ref < 0.05))
    return ~bad, offset


def implied_sample_size(beta, se, af, *, binary=False, reported_n=None):
    """Effective sample size the summary statistics behave as if they had.

    Solving ``sd_ss == sd_ref`` for ``n`` gives, per variant,

        ``n = (c^2 / (2 f (1 - f)) - beta^2) / se^2``

    with ``c = 2`` for a case/control trait. The median over variants is the
    sample size the file is internally consistent with, which is not always the
    one it reports.

    This matters because ``n_eff`` enters the model directly through
    ``ldpred3.standardize_betas``. For the tested binary CARDIoGRAMplusC4D CAD
    file the ratio was 0.570: reported 162,973 against an implied 92,966. That
    comparison is identifiable because the case/control scale fixes ``c``;
    quantitative-trait comparisons do not provide the same check.

    Two causes produce this and cannot be separated from the file alone:
    genomic control inflating the standard errors (``se_dgc`` and similar), and
    the pooled ``4/(1/n_case + 1/n_ctrl)`` formula, which overstates the
    effective size of a meta-analysis unless every contributing cohort shares
    the same case/control ratio -- the correct form being the sum of per-cohort
    effective sizes. The implied value absorbs both without needing to know
    which applies.

    For a quantitative trait the phenotype scale is unknown. Neither an
    absolute effective N nor its ratio to ``reported_n`` is identifiable from
    ``beta``, ``se`` and allele frequency alone: calibrating the unknown scale
    from ``reported_n`` would force the ratio to one by construction. The
    function therefore returns ``nan`` for ``median`` and ``ratio``, and
    ``False`` for ``consistent``, when ``binary=False``. Use externally
    standardised effects or a trait-specific method when absolute quantitative
    N is required.

    Returns
    -------
    dict
        ``median`` (implied effective N), ``ratio`` (to ``reported_n``, or
        ``nan``), and ``consistent`` (ratio within 0.9 to 1.1). For a
        quantitative trait the first two are ``nan`` and the last is ``False``
        because the absolute scale is unidentified.
    """
    _validate_boolean_controls(binary=binary)
    beta = np.asarray(beta, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    af = np.asarray(af, dtype=np.float64)
    if (beta.ndim != 1 or se.ndim != 1 or af.ndim != 1
            or beta.size == 0 or se.shape != beta.shape or af.shape != beta.shape):
        raise ValueError("beta, se and af must be non-empty equal-length vectors")
    if not np.all(np.isfinite(beta)):
        raise ValueError("beta contains non-finite values; filter them first")
    if not np.all(np.isfinite(se)) or np.any(se <= 0):
        raise ValueError("se must contain finite positive values")
    if not np.all(np.isfinite(af)) or np.any((af <= 0) | (af >= 1)):
        raise ValueError("af must contain finite values strictly between 0 and 1")
    sd_ref = np.sqrt(2.0 * af * (1.0 - af))
    reported = None
    if reported_n is not None:
        reported = np.asarray(reported_n, dtype=np.float64)
        if reported.ndim == 0:
            reported = np.full(beta.shape, float(reported))
        elif reported.shape != beta.shape:
            raise ValueError(
                "reported_n must be a scalar or a vector matching beta")
        if not np.all(np.isfinite(reported)) or np.any(reported <= 0):
            raise ValueError("reported_n must contain finite positive values")
    if not binary:
        return {"median": float("nan"), "ratio": float("nan"),
                "consistent": False}
    c = 2.0
    implied = (c ** 2 / sd_ref ** 2 - beta ** 2) / se ** 2
    median = float(np.median(implied))
    ratio = float("nan")
    if reported is not None:
        reported_median = float(np.median(reported))
        ratio = median / reported_median
    return {"median": median, "ratio": ratio,
            "consistent": bool(np.isfinite(ratio) and 0.9 <= ratio <= 1.1)}
