"""Summary-statistic quality control against the LD reference you will fit with.

bipred does not harmonize summary statistics or build LD, and this module does
not change that. What it adds is the one check that cannot be done without the
LD: whether a variant's reported effect is *consistent with the variants
correlated with it*. Every filter a user can apply beforehand -- minor allele
frequency, imputation quality, a chi-square cap, per-variant sample size --
judges a variant in isolation. None of them can see that a variant disagrees
with its own neighbourhood, and that disagreement is what makes a bivariate
Gibbs sampler place large opposing effects on variants in near-perfect LD.

The check is the one introduced by DENTIST (Chen et al. 2021,
*Nature Communications* 12:7117). Within a window, split the variants at random
into two halves and predict each z-score in one half from the other half
through the LD::

    zhat_a = R[a,B] pinv(R[B,B]) z_B
    T_a    = (z_a - zhat_a)^2 / (1 - R[a,B] pinv(R[B,B]) R[B,a])   ~ chi2_1

Variants whose observed z is far from what their neighbours predict are
dropped, and the split is repeated so each variant is tested from several
directions.

Running the statistic against the blocks you will fit with is deliberate, not a
simplification of the published tool. An inconsistency only means anything
relative to the LD the model will actually use; testing against a third-party
panel measures a matrix the sampler never sees.

Why this is in bipred at all, given the package otherwise refuses to touch
summary statistics: a bivariate fit tolerates far less LD inconsistency than a
univariate one. On a real LDL x CAD analysis, ldpred3's univariate sampler
consumed entirely unfiltered summary statistics without trouble
(``sum(beta^2)`` 0.22) while the bivariate fit on the identical blocks diverged
(``sum(beta^2)`` 157.5, posterior means 110 times the slab SD it had itself
inferred, genetic variance still climbing at the last iteration). Removing the
41,775 LD-inconsistent variants this module finds -- 4.7% of 887,361 -- moved
that fit from divergent to converged, with r_g going from +0.12 to +0.28
against a cross-trait LDSC screen of +0.19.

Typical use, before either :func:`bipred.ldpred3_auto_bivariate_blocks` or a
univariate fit::

    from bipred.qc import dentist

    keep = dentist(blocks, beta_hat1 / se1) & dentist(blocks, beta_hat2 / se2)
    # then subset blocks and both traits to `keep` before fitting
"""

from __future__ import annotations

import numpy as np

from ldpred3 import LowRankLD

__all__ = ["dentist", "dentist_statistic", "in_long_range_ld",
           "sd_consistency", "implied_sample_size",
           "LONG_RANGE_LD_HG19", "APOE_HG19"]

#: Variants per window. The split-half uses about half of this a side, so the
#: pseudo-inverse stays small enough to be cheap while the window still spans
#: more LD than any realistic correlation reaches.
DEFAULT_WINDOW = 1000
#: Windows below this are skipped: the split-half has too few variants a side
#: for the prediction to mean anything.
MIN_WINDOW = 50
#: Eigenvalues below this fraction of the largest are dropped rather than
#: inverted. Inverting a near-null direction is exactly how an ill-conditioned
#: block manufactures an enormous prediction, which would make this test
#: generate the pathology it exists to detect.
DEFAULT_EIGENVALUE_FLOOR = 1e-3
#: chi2_1 at p = 5e-8, DENTIST's own default.
DEFAULT_THRESHOLD = 29.72
DEFAULT_ROUNDS = 4

#: Long-range LD regions, GRCh37/hg19, as ``(chrom, start, end, label)``.
#:
#: The 24 regions of Price et al. 2008 (*Am J Hum Genet* 83:132-135), which are
#: the conventional exclusion list for anything that models LD: inversions and
#: other segments where correlation extends far beyond the usual few hundred
#: kilobases, so a block-diagonal or windowed LD approximation is worst there.
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
#: APOE, hg19. Excluded for genome-wide *estimation*; keep it for prediction.
APOE_HG19 = ("19", 44_912_079, 45_912_079, "APOE")


def _window_ld(block, local):
    """Dense LD submatrix for ``local`` positions inside one block.

    A low-rank block is never densified in full: the submatrix of
    ``U U' + diag(d)`` is ``U[w] U[w]' + diag(d[w])``, exact for that
    representation and cheap at window scale, so a 12,000-variant block costs
    no more here than any other.
    """
    if isinstance(block, LowRankLD):
        factor = block.U[local].astype(np.float64) * (block.scale or 1.0)
        out = factor @ factor.T
        out[np.diag_indices(len(local))] += np.asarray(
            block.residual_diag, dtype=np.float64)[local]
        return out
    return np.asarray(block, dtype=np.float64)[np.ix_(local, local)]


def dentist_statistic(ld, z, predictors, targets, *,
                      eigenvalue_floor=DEFAULT_EIGENVALUE_FLOOR):
    """DENTIST ``T`` for ``targets``, predicted from ``predictors``.

    ``ld`` is a dense correlation submatrix, ``z`` the matching z-scores, and
    the two index arrays are disjoint positions into both. Returns one
    chi2_1-distributed statistic per target.
    """
    ld = np.asarray(ld, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    within = ld[np.ix_(predictors, predictors)]
    across = ld[np.ix_(targets, predictors)]
    values, vectors = np.linalg.eigh(within)
    keep = values > eigenvalue_floor * max(float(values.max()), 1e-12)
    if not keep.any():
        return np.zeros(len(targets))
    retained = vectors[:, keep]
    scaled = retained / values[keep]                  # V diag(1/lambda)
    predicted = across @ (scaled @ (retained.T @ z[predictors]))
    # 1 - r' pinv(R) r, per target, without ever forming pinv(R).
    leverage = np.einsum("ij,ij->i", across @ scaled, across @ retained)
    return (z[targets] - predicted) ** 2 / np.clip(1.0 - leverage, 1e-6, None)


def dentist(blocks, z, *, rounds=DEFAULT_ROUNDS, window=DEFAULT_WINDOW,
            threshold=DEFAULT_THRESHOLD,
            eigenvalue_floor=DEFAULT_EIGENVALUE_FLOOR, seed=0, verbose=False):
    """Boolean keep-mask over the variants ``blocks`` spans.

    Parameters
    ----------
    blocks : list of (R, idx)
        The same blocks you will fit with, indices partitioning ``0..m-1``.
    z : array_like (m,)
        Z-scores for one trait, ``beta / se``, in the blocks' variant order.
        Run this once per trait and intersect the masks.
    rounds : int
        Passes with fresh random splits. Outliers are removed as they are
        found, so a later pass can see variants that were masked by a bad
        neighbour in an earlier one. Stops early when a pass drops nothing.
    window, threshold, eigenvalue_floor, seed
        See the module constants.

    Returns
    -------
    ndarray of bool
        ``True`` for variants to keep. Counts per round go to stdout under
        ``verbose``.
    """
    total = sum(len(idx) for _, idx in blocks)
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (total,):
        raise ValueError(
            f"z has {z.shape} entries but the blocks span {total} variants")
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values; filter them first")
    rng = np.random.default_rng(seed)
    keep = np.ones(total, dtype=bool)
    for round_no in range(int(rounds)):
        dropped = 0
        for block, idx in blocks:
            live = np.where(keep[idx])[0]
            if live.size < MIN_WINDOW:
                continue
            for start in range(0, live.size, window):
                local = live[start:start + window]
                if local.size < MIN_WINDOW:
                    continue
                ld = _window_ld(block, local)
                zw = z[idx[local]]
                order = rng.permutation(local.size)
                half = local.size // 2
                first, second = order[:half], order[half:]
                for targets, predictors in ((first, second), (second, first)):
                    stat = dentist_statistic(
                        ld, zw, predictors, targets,
                        eigenvalue_floor=eigenvalue_floor)
                    bad = targets[stat > threshold]
                    if bad.size:
                        keep[idx[local[bad]]] = False
                        dropped += int(bad.size)
        if verbose:
            print(f"  dentist round {round_no + 1}: dropped {dropped:,}, "
                  f"{keep.sum():,} remain", flush=True)
        if dropped == 0:
            break
    return keep


def in_long_range_ld(chrom, pos, *, include_apoe=True, regions=None):
    """Mask of variants inside a long-range LD region (hg19).

    Exclude these before estimating ``rg`` or ``h2``: they are where a
    block-diagonal or windowed LD approximation is least accurate, so they
    contribute the largest discrepancies between the summary statistics and the
    reference. On real LDL x CAD data the full list removed 6,461 variants,
    three times what the MHC alone accounted for.

    **Do not exclude them when the output is a polygenic score.** APOE is the
    strongest lipid locus in the genome; dropping it improves a genome-wide
    variance-component estimate and throws away real predictive signal. Since
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
    one implied by the reference allele frequency. Catches a wrong sample size,
    a wrong standard error or a wrong frequency -- combinations that no single
    threshold can see, because each number is individually plausible and only
    their product is impossible.

    Keep the published thresholds. Tightening them is measurably worthless: on
    real LDL x CAD data, ``lower=0.8, upper=0.03`` removed a further 142,282
    variants and moved the fit's cancellation ratio from 264.8 to 276.1, which
    is to say it removed a fifth of the genome and made the answer slightly
    worse. What the check cannot do is detect an inconsistency that is not a
    property of any single variant; that is :func:`dentist`'s job.

    Apply a MAF filter first. ``sd_ref`` is unstable at low frequency, and the
    normalisation is a quantile of whatever variants survive, so filtering
    first improves it.

    Returns
    -------
    (ndarray of bool, float)
        The keep mask, and the *unnormalised* median of
        ``sd_ss / sd_ref``. That second value is a diagnostic in its own
        right: ~1.0 means the reported sample size and standard errors are
        mutually consistent, and a large departure means they are not. See
        :func:`implied_sample_size`.
    """
    af = np.asarray(af, dtype=np.float64)
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
    ``ldpred3.standardize_betas``. On four real GWAS the implied and reported
    values agreed to within 1%. On CARDIoGRAMplusC4D CAD the ratio was 0.570 --
    reported 162,973 against an implied 92,966 -- and fitting the reported
    value understated that trait's h2 by the same factor: 0.0401 became 0.0706.
    Cross-trait LDSC moves with it, since ``n_eff`` scales its estimate too, so
    the correction does not reconcile the two -- LDSC on the same corrected data
    gives 0.1205, and bipred's 0.0706 remains 0.59x of it.

    Two causes produce this and cannot be separated from the file alone:
    genomic control inflating the standard errors (``se_dgc`` and similar), and
    the pooled ``4/(1/n_case + 1/n_ctrl)`` formula, which overstates the
    effective size of a meta-analysis unless every contributing cohort shares
    the same case/control ratio -- the correct form being the sum of per-cohort
    effective sizes. The implied value absorbs both without needing to know
    which applies.

    For a quantitative trait the phenotype scale is unknown, so only the
    *ratio* to ``reported_n`` is meaningful; ``c`` is calibrated from the median
    when ``reported_n`` is given, and the returned median is then reported on
    that calibrated scale.

    Returns
    -------
    dict
        ``median`` (implied effective N), ``ratio`` (to ``reported_n``, or
        ``nan``), and ``consistent`` (ratio within 0.9 to 1.1).
    """
    beta = np.asarray(beta, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    af = np.asarray(af, dtype=np.float64)
    sd_ref = np.sqrt(2.0 * af * (1.0 - af))
    usable = (sd_ref > 0) & np.isfinite(se) & (se > 0) & np.isfinite(beta)
    if not usable.any():
        raise ValueError("no usable variants: need finite beta, se > 0, 0 < af < 1")
    beta, se, sd_ref = beta[usable], se[usable], sd_ref[usable]
    if binary:
        c = 2.0
    elif reported_n is not None:
        n = np.broadcast_to(np.asarray(reported_n, dtype=np.float64),
                            usable.shape)[usable]
        c = float(np.median(sd_ref * np.sqrt(n * se ** 2 + beta ** 2)))
    else:
        c = 1.0
    implied = (c ** 2 / sd_ref ** 2 - beta ** 2) / se ** 2
    median = float(np.median(implied))
    ratio = float("nan")
    if reported_n is not None:
        reported_median = float(np.median(
            np.broadcast_to(np.asarray(reported_n, dtype=np.float64),
                            usable.shape)[usable]))
        if reported_median > 0:
            ratio = median / reported_median
    return {"median": median, "ratio": ratio,
            "consistent": bool(np.isfinite(ratio) and 0.9 <= ratio <= 1.1)}
