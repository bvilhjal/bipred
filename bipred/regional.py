"""Regional (per-locus) genetic correlation from a fitted bivariate model.

`ldpred3_auto_bivariate_blocks` reports one genome-wide `rg`. This module turns
the same fit into a **per-region** genetic correlation, by restricting the
LD-aware quadratic forms to each region's variants:

    rg_r = (b1' R b2)_r / sqrt( (b1' R b1)_r * (b2' R b2)_r )

evaluated on the posterior-mean effects. LD is block-diagonal, so a region's
quadratic is the sum of its within-block contributions and no cross-block terms
are dropped. All three LD representations bipred accepts are supported without
materialising a dense matrix: dense float32, dense int8 (dequantised on the fly),
and `LowRankLD` factors, for which

    (a' R b)_sub = (W_sub' a)·(W_sub' b) + sum_i residual_i a_i b_i .

Posterior-mean effects are used deliberately rather than the sampled-quadratic
ratio that the genome-wide `rg` uses. The sampled ratio inflates its denominator
with posterior noise, which matters more per region than genome-wide because a
region has far fewer variants; in the benchmarks behind this module the
posterior-mean estimator had roughly half the RMSE at null and strong regions
alike.

**Two biases are known and are not corrected here.** Read them before
interpreting output; both are quantified in
`research/cross_corr_estimation/RESULTS_REGIONAL.md`.

1. *Sample overlap contaminates every region identically.* If the two GWAS share
   samples and `cross_corr` was not supplied to the fit, the same spurious
   covariance is added to every region at once — in simulation, regions whose
   true rg is zero read about **0.26**. It does not average out across regions
   and cannot be estimated within one, because a region has too few variants.
   Supply `cross_corr` to the fit whenever the cohorts may overlap.
2. *Regional estimates are shrunk toward the genome-wide correlation.* The
   sampler carries a single effect covariance for the whole genome, so every
   per-SNP posterior borrows across traits at the genome-wide rate. Strong
   regions are attenuated and null regions pulled up (about -0.07 and +0.06
   respectively in simulation). This is a property of reading regional structure
   out of a genome-wide model; it is unaffected by `cross_corr`, and it does not
   diminish with larger regions.

Consequently these estimates are more trustworthy for *ranking and comparing*
regions than as calibrated absolute values.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ldpred3.ldpred3 import LowRankLD, _validate_blocks

from .bivariate import _prepare_lowrank_block

__all__ = ["RegionalRgResult", "regional_rg"]

_Q8_SCALE = 1.0 / 127.0


@dataclass
class RegionalRgResult:
    """Per-region genetic correlation and the quadratics it is built from.

    ``region`` holds the region labels in first-appearance order; every other
    array is aligned to it. ``gvar1``/``gvar2`` are the regions' LD-aware genetic
    variances and ``gcov`` their genetic covariance, so a caller may re-derive
    ``rg`` or aggregate regions without refitting. ``rg`` is NaN where a region's
    variance is non-positive (possible with int8-quantised LD, which is not
    guaranteed positive definite) or where the region has no variants.
    """

    region: np.ndarray
    rg: np.ndarray
    gcov: np.ndarray
    gvar1: np.ndarray
    gvar2: np.ndarray
    n_variants: np.ndarray

    def __len__(self):
        return int(self.region.size)


def _dense_quadratics(block, scale, sub, b1, b2):
    """Three quadratic forms on one dense sub-block, dequantising if int8."""
    sl = block[np.ix_(sub, sub)]
    R = np.asarray(sl, dtype=np.float64)
    if scale != 1.0:
        R = R * scale
    x, y = b1[sub], b2[sub]
    Rx, Ry = R @ x, R @ y
    return float(x @ Rx), float(x @ Ry), float(y @ Ry)


def _lowrank_quadratics(W, residual, sub, b1, b2):
    """Three quadratic forms on a low-rank sub-block, without densifying it."""
    Ws = W[sub]
    x, y = b1[sub], b2[sub]
    px, py = Ws.T @ x, Ws.T @ y
    res = residual[sub]
    return (float(px @ px + np.sum(res * x * x)),
            float(px @ py + np.sum(res * x * y)),
            float(py @ py + np.sum(res * y * y)))


def regional_rg(beta1, beta2, blocks, regions, *, min_variants=1,
                allow_legacy_lowrank=False, clip=True):
    """Per-region genetic correlation from posterior-mean effects.

    Parameters
    ----------
    beta1, beta2 : array_like
        Posterior-mean standardised effects for the two traits, length ``m``.
        These are :attr:`BivariateResult.beta1_est` / ``beta2_est``.
    blocks : sequence
        The same LD blocks passed to the fit: ``(R, idx)`` pairs where ``R`` is a
        dense float/int8 matrix or a :class:`LowRankLD` factor.
    regions : array_like
        Length-``m`` region label per variant. Labels may be integers or strings;
        variants sharing a label form one region, and regions need not be
        contiguous. Use ``None``-free labels — every variant must be assigned.
    min_variants : int, default 1
        Regions with fewer variants than this report NaN ``rg``. Small regions
        are noisy; raising this is a convenience, not a correction.
    allow_legacy_lowrank : bool, default False
        Forwarded to the low-rank adapter, matching the fit's own flag.
    clip : bool, default True
        Clip ``rg`` into ``[-1, 1]``. Values outside it are possible only through
        non-positive-definite LD (int8 quantisation); clipping hides the symptom,
        so set ``False`` when diagnosing.

    Returns
    -------
    RegionalRgResult

    Notes
    -----
    See the module docstring: uncorrected sample overlap inflates **every**
    region, and all regional estimates are shrunk toward the genome-wide
    correlation. Neither is corrected here.
    """
    b1 = np.asarray(beta1, dtype=np.float64).ravel()
    b2 = np.asarray(beta2, dtype=np.float64).ravel()
    if b1.shape != b2.shape:
        raise ValueError(
            f"beta1 and beta2 must have the same length; got {b1.size} and "
            f"{b2.size}")
    if b1.size == 0:
        raise ValueError("beta1 and beta2 must be non-empty")
    if not (np.all(np.isfinite(b1)) and np.all(np.isfinite(b2))):
        raise ValueError("beta1 and beta2 must contain only finite values")
    m = b1.size

    if isinstance(min_variants, bool) or not isinstance(min_variants, (int, np.integer)):
        raise TypeError("min_variants must be an integer")
    min_variants = int(min_variants)
    if min_variants < 1:
        raise ValueError("min_variants must be >= 1")

    labels = np.asarray(regions).ravel()
    if labels.size != m:
        raise ValueError(
            f"regions must have one label per variant; got {labels.size} for "
            f"{m} variants")

    uniq, inverse = np.unique(labels, return_inverse=True)
    # report in first-appearance order rather than sorted label order
    first = np.full(uniq.size, m, dtype=np.int64)
    for pos, code in enumerate(inverse):
        if first[code] == m:
            first[code] = pos
    order = np.argsort(first, kind="stable")
    remap = np.empty(uniq.size, dtype=np.int64)
    remap[order] = np.arange(uniq.size)
    code = remap[inverse]
    region_ids = uniq[order]
    n_reg = uniq.size

    gvar1 = np.zeros(n_reg)
    gcov = np.zeros(n_reg)
    gvar2 = np.zeros(n_reg)
    counts = np.bincount(code, minlength=n_reg).astype(np.int64)

    for R, idx in _validate_blocks(blocks, m, contiguous=True):
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if isinstance(R, LowRankLD):
            U, row_scales, residual = _prepare_lowrank_block(
                R, allow_legacy=allow_legacy_lowrank)
            W = row_scales[:, None] * np.asarray(U, dtype=np.float64)
            dense = None
        else:
            arr = np.asarray(R)
            scale = _Q8_SCALE if arr.dtype == np.int8 else 1.0
            dense = (arr, scale)
            W = residual = None

        blk_codes = code[idx]
        for c in np.unique(blk_codes):
            sub = np.flatnonzero(blk_codes == c)
            if dense is not None:
                q11, q12, q22 = _dense_quadratics(dense[0], dense[1], sub,
                                                  b1[idx], b2[idx])
            else:
                q11, q12, q22 = _lowrank_quadratics(W, residual, sub,
                                                    b1[idx], b2[idx])
            gvar1[c] += q11
            gcov[c] += q12
            gvar2[c] += q22

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt(gvar1 * gvar2)
        rg = np.where(denom > 0.0, gcov / denom, np.nan)
    rg[counts < min_variants] = np.nan
    if clip:
        rg = np.clip(rg, -1.0, 1.0)

    return RegionalRgResult(region=region_ids, rg=rg, gcov=gcov, gvar1=gvar1,
                            gvar2=gvar2, n_variants=counts)
