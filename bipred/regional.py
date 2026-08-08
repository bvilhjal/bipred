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

The calculation uses the LD representation supplied to :func:`regional_rg`; it
does not replay the fit's ``ld_int8`` policy. Under the default ``ld_int8=False``
a fit preserves D8/D32 values, copying non-contiguous storage when needed, and
normalises other dense floats and floating low-rank factors to D32. This
function makes the same low-rank normalisation; exact dense alignment requires
D8/D32 inputs or passing the same normalised representation to both calls.
In-fit quantisation (``ld_int8=True`` or ``None``) creates a private copy for
float blocks.

Posterior-mean effects are used deliberately rather than the sampled-quadratic
ratio that the genome-wide `rg` uses. The sampled ratio inflates its denominator
with posterior noise, which matters more per region than genome-wide because a
region has far fewer variants.

**Two biases are known and are not corrected here.** Read them before
interpreting output; see `docs/rg.md` for guidance.

1. *Sample overlap contaminates every region.* If the two GWAS share samples
   and `cross_corr` was not supplied to the fit, spurious covariance affects
   every region, with magnitude depending on local LD, N, and shrinkage. It
   does not average out across regions and cannot be estimated reliably within
   one. Supply `cross_corr` to the fit whenever the cohorts may overlap.
2. *Regional estimates are shrunk toward the genome-wide correlation.* The
   sampler carries a single effect covariance for the whole genome, so every
   per-SNP posterior borrows across traits at the genome-wide rate. This is a
   property of reading regional structure out of a genome-wide model; it is
   unaffected by `cross_corr`.

Consequently these estimates are more trustworthy for *ranking and comparing*
regions than as calibrated absolute values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ldpred3 import LowRankLD

from ._ldpred3_compat import (
    _Q8,
    _validate_blocks,
    _validate_boolean_controls,
)

__all__ = ["RegionalRgResult", "regional_rg"]

_Q8_SCALE = 1.0 / _Q8
_DENSE_ROW_CHUNK = 256


@dataclass
class RegionalRgResult:
    """Per-region genetic correlation and the quadratics it is built from.

    ``region`` holds the region labels in first-appearance order; every other
    array is aligned to it. ``gvar1``/``gvar2`` are the regions' LD-aware genetic
    variances and ``gcov`` their genetic covariance, so a caller may re-derive
    ``rg`` without refitting. ``rg`` is NaN where either evaluated variance is
    non-positive or the region has fewer than ``min_variants`` variants.

    Summing these quantities across regions is **not** the quadratic of their
    union unless the regions lie in different LD blocks. Each region's
    quadratic covers only its own variants, so merging two regions inside one
    block drops the cross-region LD terms between them, and on strongly
    correlated variants those terms carry a large share of the union's value.
    To aggregate, re-run :func:`regional_rg` with the merged labels.
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
    # Region labels are commonly contiguous. In that case a basic slice is a
    # view. Interleaved regions are multiplied in bounded row slabs, avoiding a
    # len(sub)^2 gather. Neither path upcasts a quadratic-size array: a 12,169-
    # variant float32 block would otherwise need a 1.10 GiB float64 temporary.
    contiguous = sub.size == 1 or np.all(np.diff(sub) == 1)
    x, y = b1[sub], b2[sub]
    if contiguous:
        start = int(sub[0])
        sl = block[start:start + sub.size, start:start + sub.size]
        # Mixed-dtype ``@`` materialises a float64 copy of ``sl`` in NumPy.
        # Einsum accumulates in float64 with only O(len(sub)) outputs.
        Rx = np.einsum("ij,j->i", sl, x, dtype=np.float64,
                       optimize=False)
        Ry = np.einsum("ij,j->i", sl, y, dtype=np.float64,
                       optimize=False)
    else:
        Rx = np.empty(sub.size, dtype=np.float64)
        Ry = np.empty(sub.size, dtype=np.float64)
        for lo in range(0, sub.size, _DENSE_ROW_CHUNK):
            hi = min(lo + _DENSE_ROW_CHUNK, sub.size)
            rows = sub[lo:hi]
            slab = block[np.ix_(rows, sub)]
            Rx[lo:hi] = np.einsum(
                "ij,j->i", slab, x, dtype=np.float64, optimize=False)
            Ry[lo:hi] = np.einsum(
                "ij,j->i", slab, y, dtype=np.float64, optimize=False)
    return (scale * float(x @ Rx), scale * float(x @ Ry),
            scale * float(y @ Ry))


def _lowrank_quadratics(U, scale, residual, sub, b1, b2):
    """Three quadratic forms on a low-rank sub-block, without densifying it."""
    Us = np.asarray(U[sub], dtype=np.float64)
    x, y = b1[sub], b2[sub]
    px = scale * (Us.T @ x)
    py = scale * (Us.T @ y)
    res = residual[sub]
    return (float(px @ px + np.sum(res * x * x)),
            float(px @ py + np.sum(res * x * y)),
            float(py @ py + np.sum(res * y * y)))


def regional_rg(beta1, beta2, blocks, regions, *, min_variants=1, clip=True):
    """Per-region genetic correlation from posterior-mean effects.

    Parameters
    ----------
    beta1, beta2 : array_like
        One-dimensional posterior-mean standardised effects for the two traits,
        length ``m``. These are :attr:`BivariateResult.beta1_est` /
        ``beta2_est``.
    blocks : sequence
        ``(R, idx)`` pairs where ``R`` is a dense float/int8 matrix or a
        :class:`LowRankLD` factor, normally the same logical blocks passed to the
        fit. This function evaluates the representation supplied here. With the
        default ``ld_int8=False``, the fit preserves D8/D32 values, copying
        non-contiguous storage when needed, but normalises other dense floats
        and floating low-rank factors to D32. This function makes the same
        low-rank normalisation; exact dense alignment requires supplying D8/D32
        to both calls. In-fit quantisation (``ld_int8=True`` or ``None``)
        creates a private copy for float blocks; quantise the blocks yourself
        and pass those to both calls instead.
    regions : array_like
        One-dimensional length-``m`` region label per variant. Labels may be
        integers or strings; variants sharing a label form one region, and
        regions need not be contiguous. ``None`` is not a valid label because
        every variant must be assigned.
    min_variants : int, default 1
        Regions with fewer variants than this report NaN ``rg``. Small regions
        are noisy; raising this is a convenience, not a correction.
    clip : bool, default True
        Clip each finite raw ratio into ``[-1, 1]``. A raw ``|rg| > 1`` violates
        Cauchy--Schwarz for the evaluated quadratic form and usually indicates a
        non-positive-semidefinite regional LD submatrix, for example after int8
        quantisation, malformed input, or numerical error. Clipping does not
        repair the underlying ``gvar1``, ``gvar2``, or ``gcov``; set ``False`` and
        inspect them when diagnosing. Non-positive variances remain NaN either
        way.

    Returns
    -------
    RegionalRgResult

    Notes
    -----
    See the module docstring: uncorrected sample overlap can bias **every**
    region, and all regional estimates are shrunk toward the genome-wide
    correlation. Neither is corrected here.
    """
    _validate_boolean_controls(clip=clip)
    b1 = np.asarray(beta1, dtype=np.float64)
    b2 = np.asarray(beta2, dtype=np.float64)
    if b1.ndim != 1 or b2.ndim != 1:
        raise ValueError("beta1 and beta2 must be one-dimensional vectors")
    if b1.shape != b2.shape:
        raise ValueError(
            f"beta1 and beta2 must have the same length; got {b1.size} and "
            f"{b2.size}")
    if b1.size == 0:
        raise ValueError("beta1 and beta2 must be non-empty")
    if not (np.all(np.isfinite(b1)) and np.all(np.isfinite(b2))):
        raise ValueError("beta1 and beta2 must contain only finite values")
    m = b1.size

    # D8 and D32 match the default fitter's numeric representation. Other dense
    # inputs are accepted for exploration, but the fit first normalises them to
    # D32.

    if (isinstance(min_variants, (bool, np.bool_))
            or not isinstance(min_variants, (int, np.integer))):
        raise TypeError("min_variants must be an integer")
    min_variants = int(min_variants)
    if min_variants < 1:
        raise ValueError("min_variants must be >= 1")

    labels = np.asarray(regions)
    if labels.ndim != 1:
        raise ValueError("regions must be a one-dimensional label vector")
    if labels.size != m:
        raise ValueError(
            f"regions must have one label per variant; got {labels.size} for "
            f"{m} variants")
    if labels.dtype == object and any(label is None for label in labels):
        raise ValueError("regions must not contain None labels")
    # A float ``regions`` array using NaN as a missing sentinel would otherwise
    # slip past the None check (float dtype, not object) and be pooled by
    # np.unique into one spurious cross-genome region. Reject it like None.
    if labels.dtype.kind == "f" and not np.all(np.isfinite(labels)):
        raise ValueError("regions must not contain non-finite labels")

    # ``return_index`` already gives each unique label's first occurrence, so the
    # first-appearance order is one stable argsort -- no per-variant Python loop.
    uniq, first, inverse = np.unique(labels, return_index=True,
                                     return_inverse=True)
    inverse = inverse.ravel()
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
            raw_factor = np.asarray(R.U)
            U = (raw_factor if raw_factor.dtype == np.int8
                 else np.asarray(raw_factor, dtype=np.float32))
            factor_scale = float(R.scale)
            residual = np.asarray(R.residual_diag, dtype=np.float32)
            dense = None
        else:
            arr = np.asarray(R)
            scale = _Q8_SCALE if arr.dtype == np.int8 else 1.0
            dense = (arr, scale)
            U = residual = None
            factor_scale = 1.0

        blk_codes = code[idx]
        # Gather the block's effects once, not once per region within the block.
        b1_blk, b2_blk = b1[idx], b2[idx]
        for c in np.unique(blk_codes):
            sub = np.flatnonzero(blk_codes == c)
            if dense is not None:
                q11, q12, q22 = _dense_quadratics(dense[0], dense[1], sub,
                                                  b1_blk, b2_blk)
            else:
                q11, q12, q22 = _lowrank_quadratics(
                    U, factor_scale, residual, sub,
                    b1_blk, b2_blk,
                )
            gvar1[c] += q11
            gcov[c] += q12
            gvar2[c] += q22

    valid = (gvar1 > 0.0) & (gvar2 > 0.0)
    rg = np.full(n_reg, np.nan)
    rg[valid] = gcov[valid] / np.sqrt(gvar1[valid] * gvar2[valid])
    rg[counts < min_variants] = np.nan
    if bool(clip):
        rg = np.clip(rg, -1.0, 1.0)

    return RegionalRgResult(region=region_ids, rg=rg, gcov=gcov, gvar1=gvar1,
                            gvar2=gvar2, n_variants=counts)
