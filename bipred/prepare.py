"""Align two GWAS files to an ldpred3 LD cache for a bivariate fit.

The sampler still wants cache-ordered standardized effects and contiguous
``0..m-1`` blocks. This module is the on-ramp from the artifacts ldpred3
already writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ldpred3 import n_eff_case_control
from ldpred3.interop import (
    PreparedLDCache,
    VariantTable,
    harmonize,
    load_ld_blocks,
    qc_sumstats,
    read_sumstats,
    standardize_betas,
    subset_ld_blocks,
)


__all__ = ["PreparedBivariate", "prepare_bivariate_sumstats", "subset_blocks"]


@dataclass
class PreparedBivariate:
    """Cache-aligned inputs for :func:`ldpred3_auto_bivariate_blocks`.

    Provenance arrays are in the same order as ``beta_hat*`` so
    :meth:`BivariateResult.write_weights` can write a frozen-scale file.
    """

    blocks: list
    beta_hat1: np.ndarray
    beta_hat2: np.ndarray
    n_eff1: np.ndarray
    n_eff2: np.ndarray
    id: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    effect_allele: np.ndarray
    other_allele: np.ndarray
    af: Optional[np.ndarray] = None
    log: dict = field(default_factory=dict)
    _ld_owner: object = field(default=None, repr=False)

    def close(self):
        """Release a memory-mapped cache owned by this prepared panel.

        Ordinary NPZ caches need no explicit cleanup.  For an mmap cache, keep
        this object alive until fitting has finished, or use it as a context
        manager.  Its LD views must not be used after ``close()``.
        """
        owner, self._ld_owner = self._ld_owner, None
        if owner is not None:
            # Drop our views before unmapping their payload.  Keeping a dangling
            # ndarray around after mmap.close() can segfault on later access.
            self.blocks.clear()
            owner.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def subset_blocks(blocks, keep):
    """Restrict blocks to ``keep``, re-tiled to a contiguous ``0..m'-1``.

    ``keep`` is an exact-length boolean mask over the current ``0..m-1``
    cover, or an integral collection of global indices (including a set).
    Invalid, duplicate, negative and out-of-range indices are rejected.
    Singletons remain valid blocks.  Complete dense blocks are reused and
    consecutive principal subsets remain views; low-rank subsets select the
    factor rows before any dense fallback.

    This is a compatibility name for LDpred3's public interoperability
    boundary.  The second return value contains the retained source indices in
    cache order.
    """
    return subset_ld_blocks(blocks, keep, return_indices=True)


def _cache_variants(ids, meta):
    counted = meta.get("counted_allele")
    other = meta.get("other_allele")
    chrom = meta.get("chrom")
    pos = meta.get("pos")
    if counted is None or other is None or chrom is None or pos is None:
        raise ValueError(
            "ld_cache lacks allele/coordinate provenance; rebuild it with "
            "the current ldpred3 and ld_out=")
    ids = np.asarray(ids)
    return VariantTable(
        chrom=np.asarray(chrom), id=ids, cm=np.zeros(len(ids)),
        pos=np.asarray(pos), a1=np.asarray(counted), a2=np.asarray(other))


def _resolve_n_eff(n_eff, n_cases, n_controls, label):
    has_counts = n_cases is not None or n_controls is not None
    if has_counts:
        if n_cases is None or n_controls is None:
            raise ValueError(f"{label}: n_cases and n_controls must be given "
                             "together")
        if n_eff is not None:
            raise ValueError(
                f"{label}: pass either a scalar n_eff or n_cases/n_controls, "
                "not both")
        return n_eff_case_control(float(n_cases), float(n_controls))
    return n_eff


def _align_one(path, variants, *, n_eff, qc, qc_params, columns, label):
    try:
        columns = {} if columns is None else dict(columns)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: columns must be a field-to-column mapping") \
            from None
    column_log = dict(columns)
    column_n = columns.pop("n_eff", None)
    if column_n is not None and n_eff is not None:
        raise ValueError(
            f"{label}: pass either a scalar n_eff or an n_eff column, not both")
    ss = read_sumstats(
        path, n_eff=column_n if column_n is not None else n_eff, **columns)
    qc_log = {}
    effective_qc = {}
    if qc:
        try:
            effective_qc = {} if qc_params is None else dict(qc_params)
        except (TypeError, ValueError):
            raise ValueError("qc_params must be a mapping") from None
        keep, qc_log = qc_sumstats(ss, **effective_qc)
        ss = ss.subset(keep)
    h = harmonize(ss, variants, drop_ambiguous=True)
    m = len(variants.id)
    beta = np.full(m, np.nan)
    se = np.full(m, np.nan)
    n_vec = np.full(m, np.nan)
    eaf = np.full(m, np.nan)
    z = np.full(m, np.nan)
    if len(h):
        idx = np.asarray(h.var_index)
        beta[idx] = h.beta
        se[idx] = h.se
        n_vec[idx] = np.asarray(h.n_eff, dtype=float)
        src = ss.eaf[h.src_index] if h.src_index is not None else None
        if src is not None:
            eaf[idx] = np.where(h.flipped, 1.0 - src, src)
        good_se = se[idx] > 0
        z[idx[good_se]] = h.beta[good_se] / h.se[good_se]
    std = np.full(m, np.nan)
    ok = (np.isfinite(beta) & np.isfinite(se) & np.isfinite(n_vec)
          & (se > 0) & (n_vec > 0))
    if ok.any():
        std[ok] = standardize_betas(beta[ok], se[ok], n_vec[ok])[0]
    return std, n_vec, z, eaf, {
        "label": label, "columns": column_log, "qc_enabled": bool(qc),
        "qc_params": effective_qc, "qc": qc_log,
        "harmonize": dict(h.log),
        "n_matched": int(ok.sum()),
    }


def prepare_bivariate_sumstats(
        ld_cache, sumstats1, sumstats2, *, n_eff1=None, n_eff2=None,
        n_cases1=None, n_controls1=None, n_cases2=None, n_controls2=None,
        columns1=None, columns2=None, qc=True, qc_params=None, screen=False,
        screen_rounds=4, screen_window=1000, screen_threshold=29.72,
        screen_eigenvalue_floor=1e-3, screen_seed=0, screen_ncores=1,
        screen_verbose=False, min_af_corr=None):
    """Load an ldpred3 cache and two GWAS files; return a joint-fit panel.

    Both files are QC'd (optional), harmonized to the cache's counted allele,
    and converted with :func:`ldpred3.standardize_betas`. The returned blocks
    are the cache restricted to variants present in both files, re-tiled to
    ``0..m'-1``. Case/control counts become ``n_eff`` via
    :func:`ldpred3.n_eff_case_control`. A trait may take a scalar ``n_eff`` or
    case/control counts, not both.

    ``columns1`` / ``columns2`` map canonical LDpred3 fields to file columns.
    ``qc_params`` is passed to :func:`ldpred3.qc.qc_sumstats` for both traits.

    ``screen=True`` first forms the joint finite panel, then applies
    :func:`bipred.qc.ld_consistency_screen` to each trait's raw GWAS z-score
    (not ``beta_hat / se``) against that panel's principal LD submatrices.
    Missing or QC-dropped variants are never represented as zero-effect
    observations.  The ``screen_*`` arguments control that diagnostic.
    ``min_af_corr`` is a lower bound on ``corr(GWAS EAF, cache AF)`` after
    allele alignment; near −1 means the frequency column is inverted.

    A path-loaded memory-mapped cache remains owned by the returned object. Use
    ``with prepare_bivariate_sumstats(...) as prep:`` or call ``prep.close()``
    after fitting. A caller-supplied :class:`ldpred3.interop.PreparedLDCache`
    retains ownership instead, so its surrounding context must remain open.
    """
    n1 = _resolve_n_eff(n_eff1, n_cases1, n_controls1, "trait 1")
    n2 = _resolve_n_eff(n_eff2, n_cases2, n_controls2, "trait 2")
    shared_cache = isinstance(ld_cache, PreparedLDCache)
    if shared_cache:
        if ld_cache.closed:
            raise ValueError("prepared LD cache is closed")
        blocks, ids, meta = (
            ld_cache.blocks, ld_cache.variant_ids, ld_cache.metadata)
        owner = None
    else:
        blocks, ids, meta = load_ld_blocks(ld_cache, return_metadata=True)
        owner = blocks if getattr(blocks, "close", None) is not None else None
    try:
        variants = _cache_variants(ids, meta)
        std1, nv1, z1, eaf1, log1 = _align_one(
            sumstats1, variants, n_eff=n1, qc=qc, qc_params=qc_params,
            columns=columns1, label="trait1")
        std2, nv2, z2, eaf2, log2 = _align_one(
            sumstats2, variants, n_eff=n2, qc=qc, qc_params=qc_params,
            columns=columns2, label="trait2")
        keep = np.isfinite(std1) & np.isfinite(std2)
        ref_af = meta.get("reference_af")
        af_corr = {}
        if ref_af is not None:
            ref_af = np.asarray(ref_af, dtype=float)
            for name, eaf in (("trait1", eaf1), ("trait2", eaf2)):
                mask = keep & np.isfinite(eaf) & np.isfinite(ref_af)
                enough = (int(mask.sum()) >= 10
                          and np.ptp(eaf[mask]) > 0.0
                          and np.ptp(ref_af[mask]) > 0.0)
                af_corr[name] = (
                    float(np.corrcoef(eaf[mask], ref_af[mask])[0, 1])
                    if enough else float("nan"))
            if min_af_corr is not None:
                try:
                    threshold = float(min_af_corr)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(
                        "min_af_corr must be a finite value in [-1, 1]") \
                        from None
                if isinstance(min_af_corr, (bool, np.bool_)):
                    raise ValueError(
                        "min_af_corr must be a finite value in [-1, 1]")
                if not np.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
                    raise ValueError("min_af_corr must be a finite value in [-1, 1]")
                for name, corr in af_corr.items():
                    if not np.isfinite(corr):
                        raise ValueError(
                            f"{name} lacks 10 jointly observed, varying finite "
                            "effect-allele frequencies; min_af_corr cannot be "
                            "evaluated")
                    if corr < threshold:
                        raise ValueError(
                            f"{name} effect-allele frequency correlates "
                            f"{corr:.3f} with the LD-cache AF (required "
                            f">= {threshold:g}); the file may be "
                            "on the other allele")
        elif min_af_corr is not None:
            raise ValueError(
                "min_af_corr requires reference_af in the LD cache")
        n_joint = int(keep.sum())
        if n_joint < 2:
            raise ValueError("the two GWAS share fewer than two cache variants")
        tiled, joint_indices = subset_blocks(blocks, keep)
        n_screen_drop = 0
        screen_log = None
        if screen:
            from .qc import ld_consistency_screen

            screen_options = dict(
                rounds=screen_rounds, window=screen_window,
                threshold=screen_threshold,
                eigenvalue_floor=screen_eigenvalue_floor, seed=screen_seed,
                ncores=screen_ncores, verbose=screen_verbose)
            screen_log = dict(screen_options)
            screen_keep = (
                ld_consistency_screen(tiled, z1[joint_indices], **screen_options)
                & ld_consistency_screen(
                    tiled, z2[joint_indices], **screen_options))
            n_screen_drop = n_joint - int(screen_keep.sum())
            if int(screen_keep.sum()) < 2:
                raise ValueError(
                    "LD-consistency screening left fewer than two joint variants")
            if not np.all(screen_keep):
                # Re-select once from the owning cache so the final panel does
                # not retain an intermediate dense subset as well as its child.
                final_indices = joint_indices[screen_keep]
                if tiled is not blocks:
                    tiled.clear()
                tiled, kept = subset_blocks(blocks, final_indices)
            else:
                kept = joint_indices
        else:
            kept = joint_indices
        af = (np.asarray(ref_af, dtype=float)[kept] if ref_af is not None
              else None)
        return PreparedBivariate(
            blocks=tiled,
            beta_hat1=np.ascontiguousarray(std1[kept]),
            beta_hat2=np.ascontiguousarray(std2[kept]),
            n_eff1=np.ascontiguousarray(nv1[kept]),
            n_eff2=np.ascontiguousarray(nv2[kept]),
            id=np.asarray(ids)[kept],
            chrom=np.asarray(variants.chrom)[kept],
            pos=np.asarray(variants.pos)[kept],
            effect_allele=np.asarray(variants.a1)[kept],
            other_allele=np.asarray(variants.a2)[kept],
            af=af,
            log={"trait1": log1, "trait2": log2, "n_cache": int(len(ids)),
                 "n_joint": n_joint, "n_kept": int(kept.size),
                 "n_screen_drop": n_screen_drop, "af_corr": af_corr,
                 "screen": bool(screen), "screen_params": screen_log,
                 "prepared_cache": shared_cache},
            _ld_owner=owner,
        )
    except BaseException:
        if owner is not None:
            if "tiled" in locals() and tiled is not blocks:
                tiled.clear()
            owner.close()
        raise
