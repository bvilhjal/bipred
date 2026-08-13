"""Align two GWAS files to an ldpred3 LD cache for a bivariate fit.

The sampler still wants cache-ordered standardized effects and contiguous
``0..m-1`` blocks. This module is the on-ramp from the artifacts ldpred3
already writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ldpred3 import LowRankLD, n_eff_case_control, standardize_betas
from ldpred3.genotype_io import VariantTable
from ldpred3.harmonize import harmonize
from ldpred3.ld import load_ld_blocks
from ldpred3.ld_repr import dense_ld, lowrank_ld
from ldpred3.qc import qc_sumstats
from ldpred3.sumstats import read_sumstats


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
    af: np.ndarray
    log: dict = field(default_factory=dict)


def subset_blocks(blocks, keep):
    """Restrict blocks to ``keep``, re-tiled to a contiguous ``0..m'-1``.

    ``keep`` is a boolean mask over the current ``0..m-1`` cover, or a
    sequence of those global indices. A low-rank factor keeps its selected
    rows unless so few variants survive that the rank would exceed the
    submatrix — then it is densified and re-encoded with the same policy
    the reference uses (LR8 at size >= 1500, else dense float32).
    """
    keep = np.asarray(keep)
    if keep.dtype == bool:
        keep_set = set(np.flatnonzero(keep).tolist())
    else:
        keep_set = {int(i) for i in np.asarray(keep, dtype=np.int64).ravel()}
    new, kept_global = [], []
    for R, idx in blocks:
        loc = np.array([j for j, g in enumerate(idx) if int(g) in keep_set],
                       dtype=np.int64)
        if loc.size < 2:
            continue
        if isinstance(R, LowRankLD):
            rank = int(R.U.shape[1])
            if loc.size >= rank:
                resid = None if R.residual_diag is None else np.ascontiguousarray(
                    np.asarray(R.residual_diag)[loc])
                sub = LowRankLD(np.ascontiguousarray(R.U[loc]), int(loc.size),
                                R.scale, residual_diag=resid)
            else:
                dense = np.asarray(dense_ld(R), dtype=np.float64)[np.ix_(loc, loc)]
                sub = (lowrank_ld(dense, variance=0.99, quantize=True)
                       if loc.size >= 1500
                       else np.ascontiguousarray(dense.astype(np.float32)))
        else:
            sub = np.ascontiguousarray(np.asarray(R)[np.ix_(loc, loc)])
        new.append(sub)
        kept_global.append(np.asarray(idx)[loc])
    if not new:
        raise ValueError("subset_blocks left no block with at least two variants")
    kept_global = np.concatenate(kept_global)
    sizes = [b.U.shape[0] if isinstance(b, LowRankLD) else b.shape[0]
             for b in new]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    tiled = [(b, np.arange(offsets[i], offsets[i + 1], dtype=np.int64))
             for i, b in enumerate(new)]
    return tiled, kept_global


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
    if n_cases is not None or n_controls is not None:
        if n_cases is None or n_controls is None:
            raise ValueError(f"{label}: n_cases and n_controls must be given "
                             "together")
        return n_eff_case_control(float(n_cases), float(n_controls))
    return n_eff


def _align_one(path, variants, *, n_eff, qc, label):
    ss = read_sumstats(path, n_eff=n_eff)
    qc_log = {}
    if qc:
        keep, qc_log = qc_sumstats(ss)
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
        "label": label, "qc": qc_log, "harmonize": dict(h.log),
        "n_matched": int(ok.sum()),
    }


def prepare_bivariate_sumstats(
        ld_cache, sumstats1, sumstats2, *, n_eff1=None, n_eff2=None,
        n_cases1=None, n_controls1=None, n_cases2=None, n_controls2=None,
        qc=True, screen=False, min_af_corr=None):
    """Load an ldpred3 cache and two GWAS files; return a joint-fit panel.

    Both files are QC'd (optional), harmonized to the cache's counted allele,
    and converted with :func:`ldpred3.standardize_betas`. The returned blocks
    are the cache restricted to variants present in both files, re-tiled to
    ``0..m'-1``. Case/control counts become ``n_eff`` via
    :func:`ldpred3.n_eff_case_control`.

    ``screen=True`` applies :func:`bipred.qc.ld_consistency_screen` to each
    trait's raw GWAS z-score (not ``beta_hat / se``) and keeps the intersection.
    ``min_af_corr`` is a lower bound on ``corr(GWAS EAF, cache AF)`` after
    allele alignment; near −1 means the frequency column is inverted.
    """
    n1 = _resolve_n_eff(n_eff1, n_cases1, n_controls1, "trait 1")
    n2 = _resolve_n_eff(n_eff2, n_cases2, n_controls2, "trait 2")
    loaded = load_ld_blocks(ld_cache, return_metadata=True)
    blocks, ids, meta = loaded
    try:
        variants = _cache_variants(ids, meta)
        std1, nv1, z1, eaf1, log1 = _align_one(
            sumstats1, variants, n_eff=n1, qc=qc, label="trait1")
        std2, nv2, z2, eaf2, log2 = _align_one(
            sumstats2, variants, n_eff=n2, qc=qc, label="trait2")
        keep = np.isfinite(std1) & np.isfinite(std2)
        ref_af = meta.get("reference_af")
        af_corr = {}
        if ref_af is not None:
            ref_af = np.asarray(ref_af, dtype=float)
            for name, eaf in (("trait1", eaf1), ("trait2", eaf2)):
                mask = keep & np.isfinite(eaf) & np.isfinite(ref_af)
                af_corr[name] = (float(np.corrcoef(eaf[mask], ref_af[mask])[0, 1])
                                 if int(mask.sum()) >= 10 else float("nan"))
            if min_af_corr is not None:
                for name, corr in af_corr.items():
                    if np.isfinite(corr) and corr < float(min_af_corr):
                        raise ValueError(
                            f"{name} effect-allele frequency correlates "
                            f"{corr:.3f} with the LD-cache AF (required "
                            f">= {float(min_af_corr):g}); the file may be "
                            "on the other allele")
        if screen:
            from .qc import ld_consistency_screen
            z1u = np.where(np.isfinite(z1), z1, 0.0)
            z2u = np.where(np.isfinite(z2), z2, 0.0)
            keep = (keep
                    & ld_consistency_screen(blocks, z1u)
                    & ld_consistency_screen(blocks, z2u))
        if int(keep.sum()) < 2:
            raise ValueError("the two GWAS share fewer than two cache variants")
        tiled, kept = subset_blocks(blocks, keep)
        af = (np.asarray(ref_af, dtype=float)[kept] if ref_af is not None
              else np.full(kept.size, np.nan))
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
                 "n_kept": int(kept.size), "af_corr": af_corr,
                 "screen": bool(screen)},
        )
    finally:
        close = getattr(blocks, "close", None)
        if close is not None:
            close()
