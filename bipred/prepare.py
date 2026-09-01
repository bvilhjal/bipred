"""Align one or two GWAS files to an ldpred3 LD cache.

The sampler still wants cache-ordered standardized effects and contiguous
``0..m-1`` blocks. A prepared trait instead stays sparse in the full cache's
index space, so QC and harmonization can be reused before a pair is formed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Optional
import warnings

import numpy as np

from ldpred3 import n_eff_case_control
from ldpred3.harmonize import (
    BUILD_MISMATCH_FRACTION as _BUILD_MISMATCH_FRACTION,
    REFERENCE_COVERAGE_WARN as _REFERENCE_COVERAGE_WARN,
    SEVERE_REFERENCE_COVERAGE as _SEVERE_REFERENCE_COVERAGE,
)
from ldpred3.interop import (
    PreparedLDCache,
    VariantTable,
    harmonize,
    prepare_ld_cache,
    qc_sumstats,
    read_sumstats,
    standardize_betas,
    subset_ld_blocks,
)

from . import _progress
from ._ldpred3_compat import _variant_indices


__all__ = [
    "PreparedTrait", "PreparedBivariate", "prepare_trait_sumstats",
    "screen_prepared_trait", "pair_prepared_traits",
    "prepare_bivariate_sumstats", "subset_blocks",
]


@dataclass
class PreparedTrait:
    """One GWAS aligned sparsely to the full LD-cache index space.

    ``indices`` is strictly increasing and maps every aligned element of
    ``beta_hat``, ``n_eff``, ``z``, and ``eaf`` to the corresponding variant
    in the full cache. Only usable variants are stored; missing and QC-dropped
    variants are absent rather than represented as zeros or NaNs. ``n_cache``
    records the full reference length and is checked again when traits are
    paired.

    The object owns no LD memory and its arrays can be serialized directly.
    A persistent cache must still be keyed by the identity or content hash of
    the LD reference: equal ``n_cache`` values alone do not establish that two
    references contain the same variants in the same order.
    """

    indices: np.ndarray
    beta_hat: np.ndarray
    n_eff: np.ndarray
    z: np.ndarray
    eaf: np.ndarray
    n_cache: int
    log: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.indices)


@dataclass
class PreparedBivariate:
    """Cache-aligned inputs for :func:`ldpred3_auto_bivariate_blocks`.

    Provenance arrays are in the same order as ``beta_hat*`` so
    :meth:`BivariateResult.write_weights` can write a frozen-scale file.
    ``cache_indices`` maps those rows back to the full LD-reference order;
    this lets callers subset reference-wide invariants such as precomputed LD
    scores without recomputing them on the fitted principal submatrix.
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
    cache_indices: Optional[np.ndarray] = None
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


def _consume_subset_blocks(blocks, keep, n_cache):
    """Destructively form a principal subset one source block at a time.

    This private path is for a cache whose ordinary in-memory LD list has been
    explicitly surrendered by its caller.  Each source-list slot is released
    as soon as that block has been transferred or subset, so all full blocks
    cannot remain resident beside all of their copied principal subsets.

    The source is validated completely before its first mutation.  Once
    mutation begins, success and failure both leave ``blocks`` empty; a caller
    must consequently treat the surrounding :class:`PreparedLDCache` as
    consumed.  Memory-mapped list owners are rejected because their views need
    a live owner and already avoid loading the full payload into resident RAM.
    """
    if type(blocks) is not list:
        raise ValueError(
            "consume_ld_cache=True requires an ordinary in-memory LD block "
            "list; memory-mapped or custom block owners cannot be consumed")
    if (isinstance(n_cache, (bool, np.bool_))
            or not isinstance(n_cache, (int, np.integer))
            or int(n_cache) < 1):
        raise ValueError("n_cache must be a positive integer")
    n_cache = int(n_cache)

    values = np.asarray(keep)
    if (values.ndim != 1
            or not np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.bool_)):
        raise ValueError(
            "consumed LD selection must be a one-dimensional integer array")
    keep = values.astype(np.int64, copy=False)
    if keep.size == 0:
        raise ValueError("consumed LD selection contains no variants")
    if keep[0] < 0 or keep[-1] >= n_cache:
        raise IndexError(
            f"consumed LD selection indices must lie in [0, {n_cache})")
    if keep.size > 1 and np.any(np.diff(keep) <= 0):
        raise ValueError(
            "consumed LD selection must be strictly increasing and unique")

    # This one-element principal subset makes LDpred3 validate every source
    # block's shape and exact 0..M-1 tiling without materializing the requested
    # (potentially multi-gigabyte) subset.  The probe itself is at most 1x1.
    probe, _ = subset_ld_blocks(blocks, np.array([0], dtype=np.int64),
                                return_indices=True)
    if probe is not blocks:
        probe.clear()
    total = int(np.asarray(blocks[-1][1], dtype=np.int64)[-1]) + 1
    if total != n_cache:
        raise ValueError(
            f"LD blocks cover {total} variants, expected {n_cache}")

    out = []
    out_start = 0
    source = local_source = partial = subset = None
    try:
        for block_i in range(len(blocks)):
            source, source_indices = blocks[block_i]
            start = int(source_indices[0])
            stop = int(source_indices[-1]) + 1
            left = int(np.searchsorted(keep, start, side="left"))
            right = int(np.searchsorted(keep, stop, side="left"))

            # Transfer ownership before allocating the principal subset.  If
            # allocation fails, the exception path drops this local reference
            # and all as-yet-unvisited source slots as well.
            blocks[block_i] = None
            if left == right:
                source = None
                continue

            local_keep = keep[left:right] - start
            k_source = stop - start
            k_keep = int(local_keep.size)
            full = (k_keep == k_source and local_keep[0] == 0
                    and local_keep[-1] == k_source - 1)
            if full:
                subset = source
            else:
                local_source = [(
                    source, np.arange(k_source, dtype=np.int64))]
                partial = subset_ld_blocks(local_source, local_keep)
                if len(partial) != 1:
                    raise ValueError(
                        "one source LD block produced an invalid principal "
                        "subset")
                subset = partial[0][0]
            out.append((
                subset,
                np.arange(out_start, out_start + k_keep, dtype=np.int64),
            ))
            out_start += k_keep

            # CPython can now release a copied source block before allocation
            # begins for the next block.  Views deliberately retain only the
            # one source allocation they view, without making a second copy.
            source = local_source = partial = subset = None

        blocks.clear()
        if out_start != keep.size:
            raise ValueError(
                "LD blocks do not cover every selected variant")
        return out, keep
    except BaseException:
        # Destructive operation: never leave a partly consumed cache looking
        # reusable, and release any subset allocations completed so far.
        out.clear()
        blocks.clear()
        source = local_source = partial = subset = None
        raise


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


def _open_cache(ld_cache):
    """Return ``(PreparedLDCache, owned_here)`` after the usual validation."""
    if isinstance(ld_cache, PreparedLDCache):
        if ld_cache.closed:
            raise ValueError("prepared LD cache is closed")
        return ld_cache, False
    return prepare_ld_cache(ld_cache), True


def _cache_variant_table(cache):
    """Build the harmonization index once per reusable prepared cache."""
    variants = getattr(cache, "_bipred_variant_table", None)
    if variants is None:
        variants = _cache_variants(cache.variant_ids, cache.metadata)
        # PreparedLDCache is a mutable read-only-lifetime handle. Publishing a
        # duplicate build from concurrent callers is harmless; harmonize has
        # the same benign race for its own indices.
        cache._bipred_variant_table = variants
    return variants


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


def _norm_chrom_label(value):
    """Strip a ``chr`` prefix and canonicalise case; enough to catch build mix-ups."""
    text = str(value).strip()
    if text[:3].lower() == "chr":
        text = text[3:]
    return text.upper()


# Coverage thresholds live in ldpred3, not here, so the CLI and this package
# diagnose a thin alignment with the same numbers. Two copies of 0.5 in two
# repos is precisely how a "shared" threshold stops being shared.
#
#: Warn when the aligned trait covers less than this fraction of the LD
#: reference. LDSC's ``M``, the ``h2`` denominator, and the LD-consistency
#: screen's window coverage are all defined on the *full* reference, so a
#: sparsely covered panel changes what those numbers mean.
DEFAULT_REFERENCE_COVERAGE_WARN = _REFERENCE_COVERAGE_WARN
#: Below this fraction the fit is no longer a genome-wide analysis of that
#: reference; the wording escalates and names the likely causes.
SEVERE_REFERENCE_COVERAGE = _SEVERE_REFERENCE_COVERAGE
#: Call a coordinate/build mismatch when at least this fraction of the
#: unmatched rows carry an identifier the reference holds at a *different*
#: locus. Anything less is the ordinary case of two different variant sets.
DEFAULT_BUILD_MISMATCH_FRACTION = _BUILD_MISMATCH_FRACTION


class _ReferenceLoci:
    """Loci carrying each reference identifier, over ldpred3's cached index.

    ``harmonize`` builds an identifier -> row-index map and memoises it on the
    variant table, and a caller preparing several traits against one reference
    pays for it once. Borrowing that map and reading ``chrom``/``pos`` on
    demand keeps this lookup free; the private ``{id: {(chrom, pos)}}`` dict it
    replaces cost roughly half a gigabyte per call on a 1.4M-variant reference,
    and two calls could be live at once during a re-anchored preparation.
    """

    __slots__ = ("_by_id", "_chrom", "_pos", "_memo")

    def __init__(self, variants):
        self._by_id = _variant_indices(variants)[1]
        self._chrom = variants.chrom
        self._pos = variants.pos
        self._memo = {}

    def __len__(self):
        return len(self._by_id)

    def _norm(self, raw):
        norm = self._memo.get(raw)
        if norm is None:
            norm = self._memo[raw] = _norm_chrom_label(raw)
        return norm

    def loci(self, key):
        """Normalised ``(chrom, pos)`` pairs for ``key``, or ``None``."""
        rows = self._by_id.get(key)
        if not rows:
            return None
        # An identifier at one locus with several allele records is the common
        # case, so de-duplicate: callers ask "how many distinct loci", not
        # "how many rows".
        return {(self._norm(self._chrom[gi]), int(self._pos[gi]))
                for gi in rows}


def _diagnose_unmatched(sumstats, variants, harmonized):
    """Split the unmatched GWAS rows into build-mismatch and absent-variant.

    ``harmonize`` rejects a unique identifier whose chrom/pos disagrees with
    the reference rather than accepting it at the wrong locus, so a whole-file
    coordinate error (GRCh38 statistics against a GRCh37 reference, or CHR and
    BP read from the wrong columns) leaves *no* trace beyond a collapsed match
    count. Recover the distinction here: an unmatched row whose identifier the
    reference does hold, at a locus that disagrees, is a coordinate problem,
    not a variant the reference lacks.
    """
    n_rows = len(sumstats.id)
    matched_rows = np.zeros(n_rows, dtype=bool)
    if harmonized.src_index is not None and len(harmonized):
        matched_rows[np.asarray(harmonized.src_index, dtype=np.int64)] = True
    unmatched = np.where(~matched_rows)[0]
    report = {"n_unmatched_rows": int(unmatched.size),
              "n_unmatched_id_elsewhere": 0,
              "n_unmatched_id_absent": 0,
              "n_unmatched_id_missing": 0,
              "examples": []}
    if unmatched.size == 0:
        return report
    index = _ReferenceLoci(variants)
    ss_id, ss_chrom, ss_pos = sumstats.id, sumstats.chrom, sumstats.pos
    memo = {}
    for k in unmatched:
        key = ss_id[k]
        if not key:
            report["n_unmatched_id_missing"] += 1
            continue
        loci = index.loci(key)
        if loci is None:
            report["n_unmatched_id_absent"] += 1
            continue
        raw = ss_chrom[k]
        norm = memo.get(raw)
        if norm is None:
            norm = memo[raw] = _norm_chrom_label(raw)
        try:
            here = int(ss_pos[k])
        except (TypeError, ValueError):
            here = 0
        if (norm, here) in loci:
            # Present at the same locus: the row failed on alleles, on a
            # duplicate, or on a non-finite effect, all of which harmonize
            # already counts separately.
            continue
        report["n_unmatched_id_elsewhere"] += 1
        if len(report["examples"]) < 3:
            ref_chrom, ref_pos = sorted(loci)[0]
            report["examples"].append(
                f"{key} GWAS {norm}:{here} vs reference "
                f"{ref_chrom}:{ref_pos}")
    return report


def _reanchor_on_identifier(sumstats, variants, *, label,
                            warning_stacklevel):
    """Move each row onto the reference's own coordinates for its identifier.

    This is the coordinate repair for a build mismatch when the identifiers
    are trustworthy and shared, which is the ordinary case for a GWAS Catalog
    harmonised file (``hm_rsid`` is dbSNP-mapped) against an rsID-keyed
    reference. It is *not* a chain-file liftover: no interval mapping is
    consulted, and no coordinate is invented. Each row whose identifier the
    reference holds at exactly one locus is re-stamped with that locus, so
    the pair is thereafter compared on the reference's build, and alleles are
    still checked afterwards by :func:`harmonize` exactly as before.

    Two classes of row are *dropped* rather than re-anchored:

    1. an identifier the reference does not hold. Its coordinates are on the
       other build, so leaving it in place would expose it to positional
       matching, where a GRCh38 coordinate can land on a *different* variant's
       GRCh37 coordinate and match on alleles by chance. A wrong-variant match
       is worse than a lost variant.
    2. an identifier the reference holds at more than one locus. Nothing but
       the coordinates could choose between them, and the coordinates are the
       quantity under repair.

    Prefer a properly lifted-over file with its chain and its failed mappings
    recorded. Use this when the alternative is discarding 99% of the GWAS, and
    read the returned counts: they are the audit trail.
    """
    ids = sumstats.id
    n_rows = len(ids)
    index = _ReferenceLoci(variants)
    chrom = np.array(sumstats.chrom, dtype=object)
    pos = np.array(sumstats.pos, dtype=np.int64)
    keep = np.ones(n_rows, dtype=bool)
    memo = {}
    n_anchored = n_moved = n_ambiguous = n_absent = n_missing_id = 0
    for k in range(n_rows):
        key = ids[k]
        if not key:
            n_missing_id += 1
            keep[k] = False
            continue
        loci = index.loci(key)
        if loci is None:
            n_absent += 1
            keep[k] = False
            continue
        if len(loci) > 1:
            n_ambiguous += 1
            keep[k] = False
            continue
        ref_chrom, ref_pos = next(iter(loci))
        raw = chrom[k]
        norm = memo.get(raw)
        if norm is None:
            norm = memo[raw] = _norm_chrom_label(raw)
        if norm != ref_chrom or int(pos[k]) != ref_pos:
            n_moved += 1
        chrom[k] = ref_chrom
        pos[k] = ref_pos
        n_anchored += 1
    log = {"applied": True, "n_rows": int(n_rows),
           "n_anchored": int(n_anchored), "n_moved": int(n_moved),
           "n_dropped_ambiguous_locus": int(n_ambiguous),
           "n_dropped_absent_identifier": int(n_absent),
           "n_dropped_missing_identifier": int(n_missing_id)}
    anchored = replace(sumstats, chrom=chrom, pos=pos).subset(keep)
    if n_moved:
        warnings.warn(
            f"{label}: re-anchored {n_moved:,} of {n_rows:,} rows onto the LD "
            "reference's coordinates for their identifiers, because the two "
            "sides disagreed on where those variants are. The fit is "
            "therefore on the reference's genome build, keyed on identifier "
            f"rather than on position. {n_absent:,} rows carry an identifier "
            f"the reference does not hold and {n_ambiguous:,} an identifier "
            "it holds at several loci; both were dropped rather than matched "
            "positionally on the wrong build. This is not a chain-file "
            "liftover: record it as identifier-keyed re-anchoring, and "
            "prefer a properly lifted-over input where one is available.",
            RuntimeWarning, stacklevel=warning_stacklevel)
    return anchored, log


def _reference_overlap(sumstats, variants, harmonized, n_matched, *, label,
                       warning_stacklevel):
    """Record the reference-coverage fractions, and warn when coverage is low.

    Zero overlap is an error elsewhere. What this catches is the case that
    still produces numbers: a fit that runs on a small, unrepresentative
    corner of the LD reference.
    """
    n_rows = int(len(sumstats.id))
    n_cache = int(len(variants.id))
    overlap = {
        "n_sumstats_offered": n_rows,
        "n_cache": n_cache,
        "n_matched": int(n_matched),
        "frac_of_reference": (float(n_matched) / n_cache) if n_cache else 0.0,
        "frac_of_sumstats": (float(n_matched) / n_rows) if n_rows else 0.0,
        "diagnosed": False,
    }
    if overlap["frac_of_reference"] >= DEFAULT_REFERENCE_COVERAGE_WARN:
        overlap["reason"] = "adequate_coverage"
        return overlap
    if n_rows == 0:
        # Nothing reached harmonization, so the reference says nothing about
        # why. Naming the build here would be a guess.
        overlap.update(reason="no_rows_offered", diagnosed=False)
        warnings.warn(
            f"{label}: summary-statistics QC removed every row before "
            "alignment to the LD reference, so no variant could match. Check "
            "the QC filters against the file's columns -- a missing or "
            "unparsed sample-size, allele-frequency, or standard-error "
            "column removes every row -- before suspecting the reference.",
            RuntimeWarning, stacklevel=warning_stacklevel)
        return overlap
    diagnosis = _diagnose_unmatched(sumstats, variants, harmonized)
    overlap.update(diagnosis, diagnosed=True)
    n_unmatched = diagnosis["n_unmatched_rows"]
    elsewhere = diagnosis["n_unmatched_id_elsewhere"]
    build_mismatch = bool(
        n_unmatched
        and elsewhere / n_unmatched >= DEFAULT_BUILD_MISMATCH_FRACTION)
    overlap["build_mismatch_suspected"] = build_mismatch
    absent = diagnosis["n_unmatched_id_absent"]
    overlap["reason"] = (
        "build_mismatch" if build_mismatch else
        "absent_identifiers" if absent and absent >= elsewhere else
        "rows_rejected_at_matched_locus" if n_unmatched else
        "sparse_sumstats")
    severe = overlap["frac_of_reference"] < SEVERE_REFERENCE_COVERAGE
    lines = [
        f"{label}: only {int(n_matched):,} of {n_cache:,} LD-reference "
        f"variants are covered ({overlap['frac_of_reference']:.1%}), from "
        f"{n_rows:,} GWAS rows offered to harmonization."]
    if severe:
        lines.append(
            "A fit on this few reference variants is not a genome-wide "
            "analysis of that reference: the LD blocks are nearly empty, the "
            "LD-consistency screen leaves most windows below its minimum "
            "size, and LDSC keeps the full reference M, so h2 and "
            "polygenicity are on a denominator the data do not populate.")
    else:
        lines.append(
            "LDSC keeps the full reference M and the LD blocks stay sparse, "
            "so h2, polygenicity, and screen coverage are all affected.")
    if build_mismatch:
        lines.append(
            f"{elsewhere:,} of the {n_unmatched:,} unmatched rows carry an "
            "identifier the reference does hold, at coordinates that "
            "disagree. That is the signature of a genome-build mismatch "
            "(GRCh38 summary statistics against a GRCh37 reference, or the "
            "reverse) or of CHR/BP read from the wrong columns, not of a "
            "reference that lacks those variants. Lift the summary "
            "statistics over to the reference's build -- recording the chain "
            "file and the failed mappings -- or use a reference on the GWAS "
            "build; do not drop the coordinates to force an "
            "identifier-only match.")
        if diagnosis["examples"]:
            lines.append("Disagreeing loci: "
                         + "; ".join(diagnosis["examples"]) + ".")
    elif overlap["reason"] == "absent_identifiers":
        lines.append(
            f"{absent:,} of the {n_unmatched:,} unmatched rows carry an "
            "identifier the reference does not hold at all, so the two "
            "variant sets differ rather than disagreeing on coordinates.")
    elif overlap["reason"] == "rows_rejected_at_matched_locus":
        lines.append(
            f"The {n_unmatched:,} unmatched rows are mostly at loci the "
            "reference does hold, so they were rejected on alleles, on "
            "strand ambiguity, on a duplicate identifier, or on a "
            "non-finite effect rather than on coordinates; read the "
            "harmonize counts in the log.")
    else:
        lines.append(
            "Every offered row matched, so the GWAS itself covers only this "
            "part of the reference -- a hits-only or array-restricted "
            "deposition rather than a misalignment. Genome-wide estimands "
            "cannot be read off such a subset.")
    warnings.warn(" ".join(lines), RuntimeWarning,
                  stacklevel=warning_stacklevel)
    return overlap


def _locus_mismatch(sumstats, variants, harmonized):
    """Rows whose rsID matched a reference variant at a different chrom/pos.

    Current ``harmonize`` already rejects a unique rsID at the wrong locus
    when both sides carry coordinates. This remains as a defense for older
    ldpred3 releases that accepted those rows.
    """
    src_pos = np.asarray(sumstats.pos, dtype=np.int64)[harmonized.src_index]
    tgt_pos = np.asarray(variants.pos, dtype=np.int64)[harmonized.var_index]
    comparable = (src_pos > 0) & (tgt_pos > 0)
    if not comparable.any():
        return np.zeros(len(harmonized), dtype=bool)
    src_chrom = np.asarray(sumstats.chrom, dtype=object)[harmonized.src_index]
    tgt_chrom = np.asarray(variants.chrom, dtype=object)[harmonized.var_index]
    chrom_bad = np.fromiter(
        (_norm_chrom_label(a) != _norm_chrom_label(b)
         for a, b in zip(src_chrom, tgt_chrom)),
        dtype=bool, count=len(harmonized))
    return comparable & ((src_pos != tgt_pos) | chrom_bad)


def _align_one(path, variants, *, n_eff, qc, qc_params, columns, label,
               reanchor_on_identifier=False):
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
    if column_n is not None and not isinstance(column_n, str):
        # read_sumstats treats only a *string* n_eff as a column override; any
        # other n_eff= value becomes a forced per-variant scalar (and an
        # "n_eff" key in **columns binds that same parameter). An integer
        # index therefore goes in as its digit string, which _build_colmap
        # resolves by position; anything else is rejected rather than
        # silently coerced.
        if not isinstance(column_n, (int, np.integer)):
            raise ValueError(
                f"{label}: columns['n_eff'] must be a column name or a "
                f"zero-based integer index, not {column_n!r}")
        column_n = str(int(column_n))
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
    reanchor_log = {"applied": False}
    if reanchor_on_identifier:
        ss, reanchor_log = _reanchor_on_identifier(
            ss, variants, label=label, warning_stacklevel=4)
    h = harmonize(ss, variants, drop_ambiguous=True)
    beta = np.asarray(h.beta, dtype=float)
    se = np.asarray(h.se, dtype=float)
    n_vec = np.asarray(h.n_eff, dtype=float)
    eaf = np.full(len(h), np.nan)
    if len(h):
        src = ss.eaf[h.src_index] if h.src_index is not None else None
        if src is not None:
            eaf = np.where(h.flipped, 1.0 - src, src)
    ok = (np.isfinite(beta) & np.isfinite(se) & np.isfinite(n_vec)
          & (se > 0) & (n_vec > 0))
    n_invalid = int((~ok).sum()) if len(h) else 0
    n_locus_mismatch = 0
    if len(h) and h.src_index is not None:
        mismatch = _locus_mismatch(ss, variants, h)
        n_locus_mismatch = int(np.count_nonzero(mismatch & ok))
        ok = ok & ~mismatch
        if n_locus_mismatch:
            warnings.warn(
                f"{label}: dropped {n_locus_mismatch} rsID match(es) whose "
                "chrom/pos disagree with the LD reference (possible mixed "
                "genome build).",
                RuntimeWarning, stacklevel=3)
    overlap = _reference_overlap(
        ss, variants, h, int(ok.sum()), label=label, warning_stacklevel=3)
    indices = np.asarray(h.var_index, dtype=np.int64)[ok]
    # Keep this boundary correct even if an interoperability backend returns
    # matches in source-file order. Cache-order sorting makes later sparse
    # intersections deterministic; ``harmonize`` already guarantees unique
    # target indices, and pairing validates that invariant again.
    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    std = np.empty(int(ok.sum()), dtype=float)
    if ok.any():
        std[:] = standardize_betas(beta[ok], se[ok], n_vec[ok])[0][order]
    return PreparedTrait(
        indices=np.ascontiguousarray(indices),
        beta_hat=np.ascontiguousarray(std),
        n_eff=np.ascontiguousarray(n_vec[ok][order]),
        z=np.ascontiguousarray((beta[ok] / se[ok])[order]),
        eaf=np.ascontiguousarray(np.asarray(eaf, dtype=float)[ok][order]),
        n_cache=int(len(variants.id)),
        log={
            "label": label, "columns": column_log, "qc_enabled": bool(qc),
            "qc_params": effective_qc, "qc": qc_log,
            "harmonize": dict(h.log),
            "n_harmonized": int(len(h)),
            "n_invalid": n_invalid,
            "n_locus_mismatch": n_locus_mismatch,
            "n_matched": int(ok.sum()),
            "reference_overlap": overlap,
            "reanchor": reanchor_log,
        },
    )


def _trait_label(trait, fallback):
    if isinstance(trait.log, dict):
        label = trait.log.get("label")
        if isinstance(label, str) and label:
            return label
    return fallback


def _require_usable(trait, fallback):
    if len(trait) == 0:
        message = (
            f"{_trait_label(trait, fallback)}: all GWAS variants were removed "
            "by sumstats QC or harmonization against the LD reference (no "
            "usable variant remains)")
        overlap = trait.log.get("reference_overlap") \
            if isinstance(trait.log, dict) else None
        reason = overlap.get("reason") if isinstance(overlap, dict) else None
        if reason == "build_mismatch":
            message += (
                f"; {overlap['n_unmatched_id_elsewhere']:,} rows carry an "
                "identifier the reference holds at different coordinates, so "
                "check the genome build of both sides before anything else")
        elif reason == "no_rows_offered":
            message += (
                "; summary-statistics QC removed every row before alignment, "
                "so the reference was never consulted -- check the QC "
                "filters against the file's columns (a missing or unparsed "
                "sample-size, frequency, or standard-error column removes "
                "every row)")
        elif reason == "absent_identifiers":
            message += (
                f"; {overlap.get('n_unmatched_id_absent', 0):,} rows carry an "
                "identifier the reference does not hold, so the variant sets "
                "differ (check the build, the ancestry panel, and the "
                "identifier column)")
        elif reason == "rows_rejected_at_matched_locus":
            message += (
                f"; the {overlap.get('n_unmatched_rows', 0):,} unmatched rows "
                "are mostly at loci the reference does hold, so read the "
                "harmonize allele, strand, duplicate, and non-finite counts "
                "in the log")
        raise ValueError(message)


def _prepare_trait(cache, sumstats, *, n_eff, qc, qc_params, columns, label,
                   require_usable=True, reanchor_on_identifier=False):
    trait = _align_one(
        sumstats, _cache_variant_table(cache), n_eff=n_eff, qc=qc,
        qc_params=qc_params, columns=columns, label=label,
        reanchor_on_identifier=reanchor_on_identifier)
    if require_usable:
        _require_usable(trait, label)
    return trait


def prepare_trait_sumstats(
        ld_cache, sumstats, *, n_eff=None, n_cases=None, n_controls=None,
        columns=None, qc=True, qc_params=None, label="trait", progress=None,
        reanchor_on_identifier=False):
    """QC, harmonize, and standardize one GWAS against a full LD cache.

    The returned :class:`PreparedTrait` stores only usable variants, with
    strictly increasing ``indices`` into the full cache. It can therefore be
    serialized once and paired repeatedly with other traits through
    :func:`pair_prepared_traits`; LD blocks are deliberately not retained.
    Use :func:`screen_prepared_trait` to apply the LD-consistency diagnostic
    to this trait's own principal reference panel before pairing.

    Pass either scalar ``n_eff`` or ``n_cases`` and ``n_controls``. ``columns``
    maps canonical LDpred3 fields to file columns, and ``qc_params`` is passed
    to :func:`ldpred3.qc.qc_sumstats`. A path-loaded cache is closed before
    return. A caller-supplied :class:`ldpred3.interop.PreparedLDCache` remains
    caller-owned and reuses its cached harmonization index across traits.

    ``progress``, when given, receives two step events: loading the LD
    reference, then reading/QC/harmonizing/standardizing the trait.

    ``reanchor_on_identifier`` repairs a genome-build mismatch by taking each
    row's coordinates from the reference entry that carries its identifier;
    see :func:`_reanchor_on_identifier` for what it drops and why. It is off
    by default: a build mismatch should be seen, not absorbed. The log records
    the overlap fractions under ``reference_overlap`` either way.
    """
    resolved_n = _resolve_n_eff(n_eff, n_cases, n_controls, label)
    _progress.validate(progress)
    _progress.report(progress, "load LD reference", 0, 2, unit="step")
    cache, owned = _open_cache(ld_cache)
    try:
        _progress.report(
            progress, f"read, QC, harmonize, and standardize {label}", 1, 2,
            unit="step")
        return _prepare_trait(
            cache, sumstats, n_eff=resolved_n, qc=qc, qc_params=qc_params,
            columns=columns, label=label,
            reanchor_on_identifier=reanchor_on_identifier)
    finally:
        if owned:
            cache.close()


def _validated_trait(trait, n_cache, fallback):
    """Validate a cached sparse trait and return normalized array views."""
    if not isinstance(trait, PreparedTrait):
        raise TypeError(f"{fallback} must be a PreparedTrait")
    label = _trait_label(trait, fallback)
    if (isinstance(trait.n_cache, (bool, np.bool_))
            or not isinstance(trait.n_cache, (int, np.integer))):
        raise ValueError(f"{label}: n_cache must be an integer")
    if int(trait.n_cache) != n_cache:
        raise ValueError(
            f"{label}: prepared against {int(trait.n_cache):,} cache variants, "
            f"but this LD cache has {n_cache:,}")
    indices = np.asarray(trait.indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) \
            or np.issubdtype(indices.dtype, np.bool_):
        raise ValueError(f"{label}: indices must be a one-dimensional integer array")
    if len(indices) and (np.any(indices < 0) or np.any(indices >= n_cache)):
        raise ValueError(f"{label}: indices must lie in [0, {n_cache})")
    indices = indices.astype(np.int64, copy=False)
    if len(indices) > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError(
            f"{label}: indices must be strictly increasing and unique")

    arrays = {}
    for name in ("beta_hat", "n_eff", "z", "eaf"):
        try:
            values = np.asarray(getattr(trait, name), dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{label}: {name} must be a numeric vector") \
                from None
        if values.ndim != 1 or len(values) != len(indices):
            raise ValueError(
                f"{label}: {name} must be one-dimensional and aligned with "
                "indices")
        arrays[name] = values
    if not np.all(np.isfinite(arrays["beta_hat"])):
        raise ValueError(f"{label}: beta_hat must be finite")
    if np.any(np.abs(arrays["beta_hat"]) >= 1.0):
        raise ValueError(
            f"{label}: beta_hat must satisfy |beta_hat| < 1 "
            "(ldpred3.standardize_betas returns values in (-1, 1); raw "
            "z-scores are not standardized effects)")
    if (not np.all(np.isfinite(arrays["n_eff"]))
            or np.any(arrays["n_eff"] <= 0)):
        raise ValueError(f"{label}: n_eff must be finite and positive")
    if not np.all(np.isfinite(arrays["z"])):
        raise ValueError(f"{label}: z must be finite")
    eaf = arrays["eaf"]
    if np.any(np.isinf(eaf)):
        raise ValueError(f"{label}: eaf must be finite or NaN")
    finite_eaf = np.isfinite(eaf)
    if np.any((eaf[finite_eaf] < 0) | (eaf[finite_eaf] > 1)):
        raise ValueError(f"{label}: finite eaf values must lie in [0, 1]")
    return indices, arrays, label


def screen_prepared_trait(
        ld_cache, trait, *, rounds=4, window=1000, threshold=29.72,
        eigenvalue_floor=1e-3, seed=0, ncores=1, verbose=False,
        progress=None):
    """Screen one prepared GWAS against its exact principal LD panel.

    The sparse trait is validated against ``ld_cache`` and its cache indices
    select the corresponding rows inside each source block. The complete
    principal panel is never materialized: the diagnostic forms only its
    window-sized submatrices. It receives the raw, harmonized GWAS z vector—not
    standardized ``beta_hat``—and the returned :class:`PreparedTrait` contains
    only retained rows. The input is never mutated and the returned object owns
    no LD memory.

    ``rounds``, ``window``, ``threshold``, ``eigenvalue_floor``, ``seed``,
    ``ncores``, and ``verbose`` are forwarded to
    :func:`bipred.qc.ld_consistency_screen`. ``progress`` receives that
    function's per-block events, labelled with the trait's recorded label. A
    path-loaded ordinary or memory-mapped cache is closed before return.
    Fewer than two retained variants is an error because no bivariate panel
    can subsequently be formed.
    """
    from .qc import _ld_consistency_screen_selected

    _progress.validate(progress)
    cache, owned = _open_cache(ld_cache)
    try:
        n_cache = int(len(cache.variant_ids))
        indices, arrays, label = _validated_trait(trait, n_cache, "trait")
        n_input = int(len(indices))
        if n_input < 2:
            raise ValueError(
                f"{label}: LD-consistency screening left fewer than two "
                "variants")

        options = {
            "rounds": rounds, "window": window, "threshold": threshold,
            "eigenvalue_floor": eigenvalue_floor, "seed": seed,
            "ncores": ncores, "verbose": verbose,
        }
        stats = {}
        keep = np.asarray(_ld_consistency_screen_selected(
            cache.blocks, indices, arrays["z"], **options, progress=progress,
            progress_label=f"LD consistency screen, {label}",
            stats_out=stats))
        if keep.dtype != np.bool_ or keep.shape != (n_input,):
            raise ValueError(
                f"{label}: LD-consistency screen returned an invalid mask")
        n_kept = int(keep.sum())
        if n_kept < 2:
            raise ValueError(
                f"{label}: LD-consistency screening left fewer than two "
                "variants")

        log = deepcopy(trait.log) if isinstance(trait.log, dict) else {}
        log["screen"] = True
        log["ld_consistency_screen"] = {
            "n_input": n_input,
            "n_tested": int(stats.get("n_tested", n_input)),
            "n_untested": int(stats.get("n_untested", 0)),
            "n_kept": n_kept,
            "n_dropped": n_input - n_kept,
            "parameters": dict(options),
        }
        return PreparedTrait(
            indices=np.ascontiguousarray(indices[keep], dtype=np.int64),
            beta_hat=np.ascontiguousarray(arrays["beta_hat"][keep]),
            n_eff=np.ascontiguousarray(arrays["n_eff"][keep]),
            z=np.ascontiguousarray(arrays["z"][keep]),
            eaf=np.ascontiguousarray(arrays["eaf"][keep]),
            n_cache=n_cache, log=log)
    finally:
        # The selected-row screen is synchronous: every worker and source view
        # has gone out of scope before an mmap owner is released here.
        if owned:
            cache.close()


def _pair_prepared(
        cache, trait1, trait2, *, screen=False, screen_rounds=4,
        screen_window=1000, screen_threshold=29.72,
        screen_eigenvalue_floor=1e-3, screen_seed=0, screen_ncores=1,
        screen_verbose=False, min_af_corr=None, progress=None,
        before_screen=None, prepared_cache=True, consume_ld_cache=False):
    blocks = cache.blocks
    if consume_ld_cache and screen:
        raise ValueError(
            "consume_ld_cache=True requires screen=False; screen each "
            "prepared trait before pairing")
    ids = cache.variant_ids
    meta = cache.metadata
    n_cache = int(len(ids))
    idx1, arrays1, label1 = _validated_trait(trait1, n_cache, "trait1")
    idx2, arrays2, label2 = _validated_trait(trait2, n_cache, "trait2")
    _require_usable(trait1, "trait1")
    _require_usable(trait2, "trait2")

    joint_indices, in1, in2 = np.intersect1d(
        idx1, idx2, assume_unique=True, return_indices=True)
    ref_af = meta.get("reference_af")
    af_corr = {}
    if ref_af is not None:
        ref_af = np.asarray(ref_af, dtype=float)
        joint_ref_af = ref_af[joint_indices]
        for name, eaf in (("trait1", arrays1["eaf"][in1]),
                          ("trait2", arrays2["eaf"][in2])):
            mask = np.isfinite(eaf) & np.isfinite(joint_ref_af)
            enough = (int(mask.sum()) >= 10
                      and np.ptp(eaf[mask]) > 0.0
                      and np.ptp(joint_ref_af[mask]) > 0.0)
            af_corr[name] = (
                float(np.corrcoef(eaf[mask], joint_ref_af[mask])[0, 1])
                if enough else float("nan"))
        if min_af_corr is not None:
            try:
                threshold = float(min_af_corr)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    "min_af_corr must be a finite value in [-1, 1]") from None
            if isinstance(min_af_corr, (bool, np.bool_)):
                raise ValueError("min_af_corr must be a finite value in [-1, 1]")
            if not np.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
                raise ValueError("min_af_corr must be a finite value in [-1, 1]")
            for label, corr in ((label1, af_corr["trait1"]),
                                (label2, af_corr["trait2"])):
                if not np.isfinite(corr):
                    raise ValueError(
                        f"{label} lacks 10 jointly observed, varying finite "
                        "effect-allele frequencies; min_af_corr cannot be "
                        "evaluated")
                if corr < threshold:
                    raise ValueError(
                        f"{label} effect-allele frequency correlates "
                        f"{corr:.3f} with the LD-cache AF (required "
                        f">= {threshold:g}); the file may be on the other "
                        "allele")
    elif min_af_corr is not None:
        raise ValueError("min_af_corr requires reference_af in the LD cache")

    n_joint = int(len(joint_indices))
    if n_joint < 2:
        raise ValueError("the two GWAS share fewer than two cache variants")
    tiled = None
    try:
        if consume_ld_cache:
            tiled, joint_indices = _consume_subset_blocks(
                blocks, joint_indices, n_cache)
        else:
            tiled, joint_indices = subset_blocks(blocks, joint_indices)
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
            if before_screen is not None:
                before_screen()
            screen_keep = (
                ld_consistency_screen(
                    tiled, arrays1["z"][in1], **screen_options,
                    progress=progress,
                    progress_label="LD consistency screen, trait 1")
                & ld_consistency_screen(
                    tiled, arrays2["z"][in2], **screen_options,
                    progress=progress,
                    progress_label="LD consistency screen, trait 2"))
            n_screen_drop = n_joint - int(screen_keep.sum())
            if int(screen_keep.sum()) < 2:
                raise ValueError(
                    "LD-consistency screening left fewer than two joint variants")
            if not np.all(screen_keep):
                final_indices = joint_indices[screen_keep]
                if tiled is not blocks:
                    tiled.clear()
                tiled, kept = subset_blocks(blocks, final_indices)
                in1 = in1[screen_keep]
                in2 = in2[screen_keep]
            else:
                kept = joint_indices
        else:
            kept = joint_indices
        af = (np.asarray(ref_af, dtype=float)[kept] if ref_af is not None
              else None)
        variants = _cache_variant_table(cache)
        return PreparedBivariate(
            blocks=tiled,
            beta_hat1=np.ascontiguousarray(arrays1["beta_hat"][in1]),
            beta_hat2=np.ascontiguousarray(arrays2["beta_hat"][in2]),
            n_eff1=np.ascontiguousarray(arrays1["n_eff"][in1]),
            n_eff2=np.ascontiguousarray(arrays2["n_eff"][in2]),
            id=np.asarray(ids)[kept],
            chrom=np.asarray(variants.chrom)[kept],
            pos=np.asarray(variants.pos)[kept],
            effect_allele=np.asarray(variants.a1)[kept],
            other_allele=np.asarray(variants.a2)[kept],
            af=af,
            cache_indices=np.ascontiguousarray(kept, dtype=np.int64),
            log={
                "trait1": trait1.log, "trait2": trait2.log,
                "n_cache": n_cache, "n_joint": n_joint,
                "n_kept": int(kept.size),
                "n_screen_drop": n_screen_drop, "af_corr": af_corr,
                "screen": bool(screen), "screen_params": screen_log,
                "prepared_cache": bool(prepared_cache),
                **({"ld_cache_consumed": True} if consume_ld_cache else {}),
            },
        )
    except BaseException:
        if tiled is not None and tiled is not blocks:
            tiled.clear()
        raise


def _transfer_cache_owner(cache, owned, prep):
    """Keep mmap payloads alive in ``prep``; close ordinary owned handles."""
    if not owned:
        return prep
    if getattr(cache.blocks, "close", None) is not None:
        prep._ld_owner = cache
    else:
        cache.close()
    return prep


def pair_prepared_traits(
        ld_cache, trait1, trait2, *, screen=False, screen_rounds=4,
        screen_window=1000, screen_threshold=29.72,
        screen_eigenvalue_floor=1e-3, screen_seed=0, screen_ncores=1,
        screen_verbose=False, min_af_corr=None, progress=None,
        consume_ld_cache=False):
    """Intersect two :class:`PreparedTrait` objects and return a fit panel.

    Pairing is cheap relative to parsing and harmonization: it intersects the
    sparse cache indices, subsets the LD blocks, checks aligned allele
    frequencies, and optionally runs the pair-specific LD-consistency screen.
    New workflows that require trait-local screening can call
    :func:`screen_prepared_trait` on each input first and pair the filtered
    traits with ``screen=False``. The pair-level ``screen`` option remains for
    compatibility and deliberately retains its joint-panel semantics.
    Both traits must come from this exact LD-cache generation; ``n_cache`` and
    sparse-index structure are validated here, while a persistent caller must
    use the cache's identity or content hash as its storage key.

    ``consume_ld_cache=True`` is an explicitly destructive, low-peak option for
    a final pairing from an ordinary in-memory cache. It forms the principal
    subset block by block and closes the input cache, releasing each full
    source block before processing the next. This prevents the complete full
    panel and complete copied subset from coexisting. The option requires
    ``screen=False`` (screen each trait first), rejects memory-mapped caches,
    and requires exclusive access: no other job may be reading that cache.
    It makes a caller-supplied :class:`PreparedLDCache` unusable after a
    successful pairing or a failure after consumption starts. The default is
    non-destructive and backward compatible.

    ``progress`` is forwarded to the per-block screen. A path-loaded mmap cache
    remains owned by the returned :class:`PreparedBivariate`, as in
    :func:`prepare_bivariate_sumstats`, when ``consume_ld_cache=False``.
    """
    if not isinstance(consume_ld_cache, (bool, np.bool_)):
        raise TypeError("consume_ld_cache must be a boolean")
    consume_ld_cache = bool(consume_ld_cache)
    if consume_ld_cache and screen:
        raise ValueError(
            "consume_ld_cache=True requires screen=False; screen each "
            "prepared trait before pairing")
    _progress.validate(progress)
    cache, owned = _open_cache(ld_cache)
    try:
        if consume_ld_cache and type(cache.blocks) is not list:
            raise ValueError(
                "consume_ld_cache=True requires an ordinary in-memory LD "
                "block list; memory-mapped or custom block owners cannot be "
                "consumed")
        prep = _pair_prepared(
            cache, trait1, trait2, screen=screen,
            screen_rounds=screen_rounds, screen_window=screen_window,
            screen_threshold=screen_threshold,
            screen_eigenvalue_floor=screen_eigenvalue_floor,
            screen_seed=screen_seed, screen_ncores=screen_ncores,
            screen_verbose=screen_verbose, min_af_corr=min_af_corr,
            progress=progress, prepared_cache=not owned,
            consume_ld_cache=consume_ld_cache)
    except BaseException:
        # An empty ordinary list means destructive subsetting started. Close a
        # caller-owned handle too, so it cannot masquerade as reusable after a
        # partial allocation failure. Pre-subset validation failures leave a
        # caller-owned cache untouched.
        if owned or (consume_ld_cache and not cache.blocks):
            cache.close()
        raise
    if consume_ld_cache:
        cache.close()
        return prep
    return _transfer_cache_owner(cache, owned, prep)


def prepare_bivariate_sumstats(
        ld_cache, sumstats1, sumstats2, *, n_eff1=None, n_eff2=None,
        n_cases1=None, n_controls1=None, n_cases2=None, n_controls2=None,
        columns1=None, columns2=None, qc=True, qc_params=None, screen=False,
        screen_rounds=4, screen_window=1000, screen_threshold=29.72,
        screen_eigenvalue_floor=1e-3, screen_seed=0, screen_ncores=1,
        screen_verbose=False, min_af_corr=None, progress=None,
        reanchor_on_identifier=False):
    """Load an ldpred3 cache and two GWAS files; return a joint-fit panel.

    Both files are QC'd (optional), harmonized to the cache's counted allele,
    and converted with :func:`ldpred3.standardize_betas`. The returned blocks
    are the cache restricted to variants present in both files, re-tiled to
    ``0..m'-1``. Case/control counts become ``n_eff`` via
    :func:`ldpred3.n_eff_case_control`. A trait may take a scalar ``n_eff`` or
    case/control counts, not both.

    ``columns1`` / ``columns2`` map canonical LDpred3 fields to file columns.
    ``qc_params`` is passed to :func:`ldpred3.qc.qc_sumstats` for both traits.
    ``reanchor_on_identifier`` applies the identifier-keyed coordinate repair
    of :func:`prepare_trait_sumstats` to both traits.

    ``screen=True`` first forms the joint finite panel, then applies
    :func:`bipred.qc.ld_consistency_screen` to each trait's raw GWAS z-score
    (not ``beta_hat / se``) against that panel's principal LD submatrices.
    Missing or QC-dropped variants are never represented as zero-effect
    observations.  The ``screen_*`` arguments control that diagnostic.
    ``min_af_corr`` is a lower bound on ``corr(GWAS EAF, cache AF)`` after
    allele alignment; near −1 means the frequency column is inverted.

    ``progress``, if given, is called with one event dict per step of the
    work -- loading the reference, reading each trait, harmonizing, and, when
    ``screen=True``, once per block of each trait's LD consistency screen,
    which at genome scale dominates everything else here. It cannot change
    the result; see :mod:`bipred._progress`.

    A path-loaded memory-mapped cache remains owned by the returned object. Use
    ``with prepare_bivariate_sumstats(...) as prep:`` or call ``prep.close()``
    after fitting. A caller-supplied :class:`ldpred3.interop.PreparedLDCache`
    retains ownership instead, so its surrounding context must remain open.

    This convenience function prepares each trait through the same sparse
    boundary as :func:`prepare_trait_sumstats`, then pairs them. Call that
    function directly when one trait will be reused across several pairs.
    """
    n1 = _resolve_n_eff(n_eff1, n_cases1, n_controls1, "trait 1")
    n2 = _resolve_n_eff(n_eff2, n_cases2, n_controls2, "trait 2")
    _progress.validate(progress)
    # The screen is one coarse step here, but reports per block of its own.
    n_steps = 5 if screen else 4
    _progress.report(progress, "load LD reference", 0, n_steps,
                     unit="step")
    cache, owned = _open_cache(ld_cache)
    try:
        _progress.report(progress, "read and QC trait 1", 1, n_steps,
                         unit="step")
        trait1 = _prepare_trait(
            cache, sumstats1, n_eff=n1, qc=qc, qc_params=qc_params,
            columns=columns1, label="trait1", require_usable=False,
            reanchor_on_identifier=reanchor_on_identifier)
        _progress.report(progress, "read and QC trait 2", 2, n_steps,
                         unit="step")
        trait2 = _prepare_trait(
            cache, sumstats2, n_eff=n2, qc=qc, qc_params=qc_params,
            columns=columns2, label="trait2", require_usable=False,
            reanchor_on_identifier=reanchor_on_identifier)
        _progress.report(progress, "harmonize against the LD reference",
                         3, n_steps, unit="step")
        # A trait with zero usable variants (QC dropped all, or none matched
        # the reference) must name itself: the joint "fewer than two cache
        # variants" error below cannot say which file was unusable, and the
        # web service attributes catalog outcomes per trait from this label.
        _require_usable(trait1, "trait1")
        _require_usable(trait2, "trait2")

        def before_screen():
            _progress.report(progress, "LD consistency screen", 4,
                             n_steps, unit="step")

        prep = _pair_prepared(
            cache, trait1, trait2, screen=screen,
            screen_rounds=screen_rounds, screen_window=screen_window,
            screen_threshold=screen_threshold,
            screen_eigenvalue_floor=screen_eigenvalue_floor,
            screen_seed=screen_seed, screen_ncores=screen_ncores,
            screen_verbose=screen_verbose, min_af_corr=min_af_corr,
            progress=progress, before_screen=before_screen,
            prepared_cache=not owned,
        )
        return _transfer_cache_owner(cache, owned, prep)
    except BaseException:
        if owned:
            cache.close()
        raise
