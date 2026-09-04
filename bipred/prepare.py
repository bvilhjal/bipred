"""Pair two prepared GWAS traits against one LD cache.

Single-trait preparation -- :class:`PreparedTrait`, :func:`prepare_trait_sumstats`,
:func:`screen_prepared_trait` -- lives in :mod:`ldpred3.prepare` and is
re-exported here unchanged; it never held two-trait logic, and a univariate
pipeline should not have to import this package to run it. What this module
adds is the pairing: a :class:`PreparedBivariate` holding two traits on their
joint variant set, tiled to contiguous blocks for the bivariate sampler, with
the optional LD-consistency screen applied to that joint set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ldpred3.ld import subset_ld_blocks
from ldpred3.prepare import (  # noqa: F401  -- re-exports, plus the helpers pairing shares
    PreparedTrait,
    _cache_variant_table,
    _open_cache,
    _prepare_trait,
    _require_usable,
    _resolve_n_eff,
    _validated_trait,
    prepare_trait_sumstats,
    screen_prepared_trait,
)
from ldpred3.prepare import (  # noqa: F401
    DEFAULT_BUILD_MISMATCH_FRACTION, DEFAULT_REFERENCE_COVERAGE_WARN,
    SEVERE_REFERENCE_COVERAGE,
)
from . import _progress

__all__ = [
    "PreparedTrait", "PreparedBivariate", "prepare_trait_sumstats",
    "screen_prepared_trait", "pair_prepared_traits",
    "prepare_bivariate_sumstats", "subset_blocks",
]


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
