"""Summary-statistic quality control against the LD reference you will fit with.

bipred does not harmonize summary statistics or build LD, and this module does
not change that. What it adds is an LD-dependent check: whether a variant's
reported effect is *consistent with the variants
correlated with it*. Every filter a user can apply beforehand -- minor allele
frequency, imputation quality, per-variant sample size --
judges a variant in isolation and therefore cannot see disagreement with its
neighbourhood. A chi-square cap is not in that list: it is an LDSC-row
filter (:func:`bipred.ldsc.ldsc_chi2_mask`), not a joint-fit mask. Neighbourhood
disagreement can make a bivariate Gibbs sampler place large opposing effects
on variants in near-perfect LD.

The LD-consistency screen is inspired by DENTIST (Chen et al. 2021,
*Nature Communications* 12:7117). It uses DENTIST's central split-half
statistic: within a window, split the variants at random into two halves and
predict each z-score in one half from the other half through the LD::

    zhat_a = R[a,B] pinv(R[B,B]) z_B
    T_a    = (z_a - zhat_a)^2 / (1 - R[a,B] pinv(R[B,B]) R[B,a])   ~ chi2_1

Variants whose observed z is far from what their neighbours predict are
dropped, and the split is repeated so each variant is tested from several
directions. This is not a reproduction of the published DENTIST pipeline: its
window construction, repeated-partition schedule, eigenvalue regularisation,
and removal policy differ. Published calibration of that full procedure does
not transfer automatically to this smaller screen.

Running the statistic against the blocks you will fit with is deliberate. Under
the default fit policy, the screen evaluates the same numeric D8, D32, or
low-rank representation: other dense floats are first rounded to D32, as in the
fitter. The legacy in-fit quantisation options create a private D8 copy that the
screen cannot replay; pre-quantise when exact alignment matters.

The default threshold is chi2_1 at p = 5e-8, which is a **null** calibration
of the split-half statistic. Split-half leverage is smaller than the
full-window leverage of published DENTIST, so the same cutoff flags a lower
|z| -- including a planted true effect at |z| ≈ 10 on AR(1) ρ = 0.9. Those
flags are not drops. Each flagged variant is confirmed against the
full-window precision-form statistic
``T_j = (Ω z)_j² / Ω_jj`` with ``Ω = (R + 0.01 I)^{-1}``, and only if it has
a neighbour with ``|r| >= 0.1``. A large z whose precision residual is no
larger than twice its own z is a private effect consistent with the LD
(APOE-scale hits included). A weak z whose sign happens to disagree with a
weak neighbour is sampling noise: under the working model ``T_j`` is
chi²_1, and the default p = 5e-8 cut keeps it. What is dropped is a
residual that is itself genome-wide significant given the neighbours --
the LDL × CAD failure mode this screen exists to catch -- not a small
effect and not a chance sign at |z| ≈ 1. The locus list :data:`APOE_HG19`
is a separate sensitivity for long-range lipid structure, not a shield
this screen needs in order to keep a consistent hit.

Blocks (or surviving live sets) smaller than :data:`MIN_WINDOW` are not
evaluated and are kept. A sparse prepared trait can therefore be largely
unscreened while the log still records that a screen ran; the returned mask
and the ``n_tested`` log field say which. Thresholded or indefinite LD
references can also manufacture a large drop rate of *null* z-scores, because
the eigenvalue floor and the leverage then disagree; that is a warning, not a
quiet success.

Why this is in bipred at all, given the package otherwise refuses to touch
summary statistics: one real LDL x CAD analysis exposed LD inconsistency that
made the bivariate fit diverge while the corresponding univariate fit remained
stable. That case motivates an explicit pre-fit screen; it does not establish a
general ordering of bivariate and univariate tolerance.

Typical use, before either :func:`bipred.ldpred3_auto_bivariate_blocks` or a
univariate fit::

    from bipred.qc import ld_consistency_screen

    keep = (ld_consistency_screen(blocks, z1)
            & ld_consistency_screen(blocks, z2))
    # then subset blocks and both traits to `keep` before fitting
"""

from __future__ import annotations

import functools
import warnings

import numpy as np

from ldpred3 import LowRankLD

from . import _progress
from ._ldpred3_compat import (
    _Q8,
    _finite_control,
    _integer_at_least,
    _validate_blocks,
    _validate_boolean_controls,
    _validate_seed,
)

__all__ = ["ld_consistency_screen", "dentist", "dentist_statistic",
           "in_long_range_ld",
           "sd_consistency", "implied_sample_size",
           "LONG_RANGE_LD_HG19", "APOE_HG19"]

#: Variants per window. The split-half uses about half of this a side, which
#: bounds the pseudo-inverse cost. A count window has no fixed physical width;
#: use long-range-LD exclusions as a separate sensitivity analysis.
DEFAULT_WINDOW = 1000
#: Windows below this are skipped: the split-half has too few variants a side
#: for the prediction to mean anything.
MIN_WINDOW = 50
#: Eigenvalues below this fraction of the largest are dropped rather than
#: inverted. Inverting a near-null direction is exactly how an ill-conditioned
#: block manufactures an enormous prediction, which would make this test
#: generate the pathology it exists to detect.
DEFAULT_EIGENVALUE_FLOOR = 1e-3
#: chi2_1 at p = 5e-8. This calibrates the split-half statistic, not the full
#: screening pipeline; validate the resulting mask for the study at hand.
#: Confirmed drops additionally require the full-window LOO statistic above
#: this threshold; LD-consistent large effects are kept.
DEFAULT_THRESHOLD = 29.72
DEFAULT_ROUNDS = 4
#: Confirm a split-half flag only when a surviving neighbour has at least
#: this absolute correlation. Otherwise ``T_j`` reduces to the association's
#: own z-score and would reject isolated genuine signals.
DEFAULT_MIN_NEIGHBOR_R = 0.1
#: Ridge on the full-window confirmation inverse, matching
#: ``ldpred3.dentist_outlier_mask``.
DEFAULT_LOO_RIDGE = 0.01
#: ``|(Ω z)_j| / |z_j|`` for an LD-consistent private effect is ~1. A
#: neighbour-inconsistent z (wrong sign relative to the LD prediction, or a
#: mismatched reference) is several times larger. A flagged variant below
#: this ratio is kept as signal, not dropped as QC.
DEFAULT_PRIVATE_Z_RATIO = 2.0
#: Warn when the screen drops at least this fraction of *tested* variants.
#: Under the working model at :data:`DEFAULT_THRESHOLD` the expected rate is
#: near zero; several percent means the LD is not the matrix the z-scores
#: were drawn against (thresholded or indefinite references are typical).
DEFAULT_DROP_FRACTION_WARN = 0.05
#: Warn when at least this fraction of variants never entered a window of
#: size :data:`MIN_WINDOW`. Untested variants are kept.
DEFAULT_UNTESTED_FRACTION_WARN = 0.05

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


def _window_ld(block, local):
    """Dense LD submatrix for ``local`` positions inside one block.

    A low-rank block is never densified in full: the submatrix of
    ``U U' + diag(d)`` is ``U[w] U[w]' + diag(d[w])``, exact for that
    representation and cheap at window scale, so a 12,000-variant block costs
    no more here than any other.
    """
    if isinstance(block, LowRankLD):
        raw_factor = np.asarray(block.U)[local]
        if raw_factor.dtype == np.int8:
            factor = raw_factor.astype(np.float64) * block.scale
        else:
            # The fitter normalises every floating factor to contiguous D32.
            # Round only the requested rows before widening, matching that
            # payload without copying the full factor.
            factor = (raw_factor.astype(np.float32).astype(np.float64)
                      * block.scale)
        out = factor @ factor.T
        out[np.diag_indices(len(local))] += np.asarray(
            block.residual_diag, dtype=np.float64)[local]
        return out
    raw = np.asarray(block)
    # Slice in the storage dtype before widening.  Casting a whole D32 block
    # here made every 1,000-variant window allocate a float64 copy of the full
    # block; on the largest public panels that temporary exceeded a gigabyte.
    window = raw[np.ix_(local, local)]
    if raw.dtype == np.int8:
        # Dense D8 uses ldpred3's round(R * 127) representation.  Treating its
        # stored integers as correlations makes an otherwise clean panel look
        # maximally inconsistent.
        return np.asarray(window, dtype=np.float64) * (1.0 / _Q8)
    # The default fitter normalises non-D8 dense input to D32 once. Cast the
    # window through float32 before widening so QC evaluates those same values
    # without allocating a full-block copy.
    return np.asarray(window, dtype=np.float32).astype(np.float64)


def _dentist_statistic(ld, z, predictors, targets, eigenvalue_floor):
    """Unchecked split-half statistic used inside the screened window loop."""
    within = ld[np.ix_(predictors, predictors)]
    across = ld[np.ix_(targets, predictors)]
    values, vectors = np.linalg.eigh(within)
    keep = values > eigenvalue_floor * max(float(values.max()), 1e-12)
    if not keep.any():
        return np.zeros(len(targets))
    retained = vectors[:, keep]
    values = values[keep]
    # ``retained / values`` scales columns, so ``across @ (retained / values)``
    # is ``(across @ retained) / values``: one ``t x p x r`` product serves both
    # the prediction and the leverage, where forming ``scaled`` separately paid
    # for that product twice. ``predicted`` then reads off the ``t x r`` result
    # rather than ``across``, which is cheaper again.
    projected = across @ retained                     # the only O(t p r) work
    predicted = projected @ ((retained.T @ z[predictors]) / values)
    # 1 - r' pinv(R) r, per target, without ever forming pinv(R).
    leverage = ((projected * projected) / values).sum(axis=1)
    return (z[targets] - predicted) ** 2 / np.clip(1.0 - leverage, 1e-6, None)


def dentist_statistic(ld, z, predictors, targets, *,
                      eigenvalue_floor=DEFAULT_EIGENVALUE_FLOOR):
    """DENTIST-inspired ``T`` for ``targets``, predicted from ``predictors``.

    ``ld`` is a dense correlation submatrix, ``z`` the matching z-scores, and
    the two index arrays are disjoint positions into both. Under the working
    Gaussian model and exact LD, each returned statistic is approximately
    chi2_1. Estimated or quantized LD and eigenvalue truncation change that
    calibration.
    """
    ld = np.asarray(ld, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if ld.ndim != 2 or ld.shape[0] != ld.shape[1] or ld.shape[0] == 0:
        raise ValueError("ld must be a non-empty square matrix")
    if z.shape != (ld.shape[0],) or not np.all(np.isfinite(z)):
        raise ValueError("z must be a matching finite vector")
    if not np.all(np.isfinite(ld)):
        raise ValueError("ld must contain only finite values")

    def indices(value, name):
        value = np.asarray(value)
        if (value.ndim != 1 or value.size == 0
                or not np.issubdtype(value.dtype, np.integer)):
            raise ValueError(f"{name} must be a non-empty integer vector")
        value = value.astype(np.int64, copy=False)
        if (np.any((value < 0) | (value >= z.size))
                or np.unique(value).size != value.size):
            raise ValueError(f"{name} contains invalid or repeated indices")
        return value

    predictors = indices(predictors, "predictors")
    targets = indices(targets, "targets")
    if np.intersect1d(predictors, targets).size:
        raise ValueError("predictors and targets must be disjoint")
    eigenvalue_floor = _finite_control(
        "eigenvalue_floor", eigenvalue_floor, lower=0.0)
    if eigenvalue_floor >= 1:
        raise ValueError("eigenvalue_floor must be < 1")
    return _dentist_statistic(
        ld, z, predictors, targets, eigenvalue_floor)


def _pool_is_worthwhile(ncores, n_blocks):
    """Whether to settle blocks concurrently, or run them one at a time.

    The pool nests over BLAS, so it is *useful* only when BLAS is pinned to one
    thread -- otherwise the parallelism already exists inside each ``eigh`` and
    a second layer merely oversubscribes the same cores.

    It is *safe* only when the loaded BLAS is reentrant, and this screen's
    concurrent call is ``np.linalg.eigh`` -- the routine ldpred3 measured
    returning silently wrong answers under an OpenMP-layer OpenBLAS. So this
    takes the conservative branch of ldpred3's gate and never nests on the
    environment-variable hint alone: without ``threadpoolctl`` installed to
    confirm reentrancy, the screen stays serial no matter what ``ncores`` says.
    """
    if ncores < 2 or n_blocks <= 1:
        return False
    # Local import: the seam pulls in ``ldpred3.ld`` on first access, and a
    # single-core screen should not pay for it.
    from ._ldpred3_compat import _blas_pool_safe
    # The flag selects the conservative branch, named there for the low-rank
    # route because that is the one built on ``eigh``. This screen is too.
    return _blas_pool_safe(True)


def _selected_window_ld(block, source_rows, local, dense_lowrank):
    """Window from selected source rows without forming their full panel.

    ``subset_ld_blocks`` normally keeps a low-rank representation.  Its one
    exception is a partial subset with fewer rows than factor columns, which it
    converts to D32.  Replaying that conversion at window scale preserves the
    representation screened by the established subset-then-screen path while
    bounding the dense allocation by ``window ** 2``.
    """
    rows = source_rows[local]
    if not dense_lowrank:
        return _window_ld(block, rows)

    raw_factor = np.asarray(block.U)[rows]
    factor = np.asarray(raw_factor, dtype=np.float32)
    if block.is_int8:
        factor = factor * np.float32(block.scale)
    out = factor @ factor.T
    out[np.diag_indices(rows.size)] += np.asarray(
        block.residual_diag)[rows].astype(np.float32, copy=False)
    return np.asarray(out, dtype=np.float32).astype(np.float64)


def _window_has_neighbor(ld, min_neighbor_r):
    """Whether each window row has an off-diagonal partner above the floor."""
    off = np.abs(ld, dtype=np.float64)
    np.fill_diagonal(off, 0.0)
    return off.max(axis=1) >= min_neighbor_r


def _precision_loo(ld, z, ridge):
    """Full-window ``T_j = (Ω z)_j² / Ω_jj`` and the precision residual ``Ω z``."""
    k = ld.shape[0]
    regularized = np.array(ld, dtype=np.float64, order="C")
    regularized.flat[::k + 1] += ridge
    try:
        omega = np.linalg.inv(regularized)
    except np.linalg.LinAlgError:
        omega = np.linalg.pinv(regularized)
    tvec = omega @ z
    stat = (tvec * tvec) / np.maximum(np.diag(omega), 1e-12)
    return stat, tvec


def _confirmed_drops(ld, z, candidates, threshold, *,
                     min_neighbor_r=DEFAULT_MIN_NEIGHBOR_R,
                     ridge=DEFAULT_LOO_RIDGE,
                     private_z_ratio=DEFAULT_PRIVATE_Z_RATIO):
    """Split-half flags that fail full-window LOO as LD-inconsistent.

    A large z whose ``|(Ω z)_j|`` is no larger than ``private_z_ratio`` times
    ``|z_j|`` is a private effect consistent with ``R``. Isolated variants
    (no neighbour above ``min_neighbor_r``) are kept because ``T_j`` then
    reduces to the association's own z-score.
    """
    if candidates.size == 0:
        return candidates
    has_nbr = _window_has_neighbor(ld, min_neighbor_r)
    stat, tvec = _precision_loo(ld, z, ridge)
    private = np.abs(tvec) <= private_z_ratio * np.abs(z)
    take = ((stat[candidates] > threshold)
            & has_nbr[candidates]
            & ~private[candidates])
    return candidates[take]


def _settle_block(task, *, z, window, threshold, eigenvalue_floor):
    """Screen one block through every round, and return its own keep-mask.

    Blocks tile disjoint variant ranges, and each round re-reads only the
    survivors of the block it is screening, so a block's whole schedule --
    every round, every window -- is a function of that block, its z-scores and
    its own random streams. Returning the block's mask rather than writing into
    a shared one keeps the serial and pooled paths identical by construction
    rather than by argument.
    """
    block, idx, source_rows, dense_lowrank, round_seeds = task
    keep = np.ones(idx.size, dtype=bool)
    tested = np.zeros(idx.size, dtype=bool)
    zb = z[idx]
    dropped = np.zeros(len(round_seeds), dtype=np.int64)
    for round_no, round_seed in enumerate(round_seeds):
        rng = np.random.default_rng(round_seed)
        live = np.where(keep)[0]
        if live.size < MIN_WINDOW:
            continue
        n_before = int(keep.sum())
        # Tile from 0, then slide the last window to live.size - window so a
        # remainder shorter than MIN_WINDOW is still tested in a full
        # neighbourhood instead of being silently treated as consistent.
        starts = list(range(0, max(live.size - window + 1, 1), window))
        tail = max(0, live.size - window)
        if starts[-1] != tail:
            starts.append(tail)
        for start in starts:
            local = live[start:start + window]
            if local.size < MIN_WINDOW:
                continue
            tested[local] = True
            if source_rows is None:
                ld = _window_ld(block, local)
            else:
                ld = _selected_window_ld(
                    block, source_rows, local, dense_lowrank)
            zw = zb[local]
            order = rng.permutation(local.size)
            half = local.size // 2
            first, second = order[:half], order[half:]
            flagged = []
            for targets, predictors in ((first, second), (second, first)):
                stat = _dentist_statistic(
                    ld, zw, predictors, targets, eigenvalue_floor)
                bad = targets[stat > threshold]
                if bad.size:
                    flagged.append(bad)
            if flagged:
                candidates = np.unique(np.concatenate(flagged))
                drop = _confirmed_drops(ld, zw, candidates, threshold)
                if drop.size:
                    keep[local[drop]] = False
        # Unique drops this round. Overlapping last-window tiles would
        # otherwise count the same variant twice in the verbose printout.
        dropped[round_no] = n_before - int(keep.sum())
    return keep, dropped, tested


def _run_consistency_screen(
        specs, z, total, *, rounds, window, threshold, eigenvalue_floor,
        seed, ncores, verbose, progress, progress_label,
        warning_stacklevel, stats_out=None):
    """Validate controls and run already validated logical block specs."""
    _validate_boolean_controls(verbose=verbose)
    _progress.validate(progress)
    seed = _validate_seed(seed)
    rounds = _integer_at_least("rounds", rounds, 1)
    window = _integer_at_least("window", window, MIN_WINDOW)
    threshold = _finite_control("threshold", threshold)
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    eigenvalue_floor = _finite_control(
        "eigenvalue_floor", eigenvalue_floor, lower=0.0)
    if eigenvalue_floor >= 1:
        raise ValueError("eigenvalue_floor must be < 1")
    ncores = _integer_at_least("ncores", ncores, 1)

    # One independent stream per (logical block, round), derived in a fixed
    # order. A selected-row logical block is exactly the nonempty block that
    # ``subset_ld_blocks`` would have emitted, so it receives the same child.
    root = np.random.SeedSequence(seed)
    tasks = [(*spec, child.spawn(rounds))
             for spec, child in zip(specs, root.spawn(len(specs)))]
    settle = functools.partial(
        _settle_block, z=z, window=window, threshold=threshold,
        eigenvalue_floor=eigenvalue_floor)

    # ``executor.map`` yields in submission order, so consuming it here
    # reports a monotone count from this thread; a callback handed to the
    # workers instead would need the caller to lock.
    def _settled(source):
        out = []
        for done, item in enumerate(source, 1):
            out.append(item)
            _progress.report(progress, progress_label, done, len(tasks),
                             unit="block")
        return out

    if _pool_is_worthwhile(ncores, len(tasks)):
        from concurrent.futures import ThreadPoolExecutor
        # Each task materialises only its own window at a time, so at most
        # ``ncores`` dense windows are live -- not ``ncores`` whole blocks.
        with ThreadPoolExecutor(max_workers=ncores) as executor:
            results = _settled(executor.map(settle, tasks))
    else:
        if ncores > 1 and len(tasks) > 1:
            # The gate can only have blocked on the BLAS conditions; say which
            # one, because "ncores=8 requested, ran serial anyway" otherwise
            # reads as a no-op flag.
            from ._ldpred3_compat import _blas_runtime_info
            threads, nested_safe = _blas_runtime_info()
            if threads is None:
                reason = ("threadpoolctl is not installed, so BLAS "
                          "reentrancy cannot be confirmed")
            elif threads != 1:
                reason = (f"the loaded BLAS is using {threads} threads "
                          "(the pool would oversubscribe them)")
            else:
                reason = "the loaded BLAS is not reentrant"
            warnings.warn(
                f"screen ncores={ncores} requested but the screen is running "
                f"serial: {reason}. Pin one BLAS thread (e.g. "
                "OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1) and install "
                "threadpoolctl to enable the block pool.", RuntimeWarning,
                stacklevel=warning_stacklevel)
        results = _settled(settle(task) for task in tasks)

    keep = np.ones(total, dtype=bool)
    tested = np.zeros(total, dtype=bool)
    per_round = np.zeros(rounds, dtype=np.int64)
    for (block_keep, dropped, block_tested), (_block, idx, _rows, _dense) in zip(
            results, specs):
        keep[idx] = block_keep
        tested[idx] = block_tested
        per_round += dropped
    n_tested = int(tested.sum())
    n_untested = int((~tested).sum())
    n_dropped = int((~keep).sum())
    if stats_out is not None:
        stats_out.update(n_tested=n_tested, n_untested=n_untested,
                         n_dropped=n_dropped)
    if n_tested and (n_dropped / n_tested) > DEFAULT_DROP_FRACTION_WARN:
        warnings.warn(
            f"LD-consistency screen dropped {n_dropped:,} of {n_tested:,} "
            f"tested variants ({n_dropped / n_tested:.0%}). At the default "
            "chi2 threshold the expected null rate is near zero; rates of "
            "several percent usually mean the LD reference is thresholded "
            "or indefinite (the eigenvalue floor and the leverage then "
            "disagree). Inspect the reference before treating the mask as QC.",
            RuntimeWarning, stacklevel=warning_stacklevel)
    if n_untested and (n_tested == 0
                       or n_untested / total >= DEFAULT_UNTESTED_FRACTION_WARN):
        warnings.warn(
            f"LD-consistency screen tested {n_tested:,} of {total:,} variants "
            f"({n_untested:,} never entered a window of size {MIN_WINDOW} and "
            "were kept unevaluated). A sparse panel can be largely unscreened "
            "while still recording that a screen ran.",
            RuntimeWarning, stacklevel=warning_stacklevel)
    if verbose:
        # Unique per-round counts: a variant dropped in an overlapping window
        # is counted once, and earlier-round casualties are not in later
        # ``live`` sets, so the running total is exact.
        remaining = total
        for round_no, count in enumerate(per_round):
            remaining -= int(count)
            print(f"  LD screen round {round_no + 1}: dropped {count:,}, "
                  f"{remaining:,} remain", flush=True)
        extra = (f" ({n_untested:,} untested, kept without evaluation)"
                 if n_untested else "")
        print(f"  LD screen tested {n_tested:,} of {total:,} variants{extra}",
              flush=True)
    return keep


def _ld_consistency_screen_selected(
        blocks, selection, z, *, rounds=DEFAULT_ROUNDS, window=DEFAULT_WINDOW,
        threshold=DEFAULT_THRESHOLD,
        eigenvalue_floor=DEFAULT_EIGENVALUE_FLOOR, seed=0, ncores=1,
        verbose=False, progress=None,
        progress_label="LD consistency screen", stats_out=None):
    """Screen a sparse principal panel without materialising that panel.

    ``selection`` is a strictly increasing vector in the full cache's index
    space, while ``z`` is already aligned to it.  Every task retains its source
    LD block plus only local row indices.  Dense work is consequently limited
    to the same window-sized matrices as :func:`ld_consistency_screen`; mmap
    views remain owned by the caller for the duration of this synchronous call.

    This is an internal preparation seam, not a second public screening API.
    Its output, seed streams, progress events, and verbose counts match first
    taking ``subset_ld_blocks(blocks, selection)`` and screening that result.
    """
    try:
        source = list(blocks)
        source_total = sum(len(idx) for _block, idx in source)
    except (TypeError, ValueError):
        raise ValueError(
            "blocks must be a sequence of (LD, index) pairs") from None
    source = _validate_blocks(source, source_total, contiguous=True)
    expected = 0
    for _block, idx in source:
        if int(idx[0]) != expected:
            raise ValueError("LD blocks must tile contiguous indices 0..m-1")
        expected += int(idx.size)

    values = np.asarray(selection)
    if (values.ndim != 1
            or not np.issubdtype(values.dtype, np.integer)
            or np.issubdtype(values.dtype, np.bool_)):
        raise ValueError("selection must be a one-dimensional integer array")
    selection = values.astype(np.int64, copy=False)
    if selection.size == 0:
        raise ValueError("selection contains no variants")
    if selection[0] < 0 or selection[-1] >= source_total:
        raise IndexError(
            f"selection indices must lie in [0, {source_total})")
    if selection.size > 1 and np.any(np.diff(selection) <= 0):
        raise ValueError("selection must be strictly increasing and unique")

    z = np.asarray(z, dtype=np.float64)
    if z.shape != (selection.size,):
        raise ValueError(
            f"z has shape {z.shape}, but selection spans "
            f"{selection.size} variants")
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values; filter them first")

    specs = []
    out_start = 0
    for block, idx in source:
        start = int(idx[0])
        stop = int(idx[-1]) + 1
        left = int(np.searchsorted(selection, start, side="left"))
        right = int(np.searchsorted(selection, stop, side="left"))
        if left == right:
            continue
        source_rows = selection[left:right] - start
        count = int(source_rows.size)
        logical = np.arange(out_start, out_start + count, dtype=np.int64)
        full = (count == idx.size and source_rows[0] == 0
                and source_rows[-1] == idx.size - 1)
        dense_lowrank = (isinstance(block, LowRankLD) and not full
                         and np.asarray(block.U).shape[1] > count)
        specs.append((block, logical, source_rows, dense_lowrank))
        out_start += count
    if out_start != selection.size:
        raise ValueError("LD blocks do not cover every selected variant")

    return _run_consistency_screen(
        specs, z, int(selection.size), rounds=rounds, window=window,
        threshold=threshold, eigenvalue_floor=eigenvalue_floor, seed=seed,
        ncores=ncores, verbose=verbose, progress=progress,
        progress_label=progress_label, warning_stacklevel=4,
        stats_out=stats_out)


def ld_consistency_screen(
        blocks, z, *, rounds=DEFAULT_ROUNDS, window=DEFAULT_WINDOW,
        threshold=DEFAULT_THRESHOLD,
        eigenvalue_floor=DEFAULT_EIGENVALUE_FLOOR, seed=0, ncores=1,
        verbose=False, progress=None, progress_label="LD consistency screen"):
    """DENTIST-inspired keep-mask over the variants ``blocks`` spans.

    This uses DENTIST's split-half statistic, but it is not the complete
    published DENTIST procedure. See the module documentation for the scope of
    the name and the differences that matter for calibration.

    Parameters
    ----------
    blocks : list of (R, idx)
        The same blocks you will fit with, indices partitioning ``0..m-1``.
        D8, D32, and low-rank values match the default fitter; other dense
        floats are normalised to D32. Legacy in-fit quantisation is not replayed.
    z : array_like (m,)
        Z-scores for one trait, ``beta / se``, in the blocks' variant order.
        Run this once per trait and intersect the masks. The split-half model
        assumes broadly comparable per-variant sample sizes within a window;
        substantial N variation needs study-specific validation or
        stratification.
    rounds : int
        Passes with fresh random splits. Outliers are removed as they are
        found, so a later pass can see variants that were masked by a bad
        neighbour in an earlier one. Every requested pass is run: a split that
        drops nothing says nothing about a later, independent split.
    ncores : int, default 1
        Screen this many blocks concurrently. Blocks are independent and each
        draws its own random splits, so the mask is identical to ``ncores=1``
        whatever the pool does. The pool nests over BLAS and is therefore taken
        **only when BLAS is pinned to one thread and ``threadpoolctl`` confirms
        the loaded library is reentrant** -- the concurrent call is
        ``np.linalg.eigh``, which is exactly the routine that miscomputes under
        a non-reentrant BLAS, so it never nests on an environment-variable
        guess. Pin BLAS (``OMP_NUM_THREADS=1``) to opt in. Peak memory rises
        from one window's dense LD to ``ncores`` of them.
    window, threshold, eigenvalue_floor, seed
        See the module constants. Each block is tiled into ``window``-sized
        chunks; the last window is slid so a short remainder is tested in a
        full neighbourhood rather than skipped.

    progress : callable or None
        Called with one event dict per finished block --
        ``{"step": progress_label, "done": blocks_done, "total": n_blocks,
        "unit": "block"}``
        -- from this thread, never from a pool worker. Blocks are the natural
        unit: each settles all of its rounds together, and at genome scale
        there are thousands of them, so the count moves steadily. Reporting
        cannot change the mask. See :mod:`bipred._progress`.
    progress_label : str
        The ``step`` name in those events. A caller screening two traits
        wants to say which one is running.

    Returns
    -------
    ndarray of bool
        ``True`` for variants to keep. Untested variants (live set smaller
        than :data:`MIN_WINDOW`) stay ``True``. Counts per round go to stdout
        under ``verbose``; because each block now runs all of its rounds
        together, those counts are reported once the screen finishes rather
        than as each round completes. High drop rates and large untested
        fractions emit :class:`RuntimeWarning`.
    """
    try:
        blocks = list(blocks)
        total = sum(len(idx) for _block, idx in blocks)
    except (TypeError, ValueError):
        raise ValueError(
            "blocks must be a sequence of (LD, index) pairs") from None
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (total,):
        raise ValueError(
            f"z has shape {z.shape}, but the blocks span {total} variants")
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values; filter them first")
    blocks = _validate_blocks(blocks, total)
    specs = [(block, idx, None, False) for block, idx in blocks]
    return _run_consistency_screen(
        specs, z, total, rounds=rounds, window=window,
        threshold=threshold, eigenvalue_floor=eigenvalue_floor, seed=seed,
        ncores=ncores, verbose=verbose, progress=progress,
        progress_label=progress_label, warning_stacklevel=3)


def dentist(blocks, z, *, rounds=DEFAULT_ROUNDS, window=DEFAULT_WINDOW,
            threshold=DEFAULT_THRESHOLD,
            eigenvalue_floor=DEFAULT_EIGENVALUE_FLOOR, seed=0, ncores=1,
            verbose=False, progress=None,
            progress_label="LD consistency screen"):
    """Compatibility name for :func:`ld_consistency_screen`.

    No warning is emitted: existing pipelines keep working, while new code can
    use the more accurate name without implying the full published procedure.
    """
    return ld_consistency_screen(
        blocks, z, rounds=rounds, window=window, threshold=threshold,
        eigenvalue_floor=eigenvalue_floor, seed=seed, ncores=ncores,
        verbose=verbose, progress=progress, progress_label=progress_label)


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
