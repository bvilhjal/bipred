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

__all__ = ["dentist", "dentist_statistic"]

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
