"""MiXeR-style polygenic-overlap recovery — realistic LD.

The bivariate LDpred3 sampler fits a four-state causal mixture
``(pi00, pi10, pi01, pi11)`` = neither / trait-1-only / trait-2-only / both
causal. That is exactly the bivariate causal-mixture model MiXeR (Frei et al.
2019) uses, so ``BivariateResult.mixer`` reports the same quantities:
per-trait polygenicity, the shared-causal fraction (polygenic overlap), the
within-shared effect correlation ``rho_beta`` and the overlap decomposition of
``rg`` (``rho_beta * pi11 / sqrt(pi1 pi2)``).

This benchmark stress-tests those readouts against known mixture parameters on
realistic non-repeating coalescent LD (reusing ``rg_architectures``' cached
segments and finite reference panels). Genetic-correlation estimates are scored
against each replicate's realized population-LD correlation. Eight sweeps use
fresh-phenotype replicates on fixed genotypes:

  * ``overlap``  -- vary the shared-causal fraction 0..1 at fixed per-trait
    polygenicity; the headline MiXeR quantity. Checks frac_shared + rg tracking.
  * ``rho``      -- vary the within-shared effect correlation over a *signed*
    grid; checks rho_beta and that rg = rho_beta * overlap, and that a negative
    correlation is returned negative rather than folded.
  * ``polygenicity`` -- vary the per-trait causal fraction with the overlap held
    fixed. Every other sweep here sits at p=0.10, so the ``overlap`` sweep's
    upward bias in ``frac_shared`` could belong to the estimator or to that one
    architecture. It is **not** a constant: the bias is a function of
    polygenicity, small where the causal set is sparse and large where it is
    dense, which is what decides whether a reported overlap can be read at all.
  * ``asymmetry`` -- vary trait 2's causal fraction against a fixed trait 1, so
    the traits are *unequally* polygenic. Every other sweep sets pi1 = pi2,
    where ``frac_shared = pi11 / min(pi1, pi2)`` cannot be told apart from
    ``pi11 / pi1`` or from the ``sqrt(pi1 pi2)`` denominator of
    ``rg_from_overlap``. Includes the containment arm, where the sparse trait
    lies wholly inside the dense one: a complete overlap that still implies
    only ``rho_beta * sqrt(pi2 / pi1)`` of genetic correlation.
  * ``power``    -- vary N (hence N*h2/M) at fixed architecture; shows how much
    signal the overlap estimate needs to be meaningful.
  * ``ldmatch``  -- fit the same data on the finite reference panel vs the **exact
    in-sample (population) LD**; separates the **LD-reference-mismatch** part (the
    ref-minus-in-sample gap) from behavior that persists under matched in-sample
    LD.
  * ``calibration`` -- on the (realistic) finite reference panel, compares the
    naive count with the **noise-inflation fix** (``noise_inflation=True``): the
    learned per-trait lambda changes the fitted polygenicity, and reports whether
    the empirical 95% retained-iterate interval includes the true causal count.
  * ``unical`` -- **univariate-anchored calibration** (``res.mixer_calibrated``):
    the four-state count over-counts *more* than a univariate fit (LD-spreading is
    amplified across the four states), so this swaps the joint per-trait counts for
    two univariate ``ldpred3_auto_infer`` runs while keeping the joint shared
    fraction. Reports the joint and calibrated per-trait and shared counts.

The CSV records relative polygenicity, target and realized genetic correlation,
paired rg error, and estimator variability. Interpret the refreshed artifact
rather than assuming calibration or robustness from model structure alone.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/mixer_overlap.py

Env overrides: ``SWEEP``
(overlap,rho,polygenicity,asymmetry,power,ldmatch,calibration,unical
or a subset), ``REPS``,
``OUT``, plus ``NB`` / ``K`` / ``MUT_RATE`` (via rg_architectures) to change ``m``.
"""
import os
import sys
import csv
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rg_architectures as R                                    # noqa: E402
from bipred import ldpred3_auto_bivariate_blocks                # noqa: E402
from ldpred3 import ldpred3_auto_infer                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M = R.M
H2 = 0.5
REPS = int(os.environ.get("REPS", "8"))
BURN, ITER = 200, 300

# Per-trait causal count for the overlap / rho / power sweeps (fixed p per trait).
NCAUSAL = max(int(round(0.10 * M)), 20)     # p = 0.10 per trait

# Per-trait causal fractions for the polygenicity sweep, spanning sparse to
# dense on the same panel. All four land on an exact count at this m.
P_GRID = (0.01, 0.03, 0.10, 0.30)

# The exact population LD the sumstats are generated from (the in-sample / oracle
# LD, zero reference mismatch), for the ld-match control sweep.
TRUE_BLOCKS = [(R.POP_R[b].astype(np.float32), R.IDX[b]) for b in range(R.NB)]


def _sim_mixture(rng, n1_causal, n2_causal, n_shared, rho_beta):
    """Two traits with exactly ``n1_causal`` / ``n2_causal`` causal SNPs,
    ``n_shared`` of them in common, shared effects correlated ``rho_beta``;
    each trait scaled to h2=H2.

    The per-trait counts may differ. ``frac_shared`` in the returned truth
    follows the estimator's convention, ``pi11 / min(pi1, pi2)`` -- the shared
    fraction *of the less polygenic trait* -- so a sparse trait wholly inside a
    dense one is 1.0 however much larger the dense trait is. The rg target uses
    the other denominator, ``sqrt(pi1 pi2)``; the two coincide only when the
    traits are equally polygenic.

    Returns the effects and their exact mixture truth. The LD-weighted rg
    realized by finite effects is computed separately."""
    n_uniq1 = n1_causal - n_shared
    n_uniq2 = n2_causal - n_shared
    if min(n_uniq1, n_uniq2, n_shared) < 0:
        raise ValueError("n_shared cannot exceed either per-trait causal count")
    need = n_shared + n_uniq1 + n_uniq2
    picks = rng.choice(M, need, replace=False)
    shared = picks[:n_shared]
    u1 = picks[n_shared:n_shared + n_uniq1]
    u2 = picks[n_shared + n_uniq1:]
    b1 = np.zeros(M)
    b2 = np.zeros(M)
    b1[u1] = rng.standard_normal(n_uniq1)
    b2[u2] = rng.standard_normal(n_uniq2)
    if n_shared:
        L = np.linalg.cholesky([[1.0, rho_beta], [rho_beta, 1.0]])
        raw = L @ rng.standard_normal((2, n_shared))
        b1[shared] = raw[0]
        b2[shared] = raw[1]
    b1 *= np.sqrt(H2 / R.gv(b1, b1))
    b2 *= np.sqrt(H2 / R.gv(b2, b2))
    pi1 = n1_causal / M
    pi2 = n2_causal / M
    pi11 = n_shared / M
    # Both from the integer counts actually simulated, not from the requested
    # fraction. ``n_shared`` is a rounded count, so on a sparse grid the two
    # differ: 0.75 of 50 causal variants is 38 shared, a realized 0.76. The
    # requested value was recorded as truth until 2026-09-14, which overstated
    # the reported bias by up to 0.01 in exactly the sparse cells the
    # polygenicity sweep's claim rests on.
    frac_shared = n_shared / min(n1_causal, n2_causal)
    # Generating target under independent variants and equal h2. Finite effects
    # under LD generally realize a different genetic correlation.
    rg_target = rho_beta * n_shared / np.sqrt(n1_causal * n2_causal)
    truth = {"pi1": pi1, "pi2": pi2, "pi11": pi11,
             "frac_shared": frac_shared,
             "rho_beta": rho_beta,
             "rg_target": float(rg_target)}
    return b1, b2, truth


def _sim_overlap(rng, n_causal, frac_shared, rho_beta):
    """Equal-polygenicity case: both traits have ``n_causal`` causal SNPs and
    ``frac_shared`` of them are shared. Draw order matches :func:`_sim_mixture`
    exactly, so the symmetric sweeps are unaffected by the generalisation."""
    n_shared = int(round(frac_shared * n_causal))
    return _sim_mixture(rng, n_causal, n_causal, n_shared, rho_beta)


def _fit(ref, b1, b2, n1, n2, rep):
    bh1, bh2 = R.sumstats_pair(b1, b2, n1, n2, np.random.default_rng(50000 + rep))
    return ldpred3_auto_bivariate_blocks(ref, bh1, bh2, n1, n2,
                                         burn_in=BURN, num_iter=ITER, seed=rep)


def _cell(n_causal, frac_shared, rho_beta, n1, n2, base_seed, n2_causal=None):
    """Average the mixer readouts over REPS fresh phenotypes on fixed genotypes.

    ``n2_causal`` makes the two traits unequally polygenic; ``frac_shared`` is
    then the fraction of the *sparser* trait that is shared, matching the
    estimator's ``pi11 / min(pi1, pi2)``. Left at ``None`` the cell is the
    symmetric one and draws exactly as before."""
    fs, rb, rg, rgo, rel1, rel2, realized = [], [], [], [], [], [], []
    truth = None
    for rep in range(REPS):
        ref, _ = R.ref_panel(rep)
        rng = np.random.default_rng(base_seed + rep)
        if n2_causal is None:
            b1, b2, truth = _sim_overlap(rng, n_causal, frac_shared, rho_beta)
        else:
            n_shared = int(round(frac_shared * min(n_causal, n2_causal)))
            b1, b2, truth = _sim_mixture(rng, n_causal, n2_causal, n_shared,
                                         rho_beta)
        realized.append(R.realized_rg(b1, b2))
        res = _fit(ref, b1, b2, n1, n2, rep)
        mx = res.mixer
        fs.append(mx["frac_shared"])
        rb.append(mx["rho_beta"])
        rg.append(res.rg)
        rgo.append(mx["rg_from_overlap"])
        rel1.append(mx["polygenicity"][0] / truth["pi1"])
        rel2.append(mx["polygenicity"][1] / truth["pi2"])
    m = lambda a: round(float(np.mean(a)), 3)      # noqa: E731
    s = lambda a: round(float(np.std(a)), 3)       # noqa: E731
    realized = np.asarray(realized, float)
    rg = np.asarray(rg, float)
    rgo = np.asarray(rgo, float)
    return {"frac_shared_hat": m(fs), "frac_shared_sd": s(fs),
            "rho_beta_hat": m(rb), "rho_beta_sd": s(rb),
            "rg_hat": m(rg), "rg_sd": s(rg),
            "rg_mae_realized": m(np.abs(rg - realized)),
            "rg_overlap_hat": m(rgo), "rg_overlap_sd": s(rgo),
            "rg_overlap_mae_realized": m(np.abs(rgo - realized)),
            "rg_realized": m(realized), "rg_realized_sd": s(realized),
            "rel_poly": m(rel1 + rel2), "rel_poly_sd": s(rel1 + rel2),
            # Per-trait as well as pooled: under asymmetry the two traits are
            # recovered differently and the pooled mean hides which one moved.
            "rel_poly1": m(rel1), "rel_poly1_sd": s(rel1),
            "rel_poly2": m(rel2), "rel_poly2_sd": s(rel2),
            "true_pi1": round(truth["pi1"], 4),
            "true_pi2": round(truth["pi2"], 4),
            "true_pi11": round(truth["pi11"], 4),
            "poly_ratio": round(truth["pi1"] / truth["pi2"], 3),
            "frac_shared_target": round(truth["frac_shared"], 3),
            "rho_beta_target": round(truth["rho_beta"], 3),
            "rg_target": round(truth["rg_target"], 3)}


def sweep_overlap(rows):
    print(f"\n== overlap sweep (p=0.10/trait, rho_beta=0.8, N={R.N1}/{R.N2}) ==",
          flush=True)
    print(f"{'frac_shared':>11} | {'est':>13} | {'rho_beta':>13} | "
          f"{'rg target/real':>14} {'rg_hat':>13} {'rg_ovl':>6}", flush=True)
    for i, frac in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        r = _cell(NCAUSAL, frac, 0.8, R.N1, R.N2, base_seed=1000 + 20 * i)
        r["sweep"] = "overlap"
        rows.append(r)
        print(f"{frac:>11.2f} | {r['frac_shared_hat']:>6.2f}±{r['frac_shared_sd']:<5} "
              f"| {r['rho_beta_hat']:>6.2f}±{r['rho_beta_sd']:<5} | "
              f"{r['rg_target']:>5.2f}/{r['rg_realized']:<5.2f} "
              f"{r['rg_hat']:>6.2f}±{r['rg_sd']:<5} "
              f"{r['rg_overlap_hat']:>6.2f}", flush=True)


def sweep_rho(rows):
    """Within-shared effect correlation, including negative values.

    The grid is signed because a sign error in the covariance readout would be
    invisible on a non-negative grid, and real pairs are routinely negative
    (HDL x TG sits near -0.5). ``rho_beta`` and ``rg`` should track the target
    with the sign preserved and no asymmetry in magnitude."""
    print(f"\n== rho_beta sweep (p=0.10/trait, frac_shared=0.5, N={R.N1}/{R.N2}) ==",
          flush=True)
    print(f"{'rho_beta':>8} | {'rho_beta_hat':>13} | {'frac_shared':>13} | "
          f"{'rg target/real':>14} {'rg_hat':>13}", flush=True)
    for i, rho in enumerate([-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]):
        r = _cell(NCAUSAL, 0.5, rho, R.N1, R.N2, base_seed=2000 + 20 * i)
        r["sweep"] = "rho"
        rows.append(r)
        print(f"{rho:>8.2f} | {r['rho_beta_hat']:>6.2f}±{r['rho_beta_sd']:<5} "
              f"| {r['frac_shared_hat']:>6.2f}±{r['frac_shared_sd']:<5} | "
              f"{r['rg_target']:>5.2f}/{r['rg_realized']:<5.2f} "
              f"{r['rg_hat']:>6.2f}±{r['rg_sd']:<5}", flush=True)


def sweep_polygenicity(rows):
    """How the ``frac_shared`` bias moves with per-trait polygenicity.

    The other sweeps hold p at 0.10, so the overlap sweep alone cannot say
    whether its upward bias is a fixed property of the estimator. Vary p over two
    orders of magnitude with the overlap target held at three values, and the
    bias is not fixed: it tracks polygenicity. Read the ``bias`` column against
    the ``sd`` beside it -- a bias inside one sampling SD is not a finding, and
    the point of the sweep is where that stops being true.
    """
    print("\n== polygenicity sweep (rho_beta=0.8, "
          f"N={R.N1}/{R.N2}) ==", flush=True)
    print(f"{'p/trait':>8} {'causal':>7} {'target':>7} | {'frac_shared':>13} "
          f"{'bias':>7} | {'rel_poly':>8} | {'rg real/hat':>13}", flush=True)
    for i, p in enumerate(P_GRID):
        n_causal = max(int(round(p * M)), 20)
        for j, frac in enumerate([0.25, 0.5, 0.75]):
            r = _cell(n_causal, frac, 0.8, R.N1, R.N2,
                      base_seed=7000 + 200 * i + 20 * j)
            r["sweep"] = "polygenicity"
            # The requested p is recovered exactly by true_pi1 = n_causal / M,
            # so the grid needs no column of its own.
            rows.append(r)
            print(f"{p:>8.2f} {n_causal:>7} {frac:>7.2f} | "
                  f"{r['frac_shared_hat']:>6.2f}±{r['frac_shared_sd']:<6} "
                  f"{r['frac_shared_hat'] - frac:>+7.3f} | "
                  f"{r['rel_poly']:>8.2f} | "
                  f"{r['rg_realized']:>5.2f}/{r['rg_hat']:<5.2f}", flush=True)
        mean_bias = float(np.mean([row["frac_shared_hat"]
                                   - row["frac_shared_target"]
                                   for row in rows
                                   if row["sweep"] == "polygenicity"
                                   and row["true_pi1"] == round(n_causal / M, 3)]))
        print(f"{'':8} {'':7} {'mean':>7} | {'':13} {mean_bias:>+7.3f}",
              flush=True)


def sweep_asymmetry(rows):
    """Unequally polygenic traits -- the case the other sweeps cannot reach.

    Every other sweep here sets pi1 = pi2, where the estimator's
    ``frac_shared = pi11 / min(pi1, pi2)`` is indistinguishable from
    ``pi11 / pi1`` or from the ``sqrt(pi1 pi2)`` denominator that
    ``rg_from_overlap`` uses. Holding trait 1 at p=0.10 and thinning trait 2 to
    a tenth of that separates them, and covers the configuration real pairs
    actually take: a dense trait against a sparse one.

    The ``frac_shared=1.0`` arm is containment -- every causal variant of the
    sparse trait is also causal for the dense one. It is the case most often
    misread: the overlap is complete by construction, yet the genetic
    correlation it implies is only ``rho_beta * sqrt(pi2 / pi1)``, so at a
    10x polygenicity ratio a *complete* overlap still means rg about 0.25.
    Reported here side by side so the artifact shows both numbers at once."""
    print("\n== asymmetry sweep (trait1 p=0.10, rho_beta=0.8, "
          f"N={R.N1}/{R.N2}) ==", flush=True)
    print(f"{'p2':>6} {'ratio':>6} {'shared':>7} | {'frac_shared t/est':>18} | "
          f"{'rel poly 1 / 2':>15} | {'rg t/real/hat':>19}", flush=True)
    n1_causal = NCAUSAL
    for i, p2 in enumerate((0.10, 0.05, 0.02, 0.01)):
        n2_causal = max(int(round(p2 * M)), 20)
        for j, frac in enumerate((0.5, 1.0)):
            r = _cell(n1_causal, frac, 0.8, R.N1, R.N2,
                      base_seed=8000 + 200 * i + 20 * j, n2_causal=n2_causal)
            r["sweep"] = "asymmetry"
            rows.append(r)
            print(f"{p2:>6.2f} {r['poly_ratio']:>6.1f} "
                  f"{int(round(r['true_pi11'] * M)):>7} | "
                  f"{r['frac_shared_target']:>8.2f}/"
                  f"{r['frac_shared_hat']:<5.2f}±{r['frac_shared_sd']:<4} | "
                  f"{r['rel_poly1']:>6.2f} /{r['rel_poly2']:>6.2f} | "
                  f"{r['rg_target']:>5.2f}/{r['rg_realized']:>5.2f}/"
                  f"{r['rg_hat']:<5.2f}±{r['rg_sd']:<4}", flush=True)


def sweep_power(rows):
    print("\n== power sweep (p=0.10/trait, frac_shared=0.5, rho_beta=0.8) ==",
          flush=True)
    print(f"{'N':>8} {'Nh2/M':>6} | {'frac_shared':>13} | {'rho_beta':>13} | "
          f"{'rg real/hat':>13} | {'rel_poly':>8}", flush=True)
    for i, n in enumerate([1000, 2500, 5000, 10000, 20000]):
        r = _cell(NCAUSAL, 0.5, 0.8, n, n, base_seed=3000 + 20 * i)
        r["sweep"] = "power"
        r["N"] = n
        rows.append(r)
        print(f"{n:>8} {n*H2/M:>6.2f} | {r['frac_shared_hat']:>6.2f}±{r['frac_shared_sd']:<5} "
              f"| {r['rho_beta_hat']:>6.2f}±{r['rho_beta_sd']:<5} | "
              f"{r['rg_realized']:>5.2f}/{r['rg_hat']:<5.2f} | "
              f"{r['rel_poly']:>8.2f}", flush=True)


def sweep_ldmatch(rows):
    """Separate the LD-reference-mismatch part from the LD-spreading part: fit the
    same data on the finite reference panel vs the exact in-sample (population) LD.
    The ref-minus-in-sample gap is the mismatch part; the in-sample count itself
    (which stays inflated at low power) is the LD-spreading part."""
    print("\n== LD-match control (p=0.10/trait, frac_shared=0.5, rho_beta=0.8) ==",
          flush=True)
    print(f"{'N':>8} {'Nh2/M':>6} | {'rg real':>7} | "
          f"{'ref pi/true':>12} {'ref rg':>7} | "
          f"{'insample pi/true':>16} {'ins rg':>7}", flush=True)
    for i, n in enumerate([1000, 2500, 5000, 10000, 20000]):
        rp, rr, tp, tr, realized = [], [], [], [], []
        for rep in range(REPS):
            ref, _ = R.ref_panel(rep)
            rng = np.random.default_rng(4000 + 20 * i + rep)
            b1, b2, truth = _sim_overlap(rng, NCAUSAL, 0.5, 0.8)
            realized.append(R.realized_rg(b1, b2))
            bh1, bh2 = R.sumstats_pair(b1, b2, n, n, rng)
            res_r = ldpred3_auto_bivariate_blocks(ref, bh1, bh2, n, n,
                                                  burn_in=BURN, num_iter=ITER, seed=rep)
            res_t = ldpred3_auto_bivariate_blocks(TRUE_BLOCKS, bh1, bh2, n, n,
                                                  burn_in=BURN, num_iter=ITER, seed=rep)
            relr = 0.5 * sum(res_r.mixer["polygenicity"]) / truth["pi1"]
            relt = 0.5 * sum(res_t.mixer["polygenicity"]) / truth["pi1"]
            rp.append(relr); rr.append(res_r.rg)
            tp.append(relt); tr.append(res_t.rg)
        m = lambda a: round(float(np.mean(a)), 3)      # noqa: E731
        s = lambda a: round(float(np.std(a)), 3)       # noqa: E731
        realized = np.asarray(realized, float)
        rr = np.asarray(rr, float)
        tr = np.asarray(tr, float)
        r = {"sweep": "ldmatch", "N": n,
             "ref_relpoly": m(rp), "ref_relpoly_sd": s(rp),
             "ref_rg": m(rr), "ref_rg_sd": s(rr),
             "ref_rg_mae_realized": m(np.abs(rr - realized)),
             "insample_relpoly": m(tp), "insample_relpoly_sd": s(tp),
             "insample_rg": m(tr), "insample_rg_sd": s(tr),
             "insample_rg_mae_realized": m(np.abs(tr - realized)),
             "rg_target": round(truth["rg_target"], 3),
             "rg_realized": m(realized), "rg_realized_sd": s(realized)}
        rows.append(r)
        print(f"{n:>8} {n*H2/M:>6.2f} | {r['rg_realized']:>7.2f} | "
              f"{r['ref_relpoly']:>6.2f}±{r['ref_relpoly_sd']:<5}"
              f" {r['ref_rg']:>7.2f} | {r['insample_relpoly']:>10.2f}"
              f"±{r['insample_relpoly_sd']:<5} {r['insample_rg']:>7.2f}", flush=True)


def sweep_calibration(rows):
    """Absolute-count calibration and retained-iterate interval inclusion on the
    finite reference panel (the realistic, mismatched-LD case), across power.

    Compares ``noise_inflation=False`` and ``True``. The latter learns a
    per-trait lambda >= 1 from residual misfit. Also reports whether the empirical
    95% interval from ``mixer_iterate_summary`` includes the true per-trait
    causal count. This is not Bayesian credible-interval coverage."""
    print("\n== count calibration + retained-iterate inclusion (ref-panel LD, "
          "p=0.10/trait, frac_shared=0.5, rho_beta=0.8) ==", flush=True)
    print(f"{'N':>8} {'Nh2/M':>6} | {'rel off':>7} {'rel ON':>6} {'lam':>5} | "
          f"{'hit off':>7} {'hit ON':>6} | {'rg real':>7} {'rg off':>6} {'rg ON':>6}",
          flush=True)
    true_n1 = NCAUSAL
    for i, n in enumerate([1000, 2500, 5000, 10000, 20000]):
        ro, rn, lam, covo, covn, rgo, rgn, realized = [], [], [], 0, 0, [], [], []
        for rep in range(REPS):
            ref, _ = R.ref_panel(rep)
            rng = np.random.default_rng(5000 + 20 * i + rep)
            b1, b2, truth = _sim_overlap(rng, NCAUSAL, 0.5, 0.8)
            realized.append(R.realized_rg(b1, b2))
            bh1, bh2 = R.sumstats_pair(b1, b2, n, n, rng)
            off = ldpred3_auto_bivariate_blocks(ref, bh1, bh2, n, n, burn_in=BURN,
                                                num_iter=ITER, seed=rep)
            on = ldpred3_auto_bivariate_blocks(ref, bh1, bh2, n, n, burn_in=BURN,
                                               num_iter=ITER, noise_inflation=True,
                                               seed=rep)
            ro.append(0.5 * sum(off.mixer["polygenicity"]) / truth["pi1"])
            rn.append(0.5 * sum(on.mixer["polygenicity"]) / truth["pi1"])
            lam.append(0.5 * (on.noise_scale[0] + on.noise_scale[1]))
            rgo.append(off.rg); rgn.append(on.rg)
            for res, hit in ((off, "o"), (on, "n")):
                interval = res.mixer_iterate_summary()["n_causal"][0]["interval"]
                covered = interval[0] <= true_n1 <= interval[1]
                if hit == "o":
                    covo += covered
                else:
                    covn += covered
        m = lambda a: round(float(np.mean(a)), 3)  # noqa: E731
        s = lambda a: round(float(np.std(a)), 3)   # noqa: E731
        realized = np.asarray(realized, float)
        rgo = np.asarray(rgo, float)
        rgn = np.asarray(rgn, float)
        # Keep the historical cov_* CSV columns so regenerated artifacts remain
        # comparable. They now mean empirical iterate-interval inclusion, not
        # posterior/credible-interval coverage.
        r = {"sweep": "calibration", "N": n,
             "rg_target": round(truth["rg_target"], 3),
             "rg_realized": m(realized), "rg_realized_sd": s(realized),
             "rel_off": m(ro), "rel_on": m(rn), "lam": m(lam),
             "cov_off": covo / REPS, "cov_on": covn / REPS,
             "rg_off": m(rgo), "rg_off_sd": s(rgo),
             "rg_off_mae_realized": m(np.abs(rgo - realized)),
             "rg_on": m(rgn), "rg_on_sd": s(rgn),
             "rg_on_mae_realized": m(np.abs(rgn - realized))}
        rows.append(r)
        print(f"{n:>8} {n*H2/M:>6.2f} | {r['rel_off']:>7.2f} {r['rel_on']:>6.2f} "
              f"{r['lam']:>5.2f} | {covo}/{REPS:<5} {covn}/{REPS:<4} | "
              f"{r['rg_realized']:>7.2f} {r['rg_off']:>6.2f} {r['rg_on']:>6.2f}",
              flush=True)


def sweep_unical(rows):
    """Univariate-anchored count calibration (``res.mixer_calibrated``) vs truth.

    ``mixer_calibrated`` replaces the joint per-trait polygenicities with two
    univariate ``ldpred3_auto_infer`` estimates, retains the joint shared
    fraction, and rebuilds ``n_shared`` on that scale. This reports joint and
    calibrated counts against the known counts without presuming improvement."""
    ureps = min(REPS, 6)
    print("\n== univariate-anchored calibration (ref-panel LD, p=0.10/trait, "
          f"frac_shared=0.5, rho_beta=0.8, {ureps} reps) ==", flush=True)
    print(f"{'N':>8} {'Nh2/M':>6} | {'joint n1/t':>10} {'calib n1/t':>10} | "
          f"{'joint sh/t':>10} {'calib sh/t':>10} | {'rg real/hat':>11}", flush=True)
    true_n1 = NCAUSAL
    true_nsh = max(int(round(0.5 * NCAUSAL)), 1)
    for i, n in enumerate([1000, 2500, 5000, 10000, 20000]):
        jp, cp, jsh, csh, rgv, realized = [], [], [], [], [], []
        for rep in range(ureps):
            ref, _ = R.ref_panel(rep)
            rng = np.random.default_rng(6000 + 20 * i + rep)
            b1, b2, truth = _sim_overlap(rng, NCAUSAL, 0.5, 0.8)
            realized.append(R.realized_rg(b1, b2))
            bh1, bh2 = R.sumstats_pair(b1, b2, n, n, rng)
            res = ldpred3_auto_bivariate_blocks(ref, bh1, bh2, n, n,
                                                burn_in=BURN, num_iter=ITER, seed=rep)
            inf1 = ldpred3_auto_infer(ref, bh1, float(n), n_chains=6,
                                      burn_in=150, num_iter=150, seed=rep)
            inf2 = ldpred3_auto_infer(ref, bh2, float(n), n_chains=6,
                                      burn_in=150, num_iter=150, seed=rep)
            jm = res.mixer
            cal = res.mixer_calibrated(inf1, inf2)
            jp.append(0.5 * (jm["n_causal"][0] + jm["n_causal"][1]) / true_n1)
            cp.append(0.5 * (cal["n_causal"][0] + cal["n_causal"][1]) / true_n1)
            jsh.append(jm["n_shared"] / true_nsh)
            csh.append(cal["n_shared"] / true_nsh)
            rgv.append(res.rg)
        m = lambda a: round(float(np.mean(a)), 3)      # noqa: E731
        s = lambda a: round(float(np.std(a)), 3)       # noqa: E731
        realized = np.asarray(realized, float)
        rgv = np.asarray(rgv, float)
        r = {"sweep": "unical", "N": n,
             "rg_target": round(truth["rg_target"], 3),
             "rg_realized": m(realized), "rg_realized_sd": s(realized),
             "joint_relpoly": m(jp), "joint_relpoly_sd": s(jp),
             "calib_relpoly": m(cp), "calib_relpoly_sd": s(cp),
             "joint_relshared": m(jsh), "calib_relshared": m(csh),
             "rg_hat": m(rgv), "rg_sd": s(rgv),
             "rg_mae_realized": m(np.abs(rgv - realized))}
        rows.append(r)
        print(f"{n:>8} {n*H2/M:>6.2f} | "
              f"{r['joint_relpoly']:>5.2f}±{r['joint_relpoly_sd']:<4} "
              f"{r['calib_relpoly']:>5.2f}±{r['calib_relpoly_sd']:<4} | "
              f"{r['joint_relshared']:>10.2f} {r['calib_relshared']:>10.2f} | "
              f"{r['rg_realized']:>5.2f}/{r['rg_hat']:<5.2f}", flush=True)


SWEEPS = {"overlap": sweep_overlap, "rho": sweep_rho,
          "polygenicity": sweep_polygenicity, "asymmetry": sweep_asymmetry,
          "power": sweep_power,
          "ldmatch": sweep_ldmatch, "calibration": sweep_calibration,
          "unical": sweep_unical}


def _write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ov = [r for r in rows if r["sweep"] == "overlap"]
    rh = [r for r in rows if r["sweep"] == "rho"]
    pg = [r for r in rows if r["sweep"] == "polygenicity"]
    asy = [r for r in rows if r["sweep"] == "asymmetry"]
    pw = [r for r in rows if r["sweep"] == "power"]
    lm = [r for r in rows if r["sweep"] == "ldmatch"]
    uc = [r for r in rows if r["sweep"] == "unical"]
    npan = sum(bool(g) for g in (ov, rh, pg, asy, pw, lm, uc))
    if npan == 0:
        return
    fig, ax = plt.subplots(1, npan, figsize=(3.7 * npan, 3.6))
    ax = [ax] if npan == 1 else list(ax)
    panels = iter(ax)

    if ov:
        a = next(panels)
        x = [r["frac_shared_target"] for r in ov]
        a.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5)
        a.errorbar(x, [r["frac_shared_hat"] for r in ov],
                   [r["frac_shared_sd"] for r in ov], fmt="o-", ms=4, capsize=2,
                   color="C0", label="frac_shared")
        a.errorbar([r["rg_realized"] for r in ov], [r["rg_hat"] for r in ov],
                   [r["rg_sd"] for r in ov], fmt="s-", ms=4, capsize=2,
                   color="C3", label="rg")
        a.set_xlabel("target frac_shared / mean realized rg")
        a.set_ylabel("estimated")
        a.set_title("overlap sweep")
        a.legend(fontsize=8)
    if rh:
        a = next(panels)
        x = [r["rho_beta_target"] for r in rh]
        # The grid is signed, so the identity line has to span it; a 0..1 line
        # left the negative half of the panel without a reference.
        lo = min([*x, *(r["rg_realized"] for r in rh)])
        hi = max([*x, *(r["rg_realized"] for r in rh)])
        a.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=.5)
        a.errorbar(x, [r["rho_beta_hat"] for r in rh],
                   [r["rho_beta_sd"] for r in rh], fmt="o-", ms=4, capsize=2,
                   color="C0", label="rho_beta")
        a.errorbar([r["rg_realized"] for r in rh], [r["rg_hat"] for r in rh],
                   [r["rg_sd"] for r in rh], fmt="s-", ms=4, capsize=2,
                   color="C3", label="rg")
        a.set_xlabel("target rho_beta / mean realized rg")
        a.set_title("rho_beta sweep")
        a.legend(fontsize=8)
    if pg:
        a = next(panels)
        a.axhline(0.0, ls=":", c="k", lw=1, alpha=.6)
        for frac, colour in zip((0.25, 0.5, 0.75), ("C0", "C2", "C3")):
            cells = [r for r in pg if r["frac_shared_target"] == frac]
            a.errorbar([r["true_pi1"] for r in cells],
                       [r["frac_shared_hat"] - frac for r in cells],
                       [r["frac_shared_sd"] for r in cells],
                       fmt="o-", ms=4, capsize=2, color=colour,
                       label=f"target {frac:g}")
        a.set_xscale("log")
        a.set_xlabel("causal fraction per trait")
        a.set_ylabel("frac_shared: est − true")
        a.set_title("polygenicity sweep")
        a.legend(fontsize=8)
    if asy:
        a = next(panels)
        a.axhline(1.0, ls=":", c="k", lw=1, alpha=.6)
        for frac, colour, mark in ((0.5, "C0", "o"), (1.0, "C2", "s")):
            cells = [r for r in asy if abs(r["frac_shared_target"] - frac) < 0.02]
            if not cells:
                continue
            x = [r["poly_ratio"] for r in cells]
            a.errorbar(x, [r["frac_shared_hat"] for r in cells],
                       [r["frac_shared_sd"] for r in cells],
                       fmt=mark + "-", ms=4, capsize=2, color=colour,
                       label=f"frac_shared (true {frac:g})")
            a.plot(x, [r["rg_hat"] for r in cells], mark + "--", ms=4,
                   color=colour, alpha=.6, label=f"rg (true {frac:g} overlap)")
            a.plot(x, [r["rg_realized"] for r in cells], "kx", ms=5, alpha=.5)
        a.set_xscale("log")
        a.set_xlabel("polygenicity ratio pi1 / pi2")
        a.set_ylabel("estimate")
        a.set_title("asymmetry sweep")
        a.legend(fontsize=6)
    if pw:
        a = next(panels)
        x = [r["N"] * H2 / M for r in pw]
        a.axhline(0.5, ls="--", c="C0", lw=1, alpha=.6)
        a.axhline(1.0, ls=":", c="C2", lw=1, alpha=.6)
        a.errorbar(x, [r["frac_shared_hat"] for r in pw],
                   [r["frac_shared_sd"] for r in pw], fmt="o-", ms=4, capsize=2,
                   color="C0", label="frac_shared (true .5)")
        a.plot(x, [r["rel_poly"] for r in pw], "^-", ms=4, color="C2",
               label="rel. polygenicity")
        a.set_xscale("log")
        a.set_xlabel("N·h²/M")
        a.set_title("power sweep")
        a.legend(fontsize=8)
    if lm:
        a = next(panels)
        x = [r["N"] * H2 / M for r in lm]
        a.axhline(1.0, ls=":", c="k", lw=1, alpha=.6)
        a.errorbar(x, [r["ref_relpoly"] for r in lm],
                   [r["ref_relpoly_sd"] for r in lm], fmt="s-", ms=4, capsize=2,
                   color="C3", label="ref-panel LD")
        a.errorbar(x, [r["insample_relpoly"] for r in lm],
                   [r["insample_relpoly_sd"] for r in lm], fmt="o-", ms=4, capsize=2,
                   color="C2", label="in-sample LD")
        a.set_xscale("log")
        a.set_xlabel("N·h²/M")
        a.set_ylabel("polygenicity: est / true")
        a.set_title("LD-match control")
        a.legend(fontsize=8)
    if uc:
        a = next(panels)
        x = [r["N"] * H2 / M for r in uc]
        a.axhline(1.0, ls=":", c="k", lw=1, alpha=.6)
        a.errorbar(x, [r["joint_relpoly"] for r in uc],
                   [r["joint_relpoly_sd"] for r in uc], fmt="s-", ms=4, capsize=2,
                   color="C3", label="joint per-trait")
        a.errorbar(x, [r["calib_relpoly"] for r in uc],
                   [r["calib_relpoly_sd"] for r in uc], fmt="o-", ms=4, capsize=2,
                   color="C0", label="calibrated per-trait")
        a.plot(x, [r["joint_relshared"] for r in uc], "s--", ms=4, color="C3",
               alpha=.6, label="joint shared")
        a.plot(x, [r["calib_relshared"] for r in uc], "o--", ms=4, color="C0",
               alpha=.6, label="calibrated shared")
        a.set_xscale("log")
        a.set_xlabel("N·h²/M")
        a.set_ylabel("count: est / true")
        a.set_title("univariate-anchored calibration")
        a.legend(fontsize=7)
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle(f"MiXeR-style overlap recovery — bivariate LDpred3 "
                 f"(realistic LD, m={M}, {REPS} reps)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "mixer_overlap.png"), dpi=130)


def _warmup():
    ref, _ = R.ref_panel(0)
    b1, b2, _ = _sim_overlap(np.random.default_rng(0), NCAUSAL, 0.5, 0.8)
    _fit(ref, b1, b2, R.N1, R.N2, 0)


def main():
    which = os.environ.get(
        "SWEEP",
        "overlap,rho,polygenicity,asymmetry,power,ldmatch,calibration,"
        "unical").split(",")
    base = os.environ.get("OUT", "mixer_overlap")
    csv_path = os.path.join(HERE, base + ".csv")
    print(f"MiXeR-style overlap recovery — realistic LD (m={M}, {R.NB} blocks, "
          f"Nref={R.NREF}, {REPS} reps, p=0.10/trait)", flush=True)
    _warmup()
    t0 = time.time()
    rows = []
    for name in which:
        SWEEPS[name](rows)
        _write_csv(csv_path, rows)
    if not os.environ.get("OUT"):
        make_figure(rows)
    print(f"\nwrote {csv_path}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
