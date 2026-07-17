"""MiXeR-style polygenic-overlap recovery — realistic LD.

The bivariate LDpred3 sampler fits a four-state causal mixture
``(pi00, pi10, pi01, pi11)`` = neither / trait-1-only / trait-2-only / both
causal. That is exactly the bivariate causal-mixture model MiXeR (Frei et al.
2019) uses, so ``BivariateResult.mixer`` reports the same quantities:
per-trait polygenicity, the shared-causal fraction (polygenic overlap), the
within-shared effect correlation ``rho_beta`` and the overlap decomposition of
``rg`` (``rho_beta * pi11 / sqrt(pi1 pi2)``).

This benchmark stress-tests those readouts against a **known** ground truth on
realistic non-repeating coalescent LD (reusing ``rg_architectures``' cached
segments and finite reference panels). Three sweeps, each with fresh-phenotype
replicates on fixed genotypes:

  * ``overlap``  -- vary the shared-causal fraction 0..1 at fixed per-trait
    polygenicity; the headline MiXeR quantity. Checks frac_shared + rg tracking.
  * ``rho``      -- vary the within-shared effect correlation; checks rho_beta
    and that rg = rho_beta * overlap.
  * ``power``    -- vary N (hence N*h2/M) at fixed architecture; shows how much
    signal the overlap estimate needs to be meaningful.
  * ``ldmatch``  -- fit the same data on the finite reference panel vs the **exact
    in-sample (population) LD**; separates the **LD-reference-mismatch** part (the
    ref-minus-in-sample gap) from the **LD-spreading** part (which persists even
    under matched in-sample LD -- ~3x at low power). rg is immune to both.
  * ``calibration`` -- on the (realistic) finite reference panel, compares the
    naive count with the **noise-inflation fix** (``noise_inflation=True``): the
    learned per-trait lambda deflates the *mismatch*-inflated polygenicity back
    toward the truth with h2/rg unchanged, and reports whether the empirical 95%
    retained-iterate interval includes the true causal count.
  * ``unical`` -- **univariate-anchored calibration** (``res.mixer_calibrated``):
    the four-state count over-counts *more* than a univariate fit (LD-spreading is
    amplified across the four states), so this swaps the joint per-trait counts for
    two univariate ``ldpred3_auto_infer`` runs while keeping the joint's reliable
    shared *fraction*. Reports, vs known truth, the joint vs calibrated per-trait
    and shared counts: the anchor is less inflated but still over-counts at low
    power, so it *reduces*, not eliminates, the bias (rg unchanged).

The first three sweeps also record the **relative** polygenicity (pi_hat /
pi_true). With the realistic ``p_init=0.02`` default the count is **well
calibrated** across the whole per-SNP power range (``count/true`` ~1.0 up to
``N*h2/M~0.5``, ~1.1-1.2 by ``N*h2/M=2``): the old ~2-3x low-power over-count was
an artifact of the previous ``p_init=0.1`` (at low power the single-chain count
is init-influenced). The residual is a mild over-count that **grows with power**
(LD-spreading: correlated SNPs recruited around each causal), a little larger on
a finite reference panel (the ``ref - in-sample`` gap; ``ldmatch`` sweep). It is
trimmed by ``noise_inflation`` (removes the mismatch part; ``calibration`` sweep)
and by univariate anchoring via ``mixer_calibrated`` (``unical`` sweep); the
Dirichlet ``pi_prior`` is a minor lever (it matters in the no-LD limit -- see
ldpred3's ``p_prior`` / docs/inference.md). The **ratios** (rg, frac_shared) stay
reliable throughout (rg immune). See docs/rg.md and docs/algorithm.md.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/mixer_overlap.py

Env overrides: ``SWEEP`` (overlap,rho,power,ldmatch,calibration,unical or a subset), ``REPS``,
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

# The exact population LD the sumstats are generated from (the in-sample / oracle
# LD, zero reference mismatch), for the ld-match control sweep.
TRUE_BLOCKS = [(R.POP_R[b].astype(np.float32), R.IDX[b]) for b in range(R.NB)]


def _sim_overlap(rng, n_causal, frac_shared, rho_beta):
    """Two traits with exactly ``n_causal`` causal SNPs each, ``frac_shared`` of
    them shared, shared effects correlated ``rho_beta``; each scaled to h2=H2.

    Returns (b1, b2) and the exact ground-truth (pi1, pi2, pi11, rho_beta, rg)."""
    n_shared = int(round(frac_shared * n_causal))
    n_uniq = n_causal - n_shared
    need = n_shared + 2 * n_uniq
    picks = rng.choice(M, need, replace=False)
    shared = picks[:n_shared]
    u1 = picks[n_shared:n_shared + n_uniq]
    u2 = picks[n_shared + n_uniq:]
    b1 = np.zeros(M)
    b2 = np.zeros(M)
    b1[u1] = rng.standard_normal(n_uniq)
    b2[u2] = rng.standard_normal(n_uniq)
    if n_shared:
        L = np.linalg.cholesky([[1.0, rho_beta], [rho_beta, 1.0]])
        raw = L @ rng.standard_normal((2, n_shared))
        b1[shared] = raw[0]
        b2[shared] = raw[1]
    b1 *= np.sqrt(H2 / R.gv(b1, b1))
    b2 *= np.sqrt(H2 / R.gv(b2, b2))
    pi1 = pi2 = n_causal / M
    pi11 = n_shared / M
    # true rg = cov(g1,g2)/sqrt(var var) = rho_beta * n_shared / n_causal (equal h2)
    rg = rho_beta * (n_shared / n_causal)
    truth = {"pi1": pi1, "pi2": pi2, "pi11": pi11,
             "frac_shared": frac_shared, "rho_beta": rho_beta, "rg": rg}
    return b1, b2, truth


def _fit(ref, b1, b2, n1, n2, rep):
    bh1, bh2 = R.sumstats_pair(b1, b2, n1, n2, np.random.default_rng(50000 + rep))
    return ldpred3_auto_bivariate_blocks(ref, bh1, bh2, n1, n2,
                                         burn_in=BURN, num_iter=ITER, seed=rep)


def _cell(n_causal, frac_shared, rho_beta, n1, n2, base_seed):
    """Average the mixer readouts over REPS fresh phenotypes on fixed genotypes."""
    fs, rb, rg, rgo, rel1, rel2 = [], [], [], [], [], []
    truth = None
    for rep in range(REPS):
        ref, _ = R.ref_panel(rep)
        rng = np.random.default_rng(base_seed + rep)
        b1, b2, truth = _sim_overlap(rng, n_causal, frac_shared, rho_beta)
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
    return {"frac_shared_hat": m(fs), "frac_shared_sd": s(fs),
            "rho_beta_hat": m(rb), "rho_beta_sd": s(rb),
            "rg_hat": m(rg), "rg_sd": s(rg),
            "rg_overlap_hat": m(rgo),
            "rel_poly": m(rel1 + rel2), "rel_poly_sd": s(rel1 + rel2),
            **{f"true_{k}": round(v, 3) for k, v in truth.items()}}


def sweep_overlap(rows):
    print(f"\n== overlap sweep (p=0.10/trait, rho_beta=0.8, N={R.N1}/{R.N2}) ==",
          flush=True)
    print(f"{'frac_shared':>11} | {'est':>13} | {'rho_beta':>13} | "
          f"{'rg(true)':>8} {'rg_hat':>13} {'rg_ovl':>6}", flush=True)
    for i, frac in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        r = _cell(NCAUSAL, frac, 0.8, R.N1, R.N2, base_seed=1000 + 20 * i)
        r["sweep"] = "overlap"
        rows.append(r)
        print(f"{frac:>11.2f} | {r['frac_shared_hat']:>6.2f}±{r['frac_shared_sd']:<5} "
              f"| {r['rho_beta_hat']:>6.2f}±{r['rho_beta_sd']:<5} | "
              f"{r['true_rg']:>8.2f} {r['rg_hat']:>6.2f}±{r['rg_sd']:<5} "
              f"{r['rg_overlap_hat']:>6.2f}", flush=True)


def sweep_rho(rows):
    print(f"\n== rho_beta sweep (p=0.10/trait, frac_shared=0.5, N={R.N1}/{R.N2}) ==",
          flush=True)
    print(f"{'rho_beta':>8} | {'rho_beta_hat':>13} | {'frac_shared':>13} | "
          f"{'rg(true)':>8} {'rg_hat':>13}", flush=True)
    for i, rho in enumerate([0.0, 0.3, 0.6, 0.9]):
        r = _cell(NCAUSAL, 0.5, rho, R.N1, R.N2, base_seed=2000 + 20 * i)
        r["sweep"] = "rho"
        rows.append(r)
        print(f"{rho:>8.2f} | {r['rho_beta_hat']:>6.2f}±{r['rho_beta_sd']:<5} "
              f"| {r['frac_shared_hat']:>6.2f}±{r['frac_shared_sd']:<5} | "
              f"{r['true_rg']:>8.2f} {r['rg_hat']:>6.2f}±{r['rg_sd']:<5}", flush=True)


def sweep_power(rows):
    print("\n== power sweep (p=0.10/trait, frac_shared=0.5, rho_beta=0.8) ==",
          flush=True)
    print(f"{'N':>8} {'Nh2/M':>6} | {'frac_shared':>13} | {'rho_beta':>13} | "
          f"{'rg_hat':>13} | {'rel_poly':>8}", flush=True)
    for i, n in enumerate([1000, 2500, 5000, 10000, 20000]):
        r = _cell(NCAUSAL, 0.5, 0.8, n, n, base_seed=3000 + 20 * i)
        r["sweep"] = "power"
        r["N"] = n
        rows.append(r)
        print(f"{n:>8} {n*H2/M:>6.2f} | {r['frac_shared_hat']:>6.2f}±{r['frac_shared_sd']:<5} "
              f"| {r['rho_beta_hat']:>6.2f}±{r['rho_beta_sd']:<5} | "
              f"{r['rg_hat']:>6.2f}±{r['rg_sd']:<5} | {r['rel_poly']:>8.2f}", flush=True)


def sweep_ldmatch(rows):
    """Separate the LD-reference-mismatch part from the LD-spreading part: fit the
    same data on the finite reference panel vs the exact in-sample (population) LD.
    The ref-minus-in-sample gap is the mismatch part; the in-sample count itself
    (which stays inflated at low power) is the LD-spreading part."""
    print("\n== LD-match control (p=0.10/trait, frac_shared=0.5, rho_beta=0.8) ==",
          flush=True)
    print(f"{'N':>8} {'Nh2/M':>6} | {'ref pi/true':>12} {'ref rg':>7} "
          f"| {'insample pi/true':>16} {'ins rg':>7}", flush=True)
    for i, n in enumerate([1000, 2500, 5000, 10000, 20000]):
        rp, rr, tp, tr = [], [], [], []
        for rep in range(REPS):
            ref, _ = R.ref_panel(rep)
            rng = np.random.default_rng(4000 + 20 * i + rep)
            b1, b2, truth = _sim_overlap(rng, NCAUSAL, 0.5, 0.8)
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
        r = {"sweep": "ldmatch", "N": n,
             "ref_relpoly": m(rp), "ref_relpoly_sd": s(rp), "ref_rg": m(rr),
             "insample_relpoly": m(tp), "insample_relpoly_sd": s(tp),
             "insample_rg": m(tr), "true_rg": 0.4}
        rows.append(r)
        print(f"{n:>8} {n*H2/M:>6.2f} | {r['ref_relpoly']:>6.2f}±{r['ref_relpoly_sd']:<5}"
              f" {r['ref_rg']:>7.2f} | {r['insample_relpoly']:>10.2f}"
              f"±{r['insample_relpoly_sd']:<5} {r['insample_rg']:>7.2f}", flush=True)


def sweep_calibration(rows):
    """Absolute-count calibration and retained-iterate interval inclusion on the
    finite reference panel (the realistic, mismatched-LD case), across power.

    Compares the naive count (``noise_inflation=False``) with the noise-inflation
    fix (``True``): the fix learns a per-trait lambda >= 1 from the residual misfit
    and deflates the mismatch-inflated polygenicity back toward the truth, while
    ``h2`` / ``rg`` are unchanged. Also reports whether the empirical 95%
    interval from ``mixer_iterate_summary`` includes the true per-trait causal
    count. This is not Bayesian credible-interval coverage."""
    print("\n== count calibration + retained-iterate inclusion (ref-panel LD, "
          "p=0.10/trait, frac_shared=0.5, rho_beta=0.8) ==", flush=True)
    print(f"{'N':>8} {'Nh2/M':>6} | {'rel off':>7} {'rel ON':>6} {'lam':>5} | "
          f"{'hit off':>7} {'hit ON':>6} | {'rg off':>6} {'rg ON':>6}", flush=True)
    true_n1 = NCAUSAL
    for i, n in enumerate([1000, 2500, 5000, 10000, 20000]):
        ro, rn, lam, covo, covn, rgo, rgn = [], [], [], 0, 0, [], []
        for rep in range(REPS):
            ref, _ = R.ref_panel(rep)
            rng = np.random.default_rng(5000 + 20 * i + rep)
            b1, b2, truth = _sim_overlap(rng, NCAUSAL, 0.5, 0.8)
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
        # Keep the historical cov_* CSV columns so regenerated artifacts remain
        # comparable. They now mean empirical iterate-interval inclusion, not
        # posterior/credible-interval coverage.
        r = {"sweep": "calibration", "N": n, "true_rg": 0.4,
             "rel_off": m(ro), "rel_on": m(rn), "lam": m(lam),
             "cov_off": covo / REPS, "cov_on": covn / REPS,
             "rg_off": m(rgo), "rg_on": m(rgn)}
        rows.append(r)
        print(f"{n:>8} {n*H2/M:>6.2f} | {r['rel_off']:>7.2f} {r['rel_on']:>6.2f} "
              f"{r['lam']:>5.2f} | {covo}/{REPS:<5} {covn}/{REPS:<4} | "
              f"{r['rg_off']:>6.2f} {r['rg_on']:>6.2f}", flush=True)


def sweep_unical(rows):
    """Univariate-anchored count calibration (``res.mixer_calibrated``) vs truth.

    The joint four-state per-trait count over-counts *more* than a univariate fit
    (LD-spreading is amplified across the four states), so ``mixer_calibrated``
    replaces the joint per-trait polygenicities with two univariate
    ``ldpred3_auto_infer`` runs while keeping the joint's reliable shared
    *fraction*, and rebuilds ``n_shared`` on that scale. On the realistic reference
    panel, vs known truth, this reports the joint vs calibrated per-trait count and
    the joint vs calibrated shared count. The univariate anchor (= the calibrated
    per-trait count) is less inflated but still over-counts at low power, so the
    calibration *reduces*, not eliminates, the bias; ``rg`` is unchanged."""
    ureps = min(REPS, 6)
    print("\n== univariate-anchored calibration (ref-panel LD, p=0.10/trait, "
          f"frac_shared=0.5, rho_beta=0.8, {ureps} reps) ==", flush=True)
    print(f"{'N':>8} {'Nh2/M':>6} | {'joint n1/t':>10} {'calib n1/t':>10} | "
          f"{'joint sh/t':>10} {'calib sh/t':>10} | {'rg':>5}", flush=True)
    true_n1 = NCAUSAL
    true_nsh = max(int(round(0.5 * NCAUSAL)), 1)
    for i, n in enumerate([1000, 2500, 5000, 10000, 20000]):
        jp, cp, jsh, csh, rgv = [], [], [], [], []
        for rep in range(ureps):
            ref, _ = R.ref_panel(rep)
            rng = np.random.default_rng(6000 + 20 * i + rep)
            b1, b2, truth = _sim_overlap(rng, NCAUSAL, 0.5, 0.8)
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
        r = {"sweep": "unical", "N": n, "true_rg": 0.4,
             "joint_relpoly": m(jp), "joint_relpoly_sd": s(jp),
             "calib_relpoly": m(cp), "calib_relpoly_sd": s(cp),
             "joint_relshared": m(jsh), "calib_relshared": m(csh),
             "rg_hat": m(rgv)}
        rows.append(r)
        print(f"{n:>8} {n*H2/M:>6.2f} | "
              f"{r['joint_relpoly']:>5.2f}±{r['joint_relpoly_sd']:<4} "
              f"{r['calib_relpoly']:>5.2f}±{r['calib_relpoly_sd']:<4} | "
              f"{r['joint_relshared']:>10.2f} {r['calib_relshared']:>10.2f} | "
              f"{r['rg_hat']:>5.2f}", flush=True)


SWEEPS = {"overlap": sweep_overlap, "rho": sweep_rho, "power": sweep_power,
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
    pw = [r for r in rows if r["sweep"] == "power"]
    lm = [r for r in rows if r["sweep"] == "ldmatch"]
    uc = [r for r in rows if r["sweep"] == "unical"]
    npan = sum(bool(g) for g in (ov, rh, pw, lm, uc))
    if npan == 0:
        return
    fig, ax = plt.subplots(1, npan, figsize=(3.7 * npan, 3.6))
    ax = [ax] if npan == 1 else list(ax)
    panels = iter(ax)

    if ov:
        a = next(panels)
        x = [r["true_frac_shared"] for r in ov]
        a.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5)
        a.errorbar(x, [r["frac_shared_hat"] for r in ov],
                   [r["frac_shared_sd"] for r in ov], fmt="o-", ms=4, capsize=2,
                   color="C0", label="frac_shared")
        a.errorbar([r["true_rg"] for r in ov], [r["rg_hat"] for r in ov],
                   [r["rg_sd"] for r in ov], fmt="s-", ms=4, capsize=2,
                   color="C3", label="rg")
        a.set_xlabel("true (frac_shared / rg)")
        a.set_ylabel("estimated")
        a.set_title("overlap sweep")
        a.legend(fontsize=8)
    if rh:
        a = next(panels)
        x = [r["true_rho_beta"] for r in rh]
        a.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5)
        a.errorbar(x, [r["rho_beta_hat"] for r in rh],
                   [r["rho_beta_sd"] for r in rh], fmt="o-", ms=4, capsize=2,
                   color="C0", label="rho_beta")
        a.errorbar([r["true_rg"] for r in rh], [r["rg_hat"] for r in rh],
                   [r["rg_sd"] for r in rh], fmt="s-", ms=4, capsize=2,
                   color="C3", label="rg")
        a.set_xlabel("true (rho_beta / rg)")
        a.set_title("rho_beta sweep")
        a.legend(fontsize=8)
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
        "SWEEP", "overlap,rho,power,ldmatch,calibration,unical").split(",")
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
