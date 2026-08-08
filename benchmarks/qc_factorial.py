"""Which summary-statistic QC steps actually matter: a 2x2x2 across four trait pairs.

Every other benchmark here except ``real_ldl_cad.py`` simulates from the model
the sampler assumes. This one exists because that class of benchmark cannot
answer a question that turned out to decide whether bipred works on real data
at all: which QC steps are load-bearing, and which are folklore.

Single-pair results are suggestive; a factorial across pairs spanning the sign
range is what settles a design. The findings under test, each established on
LDL x CAD alone and each needing to generalise before it goes in the docs:

* strictness is worthless -- 0.8x/+0.03 removed a further 142,282 variants and
  moved LDL's cancellation from 264.8 to 276.1;
* the long-range exclusion is optional once the screen runs -- keeping it gave
  rg +0.2831 against +0.2615, both converged, and retains APOE;
* the screen is irreplaceable -- no per-variant configuration converged without
  it.

Each trait is verified and calibrated before any filtering, because the two
findings that mattered most today were not filters at all: CAD's effective
sample size is overstated by 1.76x, and its SD ratio is centred at 0.755
instead of 1. Neither is fixable by dropping variants.

Screens are cached per (trait, strict, long-range) and reused across pairs, so
a trait is screened four times rather than once per pair.

Inputs are the same ~9 GB of public downloads ``real_ldl_cad.py`` documents,
plus GLGC HDL/TG and GIANT height/BMI; see ``ARMS`` below for the grid. Roughly
2.5 hours, so it is not part of ``run_all.sh``. Writes one CSV row per arm as
it completes, so a partial run is still usable.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/qc_factorial.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_ldl_cad import (                                       # noqa: E402
    CAD_NEFF, quadratic, read_aligned, subset_blocks,
)
from ldpred3 import standardize_betas                                # noqa: E402
from ldpred3.ld import load_ld_blocks                                # noqa: E402
from ldpred3.ldsc import ld_scores                                   # noqa: E402
from bipred import ldpred3_auto_bivariate_blocks, ldsc_rg            # noqa: E402
from bipred.bivariate import _rg_from_quadratics_array            # noqa: E402
from bipred.qc import (                                              # noqa: E402
    dentist, implied_sample_size, in_long_range_ld, sd_consistency,
)

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get(
    "BIPRED_WORK", os.path.expanduser("~/REPOS/ldpred3/benchmarks/.work"))
SS = os.path.join(WORK, "sumstats")
REF = os.environ.get(
    "BIPRED_LDREF", os.path.join(WORK, "ldref-hm3", "ldpred3_ldref_hm3.npz"))
OUT = os.path.join(HERE, "qc_factorial.csv")

GLGC = dict(rsid_col="rsid", a1_col="A1", a2_col="A2", beta_col="beta",
            se_col="se", n_col="N", freq_col="Freq.A1.1000G.EUR", gz=True)
TRAITS = {
    "LDL": (dict(path=f"{SS}/jointGwasMc_LDL.txt.gz", **GLGC), False),
    "HDL": (dict(path=f"{SS}/jointGwasMc_HDL.txt.gz", **GLGC), False),
    "TG": (dict(path=f"{SS}/jointGwasMc_TG.txt.gz", **GLGC), False),
    "CAD": (dict(path=f"{SS}/cad.add.160614.website.txt", rsid_col="markername",
                 a1_col="effect_allele", a2_col="noneffect_allele",
                 beta_col="beta", se_col="se_dgc", n_const=CAD_NEFF,
                 freq_col="effect_allele_freq", extra_col="median_info"), True),
    "height": (dict(path=f"{SS}/GIANT_HEIGHT_2014.txt.gz",
                    rsid_col="MarkerName", a1_col="Allele1", a2_col="Allele2",
                    beta_col="b", se_col="SE", n_col="N",
                    freq_col="Freq.Allele1.HapMapCEU", gz=True), False),
}
#: Three pairs spanning the sign range. BMI x CAD is left out as the most
#: redundant -- a second positive cross-consortium case, where LDL x CAD covers
#: that regime and has the advantage of being the pair known to diverge.
PAIRS = [("LDL", "CAD", "+0.20..0.40 positive, disjoint consortia"),
         ("height", "LDL", "~0.00 null test"),
         ("HDL", "TG", "-0.50..-0.60 negative, complete overlap")]

#: The full 2x2x2 -- strict per-variant QC, long-range exclusion, LD-consistency
#: screen -- on each pair, so main effects *and* their interactions are
#: identified rather than inferred from a fractional design. 24 arms over three
#: pairs.
#:
#: Two of the three factors were settled on LDL x CAD alone and both came out
#: null: tightening the SD check to 0.8x/+0.03 removed a further 142,282
#: variants and moved the cancellation ratio the wrong way (264.8 -> 276.1),
#: and excluding the 24 long-range regions plus APOE moved rg by 0.02 with both
#: arms converged. Running them again here tests whether those nulls hold on a
#: negative correlation and on a true zero, which is where a QC step that
#: quietly costs signal would show up.
ARMS = [(strict, drop_lr, screen)                # (strict, long-range, screen)
        for strict in (False, True)
        for drop_lr in (False, True)
        for screen in (False, True)]

LENIENT = dict(lower=0.5, upper=0.1)
STRICT = dict(lower=0.8, upper=0.03)
AF_LENIENT, AF_STRICT = 0.20, 0.10
DENTIST_ROUNDS = 4


def main():
    blocks, ids, meta = load_ld_blocks(REF, return_metadata=True)
    index = {str(r): i for i, r in enumerate(ids)}
    a1 = np.asarray(meta["counted_allele"]).astype(str)
    a0 = np.asarray(meta["other_allele"]).astype(str)
    ref_af = np.asarray(meta["reference_af"], dtype=float)
    chrom = np.asarray(meta["chrom"]).astype(str)
    pos = np.asarray(meta["pos"], dtype=np.int64)
    n_ref = meta["n_ref"]
    long_range = in_long_range_ld(chrom, pos)

    # ---- stage 0-2: parse, verify, calibrate ------------------------------
    data = {}
    print("=== verify and calibrate (before any filtering) ===", flush=True)
    print(f"{'trait':>7} {'m':>9} {'af corr':>8} {'SD offset':>10} "
          f"{'reported N':>11} {'implied N':>11} {'ratio':>7}")
    for name, (spec, binary) in TRAITS.items():
        spec = dict(spec)
        rows = read_aligned(spec.pop("path"), index, a1, a0, label="", **spec)
        idx = np.array(sorted(rows), dtype=np.int64)
        beta = np.array([rows[g][0] for g in idx])
        se = np.array([rows[g][1] for g in idx])
        n = np.array([rows[g][2] for g in idx])
        freq = np.array([rows[g][3] for g in idx])
        info = np.array([rows[g][4] for g in idx])
        af = ref_af[idx]
        ok = np.isfinite(freq) & np.isfinite(af)
        af_corr = float(np.corrcoef(freq[ok], af[ok])[0, 1])
        _, offset = sd_consistency(beta, se, n, af, binary=binary)
        sized = implied_sample_size(beta, se, af, binary=binary, reported_n=n)
        print(f"{name:>7} {idx.size:>9,} {af_corr:>+8.4f} {offset:>10.3f} "
              f"{np.median(n):>11,.0f} {sized['median']:>11,.0f} "
              f"{sized['ratio']:>7.3f}"
              f"{'' if sized['consistent'] else '   <- MISSPECIFIED'}",
              flush=True)
        # Calibrate: fit with the sample size the data behaves as if it had.
        n_fit = n * sized["ratio"] if not sized["consistent"] else n
        data[name] = dict(idx=idx, beta=beta, se=se, n=n, n_fit=n_fit,
                          freq=freq, info=info, af=af, binary=binary,
                          lr=long_range[idx])

    # ---- stage 3-4: per-variant masks, per (strict, long-range) ----------
    def per_variant(name, strict, drop_lr):
        d = data[name]
        af_max = AF_STRICT if strict else AF_LENIENT
        sd_kw = STRICT if strict else LENIENT
        keep, _ = sd_consistency(d["beta"], d["se"], d["n_fit"], d["af"],
                                 binary=d["binary"], **sd_kw)
        mask = ((np.minimum(d["af"], 1 - d["af"]) >= 0.01)
                & (np.abs(d["freq"] - d["af"]) <= af_max)
                & ~(d["info"] < 0.9)
                & ((d["beta"] / d["se"]) ** 2 <= 80)
                & (d["n"] >= 0.67 * np.median(d["n"]))
                & keep)
        if drop_lr:
            mask &= ~d["lr"]
        return mask

    screens = {}

    def screened(name):
        """Per-trait LD-consistency mask, computed once on the maximal set.

        Screening once rather than per arm is both cheaper and cleaner. A
        variant's z-score is predicted from its neighbours, so the prediction
        improves with the number of neighbours available and the largest set
        gives each variant its fairest test. More importantly, re-screening per
        arm would let the screen see a different variant set in each one, which
        entangles "what the per-variant filters did" with "what the screen was
        able to look at". Running it once on the lenient, long-range-inclusive
        set makes the screen a fixed input and leaves the filters as the only
        thing varying across arms.
        """
        if name in screens:
            return screens[name]
        d = data[name]
        mask = per_variant(name, strict=False, drop_lr=False)
        tiled, kept = subset_blocks(blocks, set(d["idx"][mask].tolist()))
        order = {g: i for i, g in enumerate(d["idx"][mask])}
        sel = np.array([order[g] for g in kept])
        started = time.perf_counter()
        keep = dentist(tiled, d["beta"][mask][sel] / d["se"][mask][sel],
                       rounds=DENTIST_ROUNDS)
        screens[name] = set(kept[keep].tolist())
        print(f"    screen {name}: {kept.size:,} -> {keep.sum():,} "
              f"({100*(1-keep.mean()):.1f}% dropped, "
              f"{time.perf_counter()-started:.0f}s)", flush=True)
        return screens[name]

    # LDSC's h2 is recorded alongside its rg: running the screen and not
    # keeping both halves of the comparison makes the heritability question
    # unanswerable after the fact, which is how the first version shipped.
    fields = ["pair", "expected", "strict", "drop_long_range", "dentist", "m",
              "ldsc_rg", "ldsc_rg_se", "ldsc_h2_1", "ldsc_h2_2", "cross_corr",
              "rg", "rg_iterate_sd", "p", "h2_1", "h2_1_iterate_sd", "h2_2", "h2_2_iterate_sd",
              "frac_shared", "frac_shared_iterate_sd", "rho_beta",
              "rg_from_overlap", "n_causal_1", "n_causal_2", "n_shared",
              "cancel_1", "cancel_2", "max_abs_beta", "drift", "warned"]
    with open(OUT, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    print(f"\n=== {len(PAIRS) * len(ARMS)} arms ===", flush=True)
    for (t1, t2, expected) in PAIRS:
        for strict, drop_lr, use_dentist in ARMS:
            d1, d2 = data[t1], data[t2]
            keep1 = set(d1["idx"][per_variant(t1, strict, drop_lr)].tolist())
            keep2 = set(d2["idx"][per_variant(t2, strict, drop_lr)].tolist())
            if use_dentist:
                keep1 = keep1 & screened(t1)
                keep2 = keep2 & screened(t2)
            shared = keep1 & keep2
            if len(shared) < 10_000:
                print(f"  {t1} x {t2} strict={strict} lr={drop_lr} "
                      f"dentist={use_dentist}: only {len(shared):,} variants, "
                      "skipped", flush=True)
                continue
            tiled, kept = subset_blocks(blocks, shared)
            p1 = {g: i for i, g in enumerate(d1["idx"])}
            p2 = {g: i for i, g in enumerate(d2["idx"])}
            s1 = np.array([p1[g] for g in kept])
            s2 = np.array([p2[g] for g in kept])
            m = kept.size
            bh1 = standardize_betas(d1["beta"][s1], d1["se"][s1], d1["n_fit"][s1])[0]
            bh2 = standardize_betas(d2["beta"][s2], d2["se"][s2], d2["n_fit"][s2])[0]
            n1, n2 = d1["n_fit"][s1], d2["n_fit"][s2]
            screen = ldsc_rg(bh1, bh2, ld_scores(tiled, n_ref=n_ref), n1, n2,
                             m_snps=m)
            cc = float(np.clip(screen.gcov_intercept, -0.95, 0.95))
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = ldpred3_auto_bivariate_blocks(
                    tiled, bh1, bh2, n1, n2, burn_in=200, num_iter=300,
                    seed=0, cross_corr=cc)
            warned = any("diverged" in str(w.message) for w in caught)
            c1 = float(np.sum(res.beta1_est ** 2)) / max(
                quadratic(tiled, res.beta1_est), 1e-12)
            c2 = float(np.sum(res.beta2_est ** 2)) / max(
                quadratic(tiled, res.beta2_est), 1e-12)
            trace = np.asarray(res.genetic_samples)
            q = len(trace) // 4
            # The MiXeR-style overlap readouts are a headline output and were
            # missing from the first version of this benchmark, so no real fit
            # had ever been inspected for them.
            mx = res.mixer
            # The sampler retains a trace, so a spread comes for free and
            # quoting a bare point estimate beside LDSC's jackknife standard
            # error would make the two look more comparable than they are.
            #
            # The columns are named "iterate_sd" rather than "sd" deliberately.
            # This is *approximate* MCMC: sigma is updated by a damped moment
            # step, not drawn from its conditional (algorithm.md step B6), so
            # the chain does not target the posterior and these are not
            # posterior standard deviations. They are the empirical spread of
            # autocorrelated retained iterates from one chain, which understates
            # uncertainty on both counts, and they carry no information about
            # error in the LD reference itself. For a defensible interval use
            # ldpred3_auto_bivariate_chains and its split-Rhat.
            rg_trace = _rg_from_quadratics_array(trace[:, 1], trace[:, 0],
                                                 trace[:, 2])
            rg_iterate_sd = float(np.nanstd(rg_trace))
            pi = np.asarray(res.pi_samples)
            union = pi[:, 1] + pi[:, 2] + pi[:, 3]
            with np.errstate(divide="ignore", invalid="ignore"):
                shared_trace = np.where(union > 0, pi[:, 3] / union, np.nan)
            row = dict(
                pair=f"{t1} x {t2}", expected=expected, strict=int(strict),
                drop_long_range=int(drop_lr), dentist=int(use_dentist), m=m,
                ldsc_rg=round(float(screen.rg), 4),
                ldsc_rg_se=round(float(screen.rg_se), 4),
                ldsc_h2_1=round(float(screen.h2[0]), 4),
                ldsc_h2_2=round(float(screen.h2[1]), 4),
                cross_corr=round(cc, 4), rg=round(float(res.rg), 4),
                rg_iterate_sd=round(rg_iterate_sd, 4),
                p=round(float(res.p), 6), h2_1=round(float(res.h2[0]), 4),
                h2_1_iterate_sd=round(float(np.std(trace[:, 0])), 4),
                h2_2=round(float(res.h2[1]), 4),
                h2_2_iterate_sd=round(float(np.std(trace[:, 2])), 4),
                frac_shared_iterate_sd=round(float(np.nanstd(shared_trace)), 4),
                frac_shared=round(float(mx["frac_shared"]), 4),
                rho_beta=round(float(mx["rho_beta"]), 4),
                rg_from_overlap=round(float(mx["rg_from_overlap"]), 4),
                n_causal_1=round(float(mx["n_causal"][0])),
                n_causal_2=round(float(mx["n_causal"][1])),
                n_shared=round(float(mx["n_shared"])),
                cancel_1=round(c1, 2),
                cancel_2=round(c2, 2),
                max_abs_beta=round(max(float(np.abs(res.beta1_est).max()),
                                      float(np.abs(res.beta2_est).max())), 4),
                # Both traits, worst case: one trait can diverge while the
                # other is estimated correctly in the same fit, and reporting
                # trait 1 alone made height x LDL look healthy when LDL's
                # cancellation ratio was 150.
                drift=round(max(
                    float(trace[-q:, 0].mean() / max(trace[:q, 0].mean(), 1e-12)),
                    float(trace[-q:, 2].mean() / max(trace[:q, 2].mean(), 1e-12))), 3),
                warned=int(warned))
            with open(OUT, "a", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fields).writerow(row)
            print(f"  {t1:>6} x {t2:<6} strict={int(strict)} lr={int(drop_lr)} "
                  f"dentist={int(use_dentist)} | m {m:>8,} | LDSC {screen.rg:>+7.4f} "
                  f"| rg {res.rg:>+7.4f} | cancel {max(c1, c2):>7.1f} | "
                  f"{'WARN' if warned else 'ok':>4} | {time.perf_counter()-started:.0f}s",
                  flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
