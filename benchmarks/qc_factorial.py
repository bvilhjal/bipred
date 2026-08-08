"""QC sensitivity analysis: a 2x2x2 factorial across three trait pairs.

The self-contained benchmark suite simulates from the model the sampler assumes.
This real-data analysis exists because that class of benchmark cannot answer how
strongly the conclusions depend on three plausible QC choices.

Single-pair results are suggestive, not universal. This factorial spans one
positive, one near-null and one negative pair and asks three scoped questions:

* does tightening the per-variant SD and allele-frequency thresholds matter;
* how much does long-range-LD exclusion move each estimate after screening;
* does the LD-consistency screen separate divergence-warning status?

The 24 arms are repeated analyses of three trait pairs, not 24 independent
biological validations. They can support sensitivity claims for these files and
this LD reference; broader recommendations need independent data.

The script refuses a dirty source tree and writes ``qc_factorial.provenance.json``
beside the CSV with the clean revision, runtime versions and verified hashes.

Each trait is checked before filtering. Absolute effective sample size is not
identified from quantitative-trait beta/SE/frequency alone, so those traits
retain their reported N. CAD is binary and has an identifiable implied-N
diagnostic, which this benchmark may use to calibrate its fitting N.

The LD-consistency mask is computed once per trait on the maximal set and reused
across pairs and factorial arms.

All six input files, their canonical acquisition URLs, and the pinned LD
converter revision are recorded in ``benchmarks/README.md`` Table 4; their
expected bytes are named in ``real_data_inputs.sha256``. The run takes about 75
minutes and is therefore not part of ``run_all.sh``. It writes one CSV row per
arm as it completes, so a partial run is still usable.

Each arm estimates a free cross-trait LDSC intercept on that arm's exact variant
set and passes the raw intercept as a ``cross_corr`` sensitivity value. This
assumes the whole intercept is correlated sampling noise; confounding can also
contribute. An intercept outside the joint fit's required ``(-1, 1)`` interval
aborts the run; it is never clipped into a superficially valid correction.

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
    implied_sample_size, in_long_range_ld, ld_consistency_screen,
    sd_consistency,
)
from benchmarks.real_data_inputs import (                            # noqa: E402
    require_clean_source, require_ldpred3_source, validate_inputs,
    write_provenance_sidecar,
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
#: Three pairs spanning sign patterns suggested by earlier studies. These
#: ranges are rough context, not truth labels for this analysis.
PAIRS = [("LDL", "CAD", "rough context: positive (~+0.15 to +0.45)"),
         ("height", "LDL", "rough context: near zero"),
         ("HDL", "TG", "rough context: negative (~-0.5 to -0.6)")]

#: The full 2x2x2 -- strict per-variant QC, long-range exclusion, LD-consistency
#: screen -- on each pair, so main effects *and* their interactions are
#: identified rather than inferred from a fractional design. 24 arms over three
#: pairs.
#:
#: Arms within a pair share the same GWAS and LD reference. The factorial
#: identifies sensitivity to these switches; it does not multiply the number
#: of independent trait-pair validations.
ARMS = [(strict, drop_lr, screen)                # (strict, long-range, screen)
        for strict in (False, True)
        for drop_lr in (False, True)
        for screen in (False, True)]

LENIENT = dict(lower=0.5, upper=0.1)
STRICT = dict(lower=0.8, upper=0.03)
AF_LENIENT, AF_STRICT = 0.20, 0.10
SCREEN_ROUNDS = 4


def _sample_size_plan(n, sized, *, binary):
    """Fitting N and printable diagnostics without inventing quantitative N."""
    if not binary:
        return n, "unidentified", "--", ""
    implied_text = f"{sized['median']:,.0f}"
    ratio_text = f"{sized['ratio']:.3f}"
    adjust = np.isfinite(sized["ratio"]) and not sized["consistent"]
    n_fit = n * sized["ratio"] if adjust else n
    note = "   <- MISSPECIFIED" if adjust else ""
    return n_fit, implied_text, ratio_text, note


def _cross_corr_from_ldsc(result):
    """Return an arm's raw intercept when it lies in the fit's domain."""
    cross_corr = float(result.gcov_intercept)
    if not np.isfinite(cross_corr) or not -1.0 < cross_corr < 1.0:
        raise ValueError(
            "arm-specific LDSC intercept must be finite and in (-1, 1); "
            f"got {cross_corr!r}. Refusing to clip it into the fitting range")
    return cross_corr


def _frac_shared_trace(pi_samples):
    """Public ``frac_shared`` estimand for every retained mixture iterate."""
    pi = np.asarray(pi_samples, dtype=np.float64)
    if pi.ndim != 2 or pi.shape[1] != 4:
        raise ValueError("pi_samples must have shape (iterations, 4)")
    smaller = np.minimum(pi[:, 1] + pi[:, 3], pi[:, 2] + pi[:, 3])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(smaller > 0, pi[:, 3] / smaller, np.nan)


def main():
    source_revision = require_clean_source()
    dependency_sources = {"ldpred3": require_ldpred3_source()}
    input_hashes = validate_inputs({
        "ldref-hm3/ldpred3_ldref_hm3.npz": REF,
        "sumstats/jointGwasMc_LDL.txt.gz": TRAITS["LDL"][0]["path"],
        "sumstats/cad.add.160614.website.txt": TRAITS["CAD"][0]["path"],
        "sumstats/jointGwasMc_HDL.txt.gz": TRAITS["HDL"][0]["path"],
        "sumstats/jointGwasMc_TG.txt.gz": TRAITS["TG"][0]["path"],
        "sumstats/GIANT_HEIGHT_2014.txt.gz": TRAITS["height"][0]["path"],
    })
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
        n_fit, implied_text, ratio_text, note = _sample_size_plan(
            n, sized, binary=binary)
        print(f"{name:>7} {idx.size:>9,} {af_corr:>+8.4f} {offset:>10.3f} "
              f"{np.median(n):>11,.0f} {implied_text:>11} "
              f"{ratio_text:>7}{note}",
              flush=True)
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
        keep = ld_consistency_screen(
            tiled, d["beta"][mask][sel] / d["se"][mask][sel],
            rounds=SCREEN_ROUNDS)
        screens[name] = set(kept[keep].tolist())
        print(f"    screen {name}: {kept.size:,} -> {keep.sum():,} "
              f"({100*(1-keep.mean()):.1f}% dropped, "
              f"{time.perf_counter()-started:.0f}s)", flush=True)
        return screens[name]

    # LDSC's h2 is recorded alongside its rg: running the screen and not
    # keeping both halves of the comparison makes the heritability question
    # unanswerable after the fact, which is how the first version shipped.
    fields = ["pair", "expected", "strict", "drop_long_range", "ld_screen", "m",
              "ldsc_rg", "ldsc_rg_se", "ldsc_h2_1", "ldsc_h2_2", "cross_corr",
              "rg", "rg_iterate_sd", "p", "h2_1", "h2_1_iterate_sd", "h2_2", "h2_2_iterate_sd",
              "frac_shared", "frac_shared_iterate_sd", "rho_beta",
              "rg_from_overlap", "n_causal_1", "n_causal_2", "n_shared",
              "cancel_1", "cancel_2", "max_abs_beta", "drift",
              "divergence_warned"]
    with open(OUT, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields,
                       lineterminator="\n").writeheader()
    sidecar = write_provenance_sidecar(
        OUT, source_revision=source_revision, input_hashes=input_hashes,
        dependency_sources=dependency_sources)
    print(f"provenance: {sidecar}", flush=True)

    print(f"\n=== {len(PAIRS) * len(ARMS)} arms ===", flush=True)
    for (t1, t2, expected) in PAIRS:
        for strict, drop_lr, use_screen in ARMS:
            d1, d2 = data[t1], data[t2]
            keep1 = set(d1["idx"][per_variant(t1, strict, drop_lr)].tolist())
            keep2 = set(d2["idx"][per_variant(t2, strict, drop_lr)].tolist())
            if use_screen:
                keep1 = keep1 & screened(t1)
                keep2 = keep2 & screened(t2)
            shared = keep1 & keep2
            if len(shared) < 10_000:
                print(f"  {t1} x {t2} strict={strict} lr={drop_lr} "
                      f"screen={use_screen}: only {len(shared):,} variants, "
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
            # Refit the sensitivity value after each filtering choice. Use the
            # raw intercept or reject it: clipping would silently change it.
            cc = _cross_corr_from_ldsc(screen)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = ldpred3_auto_bivariate_blocks(
                    tiled, bh1, bh2, n1, n2, burn_in=200, num_iter=300,
                    seed=0, cross_corr=cc)
            divergence_warned = any("diverged" in str(w.message) for w in caught)
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
            shared_trace = _frac_shared_trace(res.pi_samples)
            row = dict(
                pair=f"{t1} x {t2}", expected=expected, strict=int(strict),
                drop_long_range=int(drop_lr), ld_screen=int(use_screen), m=m,
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
                # Both traits, worst case: one trait can show severe
                # cancellation while the other remains numerically stable in
                # the same fit. Reporting trait 1 alone made height x LDL look
                # healthy when LDL's cancellation ratio was 150.
                drift=round(max(
                    float(trace[-q:, 0].mean() / max(trace[:q, 0].mean(), 1e-12)),
                    float(trace[-q:, 2].mean() / max(trace[:q, 2].mean(), 1e-12))), 3),
                divergence_warned=int(divergence_warned))
            with open(OUT, "a", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fields,
                               lineterminator="\n").writerow(row)
            print(f"  {t1:>6} x {t2:<6} strict={int(strict)} lr={int(drop_lr)} "
                  f"screen={int(use_screen)} | m {m:>8,} | LDSC {screen.rg:>+7.4f} "
                  f"| rg {res.rg:>+7.4f} | cancel {max(c1, c2):>7.1f} | "
                  f"{'DIV-WARN' if divergence_warned else 'ok':>8} | "
                  f"{time.perf_counter()-started:.0f}s",
                  flush=True)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
