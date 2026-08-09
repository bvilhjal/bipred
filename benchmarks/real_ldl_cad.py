"""Real-data end-to-end benchmark: LDL x CAD on a UK Biobank HapMap3 reference.

The self-contained simulation benchmarks draw
``beta_hat ~ N(R beta, R/N)`` from the model the sampler assumes, on
well-conditioned coalescent LD. That is the right way to measure an estimator
against a known truth, and it is structurally incapable of catching a failure
that only appears when the summary statistics disagree with the LD reference.
This benchmark exists because exactly such a failure shipped in 0.3.0: the
first real GWAS bipred was ever pointed at produced a silently diverged fit that
all thirty architecture cells had no way to detect.

The traits are chosen so prior studies provide rough context without pretending
to supply a truth label for this analysis:

* LDL  -- GLGC 2013 (Willer et al.), continuous, per-variant N.
* CAD  -- CARDIoGRAMplusC4D 2015 (Nikpay et al.), GWAS Catalog GCST003116.
  Case/control, so a trait-level effective N of 4/(1/ncase + 1/nctrl) is used.
  Its SE column is genomic-control corrected, which deflates z-scores, so CAD's
  h2 here is observed-scale at the study's case fraction *and* conservative. It
  is not a liability-scale heritability.

The two consortia are believed to be close to disjoint. Their cross-trait LDSC
intercept (~+0.02) is consistent with small correlated sampling error, but an
intercept cannot identify cohort overlap by itself. A rough 0.2-0.4 LDL-CAD
interval used by the historical analysis is descriptive context only: studies
differ in samples, phenotype definitions, models and QC, so it is not a
regression oracle and cannot override internal numerical diagnostics.

Inputs, none of them committed (about 9 GB together), are acquired as in
``benchmarks/README.md`` Table 4. In particular, the LD converter must come
from the tested ldpred3 revision
``5d86ac9d97e42c57fa31d84ff093d3bf637dc0e6``::

    # LD reference: ldpred3's converter, from figshare 19213299 (CC BY 4.0)
    python /path/to/ldpred3/benchmarks/convert_bigsnpr_ldref.py \
        --work /path/to/benchmark-work/ldref-hm3 --test

    # summary statistics
    curl -O http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_LDL.txt.gz
    curl -O https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/\
GCST003001-GCST004000/GCST003116/cad.add.160614.website.txt

Point ``BIPRED_LDREF``, ``BIPRED_LDL`` and ``BIPRED_CAD`` at them and run::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/real_ldl_cad.py

It is not part of ``run_all.sh`` for that reason. Roughly 25 minutes, dominated
by the LD-consistency screen and three joint fits.

The script refuses a dirty bipred or ldpred3 source tree and writes
``real_ldl_cad.provenance.json`` beside the CSV with both clean revisions,
runtime versions and verified hashes. It also writes
``real_ldl_cad_timing.csv``: a long-form, six-decimal wall-time record for
input checks, harmonisation, preparation, screening, LD scores, LDSC, each
joint fit, diagnostics and output. The overall total is the only overlapping
row; the leaf steps can therefore be added without double counting.

Each of the three cleaning stages is fitted because their contrast is the
result for this data set: harmonisation alone, per-variant filters, and the
same filters followed by bipred's DENTIST-inspired LD-consistency screen. This
single pair motivates the screen; it does not establish a universal QC rule.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpred3 import standardize_betas                                # noqa: E402
from bipred.bivariate import _rg_from_quadratics_array               # noqa: E402
from ldpred3.ld import load_ld_blocks                                # noqa: E402
from ldpred3.ld_repr import LowRankLD, dense_ld, lowrank_ld          # noqa: E402
from ldpred3.ldsc import ld_scores                                   # noqa: E402
from bipred import ldpred3_auto_bivariate_blocks, ldsc_rg            # noqa: E402
from bipred.qc import implied_sample_size, ld_consistency_screen     # noqa: E402
from benchmarks._benchmark_utils import StepTimings                 # noqa: E402
from benchmarks.real_data_inputs import (                            # noqa: E402
    require_clean_source, require_ldpred3_source, validate_inputs,
    write_provenance_sidecar,
)

HERE = os.path.dirname(os.path.abspath(__file__))
#: Honours BIPRED_WORK like qc_factorial.py does. Without it, pointing this
#: script at a checkout that is not under ~/REPOS meant overriding all three
#: paths below individually.
WORK = os.environ.get(
    "BIPRED_WORK", os.path.expanduser("~/REPOS/ldpred3/benchmarks/.work"))
REF = os.environ.get("BIPRED_LDREF",
                     os.path.join(WORK, "ldref-hm3", "ldpred3_ldref_hm3.npz"))
LDL = os.environ.get("BIPRED_LDL",
                     os.path.join(WORK, "sumstats", "jointGwasMc_LDL.txt.gz"))
CAD = os.environ.get("BIPRED_CAD",
                     os.path.join(WORK, "sumstats", "cad.add.160614.website.txt"))

CAD_NCASE, CAD_NCTRL = 60_801, 123_504
CAD_NEFF = 4.0 / (1.0 / CAD_NCASE + 1.0 / CAD_NCTRL)

MAF_MIN, INFO_MIN, CHI2_MAX, AF_DIFF = 0.01, 0.9, 80.0, 0.2
MHC_CHROM, MHC_START, MHC_END = "6", 25_000_000, 34_000_000
BURN_IN, NUM_ITER = 200, 300

COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
BASES = set("ACGT")

#: Rough external LDL-CAD context, not a pass/fail interval.
RG_CONTEXT = (0.2, 0.4)
#: sum(beta^2)/h2 above this is a diverged fit, not an estimate.
CANCELLATION_LIMIT = 10.0

THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _timing_path(csv_path, timing_csv=None):
    """Return an explicit timing path or derive ``<csv stem>_timing.csv``."""
    if timing_csv is not None:
        return os.fspath(timing_csv)
    stem, _ = os.path.splitext(os.fspath(csv_path))
    return f"{stem}_timing.csv"


def read_aligned(path, index, a1_ref, a0_ref, *, rsid_col, a1_col, a2_col,
                 beta_col, se_col, n_col=None, n_const=None, freq_col=None,
                 extra_col=None, gz=False, label=""):
    """``{ref_index: (beta, se, n, freq, extra)}``, aligned to the reference.

    Positive beta is always the effect of the reference's counted allele. The
    returned frequency is put on that same allele so it can be checked against
    the reference's own -- the flip *rate* cannot detect an inverted
    convention, but that correlation can.
    """
    stats = dict(rows=0, unmatched=0, indel=0, ambiguous=0, allele_mismatch=0,
                 bad_value=0, flipped=0, kept=0)
    out = {}
    opener = gzip.open if gz else open
    started = time.perf_counter()
    with opener(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        missing = [c for c in (rsid_col, a1_col, a2_col, beta_col, se_col)
                   if c not in col]
        if missing:
            raise SystemExit(f"{label}: columns {missing} absent from {path}")
        i_rs, i_a1, i_a2 = col[rsid_col], col[a1_col], col[a2_col]
        i_b, i_se = col[beta_col], col[se_col]
        i_n = col.get(n_col) if n_col else None
        i_f = col.get(freq_col) if freq_col else None
        i_x = col.get(extra_col) if extra_col else None
        for line in handle:
            f = line.rstrip("\n").split("\t")
            stats["rows"] += 1
            j = index.get(f[i_rs])
            if j is None:
                stats["unmatched"] += 1
                continue
            A1, A2 = f[i_a1].upper(), f[i_a2].upper()
            if A1 not in BASES or A2 not in BASES:
                stats["indel"] += 1
                continue
            if COMP[A1] == A2:
                stats["ambiguous"] += 1
                continue
            r1, r0 = a1_ref[j], a0_ref[j]
            if {A1, A2} != {r1, r0}:
                stats["allele_mismatch"] += 1
                continue
            try:
                beta, se = float(f[i_b]), float(f[i_se])
                n = float(f[i_n]) if i_n is not None else n_const
            except ValueError:
                stats["bad_value"] += 1
                continue
            if not (np.isfinite(beta) and np.isfinite(se) and se > 0 and n > 0):
                stats["bad_value"] += 1
                continue

            def _optional(idx):
                if idx is None:
                    return np.nan
                try:
                    return float(f[idx])
                except (ValueError, IndexError):
                    return np.nan

            freq, extra = _optional(i_f), _optional(i_x)
            if A1 == r0:
                beta, freq = -beta, 1.0 - freq
                stats["flipped"] += 1
            out[j] = (beta, se, n, freq, extra)
            stats["kept"] += 1
    print(f"{label}: {stats['rows']:,} rows in {time.perf_counter()-started:.0f}s"
          f"  kept {stats['kept']:,}  flipped {stats['flipped']:,} "
          f"({100*stats['flipped']/max(stats['kept'],1):.1f}%)")
    print(f"   dropped: unmatched {stats['unmatched']:,}, indel {stats['indel']:,},"
          f" ambiguous {stats['ambiguous']:,}, allele-mismatch "
          f"{stats['allele_mismatch']:,}, bad-value {stats['bad_value']:,}",
          flush=True)
    return out


def subset_blocks(blocks, keep):
    """Restrict blocks to ``keep`` (global indices), re-tiled to 0..m-1.

    A principal submatrix of a low-rank factor is its selected rows, unless so
    few variants survive that the retained rank would exceed the submatrix --
    then densify and re-apply the reference's own representation policy.
    """
    new, kept_global = [], []
    for R, idx in blocks:
        loc = np.array([j for j, g in enumerate(idx) if g in keep],
                       dtype=np.int64)
        if loc.size < 2:
            continue
        if isinstance(R, LowRankLD):
            if loc.size >= R.U.shape[1]:
                sub = LowRankLD(np.ascontiguousarray(R.U[loc]), loc.size, R.scale)
            else:
                dense = np.asarray(dense_ld(R), dtype=np.float64)[np.ix_(loc, loc)]
                sub = (lowrank_ld(dense, variance=0.99, quantize=True)
                       if loc.size >= 1500
                       else np.ascontiguousarray(dense.astype(np.float32)))
        else:
            sub = np.ascontiguousarray(R[np.ix_(loc, loc)])
        new.append(sub)
        kept_global.append(idx[loc])
    kept_global = np.concatenate(kept_global)
    sizes = [b.U.shape[0] if isinstance(b, LowRankLD) else b.shape[0] for b in new]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    tiled = [(b, np.arange(offsets[i], offsets[i + 1])) for i, b in enumerate(new)]
    return tiled, kept_global


def quadratic(blocks, beta):
    """``beta' R beta`` accumulated block by block."""
    total = 0.0
    for R, idx in blocks:
        b = beta[idx]
        if isinstance(R, LowRankLD):
            U = R.U.astype(np.float64) * (R.scale or 1.0)
            Rb = U @ (U.T @ b) + np.asarray(R.residual_diag) * b
        else:
            Rb = np.asarray(R, dtype=np.float64) @ b
        total += float(b @ Rb)
    return total


def fit_stage(name, blocks, bh1, bh2, n1, n2, n_ref, rows, timings,
              ldsc_chi2_max=None):
    """LDSC estimate plus joint fit for one cleaning stage.

    ``ldsc_chi2_max`` drops high-chi-square rows from the LD Score *regression*
    only, leaving the joint fit the full variant set. ``m_snps`` still counts
    every variant, as the reference implementation does: the cap excludes rows
    from the regression, it does not change the estimand.
    """
    m = bh1.size
    print(f"\n=== {name}: {len(blocks)} blocks, {m:,} variants ===", flush=True)
    started = time.perf_counter()
    scores = ld_scores(blocks, n_ref=n_ref)
    score_seconds = timings.add(
        "fit", name, "ld_scores", time.perf_counter() - started,
        m=m, n_blocks=len(blocks))
    started = time.perf_counter()
    if ldsc_chi2_max is None:
        sel = slice(None)
    else:
        z1 = bh1 * np.sqrt(n1) / np.sqrt(np.maximum(1.0 - bh1 ** 2, 1e-12))
        z2 = bh2 * np.sqrt(n2) / np.sqrt(np.maximum(1.0 - bh2 ** 2, 1e-12))
        sel = (z1 ** 2 <= ldsc_chi2_max) & (z2 ** 2 <= ldsc_chi2_max)
        print(f"  LDSC cap: {m - int(sel.sum()):,} of {m:,} rows held out of "
              f"the regression (chi2 > {ldsc_chi2_max:g}); the fit keeps all "
              f"{m:,}", flush=True)
    screen = ldsc_rg(bh1[sel], bh2[sel], scores[sel], n1[sel], n2[sel],
                     m_snps=m)
    ldsc_seconds = timings.add(
        "fit", name, "ldsc_regression", time.perf_counter() - started,
        m=m, n_blocks=len(blocks))
    del scores
    print(f"  LDSC  : rg {screen.rg:+.4f} (se {screen.rg_se:.4f})  "
          f"h2 ({screen.h2[0]:.4f}, {screen.h2[1]:.4f})  "
          f"intercept {screen.gcov_intercept:+.4f}  "
          f"(scores {score_seconds:.3f}s, regression {ldsc_seconds:.3f}s)",
          flush=True)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, n1, n2,
                                            burn_in=BURN_IN, num_iter=NUM_ITER,
                                            seed=0)
    fit_seconds = timings.add(
        "fit", name, "bivariate_fit", time.perf_counter() - started,
        m=m, n_blocks=len(blocks))
    diverged = [w for w in caught if "diverged" in str(w.message)]
    print(f"  bipred: rg {res.rg:+.4f}  p {res.p:.5f}  "
          f"({fit_seconds:.3f}s)")
    started = time.perf_counter()
    # Spread across retained iterates, named as in qc_factorial.py and carrying
    # the same warning: this is *approximate* MCMC -- sigma moves by a damped
    # moment step rather than a draw from its conditional -- so these are not
    # posterior standard deviations. They are the empirical spread of
    # autocorrelated iterates from a single chain, which understates
    # uncertainty, and they say nothing about error in the LD reference. For a
    # defensible interval use ldpred3_auto_bivariate_chains and its split-Rhat.
    trace = np.asarray(res.genetic_samples, dtype=np.float64)
    rg_trace = _rg_from_quadratics_array(trace[:, 1], trace[:, 0], trace[:, 2])
    row = dict(stage=name, m=m, ldsc_rg=round(float(screen.rg), 4),
               ldsc_rg_se=round(float(screen.rg_se), 4),
               ldsc_h2_ldl=round(float(screen.h2[0]), 4),
               ldsc_h2_cad=round(float(screen.h2[1]), 4),
               ldsc_intercept=round(float(screen.gcov_intercept), 4),
               rg=round(float(res.rg), 4),
               rg_iterate_sd=round(float(np.nanstd(rg_trace)), 4),
               h2_ldl_iterate_sd=round(float(np.nanstd(trace[:, 0])), 5),
               h2_cad_iterate_sd=round(float(np.nanstd(trace[:, 2])), 5),
               retained_iterations=int(res.retained_iterations or 0),
               p=round(float(res.p), 6),
               divergence_warned=int(bool(diverged)))
    for label, beta, h2 in (("ldl", res.beta1_est, res.h2[0]),
                            ("cad", res.beta2_est, res.h2[1])):
        total = float(np.sum(beta ** 2))
        quad = quadratic(blocks, beta)
        cancel = total / max(quad, 1e-12)
        print(f"    {label.upper()}: h2 {h2:.4f}  sum(beta^2) {total:9.3f}  "
              f"cancellation {cancel:7.1f}  max|beta| {np.abs(beta).max():.4f}")
        row[f"h2_{label}"] = round(float(h2), 4)
        row[f"cancellation_{label}"] = round(cancel, 2)
        row[f"max_abs_beta_{label}"] = round(float(np.abs(beta).max()), 4)
    trace = np.asarray(res.genetic_samples)
    drift = float(trace[-len(trace)//4:, 0].mean()
                  / max(trace[:len(trace)//4, 0].mean(), 1e-12))
    row["trace_drift_ldl"] = round(drift, 3)
    print(f"    trace drift (last quarter / first) {drift:.2f}"
          f"   divergence warning: {'yes' if diverged else 'no'}")
    for w in diverged:
        print(f"    WARNING: {str(w.message)[:200]}...")
    rows.append(row)
    timings.add("fit", name, "diagnostics", time.perf_counter() - started,
                m=m, n_blocks=len(blocks))
    return res


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=os.path.join(HERE, "real_ldl_cad.csv"))
    parser.add_argument(
        "--timing-csv",
        help="long-form step timings (default: <csv stem>_timing.csv)")
    parser.add_argument(
        "--chi2-cap", choices=("both", "ldsc", "none"), default="both",
        help="where the chi2 <= 80 filter applies. 'both' (default) is the "
             "historical behaviour: one per-variant mask feeds LD Score "
             "regression and the joint fit alike. 'ldsc' keeps the cap on the "
             "regression, which needs it -- an uncapped large-effect variant "
             "holds near-full leverage on the slope -- while the fit sees "
             "every variant, which is what its slab component is for. 'none' "
             "removes it everywhere. On lipoprotein(a) the cap removes 73%% of "
             "the LPA locus and half the trait's summed chi-square, so this "
             "is not a minor switch for concentrated architectures.")
    parser.add_argument("--rounds", type=int, default=4,
                        help="LD-consistency screening passes per trait")
    args = parser.parse_args(argv)
    timing_csv = _timing_path(args.csv, args.timing_csv)
    timings = StepTimings(timing_csv)

    with timings.measure("preflight", "bipred", "source_check"):
        source_revision = require_clean_source()
    with timings.measure("preflight", "ldpred3", "source_check"):
        dependency_sources = {"ldpred3": require_ldpred3_source()}
    with timings.measure("preflight", "inputs", "checksum_validation"):
        input_hashes = validate_inputs({
            "ldref-hm3/ldpred3_ldref_hm3.npz": REF,
            "sumstats/jointGwasMc_LDL.txt.gz": LDL,
            "sumstats/cad.add.160614.website.txt": CAD,
        })

    started = time.perf_counter()
    blocks, ids, meta = load_ld_blocks(REF, return_metadata=True)
    index = {str(r): i for i, r in enumerate(ids)}
    a1_ref = np.asarray(meta["counted_allele"]).astype(str)
    a0_ref = np.asarray(meta["other_allele"]).astype(str)
    ref_af = np.asarray(meta["reference_af"], dtype=float)
    chrom = np.asarray(meta["chrom"]).astype(str)
    pos = np.asarray(meta["pos"], dtype=np.int64)
    n_ref = meta["n_ref"]
    load_seconds = timings.add(
        "input", "LD reference", "load_and_index",
        time.perf_counter() - started, m=len(ids), n_blocks=len(blocks))
    print(f"reference: {len(blocks)} blocks, {len(ids):,} variants, "
          f"n_ref {n_ref:,}  ({load_seconds:.3f}s)\n", flush=True)

    started = time.perf_counter()
    ldl = read_aligned(LDL, index, a1_ref, a0_ref, rsid_col="rsid", a1_col="A1",
                       a2_col="A2", beta_col="beta", se_col="se", n_col="N",
                       freq_col="Freq.A1.1000G.EUR", gz=True,
                       label="LDL (GLGC 2013)")
    timings.add("input", "LDL", "harmonize", time.perf_counter() - started,
                m=len(ldl), n_blocks=len(blocks))
    started = time.perf_counter()
    cad = read_aligned(CAD, index, a1_ref, a0_ref, rsid_col="markername",
                       a1_col="effect_allele", a2_col="noneffect_allele",
                       beta_col="beta", se_col="se_dgc", n_const=CAD_NEFF,
                       freq_col="effect_allele_freq", extra_col="median_info",
                       label=f"CAD (Nikpay 2015, N_eff {CAD_NEFF:,.0f})")
    timings.add("input", "CAD", "harmonize", time.perf_counter() - started,
                m=len(cad), n_blocks=len(blocks))

    started = time.perf_counter()
    shared = np.array(sorted(set(ldl) & set(cad)), dtype=np.int64)

    def column(source, k):
        return np.array([source[g][k] for g in shared])

    b1, s1, n1, f1 = (column(ldl, 0), column(ldl, 1), column(ldl, 2),
                      column(ldl, 3))
    b2, s2, f2 = column(cad, 0), column(cad, 1), column(cad, 3)
    info = column(cad, 4)
    af = ref_af[shared]
    timings.add("preparation", "shared", "materialize_intersection",
                time.perf_counter() - started, m=shared.size,
                n_blocks=len(blocks))

    # Calibrate the effective sample size before anything else, because it
    # scales every standardized effect and therefore every h2. CAD's published
    # N_eff comes from the pooled 4/(1/ncase + 1/nctrl) formula, which
    # overstates a meta-analysis of cohorts with differing case/control ratios,
    # and its SEs are doubly genomic-control corrected. Both deflate the
    # sample size the data behaves as if it had: 162,973 reported against
    # 92,966 implied. Fitting the reported value understated CAD's h2 by that
    # factor -- the 0.0401 this benchmark recorded before this stage existed.
    started = time.perf_counter()
    reported = np.full(shared.size, CAD_NEFF)
    sized = implied_sample_size(b2, s2, af, binary=True, reported_n=reported)
    cad_n = CAD_NEFF * (1.0 if sized["consistent"] else sized["ratio"])
    timings.add("preparation", "CAD", "sample_size_calibration",
                time.perf_counter() - started, m=shared.size,
                n_blocks=len(blocks))
    print(f"\nCAD effective N: reported {CAD_NEFF:,.0f}, implied "
          f"{sized['median']:,.0f} (ratio {sized['ratio']:.3f})"
          f"{'' if sized['consistent'] else '  <- MISSPECIFIED, fitting the implied value'}",
          flush=True)

    started = time.perf_counter()
    print("\nallele-alignment check against the reference's own af "
          "(near +1 aligned, near -1 inverted -- the flip rate cannot tell):")
    for label, freq in (("LDL", f1), ("CAD", f2)):
        ok = np.isfinite(freq) & np.isfinite(af)
        print(f"  {label}: {np.corrcoef(freq[ok], af[ok])[0, 1]:+.4f}")

    rows = []
    stages = {}

    # Stage 1: harmonisation only.
    stages["harmonised"] = np.ones(shared.size, bool)
    # Stage 2: per-variant filters.
    maf = np.minimum(af, 1 - af)
    # The cap is in this mask only under --chi2-cap both. Under 'ldsc' it moves
    # to the regression rows inside fit_stage; under 'none' it is gone.
    capped = (((b1 / s1) ** 2 <= CHI2_MAX) & ((b2 / s2) ** 2 <= CHI2_MAX)
              if args.chi2_cap == "both" else np.ones(shared.size, bool))
    ldsc_cap = CHI2_MAX if args.chi2_cap == "ldsc" else None
    stages["per-variant QC"] = (
        (maf >= MAF_MIN) & ~(info < INFO_MIN) & capped
        & (np.abs(f1 - af) <= AF_DIFF) & (np.abs(f2 - af) <= AF_DIFF)
        & ~((chrom[shared] == MHC_CHROM) & (pos[shared] >= MHC_START)
            & (pos[shared] <= MHC_END)))
    timings.add("preparation", "shared", "per_variant_qc",
                time.perf_counter() - started, m=shared.size,
                n_blocks=len(blocks))

    for name in ("harmonised", "per-variant QC"):
        mask = stages[name]
        started = time.perf_counter()
        tiled, kept = subset_blocks(blocks, set(shared[mask].tolist()))
        order = {g: i for i, g in enumerate(shared[mask])}
        sel = np.array([order[g] for g in kept])
        timings.add("fit", name, "ld_subset_retile",
                    time.perf_counter() - started, m=kept.size,
                    n_blocks=len(tiled))
        started = time.perf_counter()
        n1_stage = n1[mask][sel]
        n2_stage = np.full(sel.size, cad_n)
        bh1 = standardize_betas(
            b1[mask][sel], s1[mask][sel], n1_stage)[0]
        bh2 = standardize_betas(
            b2[mask][sel], s2[mask][sel], n2_stage)[0]
        timings.add("fit", name, "standardize_effects",
                    time.perf_counter() - started, m=kept.size,
                    n_blocks=len(tiled))
        fit_stage(name, tiled, bh1, bh2, n1_stage, n2_stage, n_ref,
                  rows, timings, ldsc_chi2_max=ldsc_cap)

    # Stage 3: the LD-consistency screen, on top of stage 2.
    mask = stages["per-variant QC"]
    started = time.perf_counter()
    tiled, kept = subset_blocks(blocks, set(shared[mask].tolist()))
    order = {g: i for i, g in enumerate(shared[mask])}
    sel = np.array([order[g] for g in kept])
    timings.add("screen", "shared", "ld_subset_retile",
                time.perf_counter() - started, m=kept.size,
                n_blocks=len(tiled))
    print(f"\n=== LD-consistency screen "
          f"(bipred.qc.ld_consistency_screen, {args.rounds} rounds) ===",
          flush=True)
    started = time.perf_counter()
    keep1 = ld_consistency_screen(
        tiled, b1[mask][sel] / s1[mask][sel], rounds=args.rounds, verbose=True)
    timings.add("screen", "LDL", "ld_consistency",
                time.perf_counter() - started, m=kept.size,
                n_blocks=len(tiled))
    started = time.perf_counter()
    keep2 = ld_consistency_screen(
        tiled, b2[mask][sel] / s2[mask][sel], rounds=args.rounds, verbose=True)
    timings.add("screen", "CAD", "ld_consistency",
                time.perf_counter() - started, m=kept.size,
                n_blocks=len(tiled))
    both = keep1 & keep2
    print(f"  LDL dropped {(~keep1).sum():,}  CAD dropped {(~keep2).sum():,}  "
          f"union {(~both).sum():,}  remaining {both.sum():,} "
          f"({100*both.sum()/both.size:.1f}%)", flush=True)

    started = time.perf_counter()
    tiled2, kept2 = subset_blocks(tiled, set(np.where(both)[0].tolist()))
    sel2 = kept2
    screen_name = "+ LD-consistency screen"
    timings.add("fit", screen_name, "ld_subset_retile",
                time.perf_counter() - started, m=kept2.size,
                n_blocks=len(tiled2))
    started = time.perf_counter()
    n1_stage = n1[mask][sel][sel2]
    n2_stage = np.full(sel2.size, cad_n)
    bh1 = standardize_betas(
        b1[mask][sel][sel2], s1[mask][sel][sel2], n1_stage)[0]
    bh2 = standardize_betas(
        b2[mask][sel][sel2], s2[mask][sel][sel2], n2_stage)[0]
    timings.add("fit", screen_name, "standardize_effects",
                time.perf_counter() - started, m=kept2.size,
                n_blocks=len(tiled2))
    fit_stage(screen_name, tiled2, bh1, bh2, n1_stage, n2_stage, n_ref,
              rows, timings, ldsc_chi2_max=ldsc_cap)

    started = time.perf_counter()
    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    timings.add("output", "results", "write_csv",
                time.perf_counter() - started, m=len(rows))

    started = time.perf_counter()
    sidecar = write_provenance_sidecar(
        args.csv, source_revision=source_revision, input_hashes=input_hashes,
        dependency_sources=dependency_sources,
        run_controls={
            "cpu_count": os.cpu_count(),
            "rounds": args.rounds,
            "thread_environment": {name: os.environ.get(name)
                                   for name in THREAD_ENV},
            "timing_csv": os.path.basename(timing_csv),
        })
    timings.add("output", "provenance", "write_json",
                time.perf_counter() - started)

    started = time.perf_counter()
    final = rows[-1]
    print("\n--- diagnostic checks on the final stage ---")
    print(f"  [context] rg {final['rg']:+.4f}; rough external range "
          f"{RG_CONTEXT} is not a pass/fail check")
    checks = [
        (f"LDL cancellation {final['cancellation_ldl']:.1f} < {CANCELLATION_LIMIT}",
         final["cancellation_ldl"] < CANCELLATION_LIMIT),
        (f"CAD cancellation {final['cancellation_cad']:.1f} < {CANCELLATION_LIMIT}",
         final["cancellation_cad"] < CANCELLATION_LIMIT),
        ("no divergence warning", final["divergence_warned"] == 0),
    ]
    failures = [text for text, ok in checks if not ok]
    for text, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {text}")
    timings.add("output", "results", "regression_checks",
                time.perf_counter() - started, m=final["m"])
    total_seconds = timings.total()
    print(f"\nwrote {args.csv}, {timing_csv}, and {sidecar}")
    print(f"total {total_seconds:.3f}s")
    if failures:
        raise SystemExit(f"{len(failures)} regression check(s) failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
