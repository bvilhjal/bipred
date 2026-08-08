"""Real-data end-to-end benchmark: LDL x CAD on a UK Biobank HapMap3 reference.

Every other benchmark in this directory simulates ``beta_hat ~ N(R beta, R/N)``
from the model the sampler assumes, on well-conditioned coalescent LD. That is
the right way to measure an estimator against a known truth, and it is
structurally incapable of catching a failure that only appears when the summary
statistics disagree with the LD reference. This benchmark exists because
exactly such a failure shipped in 0.3.0: the first real GWAS bipred was ever
pointed at produced a silently diverged fit that all thirty architecture cells
had no way to detect.

The traits are chosen so that the *answer* is checkable against outside
knowledge rather than against a simulation:

* LDL  -- GLGC 2013 (Willer et al.), continuous, per-variant N.
* CAD  -- CARDIoGRAMplusC4D 2015 (Nikpay et al.), GWAS Catalog GCST003116.
  Case/control, so a trait-level effective N of 4/(1/ncase + 1/nctrl) is used.
  Its SE column is genomic-control corrected, which deflates z-scores, so CAD's
  h2 here is observed-scale at the study's case fraction *and* conservative. It
  is not a liability-scale heritability.

The two consortia are close to disjoint, so correlated sampling error is small
and the cross-trait LDSC intercept (~+0.02) confirms it rather than assuming
it. Published LDL-CAD genetic correlation is around 0.2-0.4, which is the
external anchor: a fit that lands far outside that is wrong regardless of what
its internal diagnostics say.

Inputs, none of them committed (about 9 GB together)::

    # LD reference: ldpred3's converter, from figshare 19213299 (CC BY 4.0)
    python ~/REPOS/ldpred3/benchmarks/convert_bigsnpr_ldref.py

    # summary statistics
    curl -O http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_LDL.txt.gz
    curl -O https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/\
GCST003001-GCST004000/GCST003116/cad.add.160614.website.txt

Point ``BIPRED_LDREF``, ``BIPRED_LDL`` and ``BIPRED_CAD`` at them and run::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/real_ldl_cad.py

It is not part of ``run_all.sh`` for that reason. Roughly 25 minutes, dominated
by the LD-consistency screen and three joint fits.

Each of the three cleaning stages is fitted, because the *contrast* is the
result: harmonisation alone is not enough, per-variant filters are not enough,
and the LD-consistency screen is what makes the fit converge.
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
from ldpred3.ld import load_ld_blocks                                # noqa: E402
from ldpred3.ld_repr import LowRankLD, dense_ld, lowrank_ld          # noqa: E402
from ldpred3.ldsc import ld_scores                                   # noqa: E402
from bipred import ldpred3_auto_bivariate_blocks, ldsc_rg            # noqa: E402
from bipred.qc import dentist                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.expanduser("~/REPOS/ldpred3/benchmarks/.work")
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

#: Published LDL-CAD genetic correlation, the outside anchor.
RG_LITERATURE = (0.15, 0.45)
#: sum(beta^2)/h2 above this is a diverged fit, not an estimate.
CANCELLATION_LIMIT = 10.0


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


def fit_stage(name, blocks, bh1, bh2, n1, n2, n_ref, rows):
    """LDSC screen plus joint fit for one cleaning stage."""
    m = bh1.size
    print(f"\n=== {name}: {len(blocks)} blocks, {m:,} variants ===", flush=True)
    started = time.perf_counter()
    screen = ldsc_rg(bh1, bh2, ld_scores(blocks, n_ref=n_ref), n1, n2, m_snps=m)
    print(f"  LDSC  : rg {screen.rg:+.4f} (se {screen.rg_se:.4f})  "
          f"h2 ({screen.h2[0]:.4f}, {screen.h2[1]:.4f})  "
          f"intercept {screen.gcov_intercept:+.4f}  "
          f"({time.perf_counter()-started:.0f}s)", flush=True)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = ldpred3_auto_bivariate_blocks(blocks, bh1, bh2, n1, n2,
                                            burn_in=BURN_IN, num_iter=NUM_ITER,
                                            seed=0)
    diverged = [w for w in caught if "diverged" in str(w.message)]
    print(f"  bipred: rg {res.rg:+.4f}  p {res.p:.5f}  "
          f"({time.perf_counter()-started:.0f}s)")
    row = dict(stage=name, m=m, ldsc_rg=round(float(screen.rg), 4),
               ldsc_rg_se=round(float(screen.rg_se), 4),
               ldsc_h2_ldl=round(float(screen.h2[0]), 4),
               ldsc_h2_cad=round(float(screen.h2[1]), 4),
               ldsc_intercept=round(float(screen.gcov_intercept), 4),
               rg=round(float(res.rg), 4), p=round(float(res.p), 6),
               warned=int(bool(diverged)))
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
          f"   warned: {'yes' if diverged else 'no'}")
    for w in diverged:
        print(f"    WARNING: {str(w.message)[:200]}...")
    rows.append(row)
    return res


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=os.path.join(HERE, "real_ldl_cad.csv"))
    parser.add_argument("--rounds", type=int, default=4,
                        help="LD-consistency screening passes per trait")
    args = parser.parse_args(argv)

    for path in (REF, LDL, CAD):
        if not os.path.exists(path):
            raise SystemExit(f"missing input {path}; see this module's docstring")

    started = time.perf_counter()
    blocks, ids, meta = load_ld_blocks(REF, return_metadata=True)
    index = {str(r): i for i, r in enumerate(ids)}
    a1_ref = np.asarray(meta["counted_allele"]).astype(str)
    a0_ref = np.asarray(meta["other_allele"]).astype(str)
    ref_af = np.asarray(meta["reference_af"], dtype=float)
    chrom = np.asarray(meta["chrom"]).astype(str)
    pos = np.asarray(meta["pos"], dtype=np.int64)
    n_ref = meta["n_ref"]
    print(f"reference: {len(blocks)} blocks, {len(ids):,} variants, "
          f"n_ref {n_ref:,}  ({time.perf_counter()-started:.0f}s)\n", flush=True)

    ldl = read_aligned(LDL, index, a1_ref, a0_ref, rsid_col="rsid", a1_col="A1",
                       a2_col="A2", beta_col="beta", se_col="se", n_col="N",
                       freq_col="Freq.A1.1000G.EUR", gz=True,
                       label="LDL (GLGC 2013)")
    cad = read_aligned(CAD, index, a1_ref, a0_ref, rsid_col="markername",
                       a1_col="effect_allele", a2_col="noneffect_allele",
                       beta_col="beta", se_col="se_dgc", n_const=CAD_NEFF,
                       freq_col="effect_allele_freq", extra_col="median_info",
                       label=f"CAD (Nikpay 2015, N_eff {CAD_NEFF:,.0f})")

    shared = np.array(sorted(set(ldl) & set(cad)), dtype=np.int64)

    def column(source, k):
        return np.array([source[g][k] for g in shared])

    b1, s1, n1, f1 = (column(ldl, 0), column(ldl, 1), column(ldl, 2),
                      column(ldl, 3))
    b2, s2, f2 = column(cad, 0), column(cad, 1), column(cad, 3)
    info = column(cad, 4)
    af = ref_af[shared]

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
    stages["per-variant QC"] = (
        (maf >= MAF_MIN) & ~(info < INFO_MIN)
        & ((b1 / s1) ** 2 <= CHI2_MAX) & ((b2 / s2) ** 2 <= CHI2_MAX)
        & (np.abs(f1 - af) <= AF_DIFF) & (np.abs(f2 - af) <= AF_DIFF)
        & ~((chrom[shared] == MHC_CHROM) & (pos[shared] >= MHC_START)
            & (pos[shared] <= MHC_END)))

    for name in ("harmonised", "per-variant QC"):
        mask = stages[name]
        tiled, kept = subset_blocks(blocks, set(shared[mask].tolist()))
        order = {g: i for i, g in enumerate(shared[mask])}
        sel = np.array([order[g] for g in kept])
        fit_stage(name, tiled,
                  standardize_betas(b1[mask][sel], s1[mask][sel], n1[mask][sel])[0],
                  standardize_betas(b2[mask][sel], s2[mask][sel],
                                    np.full(sel.size, CAD_NEFF))[0],
                  n1[mask][sel], np.full(sel.size, CAD_NEFF), n_ref, rows)

    # Stage 3: the LD-consistency screen, on top of stage 2.
    mask = stages["per-variant QC"]
    tiled, kept = subset_blocks(blocks, set(shared[mask].tolist()))
    order = {g: i for i, g in enumerate(shared[mask])}
    sel = np.array([order[g] for g in kept])
    print(f"\n=== LD-consistency screen (bipred.qc.dentist, {args.rounds} rounds) ===",
          flush=True)
    keep1 = dentist(tiled, b1[mask][sel] / s1[mask][sel], rounds=args.rounds,
                    verbose=True)
    keep2 = dentist(tiled, b2[mask][sel] / s2[mask][sel], rounds=args.rounds,
                    verbose=True)
    both = keep1 & keep2
    print(f"  LDL dropped {(~keep1).sum():,}  CAD dropped {(~keep2).sum():,}  "
          f"union {(~both).sum():,}  remaining {both.sum():,} "
          f"({100*both.sum()/both.size:.1f}%)", flush=True)

    tiled2, kept2 = subset_blocks(tiled, set(np.where(both)[0].tolist()))
    sel2 = kept2
    fit_stage("+ LD-consistency screen", tiled2,
              standardize_betas(b1[mask][sel][sel2], s1[mask][sel][sel2],
                                n1[mask][sel][sel2])[0],
              standardize_betas(b2[mask][sel][sel2], s2[mask][sel][sel2],
                                np.full(sel2.size, CAD_NEFF))[0],
              n1[mask][sel][sel2], np.full(sel2.size, CAD_NEFF), n_ref, rows)

    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.csv}")

    final = rows[-1]
    print("\n--- regression checks on the final stage ---")
    checks = [
        (f"rg {final['rg']:+.4f} within the published {RG_LITERATURE} range",
         RG_LITERATURE[0] <= final["rg"] <= RG_LITERATURE[1]),
        (f"LDL cancellation {final['cancellation_ldl']:.1f} < {CANCELLATION_LIMIT}",
         final["cancellation_ldl"] < CANCELLATION_LIMIT),
        (f"CAD cancellation {final['cancellation_cad']:.1f} < {CANCELLATION_LIMIT}",
         final["cancellation_cad"] < CANCELLATION_LIMIT),
        ("no divergence warning", final["warned"] == 0),
    ]
    failures = [text for text, ok in checks if not ok]
    for text, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {text}")
    if failures:
        raise SystemExit(f"{len(failures)} regression check(s) failed")
    print(f"\ntotal {time.perf_counter()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
