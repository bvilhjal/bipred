"""External validation on real data: GLGC 2013 HDL x TG vs original LDSC.

The simulated-truth benchmark (``external_overlap.py``) cannot answer whether
the tools agree on real summary statistics. This arm runs the trait pair the
repository knows best — GLGC 2013 HDL x TG, the same individuals measured for
both lipids, cross-trait intercept about -0.35 — through:

* ``ldsc_orig``   -- the original LDSC (CBIIT/ldsc PyPI port): the raw GLGC
  files munged against a HapMap3 allele list derived from the same HM3 metadata
  PLINK set that built bipred's reference (the Broad's ``w_hm3.snplist`` host
  is now requester-pays), cross-trait regression against the standard 1000G EUR
  precomputed LD scores (``eur_w_ld_chr``, Zenodo record 8182036 mirror). This
  is the tool's canonical pipeline, so its variant set is its own — not
  bipred's reference panel.
* ``bipred_ldsc`` -- ``bipred.ldsc_rg`` on the harmonized, lenient-filtered,
  LD-consistency-screened variant set from bipred's HM3 reference (the
  factorial's screened arm), with the chi2 <= 80 cap on the regression rows
  only.
* ``bipred_joint`` -- the bivariate fit on that same screened set, with
  ``cross_corr`` set to the arm's own LDSC intercept (the protocol behind the
  recorded rg ~ -0.52; the uncorrected fit is reported as ``bipred_joint_cc0``).
* ``mixer_orig``  -- only when ``MIXER_REF`` points at a local 1000G.EUR.QC
  MiXeR reference bundle (GB-scale download, not fetched by this script);
  otherwise its row records the documented skip.

Comparisons are anchored against the repository's recorded values (screened
joint rg -0.52..-0.55, ``ldsc_rg`` -0.64..-0.73, frac_shared 0.94-0.95) rather
than any external ground truth — there is none for a real pair.

The GLGC files and LDSC weights are validated against
``real_data_inputs.sha256``. Unlike the pinned historical benchmarks this
script runs against the *current* ldpred3/bipred and writes its own provenance
sidecar (recording the actual tree state, dirty or not).

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/external_hdl_tg.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_ldl_cad import read_aligned, subset_blocks                     # noqa: E402
from ldpred3 import standardize_betas                                    # noqa: E402
from ldpred3.ld import load_ld_blocks                                    # noqa: E402
from ldpred3.ldsc import ld_scores                                       # noqa: E402
from bipred import ldpred3_auto_bivariate_blocks, ldsc_rg                # noqa: E402
from bipred.qc import ld_consistency_screen, sd_consistency              # noqa: E402
from benchmarks.real_data_inputs import validate_inputs                  # noqa: E402
from _external_common import (                                           # noqa: E402
    ldsc_tools, ldsc_version_label, parse_ldsc_rg_log, probe_ldsc,
    probe_mixer, run_logged, write_external_provenance, MIXER_SRC)

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get(
    "BIPRED_WORK", os.path.expanduser("~/REPOS/ldpred3/benchmarks/.work"))
SS = os.path.join(WORK, "sumstats")
WEIGHTS = os.path.join(WORK, "ldsc-weights")
REF = os.environ.get(
    "BIPRED_LDREF", os.path.join(WORK, "ldref-hm3", "ldpred3_ldref_hm3.npz"))
OUT = os.environ.get("OUT", os.path.join(HERE, "external_hdl_tg.csv"))

GLGC = dict(rsid_col="rsid", a1_col="A1", a2_col="A2", beta_col="beta",
            se_col="se", n_col="N", freq_col="Freq.A1.1000G.EUR", gz=True)
TRAITS = {"HDL": f"{SS}/jointGwasMc_HDL.txt.gz",
          "TG": f"{SS}/jointGwasMc_TG.txt.gz"}
CHI2_MAX = 80.0          # regression-row leverage cap, as in qc_factorial
SCREEN_ROUNDS = 4
FIELDS = ["arm", "tool", "ok", "m", "rg", "rg_se", "intercept",
          "intercept_se", "h2_1", "h2_2", "frac_shared", "rho_beta",
          "rg_from_overlap", "cross_corr", "time_s", "note"]


def load_screened_panel():
    """HDL/TG on bipred's HM3 reference: harmonized, lenient-filtered, screened.

    Mirrors the ``qc_factorial`` screened arm (lenient per-variant thresholds,
    LD-consistency screen on the maximal set, then the two traits intersected).
    """
    blocks, ids, meta = load_ld_blocks(REF, return_metadata=True)
    index = {str(r): i for i, r in enumerate(ids)}
    a1 = np.asarray(meta["counted_allele"]).astype(str)
    a0 = np.asarray(meta["other_allele"]).astype(str)
    ref_af = np.asarray(meta["reference_af"], dtype=float)
    n_ref = meta["n_ref"]

    data = {}
    for name, path in TRAITS.items():
        rows = read_aligned(path, index, a1, a0, label=name, **GLGC)
        idx = np.array(sorted(rows), dtype=np.int64)
        beta = np.array([rows[g][0] for g in idx])
        se = np.array([rows[g][1] for g in idx])
        n = np.array([rows[g][2] for g in idx])
        freq = np.array([rows[g][3] for g in idx])
        af = ref_af[idx]
        keep_sd, _ = sd_consistency(beta, se, n, af, binary=False)
        mask = ((np.minimum(af, 1 - af) >= 0.01)
                & (np.abs(freq - af) <= 0.20)
                & (n >= 0.67 * np.median(n))
                & keep_sd)
        # Screen once per trait on the lenient maximal set, as qc_factorial does.
        tiled, kept = subset_blocks(blocks, set(idx[mask].tolist()))
        order = {g: i for i, g in enumerate(idx[mask])}
        sel = np.array([order[g] for g in kept])
        started = time.perf_counter()
        screen = ld_consistency_screen(
            tiled, beta[mask][sel] / se[mask][sel], rounds=SCREEN_ROUNDS)
        kept_idx = set(kept[screen].tolist())
        print(f"  {name}: {idx.size:,} aligned, {mask.sum():,} after filters, "
              f"{len(kept_idx):,} after screen "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
        data[name] = dict(kept=kept_idx, beta=dict(zip(idx.tolist(), beta)),
                          se=dict(zip(idx.tolist(), se)),
                          n=dict(zip(idx.tolist(), n)))

    shared = sorted(data["HDL"]["kept"] & data["TG"]["kept"])
    tiled, kept = subset_blocks(blocks, set(shared))
    assert (np.asarray(kept) == np.asarray(shared)).all()
    out = {}
    for name in TRAITS:
        d = data[name]
        beta = np.array([d["beta"][g] for g in kept])
        se = np.array([d["se"][g] for g in kept])
        n = np.array([d["n"][g] for g in kept])
        out[name] = dict(beta=beta, se=se, n=n)
    return tiled, kept, out, n_ref


def bipred_arm():
    """bipred ldsc_rg + joint fits (with and without the overlap correction)."""
    tiled, kept, data, n_ref = load_screened_panel()
    b1, s1_, n1 = (data["HDL"][k] for k in ("beta", "se", "n"))
    b2, s2_, n2 = (data["TG"][k] for k in ("beta", "se", "n"))
    bh1 = standardize_betas(b1, s1_, n1)[0]
    bh2 = standardize_betas(b2, s2_, n2)[0]
    ell = ld_scores(tiled, n_ref=n_ref)
    z1 = bh1 * np.sqrt(n1) / np.sqrt(np.maximum(1 - bh1 ** 2, 1e-12))
    z2 = bh2 * np.sqrt(n2) / np.sqrt(np.maximum(1 - bh2 ** 2, 1e-12))
    sel = (z1 ** 2 <= CHI2_MAX) & (z2 ** 2 <= CHI2_MAX)
    m = kept.size
    screen = ldsc_rg(bh1[sel], bh2[sel], ell[sel], n1[sel], n2[sel], m_snps=m)
    cc = float(screen.gcov_intercept)
    if not -1.0 < cc < 1.0:
        raise ValueError(f"cross-trait intercept {cc} outside (-1, 1)")

    rows = []
    row = {f: "" for f in FIELDS}
    row.update(arm="bipred_ldsc", tool=f"bipred {bipred_version()}",
               ok=1, m=m, rg=screen.rg, rg_se=screen.rg_se,
               intercept=screen.gcov_intercept, h2_1=screen.h2[0],
               h2_2=screen.h2[1])
    rows.append(row)

    import warnings
    for arm, cross in (("bipred_joint", cc), ("bipred_joint_cc0", 0.0)):
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = ldpred3_auto_bivariate_blocks(
                tiled, bh1, bh2, n1, n2, burn_in=200, num_iter=300,
                seed=0, cross_corr=cross)
        mx = res.mixer
        warned = any("diverged" in str(w.message) for w in caught)
        row = {f: "" for f in FIELDS}
        row.update(arm=arm, tool=f"bipred {bipred_version()}", ok=1, m=m,
                   rg=res.rg, h2_1=res.h2[0], h2_2=res.h2[1],
                   frac_shared=mx["frac_shared"], rho_beta=mx["rho_beta"],
                   rg_from_overlap=mx["rg_from_overlap"], cross_corr=cross,
                   time_s=round(time.perf_counter() - started, 1),
                   note="divergence warning" if warned else "")
        rows.append(row)
    return rows


def ldsc_orig_arm(work):
    """Original LDSC on the raw GLGC files with the standard 1000G weights."""
    ldsc, munge = ldsc_tools()
    os.makedirs(work, exist_ok=True)
    snplist = os.path.join(WEIGHTS, "w_hm3.snplist")
    munged = []
    started = time.perf_counter()
    for name, path in TRAITS.items():
        out = os.path.join(work, name.lower())
        if not os.path.exists(out + ".sumstats.gz"):
            run_logged([munge, "--sumstats", path, "--merge-alleles", snplist,
                        "--out", out], timeout=3600,
                       log_path=out + ".log")
        munged.append(out + ".sumstats.gz")
    # LDSC's sub_chr appends the chromosome label to the prefix directly, so
    # the -chr prefix must end in a path separator (eur_w_ld_chr/1.l2...).
    ref = os.path.join(WEIGHTS, "eur_w_ld_chr") + os.sep
    out = os.path.join(work, "hdl_tg_rg")
    run_logged([ldsc, "--rg", ",".join(munged),
                "--ref-ld-chr", ref, "--w-ld-chr", ref, "--out", out],
               timeout=7200, log_path=out + ".log")
    got = parse_ldsc_rg_log(out + ".log")
    row = {f: "" for f in FIELDS}
    row.update(arm="ldsc_orig", tool=ldsc_version_label(), ok=1,
               m="", rg=got["rg"], rg_se=got["rg_se"],
               intercept=got["intercept"], intercept_se=got["intercept_se"],
               h2_1=got["h2_1"], h2_2=got["h2_2"],
               time_s=round(time.perf_counter() - started, 1))
    return [row]


def mixer_orig_arm():
    row = {f: "" for f in FIELDS}
    ref = os.environ.get("MIXER_REF")
    if not (ref and probe_mixer()):
        row.update(arm="mixer_orig", tool="none", ok=0,
                   note="skipped: MIXER_REF (1000G.EUR.QC bundle) not provided"
                        " or gsa-mixer not built; GB-scale download is not"
                        " fetched automatically")
        return [row]
    raise NotImplementedError(
        "MIXER_REF support is a stub until a 1000G.EUR.QC bundle is staged")


def bipred_version():
    import bipred
    return bipred.__version__


def main():
    t_start = time.perf_counter()
    inputs = {"ldref-hm3/ldpred3_ldref_hm3.npz": REF}
    for name, path in TRAITS.items():
        inputs[f"sumstats/{os.path.basename(path)}"] = path
    if probe_ldsc():
        weights = {"ldsc-weights/eur_w_ld_chr.tar.gz":
                   os.path.join(WEIGHTS, "eur_w_ld_chr.tar.gz"),
                   "ldsc-weights/w_hm3.snplist":
                   os.path.join(WEIGHTS, "w_hm3.snplist")}
        inputs.update({k: v for k, v in weights.items() if os.path.isfile(v)})
    input_hashes = validate_inputs(inputs)

    rows = []
    if probe_ldsc():
        try:
            rows.extend(ldsc_orig_arm(os.path.join(WORK, "ldsc-hdl-tg")))
        except (RuntimeError, ValueError) as exc:
            row = {f: "" for f in FIELDS}
            row.update(arm="ldsc_orig", tool=ldsc_version_label(), ok=0,
                       note=str(exc)[:200].replace("\n", " "))
            rows.append(row)
    else:
        row = {f: "" for f in FIELDS}
        row.update(arm="ldsc_orig", tool="none", ok=0, note="ldsc not probed")
        rows.append(row)

    rows.extend(bipred_arm())
    rows.extend(mixer_orig_arm())

    import csv
    with open(OUT + ".tmp", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(OUT + ".tmp", OUT)
    sidecar = write_external_provenance(
        OUT,
        tools={"ldsc": ldsc_version_label(),
               "mixer": os.environ.get("MIXER_REF", "not provided"),
               "gsa_mixer_src": MIXER_SRC if probe_mixer() else "not built"},
        inputs=input_hashes,
        run_controls={"traits": "GLGC 2013 HDL x TG", "screen_rounds": 4,
                      "chi2_max": CHI2_MAX,
                      "elapsed_s": round(time.perf_counter() - t_start, 1)})
    for row in rows:
        print(f"{row['arm']:>18}  rg={row['rg']}  intercept="
              f"{row['intercept']}  ok={row['ok']} {row['note']}", flush=True)
    print(f"wrote {OUT}\nprovenance: {sidecar}", flush=True)


if __name__ == "__main__":
    main()
