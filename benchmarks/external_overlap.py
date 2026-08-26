"""External validation on simulated truth: bipred vs original MiXeR vs LDSC.

The package's own benchmark suite characterizes bipred against *its own*
reimplementations. This benchmark instead runs the **original** tools on the
same simulated data, on a panel small enough that MiXeR finishes in minutes:

* one coalescent reference panel (``N_REF`` samples x ``NB`` independent
  segments x ``K`` SNPs, MAF >= 5%), written as PLINK;
* two-trait four-state-mixture truth (exactly ``n_causal`` causal SNPs per
  trait, ``frac_shared`` of them shared, shared effects correlated
  ``rho_beta``), each trait scaled to h2 = 0.25 under the panel LD;
* marginal z-scores drawn from the exact model all three tools assume,
  ``beta_hat = R beta + C u / sqrt(N)`` with the same panel LD, N = 100,000,
  independent sampling noise across traits (no sample overlap).

Because every tool sees LD from the same panel that generated the statistics,
this is an **in-sample-LD** comparison: it asks whether the estimators agree
with each other and with the truth, not how they tolerate reference mismatch
(the suite's ``rg_architectures``/``mixer_overlap`` sweeps cover that).

Two configurations are committed:

* the default 40k-SNP panel runs all four methods. Note that at m = 40,000
  with N = 100,000 the mean chi-square is ~100 — far outside the regime LDSC
  documents (it prints "<200k SNPs ... almost always bad"), and the original
  LDSC's ratio jackknife fails there (``sqrt`` of a negative delete-block
  heritability). Those rows record the failure rather than hide it;
* ``NB=400 REPS=3 SKIP_MIXER=1 OUT=external_overlap_ldsc200k.csv`` runs the
  LDSC-scale panel (200k SNPs, LDSC's documented minimum) with bipred, bipred's
  LDSC, and the original LDSC.

Methods per replicate:

* ``bipred_joint``  -- ``ldpred3_auto_bivariate_chains`` (4 dispersed chains);
  reports ``rg``, ``h2``, and the MiXeR-style readouts ``pi1/pi2/pi11``,
  ``rho_beta``, ``rg_from_overlap``.
* ``bipred_ldsc``   -- ``bipred.ldsc_rg`` with panel LD scores.
* ``ldsc_orig``     -- the original LDSC (CBIIT/ldsc PyPI port, console
  scripts in ``benchmarks/.venv-ldsc``): ``--l2`` scores from the same PLINK
  panel (``--ld-wind-snps 2000``), ``munge_sumstats``, ``--rg`` with the custom
  scores as both ``--ref-ld`` and ``--w-ld`` (the documented non-partitioned
  choice). Note its ``<200k SNP`` warning is expected at m = 40,000.
* ``mixer_orig``    -- gsa-mixer v2.2.1 built from source: ``mixer.py ld`` on
  the panel (r2min 0.05, 10 Mb window), ``fit1`` per trait, ``fit2`` for the
  pair, all on the full panel (no ``--extract`` subsetting; at 40k SNPs it is
  unnecessary and keeps the polygenicity base unambiguous), with the
  documented fast fit sequence.

When a tool is not installed the corresponding rows are written as NaN with a
``note``; the bipred arms always run. Outputs: ``external_overlap.csv`` (one
row per cell x rep x method) plus ``external_overlap.provenance.json``.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        python benchmarks/external_overlap.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ldpred3 import ld_scores                                       # noqa: E402
from ldpred3.genotype_io import VariantTable, SampleTable, write_plink  # noqa: E402
from benchmarks.simulate import (                                   # noqa: E402
    SIMULATOR_CACHE_TAG, simulate_genotypes_by_mutation_rate)
from bipred import (                                                # noqa: E402
    ldsc_rg, ldpred3_auto_bivariate_chains)
from _external_common import (                                      # noqa: E402
    ldsc_tools, ldsc_version_label, mixer_lib_path,
    parse_ldsc_rg_log, parse_mixer_fit2_json, probe_ldsc, probe_mixer,
    run_logged, write_external_provenance, MIXER_PY, MIXER_SRC)

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("EXT_WORK", os.path.join(HERE, ".ext_work"))
# A relative OUT resolves against this directory, not the caller's cwd: the
# default is HERE-based, the harness and RESULTS.md read the artifacts from
# here, and the documented `OUT=external_overlap_ldsc200k.csv` invocation is
# run from the repo root -- where it would otherwise silently drop the CSV and
# its provenance sidecar one directory up. Pass an absolute path to override.
OUT = os.environ.get("OUT", "external_overlap.csv")
if not os.path.isabs(OUT):
    OUT = os.path.join(HERE, OUT)

N_REF = int(os.environ.get("N_REF", "3000"))
NB = int(os.environ.get("NB", "80"))        # independent coalescent segments
K = int(os.environ.get("K", "500"))         # SNPs per segment; m = NB * K
M = NB * K
SEG_LEN = float(os.environ.get("SEG_LEN", "5e6"))
MUT_RATE = float(os.environ.get("MUT_RATE", "2e-8"))
MIN_MAF = 0.05                              # matches MiXeR's default snps filter
SHRINK = 0.05                               # panel LD shrinkage toward I
N_GWAS = 100_000
H2 = 0.25
REPS = int(os.environ.get("REPS", "5"))
SKIP_LDSC = bool(os.environ.get("SKIP_LDSC"))    # skip the original-LDSC arm
SKIP_MIXER = bool(os.environ.get("SKIP_MIXER"))  # skip the original-MiXeR arm
# (causal fraction per trait, fraction of the smaller trait shared, rho_beta).
# Cell 2 probes a sparser, negatively correlated regime. Counts scale with M so
# tiny smoke-test panels stay drawable.
CELLS = {
    "shared_pos": dict(p_causal=0.01, frac_shared=0.5, rho_beta=0.6),
    "sparse_neg": dict(p_causal=0.005, frac_shared=0.2, rho_beta=-0.4),
}
MIXER_FIT_ARGS = ["--fit-sequence", "diffevo-fast", "neldermead-fast",
                  "--diffevo-fast-repeats", "2", "--kmax-pdf", "10",
                  "--downsample-factor", "1000"]
_ERF = np.vectorize(math.erf)


# --------------------------------------------------------------------------- #
#  Panel (simulated once, cached under .ext_work)                              #
# --------------------------------------------------------------------------- #
def _panel_dir():
    return os.path.join(
        WORK, f"panel_n{N_REF}_nb{NB}_k{K}_{SIMULATOR_CACHE_TAG}")


def build_panel():
    """Genotypes + PLINK fileset for the shared reference panel.

    Segments are independent coalescent chunks trimmed to ``K`` common SNPs, so
    block-diagonal LD at the segment boundaries is exact by construction. The
    PLINK map fabricates one chromosome with a 10 Mb gap between segments:
    window-based tools (MiXeR ``ld``, LDSC ``--l2``) then see only within-
    segment correlations above noise, matching what bipred's blocks assume.
    """
    pdir = _panel_dir()
    plink = os.path.join(pdir, "panel.1")
    geno_npy = os.path.join(pdir, "genotypes.npy")
    if os.path.exists(geno_npy) and os.path.exists(plink + ".bed"):
        return pdir, plink
    os.makedirs(pdir, exist_ok=True)
    cols = []
    for b in range(NB):
        mut = MUT_RATE
        for _ in range(4):               # bump density until the segment fills
            Gb = simulate_genotypes_by_mutation_rate(
                N_REF, SEG_LEN, mut_rate=mut, min_maf=MIN_MAF, seed=90000 + b)
            if Gb.shape[1] >= K:
                break
            mut *= 1.6
        if Gb.shape[1] < K:
            raise RuntimeError(
                f"segment {b} produced {Gb.shape[1]} SNPs (< K={K}); "
                "raise MUT_RATE or SEG_LEN")
        cols.append(Gb[:, :K])
    G = np.concatenate(cols, axis=1)
    np.save(geno_npy, G)

    blk = np.arange(M) // K
    within = np.arange(M) % K
    variants = VariantTable(
        chrom=np.array(["1"] * M, object),
        id=np.array([f"rs{i}" for i in range(M)], object),
        cm=blk * 20.0 + within * 0.001,
        pos=blk.astype(np.int64) * 10_000_000 + (within + 1) * 15_000,
        a1=np.array(["A"] * M, object), a2=np.array(["G"] * M, object))
    samples = SampleTable(
        fid=np.array([f"R{i}" for i in range(N_REF)], object),
        iid=np.array([f"R{i}" for i in range(N_REF)], object),
        sex=np.ones(N_REF, np.int64), pheno=np.full(N_REF, np.nan))
    write_plink(plink, G, variants, samples)
    return pdir, plink


def panel_genotypes(pdir):
    return np.load(os.path.join(pdir, "genotypes.npy"))


def panel_ld(pdir):
    """Per-segment panel LD (shrunk, float64) + Cholesky + bipred blocks.

    The same shrunk ``R`` generates the summary statistics and fits bipred, so
    bipred sees oracle-matched LD; MiXeR/LDSC compute their own LD from the
    same genotypes, which at ``N_REF`` = 3000 is near-oracle.
    """
    G = panel_genotypes(pdir).astype(np.float64)
    af = G.mean(0) / 2.0
    blocks_r, blocks_c, fit_blocks = [], [], []
    for b in range(NB):
        idx = np.arange(b * K, (b + 1) * K)
        Z = G[:, idx]
        Z = (Z - Z.mean(0)) / Z.std(0)
        R = (1.0 - SHRINK) * (Z.T @ Z) / N_REF + SHRINK * np.eye(K)
        blocks_r.append(R)
        blocks_c.append(np.linalg.cholesky(R))
        fit_blocks.append((R.astype(np.float32), idx))
    return blocks_r, blocks_c, fit_blocks, af


# --------------------------------------------------------------------------- #
#  Truth and summary statistics                                                #
# --------------------------------------------------------------------------- #
def _block_slices(blocks_r):
    """Variant slices per block, driven by the blocks' own sizes."""
    out, start = [], 0
    for R in blocks_r:
        k = R.shape[0]
        out.append(slice(start, start + k))
        start += k
    return out


def gv(blocks_r, a, b):
    """LD-weighted quadratic ``a' R b`` over the generating blocks."""
    return float(sum(a[sl] @ (R @ b[sl])
                     for R, sl in zip(blocks_r, _block_slices(blocks_r))))


def realized_rg(blocks_r, b1, b2):
    v1, v2 = gv(blocks_r, b1, b1), gv(blocks_r, b2, b2)
    if v1 <= 0 or v2 <= 0:
        return float("nan")
    return gv(blocks_r, b1, b2) / math.sqrt(v1 * v2)


def sim_effects(rng, p_causal, frac_shared, rho_beta, blocks_r):
    """Exact four-state truth, each trait scaled to h2=H2 under the panel LD."""
    m = sum(R.shape[0] for R in blocks_r)
    n_causal = max(int(round(p_causal * m)), 20)
    n_shared = int(round(frac_shared * n_causal))
    n_uniq = n_causal - n_shared
    picks = rng.choice(m, n_shared + 2 * n_uniq, replace=False)
    shared, u1 = picks[:n_shared], picks[n_shared:n_shared + n_uniq]
    u2 = picks[n_shared + n_uniq:]
    b1, b2 = np.zeros(m), np.zeros(m)
    b1[u1] = rng.standard_normal(n_uniq)
    b2[u2] = rng.standard_normal(n_uniq)
    if n_shared:
        L = np.linalg.cholesky([[1.0, rho_beta], [rho_beta, 1.0]])
        raw = L @ rng.standard_normal((2, n_shared))
        b1[shared], b2[shared] = raw[0], raw[1]
    b1 *= np.sqrt(H2 / gv(blocks_r, b1, b1))
    b2 *= np.sqrt(H2 / gv(blocks_r, b2, b2))
    truth = {"pi1": n_causal / m, "pi2": n_causal / m,
             "pi11": n_shared / m, "rho_beta": rho_beta,
             "rg_target": rho_beta * n_shared / n_causal}
    return b1, b2, truth


def sumstats_pair(blocks_r, blocks_c, b1, b2, n, rng):
    """Standardized marginals from the exact shared model, independent noise."""
    m = sum(R.shape[0] for R in blocks_r)
    bh1, bh2 = np.empty(m), np.empty(m)
    for R, C, sl in zip(blocks_r, blocks_c, _block_slices(blocks_r)):
        k = R.shape[0]
        bh1[sl] = R @ b1[sl] + (C @ rng.standard_normal(k)) / np.sqrt(n)
        bh2[sl] = R @ b2[sl] + (C @ rng.standard_normal(k)) / np.sqrt(n)
    return bh1, bh2


def write_gwas_files(stem, bh, af, n):
    """Write the two GWAS file flavours the external tools expect.

    ``<stem>.ldsc.txt`` carries per-allele ``BETA``/``SE`` (SE from the panel
    frequency under a standardized phenotype): munge_sumstats' median check
    looks at the signed column, and on the BETA scale it passes robustly,
    whereas on the Z scale the LD-amplified marginals of a strong-LD simulation
    can sit arbitrarily far from 0 at small m. munge then derives Z = BETA/SE.
    ``<stem>.mixer.txt`` carries ``Z`` directly, which is what MiXeR reads.
    """
    m = bh.shape[0]
    z = np.sqrt(n) * bh / np.sqrt(np.maximum(1.0 - bh * bh, 1e-12))
    pval = np.clip(2 * (1 - 0.5 * (1 + _ERF(np.abs(z) / 2 ** 0.5))),
                   1e-300, 1.0)
    se = 1.0 / np.sqrt(n * 2.0 * af * (1.0 - af))
    beta = z * se
    with open(stem + ".ldsc.txt", "w") as fh:
        fh.write("SNP A1 A2 BETA SE N P\n")
        for i in range(m):
            fh.write(f"rs{i} A G {beta[i]:.6g} {se[i]:.6g} {n} {pval[i]:.4g}\n")
    with open(stem + ".mixer.txt", "w") as fh:
        fh.write("SNP A1 A2 N Z\n")
        for i in range(m):
            fh.write(f"rs{i} A G {n} {z[i]:.6g}\n")


# --------------------------------------------------------------------------- #
#  Method arms                                                                 #
# --------------------------------------------------------------------------- #
def est_bipred(fit_blocks, bh1, bh2, seed):
    t0 = time.perf_counter()
    fit = ldpred3_auto_bivariate_chains(
        fit_blocks, bh1, bh2, N_GWAS, N_GWAS, n_chains=4,
        chain_ncores=1, seed=seed)
    dt = time.perf_counter() - t0
    res = fit.posterior
    mx = res.mixer
    return {"pi1": mx["polygenicity"][0], "pi2": mx["polygenicity"][1],
            "pi11": float(res.pi[3]), "rho_beta": mx["rho_beta"],
            "rg": res.rg, "rg_from_overlap": mx["rg_from_overlap"],
            "h2_1": res.h2[0], "h2_2": res.h2[1],
            "rhat_max": max((v for v in fit.basic_split_rhat.rhat.values()
                             if v == v), default=float("nan")),
            "time_s": dt}


def est_bipred_ldsc(fit_blocks, bh1, bh2):
    ell = ld_scores(fit_blocks, n_ref=N_REF)
    t0 = time.perf_counter()
    r = ldsc_rg(bh1, bh2, ell, N_GWAS, N_GWAS, n_blocks=NB)
    return {"rg": r.rg, "rg_se": r.rg_se, "intercept": r.gcov_intercept,
            "h2_1": r.h2[0], "h2_2": r.h2[1], "time_s": time.perf_counter() - t0}


def est_ldsc_orig(plink, work, gwas1, gwas2, tag):
    """Original LDSC: custom LD scores from the panel, then cross-trait --rg."""
    ldsc, munge = ldsc_tools()
    os.makedirs(work, exist_ok=True)
    scores = os.path.join(work, "panel")
    t0 = time.perf_counter()
    if not os.path.exists(scores + ".l2.ldscore.gz"):
        run_logged([ldsc, "--bfile", plink, "--l2", "--ld-wind-snps", "2000",
                    "--yes-really", "--out", scores], timeout=7200,
                   log_path=scores + ".build.log")
    s1, s2 = (os.path.join(work, f"{tag}_t{t}") for t in (1, 2))
    for src, dst in ((gwas1, s1), (gwas2, s2)):
        if not os.path.exists(dst + ".sumstats.gz"):
            run_logged([munge, "--sumstats", src, "--out", dst],
                       timeout=3600, log_path=dst + ".log")
    out = os.path.join(work, f"{tag}_rg")
    run_logged([ldsc, "--rg", f"{s1}.sumstats.gz,{s2}.sumstats.gz",
                "--ref-ld", scores, "--w-ld", scores, "--out", out],
               timeout=3600, log_path=out + ".log")
    got = parse_ldsc_rg_log(out + ".log")
    got["time_s"] = time.perf_counter() - t0
    return got


def est_mixer_orig(plink, work, gwas1, gwas2, tag):
    """Original MiXeR: build the panel .ld once, then fit1 x 2 + fit2."""
    lib = mixer_lib_path()
    mixer_py = os.path.join(MIXER_SRC, "precimed", "mixer.py")
    env = {**os.environ, "BGMG_SHARED_LIBRARY": lib}
    os.makedirs(work, exist_ok=True)
    # The --bim-file/--ld-file templates substitute @ with the chromosome label.
    bim_tpl = plink.replace("panel.1", "panel.@") + ".bim"
    ld_tpl = os.path.join(work, "panel.@.ld")
    t0 = time.perf_counter()
    ld1 = os.path.join(work, "panel.1.ld")
    if not os.path.exists(ld1):
        run_logged([MIXER_PY, mixer_py, "ld", "--bfile", plink,
                    "--r2min", "0.05", "--ldscore-r2min", "0.0001",
                    "--ld-window-kb", "10000", "--out", ld1, "--lib", lib],
                   timeout=7200, env=env,
                   log_path=os.path.join(work, "ld.log"))
    common = ["--bim-file", bim_tpl, "--ld-file", ld_tpl, "--chr2use", "1",
              "--lib", lib, "--seed", "123"]
    fits = []
    for t, gwas in ((1, gwas1), (2, gwas2)):
        out1 = os.path.join(work, f"{tag}_t{t}.fit1")
        run_logged([MIXER_PY, mixer_py, "fit1", "--trait1-file", gwas,
                    "--out", out1, *common, *MIXER_FIT_ARGS],
                   timeout=7200, env=env, log_path=out1 + ".log")
        fits.append(out1 + ".json")
    out2 = os.path.join(work, f"{tag}.fit2")
    run_logged([MIXER_PY, mixer_py, "fit2", "--trait1-file", gwas1,
                "--trait2-file", gwas2,
                "--trait1-params-file", fits[0],
                "--trait2-params-file", fits[1],
                "--out", out2, *common, *MIXER_FIT_ARGS],
               timeout=14400, env=env, log_path=out2 + ".log")
    got = parse_mixer_fit2_json(out2 + ".json")
    got["time_s"] = time.perf_counter() - t0
    return got


# --------------------------------------------------------------------------- #
#  Driver                                                                      #
# --------------------------------------------------------------------------- #
FIELDS = ["cell", "rep", "method", "tool", "ok",
          "pi1", "pi2", "pi11", "rho_beta", "rho_zero", "rg",
          "rg_from_overlap", "rg_se", "intercept", "intercept_se",
          "gcov", "gcov_se", "h2_1", "h2_2", "h2_1_se", "h2_2_se",
          "mean_chi2_1", "mean_chi2_2", "rhat_max", "time_s",
          "pi1_true", "pi2_true", "pi11_true", "rho_beta_true",
          "rg_target", "rg_realized", "note"]


def _row(cell, rep, method, tool, truth, rgr, ok=True, note=""):
    row = {f: "" for f in FIELDS}
    row.update(cell=cell, rep=rep, method=method, tool=tool, ok=int(ok),
               note=note, rg_realized=round(rgr, 6))
    for key, value in truth.items():
        row[key + "_true" if not key.startswith("rg") else key] = value
    return row


def main():
    t_start = time.perf_counter()
    pdir, plink = build_panel()
    blocks_r, blocks_c, fit_blocks, af = panel_ld(pdir)
    have_ldsc, have_mixer = probe_ldsc(), probe_mixer()
    print(f"external tools: ldsc={have_ldsc} mixer={have_mixer} "
          f"(panel {pdir})", flush=True)

    rows = []
    for cell_i, (cell, spec) in enumerate(CELLS.items()):
        for rep in range(REPS):
            rng = np.random.default_rng(700 + 100 * cell_i + rep)
            b1, b2, truth = sim_effects(rng, **spec, blocks_r=blocks_r)
            rgr = realized_rg(blocks_r, b1, b2)
            bh1, bh2 = sumstats_pair(
                blocks_r, blocks_c, b1, b2, N_GWAS,
                np.random.default_rng(50000 + 100 * cell_i + rep))
            tag = f"{cell}_{rep}"
            gwas_dir = os.path.join(pdir, "gwas")
            os.makedirs(gwas_dir, exist_ok=True)
            s1 = os.path.join(gwas_dir, tag + "_t1")
            s2 = os.path.join(gwas_dir, tag + "_t2")
            write_gwas_files(s1, bh1, af, N_GWAS)
            write_gwas_files(s2, bh2, af, N_GWAS)

            est = est_bipred(fit_blocks, bh1, bh2, seed=rep)
            row = _row(cell, rep, "bipred_joint", f"bipred {bipred_version()}",
                       truth, rgr)
            row.update(est)
            rows.append(row)
            row = _row(cell, rep, "bipred_ldsc", f"bipred {bipred_version()}",
                       truth, rgr)
            row.update(est_bipred_ldsc(fit_blocks, bh1, bh2))
            rows.append(row)

            if have_ldsc and not SKIP_LDSC:
                try:
                    est = est_ldsc_orig(plink, os.path.join(pdir, "ldsc"),
                                        s1 + ".ldsc.txt", s2 + ".ldsc.txt", tag)
                    row = _row(cell, rep, "ldsc_orig", ldsc_version_label(),
                               truth, rgr)
                    row.update(est)
                    rows.append(row)
                except (RuntimeError, ValueError) as exc:
                    row = _row(cell, rep, "ldsc_orig", ldsc_version_label(),
                               truth, rgr, ok=False,
                               note=str(exc)[:200].replace("\n", " "))
                    rows.append(row)
            else:
                note = "ldsc not probed" if not have_ldsc else "SKIP_LDSC set"
                rows.append(_row(cell, rep, "ldsc_orig", "none", truth, rgr,
                                 ok=False, note=note))

            if have_mixer and not SKIP_MIXER:
                try:
                    est = est_mixer_orig(plink, os.path.join(pdir, "mixer"),
                                         s1 + ".mixer.txt", s2 + ".mixer.txt",
                                         tag)
                    row = _row(cell, rep, "mixer_orig", mixer_tool_label(),
                               truth, rgr)
                    row.update(est)
                    rows.append(row)
                except (RuntimeError, ValueError, KeyError) as exc:
                    row = _row(cell, rep, "mixer_orig", mixer_tool_label(),
                               truth, rgr, ok=False,
                               note=str(exc)[:200].replace("\n", " "))
                    rows.append(row)
            else:
                note = "mixer not probed" if not have_mixer else "SKIP_MIXER set"
                rows.append(_row(cell, rep, "mixer_orig", "none", truth, rgr,
                                 ok=False, note=note))

            _checkpoint(rows)
            done = [r for r in rows if r["cell"] == cell and r["rep"] == rep]
            brief = "; ".join(f"{r['method']}: rg={_fmt(r.get('rg'))}"
                              for r in done)
            print(f"[{cell} rep{rep}] realized rg={rgr:+.3f}  {brief}",
                  flush=True)

    _checkpoint(rows)
    sidecar = write_external_provenance(
        OUT, tools={"ldsc": ldsc_version_label(), "mixer": mixer_tool_label(),
                    "simulator": SIMULATOR_CACHE_TAG},
        run_controls={"n_ref": N_REF, "nb": NB, "k": K, "m": M,
                      "n_gwas": N_GWAS, "h2": H2, "reps": REPS,
                      "shrink": SHRINK, "min_maf": MIN_MAF,
                      "elapsed_s": round(time.perf_counter() - t_start, 1)})
    print(f"wrote {OUT}\nprovenance: {sidecar}", flush=True)


def _fmt(value):
    try:
        return f"{float(value):+.3f}"
    except (TypeError, ValueError):
        return "nan"


def bipred_version():
    import bipred
    return bipred.__version__


def mixer_tool_label():
    rev = "unprobed"
    head = os.path.join(MIXER_SRC, ".git", "HEAD")
    if os.path.isfile(head):
        rev = "gsa-mixer src"
    return f"gsa-mixer v2.2.1 source build ({rev})"


def _checkpoint(rows):
    import csv
    with open(OUT + ".tmp", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(OUT + ".tmp", OUT)


if __name__ == "__main__":
    main()
