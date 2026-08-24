"""Synthetic demo dataset for the bipred web service.

Builds a small LD cache (AR(1)-thresholded haplotypes give realistic
distance-decaying LD without a coalescent dependency) plus two matching
GWAS files with correlated effects, so a first-time user can run the full
pipeline in seconds. This is a UI fixture, not a calibration benchmark —
``benchmarks/`` owns method-evidence generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import caches

_NUC = ("A", "C", "G", "T")
# Strand-ambiguous pairs would be dropped by harmonization; avoid them so the
# demo panel passes through whole.
_AMBIG = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}


def build_demo(out_dir: Path, *, m=12000, n_samples=1500, seed=12345,
               n_eff=100_000, h2=0.25, rg=0.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Variant map: exponential gaps, LD between neighbours decays with gap.
    # The 2 kb / 2 kb gap+decay pairing keeps mean LD scores low (~2): the
    # demo exists to exercise the UI flow, and the joint sampler's finite-panel
    # pathologies (documented in benchmarks/RESULTS.md) show up well before
    # realistic human LD strengths at this tiny reference size.
    gaps = rng.exponential(2_000.0, size=m)
    pos = np.cumsum(gaps).astype(np.int64) + 1
    chrom = np.where(np.arange(m) < int(0.6 * m), "1", "2")
    ids = np.array([f"rs{1_000_000 + i}" for i in range(m)])
    allele_pairs = []
    while len(allele_pairs) < m:
        pair = (rng.choice(_NUC), rng.choice(_NUC))
        if pair[0] != pair[1] and pair not in _AMBIG:
            allele_pairs.append(pair)
    ea = np.array([p[0] for p in allele_pairs])
    oa = np.array([p[1] for p in allele_pairs])
    af_target = rng.uniform(0.05, 0.5, size=m)

    # Haplotypes: latent AR(1) Gaussian along the sequence, thresholded at the
    # per-variant allele-frequency quantile -> dosages with HWE-like margins
    # and distance-decaying LD.
    rho = np.exp(-gaps / 2_000.0)
    rho[0] = 0.0
    lat = rng.normal(size=(2 * n_samples, m)).astype(np.float64)
    for j in range(1, m):
        lat[:, j] = rho[j] * lat[:, j - 1] + np.sqrt(1.0 - rho[j] ** 2) * lat[:, j]
    # Per-column empirical quantile at 1 - AF (np.quantile broadcasts a
    # vector q into an (m, m) array, which is not what we want).
    order = np.sort(lat, axis=0)
    k = np.clip(((1.0 - af_target) * (2 * n_samples - 1)).astype(int),
                0, 2 * n_samples - 1)
    thresh = order[k, np.arange(m)]
    hap = (lat > thresh).astype(np.int8)
    dosage = hap[0::2] + hap[1::2]
    af_obs = dosage.mean(axis=0) / 2.0

    from ldpred3 import compute_ld_blocks, save_ld_blocks
    blocks = compute_ld_blocks(np.ascontiguousarray(dosage), chrom=chrom,
                               block_size=500)
    cache = out_dir / "demo.ld.npz"
    save_ld_blocks(cache, blocks, ids, reference_af=af_obs, n_ref=n_samples,
                   counted_allele=ea, other_allele=oa, chrom=chrom, pos=pos)

    # Correlated sparse effects; marginals through the fitted LD, as in
    # examples/minimal.py. state: 1 = causal in both, 2 = trait 1 only,
    # 3 = trait 2 only, 0 = neither.
    pi11, pi1_only, pi2_only = 0.03, 0.02, 0.02
    state = rng.choice(4, size=m, p=[1.0 - pi11 - pi1_only - pi2_only,
                                     pi11, pi1_only, pi2_only])
    causal1 = (state == 1) | (state == 2)
    causal2 = (state == 1) | (state == 3)
    rho_eff = 0.8                      # effect correlation at shared variants
    s1 = np.sqrt(h2 / (m * (pi11 + pi1_only)))
    s2 = np.sqrt(h2 / (m * (pi11 + pi2_only)))
    base = rng.normal(size=m)
    beta1 = np.where(causal1, s1 * base, 0.0)
    beta2 = np.where(causal2,
                     s2 * (rho_eff * base
                           + np.sqrt(1.0 - rho_eff ** 2) * rng.normal(size=m)),
                     0.0)
    beta_hat = []
    for beta in (beta1, beta2):
        marg = np.zeros(m)
        for R, idx in blocks:
            marg[idx] = np.asarray(R) @ beta[idx]
        marg += rng.normal(scale=1.0 / np.sqrt(n_eff), size=m)
        beta_hat.append(marg)

    se = 1.0 / np.sqrt(n_eff)
    # File z must equal sqrt(n_eff) * marg so that standardize_betas recovers
    # the standardized marginal; with se = 1/sqrt(n_eff) that means BETA=marg.
    for trait, marg in enumerate(beta_hat, start=1):
        with open(out_dir / f"trait{trait}.tsv", "w") as fh:
            fh.write("SNP\tCHR\tBP\tA1\tA2\tEAF\tBETA\tSE\n")
            for i in range(m):
                fh.write(f"{ids[i]}\t{chrom[i]}\t{pos[i]}\t{ea[i]}\t{oa[i]}"
                         f"\t{af_obs[i]:.4f}\t{marg[i]:.6e}\t{se:.6e}\n")
    rg_target = float(rho_eff * pi11 / np.sqrt((pi11 + pi1_only)
                                               * (pi11 + pi2_only)))
    meta = {"n_eff1": n_eff, "n_eff2": n_eff, "m": m, "seed": seed,
            "h2_true": [h2, h2], "rg_target": rg_target}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def ensure_demo(root: Path | None = None) -> Path:
    """Build the demo dataset on first use; returns its directory."""
    out = caches.demo_cache_dir(root)
    expected = ["demo.ld.npz", "trait1.tsv", "trait2.tsv", "meta.json"]
    if not all((out / name).exists() for name in expected):
        build_demo(out)
    return out


def demo_meta(root: Path | None = None) -> dict:
    return json.loads((ensure_demo(root) / "meta.json").read_text())
