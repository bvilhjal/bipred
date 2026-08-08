"""Bivariate prediction gain across two-trait architectures with realistic LD.

The GWAS is generated from population (coalescent) LD and fitted with LD
estimated from a finite reference panel. The default run pairs a genuine
no-shrinkage control with the historical 5% regularisation arm on identical
effect and noise draws.

Build ``ld_library.npz`` with ``make_ld_library.py`` first.  The script writes
one full-precision CSV row per architecture and replicate; the printed table is
only a compact summary of that artifact. It refuses a dirty source tree and
writes a sidecar with the clean revision, package versions, run controls,
simulator cache tag, and library hash.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bipred import ldpred3_auto_bivariate_blocks                 # noqa: E402
from ldpred3 import ldpred3_by_blocks                            # noqa: E402
from benchmarks.real_data_inputs import (                        # noqa: E402
    require_clean_source, require_ldpred3_source, sha256_file,
    write_provenance_sidecar,
)


N1, N2 = 100_000, 2_000
P_CAUSAL = 0.1
H2 = 0.5
N_REF = 2_000
REPS = 6
CSV_FIELDS = (
    "architecture", "target_rg", "realized_rg", "replicate", "m", "n1",
    "n2", "n_ref", "reference_shrinkage", "p_causal", "h2", "n_causal_1",
    "n_causal_2", "n_shared_causal", "solo_r2", "joint_r2", "gain",
    "rg_est", "joint_p", "joint_h2_1", "joint_h2_2", "joint_warned",
    "joint_warning_count", "joint_implausible_warnings",
    "joint_divergence_warnings", "joint_other_warnings",
)


def _disjoint_masks(rng, m, p_causal=P_CAUSAL):
    """Two exactly disjoint Bernoulli masks with equal expected density."""
    if not 0.0 <= p_causal <= 0.5:
        raise ValueError("p_causal must be in [0, 0.5] for equal disjoint masks")
    first = rng.random(m) < p_causal
    # Conditional sampling on the complement preserves E[sum(second)] = m*p.
    second = (~first) & (rng.random(m) < p_causal / (1.0 - p_causal))
    return first, second


def _build_panels(lib_r, *, n_ref=N_REF, shrinkage=0.0, blocks=None, seed=0):
    """Population and finite-reference LD blocks, with distinct payloads."""
    lib_r = np.asarray(lib_r, dtype=np.float64)
    if lib_r.ndim != 3 or lib_r.shape[1] != lib_r.shape[2]:
        raise ValueError("the LD library must have shape (blocks, k, k)")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    nb = lib_r.shape[0] if blocks is None else min(int(blocks), lib_r.shape[0])
    if nb < 1:
        raise ValueError("at least one LD block is required")
    k = lib_r.shape[1]
    rng = np.random.default_rng(seed)
    pop, chol_pop, ref, indices = [], [], [], []
    for block_no in range(nb):
        population_ld = lib_r[block_no].copy()
        chol = np.linalg.cholesky(population_ld + 1e-4 * np.eye(k))
        genotypes = rng.standard_normal((n_ref, k)) @ chol.T
        genotypes = (genotypes - genotypes.mean(0)) / genotypes.std(0)
        reference_ld = (genotypes.T @ genotypes) / n_ref
        if shrinkage:
            reference_ld = ((1.0 - shrinkage) * reference_ld
                            + shrinkage * np.eye(k))
        idx = np.arange(block_no * k, (block_no + 1) * k)
        pop.append((population_ld.astype(np.float32), idx))
        ref.append((reference_ld.astype(np.float32), idx))
        chol_pop.append(chol)
        indices.append(idx)
    return pop, chol_pop, ref, indices


def _genetic_covariance(a, b, pop, indices):
    return sum(a[idx] @ (pop[i][0].astype(float) @ b[idx])
               for i, idx in enumerate(indices))


def _scale_effects(beta, h2, pop, indices):
    variance = _genetic_covariance(beta, beta, pop, indices)
    return beta * np.sqrt(h2 / variance) if variance > 0 else beta


def _sumstats(beta, n_eff, rng, pop, chol_pop, indices):
    marginal = np.empty(beta.size)
    for i, idx in enumerate(indices):
        marginal[idx] = (pop[i][0].astype(float) @ beta[idx]
                         + chol_pop[i] @ rng.standard_normal(idx.size)
                         / np.sqrt(n_eff))
    return marginal


def _prediction_r2(estimate, truth, pop, indices):
    numerator = _genetic_covariance(estimate, truth, pop, indices)
    denominator = (_genetic_covariance(estimate, estimate, pop, indices)
                   * _genetic_covariance(truth, truth, pop, indices))
    return float(numerator * numerator / denominator) if denominator > 0 else 0.0


def _realized_rg(beta1, beta2, pop, indices):
    covariance = _genetic_covariance(beta1, beta2, pop, indices)
    denominator = np.sqrt(
        _genetic_covariance(beta1, beta1, pop, indices)
        * _genetic_covariance(beta2, beta2, pop, indices))
    return float(covariance / denominator) if denominator > 0 else np.nan


def _shared_effects(rng, rg, m, pop, indices, *, p_causal=P_CAUSAL, h2=H2):
    causal = rng.random(m) < p_causal
    transform = np.linalg.cholesky([[1.0, rg], [rg, 1.0]])
    raw = transform @ rng.standard_normal((2, causal.sum()))
    beta1, beta2 = np.zeros(m), np.zeros(m)
    beta1[causal], beta2[causal] = raw
    return (_scale_effects(beta1, h2, pop, indices),
            _scale_effects(beta2, h2, pop, indices))


def _disjoint_effects(rng, m, pop, indices, *, p_causal=P_CAUSAL, h2=H2):
    causal1, causal2 = _disjoint_masks(rng, m, p_causal)
    beta1, beta2 = np.zeros(m), np.zeros(m)
    beta1[causal1] = rng.standard_normal(causal1.sum())
    beta2[causal2] = rng.standard_normal(causal2.sum())
    return (_scale_effects(beta1, h2, pop, indices),
            _scale_effects(beta2, h2, pop, indices))


def _write_csv(path, rows):
    """Write the full-precision, one-row-per-replicate numeric record."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _warning_counts(caught):
    counts = {"implausible": 0, "divergence": 0, "other": 0}
    for warning in caught:
        message = str(warning.message).lower()
        if "diverg" in message:
            counts["divergence"] += 1
        elif "implausible bivariate fit" in message:
            counts["implausible"] += 1
        else:
            counts["other"] += 1
    return counts


def _run_arm(args, lib_r, shrinkage):
    pop, chol_pop, ref, indices = _build_panels(
        lib_r, n_ref=args.n_ref, shrinkage=shrinkage,
        blocks=args.blocks, seed=0)
    m = sum(idx.size for idx in indices)
    cases = [
        (f"shared, rg={rg:.1f}", rg,
         lambda rng, rg=rg: _shared_effects(rng, rg, m, pop, indices))
        for rg in (0.0, 0.3, 0.6, 0.9)
    ]
    cases.append(("disjoint causal", None,
                  lambda rng: _disjoint_effects(rng, m, pop, indices)))

    rows = []
    print(f"trait2 genetic R2, finite-reference LD (Nref={args.n_ref}, "
          f"shrinkage={shrinkage:g}); N1={args.n1}, N2={args.n2}, "
          f"h2={H2}, m={m}, {args.reps} reps\n")
    print(f"{'architecture':>22} | {'alone':>6} | {'joint':>6} | "
          f"{'gain':>6} | {'rg_est':>6}")
    print("-" * 60)
    for label, target_rg, simulate in cases:
        case_rows = []
        for rep in range(args.reps):
            rng = np.random.default_rng(300 + rep)
            beta1, beta2 = simulate(rng)
            beta_hat1 = _sumstats(beta1, args.n1, rng, pop, chol_pop, indices)
            beta_hat2 = _sumstats(beta2, args.n2, rng, pop, chol_pop, indices)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = ldpred3_auto_bivariate_blocks(
                    ref, beta_hat1, beta_hat2, args.n1, args.n2,
                    burn_in=args.burn_in, num_iter=args.num_iter, seed=rep)
            solo_beta = ldpred3_by_blocks(
                ref, beta_hat2, np.full(m, float(args.n2)), method="auto",
                burn_in=args.burn_in, num_iter=args.num_iter, seed=rep)
            solo = _prediction_r2(solo_beta, beta2, pop, indices)
            joint = _prediction_r2(result.beta2_est, beta2, pop, indices)
            row = {
                "architecture": label,
                "target_rg": "" if target_rg is None else target_rg,
                "realized_rg": _realized_rg(beta1, beta2, pop, indices),
                "replicate": rep,
                "m": m,
                "n1": args.n1,
                "n2": args.n2,
                "n_ref": args.n_ref,
                "reference_shrinkage": shrinkage,
                "p_causal": P_CAUSAL,
                "h2": H2,
                "n_causal_1": int(np.count_nonzero(beta1)),
                "n_causal_2": int(np.count_nonzero(beta2)),
                "n_shared_causal": int(np.count_nonzero((beta1 != 0) & (beta2 != 0))),
                "solo_r2": solo,
                "joint_r2": joint,
                "gain": joint - solo,
                "rg_est": float(result.rg),
                "joint_p": float(result.p),
                "joint_h2_1": float(result.h2[0]),
                "joint_h2_2": float(result.h2[1]),
                "joint_warned": int(bool(caught)),
            }
            counts = _warning_counts(caught)
            row.update(
                joint_warning_count=len(caught),
                joint_implausible_warnings=counts["implausible"],
                joint_divergence_warnings=counts["divergence"],
                joint_other_warnings=counts["other"],
            )
            rows.append(row)
            case_rows.append(row)
        alone = np.mean([row["solo_r2"] for row in case_rows])
        joint = np.mean([row["joint_r2"] for row in case_rows])
        rg_est = np.mean([row["rg_est"] for row in case_rows])
        warned = sum(row["joint_warned"] for row in case_rows)
        implausible = sum(row["joint_implausible_warnings"] for row in case_rows)
        divergence = sum(row["joint_divergence_warnings"] for row in case_rows)
        other = sum(row["joint_other_warnings"] for row in case_rows)
        warning_note = (f"  WARN {warned}/{args.reps} "
                        f"(impl {implausible}, div {divergence}, other {other})"
                        if warned else "")
        print(f"{label:>22} | {alone:>6.3f} | {joint:>6.3f} | "
              f"{joint-alone:>+6.3f} | {rg_est:>+6.2f}{warning_note}")
    return rows


def _run(args, *, source_revision, dependency_sources):
    library_hash = sha256_file(args.library)
    with np.load(args.library) as archive:
        lib_r = np.array(archive["R"], dtype=np.float64)
        simulator_cache_tag = (
            str(archive["simulator_cache_tag"].item())
            if "simulator_cache_tag" in archive else "unrecorded")
    rows = []
    started = time.time()
    for shrinkage in args.shrinkage:
        rows.extend(_run_arm(args, lib_r, shrinkage))
        print()

    _write_csv(args.csv, rows)
    sidecar = write_provenance_sidecar(
        args.csv,
        source_revision=source_revision,
        input_hashes={"ld_library.npz": library_hash},
        dependency_sources=dependency_sources,
        run_controls={
            "blocks": args.blocks,
            "burn_in": args.burn_in,
            "n1": args.n1,
            "n2": args.n2,
            "n_ref": args.n_ref,
            "num_iter": args.num_iter,
            "reps": args.reps,
            "reference_shrinkage": list(args.shrinkage),
            "simulator_cache_tag": simulator_cache_tag,
        },
    )
    print(f"wrote {args.csv} and {sidecar}\n"
          f"({time.time() - started:.0f}s)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", default="ld_library.npz")
    parser.add_argument("--csv", default="benchmarks/bivariate_demo.csv")
    parser.add_argument("--blocks", type=int,
                        help="use only the first blocks (reduced smoke runs)")
    parser.add_argument("--n-ref", type=int, default=N_REF)
    parser.add_argument("--n1", type=int, default=N1)
    parser.add_argument("--n2", type=int, default=N2)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--burn-in", type=int, default=150)
    parser.add_argument("--num-iter", type=int, default=200)
    parser.add_argument("--shrinkage", type=float, nargs="+", default=[0.0, 0.05],
                        help="paired reference-LD shrinkage arms")
    args = parser.parse_args(argv)
    if args.reps < 1 or args.n_ref < 2 or args.n1 <= 0 or args.n2 <= 0:
        parser.error("reps >= 1, n-ref >= 2 and positive sample sizes required")
    if any(not 0.0 <= value <= 1.0 for value in args.shrinkage):
        parser.error("shrinkage values must be in [0, 1]")
    source_revision = require_clean_source()
    dependency_sources = {"ldpred3": require_ldpred3_source()}
    _run(args, source_revision=source_revision,
         dependency_sources=dependency_sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
