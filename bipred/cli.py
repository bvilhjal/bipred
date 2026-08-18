"""Command line: one LDpred3 cache + two GWAS files -> bivariate weights."""

from __future__ import annotations

import argparse
import math
import sys


def _column_mapping(values, parser, option):
    """Parse repeated ``FIELD=COLUMN`` specifications for one GWAS."""
    out = {}
    for value in values:
        if "=" not in value:
            parser.error(f"{option} requires FIELD=COLUMN, got {value!r}")
        field, column = (part.strip() for part in value.split("=", 1))
        if not field or not column:
            parser.error(f"{option} requires non-empty FIELD=COLUMN")
        if field in out:
            parser.error(f"{option} repeats canonical field {field!r}")
        out[field] = column
    return out


def build_parser():
    p = argparse.ArgumentParser(
        prog="bipred",
        description="Joint two-trait LDpred3 fit from an LD cache and two GWAS "
                    "files.")
    p.add_argument("--ld-cache", required=True,
                   help="cache written by ldpred3.save_ld_blocks")
    p.add_argument("--sumstats1", required=True)
    p.add_argument("--sumstats2", required=True)
    p.add_argument("--n-eff1", type=float)
    p.add_argument("--n-eff2", type=float)
    p.add_argument("--n-cases1", type=float)
    p.add_argument("--n-controls1", type=float)
    p.add_argument("--n-cases2", type=float)
    p.add_argument("--n-controls2", type=float)
    p.add_argument(
        "--column1", action="append", default=[], metavar="FIELD=COLUMN",
        help="trait-1 column override; repeat for id, chrom, pos, ea, oa, "
             "beta/or, se, pval, n_eff, eaf, or info")
    p.add_argument(
        "--column2", action="append", default=[], metavar="FIELD=COLUMN",
        help="trait-2 column override; repeat as needed")

    qc = p.add_argument_group("summary-statistic QC")
    qc.add_argument("--no-qc", action="store_true",
                    help="disable ldpred3.qc_sumstats (not recommended)")
    qc.add_argument("--min-n-ratio", type=float, default=0.7)
    qc.add_argument("--min-maf", type=float, default=0.01)
    qc.add_argument("--min-info", type=float, default=0.7)
    qc.add_argument(
        "--max-chisq", type=float,
        help="optional GWAS chi-square deletion; normally leave unset because "
             "large signals belong in the prediction model")
    qc.add_argument("--keep-duplicates", action="store_true")
    qc.add_argument(
        "--min-af-corr", type=float,
        help="minimum aligned GWAS/cache effect-allele-frequency correlation")

    screen = p.add_argument_group("LD-consistency sensitivity screen")
    screen.add_argument("--screen", action="store_true",
                        help="screen the genuine joint panel for both traits")
    screen.add_argument("--screen-rounds", type=int, default=4)
    screen.add_argument("--screen-window", type=int, default=1000)
    screen.add_argument("--screen-threshold", type=float, default=29.72)
    screen.add_argument("--screen-eigenvalue-floor", type=float, default=1e-3)
    screen.add_argument("--screen-seed", type=int)
    screen.add_argument(
        "--screen-ncores", type=int,
        help="screening threads; defaults to --ncores")
    screen.add_argument("--screen-verbose", action="store_true")

    fit = p.add_argument_group("fit")
    fit.add_argument(
        "--cross-corr", type=float, default=0.0,
        help="correlation of cross-trait sampling errors")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--ncores", type=int, default=1,
                     help="within-chain block threads")
    fit.add_argument("--n-chains", type=int, default=1,
                     help="1 for a single fit; >=2 for dispersed chains")
    fit.add_argument("--chain-ncores", type=int, default=1,
                     help="concurrent chains; cannot combine with --ncores > 1")
    fit.add_argument("--burn-in", type=int, default=200)
    fit.add_argument("--num-iter", type=int, default=200)

    out = p.add_argument_group("output")
    out.add_argument("--out-weights1", help="LDpred3 weight file for trait 1")
    out.add_argument("--out-weights2", help="LDpred3 weight file for trait 2")
    out.add_argument(
        "--hwe-frozen-scale", action="store_true",
        help="write cache-AF/HWE SD_REF columns; an approximation, not an "
             "observed fit-cohort dosage scale. Default weights use target "
             "scaling and omit AF_REF/SD_REF")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    columns1 = _column_mapping(args.column1, parser, "--column1")
    columns2 = _column_mapping(args.column2, parser, "--column2")
    if args.n_chains < 1:
        parser.error("--n-chains must be 1 or at least 2")
    if args.ncores < 1 or args.chain_ncores < 1:
        parser.error("--ncores and --chain-ncores must be at least 1")
    if not math.isfinite(args.cross_corr) or not -1.0 < args.cross_corr < 1.0:
        parser.error("--cross-corr must be finite and lie in (-1, 1)")
    if args.n_chains == 1 and args.chain_ncores != 1:
        parser.error("--chain-ncores applies only when --n-chains is at least 2")
    if args.n_chains > 1 and args.ncores > 1 and args.chain_ncores > 1:
        parser.error("choose --ncores or --chain-ncores, not both")
    if args.burn_in < 0 or args.num_iter < 1:
        parser.error("--burn-in must be non-negative and --num-iter positive")
    if args.n_chains > 1 and (args.num_iter < 4 or args.num_iter % 2):
        parser.error("multi-chain --num-iter must be even and at least 4")
    if (args.hwe_frozen_scale
            and not (args.out_weights1 or args.out_weights2)):
        parser.error("--hwe-frozen-scale requires a weight output")
    for n_eff, n_cases, n_controls, name in (
            (args.n_eff1, args.n_cases1, args.n_controls1, "trait 1"),
            (args.n_eff2, args.n_cases2, args.n_controls2, "trait 2")):
        if n_eff is not None and (n_cases is not None or n_controls is not None):
            parser.error(
                f"{name}: pass either --n-eff or --n-cases/--n-controls, "
                "not both")

    from . import (
        ldpred3_auto_bivariate_blocks,
        ldpred3_auto_bivariate_chains,
        prepare_bivariate_sumstats,
    )

    qc_params = dict(
        min_n_ratio=args.min_n_ratio, min_maf=args.min_maf,
        min_info=args.min_info, max_chisq=args.max_chisq,
        drop_duplicates=not args.keep_duplicates)
    screen_ncores = (args.ncores if args.screen_ncores is None
                     else args.screen_ncores)
    if screen_ncores < 1:
        parser.error("--screen-ncores must be at least 1")
    screen_seed = args.seed if args.screen_seed is None else args.screen_seed
    prep = prepare_bivariate_sumstats(
        args.ld_cache, args.sumstats1, args.sumstats2,
        n_eff1=args.n_eff1, n_eff2=args.n_eff2,
        n_cases1=args.n_cases1, n_controls1=args.n_controls1,
        n_cases2=args.n_cases2, n_controls2=args.n_controls2,
        columns1=columns1, columns2=columns2,
        qc=not args.no_qc, qc_params=qc_params,
        min_af_corr=args.min_af_corr, screen=args.screen,
        screen_rounds=args.screen_rounds, screen_window=args.screen_window,
        screen_threshold=args.screen_threshold,
        screen_eigenvalue_floor=args.screen_eigenvalue_floor,
        screen_seed=screen_seed, screen_ncores=screen_ncores,
        screen_verbose=args.screen_verbose)
    try:
        print(
            f"kept {prep.log['n_kept']} of {prep.log['n_cache']} cache variants "
            f"({prep.log['n_joint']} jointly observed, "
            f"{prep.log['n_screen_drop']} screen drops)",
            file=sys.stderr)
        if args.hwe_frozen_scale and prep.af is None:
            parser.error(
                "--hwe-frozen-scale requires reference_af in the LD cache; "
                "omit it to write target-scaled weights")

        fit_kwargs = dict(
            burn_in=args.burn_in, num_iter=args.num_iter,
            ncores=args.ncores, cross_corr=args.cross_corr, seed=args.seed)
        if args.n_chains == 1:
            result = ldpred3_auto_bivariate_blocks(
                prep.blocks, prep.beta_hat1, prep.beta_hat2,
                prep.n_eff1, prep.n_eff2, **fit_kwargs)
            res = result
        else:
            multi = ldpred3_auto_bivariate_chains(
                prep.blocks, prep.beta_hat1, prep.beta_hat2,
                prep.n_eff1, prep.n_eff2, n_chains=args.n_chains,
                chain_ncores=args.chain_ncores, **fit_kwargs)
            res = multi.posterior
            finite_rhat = [value for value in multi.basic_split_rhat.rhat.values()
                           if value == value]
            if finite_rhat:
                print(f"max basic split-Rhat={max(finite_rhat):.4g}",
                      file=sys.stderr)

        print(res)
        print(f"h2={res.h2}  rg={res.rg}")
        common = dict(
            id=prep.id, chrom=prep.chrom, pos=prep.pos,
            effect_allele=prep.effect_allele,
            other_allele=prep.other_allele,
            af=prep.af if args.hwe_frozen_scale else None)
        if args.out_weights1:
            res.write_weights(args.out_weights1, trait=1, **common)
            print(f"wrote {args.out_weights1}")
        if args.out_weights2:
            res.write_weights(args.out_weights2, trait=2, **common)
            print(f"wrote {args.out_weights2}")
    finally:
        prep.close()
    return 0


if __name__ == "__main__":                    # pragma: no cover
    raise SystemExit(main())
