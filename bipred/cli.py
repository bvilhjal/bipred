"""Thin CLI: ldpred3 cache + two GWAS files → bivariate fit + weight files."""

from __future__ import annotations

import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(
        prog="bipred",
        description="Joint two-trait LDpred3 fit from an ldpred3 LD cache "
                    "and two GWAS files.")
    p.add_argument("--ld-cache", required=True,
                   help="ldpred3 save_ld_blocks cache")
    p.add_argument("--sumstats1", required=True)
    p.add_argument("--sumstats2", required=True)
    p.add_argument("--n-eff1", type=float)
    p.add_argument("--n-eff2", type=float)
    p.add_argument("--n-cases1", type=float)
    p.add_argument("--n-controls1", type=float)
    p.add_argument("--n-cases2", type=float)
    p.add_argument("--n-controls2", type=float)
    p.add_argument("--screen", action="store_true",
                   help="apply ld_consistency_screen to both traits")
    p.add_argument("--out-weights1", help="ldpred3 weight file for trait 1")
    p.add_argument("--out-weights2", help="ldpred3 weight file for trait 2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ncores", type=int, default=1)
    p.add_argument("--burn-in", type=int, default=200)
    p.add_argument("--num-iter", type=int, default=200)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from . import (ldpred3_auto_bivariate_blocks, prepare_bivariate_sumstats)

    prep = prepare_bivariate_sumstats(
        args.ld_cache, args.sumstats1, args.sumstats2,
        n_eff1=args.n_eff1, n_eff2=args.n_eff2,
        n_cases1=args.n_cases1, n_controls1=args.n_controls1,
        n_cases2=args.n_cases2, n_controls2=args.n_controls2,
        screen=args.screen)
    print(f"kept {prep.log['n_kept']} of {prep.log['n_cache']} cache variants",
          file=sys.stderr)
    res = ldpred3_auto_bivariate_blocks(
        prep.blocks, prep.beta_hat1, prep.beta_hat2, prep.n_eff1, prep.n_eff2,
        burn_in=args.burn_in, num_iter=args.num_iter, ncores=args.ncores,
        seed=args.seed)
    print(res)
    print(f"h2={res.h2}  rg={res.rg}")
    common = dict(id=prep.id, chrom=prep.chrom, pos=prep.pos,
                  effect_allele=prep.effect_allele,
                  other_allele=prep.other_allele, af=prep.af)
    if args.out_weights1:
        res.write_weights(args.out_weights1, trait=1, **common)
        print(f"wrote {args.out_weights1}")
    if args.out_weights2:
        res.write_weights(args.out_weights2, trait=2, **common)
        print(f"wrote {args.out_weights2}")
    return 0
