"""How much of each trait's signal does the chi-square cap of 80 remove?

The per-variant QC in both drivers caps chi-square at 80. That cap exists for
LD Score regression, whose linear model gives an uncapped large-effect variant
near-full leverage on the slope -- ``bipred.ldsc.ldsc_rg`` says so and tells the
caller to cap before calling it. But the drivers apply the mask to the *shared*
variant set, so the cap also removes those variants from the bivariate fit,
which is a mixture model with a slab component built to hold large effects.

For a polygenic trait that costs almost nothing. For a trait whose heritability
sits in one locus it removes the trait. This script measures which is which,
counting only variants that are in the LD reference, since those are the only
ones a fit would have seen.

    python chi2_cap_audit.py --sumstats <dir> --ldref <ldpred3_ldref_hm3.npz>

Writes chi2_cap_audit.csv beside this file.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 80.0

if sys.version_info >= (3, 11):
    import tomllib
else:                                            # pragma: no cover
    import tomli as tomllib


def opener(path, gzipped):
    return gzip.open(path, "rt") if gzipped else open(path)


def audit(path, cols, gzipped, keep_ids, odds_ratio):
    """Return counts and summed chi-square, overall and above the cap."""
    with opener(path, gzipped) as fh:
        # Releases here are tab- or space-separated; sniff it from the header.
        # None means "split on any whitespace run", which is right for the
        # space-separated GLGC files and wrong for tabs with empty fields.
        first = fh.readline().rstrip("\n")
        sep = "\t" if "\t" in first else None
        header = first.split(sep)
        try:
            irs = header.index(cols["rsid"])
            ib = header.index(cols["effect"])
            ise = header.index(cols["se"])
        except ValueError as exc:
            return {"error": f"missing column: {exc}"}

        n = over = 0
        chi_all = chi_over = 0.0
        biggest = (0.0, "")
        for line in fh:
            f = line.rstrip("\n").split(sep)
            try:
                rs = f[irs]
                if rs not in keep_ids:
                    continue
                b, se = float(f[ib]), float(f[ise])
            except (ValueError, IndexError):
                continue
            if not np.isfinite(b) or not np.isfinite(se) or se <= 0:
                continue
            if odds_ratio:
                if b <= 0:
                    continue
                b = np.log(b)
            chi = (b / se) ** 2
            n += 1
            chi_all += chi
            if chi > CAP:
                over += 1
                chi_over += chi
                if chi > biggest[0]:
                    biggest = (chi, rs)
    return {"m_in_ref": n, "n_over_cap": over, "chi2_total": chi_all,
            "chi2_over_cap": chi_over, "max_chi2": biggest[0],
            "max_chi2_rsid": biggest[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumstats", required=True, help="directory of GWAS files")
    ap.add_argument("--ldref", required=True, help="ldpred3_ldref_hm3.npz")
    ap.add_argument("--out", default=os.path.join(HERE, "chi2_cap_audit.csv"))
    args = ap.parse_args()

    with open(os.path.join(HERE, "traits.toml"), "rb") as fh:
        reg = tomllib.load(fh)
    keep_ids = set(np.load(args.ldref, allow_pickle=True)["ids"].tolist())
    print(f"LD reference: {len(keep_ids):,} variants")

    rows = []
    for name, t in sorted(reg["traits"].items()):
        path = os.path.join(args.sumstats, t["file"])
        if not os.path.exists(path):
            print(f"  {name}: file not found, skipped")
            continue
        res = audit(path, reg["columns"][t["columns"]], t.get("gzipped", False),
                    keep_ids,
                    "effect_is_odds_ratio" in t.get("quirks", []))
        if "error" in res:
            print(f"  {name}: {res['error']}")
            continue
        frac = 100 * res["chi2_over_cap"] / res["chi2_total"] if \
            res["chi2_total"] else 0.0
        rows.append(dict(trait=name, **res,
                         pct_chi2_removed=round(frac, 2),
                         pct_variants_removed=round(
                             100 * res["n_over_cap"] / max(res["m_in_ref"], 1),
                             4)))
        print(f"  {name:20s} m={res['m_in_ref']:>9,}  "
              f"over cap={res['n_over_cap']:>6,}  "
              f"chi2 removed={frac:6.2f}%  max={res['max_chi2']:,.0f} "
              f"({res['max_chi2_rsid']})")

    rows.sort(key=lambda r: -r["pct_chi2_removed"])
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} traits)")


if __name__ == "__main__":
    main()
