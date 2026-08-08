"""Peak memory of a bivariate fit, by LD representation and variant count.

The suite measured time and accuracy but never the memory a *fit* costs on top
of the LD it was handed, which is the quantity that decides whether a
genome-scale run survives. It is also the quantity that regressed silently: the
fit-time ``ld_int8`` default used to quantise float blocks into a private copy,
so a caller holding a float panel paid for a second genome-scale LD payload
while their own was still alive -- k bytes per variant, invisible in every
timing table.

Two numbers are reported per cell:

``payload``
    Bytes the caller's own LD blocks occupy, computed from distinct arrays. The
    synthetic blocks have the same spectrum but never alias one payload: a
    100,000-variant D32 panel with k=500 therefore occupies about 200 MB, not
    the 1 MB an accidentally repeated object would report. This is the baseline
    the fit is measured against.
``fit peak``
    ``tracemalloc`` peak *inside* the ``ldpred3_auto_bivariate_blocks`` call,
    which is what the fit adds. Python-level allocation only -- Numba's
    workspaces are native and do not appear -- so read it as the allocation the
    driver is responsible for, not as process RSS. ``--rss`` prints one process
    high-water mark for the entire invocation; it is deliberately not attached
    to individual rows because ``ru_maxrss`` is cumulative.

Usage::

    python -m benchmarks.fit_memory                    # the default grid
    python -m benchmarks.fit_memory --m 200000 --rss
    python -m benchmarks.fit_memory --csv fit_memory.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tracemalloc
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.sweep_cost import _blocks, _sumstats                 # noqa: E402


def _payload_bytes(blocks):
    """Resident bytes of the caller's LD, counting each storage owner once."""
    def owner(array):
        while isinstance(array.base, np.ndarray):
            array = array.base
        return array

    seen = {}
    for block, _idx in blocks:
        factor = getattr(block, "U", None)
        if factor is None:
            arrays = [np.asarray(block)]
        else:
            arrays = [np.asarray(factor), np.asarray(block.residual_diag)]
        for array in arrays:
            root = owner(array)
            seen[id(root)] = root.nbytes
    return sum(seen.values())


def _peak_rss_bytes():
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return usage if sys.platform == "darwin" else usage * 1024


def _cell(kind, m, k, ld_int8):
    from bipred import ldpred3_auto_bivariate_blocks

    n_eff = 100_000.0
    blocks, label, dense = _blocks(kind, m, k)
    beta_hat1, beta_hat2 = _sumstats(dense, m, k, n_eff=n_eff)
    payload = _payload_bytes(blocks)

    kwargs = {} if ld_int8 is None else {"ld_int8": ld_int8}

    def fit():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ldpred3_auto_bivariate_blocks(blocks, beta_hat1, beta_hat2,
                                          n_eff, n_eff, burn_in=2, num_iter=2,
                                          seed=0, **kwargs)

    # Warm up outside the trace. The first fit in a process compiles the Numba
    # kernels, and their Python-side allocation lands in whichever cell happens
    # to run first -- which read as 449 bytes/variant against 121 for the same
    # configuration at twice the size.
    fit()
    tracemalloc.start()
    fit()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    row = {"representation": label, "m": m, "k": k,
           "blocks": len(blocks), "block_storage": "distinct",
           "ld_int8": "default" if ld_int8 is None else str(ld_int8),
           "payload_mb": payload / 1e6, "fit_peak_mb": peak / 1e6,
           "fit_peak_bytes_per_variant": peak / m}
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", default=[50_000, 100_000])
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--kinds", nargs="+",
                        default=["dense_f32", "dense_i8", "lr8_0.99"])
    parser.add_argument("--rss", action="store_true",
                        help="report one high-water mark for the whole run")
    parser.add_argument("--csv")
    args = parser.parse_args(argv)

    rows = []
    print(f"k={args.k}   payload = the caller's own LD; "
          f"fit peak = allocation inside the fit\n")
    header = f"{'representation':22} {'m':>7} {'ld_int8':>8} {'payload MB':>11} {'fit peak MB':>12} {'B/variant':>10}"
    print(header)
    print("-" * len(header))
    for kind in args.kinds:
        for m in args.m:
            # The current default consumes D32 as supplied. Explicit True is
            # retained on float blocks to measure the legacy private D8 copy;
            # it cannot change an already-D8 or low-rank payload.
            settings = [None] if kind != "dense_f32" else [None, True]
            for ld_int8 in settings:
                row = _cell(kind, m, args.k, ld_int8)
                rows.append(row)
                print(f"{row['representation']:22} {row['m']:7d} "
                      f"{row['ld_int8']:>8} {row['payload_mb']:11.2f} "
                      f"{row['fit_peak_mb']:12.2f} "
                      f"{row['fit_peak_bytes_per_variant']:10.1f}")
        print()

    if args.rss:
        print(f"whole-process peak RSS: {_peak_rss_bytes() / 1e6:.2f} MB\n")

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
