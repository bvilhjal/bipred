"""Draw this study's figures from results/estimates.csv.

Reads nothing else -- in particular not the summary statistics, so the figures
rebuild on a machine without 9 GB of GWAS files. Writes figures/*.pdf, which
REPORT.tex includes.

    python figures.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
#: The 2x2x2 factorial's arm 001 -- lenient filters, long-range LD kept, screen
#: on -- which is the arm every pair in this study was fitted under.
BENCH = os.path.join(REPO, "benchmarks", "qc_factorial.csv")
ARM_001 = {"strict": "0", "drop_long_range": "0", "ld_screen": "1"}
FOCAL = "Lp(a) x CAD"

INK, MUTED, HILITE = "#22303f", "#93a3b3", "#d1495b"
PALE, BAND = "#c9d3dc", "#eef1f4"

#: Where each label sits relative to its point: (dx, dy) in points, and the
#: horizontal alignment. Hand-placed because five of the eleven pairs collide
#: on the default above-centre.
LABEL = {
    "urate x gout": (0, 10, "center"),
    "dbilirubin x bilirubin": (0, -16, "center"),
    "urate x cystatinC": (0, -16, "center"),
    "gout x CRP": (-6, 8, "right"),
    "cystatinC x CRP": (12, 2, "left"),
    "ALP x GGT": (12, -5, "left"),
    "Lp(a) x CAD": (0, 11, "center"),
    "Lp(a) x VTE": (0, 10, "center"),
    "GGT x bilirubin": (0, -15, "center"),
    "SHBG x ALP": (10, -14, "left"),
    "CRP x Lp(a)": (0, -16, "center"),
}

#: Same, for the h2-agreement panel. Anchored on each trait's largest point.
LABEL_H2 = {
    "Lp(a)": (10, -3, "left"),
    "CAD": (10, -3, "left"),
    "urate": (10, -3, "left"),
    "VTE": (-2, 9, "center"),
    "dbilirubin": (0, -13, "center"),
    "bilirubin": (10, -3, "left"),
    "gout": (-9, 2, "right"),
    "CRP": (2, -13, "center"),
    "ALP": (-9, -7, "right"),
    "GGT": (-9, 2, "right"),
    "SHBG": (9, -3, "left"),
    "cystatinC": (9, 0, "left"),
    # from the QC factorial
    "height": (9, -3, "left"),
    "LDL": (10, 4, "left"),
    "HDL": (10, -4, "left"),
    "TG": (0, -14, "center"),
}

#: Same, for the rg-agreement panel, whose points fall differently.
LABEL_RG = {
    "urate x gout": (0, -15, "center"),
    "dbilirubin x bilirubin": (0, 10, "center"),
    "urate x cystatinC": (8, -3, "left"),
    "gout x CRP": (8, -3, "left"),
    "cystatinC x CRP": (9, 3, "left"),
    "ALP x GGT": (0, -15, "center"),
    "Lp(a) x CAD": (0, 11, "center"),
    "Lp(a) x VTE": (0, -15, "center"),
    "GGT x bilirubin": (0, 9, "center"),
    "SHBG x ALP": (0, -15, "center"),
    "CRP x Lp(a)": (8, 3, "left"),
    # from the QC factorial
    "LDL x CAD": (0, 10, "center"),
    "height x LDL": (-9, 4, "right"),
    "HDL x TG": (0, 10, "center"),
}


def style():
    plt.rcParams.update({
        "savefig.dpi": 220, "savefig.bbox": "tight", "font.size": 9,
        "axes.labelsize": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "text.color": INK, "xtick.color": INK,
        "ytick.color": INK, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
        "grid.color": "#dfe5ea", "grid.linewidth": 0.7,
    })


def pretty(pair):
    return pair.replace(" x ", " × ")


def load_bench():
    """Arm-001 rows of the QC factorial: the same arm, three more pairs.

    Unlike results/estimates.csv these carry `*_iterate_sd` -- the spread of
    the quantity across retained Gibbs iterates. That is Monte-Carlo
    variability of the fit, not a confidence interval: it says nothing about
    the sampling error of the GWAS behind it, which is why it comes out around
    1% of h2. Drawn as bars only to show where any bipred spread is recorded
    at all.
    """
    if not os.path.exists(BENCH):
        return []
    with open(BENCH, newline="") as fh:
        return [r for r in csv.DictReader(fh)
                if all(r[k] == v for k, v in ARM_001.items())]


def save(fig, stem):
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", stem + ".pdf"))
    plt.close(fig)


def fig_ordering(live):
    """rho_beta orders by biology; rg does not."""
    order = sorted(live, key=lambda r: float(r["rho_beta"]))
    y = range(len(order))
    rho = [float(r["rho_beta"]) for r in order]
    rg = [float(r["rg"]) for r in order]
    focal = [r["pair"] == FOCAL for r in order]

    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    for i in range(0, len(order), 2):
        ax.axhspan(i - 0.5, i + 0.5, color="#f6f8f9", zorder=0)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    for i, (a, hit) in enumerate(zip(rho, focal)):
        ax.hlines(i, 0, a, color=HILITE if hit else PALE, lw=1.6, zorder=2)
    ax.scatter(rg, y, s=40, marker="s", facecolors="white", edgecolors=MUTED,
               linewidths=1.2, zorder=3, label="rg  (genome-wide)")
    ax.scatter(rho, y, s=46, zorder=4, label="rho_beta  (shared component)",
               color=[HILITE if hit else INK for hit in focal])

    ax.set_yticks(list(y))
    ax.set_yticklabels([pretty(r["pair"]) for r in order])
    for hit, label in zip(focal, ax.get_yticklabels()):
        if hit:
            label.set_color(HILITE)
            label.set_fontweight("bold")
    ax.set_xlim(-0.25, 1.12)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("effect correlation")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    return fig


def fig_frac_shared(live):
    """frac_shared does not discriminate; rho_beta does."""
    # The uninformative middle: over this range of frac_shared, rho_beta takes
    # both null and clearly-related values. Naming it is more honest than
    # implying frac_shared carries no information at all -- across the full
    # range it plainly does.
    lo, hi = 0.60, 0.82
    inband = [r for r in live if lo <= float(r["frac_shared"]) <= hi]
    mid = [float(r["rho_beta"]) for r in inband]
    share = [float(r["frac_shared"]) for r in inband]

    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    ax.axvspan(lo, hi, color="#f4f7f9", zorder=0)
    ax.axhspan(-0.12, -0.04, color=BAND, zorder=0)
    ax.text(1.14, -0.08, "null band", fontsize=7.5, color=MUTED, ha="right",
            va="center")
    ax.annotate("", xy=(0.572, min(mid)), xytext=(0.572, max(mid)),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
    ax.text((lo + hi) / 2, 1.10,
            f"{len(mid)} pairs share {min(share):.0%}–{max(share):.0%} of "
            f"their causal set;\ntheir rho_beta spans {min(mid):+.2f} to "
            f"{max(mid):+.2f}",
            fontsize=7.5, color=INK, ha="center", va="top")
    ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
    for row in live:
        x, y = float(row["frac_shared"]), float(row["rho_beta"])
        hit = row["pair"] == FOCAL
        ax.scatter([x], [y], s=46, zorder=3, color=HILITE if hit else INK)
        dx, dy, ha = LABEL.get(row["pair"], (0, 10, "center"))
        ax.annotate(pretty(row["pair"]), (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=7,
                    color=HILITE if hit else MUTED,
                    fontweight="bold" if hit else "normal")
    ax.set_xlabel("frac_shared   (share of the smaller causal set that is shared)")
    ax.set_ylabel("rho_beta")
    # Wide enough for the least-shared pair: uncapped, CRP x Lp(a) and
    # Lp(a) x VTE sit near 0.24, and an axis starting at 0.30 dropped them
    # from the figure without saying so.
    ax.set_xlim(0.16, 1.16)
    ax.set_ylim(-0.28, 1.18)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    return fig


def fig_h2_agreement(live, bench):
    """Per-trait h2: the joint fit against LD Score regression.

    One point per trait per pairing. A trait in several pairs is estimated
    several times over slightly different variant sets, so the spread along a
    trait's line is empirical variability -- the only kind available for the
    study pairs, neither estimator recording a standard error for h2.
    """
    per_trait = {}
    for r in live:
        t1, t2 = r["pair"].split(" x ")
        per_trait.setdefault(t1, []).append(
            (float(r["ldsc_h2_1"]), float(r["h2_1"]), 0.0, "study"))
        per_trait.setdefault(t2, []).append(
            (float(r["ldsc_h2_2"]), float(r["h2_2"]), 0.0, "study"))
    for r in bench:
        t1, t2 = r["pair"].split(" x ")
        per_trait.setdefault(t1, []).append(
            (float(r["ldsc_h2_1"]), float(r["h2_1"]),
             float(r["h2_1_iterate_sd"]), "bench"))
        per_trait.setdefault(t2, []).append(
            (float(r["ldsc_h2_2"]), float(r["h2_2"]),
             float(r["h2_2_iterate_sd"]), "bench"))

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.plot([0.004, 0.52], [0.004, 0.52], ls="--", lw=0.9, color=MUTED,
            zorder=1)
    ax.text(0.0125, 0.0104, "equal", fontsize=7.5, color=MUTED, ha="left")

    for trait, points in per_trait.items():
        hit = trait == "Lp(a)"
        colour = HILITE if hit else INK
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if len(points) > 1:
            ax.plot(xs, ys, "-", lw=1.0, color=HILITE if hit else PALE,
                    zorder=2)
        for x, y, sd, source in points:
            ax.errorbar(x, y, yerr=1.96 * sd if sd else None,
                        fmt="^" if source == "bench" else "o", ms=5.5,
                        color=colour, ecolor=colour, elinewidth=1.1,
                        capsize=2, alpha=0.9, zorder=3)
        dx, dy, ha = LABEL_H2.get(trait, (9, -3, "left"))
        ax.annotate(trait, (max(xs), max(ys)), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=7, color=colour,
                    fontweight="bold" if hit else "normal")

    ax.plot([], [], "o", ms=5.5, color=INK, label="this study (11 pairs)")
    ax.plot([], [], "^", ms=5.5, color=INK,
            label="QC factorial, arm 001 (3 pairs; bars = 95% iterate SD)")
    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.0032, 0.62)
    ax.set_ylim(0.0032, 0.62)
    ax.set_aspect("equal")
    ax.set_xlabel("LDSC h2   (log scale)")
    ax.set_ylabel("bipred h2   (log scale)")
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    return fig


def fig_rg_agreement(live, bench):
    """rg: the joint fit against LD Score regression.

    Horizontal bars are LDSC's 95% block-jackknife interval, available for
    every pair. Vertical bars are the 95% iterate SD and exist only for the
    three factorial pairs -- the study CSV records no bipred spread at all.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot([-0.75, 0.90], [-0.75, 0.90], ls="--", lw=0.9, color=MUTED,
            zorder=1)
    ax.text(0.885, 0.855, "equal", fontsize=7.5, color=MUTED, ha="right")
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)

    rows = ([(r, "study") for r in live] + [(r, "bench") for r in bench])
    for row, source in rows:
        hit = row["pair"] == FOCAL
        colour = HILITE if hit else INK
        xi, yi = float(row["ldsc_rg"]), float(row["rg"])
        yerr = (1.96 * float(row["rg_iterate_sd"])
                if source == "bench" else None)
        ax.errorbar(xi, yi, xerr=1.96 * float(row["ldsc_rg_se"]), yerr=yerr,
                    fmt="^" if source == "bench" else "o", ms=5.5,
                    color=colour, ecolor=HILITE if hit else PALE,
                    elinewidth=1.3, capsize=2.5, zorder=3 if hit else 2)
        dx, dy, ha = LABEL_RG.get(row["pair"], (0, 9, "center"))
        ax.annotate(pretty(row["pair"]), (xi, yi), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=7,
                    color=colour, fontweight="bold" if hit else "normal")

    ax.plot([], [], "o", ms=5.5, color=INK, label="this study (11 pairs)")
    ax.plot([], [], "^", ms=5.5, color=INK,
            label="QC factorial, arm 001 (3 pairs)")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.82, 1.24)
    ax.set_ylim(-0.68, 0.95)
    ax.set_xlabel("LDSC rg   (horizontal bars: 95% block-jackknife interval)")
    ax.set_ylabel("bipred rg")
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    return fig


def _per_trait_n_causal(rows):
    per = {}
    for r in rows:
        t1, t2 = r["pair"].split(" x ")
        per.setdefault(t1, []).append(int(r["n_causal_1"]))
        per.setdefault(t2, []).append(int(r["n_causal_2"]))
    return per


def fig_polygenicity(live, capped):
    """Two things at once: Lp(a)'s architecture, and what the cap did to it.

    The chi-square cap removes a trait's largest effects, and the mixture then
    has to explain the same heritability with more small ones. Every trait's
    polygenicity falls when the cap comes off the fit -- by 9% for Lp(a), which
    was already at the floor, and by 93% for dbilirubin.
    """
    per_trait = _per_trait_n_causal(live)
    was = _per_trait_n_causal(capped)
    order = sorted(per_trait, key=lambda t: sum(per_trait[t]) / len(per_trait[t]))

    fig, ax = plt.subplots(figsize=(6.9, 3.8))
    for i, trait in enumerate(order):
        hit = trait == "Lp(a)"
        vals = per_trait[trait]
        old = was.get(trait, [])
        if old:
            a = sum(old) / len(old)
            b = sum(vals) / len(vals)
            ax.annotate("", xy=(b, i), xytext=(a, i),
                        arrowprops=dict(arrowstyle="->", color="#c9d3dc",
                                        lw=1.1, shrinkA=3, shrinkB=3))
            ax.plot(old, [i] * len(old), "o", ms=5, mfc="white", zorder=2,
                    mec=MUTED, mew=1.1)
        ax.plot(vals, [i] * len(vals), "o", ms=6, alpha=0.9, zorder=3,
                color=HILITE if hit else INK)
        if trait in ("Lp(a)", "CAD"):
            text = (f"{min(vals):,}" if min(vals) == max(vals)
                    else f"{min(vals):,}–{max(vals):,}")
            # Above the row: the capped markers sit to the right of the filled
            # ones and the axis label to their left, so both sides are taken.
            ax.annotate(text, (max(vals), i), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7.5,
                        color=HILITE if hit else INK)
    ax.plot([], [], "o", ms=5, mfc="white", mec=MUTED, mew=1.1,
            label="bipred with cap")
    ax.plot([], [], "o", ms=6, color=INK, label="bipred without cap")
    ax.legend(loc="upper left")
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    for label in ax.get_yticklabels():
        if label.get_text() == "Lp(a)":
            label.set_color(HILITE)
            label.set_fontweight("bold")
    ax.set_xlabel("n_causal   (log scale; one point per pairing the trait appears in)")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    return fig


def main():
    with open(os.path.join(HERE, "results", "estimates.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    # The uncapped fits are the study's estimates; the capped ones are the
    # sensitivity comparison they replaced. A diverged fit is in neither.
    live = [r for r in rows
            if r["status"] == "ok" and r["fit_chi2_cap"] == "uncapped"]
    capped = [r for r in rows
              if r["status"] == "ok" and r["fit_chi2_cap"] == "capped"]
    style()
    save(fig_ordering(live), "rho-beta-vs-rg")
    save(fig_frac_shared(live), "frac-shared")
    save(fig_polygenicity(live, capped), "polygenicity")
    bench = load_bench()
    save(fig_h2_agreement(live, bench), "h2-agreement")
    save(fig_rg_agreement(live, bench), "rg-agreement")
    print(f"wrote 5 figures from {len(live)} uncapped pairs "
          f"({len(capped)} capped for comparison) "
          f"+ {len(bench)} factorial pairs")


if __name__ == "__main__":
    main()
