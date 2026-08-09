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
    ax.set_xlim(0.30, 1.16)
    ax.set_ylim(-0.28, 1.18)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    return fig


def fig_polygenicity(live):
    """Lp(a) is two orders of magnitude less polygenic than everything else."""
    per_trait = {}
    for r in live:
        t1, t2 = r["pair"].split(" x ")
        per_trait.setdefault(t1, []).append(int(r["n_causal_1"]))
        per_trait.setdefault(t2, []).append(int(r["n_causal_2"]))
    order = sorted(per_trait, key=lambda t: sum(per_trait[t]) / len(per_trait[t]))

    fig, ax = plt.subplots(figsize=(6.9, 3.4))
    for i, trait in enumerate(order):
        hit = trait == "Lp(a)"
        vals = per_trait[trait]
        if len(vals) > 1:
            ax.hlines(i, min(vals), max(vals), color=HILITE if hit else PALE,
                      lw=1.4, zorder=1)
        ax.plot(vals, [i] * len(vals), "o", ms=6, alpha=0.85, zorder=2,
                color=HILITE if hit else INK)
        if trait in ("Lp(a)", "CAD"):
            text = (f"{min(vals):,}" if min(vals) == max(vals)
                    else f"{min(vals):,}–{max(vals):,}")
            ax.annotate(text, (max(vals), i), textcoords="offset points",
                        xytext=(9, -3), fontsize=7.5,
                        color=HILITE if hit else INK)
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
        live = [r for r in csv.DictReader(fh) if r["status"] == "ok"]
    style()
    save(fig_ordering(live), "rho-beta-vs-rg")
    save(fig_frac_shared(live), "frac-shared")
    save(fig_polygenicity(live), "polygenicity")
    print(f"wrote 3 figures from {len(live)} fitted pairs")


if __name__ == "__main__":
    main()
