"""Draw this study's figures from results/estimates.csv.

Reads nothing else -- in particular not the summary statistics. Writes
figures/*.pdf, which REPORT.tex includes.

    python figures.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CANCELLATION_LIMIT = 10.0
STAGES = ["harmonised", "per-variant QC", "+ LD-consistency screen"]

INK, MUTED, HILITE = "#22303f", "#93a3b3", "#d1495b"
ACCENT, BAND = "#2a9d8f", "#fbeaec"


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


def save(fig, stem):
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", stem + ".pdf"))
    plt.close(fig)


def by_run(rows):
    """(platform, bipred, chi2_cap) -> {stage: row}, in first-seen order."""
    runs = {}
    for r in rows:
        key = (r["platform"], r["bipred"], r.get("chi2_cap", "both"))
        runs.setdefault(key, {})[r["stage"]] = r
    return runs


def label(run):
    platform, version, cap = run
    suffix = "" if cap == "both" else ", cap on LDSC only"
    return f"{platform}, bipred {version}{suffix}"


def fig_cancellation(rows):
    """The screen is what makes the fit admissible."""
    runs = by_run(rows)
    x = range(len(STAGES))
    fig, ax = plt.subplots(figsize=(6.6, 3.9))

    ax.axhspan(CANCELLATION_LIMIT, 1e4, color=BAND, zorder=0)
    ax.axhline(CANCELLATION_LIMIT, color=HILITE, lw=1.0, ls="--", zorder=2)
    ax.text(2.04, CANCELLATION_LIMIT * 1.9, "divergence limit", fontsize=7.5,
            color=HILITE, ha="right")

    # Every run is drawn, but only the first is labelled: the two platforms
    # agree closely enough here that four legend entries would describe two
    # visible lines. That they coincide is the point of the other figure.
    for run_index, stages in enumerate(runs.values()):
        for colour, trait, key, dy in ((INK, "LDL", "cancellation_ldl", 9),
                                       (ACCENT, "CAD", "cancellation_cad", -14)):
            y = [float(stages[s][key]) for s in STAGES]
            ax.plot(x, y, color=colour, lw=1.4, marker="o", ms=5, zorder=3,
                    label=trait if not run_index else None)
            if run_index:
                continue
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:,.2f}", (xi, yi), textcoords="offset points",
                            xytext=(0, dy), ha="center", fontsize=7.5,
                            color=colour, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 2e3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(STAGES)
    ax.set_ylabel("cancellation   sum(beta²) / h²   (log scale)")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left")
    return fig


def fig_rg(rows):
    """The rg reported before the screen is not an estimate of anything."""
    runs = by_run(rows)
    x = range(len(STAGES))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    diverged = [i for i, s in enumerate(STAGES)
                if any(st[s]["divergence_warned"] == "1"
                       for st in runs.values())]
    ax.axvspan(min(diverged) - 0.5, max(diverged) + 0.5, color=BAND, zorder=0)
    ax.text((min(diverged) + max(diverged)) / 2, 0.012,
            "fit diverged — not an estimate", fontsize=8, color=HILITE,
            ha="center")
    # The two runs coincide at the diverged stages, so label them once, muted:
    # these are not estimates and should not read as a series to be compared.
    for i in diverged:
        value = float(next(iter(runs.values()))[STAGES[i]]["rg"])
        ax.annotate(f"{value:+.4f}", (i, value), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7.5, color=MUTED)

    styles = ((INK, "o", "-"), (ACCENT, "s", "-"), (HILITE, "^", "--"))
    offsets = (5, -3, -13)
    for (colour, marker, dash), dy, (run, stages) in zip(styles, offsets,
                                                         runs.items()):
        y = [float(stages[s]["rg"]) for s in STAGES]
        ax.plot(x, y, dash, color=colour, lw=1.4, marker=marker, ms=5,
                zorder=3, label=label(run))
        ax.annotate(f"{y[-1]:+.4f}", (x[-1], y[-1]), fontsize=7.5,
                    color=colour, textcoords="offset points", xytext=(8, dy))
    ax.set_xticks(list(x))
    ax.set_xticklabels(STAGES)
    ax.set_xlim(-0.5, 2.55)
    ax.set_ylim(0, 0.36)
    ax.set_ylabel("bipred rg")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 1.0))
    return fig


def main():
    with open(os.path.join(HERE, "results", "estimates.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    style()
    save(fig_cancellation(rows), "cancellation")
    save(fig_rg(rows), "rg-by-stage")
    print(f"wrote 2 figures from {len(rows)} rows, "
          f"{len(by_run(rows))} runs")


if __name__ == "__main__":
    main()
