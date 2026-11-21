"""Plot RTX 4060 within-block throughput contrasts from paired_squares.csv."""

import argparse
import csv
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def default_csv():
    for parent in Path(__file__).resolve().parents:
        for relative in ("paired_squares.csv", "data/paired_squares.csv", "benchmarks/paired_squares.csv",
                         "release/benchmarks/paired_squares.csv"):
            candidate = parent / relative
            if candidate.is_file():
                return candidate
    return Path("paired_squares.csv")


def paired_contrasts(csv_path):
    blocks = defaultdict(dict)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["carry"], int(row["block"]))
            arm = row["arm"]
            if arm in blocks[key]:
                raise ValueError(f"Duplicate observation for {key} and {arm}")
            measurement = float(row["aggregate_us"])
            if not measurement > 0:
                raise ValueError(f"Expected a positive time for {key} and {arm}")
            blocks[key][arm] = (int(row["order"]), measurement)

    arms = {"stock", "center8", "center9"}
    values = []
    for carry in ("exact", "q44"):
        orders = []
        cohort = [blocks[carry, number] for number in range(1, 7)]
        for number, block in enumerate(cohort, 1):
            if set(block) != arms or {value[0] for value in block.values()} != {1, 2, 3}:
                raise ValueError(f"Incomplete crossover in {carry}, block {number}")
            orders.append(tuple(sorted(arms, key=lambda arm: block[arm][0])))
        if set(orders) != set(permutations(sorted(arms))):
            raise ValueError(f"The six orders are not balanced in cohort {carry}")
        for comparator in ("stock", "center9"):
            gains = [100 * (block[comparator][1] / block["center8"][1] - 1)
                     for block in cohort]
            values.append((carry, comparator, gains))
    return values


def plot(values, destination, png_path=None):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })
    fig = plt.figure(figsize=(153 / 25.4, 74 / 25.4), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[0.29, 0.605, 0.105],
                            left=0.013, right=0.989, bottom=0.255, top=0.825, wspace=0)
    labels, panel, summary = [fig.add_subplot(grid[i]) for i in range(3)]
    row_y = (3.10, 2.25, 0.95, 0.10)
    dots = (-0.19, 0.12, -0.09, 0.19, -0.15, 0.10)
    ink = "#162c43"
    medium = "#46596b"
    rule = "#d5dce2"
    for ax in (labels, panel, summary):
        ax.set_ylim(-0.42, 3.67)
        ax.get_yaxis().set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axhline(1.60, color=rule, lw=0.7, zorder=0)
    for ax in (labels, summary):
        ax.get_xaxis().set_visible(False)
        ax.set_xlim(0, 1)
    labels.text(0.00, 1.02, "Carry", transform=labels.transAxes,
                color=ink, weight="bold", va="bottom")
    labels.text(0.42, 1.02, "Comparator", transform=labels.transAxes,
                color=ink, weight="bold", va="bottom")
    summary.text(0.98, 1.02, "Median", color=ink, transform=summary.transAxes,
                 weight="bold", ha="right", va="bottom")
    panel.set_xlim(-21, 75)
    panel.set_xticks([-20, 0, 20, 40, 60])
    panel.tick_params(axis="x", colors=medium, length=2.4, pad=2, width=0.6)
    panel.spines["bottom"].set_visible(True)
    panel.spines["bottom"].set_color(medium)
    panel.spines["bottom"].set_linewidth(0.65)
    panel.grid(axis="x", lw=0.55, color=rule)
    panel.axvline(0, lw=0.9, color=medium)
    panel.set_xlabel("Eight-prime throughput change (%)", color=ink, labelpad=4)

    for y, (carry, comparator, gains) in zip(row_y, values):
        if comparator == "stock":
            labels.text(0.00, y, {"exact": "Exact", "q44": "Q44"}[carry],
                        va="center", color=ink, weight="bold")
        labels.text(0.42, y, {"stock": "Baseline 0.2", "center9": "Nine-prime"}[comparator],
                    va="center", color=ink)
        mid = median(gains)
        panel.scatter(gains, [y + offset for offset in dots], marker="o", s=17,
                      linewidth=0.9, edgecolor=medium, facecolor="white", zorder=3)
        panel.scatter(mid, y, marker="D", s=32, linewidth=0.35, edgecolor="white",
                      facecolor=ink, zorder=4)
        summary.text(0.98, y, f"{mid:+.2f}%".replace("-", "−"),
                     ha="right", va="center", color=ink, weight="bold")
    legend = [Line2D([], [], color="none", marker="o", markeredgecolor=medium,
                     markerfacecolor="white", markeredgewidth=0.9, markersize=4.4),
              Line2D([], [], color="none", marker="D", markeredgecolor=ink,
                     markerfacecolor=ink, markersize=4.2)]
    fig.legend(legend, ["Matched block", "Six-block median"],
               ncol=2, loc="upper center", bbox_to_anchor=(0.59, 1.005),
               borderaxespad=0.15, frameon=False, handletextpad=0.2, columnspacing=1,
               labelcolor=medium)
    fig.text(0.604, 0.025, "Positive values favor the eight-prime path", ha="center",
             color=medium, size=7.5)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, metadata={"Title": "RTX 4060 paired throughput contrasts"})
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=230)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_csv())
    parser.add_argument("--output", type=Path, default=Path("FigAblation.pdf"))
    parser.add_argument("--png", type=Path, help="Optional raster preview")
    args = parser.parse_args()
    values = paired_contrasts(args.input)
    plot(values, args.output, args.png)
    for carry, comparator, gains in values:
        print(f"{carry:>5s} vs {comparator:7s}   "
              f"median {median(gains):+6.2f}%   "
              + ", ".join(f"{g:+.2f}%" for g in gains))


if __name__ == "__main__":
    main()
