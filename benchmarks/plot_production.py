#!/usr/bin/env python3
"""Plot hourly per-device completions from the adjacent production_windows.csv."""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DEVICES = (
    ("RTX 4060 Ti", "#254A58", ""),
    ("RTX 4060", "#79AAA4", "///"),
    ("RTX 3070", "#C5D2D7", ".."),
    ("RTX 2080 Ti", "#FAFAF7", "\\\\\\"),
)
METRICS = (("main_searches", "Main searches"), ("certificates", "Certificates"))


def hourly_rows(path):
    """Return a time-ordered set of full hours, checking fleet sums and continuity."""
    grouped = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            start = datetime.fromisoformat(row["start_utc"])
            end = datetime.fromisoformat(row["end_utc"])
            if end - start != timedelta(hours=1):
                raise ValueError(f"Unexpected window duration at {start}")
            bucket = grouped.setdefault(start, {})
            if row["device"] in bucket:
                raise ValueError(f"Duplicate {row['device']} observation at {start}")
            bucket[row["device"]] = {key: int(row[key]) for key, _ in METRICS}
    starts = sorted(grouped)
    if not starts or any(b-a != timedelta(hours=1) for a, b in zip(starts, starts[1:])):
        raise ValueError("A continuous series of hourly observations is required")
    for start, rows in grouped.items():
        if set(rows) != {"fleet", *(d[0] for d in DEVICES)}:
            raise ValueError(f"Incomplete set of device counts at {start}")
        for key, _ in METRICS:
            if rows["fleet"][key] != sum(rows[d[0]][key] for d in DEVICES):
                raise ValueError(f"Per-device {key} counts do not sum to fleet at {start}")
    return starts, grouped


def render(csv_path, destination):
    starts, rows = hourly_rows(csv_path)
    destination.mkdir(parents=True, exist_ok=True)
    ink = "#172A36"
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.edgecolor": "#879196",
        "axes.linewidth": 0.6,
        "text.color": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "axes.labelcolor": ink,
        "hatch.linewidth": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "ps.useafm": False,
    })
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(128/25.4, 101/25.4))
    fig.subplots_adjust(left=0.137, right=0.995, top=0.83, bottom=0.14, hspace=0.49)
    positions = list(range(len(starts)))
    highest = max(rows[t]["fleet"][metric] for t in starts for metric, _ in METRICS)
    ceiling = ((highest + 49) // 50) * 50
    for panel, ax, (metric, label) in zip("ab", axes, METRICS):
        bottom = [0] * len(starts)
        for device, color, hatch in DEVICES:
            counts = [rows[t][device][metric] for t in starts]
            ax.bar(positions, counts, bottom=bottom, width=0.69, color=color,
                   edgecolor=ink, linewidth=0.47, hatch=hatch, zorder=2)
            bottom = [a+b for a,b in zip(bottom, counts)]
        for i, total in enumerate(bottom):
            ax.annotate(str(total), (i,total), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
        ax.text(0, 1.025, f"({panel})  {label}", transform=ax.transAxes,
                ha="left", va="bottom", weight="bold", fontsize=9)
        ax.set_ylim(0, ceiling)
        ax.set_xlim(-0.58, len(starts)-0.42)
        ax.set_yticks(range(0, ceiling, 100))
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.tick_params(axis="both", length=0, pad=3, labelsize=8.1)
        ax.grid(axis="y", color="#CFD6D8", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[1].set_xticks(positions, [t.strftime("%H:%M") for t in starts])
    axes[1].set_xlabel("Start of one-hour interval (UTC)", labelpad=5.5)
    fig.supylabel("Local completions", x=0.017, y=0.49, fontsize=9)
    handles = [Patch(facecolor=c, edgecolor=ink, linewidth=0.5, hatch=h, label=d)
               for d,c,h in DEVICES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.993),
               frameon=False, ncol=4, columnspacing=0.85, handlelength=1.15,
               handletextpad=0.35, fontsize=8.15)
    for ext in ("pdf", "eps"):
        fig.savefig(destination / f"production.{ext}", format=ext)
    plt.close(fig)


if __name__ == "__main__":
    from_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=from_dir/"production_windows.csv")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    render(args.csv, args.output_dir)
