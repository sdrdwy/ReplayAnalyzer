import sys
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict


JUDGMENT_COLORS = {
    "PERFECT": "#00cc00",
    "GREAT": "#66ff66",
    "GOOD": "#00ccff",
    "OK": "#ffcc00",
    "MEH": "#ff8800",
    "MISS": "#ff2200",
}


def load_offsets(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    windows = data["hit_windows"]
    offsets_by_col = defaultdict(list)
    for o in data["offsets"]:
        col = o["column"]
        offsets_by_col[col].append(o["offset_ms"])

    return offsets_by_col, windows, data["metadata"], data


def draw_judgment_regions(ax, windows, ylim):
    judgment_order = ["MISS", "MEH", "OK", "GOOD", "GREAT", "PERFECT"]
    thresholds = [0, windows["meh"], windows["ok"], windows["good"],
                  windows["great"], windows["perfect"]]

    regions = []
    for i in range(len(thresholds) - 1):
        lo = thresholds[i + 1]
        hi = thresholds[i]
        if lo < hi:
            regions.append((lo, hi, judgment_order[i]))
            regions.append((-hi, -lo, judgment_order[i]))

    for lo, hi, name in regions:
        ax.axvspan(lo, hi, alpha=0.07, color=JUDGMENT_COLORS[name],
                   label=f"{name}" if f"{name}_pos" not in str(ax.get_children()) else "")


def draw_judgment_boundaries(ax, windows):
    colors = ["#ff2200", "#ff8800", "#ffcc00", "#00ccff", "#66ff66", "#00cc00"]
    labels = ["MISS", "MEH", "OK", "GOOD", "GREAT", "PERFECT"]
    vals = [windows["miss"], windows["meh"], windows["ok"],
            windows["good"], windows["great"], windows["perfect"]]
    for v, c, label in zip(vals, colors, labels):
        ax.axvline(x=v, color=c, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.axvline(x=-v, color=c, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.text(v + 1, ax.get_ylim()[1] * 0.95, label,
                fontsize=7, color=c, ha="left", va="top", rotation=90)
        ax.text(-v - 1, ax.get_ylim()[1] * 0.95, label,
                fontsize=7, color=c, ha="right", va="top", rotation=90)


def plot_offset_distribution(offsets_by_col, windows, meta, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 7))

    all_offsets = [v for vals in offsets_by_col.values() for v in vals]
    if not all_offsets:
        ax.text(0.5, 0.5, "No offset data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return

    max_abs = max(max(all_offsets), -min(all_offsets), windows["miss"] + 50)

    bins = 80
    colors_palette = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
        "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
        "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    ]

    sorted_cols = sorted(offsets_by_col.keys())
    for idx, col in enumerate(sorted_cols):
        vals = offsets_by_col[col]
        if not vals:
            continue
        color = colors_palette[idx % len(colors_palette)]
        ax.hist(vals, bins=bins, range=(-max_abs, max_abs),
                alpha=0.55, color=color, label=f"K{col} (n={len(vals)})",
                edgecolor=color, linewidth=0.3)

    draw_judgment_boundaries(ax, windows)

    ax.axvline(x=0, color="black", linewidth=1.2, linestyle="-", alpha=0.9)

    ax.set_xlabel("Offset (ms)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Offset Distribution by Column\n"
        f"{meta['player']}  —  {meta['beatmap']['artist']} - {meta['beatmap']['title']} "
        f"[{meta['beatmap']['version']}]  (CS={meta['cs']}, OD={meta['od']})",
        fontsize=12
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim(-max_abs, max_abs)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <offsets.json> [output.png]",
              file=sys.stderr)
        sys.exit(1)

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"Error: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) >= 3:
        save_path = sys.argv[2]
    else:
        save_path = json_path.replace(".json", "_distribution.png")

    offsets_by_col, windows, meta, _ = load_offsets(json_path)
    print(f"Loaded: {json_path}")
    print(f"Columns: {sorted(offsets_by_col.keys())}")
    for col, vals in sorted(offsets_by_col.items()):
        print(f"  K{col}: {len(vals)} offsets  "
              f"range=[{min(vals):.1f}, {max(vals):.1f}]")
    plot_offset_distribution(offsets_by_col, windows, meta, save_path)


if __name__ == "__main__":
    main()
