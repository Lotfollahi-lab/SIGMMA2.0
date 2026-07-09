"""Figure: panel_2d22ba31 cohort composition — section counts by tissue × split.

Stacked bars across 5 tissues × 3 splits (train/val/test). Numbers reflect the
panel-restricted manifests:
  - MANIFEST_panel_2d22ba31_train.csv (118 sections; 6 SKI silent-skip → 112 eff.)
  - MANIFEST_panel_2d22ba31_val.csv   (15)
  - MANIFEST_panel_2d22ba31_test.csv  (14)
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from sigmma.utils.tissue_map import tissue_of, GROUP_OF, PUBLIC_TT

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
PANEL_DIR = ROOT / "outputs/gex_2d/svg_panel_2d22ba31"
OUT_PNG = ROOT / "figures/gex_2d/svg_panel_2d22ba31_cohort_stats.png"
OUT_PDF = ROOT / "figures/gex_2d/svg_panel_2d22ba31_cohort_stats.pdf"

# Display groups present in the panel_2d22ba31 cohort (keyed by GROUP_OF;
# skin = beacon + CTCL merged). Glioblastoma is the old "BRA" bucket.
TISSUES = [
    ("glioblastoma", "Glioblastoma"),
    ("renal_cell_carcinoma", "Kidney"),
    ("liver", "Liver"),
    ("skin", "Skin"),
    ("whole_embryo", "Whole embryo"),
]

SPLITS = [
    ("train", "Train", "#3a6ea5"),
    ("val",   "Val",   "#cca96b"),
    ("test",  "Test",  "#a64041"),
]


def group_of(section_name: str) -> str | None:
    """Authoritative display group; None for public/unknown sections."""
    t = tissue_of(section_name)
    if t in PUBLIC_TT:
        return None
    return GROUP_OF.get(t)


def load_manifest_tissues(path: Path) -> list[str]:
    df = pd.read_csv(path)
    sec = df["sample_name"].astype(str)
    return [group_of(s) for s in sec]


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    counts = {}  # counts[tcode][split] = n
    for tcode, _ in TISSUES:
        counts[tcode] = {s[0]: 0 for s in SPLITS}
    for split_key, _, _ in SPLITS:
        manifest = PANEL_DIR / f"MANIFEST_panel_2d22ba31_{split_key}.csv"
        tissues = load_manifest_tissues(manifest)
        for t in tissues:
            if t in counts:
                counts[t][split_key] += 1

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(TISSUES))
    bottom = np.zeros(len(TISSUES))
    for split_key, split_label, color in SPLITS:
        vals = np.array([counts[t][split_key] for t, _ in TISSUES])
        bars = ax.bar(x, vals, bottom=bottom, color=color,
                      edgecolor="black", linewidth=0.4,
                      label=f"{split_label} (n={int(vals.sum())})")
        # In-bar count labels for non-trivial segments
        for xi, v, b in zip(x, vals, bottom):
            if v >= 2:
                ax.text(xi, b + v / 2, f"{int(v)}",
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += vals

    # Total per tissue above stack
    totals = np.array([sum(counts[t].values()) for t, _ in TISSUES])
    for xi, total in zip(x, totals):
        ax.text(xi, total + max(totals) * 0.02, f"{int(total)}",
                ha="center", va="bottom", fontsize=7, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _code, lbl in TISSUES])
    ax.set_ylabel("Sections (count)")
    ax.set_title("panel_2d22ba31 cohort composition (147 sections, 4952 genes)",
                 fontsize=9)
    ax.set_ylim(0, max(totals) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.3, color="#cccccc",
            alpha=0.7, zorder=-1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.text(
        0.5, 0.02,
        "panel_2d22ba31d287 (in-house Sanger Xenium). "
        "Train effective n = 112 after parquet-resolver silent-skip on 6 SKI.",
        ha="center", fontsize=6.5, color="black",
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF)
    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_PDF}")

    # Console table
    print("\nSection counts per tissue × split:")
    print(f"{'tissue':<8} {'train':>7} {'val':>5} {'test':>5} {'total':>7}")
    for tcode, _ in TISSUES:
        row = counts[tcode]
        tot = sum(row.values())
        print(f"{tcode:<8} {row['train']:>7} {row['val']:>5} {row['test']:>5} {tot:>7}")
    print(f"{'TOTAL':<8} "
          f"{sum(counts[t]['train'] for t,_ in TISSUES):>7} "
          f"{sum(counts[t]['val']   for t,_ in TISSUES):>5} "
          f"{sum(counts[t]['test']  for t,_ in TISSUES):>5} "
          f"{sum(sum(counts[t].values()) for t,_ in TISSUES):>7}")


if __name__ == "__main__":
    main()
