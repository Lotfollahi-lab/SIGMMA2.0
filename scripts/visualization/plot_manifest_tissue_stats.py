import argparse
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_MANIFEST = Path(
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST.csv"
)
DEFAULT_METADATA = Path(
    "/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sample_metadata.csv"
)
DEFAULT_OUTPUT = Path("outputs/data_overview/sections_per_tissue.png")

# STANDING RULE (user-confirmed 2026-06-06): map section->tissue from
# sample_metadata.csv (join on sample_id == manifest sample_name), never by
# parsing the image path / run folder. sample_metadata already encodes the two
# pancreas cohorts as distinct tissue_types (`pancreas` = diabetic DJ* samples;
# `fetal_pancreas` = the fetal cohort, previously mislabelled "pediatric").
TISSUE_TYPE_LABEL = {
    "beacon":                   "Skin",
    "CTCL":                     "Ctcl",
    "pediatric_ovary":          "Pediatric Ovary",
    "renal_cell_carcinoma":     "Renal Cell Carcinoma",
    "liver":                    "Liver",
    "pancreas":                 "Diabetic Pancreas",
    "fetal_pancreas":           "Fetal Pancreas",
    "adrenocortical_carcinoma": "Adrenocortical Carcinoma",
    "whole_embryo":             "Whole Embryo",
    "glioblastoma":             "Glioblastoma",
    "lung":                     "Lung",
    # External HEST-1k public panels pooled into one bar.
    "IDC":                      "Public Xenium",
    "PAAD":                     "Public Xenium",
    "SKCM":                     "Public Xenium",
    "LUAD":                     "Public Xenium",
}

# Nature-style typography + light spine/tick weights. Helvetica/Arial are not
# installed in this venv; Nimbus Sans is the Helvetica-metric-compatible
# substitute that ships with most GhostScript installs.
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.dpi": 600,
}

# Paul Tol's "muted" qualitative palette (Tol 2021) — colorblind-safe and
# lower-contrast than NPG. Used as a stable per-tissue identity so the same
# color carries across multi-panel figures.
TISSUE_PALETTE = {
    "Skin":                 "#CC6677",
    "Pediatric Ovary":      "#88CCEE",
    "Renal Cell Carcinoma": "#44AA99",
    # Pancreas is two distinct cohorts; keep them adjacent shades of the same hue.
    "Diabetic Pancreas":    "#332288",
    "Fetal Pancreas":       "#6F63B5",
    "Liver":                "#DDCC77",
    "Whole Embryo":         "#999933",
    "Glioblastoma":         "#AA4499",
    "Lung":                 "#882255",
}
PALETTE_FALLBACK = "#DDDDDD"
TEXT_COLOR = "#222222"


def _format_cells(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _cells_in_section(spatial_preprocessed_path: str) -> int:
    # Cells live in the processed AnnData h5; obsm/spatial is (n_cells, 2). Read
    # only the dataset shape so we never pull the matrix into memory.
    with h5py.File(spatial_preprocessed_path, "r") as f:
        return int(f["obsm/spatial"].shape[0])


def aggregate(manifest_path: Path, metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    # Authoritative section->tissue map; join on sample_id == manifest sample_name.
    # A sample run in two Xenium runs (e.g. Sml4-PNC-0-FFPE-1-s4 in SGP252 RUN1
    # & RUN2) appears twice in the metadata; dedupe so the left join stays 1:1
    # and does not double-count those manifest sections.
    meta = pd.read_csv(metadata_path)[["sample_id", "tissue_type"]].drop_duplicates("sample_id")
    df = df.merge(meta, left_on="sample_name", right_on="sample_id", how="left")

    missing = df["tissue_type"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} manifest sections have no sample_metadata row, "
            f"e.g. {df.loc[missing, 'sample_name'].head().tolist()}"
        )
    unknown = sorted(set(df["tissue_type"]) - set(TISSUE_TYPE_LABEL))
    if unknown:
        raise ValueError(f"Unmapped tissue_type values in TISSUE_TYPE_LABEL: {unknown}")

    df["tissue"] = df["tissue_type"].map(TISSUE_TYPE_LABEL)
    df["n_cells"] = df["spatial_preprocessed"].map(_cells_in_section)
    stats = (
        df.groupby("tissue")
        .agg(n_sections=("sample_name", "count"), n_cells=("n_cells", "sum"))
        .sort_values("n_sections", ascending=False)
        .reset_index()
    )
    return stats


def plot(stats: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Largest at top for horizontal bars.
    stats = stats.iloc[::-1].reset_index(drop=True)

    with mpl.rc_context(NATURE_RC):
        # Single-column Nature width ≈ 89 mm = 3.5". Heights scaled to leave
        # room for 8 tick rows without crowding.
        fig, ax = plt.subplots(figsize=(6, 10))
        y = np.arange(len(stats))
        colors = [TISSUE_PALETTE.get(t, PALETTE_FALLBACK) for t in stats["tissue"]]
        bars = ax.barh(
            y,
            stats["n_sections"],
            color=colors,
            edgecolor="none",
            height=0.7,
        )

        xmax = stats["n_sections"].max()
        ax.set_xlim(0, xmax * 1.28)
        for bar, n_cells in zip(bars, stats["n_cells"]):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"  {_format_cells(int(n_cells))} cells",
                ha="left",
                va="center",
                color=TEXT_COLOR,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(stats["tissue"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Number of sections")
        ax.margins(y=0.04)

        fig.tight_layout()
        fig.savefig(output_path)
        # Also drop a vector copy alongside the PNG for figure assembly.
        fig.savefig(output_path.with_suffix(".pdf"))
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    stats = aggregate(args.manifest, args.metadata)
    plot(stats, args.output)
    csv_path = args.output.with_name("tissue_stats.csv")
    stats.to_csv(csv_path, index=False)
    print(stats.to_string(index=False))
    print(f"\nSaved figure: {args.output}")
    print(f"Saved stats:  {csv_path}")


if __name__ == "__main__":
    main()
