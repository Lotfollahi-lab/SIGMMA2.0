"""Spatial-expression heat-map grid for the top SVGs of a panel.

For each selected section, re-bin the parquet to a tile×gene sparse matrix
on the section's panel, then for each top gene render the 2D log-expression
heat-map. Rows = sections, cols = genes.

Saves PNG to `figures/diagnostics/svg_verification/<panel_id>.png`.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

from sigmma.evaluation.transcript_binning import manifest_path_to_transcripts_parquet
from sigmma.evaluation.svg_selection import bin_section_expression, tissue_of


def _read_var_index(h5_path):
    with h5py.File(h5_path, "r") as f:
        var_grp = f["var"]
        idx_attr = var_grp.attrs.get("_index", "_index")
        if isinstance(idx_attr, bytes):
            idx_attr = idx_attr.decode()
        if idx_attr not in var_grp:
            for k in ("gene_name", "feature_name", "gene_symbol", "_index"):
                if k in var_grp:
                    idx_attr = k
                    break
        raw = var_grp[idx_attr][:]
    return [g.decode() if isinstance(g, (bytes, bytearray)) else str(g) for g in raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_id", default="panel_2d22ba31d287")
    ap.add_argument("--svg_panels_root",
                    default="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/svg_panels")
    ap.add_argument("--train_manifest",
                    default="/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train.csv")
    ap.add_argument("--n_top_genes", type=int, default=8)
    ap.add_argument("--sections", nargs="*", default=None,
                    help="Section IDs to plot. If omitted, picks one per tissue.")
    ap.add_argument("--tissues", nargs="*",
                    default=["beacon", "renal_cell_carcinoma", "liver", "whole_embryo"])
    ap.add_argument("--bin_size_px", type=int, default=256)
    ap.add_argument("--out", default="figures/diagnostics/svg_verification/")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load panel index + select sections
    import json
    panel_index = json.load(open(Path(args.svg_panels_root) / "panel_index.json"))
    panel_sids = [sid for sid, pid in panel_index.items() if pid == args.panel_id]

    if args.sections:
        sections = list(args.sections)
    else:
        # One per tissue
        sections = []
        for t in args.tissues:
            for sid in panel_sids:
                if tissue_of(sid) == t:
                    sections.append(sid)
                    break
    print(f"Sections: {sections}", flush=True)

    # Load top genes
    top_genes = np.load(Path(args.svg_panels_root) / args.panel_id / "svg_top100.npy",
                        allow_pickle=True)
    genes = list(top_genes[: args.n_top_genes])
    print(f"Genes: {genes}", flush=True)

    # Load manifest
    m = pd.read_csv(args.train_manifest)
    m = m.set_index("sample_name")

    # Cache binned matrices per section
    section_data = {}
    for sid in sections:
        if sid not in m.index:
            print(f"  [warn] {sid} not in manifest, skipping", flush=True)
            continue
        row = m.loc[sid]
        spatial = row["spatial_preprocessed"]
        he = row["hematoxylin_eosin_image"]
        parquet = manifest_path_to_transcripts_parquet(he)
        if parquet is None:
            print(f"  [warn] {sid} parquet unresolved, skipping", flush=True)
            continue
        var_index = _read_var_index(spatial)
        print(f"  binning {sid} ({len(var_index)} genes)...", flush=True)
        expr_sparse, bin_coords, grid_shape = bin_section_expression(
            parquet, var_index, bin_size_px=args.bin_size_px,
            qv_min=20.0, min_bin_transcripts=5,
        )
        # Gene name -> column index for this section's var_index
        gene_to_col = {g: i for i, g in enumerate(var_index)}
        # Dense (H, W) per gene
        per_gene_maps = {}
        H, W = grid_shape
        for g in genes:
            if g not in gene_to_col:
                per_gene_maps[g] = None
                continue
            col = gene_to_col[g]
            vals = expr_sparse[:, col].toarray().ravel()
            grid = np.zeros((H, W), dtype=np.float32)
            grid[bin_coords[:, 0], bin_coords[:, 1]] = vals
            # Log1p for display
            per_gene_maps[g] = np.log1p(grid)
        section_data[sid] = {
            "grid_shape": grid_shape,
            "bin_coords": bin_coords,
            "per_gene_maps": per_gene_maps,
            "tissue": tissue_of(sid) or "UNK",
        }

    if not section_data:
        print("No sections processed.", flush=True)
        return

    # ===== Plot =====
    n_rows = len(section_data)
    n_cols = len(genes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows),
                             squeeze=False)
    for i, (sid, d) in enumerate(section_data.items()):
        for j, g in enumerate(genes):
            ax = axes[i, j]
            m = d["per_gene_maps"].get(g)
            if m is None:
                ax.text(0.5, 0.5, "absent", ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                vmax = np.percentile(m[m > 0], 99) if (m > 0).any() else 1.0
                ax.imshow(m, cmap="magma", vmin=0, vmax=max(vmax, 0.01),
                          interpolation="nearest", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(g, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{d['tissue']}\n{sid[:18]}", fontsize=8)

    fig.suptitle(f"Top {n_cols} SVGs per section — panel {args.panel_id}",
                 fontsize=11)
    fig.tight_layout()
    out_path = out_dir / f"{args.panel_id}_top{n_cols}_svgs.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[saved] {out_path}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
