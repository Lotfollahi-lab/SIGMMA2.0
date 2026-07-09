"""Re-run panel aggregation with new filter thresholds using existing
per-section diagnostic npz files (saves the expensive binning + Moran's I
recompute). Updates svg_topK.npy/csv/meta.json + rejected_genes.csv in place.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sigmma.evaluation.svg_selection import (
    MergedPanel, aggregate_panel_svgs, merge_small_panels, tissue_of,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg_panels_root",
                    default="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/svg_panels")
    ap.add_argument("--panel_groups_json",
                    default="outputs/diagnostics/panel_inventory/panel_groups.json")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--min_sections_per_panel", type=int, default=5)
    ap.add_argument("--jaccard_threshold", type=float, default=0.90)
    ap.add_argument("--min_overlap_threshold", type=float, default=0.95)
    ap.add_argument("--weight_by", default="n_cells")
    ap.add_argument("--min_detection_frac", type=float, default=0.7)
    ap.add_argument("--min_morans_i_floor", type=float, default=0.05)
    ap.add_argument("--min_morans_i_floor_frac", type=float, default=0.5)
    ap.add_argument("--min_mean_log1p_expr", type=float, default=0.0001)
    args = ap.parse_args()

    root = Path(args.svg_panels_root)
    diag_root = root / "_diag"
    panel_groups = json.load(open(args.panel_groups_json))
    merged_panels = merge_small_panels(
        panel_groups,
        min_sections=args.min_sections_per_panel,
        jaccard_threshold=args.jaccard_threshold,
        min_overlap_threshold=args.min_overlap_threshold,
    )

    summary = []
    for panel in merged_panels:
        if len(panel.merged_var_index) == 0:
            continue
        per_section = []
        for sid in panel.sample_names:
            npz_path = None
            for h in panel.merged_from_hashes:
                p = diag_root / h / f"{sid}.npz"
                if p.exists():
                    npz_path = p
                    break
            if npz_path is None:
                continue
            z = np.load(npz_path, allow_pickle=True)
            per_section.append({
                "sid": sid,
                "tissue": str(z["tissue"]),
                "var_index": list(z["var_index"]),
                "morans_i": z["morans_i"],
                "bins_present": z["bins_present"],
                "mean_log1p_expression": z["mean_log1p_expression"],
                "n_cells_used": int(z["n_transcripts_used"]),
                "n_bins_used": int(z["n_bins_used"]),
            })
        if not per_section:
            continue
        top_genes, top_df, rejected_df = aggregate_panel_svgs(
            panel.merged_var_index, per_section,
            min_detection_frac=args.min_detection_frac,
            min_morans_i_floor=args.min_morans_i_floor,
            min_morans_i_floor_frac=args.min_morans_i_floor_frac,
            min_mean_log1p_expr=args.min_mean_log1p_expr,
            k=args.k, weight_by=args.weight_by,
        )
        panel_dir = root / panel.panel_id
        panel_dir.mkdir(parents=True, exist_ok=True)
        np.save(panel_dir / f"svg_top{args.k}.npy",
                np.array(top_genes, dtype=object), allow_pickle=True)
        top_df.to_csv(panel_dir / f"svg_top{args.k}.csv", index=False)
        rejected_df.to_csv(panel_dir / "rejected_genes.csv", index=False)

        # Update meta.json (overwrite)
        meta = {
            "panel_id": panel.panel_id,
            "merged_from_hashes": panel.merged_from_hashes,
            "n_sections_in_aggregate": len(per_section),
            "K": args.k,
            "weight_by": args.weight_by,
            "min_detection_frac": args.min_detection_frac,
            "min_morans_i_floor": args.min_morans_i_floor,
            "min_morans_i_floor_frac": args.min_morans_i_floor_frac,
            "min_mean_log1p_expr": args.min_mean_log1p_expr,
            "merged_var_index_size": len(panel.merged_var_index),
        }
        with open(panel_dir / f"svg_top{args.k}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        median_I = float(top_df["weighted_mean_I"].median()) if len(top_df) else float("nan")
        top1 = top_genes[0] if top_genes else None
        summary.append({
            "panel_id": panel.panel_id,
            "n_sections": len(per_section),
            "n_passed": len(top_df),
            "top1": top1,
            "median_top_K_I": median_I,
        })
        print(f"  {panel.panel_id:<45s}  n_sec={len(per_section):>3d}  "
              f"n_passed={len(top_df):>3d}  top1={top1!s:<12s}  "
              f"median_I={median_I:.4f}", flush=True)

    pd.DataFrame(summary).to_csv(root / "_panel_summary.csv", index=False)
    print(f"\n[done] aggregations re-run, summary at {root}/_panel_summary.csv")


if __name__ == "__main__":
    main()
