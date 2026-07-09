"""Phase 0b — build per-panel GEX-2D artifacts for the TRAIN-ONLY SVG reselection.

For each small panel (ovary/lung/brain) this writes
``outputs/gex_2d/svgTRAIN_panel_<HASH>/`` with:
  - gene_panel.json   (the TRAIN-ONLY SVG gene list from
                       svg_panels_trainonly_2026/<HASH>/panel_<HASH>/svg_top100.*)
  - MANIFEST_panel_<HASH>{,_train,_val,_test}.csv  (copied verbatim from the
                       published svg0529_panel_<HASH> dir — the section splits are
                       gene-independent, only the predicted gene set changed)

This keeps the section splits identical to the published benchmark while making the
target gene set leak-free (selected on train sections only).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
TRAINONLY_ROOT = ROOT / "outputs/diagnostics/svg_panels_trainonly_2026"
OUT_ROOT = ROOT / "outputs/gex_2d"

HASHES = ["1ca7fe57390a", "a87c22a4bcb9", "2df22e67cd31"]  # ovary / lung / brain


def main() -> None:
    for h in HASHES:
        src_panel = TRAINONLY_ROOT / h / f"panel_{h}"
        meta = json.loads((src_panel / "svg_top100.meta.json").read_text())
        genes = np.load(src_panel / "svg_top100.npy", allow_pickle=True).tolist()

        gene_panel = {
            "k": int(meta.get("K", len(genes))),
            "hvg_method": "panel_aware_svg_morans_i",
            "panel_id": meta.get("panel_id", f"panel_{h}"),
            "selection": "train_only",
            "n_sections_in_aggregate": meta.get("n_sections_in_aggregate"),
            "n_sections_in_panel": meta.get("n_sections_in_panel"),
            "merged_var_index_size": meta.get("merged_var_index_size"),
            "weight_by": meta.get("weight_by"),
            "min_detection_frac": meta.get("min_detection_frac"),
            "min_morans_i_floor": meta.get("min_morans_i_floor"),
            "min_morans_i_floor_frac": meta.get("min_morans_i_floor_frac"),
            "min_mean_log1p_expr": meta.get("min_mean_log1p_expr"),
            "source_npy": str(src_panel / "svg_top100.npy"),
            "source_meta": str(src_panel / "svg_top100.meta.json"),
            "panel": genes,
        }

        out_dir = OUT_ROOT / f"svgTRAIN_panel_{h}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gene_panel.json").write_text(json.dumps(gene_panel, indent=2))

        # Copy the gene-independent section-split manifests from the published panel.
        src_manifest_dir = OUT_ROOT / f"svg0529_panel_{h}"
        for tag in ("", "_train", "_val", "_test"):
            src = src_manifest_dir / f"MANIFEST_panel_{h}{tag}.csv"
            dst = out_dir / f"MANIFEST_panel_{h}{tag}.csv"
            shutil.copyfile(src, dst)

        print(f"{out_dir.name}: n_genes={len(genes)}  panel_id={gene_panel['panel_id']}  "
              f"(merged_var={gene_panel['merged_var_index_size']})")


if __name__ == "__main__":
    main()
