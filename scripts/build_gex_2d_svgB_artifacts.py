"""Phase 1 prep — build svgB panel dirs for the strict shared-gene zero-shot (B).

For each unseen panel, write ``outputs/gex_2d/svgB_panel_<HASH>/`` with:
  - gene_panel.json   panel = the SHARED genes (main-94 ∩ measured var.index),
                      in main-panel order (matches the head's output slicing).
  - MANIFEST_panel_<HASH>_eval.csv   the evaluation sections (test split; TENX-only
                      for the public-mixed panels), copied from the published panel.

The svgB data/target cache (built next) bins exactly these shared genes from the
eval sections' transcripts.parquet, so the frozen main head can be scored on them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/nfs/team361/dj16/projects/SIGMMA")
MVI = ROOT / "outputs/gex_2d/measured_var_index"
OUT_ROOT = ROOT / "outputs/gex_2d"

# panel_hash -> eval-section name prefix filter (None = all test sections)
PANELS = {
    "1ca7fe57390a": None,    # ovary
    "a87c22a4bcb9": None,    # lung
    "2df22e67cd31": None,    # brain
    "11bbd69265ed": "TENX",  # PAAD/LUAD public
    "80b4f72e3da8": "TENX",  # SKCM public
}


def main() -> None:
    for h, prefix in PANELS.items():
        mvi = json.loads((MVI / f"{h}.json").read_text())
        shared = mvi["shared"]  # main-panel order

        gene_panel = {
            "k": len(shared),
            "hvg_method": "panel_aware_svg_morans_i",
            "panel_id": f"svgB_panel_{h}",
            "selection": "zeroshot_shared_measured_varindex",
            "source": str(MVI / f"{h}.json"),
            "n_shared": len(shared),
            "panel": shared,
        }
        out_dir = OUT_ROOT / f"svgB_panel_{h}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "gene_panel.json").write_text(json.dumps(gene_panel, indent=2))

        # Eval manifest: test split filtered (TENX-only for public-mixed panels).
        src_test = OUT_ROOT / f"svg0529_panel_{h}" / f"MANIFEST_panel_{h}_test.csv"
        rows = list(csv.DictReader(open(src_test)))
        header = rows[0].keys()
        if prefix:
            rows = [r for r in rows if r["sample_name"].startswith(prefix)]
        with open(out_dir / f"MANIFEST_panel_{h}_eval.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(header))
            w.writeheader()
            w.writerows(rows)

        print(f"svgB_panel_{h}: n_shared_genes={len(shared)}  n_eval_sections={len(rows)}")


if __name__ == "__main__":
    main()
