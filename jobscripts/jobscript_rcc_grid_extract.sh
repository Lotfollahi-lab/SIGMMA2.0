#!/usr/bin/env bash
# OPTION-A: build the 64um tile GRID + SIGMMA tile embeddings for the RCC
# train+test sections, so a tile-level tumour/stroma head can be trained on the
# SAME grid-embedding space that TCGA inference uses (full-square pooling). The
# per-cell level_2_cell_type labels (already in the RCC data_h5) are aggregated
# to tiles later by train_tile_tumour_stroma.py. Forward-pass -> inference queue.
#BSUB -J rcc_grid_extract
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/rcc_grid_extract.%J.out
#BSUB -e logs/job_logs/rcc_grid_extract.%J.err
#BSUB -q inference
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=24GB:num=1"
#BSUB -n 4
#BSUB -M 48GB
#BSUB -R "select[mem>48GB] rusage[mem=48GB] span[hosts=1]"
set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
config="$run_dir/configs/cell_niche_transfer_rcc_celltype_a01_step2000.yaml"
POETRY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

grid_data="$cr/grid64/data/cell_niche_rcc"
grid_feat="$cr/grid64/sigmma/cell_niche_rcc"
grid_manifest="$run_dir/configs/rcc_grid64_manifest.csv"
mkdir -p "$grid_data" "$grid_feat"

# ---- 1. build a grid manifest (train+test) + grid data_h5 per section ----
echo "[1] build grid data_h5 for RCC train+test sections"
"$POETRY" - "$grid_data" "$grid_manifest" <<'PY'
import sys, subprocess, pandas as pd
from pathlib import Path
grid_data, grid_manifest = sys.argv[1], sys.argv[2]
m = pd.read_csv("configs/cell_niche_evaluable_slides_rcc.csv")
m = m[m.split.isin(["train", "test"])]
rows = []
for _, r in m.iterrows():
    s = r["sample_name"]; z = r["hematoxylin_eosin_image"]
    out = f"{grid_data}/{s}.h5"
    if not Path(out).exists():
        subprocess.run([sys.executable,
            "scripts/tcga_tsr/tcga_build_grid_data.py",
            "--zarr", z, "--sample", s, "--out_h5", out,
            "--tile_um", "64", "--um_per_px", "0.5", "--min_frac", "0.5"],
            check=True)
    else:
        print(f"[skip-grid] {s} exists", flush=True)
    rows.append({"sample_name": s, "hematoxylin_eosin_image": z, "data_h5": out})
pd.DataFrame(rows).to_csv(grid_manifest, index=False, lineterminator="\n")
print(f"[manifest] {len(rows)} sections -> {grid_manifest}", flush=True)
PY

# ---- 2. SIGMMA tile embeddings over the whole grid manifest (skip-existing) ----
echo "[2] extract SIGMMA tile embeddings (RCC grid)"
"$POETRY" -u "$run_dir/scripts/tcga_tsr/tcga_extract_sigmma_features.py" \
    --config "$config" --manifest "$grid_manifest" \
    --out_dir "$grid_feat" --max_cells 0 --batch_size 64

echo "[rcc-grid-extract] DONE"
