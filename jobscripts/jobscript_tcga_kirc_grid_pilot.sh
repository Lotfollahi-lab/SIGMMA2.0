#!/usr/bin/env bash
# OPTION-A PILOT (1 slide): tile-level tumour/stroma map.
# Replace per-nucleus seg with a UNIFORM 64um grid over tissue, then run the
# UNMODIFIED SIGMMA feature extractor on the grid points (full-square masks ->
# tile-level embeddings). Validates the grid->embedding plumbing end-to-end on
# TCGA-CJ-4637 before scaling to the 5 verify slides + RCC training slides.
# Forward-pass only -> inference queue.
#BSUB -J tcga_grid_pilot
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_grid_pilot.%J.out
#BSUB -e logs/job_logs/tcga_grid_pilot.%J.err
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

sample="TCGA-CJ-4637-01Z-00-DX1.5632396E-A880-4686-9E94-89E3F01AFBFB"
zarr="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/kirc_slides/81c38a4f-9d25-4ade-81b7-89b7658ad6d6/${sample}.he.zarr"

grid_data="$cr/tcga_kirc/grid64_data"
grid_feat="$cr/tcga_kirc/grid64_sigmma_36852_step2000"
grid_manifest="$run_dir/configs/tcga_kirc_grid_manifest_pilot.csv"
mkdir -p "$grid_data" "$grid_feat"

# ---- 1. build the 64um grid data_h5 (CPU, zarr) ----
echo "[1] build grid data_h5 (tile_um=64)"
"$POETRY" -u "$run_dir/scripts/tcga_tsr/tcga_build_grid_data.py" \
    --zarr "$zarr" --sample "$sample" \
    --out_h5 "$grid_data/${sample}.h5" \
    --tile_um 64 --um_per_px 0.5 --min_frac 0.5

# ---- 2. one-row grid manifest for the feature extractor ----
echo "[2] write grid manifest"
printf 'sample_name,hematoxylin_eosin_image,data_h5\n%s,%s,%s\n' \
    "$sample" "$zarr" "$grid_data/${sample}.h5" > "$grid_manifest"

# ---- 3. SIGMMA features on the grid (GPU forward pass) ----
echo "[3] extract SIGMMA tile embeddings"
"$POETRY" -u "$run_dir/scripts/tcga_tsr/tcga_extract_sigmma_features.py" \
    --config "$config" --manifest "$grid_manifest" --slide "$sample" \
    --out_dir "$grid_feat" --max_cells 0 --batch_size 64

echo "[4] summary"
"$POETRY" - "$grid_feat/${sample}.h5" <<'PY'
import sys, h5py, numpy as np
f=h5py.File(sys.argv[1],"r")
print("datasets:", list(f.keys()))
for k in f:
    try: print(f"  {k:24s} {f[k].shape} {f[k].dtype}")
    except Exception: pass
PY
echo "[grid-pilot] DONE"
