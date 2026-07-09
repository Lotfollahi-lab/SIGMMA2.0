#!/usr/bin/env bash
# CJ-4637 tile-level cell-type prediction + region overlay on the FULL-COVERAGE
# (white_thr=235) 64um grid. Extract SIGMMA tile embeddings on the rebuilt grid
# (GPU forward pass), then predict tumour/stroma/other per tile and draw the
# region-used overlay in the region_overlays_central05_full style.
# Forward-pass only -> inference queue.
#BSUB -J cj4637_grid235_overlay
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/cj4637_grid235_overlay.%J.out
#BSUB -e logs/job_logs/cj4637_grid235_overlay.%J.err
#BSUB -q inference
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=24GB:num=1"
#BSUB -n 4
#BSUB -M 48GB
#BSUB -R "select[mem>48GB] rusage[mem=48GB] span[hosts=1]"
set -euo pipefail
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
config="$run_dir/configs/cell_niche_transfer_rcc_celltype_a01_step2000.yaml"
POETRY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"
manifest="$run_dir/configs/tcga_kirc_manifest.csv"
bundle="$run_dir/outputs/cell_niche_rcc/tile_tumour_stroma_probe/z_sigmma_concat__tile_tso_binary.joblib"

s="TCGA-CJ-4637-01Z-00-DX1.5632396E-A880-4686-9E94-89E3F01AFBFB"
zarr=$(awk -F, -v s="$s" '$1==s{print $3}' "$manifest" | tr -d '\r')

grid_data="$cr/tcga_kirc/grid64_data_thr235"                 # full-coverage grid
grid_feat="$cr/tcga_kirc/grid64_sigmma_36852_step2000_thr235"
grid_manifest="$run_dir/configs/tcga_kirc_grid_manifest_cj4637_thr235.csv"
mkdir -p "$grid_feat"

# ---- 1. one-row grid manifest pointing at the 235 grid data_h5 ----
printf 'sample_name,hematoxylin_eosin_image,data_h5\n%s,%s,%s\n' \
    "$s" "$zarr" "$grid_data/$s.h5" > "$grid_manifest"

# ---- 2. SIGMMA tile embeddings on the 235 grid (GPU) ----
echo "[1] extract SIGMMA tile embeddings (235 grid)"
"$POETRY" -u "$run_dir/scripts/tcga_tsr/tcga_extract_sigmma_features.py" \
    --config "$config" --manifest "$grid_manifest" --slide "$s" \
    --out_dir "$grid_feat" --max_cells 0 --batch_size 64

# ---- 3. predict tiles + draw region overlay (tile-level) ----
echo "[2] predict tiles + draw region overlay"
"$POETRY" -u "$run_dir/scripts/tcga_tsr/tcga_grid_overlay.py" \
    --grid_data "$grid_data/$s.h5" --grid_feat "$grid_feat/$s.h5" \
    --he_zarr "$zarr" --bundle "$bundle" --sample "$s" \
    --fig_path "$run_dir/figures/clinical_outcome/tso_kirc/grid_region_overlay_thr235/${s%%.*}" \
    --field_um 1860 --um_per_px 0.5 --border_frac 0.25 --min_tumour 10 \
    --measure_frac 0.5 --reject_thr 0.75

echo "[cj4637-grid235-overlay] DONE"
