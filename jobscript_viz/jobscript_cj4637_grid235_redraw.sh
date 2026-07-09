#!/usr/bin/env bash
# Re-draw ONLY the tile-level region overlay for CJ-4637 on the 235 grid (the
# SIGMMA tile features are already cached from job 56173; the previous draw died
# on a float zarr-slice index, now fixed). CPU + zarr -> oversubscribed (<=6GB).
#BSUB -J cj4637_grid235_redraw
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/cj4637_grid235_redraw.%J.out
#BSUB -e logs/job_logs/cj4637_grid235_redraw.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
manifest="$run_dir/configs/tcga_kirc_manifest.csv"
bundle="$run_dir/outputs/cell_niche_rcc/tile_tumour_stroma_probe/z_sigmma_concat__tile_tso_binary.joblib"

s="TCGA-CJ-4637-01Z-00-DX1.5632396E-A880-4686-9E94-89E3F01AFBFB"
zarr=$(awk -F, -v s="$s" '$1==s{print $3}' "$manifest" | tr -d '\r')
grid_data="$cr/tcga_kirc/grid64_data_thr235"
grid_feat="$cr/tcga_kirc/grid64_sigmma_36852_step2000_thr235"

"$POETRY" -u "$run_dir/scripts/tcga_tsr/tcga_grid_overlay.py" \
    --grid_data "$grid_data/$s.h5" --grid_feat "$grid_feat/$s.h5" \
    --he_zarr "$zarr" --bundle "$bundle" --sample "$s" \
    --fig_path "$run_dir/figures/clinical_outcome/tso_kirc/grid_region_overlay_thr235/${s%%.*}" \
    --field_um 1860 --um_per_px 0.5 --border_frac 0.25 --min_tumour 10 \
    --measure_frac 0.5 --reject_thr 0.75

echo "[cj4637-grid235-redraw] DONE"
