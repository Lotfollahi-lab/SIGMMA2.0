#!/bin/bash
#BSUB -J tcga_coad_voronoi
#BSUB -G team361
#BSUB -q normal
#BSUB -n 1
#BSUB -M 16000
#BSUB -R "select[mem>16000] rusage[mem=16000]"
#BSUB -o logs/job_logs/tcga_coad_voronoi.%J.out
#BSUB -e logs/job_logs/tcga_coad_voronoi.%J.err

# Area-based intratumoural TSR (nearest-nucleus Voronoi within tissue mask),
# clinical van Pelt / Mesker convention. Self-contained (farm22 env passing
# unreliable). Survival is run separately after this with --stroma_cutoff 0.5.
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1

PY="$run_dir/.venv/bin/python"
feat_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_coad/sigmma_36852_step2000"
per_cell_dir="$run_dir/outputs/tcga_coad_tsr/per_cell_pred"
manifest="$run_dir/configs/tcga_coad_manifest.csv"
out_dir="$run_dir/outputs/tcga_coad_tsr_voronoi"
mkdir -p "$out_dir"

"$PY" -u scripts/tcga_tsr/tcga_voronoi_tsr.py \
    --features_dir "$feat_dir" \
    --per_cell_dir "$per_cell_dir" \
    --manifest "$manifest" \
    --out_csv "$out_dir/tcga_coad_tsr_voronoi.csv" \
    --tile_um 1000 --um_per_px 0.5 \
    --area_um 8 --max_dist_um 50 --white_thr 220 --min_tumour 1

echo "[voronoi] area CSV done -> $out_dir"
