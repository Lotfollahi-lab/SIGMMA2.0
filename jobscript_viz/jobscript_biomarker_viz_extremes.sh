#!/usr/bin/env bash
# Spatial viz of the HIGHEST (TCGA-B8-5546, stage I) and LOWEST (TCGA-BP-4337,
# stage III) predicted-PAX8 patients: H&E next to predicted PAX8. Sparse
# (200-tile) feature cache like the main spatial figure. CPU -> oversubscribed.
#BSUB -J biomarker_viz_extremes
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_viz_extremes.%J.out
#BSUB -e logs/job_logs/biomarker_viz_extremes.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 7500
#BSUB -R "select[mem>7500] rusage[mem=7500] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

feat_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_kirc/sigmma_36852_step2000"
"$POETRY_PY" -u scripts/biomarker/viz_tcga_gene_spatial.py \
    --bundle "$run_dir/outputs/biomarker/gene_regression/bundles/sigmma_concat.joblib" \
    --manifest "$run_dir/configs/tcga_kirc_manifest.csv" \
    --feat_dir "$feat_dir" \
    --slide_csv "$run_dir/outputs/biomarker/tcga_predictions/tcga_kirc_biomarker_slide.csv" \
    --gene PAX8 --marker_size 9 \
    --samples \
      TCGA-B8-5546-01Z-00-DX1.6c98ff0d-ffd9-46e7-9759-d3639e6dc07c \
      TCGA-BP-4337-01Z-00-DX1.5c88395f-317b-4cc5-8945-16c7cdb0876e \
    --out "$run_dir/figures/clinical_outcome/biomarker/spatial/tcga_kirc_pax8_extremes.png"
echo "[done] PAX8 extremes viz"
