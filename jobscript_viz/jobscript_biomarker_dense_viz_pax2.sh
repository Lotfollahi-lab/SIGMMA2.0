#!/usr/bin/env bash
# Dense predicted-PAX8 spatial viz for TCGA-BP-5174 (all-nuclei feature cache).
# CPU (zarr strip-read + matplotlib) -> oversubscribed.
#BSUB -J biomarker_dense_viz_pax2
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_dense_viz_pax2.%J.out
#BSUB -e logs/job_logs/biomarker_dense_viz_pax2.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 7500
#BSUB -R "select[mem>7500] rusage[mem=7500] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

"$POETRY_PY" -u scripts/biomarker/viz_tcga_gene_spatial.py \
    --bundle "$run_dir/outputs/biomarker/gene_regression/bundles/sigmma_concat.joblib" \
    --manifest "$run_dir/configs/tcga_bp5174_dense_manifest.csv" \
    --feat_dir "/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_kirc/dense/sigmma_36852_step2000" \
    --slide_csv "$run_dir/outputs/biomarker/tcga_predictions/tcga_kirc_biomarker_slide.csv" \
    --gene PAX2 --marker_size 4 \
    --samples TCGA-BP-5174-01Z-00-DX1.d6f8427f-fbbe-4582-a020-ede5675a7e76 \
    --out "$run_dir/figures/clinical_outcome/biomarker/spatial/tcga_bp5174_pax2_dense.png"
echo "[done] dense PAX8 viz"
