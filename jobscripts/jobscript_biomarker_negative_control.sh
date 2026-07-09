#!/usr/bin/env bash
# Negative control: apply the frozen kidney-RCC PAX8 probe to PAX8-negative
# colorectal (TCGA COAD/READ) vs PAX8-positive kidney (KIRC); compare predicted
# PAX8 distributions. Forward-only CPU sklearn -> oversubscribed.
#BSUB -J biomarker_neg_control
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_neg_control.%J.out
#BSUB -e logs/job_logs/biomarker_neg_control.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"

"$POETRY_PY" -u scripts/biomarker/biomarker_negative_control.py \
    --bundle "$run_dir/outputs/biomarker/gene_regression/bundles/sigmma_concat.joblib" \
    --gene PAX8 \
    --kirc_dir "$cr/tcga_kirc/sigmma_36852_step2000" \
    --coad_dir "$cr/tcga_coad/sigmma_36852_step2000" \
    --read_dir "$cr/tcga_read/sigmma_36852_step2000" \
    --out_dir "$run_dir/outputs/biomarker/negative_control" \
    --fig_dir "$run_dir/figures/clinical_outcome/biomarker/negative_control"
echo "[done] PAX8 negative control"
