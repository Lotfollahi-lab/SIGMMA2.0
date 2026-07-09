#!/usr/bin/env bash
# Biomarker probe, step 5: associate per-patient predicted CA9/PAX8 with
# TCGA-KIRC severity (OS/DSS survival, AJCC stage, tumour grade). CPU
# lifelines/statsmodels -> oversubscribed.
#BSUB -J biomarker_severity
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_severity.%J.out
#BSUB -e logs/job_logs/biomarker_severity.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 4000
#BSUB -R "select[mem>4000] rusage[mem=4000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
PY="$run_dir/.venv/bin/python"

clin="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-KIRC_clinical.tsv"
pred="$run_dir/outputs/biomarker/tcga_predictions/tcga_kirc_biomarker_patient.csv"
grade="$run_dir/outputs/biomarker/tcga_kirc_grade.csv"
out="$run_dir/outputs/biomarker/severity"

"$PY" -u scripts/biomarker/biomarker_severity.py \
    --pred_csv "$pred" \
    --clinical "$clin" \
    --grade_csv "$grade" \
    --out_dir "$out" \
    --fig_dir "$run_dir/figures/clinical_outcome/biomarker" \
    --min_days 30
echo "[done] biomarker severity association"
