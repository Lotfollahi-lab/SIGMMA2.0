#!/usr/bin/env bash
# PAX8 high/low (median split) vs distant metastasis (AJCC M0/M1) contingency +
# Fisher test + heatmap for TCGA-KIRC. CPU -> oversubscribed.
#BSUB -J biomarker_met_contingency
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_met_contingency.%J.out
#BSUB -e logs/job_logs/biomarker_met_contingency.%J.err
#BSUB -q oversubscribed
#BSUB -n 1
#BSUB -M 3000
#BSUB -R "select[mem>3000] rusage[mem=3000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
PY="$run_dir/.venv/bin/python"

for agg in mean q90; do
  "$PY" -u scripts/biomarker/biomarker_metastasis_contingency.py \
    --pred_csv "$run_dir/outputs/biomarker/tcga_predictions/tcga_kirc_biomarker_patient.csv" \
    --clinical "/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-KIRC_clinical.tsv" \
    --gene PAX8 --agg "$agg" --cohort TCGA-KIRC \
    --out_dir "$run_dir/outputs/biomarker/metastasis" \
    --fig_dir "$run_dir/figures/clinical_outcome/biomarker/metastasis"
done
echo "[done] PAX8 metastasis confusion"
