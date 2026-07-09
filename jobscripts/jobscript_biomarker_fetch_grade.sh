#!/usr/bin/env bash
# Biomarker probe, step 4: fetch TCGA-KIRC tumour grade (G1-G4) per case from the
# GDC clinical API (grade is absent from the local clinical TSV). Needs outbound
# HTTPS -> transfer queue (external data movement).
#BSUB -J biomarker_fetch_grade
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_fetch_grade.%J.out
#BSUB -e logs/job_logs/biomarker_fetch_grade.%J.err
#BSUB -q transfer
#BSUB -n 1
#BSUB -M 2000
#BSUB -R "select[mem>2000] rusage[mem=2000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
PY="$run_dir/.venv/bin/python"

out_csv="$run_dir/outputs/biomarker/tcga_kirc_grade.csv"
"$PY" -u scripts/biomarker/fetch_kirc_grade.py \
    --project TCGA-KIRC --out_csv "$out_csv"
echo "[done] KIRC grade fetch"
