#!/usr/bin/env bash
# Cox PH for the DICHOTOMIZED TSR-high vs TSR-low group (reference = low) on
# TCGA-KIRC full-519 cases. tcga_survival.py only fits Cox on CONTINUOUS
# stroma_fraction; this gives the binary group HR, unadjusted (M1) and adjusted
# for age (M2) and age+stage (M3), for both fixed05 and median splits.
# Reads existing outputs/tcga_kirc_tsr/{fixed05,median}/*_cases.csv -> CPU-only,
# seconds. No GPU, no pipeline re-run.
#BSUB -J tcga_kirc_cox_group
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_kirc_cox_group.%J.out
#BSUB -e logs/job_logs/tcga_kirc_cox_group.%J.err
#BSUB -q oversubscribed
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "select[mem>4000] rusage[mem=4000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
VENV_PY="$run_dir/.venv/bin/python"

"$VENV_PY" -u scripts/tcga_tsr/tcga_cox_group.py \
  --base outputs/tcga_kirc_tsr \
  --stem tcga_kirc_tsr \
  --out outputs/tcga_kirc_tsr/cox_group_hr.txt
