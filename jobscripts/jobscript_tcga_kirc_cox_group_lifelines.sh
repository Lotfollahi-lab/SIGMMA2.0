#!/usr/bin/env bash
# Binary TSR-high-vs-low Cox via lifelines CoxPHFitter (Efron ties + Schoenfeld
# PH test + concordance), re-run for all 4 endpoints on the existing per-case
# csvs. Uses the POETRY venv (it has lifelines; .venv does NOT). CPU, seconds.
#BSUB -J tcga_kirc_cox_ll
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_kirc_cox_ll.%J.out
#BSUB -e logs/job_logs/tcga_kirc_cox_ll.%J.err
#BSUB -q oversubscribed
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "select[mem>4000] rusage[mem=4000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"
export PYTHONUNBUFFERED=1

for ep in OS DSS DFI PFI; do
  ep_l=$(echo "$ep" | tr '[:upper:]' '[:lower:]')
  base="$run_dir/outputs/tcga_kirc_tsr/endpoints/$ep"
  echo "================= binary group Cox (lifelines): $ep ================="
  "$POETRY_PY" -u scripts/tcga_tsr/tcga_cox_group.py \
      --base "$base" --stem "tcga_kirc_${ep_l}_tsr" \
      --out "$base/cox_group_hr_lifelines.txt" \
    || echo "[FAILED] cox_group endpoint=$ep"
done
echo "[kirc cox_group lifelines] DONE"
