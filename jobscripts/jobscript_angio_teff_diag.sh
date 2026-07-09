#!/usr/bin/env bash
# Diagnose .X normalisation state of in-house RCC spatial_data_processed.h5 before
# trusting the angio/Teff signature scores (job 59010 flagged X as not-raw-counts
# for ALL sections, and we're on the gex-double-log fix branch). Read-only.
#BSUB -J angio_teff_diag
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/angio_teff_diag.%J.out
#BSUB -e logs/job_logs/angio_teff_diag.%J.err
#BSUB -q oversubscribed
#BSUB -n 1
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

# representative set: high-angio (FT-1), the near-zero outliers (FT-5, missing
# IFNG too), a high-teff (CV6-FT-1), and a different donor (CV9-FT-2).
"$POETRY_PY" -u scripts/angio_teff/diag_xstate.py \
    --manifest "$run_dir/configs/cell_niche_evaluable_slides_rcc.csv" \
    --samples CV1-KID-0-FT-1 CV1-KID-0-FT-5 CV6-KID-0-FT-5 CV6-KID-0-FT-1 CV9-KID-0-FT-2-S3
