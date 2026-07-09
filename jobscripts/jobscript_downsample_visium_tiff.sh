#!/bin/bash
#BSUB -J downsample_visium_tiff
#BSUB -q oversubscribed
#BSUB -G s10396
#BSUB -n 2
#BSUB -M 7500
#BSUB -R "select[mem>7500] rusage[mem=7500] span[hosts=1]"
#BSUB -o logs/job_logs/downsample_visium_tiff.%J.out
#BSUB -e logs/job_logs/downsample_visium_tiff.%J.err

set -euo pipefail
cd /nfs/team361/dj16/projects/SIGMMA

SECTION_ID="${SECTION_ID:-AT15-BRA-4-FO-F1-S31}"
TARGET_UM_PX="${TARGET_UM_PX:-0.5}"

.venv/bin/python scripts/downsample_visium_tiff.py \
    --section_id "${SECTION_ID}" \
    --target_um_px "${TARGET_UM_PX}"
