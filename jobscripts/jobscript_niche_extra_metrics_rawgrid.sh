#!/usr/bin/env bash
#BSUB -J niche-extra-rawgrid
#BSUB -G s10396
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=4GB:num=1"
#BSUB -n 2
#BSUB -M 8000
#BSUB -R "select[mem>8000] rusage[mem=8000] span[hosts=1]"
#BSUB -o logs/job_logs/niche-extra-rawgrid.%J.out
#BSUB -e logs/job_logs/niche-extra-rawgrid.%J.err

# Recompute cell-niche headline metrics (MCC + weighted F1 + Moran's I) with the
# SIGMMA representation switched to the RAW-GRID feature (z_sigmma_macro_dense_noproj):
# the same LoRA-adapted UNI2-h forward_features patch grid EVA / GEX-2D use
# (no scale pooling, no scale_mlps projection), mask-pooled per cell.
# Reads existing predictions.npz (no re-extraction / re-probe needed).

set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"

cd "$run_dir"
python -u scripts/niche_extra_metrics.py
