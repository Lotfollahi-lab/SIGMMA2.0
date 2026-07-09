#!/bin/bash
#BSUB -q oversubscribed
#BSUB -G s10396
#BSUB -n 2
#BSUB -M 4500
#BSUB -R "select[mem>4500] rusage[mem=4500]"
#BSUB -o logs/job_logs/smoke_highres_crop.%J.out
#BSUB -e logs/job_logs/smoke_highres_crop.%J.err
#BSUB -J smoke_highres_crop

set -euo pipefail
cd /nfs/team361/dj16/projects/SIGMMA
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate $ENV_PATH"
python -u scripts/_smoke_highres_crop.py
