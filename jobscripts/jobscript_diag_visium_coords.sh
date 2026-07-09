#!/bin/bash
#BSUB -q oversubscribed
#BSUB -G s10396
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "select[mem>4000] rusage[mem=4000]"
#BSUB -o logs/job_logs/diag_visium_coords.%J.out
#BSUB -e logs/job_logs/diag_visium_coords.%J.err
#BSUB -J diag_visium_coords
set -euo pipefail
cd /nfs/team361/dj16/projects/SIGMMA
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "activate failed"
python -u scripts/_diag_visium_coords.py
