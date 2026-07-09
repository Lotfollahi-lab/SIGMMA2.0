#BSUB -J viz_he_single
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/viz_he_single.%J.out
#BSUB -e logs/job_logs/viz_he_single.%J.err
#BSUB -q oversubscribed
#BSUB -n 4
#BSUB -M 6GB
#BSUB -R "select[mem>6GB] rusage[mem=6GB] span[hosts=1]"

set -e

module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate $ENV_PATH"
cd "${run_dir}"

python scripts/visualization/viz_he_single_section.py
