#BSUB -J plot_tissue_stats
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/plot_tissue_stats.%J.out
#BSUB -e logs/job_logs/plot_tissue_stats.%J.err
#BSUB -q normal
#BSUB -n 1
#BSUB -M 8GB
#BSUB -R "select[mem>8GB] rusage[mem=8GB] span[hosts=1]"

set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH"
cd "${run_dir}"

python scripts/visualization/plot_manifest_tissue_stats.py
