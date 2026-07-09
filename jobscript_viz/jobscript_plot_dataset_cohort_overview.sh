#BSUB -J plot_cohort_overview
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/plot_cohort_overview.%J.out
#BSUB -e logs/job_logs/plot_cohort_overview.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=4GB:num=1"
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

python scripts/visualization/plot_dataset_cohort_overview.py
