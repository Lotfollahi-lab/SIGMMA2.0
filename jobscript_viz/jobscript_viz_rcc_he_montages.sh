#BSUB -J viz_rcc_he
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/viz_rcc_he.%J.out
#BSUB -e logs/job_logs/viz_rcc_he.%J.err
#BSUB -q oversubscribed
#BSUB -n 8
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

python scripts/viz_rcc_he_montages.py --cohort "${COHORT:-both}"
