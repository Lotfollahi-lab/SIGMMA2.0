#!/bin/bash
#BSUB -J viz_st_emb_umap
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/viz_st_emb_umap.%J.out
#BSUB -e logs/job_logs/viz_st_emb_umap.%J.err
#BSUB -q normal
#BSUB -n 4
#BSUB -M 32GB
#BSUB -R "select[mem>32GB] rusage[mem=32GB] span[hosts=1]"

set -e

module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate $ENV_PATH"
cd "${run_dir}"

# ---------------------------------------------------------------------------
# UMAP of SIGMMA ST tile embeddings (step2000, job 36852 full test-split),
# coloured by cleaned biological tissue type and by per-cohort dataset type.
#   SCALE — one of {micro, meso, macro, all}  (default all → 3 rows)
# ---------------------------------------------------------------------------
SCALE="${SCALE:-all}"

python scripts/visualization/viz_st_emb_umap_tissue_dataset.py --scale "$SCALE"
