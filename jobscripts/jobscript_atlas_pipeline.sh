#!/bin/bash
#BSUB -J atlas_pipeline
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/atlas_pipeline.%J.out
#BSUB -e logs/job_logs/atlas_pipeline.%J.err
#BSUB -q training-parallel
#BSUB -n 8
#BSUB -M 64GB
#BSUB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]"

set -e

module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate $ENV_PATH"
cd "${run_dir}"

# ---------------------------------------------------------------------------
# Atlas pipeline driver: Steps 1–4 across one or more scales.
#
# Env vars:
#   SCALES   — space-separated subset of {micro, meso, macro, concat}.
#              Default: "micro meso macro".
#   K_GRID   — clustering K sweep grid (default "10,20,30,50").
# ---------------------------------------------------------------------------
SCALES="${SCALES:-micro meso macro}"
K_GRID="${K_GRID:-10,20,30,50}"

# Step 1 (idempotent) — niche composition cache for all 8 tissues.
python -m scripts.atlas_aggregate_niche_composition --tissue all

# Steps 2–4 per scale.
for scale in $SCALES; do
    echo "===================== scale=$scale ====================="
    python -m scripts.atlas_cluster_embeddings \
        --scale "image_emb_${scale}" --k_grid "${K_GRID}"
    python -m scripts.atlas_compute_cluster_metrics --scale "${scale}"
    python -m scripts.plot_atlas_clusters --scale "${scale}"
done
