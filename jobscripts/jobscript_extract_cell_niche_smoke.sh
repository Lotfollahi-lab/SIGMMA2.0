#BSUB -J cell-niche-smoke
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/cell-niche-smoke.%J.out
#BSUB -e logs/job_logs/cell-niche-smoke.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 8
#BSUB -M 200GB
#BSUB -R "select[mem>200GB] rusage[mem=200GB] span[hosts=1]"

set -e

module load python-3.11.6

export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
out_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/cell_niche/smoke/${LSB_JOBID}"

ENV_PATH="/nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# Phase 2 smoke test: one slide only. BK25-SKI-27-FO-1-S3 is in the
# evaluable manifest, has raw_data, and has Niches_TERRA labels.
python -u ${run_dir}/scripts/extract_cell_niche_features.py \
    --config  ${run_dir}/configs/cell_niche_transfer.yaml \
    --out_dir ${out_dir} \
    --slide   BK25-SKI-27-FO-1-S3
