#BSUB -J extract-cell-niche
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/extract-cell-niche.%J.out
#BSUB -e logs/job_logs/extract-cell-niche.%J.err
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
out_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/cell_niche/features/${LSB_JOBID}"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# Phase 2: per-cell H&E embedding extraction (SIGMMA per-scale LoRA + raw UNI
# baseline) for the niche label-transfer task. Iterates the 35 evaluable
# manifest rows in configs/cell_niche_evaluable_slides.csv. Writes one HDF5
# per slide; downstream probing loads them all together.
python -u ${run_dir}/scripts/extract_cell_niche_features.py \
    --config  ${run_dir}/configs/cell_niche_transfer.yaml \
    --out_dir ${out_dir}
