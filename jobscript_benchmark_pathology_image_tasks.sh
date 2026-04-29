# BSUB -J pathology_benchmark-SIGMMA
# BSUB -G team361
# BSUB -L /usr/bin/bash
# BSUB -o pathology_benchmark-SIGMMA.%J.out
# BSUB -e pathology_benchmark-SIGMMA.%J.err
# BSUB -q training-parallel
# BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
# BSUB -n 8
# BSUB -M 400GB
# BSUB -R "select[mem>400GB] rusage[mem=400GB] span[hosts=1]"

set -e

module load python-3.11.6

export PYTHONUNBUFFERED=1

# For torch distributed
export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=${MASTER_PORT}"

# nvidia-smi
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_dir="/nfs/team361/dj16/projects/SIGMMA"

ENV_PATH="/nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate"

[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# # Patch-level classification 
# python ./scripts/benchmark_image_pathology_task.py \
#     --sigmma_config /nfs/team361/dj16/projects/SIGMMA/configs/image_pathology_task_sigmma.yaml \
#     --datasets breakhis crc gleason mhist patchcam \
#     --download

# # WSI slide-level 
# python ./scripts/benchmark_image_pathology_task.py \
#    --sigmma_config /nfs/team361/dj16/projects/SIGMMA/configs/image_pathology_task_sigmma.yaml \
#    --datasets panda \
#    --download

# # Segmentation
# python scripts/benchmark_image_pathology_task.py \
#        --sigmma_config /nfs/team361/dj16/projects/SIGMMA/configs/image_pathology_task_sigmma.yaml \
#        --datasets monusac --backbone uni

python ./scripts/benchmark_image_pathology_task.py \
    --sigmma_config /nfs/team361/dj16/projects/SIGMMA/configs/image_pathology_task_sigmma.yaml \
    --datasets monusac