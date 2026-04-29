#BSUB -J seg-per-scale[1-3]
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o seg-per-scale.%J.%I.out
#BSUB -e seg-per-scale.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 8
#BSUB -M 400GB
#BSUB -R "select[mem>400GB] rusage[mem=400GB] span[hosts=1]"

set -e

module load python-3.11.6

export PYTHONUNBUFFERED=1
export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=${MASTER_PORT}"

export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"

ENV_PATH="/nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# Job array: each task runs one scale (LSB_JOBINDEX 1..3 → scale_idx 0..2,
# i.e. micro/meso/macro). Per-scale benchmark activates the matching LoRA
# adapter on UNI and runs EVA segmentation with 5 seeds (EVA leaderboard
# protocol). Results land under
#   logs/sigmma_per_scale/scale_${SCALE_IDX}/{monusac}/run_{0..4}.
SCALE_IDX=$((LSB_JOBINDEX - 1))
echo "[seg-per-scale] LSB_JOBINDEX=${LSB_JOBINDEX} → scale_idx=${SCALE_IDX}"

python ${run_dir}/scripts/benchmark_segmentation_per_scale.py \
    --sigmma_config ${run_dir}/configs/image_pathology_segmentation_per_scale_sigmma.yaml \
    --datasets monusac \
    --scales ${SCALE_IDX} \
    --data_root ${run_dir}/data \
    --output_root ${run_dir}/logs/sigmma_per_scale \
    --n_runs 5
