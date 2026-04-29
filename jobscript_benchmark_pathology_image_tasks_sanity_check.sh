#BSUB -J pathology_benchmark_sanity_check-SIGMMA
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o pathology_benchmark_sanity_check-SIGMMA.%J.out
#BSUB -e pathology_benchmark_sanity_check-SIGMMA.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 8
#BSUB -M 400GB
#BSUB -R "select[mem>400GB] rusage[mem=400GB] span[hosts=1]"

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

# Diagnose how LoRA fine-tuning has moved UNI: per-adapter ΔW magnitude,
# per-block / per-target-module breakdown, top-changed layers, and (for
# per_scale checkpoints) cross-adapter cosine similarity. CSV row-per-layer
# is dropped under logs/ for notebook plotting.
mkdir -p ${run_dir}/logs

# Per-scale checkpoint (3 adapters: scale_0/scale_1/scale_2). Swap to
# image_pathology_task_sigmma.yaml if the target checkpoint is single-adapter.
SIGMMA_CONFIG=${run_dir}/configs/image_pathology_segmentation_per_scale_sigmma.yaml

python ${run_dir}/scripts/diff_uni_weights.py \
    --sigmma_config ${SIGMMA_CONFIG} \
    --top_k 20 \
    --cross_adapter \
    --csv_out ${run_dir}/logs/diff_uni_weights.${LSB_JOBID}.csv