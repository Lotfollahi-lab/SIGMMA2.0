#BSUB -J linear-probing-SIGMMA
#BSUB -G s10396 
#BSUB -L /usr/bin/bash
#BSUB -o linear-probing-SIGMMA.%J.out
#BSUB -e linear-probing-SIGMMA.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 20
#BSUB -M 400GB
#BSUB -R "select[mem>400GB] rusage[mem=400GB] span[hosts=1]"

set -e

module load python-3.11.6

# For torch distributed
export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=${MASTER_PORT}"

# nvidia-smi
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_dir="/nfs/team361/dj16/projects/SIGMMA"
output_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/img_spatial_alignment/outputs"

ENV_PATH="/nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate"

[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

## ===== TASK1. Gene expression prediction =====
python -u scripts/benchmark_gex_prediction.py -zeroshot False -config_file ${run_dir}/configs/in-house_linear_probing_sigmma.yaml -out_dir ${output_dir}/outputs/in-house_linear_probing_sigmma_${LSB_JOBID}
