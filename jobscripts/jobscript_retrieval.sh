#BSUB -J retrieval-SIGMMA
#BSUB -G s10396 
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/retrieval-SIGMMA.%J.out
#BSUB -e logs/job_logs/retrieval-SIGMMA.%J.err
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
output_dir="${run_dir}/outputs/retrieval/sigmma"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"

[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# === RUN
python -u scripts/benchmark_retrieval.py -config_file ${run_dir}/configs/in-house_retrieval_sigmma.yaml -out_dir ${output_dir}/${LSB_JOBID}