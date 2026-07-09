#BSUB -J train-SIGMMA
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/train-SIGMMA.%J.out
#BSUB -e logs/job_logs/train-SIGMMA.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=2"
#BSUB -n 32
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
output_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"

[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# === Run Without DDP
# Save the config file to the output directory for reference
# mkdir -p ${output_dir}/${LSB_JOBID}
# cp ${run_dir}/configs/in-house_train_sigmma.yaml ${output_dir}/${LSB_JOBID}/
# python -u scripts/train/train_model.py -config_file ${run_dir}/configs/in-house_train_sigmma.yaml -out_dir ${output_dir}/${LSB_JOBID}

# === RUN With DDP
# Save the config file to the output directory for reference
mkdir -p ${output_dir}/${LSB_JOBID}
cp ${run_dir}/configs/in-house_train_sigmma.yaml ${output_dir}/${LSB_JOBID}/
torchrun --standalone --nproc_per_node 2 scripts/train/train_model.py -config_file ${run_dir}/configs/in-house_train_sigmma.yaml -out_dir ${output_dir}/${LSB_JOBID}