#BSUB -J train-SIGMMA-dense-attnpool
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/train-SIGMMA-dense-attnpool.%J.out
#BSUB -e logs/job_logs/train-SIGMMA-dense-attnpool.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=2"
#BSUB -n 32
#BSUB -M 400GB
#BSUB -R "select[mem>400GB && hname!='farm-gpu0504'] rusage[mem=400GB] span[hosts=1]"

set -e

module load python-3.11.6

export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=${MASTER_PORT}"

export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
output_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma_dense_attnpool"
config_file="${run_dir}/configs/in-house_train_sigmma_dense_attnpool.yaml"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

mkdir -p ${output_dir}/${LSB_JOBID}
cp ${config_file} ${output_dir}/${LSB_JOBID}/
torchrun --standalone --nproc_per_node 2 scripts/train_model.py -config_file ${config_file} -out_dir ${output_dir}/${LSB_JOBID}
