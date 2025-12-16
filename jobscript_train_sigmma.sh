#BSUB -J train-SIGMMA
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o train-SIGMMA.%J.out
#BSUB -e train-SIGMMA.%J.err
#BSUB -q training-parallel
#BSUB -U mlteam_burnin_beta
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=3"
#BSUB -n 16
#BSUB -M 300GB
#BSUB -R "select[mem>300GB] rusage[mem=300GB] span[hosts=1]"

set -e

module load python-3.11.6

# nvidia-smi
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_dir="/nfs/team361/dj16/projects/SIGMMA"
output_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/img_spatial_alignment/outputs"

ENV_PATH="/nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate"

[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# Without DDP
python -u scripts/train_model.py -config_file ${run_dir}/configs/in-house_train_sigmma.yaml -out_dir ${output_dir}/in-house_sigmma_${LSB_JOBID}

# With DDP
# torchrun --standalone --nproc_per_node 3 scripts/train_model.py -config_file ${run_dir}/configs/in-house_train_sigmma.yaml -out_dir ${output_dir}/in-house_sigmma_${LSB_JOBID} 