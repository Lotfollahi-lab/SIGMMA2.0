#BSUB -J train-SIGMMA
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o train-SIGMMA.%J.out
#BSUB -e train-SIGMMA.%J.err
#BSUB -q gpu-lotfollahi-train
#BSUB -n 16
#BSUB -M 150GB
#BSUB -R "select[mem>150GB] rusage[mem=150GB] span[hosts=1]"
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=2"

set -e

module load python-3.11.6

# nvidia-smi
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_dir="/nfs/team361/dj16/projects/SIGMMA"
output_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/img_spatial_alignment/outputs"

poetry install

poetry run which python

poetry run python -u scripts/train_model.py -config_file ${run_dir}/configs/in-house_train_sigmma.yaml -out_dir ${output_dir}/in-house_sigmma_${LSB_JOBID}