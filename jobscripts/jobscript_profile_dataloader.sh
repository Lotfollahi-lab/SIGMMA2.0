#BSUB -J profile-dataloader
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/profile-dataloader.%J.out
#BSUB -e logs/job_logs/profile-dataloader.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=4GB:num=1"
#BSUB -n 4
#BSUB -M 64GB
#BSUB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]"

set -e

module load python-3.11.6

export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

run_dir="/nfs/team361/dj16/projects/SIGMMA"
out_dir="${run_dir}/outputs/diagnostics/profile_dataloader/${LSB_JOBID}"
mkdir -p ${out_dir}

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

python -u ${run_dir}/scripts/profile_dataloader.py \
    --manifest /lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train.csv \
    --n_tiles 200 \
    --n_slides 3 \
    --fov_size 256 \
    --num_resolutions 3 \
    --out_csv ${out_dir}/profile_per_step.csv
