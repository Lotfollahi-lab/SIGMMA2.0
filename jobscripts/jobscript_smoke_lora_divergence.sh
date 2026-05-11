#BSUB -J smoke_lora_div
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/smoke_lora_div.%J.out
#BSUB -e logs/job_logs/smoke_lora_div.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=40GB:num=1"
#BSUB -n 4
#BSUB -M 80GB
#BSUB -R "select[mem>80GB] rusage[mem=80GB] span[hosts=1]"

set -e

module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate $ENV_PATH"

cd "${run_dir}"

python -m scripts.smoke_lora_divergence \
    --sigmma_config configs/cell_niche_transfer.yaml \
    --hematoxylin_zarr /lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_beacon/20241010__132615__SGP194_SKI_RUN1/BK24-SKI-27-FO-01-S3/hematoxylin_eosin.zarr \
    --num_fovs 50 \
    --fov_size 256 \
    --batch_size 4 \
    --out_dir outputs/gex_2d/smoke
