#BSUB -J inspect_obs
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/inspect_obs.%J.out
#BSUB -e logs/job_logs/inspect_obs.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=8GB:num=1"
#BSUB -n 2
#BSUB -M 32GB
#BSUB -R "select[mem>32GB] rusage[mem=32GB] span[hosts=1]"

set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH"
cd "${run_dir}"

python scripts/_inspect_anndata_obs.py \
    /lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_beacon/20241010__132615__SGP194_SKI_RUN1/BK24-SKI-28-FO-1-s3/spatial_data_processed.h5
