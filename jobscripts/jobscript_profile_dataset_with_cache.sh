#BSUB -J profile-cache
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/profile-cache.%J.out
#BSUB -e logs/job_logs/profile-cache.%J.err
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

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# Take the first slide from the train manifest
slide=$(awk -F',' 'NR==2 {print $1}' /lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train.csv)
echo "Profiling slide: ${slide}"

python -u ${run_dir}/scripts/profile_dataset_with_cache.py \
    --manifest         /lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train.csv \
    --slide            "${slide}" \
    --hier_cache_dir   /lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/hier_cache_fov256_res3_tt20000 \
    --fov_size         256 \
    --num_resolutions  3 \
    --tissue_threshold 20000 \
    --n_samples        300
