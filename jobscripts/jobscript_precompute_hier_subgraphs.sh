#BSUB -J precompute-hier
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/precompute-hier.%J.out
#BSUB -e logs/job_logs/precompute-hier.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=4GB:num=1"
#BSUB -n 32
#BSUB -M 256GB
#BSUB -R "select[mem>256GB] rusage[mem=256GB] span[hosts=1]"

set -e

module load python-3.11.6

export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

# Date suffix for the cache dir, passed as $1 (e.g. `bsub jobscript.sh 2026_05_31`).
# Defaults to today if omitted. Output: hier_cache_fov256_res3_tt20000_<DATE>.
DATE="${1:-$(date +%Y_%m_%d)}"

run_dir="/nfs/team361/dj16/projects/SIGMMA"
out_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/hier_cache_fov256_res3_tt20000_${DATE}"
echo "Building hier_cache at: ${out_dir}"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# Build cache for train, val and test manifests so eval also benefits.
for manifest in \
    /lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_train.csv \
    /lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_val.csv \
    /lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_test.csv ; do
  echo "=== ${manifest} ==="
  python -u ${run_dir}/scripts/precompute_hierarchical_subgraphs.py \
      --manifest         ${manifest} \
      --out_dir          ${out_dir} \
      --fov_size         256 \
      --num_resolutions  3 \
      --tissue_threshold 20000 \
      --n_workers        24
done
