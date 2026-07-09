#BSUB -J scale-divergence
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/scale-divergence.%J.out
#BSUB -e logs/job_logs/scale-divergence.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=40GB:num=1"
#BSUB -n 8
#BSUB -M 64GB
#BSUB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]"

set -e

module load python-3.11.6

export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
out_dir="${run_dir}/outputs/diagnostics/scale_divergence/${LSB_JOBID}"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

python -u ${run_dir}/scripts/analyze_scale_divergence.py \
    --config   ${run_dir}/configs/analyze_scale_divergence_sigmma.yaml \
    --slide    BK39-SKI-27-FO-1-S3 \
    --manifest /lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST_test.csv \
    --out_dir  ${out_dir}
