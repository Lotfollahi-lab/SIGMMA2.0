#BSUB -J niche-cross-slide-step1000
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/niche-cross-slide-step1000.%J.out
#BSUB -e logs/job_logs/niche-cross-slide-step1000.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=10GB:num=1"
#BSUB -n 8
#BSUB -M 64GB
#BSUB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]"
#BSUB -w "done(19044)"

set -e

module load python-3.11.6

export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

run_dir="/nfs/team361/dj16/projects/SIGMMA"
# Features extracted by job 19044 (cell-niche extract using 18328 step1000 ckpt).
features_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/cell_niche/features/19044"
out_dir="${run_dir}/outputs/cell_niche/${LSB_JOBID}"

ENV_PATH="/nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

python -u ${run_dir}/scripts/probe_niche_cross_slide.py \
    --features_dir ${features_dir} \
    --manifest     ${run_dir}/configs/cell_niche_evaluable_slides.csv \
    --out_dir      ${out_dir}
