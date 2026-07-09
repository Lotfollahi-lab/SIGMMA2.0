#BSUB -J retrieval-sigmma-36852-step2000
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/retrieval-sigmma-36852-step2000.%J.out
#BSUB -e logs/job_logs/retrieval-sigmma-36852-step2000.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 16
#BSUB -M 320GB
#BSUB -R "select[mem>320GB] rusage[mem=320GB] span[hosts=1]"

# Cross-modal retrieval on SIGMMA dense attn-pool gex-svg a01 (job 36852) @ step2000.
# Single-checkpoint run (no baselines). Mirrors jobscript_retrieval_full_rerun_step2000.sh
# index 1 otherwise; output dir matches the full_rerun harvest convention.

set -e

module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((29500 + RANDOM % 1000))

run_dir="/nfs/team361/dj16/projects/SIGMMA"
output_dir="${run_dir}/outputs/retrieval/full_rerun_step2000"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"
cd "${run_dir}"

cfg="in-house_retrieval_sigmma_dense_attnpool_a01_step2000.yaml"
cfg_name="${cfg%.yaml}"
job_outdir="${output_dir}/${cfg_name}"
mkdir -p "${job_outdir}"
cp "${run_dir}/configs/${cfg}" "${job_outdir}/"

echo "=========================================="
echo "retrieval ${cfg} (job 36852 / step2000)"
echo "Out: ${job_outdir}"
echo "=========================================="

python -u scripts/benchmark_retrieval.py \
    -config_file "${run_dir}/configs/${cfg}" \
    -out_dir "${job_outdir}"
