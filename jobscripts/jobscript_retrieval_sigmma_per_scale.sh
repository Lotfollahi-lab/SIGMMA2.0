#BSUB -J retrieval-sigmma-per-scale
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/retrieval-sigmma-per-scale.%J.out
#BSUB -e logs/job_logs/retrieval-sigmma-per-scale.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 20
#BSUB -M 400GB
#BSUB -R "select[mem>400GB] rusage[mem=400GB] span[hosts=1]"

set -e

module load python-3.11.6

export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=${MASTER_PORT}"

export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
output_dir="${run_dir}/outputs/retrieval/baselines"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# Re-evaluate the same SIGMMA checkpoint (in-house_sigmma_16314 step 1500,
# which the macro retrieval in 18041 used) on micro and meso scales for the
# per-scale ablation row in the retrieval table.
configs=(
    "in-house_retrieval_sigmma_micro.yaml"
    "in-house_retrieval_sigmma_meso.yaml"
)

for cfg in "${configs[@]}"; do
    cfg_name="${cfg%.yaml}"
    job_outdir="${output_dir}/${cfg_name}/${LSB_JOBID}"
    mkdir -p ${job_outdir}
    cp ${run_dir}/configs/${cfg} ${job_outdir}/

    echo "=========================================="
    echo "Running retrieval benchmark with ${cfg}"
    echo "Output dir: ${job_outdir}"
    echo "=========================================="

    python -u scripts/benchmark_retrieval.py \
        -config_file ${run_dir}/configs/${cfg} \
        -out_dir ${job_outdir}
done
