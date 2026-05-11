#BSUB -J retrieval-baselines
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/retrieval-baselines.%J.out
#BSUB -e logs/job_logs/retrieval-baselines.%J.err
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

ENV_PATH="/nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# === Baseline retrieval evaluation ===========================================
# Three baselines (CLIP / PLIP / OmiCLIP) are evaluated zero-shot from their
# official checkpoints. BLEEP requires a checkpoint produced by
# jobscript_train_baselines.sh; update the path inside its config first.
# =============================================================================

# Comment out any baseline you don't want to run this job.
# BLEEP is uncommented only after jobscript_train_baselines.sh has produced a
# checkpoint and configs/in-house_retrieval_bleep.yaml has been updated.
configs=(
    "in-house_retrieval_clip.yaml"
    "in-house_retrieval_plip.yaml"
    "in-house_retrieval_omiclip.yaml"
    "in-house_retrieval_bleep.yaml"
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
