#BSUB -J train-baselines
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/train-baselines.%J.out
#BSUB -e logs/job_logs/train-baselines.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=2"
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
output_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/baselines"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# === Baseline fine-tuning =====================================================
# All four manuscript baselines are fine-tuned on our in-house splits with
# symmetric InfoNCE on H&E <-> top-K-HVG-text (CLIP/PLIP/OmiCLIP) or H&E <->
# NEMO cell embedding (BLEEP). Comment out any baseline you don't want to run.
# ==============================================================================

configs=(
    # "in-house_train_bleep.yaml"   # ran successfully in 18189
    # "in-house_train_clip.yaml"    # ran successfully in 18189
    # "in-house_train_plip.yaml"    # ran successfully in 18189
    "in-house_train_omiclip.yaml"   # 18189 failed here (missing ckpt); rerun only this
)

for cfg in "${configs[@]}"; do
    cfg_name="${cfg%.yaml}"
    job_outdir="${output_dir}/${cfg_name}/${LSB_JOBID}"
    mkdir -p ${job_outdir}
    cp ${run_dir}/configs/${cfg} ${job_outdir}/

    echo "=========================================="
    echo "Fine-tuning baseline with ${cfg}"
    echo "Output dir: ${job_outdir}"
    echo "=========================================="

    torchrun --standalone --nproc_per_node 2 scripts/train_model.py \
        -config_file ${run_dir}/configs/${cfg} \
        -out_dir ${job_outdir}
done
