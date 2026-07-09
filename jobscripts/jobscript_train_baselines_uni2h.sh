#BSUB -J train-baselines-uni2h[1-3]
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/train-baselines-uni2h.%J.%I.out
#BSUB -e logs/job_logs/train-baselines-uni2h.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=2"
#BSUB -n 20
#BSUB -M 600GB
#BSUB -R "select[mem>600GB] rusage[mem=600GB] span[hosts=1]"

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

# === Fan-out: CLIP/PLIP/OmiCLIP fine-tuning on UNI2-h =========================
# BLEEP is handled separately by jobscript_train_baselines_uni2h_smoke.sh (which
# doubles as the BLEEP production run). Submit this array after BLEEP smoke
# clears the first ~50 steps without OOM on 76GB GPU.
# ==============================================================================

configs=(
    "in-house_train_clip_uni.yaml"
    "in-house_train_plip_uni.yaml"
    "in-house_train_omiclip_uni.yaml"
)

idx=$((LSB_JOBINDEX - 1))
cfg="${configs[$idx]}"
cfg_name="${cfg%.yaml}"
job_outdir="${output_dir}/${cfg_name}/${LSB_JOBID}_${LSB_JOBINDEX}"
mkdir -p ${job_outdir}
cp ${run_dir}/configs/${cfg} ${job_outdir}/

echo "=========================================="
echo "[ARRAY ${LSB_JOBINDEX}/3] Fine-tuning ${cfg} on UNI2-h"
echo "Output dir: ${job_outdir}"
echo "=========================================="

torchrun --standalone --nproc_per_node 2 scripts/train_model.py \
    -config_file ${run_dir}/configs/${cfg} \
    -out_dir ${job_outdir}
