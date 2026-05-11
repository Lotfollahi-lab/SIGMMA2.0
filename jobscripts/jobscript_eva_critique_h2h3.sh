#BSUB -J eva_critique_h2h3
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/eva_critique_h2h3.%J.out
#BSUB -e logs/job_logs/eva_critique_h2h3.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 8
#BSUB -M 320GB
#BSUB -R "select[mem>320GB] rusage[mem=320GB] span[hosts=1]"

set -e

module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate $ENV_PATH"

CFG="${run_dir}/configs/image_pathology_segmentation_per_scale_sigmma.yaml"
DATA="${run_dir}/data"
OUT="${run_dir}/outputs/eva_critique"
SCALES=(0 1 2)
DEC=convdec_image  # EVA upstream MoNuSAC default (ConvDecoderWithImage)

cd "${run_dir}"

# ------------------------------------------------------------
# Phase 1 — train EVA-default decoder for UNI + SIGMMA s0/s1/s2.
# batch_size=64 matches kaiko-ai/eva configs/vision/pathology/online/segmentation/monusac.yaml.
# ------------------------------------------------------------
echo "[train_decoder] model=uni dec=${DEC}"
python -m scripts.eva_critique.train_decoder \
    --sigmma_config "${CFG}" \
    --model uni --decoder "${DEC}" \
    --data_root "${DATA}" --out_root "${OUT}" \
    --max_steps 2000 --batch_size 64 --lr 0.002 --seed 42

for SCALE in "${SCALES[@]}"; do
    echo "[train_decoder] model=sigmma scale=${SCALE} dec=${DEC}"
    python -m scripts.eva_critique.train_decoder \
        --sigmma_config "${CFG}" \
        --model sigmma --decoder "${DEC}" --scale ${SCALE} \
        --data_root "${DATA}" --out_root "${OUT}" \
        --max_steps 2000 --batch_size 64 --lr 0.002 --seed 42
done

# ------------------------------------------------------------
# Phase 2 — attention caches (UNI + 3 SIGMMA scales)
# ------------------------------------------------------------
python -m scripts.eva_critique.extract_attention \
    --sigmma_config "${CFG}" --model uni \
    --data_root "${DATA}" --out_dir "${OUT}/uni"

for SCALE in "${SCALES[@]}"; do
    python -m scripts.eva_critique.extract_attention \
        --sigmma_config "${CFG}" --model sigmma --scale ${SCALE} \
        --data_root "${DATA}" --out_dir "${OUT}/sigmma_scale${SCALE}"
done

# ------------------------------------------------------------
# Phase 4 — H2 (region-stratified) + H3 (alt + instance) per SIGMMA scale.
# pipeline_eval consumes the EVA-default decoder's test_artifacts.npz.
# ------------------------------------------------------------
for SCALE in "${SCALES[@]}"; do
    SCALE_OUT="${OUT}/per_scale/scale${SCALE}"
    mkdir -p "${SCALE_OUT}"
    python -m scripts.eva_critique.pipeline_eval \
        --uni_artifacts    "${OUT}/uni/${DEC}/test_artifacts.npz" \
        --sigmma_artifacts "${OUT}/sigmma_scale${SCALE}/${DEC}/test_artifacts.npz" \
        --uni_attention    "${OUT}/uni/attention.npz" \
        --sigmma_attention "${OUT}/sigmma_scale${SCALE}/attention.npz" \
        --data_root "${DATA}" --out_dir "${SCALE_OUT}"
done

# ------------------------------------------------------------
# Phase 5 — aggregate. summarize.py is subset-tolerant for missing
# linear_probe_results.csv and decoder ablation rows.
# ------------------------------------------------------------
python -m scripts.eva_critique.summarize \
    --per_scale_root "${OUT}/per_scale" \
    --scales 0 1 2 \
    --decoder_root "${OUT}" \
    --probe_root "${OUT}" \
    --out_dir "${OUT}"
