#BSUB -J eva_critique
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/eva_critique.%J.out
#BSUB -e logs/job_logs/eva_critique.%J.err
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
SCALES=(0 1 2)   # SIGMMA LoRA adapters: 0=micro, 1=meso, 2=macro
DECODERS=(linear mlp light_unet unet_skip)
mkdir -p "${OUT}"

cd "${run_dir}"

# ------------------------------------------------------------
# Component 4 (a) — train ALL (model x decoder x scale) decoders.
# Encoder frozen everywhere; only decoder weights are trained.
# Same seed (42), same RandomResizedCrop+HFlip, same AdamW(lr=2e-3).
#   UNI:    1 encoder x 4 decoders            =  4 runs
#   SIGMMA: 3 scales  x 4 decoders            = 12 runs
# Total: 16 decoder training runs.
# ------------------------------------------------------------
for DEC in "${DECODERS[@]}"; do
    echo "[train_decoder] model=uni dec=${DEC}"
    python -m scripts.eva_critique.train_decoder \
        --sigmma_config "${CFG}" \
        --model uni --decoder "${DEC}" \
        --data_root "${DATA}" --out_root "${OUT}" \
        --max_steps 2000 --batch_size 32 --lr 0.002 --seed 42
done

for SCALE in "${SCALES[@]}"; do
    for DEC in "${DECODERS[@]}"; do
        echo "[train_decoder] model=sigmma scale=${SCALE} dec=${DEC}"
        python -m scripts.eva_critique.train_decoder \
            --sigmma_config "${CFG}" \
            --model sigmma --decoder "${DEC}" --scale ${SCALE} \
            --data_root "${DATA}" --out_root "${OUT}" \
            --max_steps 2000 --batch_size 32 --lr 0.002 --seed 42
    done
done

# ------------------------------------------------------------
# Cache last-block CLS->patch attention maps for every encoder.
# UNI: 1 attention map. SIGMMA: 3 (one per active LoRA adapter).
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
# Component 4 (b) — patch-level linear probe (CLS+mean ridge, 5-fold image-grouped).
# Run once per SIGMMA scale (UNI is repeated inside each call but only logged once
# under model='uni' — dedup downstream).
# ------------------------------------------------------------
for SCALE in "${SCALES[@]}"; do
    python -m scripts.eva_critique.linear_probe \
        --sigmma_config "${CFG}" --scale ${SCALE} \
        --data_root "${DATA}" --out_dir "${OUT}/probe_scale${SCALE}"
done

# ------------------------------------------------------------
# Components 1, 2, 3, 5 — run once per SIGMMA scale, against the SAME UNI baseline.
# Uses the LINEAR (D0, EVA-default) decoder's cached prob/pred — that is the head
# EVA actually scores. Alternative decoders are aggregated separately by summarize.py.
# ------------------------------------------------------------
for SCALE in "${SCALES[@]}"; do
    SCALE_OUT="${OUT}/per_scale/scale${SCALE}"
    mkdir -p "${SCALE_OUT}"
    python -m scripts.eva_critique.pipeline_eval \
        --uni_artifacts    "${OUT}/uni/linear/test_artifacts.npz" \
        --sigmma_artifacts "${OUT}/sigmma_scale${SCALE}/linear/test_artifacts.npz" \
        --uni_attention    "${OUT}/uni/attention.npz" \
        --sigmma_attention "${OUT}/sigmma_scale${SCALE}/attention.npz" \
        --data_root "${DATA}" --out_dir "${SCALE_OUT}"
done

# ------------------------------------------------------------
# Aggregate per-scale CSVs -> stacked CSVs + figures + one-page summary
# mapping evidence to H1/H2/H3. summarize.py is scale-aware: it concatenates
# the three per-scale folders, tags each row with its scale, and emits both
# scale-specific and pooled tables in summary.md.
# ------------------------------------------------------------
python -m scripts.eva_critique.summarize \
    --per_scale_root "${OUT}/per_scale" \
    --scales 0 1 2 \
    --decoder_root "${OUT}" \
    --probe_root "${OUT}" \
    --out_dir "${OUT}"
