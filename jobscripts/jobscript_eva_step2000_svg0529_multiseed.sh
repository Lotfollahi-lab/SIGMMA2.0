#!/bin/bash
#BSUB -J eva-step2000-multiseed[1-30]
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/eva-step2000-multiseed.%J.%I.out
#BSUB -e logs/job_logs/eva-step2000-multiseed.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 8
#BSUB -M 320GB
#BSUB -R "select[mem>320GB] rusage[mem=320GB] span[hosts=1]"

# Full-rerun EVA MoNuSAC benchmark on the STEP2000 corrected-pipeline ckpts.
# 6 conditions x 5 seeds = 30 tasks. Matches the step1000 multiseed methodology
# (seeds 42/2/6/14/21; convdec_image; max_steps 2000; AJI/PQ global + R1-restricted).
#
# Conditions (idx/5):
#   0 uni2h      UNI2-h no-FT          (model uni,  per_scale cfg)         step-independent
#   1 a01        SIGMMA a01 (36852)    (model sigmma, a01_step2000 cfg)    LoRA UNI
#   2 bleep      BLEEP-UNI FT (36858)  (baseline_config in-house_eva_bleep_uni_step2000.yaml)
#   3 clip       CLIP-UNI  FT (36642_1)
#   4 plip       PLIP-UNI  FT (36642_2)
#   5 omiclip    OmiCLIP-UNI FT (36642_3)
#
# Output: outputs/eva_step2000/<cond>/seed<seed>/<art_subdir>/convdec_image/test_artifacts.npz
#         + outputs/eva_step2000/<cond>/seed<seed>/instance_metrics_{global,r1_restricted}.csv

set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate $ENV_PATH"
cd "${run_dir}"

DATA="${run_dir}/data"
PER_SCALE_CFG="${run_dir}/configs/image_pathology_segmentation_per_scale_sigmma.yaml"

conditions=(uni2h a01 bleep clip plip omiclip)
seeds=(42 2 6 14 21)

idx=$((LSB_JOBINDEX - 1))
cond="${conditions[$((idx / 5))]}"
seed="${seeds[$((idx % 5))]}"

OUT_ROOT="${run_dir}/outputs/eva_step2000/${cond}/seed${seed}"
mkdir -p "${OUT_ROOT}"

# Build the train_decoder invocation per condition.
dec_args=(--decoder convdec_image --data_root "${DATA}" --out_root "${OUT_ROOT}"
          --max_steps 2000 --batch_size 32 --lr 0.002 --seed "${seed}")

case "${cond}" in
  uni2h)
    art_subdir="uni"
    dec_args=(--sigmma_config "${PER_SCALE_CFG}" --model uni "${dec_args[@]}")
    ;;
  a01)
    art_subdir="sigmma_shared_lora"
    cfg="${run_dir}/configs/cell_niche_transfer_dense_attnpool_a01_step2000.yaml"
    cp "${cfg}" "${OUT_ROOT}/" 2>/dev/null || true
    dec_args=(--sigmma_config "${cfg}" --model sigmma "${dec_args[@]}")
    ;;
  bleep|clip|plip|omiclip)
    art_subdir="${cond}_uni_ft"
    bcfg="${run_dir}/configs/in-house_eva_${cond}_uni_step2000.yaml"
    cp "${bcfg}" "${OUT_ROOT}/" 2>/dev/null || true
    dec_args=(--sigmma_config "${PER_SCALE_CFG}" --model "${cond}_uni_ft"
              --baseline_config_path "${bcfg}" "${dec_args[@]}")
    ;;
  *)
    echo "Unknown condition: ${cond}"; exit 1 ;;
esac

echo "=========================================="
echo "[${LSB_JOBINDEX}/30] EVA step2000  cond=${cond}  seed=${seed}  -> ${OUT_ROOT}"
echo "=========================================="

ART="${OUT_ROOT}/${art_subdir}/convdec_image/test_artifacts.npz"
if [ -f "${ART}" ]; then
    echo "[${cond}|seed${seed}] reusing existing test_artifacts.npz (skipping train_decoder)"
else
    echo "[${cond}|seed${seed}] train_decoder ${dec_args[*]}"
    python -m scripts.eva_critique.train_decoder "${dec_args[@]}"
fi

echo "[${cond}|seed${seed}] instance metrics: GLOBAL + R1-restricted"
python - <<PY
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("${run_dir}")
sys.path.insert(0, str(ROOT))
from skimage.morphology import binary_dilation, disk
from scripts.eva_critique.metrics_seg import (
    aji, detection_f1_hungarian, label_components, panoptic_quality,
)
from scripts.eva_critique.inst import prob_to_instances

art = Path("${ART}")
if not art.exists():
    sys.exit(f"missing artifacts: {art}")
d = np.load(art, allow_pickle=True)
ids, probs, gts = d["image_id"], d["prob"].astype(np.float32), d["gt"]
print(f"loaded {len(ids)} FOVs from {art}")

def per_fov(prob, gt, mask=None):
    if mask is not None:
        prob = np.where(mask, prob, 0.0)
        gt = (gt.astype(bool) & mask).astype(np.uint8)
    pred = prob_to_instances(prob, threshold=0.5)
    gti = label_components(gt.astype(bool))
    PQ, SQ, DQ, *_ = panoptic_quality(pred, gti, iou_thr=0.5)
    return dict(AJI=aji(pred, gti), PQ=PQ, SQ=SQ, DQ=DQ,
                DetF1_07=detection_f1_hungarian(pred, gti,
                                                iou_thresholds=(0.5, 0.7))["f1@0.7"])

rows_g, rows_r1 = [], []
for i in range(len(ids)):
    p, g = probs[i], gts[i]
    rows_g.append({**per_fov(p, g),     "image_id": str(ids[i])})
    R1 = binary_dilation(g.astype(bool), footprint=disk(8))
    rows_r1.append({**per_fov(p, g, R1), "image_id": str(ids[i])})

out_root = Path("${OUT_ROOT}")
for label, rows in [("global", rows_g), ("r1_restricted", rows_r1)]:
    df = pd.DataFrame(rows)
    print(f"\n{label.upper()} (${cond} seed${seed}, tau=0.5, n={len(df)}):")
    print(df[['AJI','PQ','SQ','DQ','DetF1_07']].mean().round(4).to_string())
    out = out_root / f"instance_metrics_{label}.csv"
    df.to_csv(out, index=False)
    print(f"  saved {out}")
PY

echo "[${cond}|seed${seed}] done. outputs at ${OUT_ROOT}/"
