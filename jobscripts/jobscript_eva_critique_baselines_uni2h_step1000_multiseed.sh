#!/bin/bash
#BSUB -J eva-baselines-uni2h-step1000-multiseed[1-20]
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/eva-baselines-uni2h-step1000-multiseed.%J.%I.out
#BSUB -e logs/job_logs/eva-baselines-uni2h-step1000-multiseed.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 8
#BSUB -M 320GB
#BSUB -R "select[mem>320GB] rusage[mem=320GB] span[hosts=1]"

# Multi-seed EVA MoNuSAC baseline-FT benchmark on UNI2-h: 4 models x 5 seeds = 20 tasks.
# Models: BLEEP, CLIP, PLIP, OmiCLIP — all UNI2-h init + finetuned at step1000.
# Seeds: 42 (already trained, will skip via guard), 2, 6, 14, 21.
# Output layout: ${OUT_ROOT}/${model}/seed${seed}/{${model}/convdec_image/test_artifacts.npz,instance_metrics_*.csv}
# seed=42 artifacts were migrated into seed42/ subdirs on 2026-05-19 prior to submitting this job.

set -e

module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate $ENV_PATH"
cd "${run_dir}"

CFG="${run_dir}/configs/image_pathology_segmentation_per_scale_sigmma.yaml"
DATA="${run_dir}/data"

models=("bleep_uni_ft" "clip_uni_ft" "plip_uni_ft" "omiclip_uni_ft")
baseline_cfgs=(
    "in-house_eva_bleep_uni_step1000.yaml"
    "in-house_eva_clip_uni_step1000.yaml"
    "in-house_eva_plip_uni_step1000.yaml"
    "in-house_eva_omiclip_uni_step1000.yaml"
)
seeds=(42 2 6 14 21)

idx=$((LSB_JOBINDEX - 1))
model="${models[$((idx / 5))]}"
bcfg="${run_dir}/configs/${baseline_cfgs[$((idx / 5))]}"
seed="${seeds[$((idx % 5))]}"

OUT_ROOT="${run_dir}/outputs/eva_critique_baselines_uni2h_step1000/${model}/seed${seed}"
mkdir -p "${OUT_ROOT}"

echo "=========================================="
echo "[${LSB_JOBINDEX}/20] eva_critique model=${model} seed=${seed}"
echo "Baseline cfg: ${bcfg}"
echo "Out: ${OUT_ROOT}"
echo "=========================================="

ART="${OUT_ROOT}/${model}/convdec_image/test_artifacts.npz"
if [ -f "${ART}" ]; then
    echo "[${model}|seed${seed}] reusing existing test_artifacts.npz (skipping train_decoder)"
else
    echo "[${model}|seed${seed}] train_decoder: model=${model} dec=convdec_image"
    python -m scripts.eva_critique.train_decoder \
        --sigmma_config "${CFG}" \
        --model "${model}" --decoder convdec_image \
        --baseline_config_path "${bcfg}" \
        --data_root "${DATA}" --out_root "${OUT_ROOT}" \
        --max_steps 2000 --batch_size 32 --lr 0.002 --seed ${seed}
fi

echo "[${model}|seed${seed}] instance metrics: GLOBAL + R1-restricted"
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
    rows_g.append({**per_fov(p, g),                                "image_id": str(ids[i])})
    R1 = binary_dilation(g.astype(bool), footprint=disk(8))
    rows_r1.append({**per_fov(p, g, R1),                           "image_id": str(ids[i])})

out_root = Path("${OUT_ROOT}")
for label, rows in [("global", rows_g), ("r1_restricted", rows_r1)]:
    df = pd.DataFrame(rows)
    print(f"\n{label.upper()} (${model} seed${seed}, tau=0.5, n={len(df)}):")
    print(df[['AJI','PQ','SQ','DQ','DetF1_07']].mean().round(4).to_string())
    out = out_root / f"instance_metrics_{label}.csv"
    df.to_csv(out, index=False)
    print(f"  saved {out}")
PY

echo "[${model}|seed${seed}] done. outputs at ${OUT_ROOT}/"
