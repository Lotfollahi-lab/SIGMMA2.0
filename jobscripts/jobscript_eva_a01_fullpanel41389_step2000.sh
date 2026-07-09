#!/bin/bash
#BSUB -J eva-a01-fp41389[1-5]
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/eva-a01-fp41389.%J.%I.out
#BSUB -e logs/job_logs/eva-a01-fp41389.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 8
#BSUB -M 320GB
#BSUB -R "select[mem>320GB] rusage[mem=320GB] span[hosts=1]"

# Re-run ONLY the SIGMMA condition of the EVA MoNuSAC benchmark on the
# NEW full-panel checkpoint:
#   sigmma_dense_attnpool_gex_fullpanel_a01/41389_1/checkpoint-step2000.pth
#
# Baselines (uni2h / bleep / clip / plip / omiclip) are checkpoint-independent
# and already cached under outputs/eva_step2000/<cond>/seed<seed>/ — NOT re-run.
# 5 seeds (42/2/6/14/21) to match the baseline multiseed methodology.
#
# Output: outputs/eva_step2000/a01_fullpanel41389_step2000/seed<seed>/
#           sigmma_shared_lora/convdec_image/test_artifacts.npz
#         + instance_metrics_{global,r1_restricted}.csv
# The pre-existing outputs/eva_step2000/a01/ (svg-36852 step2000) is preserved.

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
CFG="${run_dir}/configs/cell_niche_transfer_dense_attnpool_a01_step2000.yaml"

seeds=(42 2 6 14 21)
seed="${seeds[$((LSB_JOBINDEX - 1))]}"

OUT_ROOT="${run_dir}/outputs/eva_step2000/a01_fullpanel41389_step2000/seed${seed}"
mkdir -p "${OUT_ROOT}"
cp "${CFG}" "${OUT_ROOT}/" 2>/dev/null || true

art_subdir="sigmma_shared_lora"
ART="${OUT_ROOT}/${art_subdir}/convdec_image/test_artifacts.npz"

echo "=========================================="
echo "[${LSB_JOBINDEX}/5] EVA a01 fullpanel-41389 step2000  seed=${seed}  -> ${OUT_ROOT}"
echo "  ckpt: $(grep checkpoint_path "${CFG}")"
echo "=========================================="

if [ -f "${ART}" ]; then
    echo "[a01-fp|seed${seed}] reusing existing test_artifacts.npz (skipping train_decoder)"
else
    python -m scripts.eva_critique.train_decoder \
        --sigmma_config "${CFG}" --model sigmma \
        --decoder convdec_image --data_root "${DATA}" --out_root "${OUT_ROOT}" \
        --max_steps 2000 --batch_size 32 --lr 0.002 --seed "${seed}"
fi

echo "[a01-fp|seed${seed}] instance metrics: GLOBAL + R1-restricted"
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
    print(f"\n{label.upper()} (a01-fp seed${seed}, tau=0.5, n={len(df)}):")
    print(df[['AJI','PQ','SQ','DQ','DetF1_07']].mean().round(4).to_string())
    out = out_root / f"instance_metrics_{label}.csv"
    df.to_csv(out, index=False)
    print(f"  saved {out}")
PY

echo "[a01-fp|seed${seed}] done. outputs at ${OUT_ROOT}/"
