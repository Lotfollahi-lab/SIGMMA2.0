#!/usr/bin/env bash
# Existence check for SIGMMA's core value prop (memory project_sigmma_core_value_prop):
# mine H&E crops that are morphologically near-identical (high grayscale SSIM) yet
# carry OPPOSITE molecular identity (Tumour epithelium vs Nephron epithelium in RCC,
# level_2_cell_type). NO model — pure H&E + ST data mining to establish the premise
# that a molecularly-defined compartment can be morphologically subtle.
#
# CPU only (zarr crop reads + scanpy + skimage SSIM matmul) -> oversubscribed queue.
# SMOKE=1 (default) restricts to 2 sections with small caps to validate end-to-end;
# set SMOKE=0 for the full 20-section RCC sweep.
#BSUB -J morph_doppel
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/morph_doppel.%J.out
#BSUB -e logs/job_logs/morph_doppel.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 7500
#BSUB -R "select[mem>7500] rusage[mem=7500] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1

POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

CONFIG="${CONFIG:-configs/cell_niche_transfer_rcc_celltype_a01_step2000.yaml}"
CLASS_A="${CLASS_A:-Tumour epithelium}"
CLASS_B="${CLASS_B:-Nephron epithelium}"
LABEL_NAME="${LABEL_NAME:-level_2_cell_type}"
CROP_PX="${CROP_PX:-128}"
TOP_K="${TOP_K:-40}"
SMOKE="${SMOKE:-1}"
MODE="${MODE:-mine}"   # mine=render pairs | dist=cross-vs-within SSIM distributions
TAG="${TAG:-rcc_tumour_vs_nephron}"

# fail fast if deps missing in this venv
"$POETRY_PY" -c "import scanpy, skimage, zarr; print('scanpy', scanpy.__version__, '| skimage', skimage.__version__)"

extra_args=()
if [ "$SMOKE" = "1" ]; then
  extra_args+=(--sections "CV1-KID-0-FO-1,CV2-KID-0-FO-1"
               --max_per_class_per_section 120
               --shortlist 2000 --tag "${TAG}_smoke")
  echo "[run] SMOKE morph-doppelganger mining (2 sections)"
else
  extra_args+=(--max_per_class_per_section 150 --shortlist 4000 --tag "$TAG")
  echo "[run] FULL morph-doppelganger mining (all RCC sections)"
fi

"$POETRY_PY" -u scripts/mine_morph_doppelgangers.py \
    --config "$CONFIG" \
    --mode "$MODE" \
    --label_name "$LABEL_NAME" \
    --class_a "$CLASS_A" \
    --class_b "$CLASS_B" \
    --crop_px "$CROP_PX" \
    --top_k "$TOP_K" \
    "${extra_args[@]}"

echo "[done] morph-doppelganger mining"
