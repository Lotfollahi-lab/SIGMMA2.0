#!/usr/bin/env bash
# Morphological-doppelganger mining at 256px = 128um crops (macro scale: tissue
# architecture, closer to what a pathologist reads). RCC Tumour vs Nephron
# epithelium, all 20 sections. CPU only, self-contained (farm22 env-passing
# unreliable, see memory feedback_lsf_env_passing).
#BSUB -J morph_doppel256
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/morph_doppel256.%J.out
#BSUB -e logs/job_logs/morph_doppel256.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

"$POETRY_PY" -c "import scanpy, skimage, zarr; print('ok')"

echo "[run] morph-doppelganger mining @256px (128um), all RCC sections"
"$POETRY_PY" -u scripts/mine_morph_doppelgangers.py \
    --config configs/cell_niche_transfer_rcc_celltype_a01_step2000.yaml \
    --mode mine \
    --label_name level_2_cell_type \
    --class_a "Tumour epithelium" \
    --class_b "Nephron epithelium" \
    --crop_px 256 \
    --max_per_class_per_section 150 \
    --shortlist 4000 \
    --top_k 40 \
    --tag rcc_tumour_vs_nephron_crop256

echo "[done] morph-doppelganger mining @256px"
