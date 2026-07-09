#!/usr/bin/env bash
# Rigorous companion to jobscript_mine_morph_doppelgangers.sh: answer "is SSIM~0.3
# actually 'similar'?" by comparing the CROSS-type (Tumour vs Nephron) windowed-SSIM
# distribution to the WITHIN-type distributions (same min_sep constraint applied to
# both). Overlap => tumour/nephron epithelium are morphologically inseparable by
# SSIM = the compartment is molecularly-defined but morphologically subtle.
# Full 20-section RCC sweep, CPU only. Self-contained (farm22 env-passing is
# unreliable, see memory feedback_lsf_env_passing).
#BSUB -J morph_doppel_dist
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/morph_doppel_dist.%J.out
#BSUB -e logs/job_logs/morph_doppel_dist.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 7500
#BSUB -R "select[mem>7500] rusage[mem=7500] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

"$POETRY_PY" -c "import scanpy, skimage, zarr; print('ok')"

echo "[run] cross-vs-within SSIM distributions (all RCC sections)"
"$POETRY_PY" -u scripts/mine_morph_doppelgangers.py \
    --config configs/cell_niche_transfer_rcc_celltype_a01_step2000.yaml \
    --mode dist \
    --label_name level_2_cell_type \
    --class_a "Tumour epithelium" \
    --class_b "Nephron epithelium" \
    --crop_px 128 \
    --max_per_class_per_section 150 \
    --n_pairs_per_group 8000 \
    --tag rcc_tumour_vs_nephron

echo "[done] SSIM distribution diagnostic"
