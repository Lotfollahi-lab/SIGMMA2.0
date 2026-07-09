#!/usr/bin/env bash
# Clinical-outcome path 2 (angio/Teff). Compute per-cell IMmotion150/151
# angiogenesis & T-effector signature scores from the in-house RCC Xenium, to be
# used as regression targets for an H&E -> molecular-score probe (z_sigmma_concat
# -> score). See memory project_angio_teff_pipeline_design_2026_06_28.
#
# CPU only (scanpy read + score_genes), forward-data-only -> oversubscribed.
# 20 RCC evaluable sections processed sequentially; each section's full 5K panel
# measures all 11 signature genes (the 2,979-gene intersection drops Teff, but
# that gates SIGMMA's training, not this real-ST target).
#BSUB -J angio_teff_scores
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/angio_teff_scores.%J.out
#BSUB -e logs/job_logs/angio_teff_scores.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 7500
#BSUB -R "select[mem>7500] rusage[mem=7500] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1

POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

manifest="$run_dir/configs/cell_niche_evaluable_slides_rcc.csv"
out_dir="$run_dir/outputs/angio_teff/rcc_signature_scores"
mkdir -p "$out_dir"

# fail fast if scanpy is unavailable in this venv
"$POETRY_PY" -c "import scanpy, pyarrow; print('scanpy', scanpy.__version__)"

echo "[run] compute angio/Teff signature scores -> $out_dir"
"$POETRY_PY" -u scripts/angio_teff/compute_signature_scores.py \
    --manifest "$manifest" \
    --out_dir "$out_dir" \
    --target_sum 1e4 --ctrl_size 50 --seed 0

echo "[done] angio/Teff signature scores"
