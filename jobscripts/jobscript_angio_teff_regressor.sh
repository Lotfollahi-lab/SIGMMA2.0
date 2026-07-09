#!/usr/bin/env bash
# Clinical-outcome path 2, step 2: linear-probe regression of angio/Teff signature
# scores from per-cell embeddings, SIGMMA vs frozen UNI2-h (the SIGMMA-specific-
# value milestone). Joins compute_signature_scores.py per-cell parquet to the RCC
# SIGMMA + UNI feature caches by cell_id. CPU sklearn; needs >8GB at max_cells=
# 8000 on the 4608d reprs -> normal queue (no ~8GB oversubscribed cap).
# See memory project_angio_teff_pipeline_design_2026_06_28.
#BSUB -J angio_teff_regressor
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/angio_teff_regressor.%J.out
#BSUB -e logs/job_logs/angio_teff_regressor.%J.err
#BSUB -q normal
#BSUB -n 4
#BSUB -M 16000
#BSUB -R "select[mem>16000] rusage[mem=16000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
sigmma_dir="$cr/sigmma/cell_niche_rcc_2026_06_19/36852_step2000"
uni_dir="$cr/uni/cell_niche_rcc_2026_06_19/uni2h_input224_p14"
score_dir="$run_dir/outputs/angio_teff/rcc_signature_scores"
out_dir="$run_dir/outputs/angio_teff/score_regression"

"$POETRY_PY" -u scripts/angio_teff/train_score_regressor.py \
    --sigmma_dir "$sigmma_dir" \
    --uni_dir "$uni_dir" \
    --score_dir "$score_dir" \
    --manifest "$run_dir/configs/cell_niche_evaluable_slides_rcc.csv" \
    --out_dir "$out_dir" \
    --max_cells 8000 --seed 0

echo "[done] angio/Teff score regression"
