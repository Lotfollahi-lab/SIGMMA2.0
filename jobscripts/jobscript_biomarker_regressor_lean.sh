#!/usr/bin/env bash
# Biomarker probe, step 2: RidgeCV regression of CA9/PAX8 per-cell expression from
# SIGMMA vs frozen UNI2-h embeddings (the SIGMMA-value test) + save a frozen
# sigmma_concat bundle for TCGA application. Single-gene sibling of
# jobscript_angio_teff_regressor.sh. Reports section-level held-out Pearson =
# the internal 'can H&E predict this biomarker?' feasibility answer.
# CPU sklearn; needs >8GB on 4608d uni_concat -> normal queue.
#BSUB -J biomarker_regressor_lean
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_regressor_lean.%J.out
#BSUB -e logs/job_logs/biomarker_regressor_lean.%J.err
#BSUB -q oversubscribed
#BSUB -n 4
#BSUB -M 7500
#BSUB -R "select[mem>7500] rusage[mem=7500] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
sigmma_dir="$cr/sigmma/cell_niche_rcc_2026_06_19/36852_step2000"
uni_dir="$cr/uni/cell_niche_rcc_2026_06_19/uni2h_input224_p14"
score_dir="$run_dir/outputs/biomarker/rcc_gene_targets"
out_dir="$run_dir/outputs/biomarker/gene_regression_lean"

"$POETRY_PY" -u scripts/biomarker/train_gene_regressor.py \
    --sigmma_dir "$sigmma_dir" \
    --uni_dir "$uni_dir" \
    --score_dir "$score_dir" \
    --manifest "$run_dir/configs/cell_niche_evaluable_slides_rcc.csv" \
    --targets CA9 PAX8 \
    --bundle_rep sigmma_concat \
    --out_dir "$out_dir" \
    --max_cells 1500 --min_feat_cells 100 --min_eval_cells 50 --seed 0

echo "[done] biomarker gene regression"
