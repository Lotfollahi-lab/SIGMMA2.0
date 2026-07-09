#!/usr/bin/env bash
# COAD TSR with the PMC7887673 ("tutorial") stroma definition, end-to-end:
#   1. retrain the tumour/stroma/other probe with --grouping tutorial
#      (lymphocytic infiltration -> STROMA; Perivascular/Nephron -> other),
#   2. re-predict per-slide TSR on the cached COAD SIGMMA features,
#   3. survival with a FIXED 50% stroma-high cutoff (age-only Cox, no stage).
# Same z_sigmma_concat / ckpt 36852 step2000 space as the classic probe. CPU.
#   bsub < jobscripts/jobscript_tcga_coad_tutorial_tsr.sh
#BSUB -J tcga-coad-tut-tsr
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga-coad-tut-tsr.%J.out
#BSUB -e logs/job_logs/tcga-coad-tut-tsr.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=1GB:num=1"
#BSUB -n 8
#BSUB -M 64000
#BSUB -R "select[mem>64000] rusage[mem=64000] span[hosts=1]"
set -e
export PYTHONUNBUFFERED=1

run_dir="/nfs/team361/dj16/projects/SIGMMA"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
feat_dir="$cr/tcga_coad/sigmma_36852_step2000"
bundle="$run_dir/outputs/cell_niche_rcc/tumour_stroma_other_probe/z_sigmma_concat__tso_tutorial.joblib"
surv_tsv="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-COAD_slide_survival.tsv"
out_dir="$run_dir/outputs/tcga_coad_tsr_tutorial"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "$run_dir"

echo "[1/3] retrain probe (--grouping tutorial)"
python -u scripts/tcga_tsr/train_tumour_stroma_probe.py \
    --data_dir   "$cr/data/cell_niche_rcc_2026_06_19/scales_64_128_256" \
    --sigmma_dir "$cr/sigmma/cell_niche_rcc_2026_06_19/36852_step2000" \
    --manifest   configs/cell_niche_evaluable_slides_rcc.csv \
    --grouping   tutorial \
    --out_bundle "$bundle"

echo "[2/3] predict TSR on COAD"
python -u scripts/tcga_tsr/tcga_predict_tsr.py \
    --features_dir "$feat_dir" --bundle "$bundle" \
    --out_csv "$out_dir/tsr_per_slide.csv" \
    --per_cell_dir "$out_dir/per_cell_pred"

echo "[3/3] survival (fixed 50% stroma-high cutoff)"
python -u scripts/tcga_tsr/tcga_survival.py \
    --tsr_csv "$out_dir/tsr_per_slide.csv" \
    --survival_tsv "$surv_tsv" \
    --cohort COAD \
    --stroma_cutoff 0.5 \
    --out_dir "$out_dir" \
    --fig_dir "$run_dir/figures/clinical_outcome/tso_coad_tutorial"
echo "[tutorial-tsr] DONE -> $out_dir"
