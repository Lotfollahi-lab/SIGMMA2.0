#!/usr/bin/env bash
# COAD Stage C+D: apply the 3-class tumour/stroma/other probe -> per-slide TSR,
# then TSR high/low survival (Kaplan-Meier + Cox). CPU only.
#   bsub -w "ended(FEATJOBID)" < jobscripts/jobscript_tcga_coad_harvest.sh
#BSUB -J tcga-coad-harvest
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga-coad-harvest.%J.out
#BSUB -e logs/job_logs/tcga-coad-harvest.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=1GB:num=1"
#BSUB -n 2
#BSUB -M 16000
#BSUB -R "select[mem>16000] rusage[mem=16000] span[hosts=1]"
set -e
export PYTHONUNBUFFERED=1

run_dir="/nfs/team361/dj16/projects/SIGMMA"
feat_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_coad/sigmma_36852_step2000"
bundle="$run_dir/outputs/cell_niche_rcc/tumour_stroma_other_probe/z_sigmma_concat__tso.joblib"
surv_tsv="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-COAD_slide_survival.tsv"
out_dir="$run_dir/outputs/tcga_coad_tsr"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "$run_dir"

echo "[harvest] predict TSR (3-class tumour/stroma/other)"
python -u scripts/tcga_tsr/tcga_predict_tsr.py \
    --features_dir "$feat_dir" --bundle "$bundle" \
    --out_csv "$out_dir/tsr_per_slide.csv" \
    --per_cell_dir "$out_dir/per_cell_pred"

echo "[harvest] survival"
python -u scripts/tcga_tsr/tcga_survival.py \
    --tsr_csv "$out_dir/tsr_per_slide.csv" \
    --survival_tsv "$surv_tsv" \
    --cohort COAD \
    --out_dir "$out_dir" \
    --fig_dir "$run_dir/figures/clinical_outcome/tso_coad"
echo "[harvest] DONE -> $out_dir"
