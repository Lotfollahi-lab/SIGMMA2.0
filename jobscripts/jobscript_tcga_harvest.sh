#!/usr/bin/env bash
# Stage C+D: apply the frozen probe -> per-slide TSR, then TSR high/low survival.
# CPU only. Gate on the feature array:
#   bsub -w "ended(FEATJOBID)" < jobscripts/jobscript_tcga_harvest.sh
#BSUB -J tcga-harvest
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga-harvest.%J.out
#BSUB -e logs/job_logs/tcga-harvest.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1

run_dir="/nfs/team361/dj16/projects/SIGMMA"
feat_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_read/sigmma_36852_step2000"
bundle="$run_dir/outputs/cell_niche_rcc/51140/models/level_2_cell_type__z_sigmma_concat.joblib"
surv_tsv="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-READ_slide_survival.tsv"
out_dir="$run_dir/outputs/tcga_read_tsr"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "$run_dir"

echo "[harvest] predict TSR"
python -u scripts/tcga_tsr/tcga_predict_tsr.py \
    --features_dir "$feat_dir" --bundle "$bundle" \
    --out_csv "$out_dir/tsr_per_slide.csv" \
    --per_cell_dir "$out_dir/per_cell_pred"

echo "[harvest] survival"
python -u scripts/tcga_tsr/tcga_survival.py \
    --tsr_csv "$out_dir/tsr_per_slide.csv" \
    --survival_tsv "$surv_tsv" \
    --out_dir "$out_dir" \
    --fig_dir "$run_dir/figures/clinical_outcome"
echo "[harvest] DONE -> $out_dir"
