#!/usr/bin/env bash
# Biomarker probe, step 3: apply the frozen RCC-trained CA9/PAX8 Ridge probe to
# the 519 cached TCGA-KIRC SIGMMA feature HDF5s -> per-slide + per-patient
# predicted biomarker expression. Forward-only CPU sklearn -> oversubscribed.
#BSUB -J biomarker_predict_tcga
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_predict_tcga.%J.out
#BSUB -e logs/job_logs/biomarker_predict_tcga.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

feat_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_kirc/sigmma_36852_step2000"
bundle="$run_dir/outputs/biomarker/gene_regression/bundles/sigmma_concat.joblib"
out="$run_dir/outputs/biomarker/tcga_predictions"

"$POETRY_PY" -u scripts/biomarker/predict_tcga_gene.py \
    --features_dir "$feat_dir" \
    --bundle "$bundle" \
    --out_slide_csv "$out/tcga_kirc_biomarker_slide.csv" \
    --out_patient_csv "$out/tcga_kirc_biomarker_patient.csv" \
    --q_hotspot 0.90
echo "[done] TCGA biomarker prediction"
