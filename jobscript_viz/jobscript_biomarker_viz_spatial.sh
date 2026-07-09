#!/usr/bin/env bash
# Biomarker probe viz: TCGA-KIRC H&E next to SIGMMA-predicted 2D PAX8 spatial map
# (frozen in-house RCC probe on H&E-segmented nuclei). CPU (zarr strip-read +
# matplotlib) -> oversubscribed.
#BSUB -J biomarker_viz_spatial
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_viz_spatial.%J.out
#BSUB -e logs/job_logs/biomarker_viz_spatial.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 7500
#BSUB -R "select[mem>7500] rusage[mem=7500] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

feat_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_kirc/sigmma_36852_step2000"
bundle="$run_dir/outputs/biomarker/gene_regression/bundles/sigmma_concat.joblib"

# LOW / MID / HIGH predicted-PAX8 slides (from tcga_kirc_biomarker_slide.csv)
"$POETRY_PY" -u scripts/biomarker/viz_tcga_gene_spatial.py \
    --bundle "$bundle" \
    --manifest "$run_dir/configs/tcga_kirc_manifest.csv" \
    --feat_dir "$feat_dir" \
    --slide_csv "$run_dir/outputs/biomarker/tcga_predictions/tcga_kirc_biomarker_slide.csv" \
    --gene PAX8 --bin_px 6 --min_bin 2 \
    --samples \
      TCGA-AS-3778-01Z-00-DX1.E44DEEAC-DDAC-476A-A8EB-A7E3689B4121 \
      TCGA-BP-5174-01Z-00-DX1.d6f8427f-fbbe-4582-a020-ede5675a7e76 \
      TCGA-BP-4988-01Z-00-DX1.b037f17b-8fc7-46b2-adae-8e1140606fb7 \
    --out "$run_dir/figures/clinical_outcome/biomarker/spatial/tcga_kirc_pax8_spatial.png"
echo "[done] PAX8 spatial viz"
