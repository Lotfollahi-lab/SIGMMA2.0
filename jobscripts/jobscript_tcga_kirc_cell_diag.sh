#!/usr/bin/env bash
# Diagnose sparse prediction dots: count cells at each pipeline stage (segmented
# nuclei -> SIGMMA features -> predictions -> matched) + density, for the 3
# central05 overlay slides. Pinpoints segmentation vs feature-extraction loss.
#BSUB -J tcga_kirc_cell_diag
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_kirc_cell_diag.%J.out
#BSUB -e logs/job_logs/tcga_kirc_cell_diag.%J.err
#BSUB -q oversubscribed
#BSUB -n 1
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
VENV_PY="$run_dir/.venv/bin/python"
export PYTHONUNBUFFERED=1
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"

"$VENV_PY" -u scripts/tcga_tsr/tcga_cell_count_diag.py \
    --features_dir "$cr/tcga_kirc/sigmma_36852_step2000" \
    --per_cell_dir "$run_dir/outputs/tcga_kirc_tsr/per_cell_pred" \
    --manifest "$run_dir/configs/tcga_kirc_manifest.csv" \
    --samples \
      TCGA-CJ-4889-01Z-00-DX1.A584930A-586C-4CD9-980E-FB0240AD5D27 \
      TCGA-CZ-4860-01Z-00-DX1.4e1e4ab7-d82d-437e-a403-5e4cc2647751 \
      TCGA-BP-4795-01Z-00-DX1.29d32315-efd2-495c-85b5-86db265cfc9c
echo "[cell-diag] DONE"
