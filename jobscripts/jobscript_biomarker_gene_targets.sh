#!/usr/bin/env bash
# Biomarker probe, step 1: compute per-cell CA9 (CA-IX) & PAX8 expression targets
# from the in-house RCC Xenium, to serve as regression targets for an H&E ->
# biomarker probe (z_sigmma_concat -> gene expression). Single-gene sibling of
# jobscript_angio_teff_scores.sh. VIM excluded (not on the RCC Xenium panel).
#
# CPU only (scanpy read), forward-data-only -> oversubscribed. 20 RCC evaluable
# sections; each section's 5K panel measures CA9 (19/20) and PAX8 (20/20).
#BSUB -J biomarker_gene_targets
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_gene_targets.%J.out
#BSUB -e logs/job_logs/biomarker_gene_targets.%J.err
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
out_dir="$run_dir/outputs/biomarker/rcc_gene_targets"
mkdir -p "$out_dir"

"$POETRY_PY" -c "import scanpy, pyarrow; print('scanpy', scanpy.__version__)"

echo "[run] compute CA9/PAX8 per-cell targets -> $out_dir"
"$POETRY_PY" -u scripts/biomarker/compute_gene_targets.py \
    --manifest "$manifest" \
    --genes CA9 PAX8 PAX2 \
    --out_dir "$out_dir" \
    --target_sum 1e4

echo "[done] biomarker gene targets"
