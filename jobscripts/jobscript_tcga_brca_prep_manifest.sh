#!/usr/bin/env bash
# BRCA Stage 0: build the full manifest from converted *.he.zarr, then carve a
# deterministic PREVIEW subset (first 100 slides) for a sanity-check run before
# committing full-cohort (1133-slide) feature extraction.
#   bsub < jobscripts/jobscript_tcga_brca_prep_manifest.sh
#BSUB -J tcga_brca_prep
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -q inference
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=1GB:num=1"
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "select[mem>4000] rusage[mem=4000] span[hosts=1]"
#BSUB -o logs/job_logs/tcga_brca_prep.%J.out
#BSUB -e logs/job_logs/tcga_brca_prep.%J.err
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"

slides_root="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/brca_slides"
data_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_brca/data_2026_07_01"
full_csv="$run_dir/configs/tcga_brca_manifest.csv"
prev_csv="$run_dir/configs/tcga_brca_manifest_preview100.csv"
mkdir -p "$data_dir"

echo "[prep] build full BRCA manifest"
"$POETRY_PY" -u scripts/tcga_tsr/build_tcga_manifest.py \
    --slides_root "$slides_root" --data_dir "$data_dir" --out_csv "$full_csv"

echo "[prep] carve preview subset (first 100 slides + header)"
head -n 101 "$full_csv" > "$prev_csv"
echo "[prep] preview rows (incl header): $(wc -l < "$prev_csv")  -> $prev_csv"
