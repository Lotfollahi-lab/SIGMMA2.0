#!/usr/bin/env bash
# OPTION-A full-519 Stage A: build the 64um tissue GRID per KIRC slide at the
# FULL-COVERAGE foreground setting (white_thr=235 recovers pale clear-cell tumour
# the 220 gate drops). One slide per array task, skip-existing. CPU + zarr.
#   N=$(($(wc -l < configs/tcga_kirc_manifest.csv) - 1))
#   bsub -J "kirc_grid_build[1-$N]%40" < jobscripts/jobscript_tcga_kirc_grid_build_array.sh
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/kirc_grid_build.%J.%I.out
#BSUB -e logs/job_logs/kirc_grid_build.%J.%I.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
export PYTHONUNBUFFERED=1
VENV_PY="$run_dir/.venv/bin/python"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
MANIFEST="$run_dir/configs/tcga_kirc_manifest.csv"
grid_data="$cr/tcga_kirc/grid64_data_thr235"
mkdir -p "$grid_data"

i=$((LSB_JOBINDEX + 1))
sample=$(sed -n "${i}p" "$MANIFEST" | cut -d, -f1 | tr -d '\r')
zarr=$(sed -n "${i}p" "$MANIFEST" | cut -d, -f3 | tr -d '\r')
out="$grid_data/$sample.h5"

if [ -z "$sample" ]; then echo "[skip] empty row $i"; exit 0; fi
if [ -f "$out" ]; then echo "[skip] $sample grid exists"; exit 0; fi
if [ ! -e "$zarr" ]; then echo "[skip] $sample: no zarr $zarr"; exit 0; fi

"$VENV_PY" -u scripts/tcga_tsr/tcga_build_grid_data.py \
    --zarr "$zarr" --sample "$sample" --out_h5 "$out" \
    --tile_um 64 --um_per_px 0.5 --white_thr 235 --min_frac 0.5
echo "[grid-build] DONE $sample"
