#!/usr/bin/env bash
# OPTION-A full-519 Stage B: extract SIGMMA tile embeddings on the 235 grid, one
# slide per array task (forward pass only -> inference queue). Builds a per-task
# one-row grid manifest (data_h5 -> the grid64 h5) so no shared-manifest race.
# --max_cells 0 = ALL tiles (the whole point of Option A = 100% coverage).
# Gate on the build array:
#   N=$(($(wc -l < configs/tcga_kirc_manifest.csv) - 1))
#   bsub -w "done(BUILDJOBID)" -J "kirc_grid_feat[1-$N]%24" < jobscripts/jobscript_tcga_kirc_grid_feat_array.sh
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/kirc_grid_feat.%J.%I.out
#BSUB -e logs/job_logs/kirc_grid_feat.%J.%I.err
#BSUB -q inference
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=24GB:num=1"
#BSUB -n 4
#BSUB -M 48GB
#BSUB -R "select[mem>48GB] rusage[mem=48GB] span[hosts=1]"
set -euo pipefail
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
config="$run_dir/configs/cell_niche_transfer_rcc_celltype_a01_step2000.yaml"
POETRY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"
MANIFEST="$run_dir/configs/tcga_kirc_manifest.csv"
grid_data="$cr/tcga_kirc/grid64_data_thr235"
grid_feat="$cr/tcga_kirc/grid64_sigmma_36852_step2000_thr235"
tmp_man="$cr/tcga_kirc/grid64_manifests_thr235"
mkdir -p "$grid_feat" "$tmp_man"

i=$((LSB_JOBINDEX + 1))
sample=$(sed -n "${i}p" "$MANIFEST" | cut -d, -f1 | tr -d '\r')
zarr=$(sed -n "${i}p" "$MANIFEST" | cut -d, -f3 | tr -d '\r')
griddata="$grid_data/$sample.h5"

if [ -z "$sample" ]; then echo "[skip] empty row $i"; exit 0; fi
if [ ! -f "$griddata" ]; then echo "[skip] $sample: no grid data (build skipped)"; exit 0; fi
if [ -f "$grid_feat/$sample.h5" ]; then echo "[skip] $sample features exist"; exit 0; fi

m="$tmp_man/$sample.csv"
printf 'sample_name,hematoxylin_eosin_image,data_h5\n%s,%s,%s\n' \
    "$sample" "$zarr" "$griddata" > "$m"

"$POETRY" -u scripts/tcga_tsr/tcga_extract_sigmma_features.py \
    --config "$config" --manifest "$m" --slide "$sample" \
    --out_dir "$grid_feat" --max_cells 0 --batch_size 64
echo "[grid-feat] DONE $sample"
