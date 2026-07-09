#!/usr/bin/env bash
# KIRC Stage A (array): svs->zarr (if needed) + nucleus segmentation -> data_h5,
# one array task per KIRC slide. GPU queue. Mirrors the COAD seg array.
#   N=$(($(wc -l < configs/tcga_kirc_manifest.csv) - 1))
#   bsub -J "tcga_kirc_cseg[1-$N]%24" < jobscripts/jobscript_tcga_kirc_convert_seg_array.sh
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_kirc_cseg.%J.%I.out
#BSUB -e logs/job_logs/tcga_kirc_cseg.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=8GB:num=1"
#BSUB -n 4
#BSUB -M 32GB
#BSUB -R "select[mem>32GB] rusage[mem=32GB] span[hosts=1]"
set -e

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ZPY=/software/cellgen/team361/dj16/envs/zarr_v3/bin/python
SPY=/lustre/scratch126/cellgen/lotfollahi/dj16/envs/tcga_seg/bin/python
CONVERT=/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/svs_to_zarr.py
MANIFEST="$run_dir/configs/tcga_kirc_manifest.csv"
export PYTHONUNBUFFERED=1
cd "$run_dir"

i=$((LSB_JOBINDEX + 1))                       # +1 skips CSV header
line=$(sed -n "${i}p" "$MANIFEST")
sample=$(echo "$line" | cut -d, -f1 | tr -d '\r')
svs=$(echo "$line"     | cut -d, -f2 | tr -d '\r')
zarr=$(echo "$line"    | cut -d, -f3 | tr -d '\r')
data_h5=$(echo "$line" | cut -d, -f4 | tr -d '\r')

if [ -f "$data_h5" ]; then echo "[skip] $sample data_h5 exists"; exit 0; fi
if [ -f "${zarr}.no_mpp" ]; then echo "[skip] $sample no mpp metadata"; exit 0; fi

# svs is only needed to (re)build the zarr; if the zarr already exists (svs may
# have been deleted post-conversion) go straight to segmentation.
if [ ! -f "$zarr/zarr.json" ]; then
  if [ ! -f "$svs" ]; then echo "[skip] $sample svs not present and no zarr (still downloading?)"; exit 0; fi
  echo "[convert] $sample"; rm -rf "$zarr"
  "$ZPY" "$CONVERT" "$svs" "$zarr" --workers 4 --overwrite
fi
if [ ! -f "$zarr/zarr.json" ]; then echo "[skip] $sample zarr not produced"; exit 0; fi

echo "[seg] $sample"
mkdir -p "$(dirname "$data_h5")"
"$SPY" "$run_dir/scripts/tcga_tsr/tcga_segment_and_build_data.py" \
    --zarr "$zarr" --sample "$sample" --out_h5 "$data_h5" \
    --max_cells 30000 --max_tiles 200
echo "[done] $sample"
