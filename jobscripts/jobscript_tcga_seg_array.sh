#!/usr/bin/env bash
# Stage A: nucleus segmentation -> data_h5 for TCGA-READ slides (job array).
# One array task per slide (indexed into the manifest). Writes data_h5 with the
# process_slide_data schema (no labels). Uses the tcga_seg venv (cellpose).
#
# Submit AFTER the manifest is built:
#   N=$(($(wc -l < $MANIFEST) - 1))
#   bsub -J "tcga_seg[1-$N]%32" < jobscripts/jobscript_tcga_seg_array.sh
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_seg.%J.%I.out
#BSUB -e logs/job_logs/tcga_seg.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=8GB:num=1"
#BSUB -n 4
#BSUB -M 32GB
#BSUB -R "select[mem>32GB] rusage[mem=32GB] span[hosts=1]"
set -e

run_dir="/nfs/team361/dj16/projects/SIGMMA"
VENV=/lustre/scratch126/cellgen/lotfollahi/dj16/envs/tcga_seg/bin/python
MANIFEST="${MANIFEST:-$run_dir/configs/tcga_read_manifest.csv}"
export PYTHONUNBUFFERED=1
cd "$run_dir"

i=$((LSB_JOBINDEX + 1))                       # +1 to skip CSV header
line=$(sed -n "${i}p" "$MANIFEST")
sample=$(echo "$line" | cut -d, -f1)
zarr=$(echo "$line"   | cut -d, -f2)
data_h5=$(echo "$line" | cut -d, -f3)

if [ -f "$data_h5" ]; then echo "[skip] $sample data_h5 exists"; exit 0; fi
echo "[seg] task $LSB_JOBINDEX  sample=$sample"
$VENV "$run_dir/scripts/tcga_tsr/tcga_segment_and_build_data.py" \
    --zarr "$zarr" --sample "$sample" --out_h5 "$data_h5" \
    --max_cells 30000 --max_tiles 200
echo "[seg] done $sample"
