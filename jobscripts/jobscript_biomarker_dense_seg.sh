#!/usr/bin/env bash
# Dense whole-slide re-segmentation of ONE TCGA-KIRC slide (TCGA-BP-5174) for a
# dense predicted-PAX8 spatial map: ALL foreground tiles (vs the 200-tile sample
# used for the clinical scalar), so nuclei cover the whole tissue. Separate
# dense/ output so the clinical caches are untouched. GPU (Cellpose, tcga_seg).
#BSUB -J biomarker_dense_seg
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_dense_seg.%J.out
#BSUB -e logs/job_logs/biomarker_dense_seg.%J.err
#BSUB -q inference
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=8GB:num=1"
#BSUB -n 4
#BSUB -M 32GB
#BSUB -R "select[mem>32GB] rusage[mem=32GB] span[hosts=1]"
set -e
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1
SPY=/lustre/scratch126/cellgen/lotfollahi/dj16/envs/tcga_seg/bin/python

sample="TCGA-BP-5174-01Z-00-DX1.d6f8427f-fbbe-4582-a020-ede5675a7e76"
zarr="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/kirc_slides/17ac296c-a51d-4e6b-a656-c82bc1178ae9/${sample}.he.zarr"
data_h5="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_kirc/dense/data/${sample}.h5"
mkdir -p "$(dirname "$data_h5")"

echo "[dense-seg] $sample (all foreground tiles)"
"$SPY" "$run_dir/scripts/tcga_tsr/tcga_segment_and_build_data.py" \
    --zarr "$zarr" --sample "$sample" --out_h5 "$data_h5" \
    --max_tiles 100000 --max_cells 2000000
echo "[done] dense seg $sample"
