#!/usr/bin/env bash
# Rebuild the FULL 519-slide KIRC manifest from the converted *.he.zarr dirs
# (svs deleted post-conversion -> enumerate zarrs, not svs). 4-col output
# matches the seg/feature/harvest jobscripts (sample,svs,zarr,data_h5). data_dir
# points at the existing seg cache so the 115 already-done slides are reused
# (skip-existing in seg + features).
#BSUB -J kirc_build_manifest
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/kirc_build_manifest.%J.out
#BSUB -e logs/job_logs/kirc_build_manifest.%J.err
#BSUB -q oversubscribed
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "select[mem>4000] rusage[mem=4000] span[hosts=1]"
set -e
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
VENV_PY="$run_dir/.venv/bin/python"
export PYTHONUNBUFFERED=1

slides_root="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/kirc_slides"
data_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_kirc/data_2026_06_23"
out_csv="$run_dir/configs/tcga_kirc_manifest.csv"

# keep a copy of the 115-slide preview manifest before overwriting
cp "$out_csv" "$run_dir/configs/tcga_kirc_manifest_preview115.csv"

"$VENV_PY" -u scripts/tcga_tsr/build_tcga_manifest.py \
    --slides_root "$slides_root" --data_dir "$data_dir" --out_csv "$out_csv"

echo "[manifest] rows: $(( $(wc -l < "$out_csv") - 1 ))"
