#!/usr/bin/env bash
# OPTION-A full-519 Stage C: tile-level MAX-hotspot TSR over all KIRC slides
# (8-direction surround validity: 4 side strips >= surround_min tumour tiles AND
# 4 corner blocks >= corner_min) then survival on OS/DSS/PFI x {fixed05, median}.
# Gate on the feature array:
#   bsub -w "ended(FEATJOBID)" < jobscripts/jobscript_tcga_kirc_grid_harvest.sh
#BSUB -J kirc_grid_harvest
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/kirc_grid_harvest.%J.out
#BSUB -e logs/job_logs/kirc_grid_harvest.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
export PYTHONUNBUFFERED=1
POETRY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"
VENV_PY="$run_dir/.venv/bin/python"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"

manifest="$run_dir/configs/tcga_kirc_manifest.csv"
grid_data="$cr/tcga_kirc/grid64_data_thr235"
grid_feat="$cr/tcga_kirc/grid64_sigmma_36852_step2000_thr235"
bundle="$run_dir/outputs/cell_niche_rcc/tile_tumour_stroma_probe/z_sigmma_concat__tile_tso_binary.joblib"
surv_tsv="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-KIRC_slide_survival.tsv"
out_dir="$run_dir/outputs/tcga_kirc_grid_tsr"
tsr_csv="$out_dir/tcga_kirc_grid_tsr.csv"
mkdir -p "$out_dir"

# ---- 1. tile TSR over all slides (8-direction surround, central-window) ----
echo "[1/2] tile TSR (white_thr235 grid, 8-dir surround)"
"$POETRY" -u scripts/tcga_tsr/tcga_grid_tsr.py \
    --grid_data_dir "$grid_data" --grid_feat_dir "$grid_feat" \
    --manifest "$manifest" --bundle "$bundle" \
    --out_csv "$tsr_csv" --per_tile_dir "$out_dir/per_tile" \
    --tile_um 64 --field_um 1860 --um_per_px 0.5 \
    --measure_frac 0.5 --reject_thr 0.75 \
    --surround_um 64 --surround_min 5 --corner_min 1 --min_tumour 10

# ---- 2. survival: OS/DSS/PFI x {fixed05, median} ----
for ep in OS DSS PFI; do
  for split in fixed05 median; do
    cut_arg=""; [ "$split" = "fixed05" ] && cut_arg="--stroma_cutoff 0.5"
    sdir="$out_dir/endpoints/$ep/$split"; mkdir -p "$sdir"
    echo "[2/2] survival ep=$ep split=$split"
    "$VENV_PY" -u scripts/tcga_tsr/tcga_survival.py \
        --tsr_csv "$tsr_csv" --survival_tsv "$surv_tsv" --cohort KIRC \
        --endpoint "$ep" --min_os_days 30 --min_cells 0 \
        --out_dir "$sdir" \
        --fig_dir "$run_dir/figures/clinical_outcome/tso_kirc/grid_endpoints/$ep/$split" \
        $cut_arg || echo "[survival] FAILED ep=$ep split=$split"
  done
done
echo "[kirc-grid-harvest] DONE -> $out_dir"
