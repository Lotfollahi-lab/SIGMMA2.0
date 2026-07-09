#!/usr/bin/env bash
# Visualize the REGION used for protocol TSR on TCGA-KIRC: reproduces the exact
# field selection (3.46mm^2 field, 4-border valid, density+follicle gates,
# MAX-stroma hotspot) and draws, per slide, (A) whole-slide thumbnail with the
# winning field highlighted + (B) a zoom of that field with per-cell labels.
# Auto-picks low/median/high-stroma slides from the TSR csv (override --samples).
# Reads H&E zarr -> needs .venv (zarr+scipy+matplotlib, same as protocol_tsr).
#BSUB -J tcga_kirc_region_overlay
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_kirc_region_overlay.%J.out
#BSUB -e logs/job_logs/tcga_kirc_region_overlay.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
VENV_PY="$run_dir/.venv/bin/python"
export PYTHONUNBUFFERED=1
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"

"$VENV_PY" -u scripts/tcga_tsr/tcga_protocol_overlay.py \
    --features_dir "$cr/tcga_kirc/sigmma_36852_step2000" \
    --per_cell_dir "$run_dir/outputs/tcga_kirc_tsr/per_cell_pred" \
    --manifest "$run_dir/configs/tcga_kirc_manifest.csv" \
    --tsr_csv "$run_dir/outputs/tcga_kirc_tsr/tcga_kirc_tsr_3p46mm.csv" \
    --fig_dir "$run_dir/figures/clinical_outcome/tso_kirc/region_overlays"
echo "[kirc-region-overlay] DONE"
