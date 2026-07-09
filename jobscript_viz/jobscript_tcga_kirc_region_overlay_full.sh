#!/usr/bin/env bash
# Redraw the protocol-TSR region overlays for the 5 KIRC slides re-predicted at
# FULL cell density (--max_cells 0, from the 8000-cap verify job 55615). Uses the
# full-density features + per-cell preds and the central measurement window
# (--measure_frac 0.5) so the panels match the central05 estimator exactly.
# Draws ALL 5 verify slides (explicit --samples) so they can be compared 1:1 to
# the 8000-cap overlays in figures/.../region_overlays_central05/.
# Reads H&E zarr -> needs .venv (zarr+scipy+matplotlib).
#BSUB -J tcga_kirc_region_overlay_full
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_kirc_region_overlay_full.%J.out
#BSUB -e logs/job_logs/tcga_kirc_region_overlay_full.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
VENV_PY="$run_dir/.venv/bin/python"
export PYTHONUNBUFFERED=1
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
vdir="$run_dir/outputs/tcga_kirc_tsr/verify_fullcells"

"$VENV_PY" -u scripts/tcga_tsr/tcga_protocol_overlay.py \
    --features_dir "$cr/tcga_kirc/sigmma_36852_step2000_full" \
    --per_cell_dir "$vdir/per_cell_pred" \
    --manifest "$run_dir/configs/tcga_kirc_manifest.csv" \
    --tsr_csv "$vdir/tcga_kirc_tsr_central05_full.csv" \
    --measure_frac 0.5 \
    --samples \
        TCGA-A3-3374-01Z-00-DX1.560b8f9f-5195-43e0-be52-73aede19e077 \
        TCGA-BP-4795-01Z-00-DX1.29d32315-efd2-495c-85b5-86db265cfc9c \
        TCGA-BP-4995-01Z-00-DX1.e8ebf283-46f9-4d1f-a8a6-11c24d8fad46 \
        TCGA-BP-5184-01Z-00-DX1.b682d92b-fc7b-4a8c-a0f3-d1e3a309f136 \
        TCGA-CJ-4637-01Z-00-DX1.5632396E-A880-4686-9E94-89E3F01AFBFB \
    --fig_dir "$run_dir/figures/clinical_outcome/tso_kirc/region_overlays_central05_full"
echo "[kirc-region-overlay-full] DONE"
