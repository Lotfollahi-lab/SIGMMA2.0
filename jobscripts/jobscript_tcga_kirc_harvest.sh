#!/usr/bin/env bash
# Tier-4 IN-DOMAIN validation: TSR -> survival on TCGA-KIRC (kidney = in-domain
# for the RCC cell-type probe; clear-cell RCC morphology MATCHES the training
# cohort, so the RCC->non-renal label-transfer failure that nulled COAD/READ
# does NOT apply here). Predict -> protocol-faithful TSR (van Pelt/Mesker) with
# the NEW lymphocytic-follicle exclusion -> survival.
#
# Gate on the KIRC feature array:
#   bsub -w "ended(KIRC_FEATJOBID)" < jobscripts/jobscript_tcga_kirc_harvest.sh
#
# KEY CHOICES (per analysis 2026-06-23):
#  * 10-CLASS level_2 bundle (NOT the 3-class tso/protocol bundle): keeps the
#    FINE immune labels in the per-cell npz so the follicle gate can see them;
#    tcga_predict_tsr falls back to CLASSIC tumour/stroma (immune excluded).
#  * protocol_tsr.py GROUP map -> classic Mesker compartments (immune is its OWN
#    group, NOT stroma) + density-based follicle gate (drops fields with a dense
#    lymphocytic aggregate; literature: "avoid lymphocytic follicles/aggregates").
#  * survival: clinical fixed 50% AND median split.
#BSUB -J tcga_kirc_harvest
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_kirc_harvest.%J.out
#BSUB -e logs/job_logs/tcga_kirc_harvest.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1

POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"
VENV_PY="$run_dir/.venv/bin/python"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"

# ---- KIRC I/O ----
feat_dir="$cr/tcga_kirc/sigmma_36852_step2000"
manifest="$run_dir/configs/tcga_kirc_manifest.csv"
surv_tsv="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-KIRC_slide_survival.tsv"
out_dir="$run_dir/outputs/tcga_kirc_tsr"
# 10-class RCC level_2 probe (FINE labels preserved in npz -> follicle gate)
bundle="$run_dir/outputs/cell_niche_rcc/51140/models/level_2_cell_type__z_sigmma_concat.joblib"
per_cell_dir="$out_dir/per_cell_pred"
mkdir -p "$out_dir"

# ---- 1. predict per-cell types on KIRC (FINE 10-class) ----
echo "[1/3] predict per-cell types on KIRC (10-class level_2 bundle)"
"$POETRY_PY" -u scripts/tcga_tsr/tcga_predict_tsr.py \
    --features_dir "$feat_dir" --bundle "$bundle" \
    --out_csv "$out_dir/tcga_kirc_predict.csv" \
    --per_cell_dir "$per_cell_dir"

# ---- 2. protocol TSR: 3.46mm^2 field, density gate, MAX hotspot, FOLLICLE gate ----
echo "[2/3] protocol TSR + lymphocytic-follicle exclusion"
"$VENV_PY" -u scripts/tcga_tsr/tcga_protocol_tsr.py \
    --features_dir "$feat_dir" --per_cell_dir "$per_cell_dir" \
    --manifest "$manifest" \
    --out_csv "$out_dir/tcga_kirc_tsr_3p46mm.csv" \
    --field_um 1860 --um_per_px 0.5 --area_um 8 --max_dist_um 50 \
    --white_thr 220 --border_frac 0.25 --min_tumour 10 \
    --density_k 5 --density_radius_um 60 \
    --follicle_radius_um 50 --follicle_k 15 --follicle_min 30

# ---- 3. survival: clinical fixed 50% AND median ----
for split in fixed05 median; do
  cut_arg=""; [ "$split" = "fixed05" ] && cut_arg="--stroma_cutoff 0.5"
  sdir="$out_dir/$split"; mkdir -p "$sdir"
  echo "[3/3] survival split=$split"
  "$VENV_PY" -u scripts/tcga_tsr/tcga_survival.py \
      --tsr_csv "$out_dir/tcga_kirc_tsr_3p46mm.csv" \
      --survival_tsv "$surv_tsv" --cohort KIRC \
      --out_dir "$sdir" \
      --fig_dir "$run_dir/figures/clinical_outcome/tso_kirc/$split" \
      --min_os_days 30 \
      $cut_arg || echo "[survival] FAILED split=$split"
done

echo "[kirc-tier4] DONE -> $out_dir"
