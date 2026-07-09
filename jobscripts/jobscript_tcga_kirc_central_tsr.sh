#!/usr/bin/env bash
# REFINED TSR: measure the stroma/tumour ratio in the CENTRAL window of each
# hotspot field (--measure_frac 0.5) instead of the whole field. The 4-border
# test stays only a gate; the central window must itself be intratumoural
# (>= min_tumour tumour cells), which rejects the peripheral fat/stroma pockets
# that were winning the MAX hotspot (visual QC: G6-A8L6 frac 0.957 sat on
# adipose). Chain: recompute TSR -> survival (4 endpoints x 2 splits, stage-adj
# Cox) -> lifelines binary Cox -> re-visualize the region used.
#BSUB -J tcga_kirc_central_tsr
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_kirc_central_tsr.%J.out
#BSUB -e logs/job_logs/tcga_kirc_central_tsr.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
VENV_PY="$run_dir/.venv/bin/python"
POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"
export PYTHONUNBUFFERED=1
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"

feat_dir="$cr/tcga_kirc/sigmma_36852_step2000"
per_cell="$run_dir/outputs/tcga_kirc_tsr/per_cell_pred"
manifest="$run_dir/configs/tcga_kirc_manifest.csv"
surv_tsv="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-KIRC_slide_survival.tsv"
base="$run_dir/outputs/tcga_kirc_tsr/central05"
tsr_csv="$base/tcga_kirc_tsr_central05.csv"
MF=0.5
mkdir -p "$base"

# ---- 1. recompute protocol TSR, measuring the CENTRAL window ----
echo "[1/3] protocol TSR with central window (measure_frac=$MF)"
"$VENV_PY" -u scripts/tcga_tsr/tcga_protocol_tsr.py \
    --features_dir "$feat_dir" --per_cell_dir "$per_cell" --manifest "$manifest" \
    --out_csv "$tsr_csv" \
    --field_um 1860 --um_per_px 0.5 --area_um 8 --max_dist_um 50 \
    --white_thr 220 --border_frac 0.25 --min_tumour 10 \
    --density_k 5 --density_radius_um 60 \
    --follicle_radius_um 50 --follicle_k 15 --follicle_min 30 \
    --measure_frac "$MF"

# ---- 2. survival (4 endpoints x 2 splits, stage-adjusted Cox) + binary Cox ----
for ep in OS DSS DFI PFI; do
  ep_l=$(echo "$ep" | tr '[:upper:]' '[:lower:]')
  epbase="$base/endpoints/$ep"
  for split in fixed05 median; do
    cut_arg=""; [ "$split" = "fixed05" ] && cut_arg="--stroma_cutoff 0.5"
    out="$epbase/$split"
    fig="$run_dir/figures/clinical_outcome/tso_kirc/central05/$ep/$split"
    mkdir -p "$out" "$fig"
    echo "[2] survival ep=$ep split=$split"
    "$VENV_PY" -u scripts/tcga_tsr/tcga_survival.py \
        --tsr_csv "$tsr_csv" --survival_tsv "$surv_tsv" --cohort KIRC \
        --endpoint "$ep" --min_os_days 30 \
        --out_dir "$out" --fig_dir "$fig" $cut_arg \
      || echo "[FAILED] survival ep=$ep split=$split"
  done
  echo "[2] lifelines binary Cox ep=$ep"
  "$POETRY_PY" -u scripts/tcga_tsr/tcga_cox_group.py \
      --base "$epbase" --stem "tcga_kirc_${ep_l}_tsr" \
      --out "$epbase/cox_group_hr_lifelines.txt" \
    || echo "[FAILED] cox_group ep=$ep"
done

# ---- 3. re-visualize the region used (central window) ----
echo "[3/3] region overlay (central window)"
"$VENV_PY" -u scripts/tcga_tsr/tcga_protocol_overlay.py \
    --features_dir "$feat_dir" --per_cell_dir "$per_cell" --manifest "$manifest" \
    --tsr_csv "$tsr_csv" --measure_frac "$MF" \
    --fig_dir "$run_dir/figures/clinical_outcome/tso_kirc/region_overlays_central05"
echo "[kirc-central-tsr] DONE"
