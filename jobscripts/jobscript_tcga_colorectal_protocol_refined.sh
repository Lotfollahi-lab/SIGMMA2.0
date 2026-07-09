#!/bin/bash
# Protocol-faithful TSR (van Pelt/Mesker, PMC6182321) end-to-end on TCGA
# COLORECTAL = COAD + READ, in one job. Retrains the tumour/stroma/other probe
# once (protocol grouping: immune+vasculature+perivascular -> stroma; only
# normal epithelium -> other) then runs predict -> protocol TSR (3.46 mm^2
# x10-FOV fields, 4-border valid, per-field Voronoi area stroma fraction,
# density gate, MAX hotspot) -> survival (clinical fixed 50% AND median) for
# each cohort. Mirrors jobscript_tcga_coad_protocol_refined.sh, adds READ.
# NOTE: probe is RCC-trained -> colorectal is out-of-domain label transfer;
# interpret sign with care (prior COAD run showed inverted OS association).
# Two venvs: poetry (sklearn/h5py) steps 1-2, .venv (zarr/scipy/statsmodels) 3-4.
#   bsub < jobscripts/jobscript_tcga_colorectal_protocol_refined.sh
#BSUB -J tcga_colorectal_protocol_refined
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -q inference
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=1GB:num=1"
#BSUB -n 8
#BSUB -M 64000
#BSUB -R "select[mem>64000] rusage[mem=64000] span[hosts=1]"
#BSUB -o logs/job_logs/tcga_colorectal_protocol_refined.%J.out
#BSUB -e logs/job_logs/tcga_colorectal_protocol_refined.%J.err
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1

POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"
VENV_PY="$run_dir/.venv/bin/python"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
bundle="$run_dir/outputs/cell_niche_rcc/tumour_stroma_other_probe/z_sigmma_concat__protocol.joblib"

# ---- 1. retrain probe with the protocol grouping (RCC) -- shared by both ----
echo "[1] retrain probe (grouping=protocol)"
"$POETRY_PY" -u scripts/tcga_tsr/train_tumour_stroma_probe.py \
    --data_dir   "$cr/data/cell_niche_rcc_2026_06_19/scales_64_128_256" \
    --sigmma_dir "$cr/sigmma/cell_niche_rcc_2026_06_19/36852_step2000" \
    --manifest   configs/cell_niche_evaluable_slides_rcc.csv \
    --grouping   protocol \
    --out_bundle "$bundle"

surv_base="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA"

run_cohort () {
  local COH="$1" coh_lc="$2"
  local feat_dir="$cr/tcga_${coh_lc}/sigmma_36852_step2000"
  local manifest="$run_dir/configs/tcga_${coh_lc}_manifest.csv"
  local surv_tsv="$surv_base/TCGA-${COH}_slide_survival.tsv"
  local out_dir="$run_dir/outputs/tcga_${coh_lc}_tsr_protocol"
  local per_cell_dir="$out_dir/per_cell_pred_protocol"
  mkdir -p "$out_dir"

  echo "[2/$COH] predict per-cell TSR with protocol bundle"
  "$POETRY_PY" -u scripts/tcga_tsr/tcga_predict_tsr.py \
      --features_dir "$feat_dir" --bundle "$bundle" \
      --out_csv "$out_dir/tcga_${coh_lc}_tsr_protocol_predict.csv" \
      --per_cell_dir "$per_cell_dir"

  echo "[3/$COH] protocol TSR (field 1860um ~ 3.46mm^2, density gate)"
  "$VENV_PY" -u scripts/tcga_tsr/tcga_protocol_tsr.py \
      --features_dir "$feat_dir" --per_cell_dir "$per_cell_dir" \
      --manifest "$manifest" \
      --out_csv "$out_dir/tcga_${coh_lc}_tsr_protocol_3p46mm.csv" \
      --field_um 1860 --um_per_px 0.5 --area_um 8 --max_dist_um 50 \
      --white_thr 220 --border_frac 0.25 --min_tumour 10 \
      --density_k 5 --density_radius_um 60

  for split in fixed05 median; do
    local cut_arg=""; [ "$split" = "fixed05" ] && cut_arg="--stroma_cutoff 0.5"
    local sdir="$out_dir/refined_${split}"; mkdir -p "$sdir"
    echo "[4/$COH] survival split=$split"
    "$VENV_PY" -u scripts/tcga_tsr/tcga_survival.py \
        --tsr_csv "$out_dir/tcga_${coh_lc}_tsr_protocol_3p46mm.csv" \
        --survival_tsv "$surv_tsv" --cohort "$COH" \
        --out_dir "$sdir" \
        --fig_dir "$run_dir/figures/clinical_outcome/tso_${coh_lc}/refined_${split}" \
        $cut_arg || echo "[survival] FAILED $COH split=$split"
  done
  echo "[$COH] DONE -> $out_dir"
}

run_cohort COAD coad
run_cohort READ read

echo "[colorectal-protocol-refined] DONE"
