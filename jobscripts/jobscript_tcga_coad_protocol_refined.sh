#!/bin/bash
# Protocol-faithful TSR refinement (van Pelt/Mesker, PMC6182321), end-to-end on
# TCGA-COAD. Chains four steps in one job (farm22 env passing unreliable):
#   1. retrain the tumour/stroma/other probe with the PROTOCOL grouping
#      (immune + vasculature + perivascular -> stroma; only normal epithelium
#      -> other) -> new joblib bundle (classic bundle preserved).
#   2. re-predict per-cell groups on COAD with the new bundle (pred labels
#      change) -> per_cell_pred_protocol/*.npz + a slide-level TSR csv (used to
#      check the "other" reject-bucket shrinkage vs the old 57% median).
#   3. protocol TSR: 3.46 mm^2 (1860 um) x10-FOV fields, 4-border valid fields,
#      per-field Voronoi area stroma fraction with active low-cell-density
#      exclusion, slide TSR = MAX over fields (hotspot rule).
#   4. survival: clinical fixed 50% AND median split.
# Two venvs: poetry (sklearn/h5py) for steps 1-2, .venv (zarr/scipy/statsmodels)
# for steps 3-4. training-parallel + GPU-shared is the queue that currently
# admits jobs (-G s10396; team361 group gone).
#   bsub < jobscripts/jobscript_tcga_coad_protocol_refined.sh
#BSUB -J tcga_coad_protocol_refined
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=1GB:num=1"
#BSUB -n 8
#BSUB -M 64000
#BSUB -R "select[mem>64000] rusage[mem=64000] span[hosts=1]"
#BSUB -o logs/job_logs/tcga_coad_protocol_refined.%J.out
#BSUB -e logs/job_logs/tcga_coad_protocol_refined.%J.err
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1

POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"
VENV_PY="$run_dir/.venv/bin/python"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"

# ---- COAD I/O ----
feat_dir="$cr/tcga_coad/sigmma_36852_step2000"
manifest="$run_dir/configs/tcga_coad_manifest.csv"
surv_tsv="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-COAD_slide_survival.tsv"
out_dir="$run_dir/outputs/tcga_coad_tsr_protocol"
bundle="$run_dir/outputs/cell_niche_rcc/tumour_stroma_other_probe/z_sigmma_concat__protocol.joblib"
per_cell_dir="$out_dir/per_cell_pred_protocol"
mkdir -p "$out_dir"

# ---- 1. retrain probe with the protocol grouping (RCC) ----
echo "[1/4] retrain probe (grouping=protocol)"
"$POETRY_PY" -u scripts/tcga_tsr/train_tumour_stroma_probe.py \
    --data_dir   "$cr/data/cell_niche_rcc_2026_06_19/scales_64_128_256" \
    --sigmma_dir "$cr/sigmma/cell_niche_rcc_2026_06_19/36852_step2000" \
    --manifest   configs/cell_niche_evaluable_slides_rcc.csv \
    --grouping   protocol \
    --out_bundle "$bundle"

# ---- 2. re-predict per-cell groups on COAD (regenerate npz) ----
echo "[2/4] predict per-cell TSR on COAD with protocol bundle"
"$POETRY_PY" -u scripts/tcga_tsr/tcga_predict_tsr.py \
    --features_dir "$feat_dir" --bundle "$bundle" \
    --out_csv "$out_dir/tcga_coad_tsr_protocol_predict.csv" \
    --per_cell_dir "$per_cell_dir"

# ---- 3. protocol TSR: 3.46 mm^2 field + density gate, MAX hotspot ----
echo "[3/4] protocol TSR (field 1860um ~ 3.46mm^2, density gate)"
"$VENV_PY" -u scripts/tcga_tsr/tcga_protocol_tsr.py \
    --features_dir "$feat_dir" --per_cell_dir "$per_cell_dir" \
    --manifest "$manifest" \
    --out_csv "$out_dir/tcga_coad_tsr_protocol_3p46mm.csv" \
    --field_um 1860 --um_per_px 0.5 --area_um 8 --max_dist_um 50 \
    --white_thr 220 --border_frac 0.25 --min_tumour 10 \
    --density_k 5 --density_radius_um 60

# ---- 4. survival: clinical fixed 50% AND median ----
for split in fixed05 median; do
  cut_arg=""; [ "$split" = "fixed05" ] && cut_arg="--stroma_cutoff 0.5"
  sdir="$out_dir/refined_${split}"; mkdir -p "$sdir"
  echo "[4/4] survival split=$split"
  "$VENV_PY" -u scripts/tcga_tsr/tcga_survival.py \
      --tsr_csv "$out_dir/tcga_coad_tsr_protocol_3p46mm.csv" \
      --survival_tsv "$surv_tsv" --cohort COAD \
      --out_dir "$sdir" \
      --fig_dir "$run_dir/figures/clinical_outcome/tso_coad/refined_${split}" \
      $cut_arg || echo "[survival] FAILED split=$split"
done

echo "[protocol-refined] DONE -> $out_dir"
