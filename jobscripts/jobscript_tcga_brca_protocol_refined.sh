#!/usr/bin/env bash
# BRCA Stage C+D (PREVIEW): protocol-faithful TSR (van Pelt/Mesker) + survival on
# the 100-slide preview subset. Reuses the RCC-trained protocol probe bundle
# (already on disk; built by the colorectal job). predict -> protocol TSR
# (3.46 mm^2 x10-FOV fields, density gate, MAX hotspot) -> survival (fixed 50% +
# median). NOTE: RCC probe on breast = out-of-domain label transfer; sanity-check
# only. Two venvs: poetry (steps 1-2), .venv (steps 3-4).
#   bsub -w "done(FEATJOBID)" < jobscripts/jobscript_tcga_brca_protocol_refined.sh
#BSUB -J tcga_brca_protocol_refined
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -q inference
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=1GB:num=1"
#BSUB -n 8
#BSUB -M 64000
#BSUB -R "select[mem>64000] rusage[mem=64000] span[hosts=1]"
#BSUB -o logs/job_logs/tcga_brca_protocol_refined.%J.out
#BSUB -e logs/job_logs/tcga_brca_protocol_refined.%J.err
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1

POETRY_PY="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/python"
VENV_PY="$run_dir/.venv/bin/python"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"

feat_dir="$cr/tcga_brca/sigmma_36852_step2000"
manifest="$run_dir/configs/tcga_brca_manifest_preview100.csv"
surv_tsv="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-BRCA_slide_survival.tsv"
out_dir="$run_dir/outputs/tcga_brca_tsr_protocol_preview"
bundle="$run_dir/outputs/cell_niche_rcc/tumour_stroma_other_probe/z_sigmma_concat__protocol.joblib"
per_cell_dir="$out_dir/per_cell_pred_protocol"
mkdir -p "$out_dir"

echo "[1/3] predict per-cell TSR with protocol bundle"
"$POETRY_PY" -u scripts/tcga_tsr/tcga_predict_tsr.py \
    --features_dir "$feat_dir" --bundle "$bundle" \
    --out_csv "$out_dir/tcga_brca_tsr_protocol_predict.csv" \
    --per_cell_dir "$per_cell_dir"

echo "[2/3] protocol TSR (field 1860um ~ 3.46mm^2, density gate)"
"$VENV_PY" -u scripts/tcga_tsr/tcga_protocol_tsr.py \
    --features_dir "$feat_dir" --per_cell_dir "$per_cell_dir" \
    --manifest "$manifest" \
    --out_csv "$out_dir/tcga_brca_tsr_protocol_3p46mm.csv" \
    --field_um 1860 --um_per_px 0.5 --area_um 8 --max_dist_um 50 \
    --white_thr 220 --border_frac 0.25 --min_tumour 10 \
    --density_k 5 --density_radius_um 60

for split in fixed05 median; do
  cut_arg=""; [ "$split" = "fixed05" ] && cut_arg="--stroma_cutoff 0.5"
  sdir="$out_dir/refined_${split}"; mkdir -p "$sdir"
  echo "[3/3] survival split=$split"
  "$VENV_PY" -u scripts/tcga_tsr/tcga_survival.py \
      --tsr_csv "$out_dir/tcga_brca_tsr_protocol_3p46mm.csv" \
      --survival_tsv "$surv_tsv" --cohort BRCA \
      --out_dir "$sdir" \
      --fig_dir "$run_dir/figures/clinical_outcome/tso_brca_preview/refined_${split}" \
      $cut_arg || echo "[survival] FAILED split=$split"
done
echo "[brca-protocol-refined preview] DONE -> $out_dir"
