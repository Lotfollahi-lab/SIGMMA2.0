#!/bin/bash
#BSUB -J tcga_coad_protocol_tsr
#BSUB -G s10396
#BSUB -q training-normal
#BSUB -n 1
#BSUB -M 16000
#BSUB -R "select[mem>16000] rusage[mem=16000]"
#BSUB -o logs/job_logs/tcga_coad_protocol_tsr.%J.out
#BSUB -e logs/job_logs/tcga_coad_protocol_tsr.%J.err

# Protocol-faithful TSR (van Pelt/Mesker, PMC6182321): 4-border valid fields,
# per-field Voronoi area stroma fraction, slide TSR = MAX over fields (hotspot /
# heterogeneity rule), fixed 50% cutoff. Self-contained (farm22 env passing
# unreliable). Survival run after with --stroma_cutoff 0.5 AND median.
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"
cd "$run_dir"
export PYTHONUNBUFFERED=1

PY="$run_dir/.venv/bin/python"
feat_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_coad/sigmma_36852_step2000"
per_cell_dir="$run_dir/outputs/tcga_coad_tsr/per_cell_pred"
manifest="$run_dir/configs/tcga_coad_manifest.csv"
surv_tsv="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-COAD_slide_survival.tsv"
out_dir="$run_dir/outputs/tcga_coad_tsr_protocol"
mkdir -p "$out_dir"

# --- 1mm field (x10 FOV analog) ---
"$PY" -u scripts/tcga_tsr/tcga_protocol_tsr.py \
    --features_dir "$feat_dir" --per_cell_dir "$per_cell_dir" \
    --manifest "$manifest" \
    --out_csv "$out_dir/tcga_coad_tsr_protocol_1mm.csv" \
    --field_um 1000 --um_per_px 0.5 --area_um 8 --max_dist_um 50 \
    --white_thr 220 --border_frac 0.25 --min_tumour 10

# --- 2mm field (larger x10 FOV) sensitivity ---
"$PY" -u scripts/tcga_tsr/tcga_protocol_tsr.py \
    --features_dir "$feat_dir" --per_cell_dir "$per_cell_dir" \
    --manifest "$manifest" \
    --out_csv "$out_dir/tcga_coad_tsr_protocol_2mm.csv" \
    --field_um 2000 --um_per_px 0.5 --area_um 8 --max_dist_um 50 \
    --white_thr 220 --border_frac 0.25 --min_tumour 10

# --- survival: clinical fixed 50% AND median, for both field sizes ---
for fld in 1mm 2mm; do
  for split in fixed05 median; do
    cut_arg=""; [ "$split" = "fixed05" ] && cut_arg="--stroma_cutoff 0.5"
    sdir="$out_dir/${fld}_${split}"; mkdir -p "$sdir"
    echo "[survival] field=$fld split=$split"
    "$PY" -u scripts/tcga_tsr/tcga_survival.py \
        --tsr_csv "$out_dir/tcga_coad_tsr_protocol_${fld}.csv" \
        --survival_tsv "$surv_tsv" --cohort COAD \
        --out_dir "$sdir" \
        --fig_dir "$run_dir/figures/clinical_outcome/tso_coad/protocol_${fld}_${split}" \
        $cut_arg || echo "[survival] FAILED field=$fld split=$split"
  done
done

echo "[protocol-tsr] DONE -> $out_dir"
