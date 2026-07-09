#!/usr/bin/env bash
# VERIFY the 8000-cell subsample cap does not bias the central-window TSR.
# For 5 slides spanning stroma_fraction, re-extract SIGMMA features at FULL
# density (--max_cells 0), re-predict, recompute central05 TSR, and compare
# per-slide stroma_fraction vs the 8000-cap values. If the ratio is stable the
# cap is just a visual-sparsity issue; if it shifts, all 519 need re-extraction.
# GPU (ViT-H) for extraction, then CPU predict/TSR/compare on the same node.
#BSUB -J tcga_kirc_verify_full
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_kirc_verify_full.%J.out
#BSUB -e logs/job_logs/tcga_kirc_verify_full.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=24GB:num=1"
#BSUB -n 4
#BSUB -M 64GB
#BSUB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]"
set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
config="$run_dir/configs/cell_niche_transfer_rcc_celltype_a01_step2000.yaml"
manifest="$run_dir/configs/tcga_kirc_manifest.csv"
bundle="$run_dir/outputs/cell_niche_rcc/51140/models/level_2_cell_type__z_sigmma_concat.joblib"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
VENV_PY="$run_dir/.venv/bin/python"

out_full="$cr/tcga_kirc/sigmma_36852_step2000_full"
vdir="$run_dir/outputs/tcga_kirc_tsr/verify_fullcells"
per_cell_full="$vdir/per_cell_pred"
tsr_full="$vdir/tcga_kirc_tsr_central05_full.csv"
mkdir -p "$out_full" "$per_cell_full" "$vdir"

SLIDES=(
  TCGA-BP-4995-01Z-00-DX1.e8ebf283-46f9-4d1f-a8a6-11c24d8fad46
  TCGA-A3-3374-01Z-00-DX1.560b8f9f-5195-43e0-be52-73aede19e077
  TCGA-CJ-4637-01Z-00-DX1.5632396E-A880-4686-9E94-89E3F01AFBFB
  TCGA-BP-5184-01Z-00-DX1.b682d92b-fc7b-4a8c-a0f3-d1e3a309f136
  TCGA-BP-4795-01Z-00-DX1.29d32315-efd2-495c-85b5-86db265cfc9c
)

# ---- 1. full-density feature extraction (GPU), one slide at a time ----
source "$ENV_PATH"
for s in "${SLIDES[@]}"; do
  if [ -f "$out_full/$s.h5" ]; then echo "[skip-extract] $s exists"; continue; fi
  echo "[1] extract FULL cells: $s"
  python -u "$run_dir/scripts/tcga_tsr/tcga_extract_sigmma_features.py" \
      --config "$config" --manifest "$manifest" --slide "$s" \
      --out_dir "$out_full" --max_cells 0 --batch_size 64
done

# ---- 2. predict per-cell types (10-class bundle) on full features ----
echo "[2] predict per-cell types (full)"
python -u "$run_dir/scripts/tcga_tsr/tcga_predict_tsr.py" \
    --features_dir "$out_full" --bundle "$bundle" \
    --out_csv "$vdir/tcga_kirc_predict_full.csv" \
    --per_cell_dir "$per_cell_full"

# ---- 3. central05 TSR on full features ----
echo "[3] protocol TSR (central window, full cells)"
"$VENV_PY" -u scripts/tcga_tsr/tcga_protocol_tsr.py \
    --features_dir "$out_full" --per_cell_dir "$per_cell_full" --manifest "$manifest" \
    --out_csv "$tsr_full" \
    --field_um 1860 --um_per_px 0.5 --area_um 8 --max_dist_um 50 \
    --white_thr 220 --border_frac 0.25 --min_tumour 10 \
    --density_k 5 --density_radius_um 60 \
    --follicle_radius_um 50 --follicle_k 15 --follicle_min 30 \
    --measure_frac 0.5

# ---- 4. compare 8000-cap vs full ----
echo "[4] compare cap vs full"
"$VENV_PY" -u scripts/tcga_tsr/tcga_tsr_compare.py \
    --cap_csv "$run_dir/outputs/tcga_kirc_tsr/central05/tcga_kirc_tsr_central05.csv" \
    --full_csv "$tsr_full"
echo "[verify-fullcells] DONE"
