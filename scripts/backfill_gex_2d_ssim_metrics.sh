#!/bin/bash
# Re-run gex_2d phase=eval against existing trained heads to back-fill new
# MSE / SSIM metrics. No head retraining; this is purely post-hoc.
#
# Covers:
#   - GEX-aware sweep at step1000: α ∈ {a01, a02, a04} × repr ∈ {uni, concat, micro, meso, macro}
#   - UNI2-h cohort baseline: repr=uni (head_type=linear_sweep)
#
# All evals overwrite the existing JSON in place (the new schema is a superset
# of the old schema — nothing is lost).

set -e
cd /nfs/team361/dj16/projects/SIGMMA
source /nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate

CACHE="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
DATA_CACHE="${CACHE}/data/gex_2d/dispersion_g14_fov256"
UNI_CACHE="${CACHE}/uni/gex_2d/uni2h_fov256_p14"
PANEL_DIR="outputs/gex_2d/disp"

common_sweep_flags=(
    --hvg_method dispersion --target_grid 14 --cache_target_grid 56 --fov_size 256
    --patch_size_hint 14
    --batch_size 64 --in_memory
    --panel_dir "$PANEL_DIR"
)

# ---------------- sweep (a01/a02/a04 × 5 reprs) ----------------
declare -A SIGMMA_CACHE
SIGMMA_CACHE[a01]="${CACHE}/sigmma/gex_2d/25923_1_step1000"
SIGMMA_CACHE[a02]="${CACHE}/sigmma/gex_2d/25923_2_step1000"
SIGMMA_CACHE[a04]="${CACHE}/sigmma/gex_2d/25923_3_step1000"

for alpha in a01 a02 a04; do
  out_root="outputs/gex_2d/gex_sweep_step1000/${alpha}"
  for repr in uni concat micro meso macro; do
    echo
    echo "=== sweep ${alpha} ${repr} ==="
    python -m scripts.probe_gex_2d \
      --phase eval --input_repr "$repr" --head_type mlp \
      --data_cache_dir "$DATA_CACHE" --uni_cache_dir "$UNI_CACHE" \
      --sigmma_cache_dir "${SIGMMA_CACHE[$alpha]}" \
      --out_root "$out_root" \
      "${common_sweep_flags[@]}"
  done
done

# ---------------- UNI2-h cohort baseline (uni, linear_sweep) ----------------
echo
echo "=== baseline uni2h uni (linear_sweep) ==="
python -m scripts.probe_gex_2d \
  --phase eval --input_repr uni --head_type linear_sweep \
  --data_cache_dir "$DATA_CACHE" --uni_cache_dir "$UNI_CACHE" \
  --out_root "outputs/gex_2d/disp_g14_cohort_uni2h" \
  "${common_sweep_flags[@]}"

# ---------------- UNI2-h-init baseline FT step1000 (uni, linear_sweep) ----------------
# Each baseline has its own uni cache from the earlier cache phase (job 26963).
for backbone in bleep_uni clip_uni plip_uni omiclip_uni; do
  echo
  echo "=== baseline ${backbone} uni (linear_sweep) ==="
  python -m scripts.probe_gex_2d \
    --phase eval --input_repr uni --head_type linear_sweep \
    --data_cache_dir "$DATA_CACHE" \
    --uni_cache_dir "${CACHE}/uni/gex_2d/${backbone}_step1000" \
    --out_root "outputs/gex_2d/disp_g14_cohort_uni2h_step1000/${backbone}" \
    "${common_sweep_flags[@]}"
done

echo
echo "[DONE] back-filled MSE + SSIM metrics on all step1000 sweep + UNI2-h baseline + 4 baseline FT"
