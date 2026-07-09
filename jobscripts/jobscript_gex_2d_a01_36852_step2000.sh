#!/bin/bash
#BSUB -J gex2d-a01-36852
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/gex2d-a01-36852.%J.out
#BSUB -e logs/job_logs/gex2d-a01-36852.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=40GB:num=1"
#BSUB -n 4
#BSUB -M 160GB
#BSUB -R "select[mem>160GB] rusage[mem=160GB] span[hosts=1]"

# 2D GEX prediction for SIGMMA on the SAME checkpoint as retrieval/niche
#   sigmma_dense_attnpool_gex_svg_a01/36852/checkpoint-step2000.pth
#
# Caches are per-section HDF5 (new format). data + sigmma caches already
# converted from npz->h5; the 36852 sigmma feature cache is the complete
# superset (889 entries, ⊇ the old 41389 cache), so phase=cache is a no-op
# (existing .h5 are skipped) + writes cache_meta.json. Then a
# probe-head train+eval for the 4 SIGMMA reprs (concat/micro/meso/macro).
# uni repr is skipped (uni2h feature cache was deleted); baselines are NOT
# re-run — their Jun-3 result jsons are reused for comparison.

set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH"
cd "${run_dir}"

cache_root="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
data_cache_dir="${cache_root}/data/gex_2d/svg0529_panel_35a7bd78801c_g14_fov256"
sigmma_cache_dir="${cache_root}/sigmma/gex_2d/36852_step2000"
panel_dir="outputs/gex_2d/svg0529_panel_35a7bd78801c"
sigmma_config="${run_dir}/configs/cell_niche_transfer_dense_attnpool_a01_step2000.yaml"

train_manifest="${run_dir}/${panel_dir}/MANIFEST_panel_35a7bd78801c_train.csv"
val_manifest="${run_dir}/${panel_dir}/MANIFEST_panel_35a7bd78801c_val.csv"

out_root="outputs/gex_2d/svg0529_step2000/sigmma_a01_36852_step2000"
mkdir -p "$out_root"
cp "$sigmma_config" "${out_root}/"

common_args=(
    --data_cache_dir "$data_cache_dir"
    --panel_dir "$panel_dir"
    --target_grid 14 --cache_target_grid 56 --fov_size 256
    --patch_size_hint 14
    --train_manifest "$train_manifest"
    --val_manifest "$val_manifest"
)

echo "=========================================="
echo "STEP 1/3: fill missing SIGMMA feature cache (36852 step2000)"
echo "  $(grep checkpoint_path "${sigmma_config}")"
echo "=========================================="
python -m scripts.probe_gex_2d --phase cache --backbone sigmma \
    "${common_args[@]}" \
    --sigmma_cache_dir "$sigmma_cache_dir" \
    --sigmma_config "$sigmma_config"

echo "=========================================="
echo "STEP 2/3 + 3/3: probe-head train + eval (concat/micro/meso/macro)"
echo "=========================================="
for repr in concat micro meso macro; do
    echo "---- repr=${repr} : train ----"
    python -m scripts.probe_gex_2d --phase train --input_repr "$repr" \
        "${common_args[@]}" --head_type mlp \
        --sigmma_cache_dir "$sigmma_cache_dir" \
        --sigmma_config "$sigmma_config" \
        --num_steps 2000 --batch_size 64 \
        --out_root "$out_root"
    echo "---- repr=${repr} : eval ----"
    python -m scripts.probe_gex_2d --phase eval --input_repr "$repr" \
        "${common_args[@]}" --head_type mlp \
        --sigmma_cache_dir "$sigmma_cache_dir" \
        --sigmma_config "$sigmma_config" \
        --num_steps 2000 --batch_size 64 \
        --out_root "$out_root"
done

echo "=== DONE: results under ${out_root}/results/{concat,micro,meso,macro}.json ==="
