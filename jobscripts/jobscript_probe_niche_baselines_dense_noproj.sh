#BSUB -J niche-base-noproj
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/niche-base-noproj.%J.out
#BSUB -e logs/job_logs/niche-base-noproj.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=2GB:num=1"
#BSUB -n 8
#BSUB -M 100GB
#BSUB -R "select[mem>100GB] rusage[mem=100GB] span[hosts=1]"
# CPU-only probe still needs a token GPU alloc to satisfy esub.

# Fair raw-grid comparison: re-probe the 4 FT baselines on their RAW-GRID
# (dense_noproj) macro feature — the same no-projection UNI2-h forward_features
# feature SIGMMA now uses. Also re-evaluates projected z_<b>_macro as a sanity
# check against the existing 27663 run. Skin in-house, section-level split.
set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

run_dir="/nfs/team361/dj16/projects/SIGMMA"
cache_root="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
data_dir="${cache_root}/data/cell_niche/scales_64_128_256"
base_root="${cache_root}/cell_niche/features_baselines/27197"
out_dir="${run_dir}/outputs/cell_niche/baselines_dense_noproj_${LSB_JOBID}"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "${run_dir}"
mkdir -p "${out_dir}"

python -u ${run_dir}/scripts/probe_niche_cross_slide.py \
    --data_dir "${data_dir}" \
    --baseline_features_dirs \
        "${base_root}/bleep" "${base_root}/clip" \
        "${base_root}/plip" "${base_root}/omiclip" \
    --manifest "${run_dir}/configs/cell_niche_evaluable_slides.csv" \
    --embeddings \
        z_bleep_macro z_clip_macro z_plip_macro z_omiclip_macro \
        z_bleep_macro_dense_noproj z_clip_macro_dense_noproj \
        z_plip_macro_dense_noproj z_omiclip_macro_dense_noproj \
    --out_dir "${out_dir}"
echo "[niche-base-noproj] DONE -> ${out_dir}"
