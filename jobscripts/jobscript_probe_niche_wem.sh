#BSUB -J niche-wem-probe
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/niche-wem-probe.%J.out
#BSUB -e logs/job_logs/niche-wem-probe.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=2GB:num=1"
#BSUB -n 16
#BSUB -M 200GB
#BSUB -R "select[mem>200GB] rusage[mem=200GB] span[hosts=1]"
# CPU-only LogReg probe still needs a token GPU alloc to satisfy esub.

# whole_embryo cross-slide niche probe (label = niche40). Same headline recipe
# as skin: UNI2-h raw-grid macro (z_uni_macro) vs SIGMMA raw-grid macro
# (z_sigmma_macro_dense_noproj) vs the 4 fine-tuned UNI baselines. Section-level
# split (16 train / 2 val / 2 test). z_sigmma_macro is kept as a projected-vs-raw
# sanity column. Depends on jobscript_extract_cell_niche_wem.sh.
set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

run_dir="/nfs/team361/dj16/projects/SIGMMA"
cache_root="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
data_dir="${cache_root}/data/cell_niche_wem_step2000/scales_64_128_256"
uni_dir="${cache_root}/uni/cell_niche_wem_step2000/uni2h_input224_p14"
sigmma_dir="${cache_root}/sigmma/cell_niche_wem/36852_step2000"
baseline_root="${cache_root}/cell_niche_wem/features_baselines/116348"
out_dir="${run_dir}/outputs/cell_niche/wem_step2000_${LSB_JOBID}"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "${run_dir}"
mkdir -p "${out_dir}"

python -u ${run_dir}/scripts/probe_niche_cross_slide.py \
    --data_dir     "${data_dir}" \
    --uni_dir      "${uni_dir}" \
    --sigmma_dir   "${sigmma_dir}" \
    --manifest     "${run_dir}/configs/cell_niche_evaluable_slides_whole_embryo.csv" \
    --label_fields niche40 \
    --baseline_features_dirs \
        "${baseline_root}/bleep" "${baseline_root}/clip" \
        "${baseline_root}/plip"  "${baseline_root}/omiclip" \
    --embeddings \
        z_uni_macro \
        z_sigmma_macro z_sigmma_macro_dense_noproj \
        z_bleep_macro z_clip_macro z_plip_macro z_omiclip_macro \
    --out_dir "${out_dir}"
echo "[niche-wem-probe] DONE -> ${out_dir}"
