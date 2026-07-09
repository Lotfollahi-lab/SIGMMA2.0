#BSUB -J niche-rcc-probe
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/niche-rcc-probe.%J.out
#BSUB -e logs/job_logs/niche-rcc-probe.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=2GB:num=1"
#BSUB -n 16
#BSUB -M 200GB
#BSUB -R "select[mem>200GB] rusage[mem=200GB] span[hosts=1]"
# NOTE: farm group/queue drift — re-verify with `bqueues -u dj16`. CPU-only probe
# still needs a token GPU alloc to satisfy esub.

# RCC cross-slide cell-TYPE probe: LogReg on level_2_cell_type (10 classes incl.
# Tumour vs Nephron epithelium) over UNI2-h + SIGMMA scales/concat. Section-level
# split (donors mixed) from cell_niche_evaluable_slides_rcc.csv. Emits per-section
# MCC + weighted F1. Depends on RCC extract job (data/uni/sigmma stages).
# To add the 4 FT baselines (BLEEP/CLIP/PLIP/OmiCLIP): run
# extract_cell_niche_features_baselines.py --niche_config <rcc config> first, then
# append --baseline_features_dirs + z_{bleep,clip,plip,omiclip}_macro embeddings.
set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

run_dir="/nfs/team361/dj16/projects/SIGMMA"
cache_root="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
data_dir="${cache_root}/data/cell_niche_rcc/scales_64_128_256"
uni_dir="${cache_root}/uni/cell_niche_rcc/uni2h_input224_p14"
sigmma_dir="${cache_root}/sigmma/cell_niche_rcc/36852_step2000"
out_dir="${run_dir}/outputs/cell_niche_rcc/${LSB_JOBID}"
fig_dir="${run_dir}/figures/clinical_outcome/${LSB_JOBID}"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "${run_dir}"
mkdir -p "${out_dir}"

python -u ${run_dir}/scripts/probe_niche_cross_slide.py \
    --data_dir   "${data_dir}" \
    --uni_dir    "${uni_dir}" \
    --sigmma_dir "${sigmma_dir}" \
    --manifest   "${run_dir}/configs/cell_niche_evaluable_slides_rcc.csv" \
    --label_fields level_2_cell_type \
    --embeddings \
        z_uni_macro z_uni_macro_cls \
        z_sigmma_micro z_sigmma_meso z_sigmma_macro z_sigmma_concat \
    --out_dir "${out_dir}" \
    --fig_dir "${fig_dir}"
echo "[rcc-probe] DONE -> ${out_dir} (figs -> ${fig_dir})"
