#BSUB -J extract-cell-niche-wem
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/extract-cell-niche-wem.%J.out
#BSUB -e logs/job_logs/extract-cell-niche-wem.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 8
#BSUB -M 200GB
#BSUB -R "select[mem>200GB] rusage[mem=200GB] span[hosts=1]"

# whole_embryo niche features (data -> uni -> sigmma) mirroring the skin
# extract, but on the WEM manifest + niche40 label schema. The three stages are
# sequential: uni/sigmma both read the tile crops written by the data stage.
# Baseline (BLEEP/CLIP/PLIP/OmiCLIP) features already exist under
# .../cache/cell_niche_wem/features_baselines/116348 (job 116348).
set -e

module load python-3.11.6

export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
config="${run_dir}/configs/cell_niche_transfer_whole_embryo_a01_step2000.yaml"
cache_root="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"

data_dir="${cache_root}/data/cell_niche_wem_step2000/scales_64_128_256"
uni_dir="${cache_root}/uni/cell_niche_wem_step2000/uni2h_input224_p14"
sigmma_dir="${cache_root}/sigmma/cell_niche_wem/36852_step2000"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "${run_dir}"

echo "[wem-extract] stage=data -> ${data_dir}"
python -u ${run_dir}/scripts/extract_cell_niche_features.py \
    --stage    data \
    --config   "${config}" \
    --data_dir "${data_dir}"

echo "[wem-extract] stage=uni -> ${uni_dir}"
python -u ${run_dir}/scripts/extract_cell_niche_features.py \
    --stage    uni \
    --config   "${config}" \
    --data_dir "${data_dir}" \
    --uni_dir  "${uni_dir}"

echo "[wem-extract] stage=sigmma -> ${sigmma_dir}"
python -u ${run_dir}/scripts/extract_cell_niche_features.py \
    --stage      sigmma \
    --config     "${config}" \
    --data_dir   "${data_dir}" \
    --sigmma_dir "${sigmma_dir}"

echo "[wem-extract] DONE"
