#BSUB -J gex2d-baselines-uni2h-step1000[1-4]
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/gex2d-baselines-uni2h-step1000.%J.%I.out
#BSUB -e logs/job_logs/gex2d-baselines-uni2h-step1000.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=40GB:num=1"
#BSUB -n 4
#BSUB -M 80GB
#BSUB -R "select[mem>80GB] rusage[mem=80GB] span[hosts=1]"

# GEX-2D cohort eval for the 4 UNI2-h-init baseline FT step1000 ckpts:
#   index 1 -> bleep_uni  (26355)
#   index 2 -> clip_uni   (26375_1)
#   index 3 -> plip_uni   (26375_2)
#   index 4 -> omiclip_uni (26375_3)
# Per array element: cache FT features → train linear-sweep head → eval.

set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${LSB_JOBINDEX}" in
    1) backbone="bleep_uni";   ckpt_run="26355" ;;
    2) backbone="clip_uni";    ckpt_run="26375_1" ;;
    3) backbone="plip_uni";    ckpt_run="26375_2" ;;
    4) backbone="omiclip_uni"; ckpt_run="26375_3" ;;
    *) echo "Unknown LSB_JOBINDEX=${LSB_JOBINDEX}"; exit 1 ;;
esac

# omiclip_uni training output dir is "in-house_train_omiclip_uni" (the
# bleep_uni/clip_uni/plip_uni naming is identical).
ckpt_dir="in-house_train_${backbone}"
ft_ckpt="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/baselines/${ckpt_dir}/${ckpt_run}/checkpoint-step1000.pth"

run_dir="/nfs/team361/dj16/projects/SIGMMA"
cache_root="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
# Shared data cache (target/mask only — ckpt-independent).
data_cache_dir="${cache_root}/data/gex_2d/dispersion_g14_fov256"
# Per-baseline UNI features cache (depends on FT ckpt + UNI variant).
uni_cache_dir="${cache_root}/uni/gex_2d/${backbone}_step1000"

out_root="outputs/gex_2d/disp_g14_cohort_uni2h_step1000/${backbone}"
panel_dir="outputs/gex_2d/disp"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"
cd "${run_dir}"

mkdir -p "${out_root}"

common=(
    --data_cache_dir "$data_cache_dir"
    --uni_cache_dir "$uni_cache_dir"
    --panel_dir "$panel_dir"
    --out_root "$out_root"
    --hvg_method dispersion --target_grid 14 --cache_target_grid 56 --fov_size 256
    --patch_size_hint 14
)

echo "=========================================="
echo "[${backbone}] FT ckpt: ${ft_ckpt}"
echo "[${backbone}] out_root: ${out_root}"
echo "=========================================="

# PHASE 1: cache FT features (uses --backbone <baseline_uni> + --ft_ckpt_path).
echo "[${backbone}] PHASE 1: cache ${backbone} features -> ${uni_cache_dir}"
python -m scripts.probe_gex_2d --phase cache --backbone "$backbone" \
    --ft_ckpt_path "$ft_ckpt" \
    "${common[@]}"

# PHASE 2+3: train + eval the linear-sweep head on cached features (input_repr=uni).
echo "[${backbone}] PHASE 2: train linear_sweep head on cached features"
python -m scripts.probe_gex_2d --phase train \
    --input_repr uni --head_type linear_sweep \
    --num_steps 2000 --batch_size 64 --in_memory \
    "${common[@]}"

echo "[${backbone}] PHASE 3: eval"
python -m scripts.probe_gex_2d --phase eval \
    --input_repr uni --head_type linear_sweep \
    --batch_size 64 --in_memory \
    "${common[@]}"

echo "[${backbone}] DONE — results under ${out_root}/results_linear_sweep/"
