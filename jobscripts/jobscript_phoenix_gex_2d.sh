#BSUB -J phoenix-gex2d[1-14]
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/phoenix-gex2d.%J.%I.out
#BSUB -e logs/job_logs/phoenix-gex2d.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=80GB:num=1"
#BSUB -n 4
#BSUB -M 64GB
#BSUB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]"

# Section-parallel zero-shot Phoenix eval on the panel_2d22ba31 SVG-100 TEST
# cohort (14 sections). Each array task evaluates ONE section (Path B, K=14
# dense bins, n_samples=8) and saves its raw per-FOV arrays as a shard via
# --save_raw. After all 14 finish, merge into one globally-pooled JSON:
#
#   python -m scripts.merge_phoenix_raw_gex_2d \
#       --out_root <OUT_DIR> --panel_path <PANEL> --target_grid 14 --fov_size 256
#
# Phoenix is a generative flow model: each FOV = 196 bins x n_samples=8 adaptive
# ODE solves over a DINOv2-giant encoder => ~70 s/FOV (measured, job 32535), NOT
# the ~6 s/FOV a regressor would take. Full sections (400-5097 FOVs) would be
# 8 h to 4 DAYS each. We therefore --max_fovs_per_slide 250 (seeded random
# subset): pooled per-gene Pearson + per-slide SSIM stay unbiased on a
# representative subset, and each task drops to ~5 h. Merge keeps per_gene_pearson
# pooled over all sections, matching the stored baselines.

set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH"
cd "${run_dir}"

PANEL_PATH="${run_dir}/outputs/gex_2d/svg_panel_2d22ba31/gene_panel.json"
TEST_MANIFEST="${run_dir}/outputs/gex_2d/svg_panel_2d22ba31/MANIFEST_panel_2d22ba31_test.csv"
DATA_CACHE_DIR="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/data/gex_2d/svg_panel_2d22ba31_g14_fov256"
OUT_DIR="${run_dir}/outputs/gex_2d/svg_panel_2d22ba31_step1000_test/phoenix"
mkdir -p "$OUT_DIR"

# Pick this array task's section (skip header line).
SECTION=$(sed -n "$((LSB_JOBINDEX + 1))p" "$TEST_MANIFEST" | cut -d, -f1)
echo "=== Phoenix array task ${LSB_JOBINDEX}: section ${SECTION} ==="

python -m scripts.predict_baseline_gex_2d \
    --baseline phoenix \
    --panel_path "$PANEL_PATH" \
    --val_manifest "$TEST_MANIFEST" \
    --data_cache_dir "$DATA_CACHE_DIR" \
    --out_root "$OUT_DIR" \
    --mode bin \
    --target_grid 14 --fov_size 256 \
    --crop_px 112 --n_samples 8 --phoenix_batch 32 \
    --smoke_slide "$SECTION" \
    --max_fovs_per_slide 250 --fov_sample_seed 0 \
    --save_raw

echo "=== section ${SECTION} DONE ==="
