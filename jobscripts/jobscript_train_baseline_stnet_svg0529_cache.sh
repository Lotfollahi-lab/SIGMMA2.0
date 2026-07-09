#BSUB -J train-stnet-svg0529
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/train-stnet-svg0529.%J.out
#BSUB -e logs/job_logs/train-stnet-svg0529.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=40GB:num=1"
#BSUB -n 4
#BSUB -M 80GB
#BSUB -R "select[mem>80GB] rusage[mem=80GB] span[hosts=1]"

# ST-Net regression baseline on the NEW svg_panels_2026_05_29 main 5K cohort
# (panel_35a7bd78801c). Uses the Stage-1 binned-target cache built by job
# 35135/35243 (svg0529_panel_35a7bd78801c_g14_fov256) to skip per-iter
# load_slide_transcripts + bin_fov. Part of the full re-run on the corrected
# pipeline (see project_full_rerun_plan_2026_05_29).

set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate $ENV_PATH"
cd "${run_dir}"

PANEL_BASE="${run_dir}/outputs/gex_2d/svg0529_panel_35a7bd78801c"
PANEL_PATH="${PANEL_BASE}/gene_panel.json"
TRAIN_MANIFEST="${PANEL_BASE}/MANIFEST_panel_35a7bd78801c_train.csv"
VAL_MANIFEST="${PANEL_BASE}/MANIFEST_panel_35a7bd78801c_val.csv"
DATA_CACHE_DIR="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/data/gex_2d/svg0529_panel_35a7bd78801c_g14_fov256"
OUT_DIR="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/baselines/stnet_svg0529_cache"
mkdir -p "$OUT_DIR"

echo "=== ST-Net (svg0529 cache) → ${OUT_DIR} ==="
python -m scripts.train_baseline_stnet \
    --train_manifest "$TRAIN_MANIFEST" \
    --val_manifest "$VAL_MANIFEST" \
    --panel_path "$PANEL_PATH" \
    --data_cache_dir "$DATA_CACHE_DIR" \
    --out_dir "$OUT_DIR" \
    --num_steps 5000 \
    --batch_size 64 \
    --lr 1e-4 \
    --target_grid 14 \
    --fov_size 256

echo "=== ST-Net DONE ==="
