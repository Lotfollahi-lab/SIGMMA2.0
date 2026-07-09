#!/usr/bin/env bash
# Stage B: SIGMMA feature extraction (z_sigmma_*) for all TCGA data_h5s.
# Loads the model once and iterates the manifest. Uses the sigmma poetry venv.
#   bsub < jobscripts/jobscript_tcga_sigmma_features.sh
#BSUB -J tcga-sigmma-feats
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga-sigmma-feats.%J.out
#BSUB -e logs/job_logs/tcga-sigmma-feats.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=24GB:num=1"
#BSUB -n 8
#BSUB -M 96GB
#BSUB -R "select[mem>96GB] rusage[mem=96GB] span[hosts=1]"
set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
config="$run_dir/configs/cell_niche_transfer_rcc_celltype_a01_step2000.yaml"
MANIFEST="${MANIFEST:-$run_dir/configs/tcga_read_manifest.csv}"
out_dir="${OUT_DIR:-/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_read/sigmma_36852_step2000}"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "$run_dir"

python -u "$run_dir/scripts/tcga_tsr/tcga_extract_sigmma_features.py" \
    --config "$config" --manifest "$MANIFEST" --out_dir "$out_dir"
echo "[sigmma-feats] DONE -> $out_dir"
