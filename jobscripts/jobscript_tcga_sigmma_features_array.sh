#!/usr/bin/env bash
# Stage B (array): SIGMMA feature extraction, one slide per array task, so 166
# slides extract in parallel. Each task loads the model once and runs --slide.
#   N=$(($(wc -l < configs/tcga_read_manifest.csv) - 1))
#   bsub -w "done(SEGJOBID)" -J "tcga_feat[1-$N]%24" < jobscripts/jobscript_tcga_sigmma_features_array.sh
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_feat.%J.%I.out
#BSUB -e logs/job_logs/tcga_feat.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=24GB:num=1"
#BSUB -n 4
#BSUB -M 64GB
#BSUB -R "select[mem>64GB] rusage[mem=64GB] span[hosts=1]"
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

i=$((LSB_JOBINDEX + 1))
sample=$(sed -n "${i}p" "$MANIFEST" | cut -d, -f1)
data_h5=$(sed -n "${i}p" "$MANIFEST" | cut -d, -f4)
if [ ! -f "$data_h5" ]; then echo "[skip] $sample: no data_h5 (seg skipped)"; exit 0; fi
if [ -f "$out_dir/$sample.h5" ]; then echo "[skip] $sample features exist"; exit 0; fi

python -u "$run_dir/scripts/tcga_tsr/tcga_extract_sigmma_features.py" \
    --config "$config" --manifest "$MANIFEST" --slide "$sample" --out_dir "$out_dir" \
    --max_cells 8000 --batch_size 64
echo "[feat] DONE $sample"
