#!/usr/bin/env bash
# KIRC z_uni control (array): raw UNI2-h feature extraction, one slide per task.
# Forward-pass only -> inference queue (see memory: inference-queue-for-inference-jobs).
#   N=$(($(wc -l < configs/tcga_kirc_manifest.csv) - 1))
#   bsub -J "tcga_kirc_uni[1-$N]%24" < jobscripts/jobscript_tcga_kirc_uni_features_array.sh
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_kirc_uni.%J.%I.out
#BSUB -e logs/job_logs/tcga_kirc_uni.%J.%I.err
#BSUB -q inference
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
MANIFEST="$run_dir/configs/tcga_kirc_manifest.csv"
out_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_kirc/uni_2026_06_19"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "$run_dir"

i=$((LSB_JOBINDEX + 1))
sample=$(sed -n "${i}p" "$MANIFEST" | cut -d, -f1 | tr -d '\r')
data_h5=$(sed -n "${i}p" "$MANIFEST" | cut -d, -f4 | tr -d '\r')
if [ ! -f "$data_h5" ]; then echo "[skip] $sample: no data_h5 (seg skipped)"; exit 0; fi
if [ -f "$out_dir/$sample.h5" ]; then echo "[skip] $sample features exist"; exit 0; fi

python -u "$run_dir/scripts/tcga_tsr/tcga_extract_uni_features.py" \
    --config "$config" --manifest "$MANIFEST" --slide "$sample" --out_dir "$out_dir" \
    --max_cells 8000 --batch_size 64
echo "[uni] DONE $sample"
