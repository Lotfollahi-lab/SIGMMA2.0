#!/usr/bin/env bash
# Dense SIGMMA feature extraction for ALL nuclei of the dense-segmented
# TCGA-BP-5174 (max_cells 0 = no subsample), reading the dense data.h5 via a
# one-row manifest. GPU forward pass (sigmma venv). Feeds the dense PAX8 viz.
#BSUB -J biomarker_dense_feat
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/biomarker_dense_feat.%J.out
#BSUB -e logs/job_logs/biomarker_dense_feat.%J.err
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
manifest="$run_dir/configs/tcga_bp5174_dense_manifest.csv"
out_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache/tcga_kirc/dense/sigmma_36852_step2000"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "$run_dir"

sample="TCGA-BP-5174-01Z-00-DX1.d6f8427f-fbbe-4582-a020-ede5675a7e76"
python -u "$run_dir/scripts/tcga_tsr/tcga_extract_sigmma_features.py" \
    --config "$config" --manifest "$manifest" --slide "$sample" --out_dir "$out_dir" \
    --max_cells 0 --batch_size 64
echo "[feat] DONE dense $sample"
