#!/bin/bash
#BSUB -J cross_tissue_extract
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/cross_tissue_extract.%J.out
#BSUB -e logs/job_logs/cross_tissue_extract.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=40GB:num=1"
#BSUB -n 8
#BSUB -M 80GB
#BSUB -R "select[mem>80GB] rusage[mem=80GB] span[hosts=1]"

set -e

module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate $ENV_PATH"
cd "${run_dir}"

# ---------------------------------------------------------------------------
# Drive Phase A1 cross-tissue patch-level embedding extraction.
#
# Tunables via env vars:
#   TISSUE       — one of {skin, glioblastoma, liver, lung, pancreas,
#                  pediatric_ovary, renal_cell_carcinoma, whole_embryo, all}
#   MAX_SLIDES   — cap test slides per tissue (smoke convenience). Empty = all.
#   BATCH_SIZE   — DataLoader batch size (default 128).
#   NUM_WORKERS  — DataLoader workers (default 8).
#   CONFIG       — retrieval YAML to clone (default in-house_retrieval_sigmma.yaml).
#   OUT_DIR      — embedding output dir.
# ---------------------------------------------------------------------------
TISSUE="${TISSUE:-renal_cell_carcinoma}"
MAX_SLIDES="${MAX_SLIDES:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CONFIG="${CONFIG:-configs/in-house_retrieval_sigmma.yaml}"
OUT_DIR="${OUT_DIR:-outputs/cross_tissue_retrieval/embeddings}"

extra_args=()
[ -n "$MAX_SLIDES" ] && extra_args+=(--max_slides "$MAX_SLIDES")

python -m scripts.cross_tissue_extract_patches \
    --config "$CONFIG" \
    --tissue "$TISSUE" \
    --out_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    "${extra_args[@]}"
