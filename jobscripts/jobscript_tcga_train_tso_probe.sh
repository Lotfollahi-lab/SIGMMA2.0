#!/usr/bin/env bash
# Train the cross-tissue tumour/stroma/other probe on RCC SIGMMA features.
# CPU LogReg over z_sigmma_concat (560k train cells x 1536d) -> reusable bundle
# consumed by tcga_predict_tsr.py. Needs ~64GB (X + scaled copy + lbfgs).
#   bsub < jobscripts/jobscript_tcga_train_tso_probe.sh
#BSUB -J tcga-tso-probe
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga-tso-probe.%J.out
#BSUB -e logs/job_logs/tcga-tso-probe.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=1GB:num=1"
#BSUB -n 8
#BSUB -M 64000
#BSUB -R "select[mem>64000] rusage[mem=64000] span[hosts=1]"
set -e
export PYTHONUNBUFFERED=1

run_dir="/nfs/team361/dj16/projects/SIGMMA"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "$run_dir"

python -u scripts/tcga_tsr/train_tumour_stroma_probe.py \
    --data_dir   "$cr/data/cell_niche_rcc_2026_06_19/scales_64_128_256" \
    --sigmma_dir "$cr/sigmma/cell_niche_rcc_2026_06_19/36852_step2000" \
    --manifest   configs/cell_niche_evaluable_slides_rcc.csv \
    --out_bundle outputs/cell_niche_rcc/tumour_stroma_other_probe/z_sigmma_concat__tso.joblib
echo "[tso-probe] DONE"
