#!/usr/bin/env bash
# TCGA-KIRC stage-separability (POWERED): z_uni vs z_sigmma, pT1 vs pT3+.
# Patient-grouped CV AUROC + paired (sigmma-uni) delta + UMAP/silhouette.
# Submit AFTER the z_uni array ends:
#   bsub -w "ended(<UNI_ARRAY_JOBID>)" < jobscripts/jobscript_stage_separability_tcga.sh
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/stage_sep_tcga.%J.out
#BSUB -e logs/job_logs/stage_sep_tcga.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6GB
#BSUB -R "select[mem>6GB] rusage[mem=6GB] span[hosts=1]"
set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1

run_dir="/nfs/team361/dj16/projects/SIGMMA"
data_root="/lustre/scratch126/cellgen/lotfollahi/dj16"
UNI_DIR="$data_root/projects/sigmma_outputs/cache/tcga_kirc/uni_2026_06_19"
SIG_DIR="$data_root/projects/sigmma_outputs/cache/tcga_kirc/sigmma_36852_step2000"
CLIN="$data_root/data/public_TCGA/TCGA-KIRC_clinical.tsv"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "$run_dir"

n_uni=$(ls "$UNI_DIR"/*.h5 2>/dev/null | wc -l)
echo "[stage_sep] z_uni cached slides: $n_uni"

python -u "$run_dir/scripts/tcga_tsr/stage_separability.py" \
    --cohort tcga \
    --uni_dir "$UNI_DIR" --sigmma_dir "$SIG_DIR" \
    --emb_key macro_cls \
    --clinical "$CLIN" \
    --out_dir "$run_dir/outputs/stage_separability" \
    --fig_dir "$run_dir/figures/clinical_outcome/stage_separability" \
    --n_splits 5 --n_repeats 10 --seed 42
echo "[stage_sep] DONE tcga"
