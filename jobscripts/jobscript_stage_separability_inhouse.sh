#!/usr/bin/env bash
# In-house RCC stage-separability (qualitative POC): z_uni vs z_sigmma, pT1 vs pT3+.
# NOT powered (pT1 = 1 donor CV1) -> supervised probe auto-skips; emits UMAP panel.
#   bsub < jobscripts/jobscript_stage_separability_inhouse.sh
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/stage_sep_inhouse.%J.out
#BSUB -e logs/job_logs/stage_sep_inhouse.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6GB
#BSUB -R "select[mem>6GB] rusage[mem=6GB] span[hosts=1]"
set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1

run_dir="/nfs/team361/dj16/projects/SIGMMA"
data_root="/lustre/scratch126/cellgen/lotfollahi/dj16"
UNI_DIR="$data_root/projects/sigmma_outputs/cache/uni/cell_niche_rcc_2026_06_19/uni2h_input224_p14"
SIG_DIR="$data_root/projects/sigmma_outputs/cache/sigmma/cell_niche_rcc_2026_06_19/36852_step2000"
META="$data_root/data/sanger_in_house_xenium/sanger_renal_cell_carcinoma/metadata.csv"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate: ${ENV_PATH}"
cd "$run_dir"

python -u "$run_dir/scripts/tcga_tsr/stage_separability.py" \
    --cohort inhouse \
    --uni_dir "$UNI_DIR" --sigmma_dir "$SIG_DIR" \
    --emb_key macro_cls \
    --metadata "$META" \
    --out_dir "$run_dir/outputs/stage_separability" \
    --fig_dir "$run_dir/figures/clinical_outcome/stage_separability"
echo "[stage_sep] DONE inhouse"
