#!/usr/bin/env bash
#BSUB -J kirc_risk_screen
#BSUB -G s10396
#BSUB -q oversubscribed
#BSUB -L /usr/bin/bash
#BSUB -n 1
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000]"
#BSUB -o logs/job_logs/kirc_risk_screen.%J.out
#BSUB -e logs/job_logs/kirc_risk_screen.%J.err

module load python-3.11.6
export PYTHONUNBUFFERED=1
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate $ENV_PATH"

cd /nfs/team361/dj16/projects/SIGMMA

python -m scripts.tcga_tsr.tcga_clinical_risk_screen \
    --clinical_tsv /lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-KIRC_clinical.tsv \
    --out_dir outputs/clinical_outcome \
    --fig_dir figures/clinical_outcome \
    --endpoints OS DSS PFI DFI \
    --cohort KIRC
