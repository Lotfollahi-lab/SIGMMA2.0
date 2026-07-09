#!/usr/bin/env bash
# KIRC TSR->survival sensitivity to the early-censored exclusion (--min_os_days).
# Re-runs ONLY the survival step (CPU, seconds) on the existing TSR csv; no
# seg/feature/predict. Sweeps min_os_days for both fixed-0.5 and median splits.
# Deaths are always kept; only event=0 cases with OS < cut are dropped.
#BSUB -J kirc_os_sens
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/kirc_os_sens.%J.out
#BSUB -e logs/job_logs/kirc_os_sens.%J.err
#BSUB -q oversubscribed
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "select[mem>4000] rusage[mem=4000] span[hosts=1]"
set -e
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
VENV_PY="$run_dir/.venv/bin/python"
export PYTHONUNBUFFERED=1

tsr_csv="$run_dir/outputs/tcga_kirc_tsr/tcga_kirc_tsr_3p46mm.csv"
surv_tsv="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-KIRC_slide_survival.tsv"

for minos in 0 30 45 60 90; do
  for split in fixed05 median; do
    cut_arg=""; [ "$split" = "fixed05" ] && cut_arg="--stroma_cutoff 0.5"
    out="$run_dir/outputs/tcga_kirc_tsr/sens_os${minos}/${split}"
    fig="$run_dir/figures/clinical_outcome/tso_kirc/sens_os${minos}/${split}"
    mkdir -p "$out"
    echo "==================== min_os_days=$minos  split=$split ===================="
    "$VENV_PY" -u scripts/tcga_tsr/tcga_survival.py \
        --tsr_csv "$tsr_csv" --survival_tsv "$surv_tsv" --cohort KIRC \
        --min_os_days "$minos" \
        --out_dir "$out" --fig_dir "$fig" $cut_arg \
      || echo "[FAILED] min_os=$minos split=$split"
  done
done
echo "[os-sensitivity] DONE"
