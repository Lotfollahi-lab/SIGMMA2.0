#!/usr/bin/env bash
# Canonical KIRC TSR->survival with the adopted early-censored exclusion
# (--min_os_days 30): drops event=0 cases followed <30 days (peri-op/registration
# artifacts, e.g. TCGA-A3-A6NN at 3 days); deaths always kept. Re-runs ONLY the
# survival step (CPU, seconds) on the existing TSR csv -> overwrites the top-level
# fixed05/ and median/ headline outputs + canonical KM figs.
#BSUB -J kirc_surv_minos30
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/kirc_surv_minos30.%J.out
#BSUB -e logs/job_logs/kirc_surv_minos30.%J.err
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
MINOS=30

for split in fixed05 median; do
  cut_arg=""; [ "$split" = "fixed05" ] && cut_arg="--stroma_cutoff 0.5"
  out="$run_dir/outputs/tcga_kirc_tsr/${split}"
  fig="$run_dir/figures/clinical_outcome/tso_kirc/${split}"
  mkdir -p "$out" "$fig"
  echo "==================== min_os_days=$MINOS  split=$split ===================="
  "$VENV_PY" -u scripts/tcga_tsr/tcga_survival.py \
      --tsr_csv "$tsr_csv" --survival_tsv "$surv_tsv" --cohort KIRC \
      --min_os_days "$MINOS" \
      --out_dir "$out" --fig_dir "$fig" $cut_arg \
    || echo "[FAILED] split=$split"
done
echo "[kirc-survival min_os_days=$MINOS] DONE"
