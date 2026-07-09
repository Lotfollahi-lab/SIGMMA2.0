#!/usr/bin/env bash
# Re-run KIRC TSR->survival for ALL FOUR TCGA-CDR endpoints (OS/DSS/DFI/PFI),
# not just OS. clinical metadata was updated to carry DSS/DSS_time_days,
# DFI/DFI_time_days, PFI/PFI_time_days (event col is now bare <EP>, e.g. "OS").
# Only the OUTCOME label changes -> TSR/embeddings untouched, survival step only
# (CPU, seconds). Per endpoint x {fixed05, median}:
#   * KM + log-rank + continuous Cox (NOW age + AJCC-stage adjusted)
#   * binary TSR-high-vs-low Cox (M1 / +age / +age+stage) via tcga_cox_group.py
# --min_os_days 30 (early-censored drop) applied to EACH endpoint; events kept.
#BSUB -J tcga_kirc_endpoints
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tcga_kirc_endpoints.%J.out
#BSUB -e logs/job_logs/tcga_kirc_endpoints.%J.err
#BSUB -q oversubscribed
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "select[mem>4000] rusage[mem=4000] span[hosts=1]"
set -euo pipefail
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
VENV_PY="$run_dir/.venv/bin/python"
export PYTHONUNBUFFERED=1

tsr_csv="$run_dir/outputs/tcga_kirc_tsr/tcga_kirc_tsr_3p46mm.csv"
surv_tsv="/lustre/scratch126/cellgen/lotfollahi/dj16/data/public_TCGA/TCGA-KIRC_slide_survival.tsv"
MINOS=30

for ep in OS DSS DFI PFI; do
  ep_l=$(echo "$ep" | tr '[:upper:]' '[:lower:]')
  base="$run_dir/outputs/tcga_kirc_tsr/endpoints/$ep"
  for split in fixed05 median; do
    cut_arg=""; [ "$split" = "fixed05" ] && cut_arg="--stroma_cutoff 0.5"
    out="$base/$split"
    fig="$run_dir/figures/clinical_outcome/tso_kirc/endpoints/$ep/$split"
    mkdir -p "$out" "$fig"
    echo "================= endpoint=$ep split=$split ================="
    "$VENV_PY" -u scripts/tcga_tsr/tcga_survival.py \
        --tsr_csv "$tsr_csv" --survival_tsv "$surv_tsv" --cohort KIRC \
        --endpoint "$ep" --min_os_days "$MINOS" \
        --out_dir "$out" --fig_dir "$fig" $cut_arg \
      || echo "[FAILED] endpoint=$ep split=$split"
  done
  echo "----------------- binary group Cox: $ep -----------------"
  "$VENV_PY" -u scripts/tcga_tsr/tcga_cox_group.py \
      --base "$base" --stem "tcga_kirc_${ep_l}_tsr" \
      --out "$base/cox_group_hr.txt" \
    || echo "[FAILED] cox_group endpoint=$ep"
done
echo "[kirc-endpoints] DONE"
