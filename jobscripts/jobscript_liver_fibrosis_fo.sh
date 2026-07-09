#!/bin/bash
#BSUB -J liver_fib_fo[1-8]
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/liver_fib_fo.%J.%I.out
#BSUB -e logs/job_logs/liver_fib_fo.%J.%I.err
#BSUB -q normal
#BSUB -n 4
#BSUB -M 100GB
#BSUB -R "select[mem>100GB] rusage[mem=100GB] span[hosts=1]"

set -e

module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH" || echo "Failed to activate $ENV_PATH"
cd "${run_dir}"

# ---------------------------------------------------------------------------
# Liver fibrosis score + spatial fibrotic-region figures for the SGP302
# fresh-frozen (FO) liver cohort (panel d74da03b, carries ACTA2/COL1A1/COL3A1/
# FN1/SPARC). One array index per section.
#   fibrotic mask = top TOP_PCT% of cells by fibrosis score (per-section adaptive)
#   outputs -> figures/cross_tissue_retrieval/liver_fibrosis/
# ---------------------------------------------------------------------------
TOP_PCT="${TOP_PCT:-12}"
SECTIONS_FILE="configs/liver_fibrosis_fo_sections.txt"
OUTDIR="figures/cross_tissue_retrieval/liver_fibrosis"

SECTION=$(sed -n "${LSB_JOBINDEX}p" "${SECTIONS_FILE}")
if [ -z "${SECTION}" ]; then
  echo "no section for index ${LSB_JOBINDEX}"; exit 0
fi
echo "[$(date)] index ${LSB_JOBINDEX} -> ${SECTION}  (top ${TOP_PCT}%)"

cd scripts
python liver_fibrosis_2d.py \
    --section "${SECTION}" \
    --top_pct "${TOP_PCT}" \
    --show_hepatocyte \
    --outdir "../${OUTDIR}"
