#!/bin/bash
#BSUB -J emt_2d[1-54]
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/emt_2d.%J.%I.out
#BSUB -e logs/job_logs/emt_2d.%J.%I.err
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
# Generate 2D E/M (EMT) score figures for every RCC (KID) section.
# One array index per section; sections without usable cancer/compartment
# labels (donors CV3/4/5/7/8 + serial-slice mismatches) raise SystemExit and
# are skipped — that is expected, not a failure.
#   gene_set = union (full core16 + Mak2016 literature union)
#   outputs  -> figures/cross_tissue_retrieval/emt_2d/
#
# MODE (env var) selects the pooled-anchor workflow pass:
#   MODE=dump    pass 1 — compute scores, dump per-section anchor stats, no figures
#                (then run scripts/emt_2d_reduce_anchors.py to pool the dumps)
#   MODE=render  pass 2 — render figures using the cohort-pooled anchor file (default)
# ---------------------------------------------------------------------------
GENE_SET="${GENE_SET:-union}"
MODE="${MODE:-render}"
SECTIONS_FILE="configs/emt_2d_kid_sections.txt"
DUMP_DIR="figures/cross_tissue_retrieval/emt_2d/anchor_stats"
ANCHOR_FILE="figures/cross_tissue_retrieval/emt_2d/anchor_stats/anchors_${GENE_SET}.json"
SECTION=$(sed -n "${LSB_JOBINDEX}p" "${SECTIONS_FILE}")

echo "[array ${LSB_JOBINDEX}] section=${SECTION} gene_set=${GENE_SET} mode=${MODE}"
if [ -z "${SECTION}" ]; then
    echo "no section for index ${LSB_JOBINDEX}; exiting"
    exit 0
fi

CELL_DIR="figures/cross_tissue_retrieval/emt_2d/cell_scores"
if [ "${MODE}" = "dump" ]; then
    EXTRA="--anchors_only --dump_anchor_stats ../${DUMP_DIR}"
else
    # render mode also banks per-cell tumour-epi scores so dispersion / drift /
    # spatial-front metrics can be recomputed later without re-scoring.
    EXTRA="--anchor_file ../${ANCHOR_FILE} --dump_cell_scores ../${CELL_DIR}"
fi

# emt_2d_score imports its sibling emt_signature_spatial by name, so run from
# the scripts/ dir (matches how the single-section runs were invoked).
cd scripts
python emt_2d_score.py --section "${SECTION}" --gene_set "${GENE_SET}" \
    --outdir ../figures/cross_tissue_retrieval/emt_2d ${EXTRA} \
    || echo "[array ${LSB_JOBINDEX}] ${SECTION} skipped/failed (likely no usable cancer labels)"
