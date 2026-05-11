#BSUB -J probe_gex_2d
#BSUB -G team361
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/probe_gex_2d.%J.out
#BSUB -e logs/job_logs/probe_gex_2d.%J.err
#BSUB -q gpu-lotfollahi-train
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=40GB:num=1"
#BSUB -n 4
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

# ----------------------------------------------------------------------------
# Driver dispatch: pass any combination of arguments via env vars below, OR
# pass them on the bsub command line as positional args. Defaults shown run
# the smoke 0b path (single slide, both backbones, both reprs).
#
# Examples:
#   # Full Phase A cache for SIGMMA across all train+val slides:
#   bsub < jobscripts/jobscript_probe_gex_2d.sh \
#       PHASE=cache BACKBONE=sigmma SMOKE_SLIDE=
#
#   # Train + eval the concat repr on the cohort:
#   bsub < jobscripts/jobscript_probe_gex_2d.sh \
#       PHASE=train INPUT_REPR=concat SMOKE_SLIDE=
#       bsub < jobscripts/jobscript_probe_gex_2d.sh \
#       PHASE=eval  INPUT_REPR=concat SMOKE_SLIDE=
# ----------------------------------------------------------------------------

PHASE="${PHASE:-smoke0b}"
BACKBONE="${BACKBONE:-}"
INPUT_REPR="${INPUT_REPR:-}"
SMOKE_SLIDE="${SMOKE_SLIDE:-BK24-SKI-27-FO-01-S3}"
NUM_STEPS="${NUM_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

extra_args=()
[ -n "$SMOKE_SLIDE" ] && extra_args+=(--smoke_slide "$SMOKE_SLIDE")
[ -n "$BACKBONE" ]    && extra_args+=(--backbone "$BACKBONE")
[ -n "$INPUT_REPR" ]  && extra_args+=(--input_repr "$INPUT_REPR")

case "$PHASE" in
  hvg|cache|train|eval|all)
    python -m scripts.probe_gex_2d \
        --phase "$PHASE" \
        --num_steps "$NUM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        "${extra_args[@]}"
    ;;
  smoke0b)
    # Smoke 0b: single slide, both backbones cached, then train+eval all
    # four input_reprs. Pass criterion is in
    # /nfs/users/nfs_d/dj16/.claude/plans/mutable-hugging-honey.md (median
    # per-gene Pearson ≥ 0.10 within-slide).
    echo "[smoke0b] slide=$SMOKE_SLIDE"
    python -m scripts.probe_gex_2d --phase hvg
    python -m scripts.probe_gex_2d --phase cache --backbone sigmma --smoke_slide "$SMOKE_SLIDE"
    python -m scripts.probe_gex_2d --phase cache --backbone uni    --smoke_slide "$SMOKE_SLIDE"
    for repr in uni concat micro meso macro; do
      echo "[smoke0b] train repr=$repr"
      python -m scripts.probe_gex_2d --phase train --input_repr "$repr" \
          --smoke_slide "$SMOKE_SLIDE" \
          --num_steps "$NUM_STEPS" --batch_size "$BATCH_SIZE"
      echo "[smoke0b] eval repr=$repr"
      python -m scripts.probe_gex_2d --phase eval  --input_repr "$repr" \
          --smoke_slide "$SMOKE_SLIDE" \
          --batch_size "$BATCH_SIZE"
    done
    echo "[smoke0b] DONE — see outputs/gex_2d/results/*.json"
    ;;
  *)
    echo "unknown PHASE=$PHASE" >&2
    exit 1
    ;;
esac
