#BSUB -J probe_gex_2d
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/probe_gex_2d.%J.out
#BSUB -e logs/job_logs/probe_gex_2d.%J.err
#BSUB -q training-parallel
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
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
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
HEAD_TYPE="${HEAD_TYPE:-mlp}"
LOSS="${LOSS:-mse}"          # mse | lognormal | zilognormal (non-mse needs HEAD_TYPE=mlp)
IN_MEMORY="${IN_MEMORY:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
HVG_METHOD="${HVG_METHOD:-mean}"
TARGET_GRID="${TARGET_GRID:-56}"
CACHE_TARGET_GRID="${CACHE_TARGET_GRID:-56}"
OUT_ROOT="${OUT_ROOT:-}"
PANEL_DIR="${PANEL_DIR:-}"
CACHE_ROOT="${CACHE_ROOT:-}"
SIGMMA_CONFIG="${SIGMMA_CONFIG:-}"

# Only pass --loss when non-default so the legacy mse path stays byte-identical.
loss_arg=()
[ "$LOSS" != "mse" ]    && loss_arg+=(--loss "$LOSS")

extra_args=()
[ -n "$SMOKE_SLIDE" ]   && extra_args+=(--smoke_slide "$SMOKE_SLIDE")
[ -n "$BACKBONE" ]      && extra_args+=(--backbone "$BACKBONE")
[ -n "$INPUT_REPR" ]    && extra_args+=(--input_repr "$INPUT_REPR")
[ -n "$HEAD_TYPE" ]     && extra_args+=(--head_type "$HEAD_TYPE")
[ -n "$IN_MEMORY" ]     && extra_args+=(--in_memory)
[ -n "$OUT_ROOT" ]      && extra_args+=(--out_root "$OUT_ROOT")
[ -n "$PANEL_DIR" ]     && extra_args+=(--panel_dir "$PANEL_DIR")
[ -n "$CACHE_ROOT" ]    && extra_args+=(--cache_root "$CACHE_ROOT")
[ -n "$SIGMMA_CONFIG" ] && extra_args+=(--sigmma_config "$SIGMMA_CONFIG")
extra_args+=("${loss_arg[@]}")
extra_args+=(--hvg_method "$HVG_METHOD")
extra_args+=(--target_grid "$TARGET_GRID")
extra_args+=(--cache_target_grid "$CACHE_TARGET_GRID")

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
    # five input_reprs under head_type=$HEAD_TYPE. Pass criterion is in
    # /nfs/users/nfs_d/dj16/.claude/plans/mutable-hugging-honey.md (median
    # per-gene Pearson >= 0.10 within-slide).
    echo "[smoke0b] slide=$SMOKE_SLIDE head_type=$HEAD_TYPE"
    python -m scripts.probe_gex_2d --phase hvg
    python -m scripts.probe_gex_2d --phase cache --backbone sigmma --smoke_slide "$SMOKE_SLIDE"
    python -m scripts.probe_gex_2d --phase cache --backbone uni    --smoke_slide "$SMOKE_SLIDE"
    for repr in uni concat micro meso macro; do
      echo "[smoke0b] train repr=$repr head=$HEAD_TYPE"
      python -m scripts.probe_gex_2d --phase train --input_repr "$repr" \
          --head_type "$HEAD_TYPE" \
          ${IN_MEMORY:+--in_memory} \
          ${NUM_WORKERS:+--num_workers "$NUM_WORKERS"} \
          --smoke_slide "$SMOKE_SLIDE" \
          --num_steps "$NUM_STEPS" --batch_size "$BATCH_SIZE"
      echo "[smoke0b] eval repr=$repr head=$HEAD_TYPE"
      python -m scripts.probe_gex_2d --phase eval  --input_repr "$repr" \
          --head_type "$HEAD_TYPE" \
          ${IN_MEMORY:+--in_memory} \
          ${NUM_WORKERS:+--num_workers "$NUM_WORKERS"} \
          --smoke_slide "$SMOKE_SLIDE" \
          --batch_size "$BATCH_SIZE"
    done
    if [ "$HEAD_TYPE" = "mlp" ]; then
      echo "[smoke0b] DONE — see outputs/gex_2d/results/*.json"
    else
      echo "[smoke0b] DONE — see outputs/gex_2d/results_${HEAD_TYPE}/*.json"
    fi
    ;;
  smoke0c)
    # Smoke 0c: single slide, ablate {HVG method, target_grid} against smoke 0b.
    # Diagnostic: does a dispersion-based panel or coarser binning lift the
    # Δ(SIGMMA − UNI) signal that smoke 0b found to be null? Cache always
    # binned at CACHE_TARGET_GRID (default 56); train/eval downsamples to
    # TARGET_GRID via sum-pooling.
    #
    # Env contract:
    #   HVG_METHOD         = mean | dispersion
    #   TARGET_GRID        = 56 | 28 | 14 | 7
    #   CACHE_TARGET_GRID  = 56  (default; the cache native resolution)
    #   OUT_ROOT           = outputs/gex_2d/${HVG_METHOD}_g${TARGET_GRID}
    #   PANEL_DIR          = outputs/gex_2d/panels/${HVG_METHOD}
    #   CACHE_ROOT         = .../cache/gex_2d_${HVG_METHOD}
    #
    if [ -z "$OUT_ROOT" ] || [ -z "$PANEL_DIR" ] || [ -z "$CACHE_ROOT" ]; then
      echo "smoke0c requires OUT_ROOT, PANEL_DIR, CACHE_ROOT to be set" >&2
      exit 2
    fi
    echo "[smoke0c] slide=$SMOKE_SLIDE method=$HVG_METHOD target_grid=$TARGET_GRID cache_grid=$CACHE_TARGET_GRID"
    echo "[smoke0c]   panel_dir=$PANEL_DIR cache_root=$CACHE_ROOT out_root=$OUT_ROOT"
    common_args=(
      --hvg_method "$HVG_METHOD"
      --target_grid "$TARGET_GRID"
      --cache_target_grid "$CACHE_TARGET_GRID"
      --panel_dir "$PANEL_DIR"
      --cache_root "$CACHE_ROOT"
      --out_root "$OUT_ROOT"
    )
    [ -n "$SIGMMA_CONFIG" ] && common_args+=(--sigmma_config "$SIGMMA_CONFIG")
    python -m scripts.probe_gex_2d --phase hvg "${common_args[@]}"
    python -m scripts.probe_gex_2d --phase cache --backbone sigmma --smoke_slide "$SMOKE_SLIDE" "${common_args[@]}"
    python -m scripts.probe_gex_2d --phase cache --backbone uni    --smoke_slide "$SMOKE_SLIDE" "${common_args[@]}"
    # Multi-seed head training: N=${NUM_SEEDS:-5} fixed seeds per input_repr,
    # same train/val split (--seed). Eval phase aggregates best_seed*.pt to
    # mean ± std automatically.
    NUM_SEEDS="${NUM_SEEDS:-5}"
    for repr in uni concat macro meso micro; do
      for s in $(seq 0 $((NUM_SEEDS - 1))); do
        echo "[smoke0c] train repr=$repr head=$HEAD_TYPE seed=$s"
        python -m scripts.probe_gex_2d --phase train --input_repr "$repr" \
            --head_type "$HEAD_TYPE" \
            --head_seed "$s" \
            ${IN_MEMORY:+--in_memory} \
            ${NUM_WORKERS:+--num_workers "$NUM_WORKERS"} \
            --smoke_slide "$SMOKE_SLIDE" \
            --num_steps "$NUM_STEPS" --batch_size "$BATCH_SIZE" \
            "${loss_arg[@]}" "${common_args[@]}"
      done
      echo "[smoke0c] eval repr=$repr head=$HEAD_TYPE (aggregating N=$NUM_SEEDS seeds)"
      python -m scripts.probe_gex_2d --phase eval --input_repr "$repr" \
          --head_type "$HEAD_TYPE" \
          ${IN_MEMORY:+--in_memory} \
          ${NUM_WORKERS:+--num_workers "$NUM_WORKERS"} \
          --smoke_slide "$SMOKE_SLIDE" \
          --batch_size "$BATCH_SIZE" \
          "${loss_arg[@]}" "${common_args[@]}"
    done
    res_sub="results"; [ "$LOSS" != "mse" ] && res_sub="results_${LOSS}"
    echo "[smoke0c] DONE — see ${OUT_ROOT}/${res_sub}/*.json"
    ;;
  *)
    echo "unknown PHASE=$PHASE" >&2
    exit 1
    ;;
esac
