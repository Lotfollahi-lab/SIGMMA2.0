#!/usr/bin/env bash
# Phase 0a — train-only SVG reselection for the small-panel tissues (ovary/lung/brain).
#
# Reviewer-driven: the published svg_panels_2026_05_29 selected SVGs on the FULL
# manifest (all sections, incl. held-out) — a target-definition leak for the
# panel-specific SUPERVISED readout. Here we reselect each panel's SVG using ONLY
# its train sections.
#
# We REUSE the 0529 per-section Moran's I diagnostics (_diag/<hash>/<sid>.npz) by
# symlinking ONLY the train-section npz into a fresh out_dir, so binning is skipped
# (skip-if-exists) and only the panel aggregation re-runs over the train subset.
# Binning params are identical to 0529, so the reused per-section Moran's I is exact.
set -euo pipefail

ROOT=/nfs/team361/dj16/projects/SIGMMA
SVG0529=/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/svg_panels_2026_05_29
INVENTORY="$ROOT/outputs/diagnostics/panel_inventory_2026_05_29/panel_groups.json"
OUTROOT="$ROOT/outputs/diagnostics/svg_panels_trainonly_2026"
cd "$ROOT"

# Match the 0529 argv exactly (jaccard/consolidate irrelevant for single-hash panels,
# but min_mean_log1p_expr=0.0001 and k=100 matter).
COMMON_ARGS=(
  --panel_groups_json "$INVENTORY"
  --k 100 --bin_size_px 256 --qv_min 20.0 --min_bin_transcripts 5
  --min_present_bins 50 --min_sections_per_panel 5
  --jaccard_threshold 0.5 --min_overlap_threshold 0.95 --consolidate_stable_jaccard 0.5
  --weight_by n_cells
  --min_detection_frac 0.7 --min_morans_i_floor 0.05 --min_morans_i_floor_frac 0.5
  --min_mean_log1p_expr 0.0001
  --n_workers 8
)

for HASH in 1ca7fe57390a a87c22a4bcb9 2df22e67cd31; do
  echo "=================== panel $HASH ==================="
  PANELDIR="$ROOT/outputs/gex_2d/svg0529_panel_${HASH}"
  TRAIN_MANIFEST="$PANELDIR/MANIFEST_panel_${HASH}_train.csv"
  OUT="$OUTROOT/$HASH"
  mkdir -p "$OUT/_diag/$HASH"

  # Symlink ONLY the train-section diagnostics from the 0529 run.
  n_link=0
  while IFS=, read -r sid _rest; do
    [ "$sid" = "sample_name" ] && continue
    src="$SVG0529/_diag/$HASH/${sid}.npz"
    if [ -f "$src" ]; then
      ln -sf "$src" "$OUT/_diag/$HASH/${sid}.npz"
      n_link=$((n_link+1))
    else
      echo "  [warn] no 0529 diag for train section $sid (will be re-binned)"
    fi
  done < "$TRAIN_MANIFEST"
  echo "  symlinked $n_link train-section diagnostics"

  python -m scripts.compute_panel_aware_svgs \
    "${COMMON_ARGS[@]}" \
    --train_manifest "$TRAIN_MANIFEST" \
    --limit_panels "$HASH" \
    --out_dir "$OUT"
done

echo "=================== train-only SVG summary ==================="
for HASH in 1ca7fe57390a a87c22a4bcb9 2df22e67cd31; do
  CSV="$OUTROOT/$HASH/panel_${HASH}/svg_top100.csv"
  if [ -f "$CSV" ]; then
    n=$(( $(wc -l < "$CSV") - 1 ))
    echo "  $HASH: train-only SVG retained = $n genes  ($CSV)"
  else
    echo "  $HASH: NO svg_top100.csv produced — check log"
  fi
done
