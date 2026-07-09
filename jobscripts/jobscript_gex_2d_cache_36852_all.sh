#!/bin/bash
#BSUB -J gex2d-cache-36852-all
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/gex2d-cache-36852-all.%J.out
#BSUB -e logs/job_logs/gex2d-cache-36852-all.%J.err
#BSUB -q inference
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=40GB:num=1"
#BSUB -n 4
#BSUB -M 160GB
#BSUB -R "select[mem>160GB] rusage[mem=160GB] span[hosts=1]"

# ============================================================================
# Rebuild the gex_2d SIGMMA feature cache UNIFIED on ONE checkpoint:
#   sigmma_dense_attnpool_gex_svg_a01/36852/checkpoint-step2000.pth
#   (via configs/cell_niche_transfer_dense_attnpool_a01_step2000.yaml)
#
# The current sigmma/gex_2d/36852_step2000/*.h5 are STALE: 442/446 were computed
# by the (now-deleted) fullpanel_a01/41389_1 checkpoint and merely moved into the
# 36852_step2000 dir; only the 4 IDC sections are genuine 36852. This job
# recomputes ALL sections with 36852.
#
# Section set = every section in the consolidated manifest sanger_MANIFEST.csv
# (450 unique: 439 in-house + 11 public HEST). The sigmma feature cache is
# IMAGE-ONLY (micro/meso/macro patch tokens); it needs FOV bboxes from a data
# cache + the H&E zarr. No spatial graph / hier_cache.
#
# FOV bboxes are panel-INDEPENDENT (iter_fov_bboxes = index_zarr tissue mask +
# fov_size), so:
#   * 336 sections already have bboxes in the svg0529 data cache  -> reuse
#   * the remaining ~114 get a fresh data cache built here (any panel; only the
#     bboxes are consumed, targets are ignored by the sigmma cache).
#
# MODE=verify (default): cache the 4 genuine-36852 IDC sections into a TEMP dir
#   and numerically compare to the existing 36852 h5 -> proves this pipeline
#   reproduces the existing-method output. Touches nothing real.
# MODE=full: backup stale h5, then recompute every section on 36852.
# ============================================================================

set -e
module load python-3.11.6
export PYTHONUNBUFFERED=1
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f "$ENV_PATH" ] && source "$ENV_PATH"
cd "${run_dir}"

MODE="${MODE:-verify}"

cache_root="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
manifest="/lustre/scratch126/cellgen/lotfollahi/dj16/data/sanger_in_house_xenium/sanger_MANIFEST.csv"
sigmma_config="${run_dir}/configs/cell_niche_transfer_dense_attnpool_a01_step2000.yaml"
panel_dir="outputs/gex_2d/svg0529_panel_35a7bd78801c"           # 94-gene panel; only used to satisfy cache_data (targets unused)

sigmma_cache_dir="${cache_root}/sigmma/gex_2d/36852_step2000"
backup_dir="${cache_root}/sigmma/gex_2d/36852_step2000_backup_41389"
data_cache_svg="${cache_root}/data/gex_2d/svg0529_panel_35a7bd78801c_g14_fov256"   # 336 sections, reuse
data_cache_extra="${cache_root}/data/gex_2d/gex2d_extra_g14_fov256"                # NEW, for the ~114 missing

work="outputs/gex_2d/cache_36852_all"
mkdir -p "$work"

cache_args=(--target_grid 14 --cache_target_grid 56 --fov_size 256 --patch_size_hint 14 --panel_dir "$panel_dir")

echo "[cfg] $(grep checkpoint_path "$sigmma_config")"

# --- build split manifests from the consolidated manifest --------------------
# S336 = sections that already have a data cache (reuse svg0529 bboxes)
ls "$data_cache_svg"/*.h5 | xargs -n1 basename | sed 's/\.h5$//' | sort > "$work/have_sections.txt"
head -1 "$manifest" > "$work/manifest_empty.csv"
# have = rows whose section is in S336 ; missing = the rest
awk -F, 'NR==FNR{keep[$1]=1;next} FNR==1{print;next} ($1 in keep)'  "$work/have_sections.txt" "$manifest" > "$work/manifest_have.csv"
awk -F, 'NR==FNR{keep[$1]=1;next} FNR==1{print;next} !($1 in keep)' "$work/have_sections.txt" "$manifest" > "$work/manifest_missing.csv"
# idc = the 4 genuine-36852 reference sections
head -1 "$manifest" > "$work/manifest_idc.csv"
grep -E '^(NCBI783|NCBI785|TENX95|TENX99),' "$manifest" >> "$work/manifest_idc.csv"
echo "[split] have=$(($(wc -l < "$work/manifest_have.csv")-1)) missing=$(($(wc -l < "$work/manifest_missing.csv")-1)) idc=$(($(wc -l < "$work/manifest_idc.csv")-1))"

# ============================================================================
if [ "$MODE" = "verify" ]; then
  echo "=========================================="
  echo "VERIFY: recompute 4 IDC sections on 36852 into TEMP, compare to existing"
  echo "=========================================="
  v_data="${cache_root}/data/gex_2d/_verify_idc_g14_fov256"
  v_sig="${cache_root}/sigmma/gex_2d/_verify_idc_36852"
  rm -rf "$v_data" "$v_sig"

  echo "--- data cache (bboxes) for IDC ---"
  python -m scripts.probe_gex_2d --phase cache --backbone data \
      "${cache_args[@]}" \
      --data_cache_dir "$v_data" \
      --train_manifest "$work/manifest_idc.csv" --val_manifest "$work/manifest_empty.csv"

  echo "--- sigmma cache for IDC (36852) ---"
  python -m scripts.probe_gex_2d --phase cache --backbone sigmma \
      "${cache_args[@]}" \
      --data_cache_dir "$v_data" \
      --sigmma_cache_dir "$v_sig" \
      --sigmma_config "$sigmma_config" \
      --train_manifest "$work/manifest_idc.csv" --val_manifest "$work/manifest_empty.csv"

  echo "--- compare new vs existing 36852 h5 ---"
  python3 - "$v_sig" "$sigmma_cache_dir" NCBI783 NCBI785 TENX95 TENX99 <<'PY'
import sys, os, h5py, numpy as np
newdir, refdir = sys.argv[1], sys.argv[2]
secs = sys.argv[3:]
allok = True
for s in secs:
    a_p, b_p = os.path.join(newdir, s+'.h5'), os.path.join(refdir, s+'.h5')
    if not (os.path.exists(a_p) and os.path.exists(b_p)):
        print(f'{s}: MISSING new={os.path.exists(a_p)} ref={os.path.exists(b_p)}'); allok=False; continue
    with h5py.File(a_p) as a, h5py.File(b_p) as b:
        ok = True; msg = []
        if not np.array_equal(a['fov_id'][:], b['fov_id'][:]):
            ok=False; msg.append('fov_id DIFFER')
        for k in ('micro','meso','macro'):
            va, vb = a[k][:].astype(np.float32), b[k][:].astype(np.float32)
            if va.shape != vb.shape:
                ok=False; msg.append(f'{k} shape {va.shape} vs {vb.shape}'); continue
            md = float(np.max(np.abs(va-vb))) if va.size else 0.0
            msg.append(f'{k} max|Δ|={md:.2e}')
            if md > 1e-2: ok=False
        print(f'{s}: {"IDENTICAL" if ok else "DIFFER"}  ({"; ".join(msg)})')
        allok = allok and ok
print('VERIFY:', 'PASS' if allok else 'FAIL')
PY
  echo "=== verify done (temp caches left at ${v_sig} / ${v_data}) ==="
  exit 0
fi

# ============================================================================
if [ "$MODE" = "full" ]; then
  echo "=========================================="
  echo "FULL: backup stale h5, recompute ALL sections on 36852"
  echo "=========================================="

  # STEP 0: back up stale (41389) h5 so the sigmma cache does not skip them.
  if [ -d "$sigmma_cache_dir" ] && [ ! -d "$backup_dir" ]; then
    echo "[backup] $sigmma_cache_dir -> $backup_dir"
    mv "$sigmma_cache_dir" "$backup_dir"
  fi
  mkdir -p "$sigmma_cache_dir"

  # STEP 1: build data cache (bboxes) for the ~114 sections without one.
  echo "--- STEP 1: data cache for missing sections -> $data_cache_extra ---"
  python -m scripts.probe_gex_2d --phase cache --backbone data \
      "${cache_args[@]}" \
      --data_cache_dir "$data_cache_extra" \
      --train_manifest "$work/manifest_missing.csv" --val_manifest "$work/manifest_empty.csv"

  # STEP 2a: sigmma cache for the 336 sections that reuse the svg0529 data cache.
  echo "--- STEP 2a: sigmma cache (36852) for HAVE sections (svg0529 bboxes) ---"
  python -m scripts.probe_gex_2d --phase cache --backbone sigmma \
      "${cache_args[@]}" \
      --data_cache_dir "$data_cache_svg" \
      --sigmma_cache_dir "$sigmma_cache_dir" \
      --sigmma_config "$sigmma_config" \
      --train_manifest "$work/manifest_have.csv" --val_manifest "$work/manifest_empty.csv"

  # STEP 2b: sigmma cache for the missing sections (new bboxes).
  echo "--- STEP 2b: sigmma cache (36852) for MISSING sections (new bboxes) ---"
  python -m scripts.probe_gex_2d --phase cache --backbone sigmma \
      "${cache_args[@]}" \
      --data_cache_dir "$data_cache_extra" \
      --sigmma_cache_dir "$sigmma_cache_dir" \
      --sigmma_config "$sigmma_config" \
      --train_manifest "$work/manifest_missing.csv" --val_manifest "$work/manifest_empty.csv"

  n=$(ls "$sigmma_cache_dir"/*.h5 2>/dev/null | wc -l)
  echo "=== FULL done: ${n} section h5 in ${sigmma_cache_dir} (backup at ${backup_dir}) ==="
  exit 0
fi

echo "unknown MODE=$MODE (use verify|full)" >&2
exit 1
