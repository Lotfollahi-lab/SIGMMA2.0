#!/usr/bin/env bash
# OPTION-A stage 2 (re-run after the .T column-name fix): apply the trained tile
# head to the TCGA-CJ-4637 grid -> tumour/stroma map + tile-count TSR. Bundle +
# grid embeddings already exist; CPU only.
#BSUB -J tile_apply_cj4637
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/tile_apply_cj4637.%J.out
#BSUB -e logs/job_logs/tile_apply_cj4637.%J.err
#BSUB -q oversubscribed
#BSUB -n 2
#BSUB -M 6000
#BSUB -R "select[mem>6000] rusage[mem=6000] span[hosts=1]"
set -e
export PYTHONUNBUFFERED=1
run_dir="/nfs/team361/dj16/projects/SIGMMA"; cd "$run_dir"
cr="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/cache"
ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
source "$ENV_PATH"
bundle="$run_dir/outputs/cell_niche_rcc/tile_tumour_stroma_probe/z_sigmma_concat__tile_tso_classic.joblib"

python -u scripts/tcga_tsr/tcga_grid_tsr.py \
    --grid_data_dir "$cr/tcga_kirc/grid64_data" \
    --grid_feat_dir "$cr/tcga_kirc/grid64_sigmma_36852_step2000" \
    --manifest configs/tcga_kirc_grid_manifest_pilot.csv \
    --bundle "$bundle" \
    --out_csv outputs/tcga_kirc_tsr/grid_pilot/cj4637_grid_tsr.csv \
    --per_tile_dir outputs/tcga_kirc_tsr/grid_pilot/per_tile \
    --fig_dir figures/clinical_outcome/tso_kirc/grid_map \
    --tile_um 64 --field_um 1860 --border_frac 0.25 --min_tumour 10 --measure_frac 0.5
echo "[tile-apply-cj4637] DONE"
