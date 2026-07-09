#BSUB -J train-SIGMMA-gex-fullpanel-sweep[1-4]
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/train-SIGMMA-gex-fullpanel-sweep.%J.%I.out
#BSUB -e logs/job_logs/train-SIGMMA-gex-fullpanel-sweep.%J.%I.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=yes:gmem=76GB:num=2"
#BSUB -n 32
#BSUB -M 600GB
#BSUB -R "select[mem>600GB && hname!='farm-gpu0504'] rusage[mem=600GB] span[hosts=1]"

# Assay-intersection full-gene GEX-aware sweep (replaces the failed use_full_panel
# job 40837). Sections grouped by assay panel (gene_panel_info) with per-panel
# gene axis = intersection of measured var.index (full_panels_2026_06_06). Uses
# the panel-aware SVG loader path (svg_panel_dir + k_max=5050), so it avoids the
# strict identical-var.index requirement that broke use_full_panel mode.
#   index 1 -> alpha 0.1   2 -> alpha 0.2   3 -> alpha 0.4   4 -> alpha 0.8

set -e
module load python-3.11.6
export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=${MASTER_PORT}"
export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${LSB_JOBINDEX}" in
    1) alpha_tag="a01" ;;
    2) alpha_tag="a02" ;;
    3) alpha_tag="a04" ;;
    4) alpha_tag="a08" ;;
    *) echo "Unknown LSB_JOBINDEX=${LSB_JOBINDEX}"; exit 1 ;;
esac

run_dir="/nfs/team361/dj16/projects/SIGMMA"
output_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma_dense_attnpool_gex_fullpanel_${alpha_tag}"
config_file="${run_dir}/configs/in-house_train_sigmma_dense_attnpool_gex_fullpanel_${alpha_tag}.yaml"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

mkdir -p ${output_dir}/${LSB_JOBID}_${LSB_JOBINDEX}
cp ${config_file} ${output_dir}/${LSB_JOBID}_${LSB_JOBINDEX}/
torchrun --standalone --nproc_per_node 2 scripts/train/train_model.py -config_file ${config_file} -out_dir ${output_dir}/${LSB_JOBID}_${LSB_JOBINDEX}
