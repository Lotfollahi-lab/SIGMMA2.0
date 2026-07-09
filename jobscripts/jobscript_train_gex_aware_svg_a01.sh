#BSUB -J train-SIGMMA-gex-svg-a01
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o logs/job_logs/train-SIGMMA-gex-svg-a01.%J.out
#BSUB -e logs/job_logs/train-SIGMMA-gex-svg-a01.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=yes:gmem=76GB:num=2"
#BSUB -n 32
#BSUB -M 600GB
#BSUB -R "select[mem>600GB && hname!='farm-gpu0504'] rusage[mem=600GB] span[hosts=1]"

# Single a01 (alpha=0.1) panel-aware SVG retrain on the corrected pipeline:
#   - hier_cache _2026_05_31 (post pixel->um data fix)
#   - svg_panels_2026_05_29 (unaffected by coord bug; raw-parquet based)
# Derived from jobscript_train_gex_aware_svg_sweep.sh, pinned to a01 only.

set -e

module load python-3.11.6

export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=${MASTER_PORT}"

export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
output_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/sigmma_outputs/checkpoints/sigmma_dense_attnpool_gex_svg_a01"
config_file="${run_dir}/configs/in-house_train_sigmma_dense_attnpool_gex_svg_a01.yaml"

ENV_PATH="/nfs/team361/dj16/projects/SIGMMA/.venv/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

mkdir -p ${output_dir}/${LSB_JOBID}
cp ${config_file} ${output_dir}/${LSB_JOBID}/
torchrun --standalone --nproc_per_node 2 scripts/train_model.py -config_file ${config_file} -out_dir ${output_dir}/${LSB_JOBID}
