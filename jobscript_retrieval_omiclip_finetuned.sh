#BSUB -J retrieval-omiclip-ft
#BSUB -G s10396
#BSUB -L /usr/bin/bash
#BSUB -o retrieval-omiclip-ft.%J.out
#BSUB -e retrieval-omiclip-ft.%J.err
#BSUB -q training-parallel
#BSUB -gpu "mode=shared:j_exclusive=no:gmem=76GB:num=1"
#BSUB -n 20
#BSUB -M 400GB
#BSUB -R "select[mem>400GB] rusage[mem=400GB] span[hosts=1]"

set -e

module load python-3.11.6

export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=${MASTER_PORT}"

export PATH="/nfs/users/nfs_d/dj16/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_dir="/nfs/team361/dj16/projects/SIGMMA"
output_dir="/lustre/scratch126/cellgen/lotfollahi/dj16/projects/img_spatial_alignment/outputs"

ENV_PATH="/nfs/team361/dj16/.cache/pypoetry/virtualenvs/sigmma-7B1Zyy1z-py3.11/bin/activate"
[ -f $ENV_PATH ] && source $ENV_PATH || echo "Failed to activate: ${ENV_PATH}"

# Standalone retrieval run for fine-tuned OmiCLIP (job 18306, step 1000).
# Runs in parallel with 18322 which is finishing CLIP/PLIP fine-tuned.
cfg="in-house_retrieval_omiclip_finetuned.yaml"
cfg_name="${cfg%.yaml}"
job_outdir="${output_dir}/${cfg_name}_${LSB_JOBID}"
mkdir -p ${job_outdir}
cp ${run_dir}/configs/${cfg} ${job_outdir}/

echo "=========================================="
echo "Running retrieval benchmark with ${cfg}"
echo "Output dir: ${job_outdir}"
echo "=========================================="

python -u scripts/benchmark_retrieval.py \
    -config_file ${run_dir}/configs/${cfg} \
    -out_dir ${job_outdir}
