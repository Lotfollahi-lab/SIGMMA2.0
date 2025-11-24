#!/bin/bash
#BSUB -q normal # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-cellgeni-a100)
#BSUB -G team292 # groupname for billing
#BSUB -o %J_harmony.out # output file
#BSUB -e %J_harmony.err # error file
#BSUB -M 50G  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50G] rusage[mem=50G]' # RAM memory part 1. Default: 100MB
#BSUB -n 1 # number of cores

# activate virtual environment
source /software/cellgen/team292/mm58/venvs/cvpr_env/bin/activate

# run main script
python run_harmony.py

echo "Finished harmony"
