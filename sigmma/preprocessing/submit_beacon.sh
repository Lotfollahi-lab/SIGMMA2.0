#!/bin/bash
#BSUB -q yesterday # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-cellgeni-a100)
#BSUB -G team361 # groupname for billing
#BSUB -o %J_harmony_beacon.out # output file
#BSUB -e %J_harmony_beacon.err # error file
#BSUB -M 100G  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>100G] rusage[mem=100G]' # RAM memory part 1. Default: 100MB
#BSUB -n 1 # number of cores

# activate virtual environment
source /software/cellgen/team292/mm58/venvs/cvpr_env/bin/activate

# run main script
python run_harmony_beacon.py

echo "Finished harmony_beacon"
