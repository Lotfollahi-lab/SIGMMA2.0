#!/bin/bash
#BSUB -q gpu-normal # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-cellgeni-a100)
#BSUB -gpu 'mode=exclusive_process:num=1' # request for exclusive access to gpu 
#BSUB -G team292 # groupname for billing
#BSUB -o %J_train.out # output file
#BSUB -e %J_train.err # error file
#BSUB -M 80G  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>80G] rusage[mem=80G]' # RAM memory part 1. Default: 100MB
#BSUB -n 8 # number of cores

# activate conda environment
module load  cuda-12.1.1 

source /software/cellgen/team292/mm58/venvs/cvpr_env/bin/activate

python ../scripts/train_model.py \
    -config_file ../configs/skin_finetune_BLEEP.yaml \
    -out_dir output_skin_temperature_01

echo "Finished skin"
