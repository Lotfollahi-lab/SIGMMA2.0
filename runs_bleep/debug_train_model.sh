#!/bin/bash

source /software/cellgen/team292/mm58/venvs/cvpr_env/bin/activate

python ../scripts/train_model.py \
    -config_file ../configs/hest1k-LUAD_finetune_BLEEP.yaml \
    -out_dir run_1 
