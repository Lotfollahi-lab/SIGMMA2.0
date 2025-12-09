# SIGMMA, σ
Spatial transcriptome–histology Image representation learning via hierarchical Graph-based Multi-scale Multi-modal Alignment

## Installation
The environment required to run SIGMMA can be installed like so:

Install the project

- You can install the requirements with [poetry](https://github.com/python-poetry/poetry):  
  `poetry install`

Note. [*STEMO*](https://github.com/Lotfollahi-lab/stemo), [*UNI*](https://github.com/mahmoodlab/UNI), and [*LOKI*](https://github.com/GuangyuWangLab2021/Loki) must be installed separately and their local paths should be provided in the pyproject.toml before environment installation.

## Data Preparation
You can download dataset in the following [link](https://huggingface.co/datasets/AnonymousAuthor42/preprocessed_hest1k_LUAD)

## Model Training
```
python scripts/train_model.py -config_file ${run_dir}/configs/hest1k-LUAD_train_sigmma.yaml -out_dir ${output_dir}/hest1k-LUAD_sigmma
```

Note. [*UNI*](https://huggingface.co/MahmoodLab/UNI) checkpoint must be downloaded separately and its path provided to *pretrained_image_encoder_ckpt_path* in the configuration.

## Downstream Tasks
``` 
# 1. gene expression prediction task
python scripts/benchmark_gex_prediction.py -config_file ${run_dir}/configs/hest1k-LUAD_linear_probing_sigmma.yaml -out_dir ${output_dir}/outputs/hest1k-LUAD_linear_probing_gex_prediction_sigmma

# 2. cross-modal retrieval
python -u scripts/benchmark_retrieval.py -config_file ${run_dir}/configs/hest1k-LUAD_retrieval_sigmma.yaml -out_dir ${output_dir}/hest1k-LUAD_retrieval_sigmma
```

Note. SIGMMA checkpoint from *Model training* step should be provided to checkpoint_path in the configurations.
