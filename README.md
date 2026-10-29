# SIGMMA, σ
Spatial transcriptome–histology Image representation learning via hierarchical Graph-based Multi-scale Multi-modal Alignment

## Installation
SIGMMA requires Python 3.11 and PyTorch 2.3.

DGL >= 2.2 is not published on PyPI, so install it from the DGL wheel index first:

```
# GPU (CUDA 12.1)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
# or CPU only
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html

pip install sigmma
```

Optional extras:

| extra | contents |
|---|---|
| `sigmma[analysis]` | downstream analysis used by `scripts/` and `notebooks/` (scvi-tools, liana, gseapy, …) |

Note. The [*UNI*](https://huggingface.co/MahmoodLab/UNI) checkpoint is gated on Hugging Face and must be downloaded separately.

### Development environment
The full research environment (all extras, exact pins, the CUDA DGL wheel and local
[*NEMO*](https://github.com/Lotfollahi-lab/pantheome), [*UNI*](https://github.com/mahmoodlab/UNI)
and [*LOKI*](https://github.com/GuangyuWangLab2021/Loki) checkouts) is the `research` dependency
group. The benchmark baselines (`sigmma/baselines`) are only available from the source repo.
Point the paths under `[tool.uv.sources]` in `pyproject.toml` at your checkouts, then:

```
uv sync
```

### Data paths
Cluster-specific default paths can be overridden with environment variables:
`SIGMMA_CACHE_ROOT`, `SIGMMA_IN_HOUSE_DATA_ROOT`, `SIGMMA_PUBLIC_DATA_ROOT`,
`SIGMMA_NFS_DATA_ROOT`, `SIGMMA_SAMPLE_METADATA_CSV`, `SIGMMA_SECTION_PANEL_CSV`.

## Data Preparation
You can download dataset in the following [link](https://huggingface.co/datasets/AnonymousAuthor42/preprocessed_hest1k_LUAD)

## Model Training
```
torchrun --standalone --nproc_per_node 1 -m sigmma.cli.train -config_file ${run_dir}/configs/hest1k-LUAD_train_sigmma.yaml -out_dir ${output_dir}/hest1k-LUAD_sigmma
```

Note. [*UNI*](https://huggingface.co/MahmoodLab/UNI) checkpoint path should be provided to *pretrained_image_encoder_ckpt_path* in the configuration.

## Downstream Tasks
``` 
# 1. gene expression prediction task
python scripts/benchmark_gex_prediction.py -config_file ${run_dir}/configs/hest1k-LUAD_linear_probing_sigmma.yaml -out_dir ${output_dir}/outputs/hest1k-LUAD_linear_probing_gex_prediction_sigmma

# 2. cross-modal retrieval
sigmma-retrieval -config_file ${run_dir}/configs/hest1k-LUAD_retrieval_sigmma.yaml -out_dir ${output_dir}/hest1k-LUAD_retrieval_sigmma
```

Note. SIGMMA checkpoint from *Model training* step should be provided to checkpoint_path in the configurations.
