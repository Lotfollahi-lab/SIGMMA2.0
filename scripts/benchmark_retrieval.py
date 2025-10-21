# Retrieval evaluation logic adapted from BLIP (https://github.com/salesforce/BLIP/blob/main/eval_retrieval_video.py)

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from ml_collections.config_dict import ConfigDict
import yaml

from model.utils import load_dataset, load_model, retrieval_score, eval_retrieval, WandbLogger
from model.datasets import multimodal_collate

import wandb

def main(args):
    dataset_name = args.config_file.split('/')[-1].split('_')[0]
    config_file = yaml.load(open(args.config_file, "r"), yaml.FullLoader)
    config = ConfigDict(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.outdir = config.get("outdir") if args.out_dir is None else args.out_dir
    if os.path.exists(config.outdir) == False:
        os.makedirs(config.outdir)

    print(f"dataset_name: {dataset_name}")
    g = torch.Generator()
    g.manual_seed(42)

    if config.dataset.split_type == "section_split":
        # == Query dataset preparation ==
        query_dataset = load_dataset(config.dataset.query_dataset)
        sequential_sampler = SequentialSampler(query_dataset)
        query_dataloader = DataLoader(query_dataset, batch_size=config.test.batch_size, sampler=sequential_sampler, num_workers=config.test.num_workers, collate_fn=multimodal_collate)

        # == Reference dataset preparation ==
        target_dataset = load_dataset(config.dataset.reference_dataset)
        reference_dataset = torch.utils.data.ConcatDataset([target_dataset, query_dataset])
        sequential_sampler = SequentialSampler(reference_dataset)
        reference_dataloader = DataLoader(reference_dataset, batch_size=config.test.batch_size, sampler=sequential_sampler, num_workers=config.test.num_workers, collate_fn=multimodal_collate)

    elif config.dataset.split_type == "pooled_FOV_split":
        dataset = load_dataset(config.dataset)
        query_indices = np.load(config.dataset.test.indices, allow_pickle=True)

        # == Reference dataset preparation ==
        reference_dataset = dataset
        random_sampler = RandomSampler(reference_dataset, replacement=False, num_samples=min(len(reference_dataset), config.test.get("num_samples", len(reference_dataset))), generator=g)
        reference_dataloader = DataLoader(reference_dataset, batch_size=config.test.batch_size, sampler=random_sampler, num_workers=config.test.num_workers, generator=g, collate_fn=multimodal_collate)

        # == Query dataset preparation ==
        query_dataset = torch.utils.data.Subset(dataset, query_indices)
        sequential_sampler = SequentialSampler(query_dataset)
        query_dataloader = DataLoader(query_dataset, batch_size=config.test.batch_size, sampler=sequential_sampler, num_workers=config.test.num_workers, generator=g, collate_fn=multimodal_collate)

    # Add wandb logging
    if args.zeroshot:
        wandb_config = {
            "entity": None,
            "project": "img-spatial-alignment",
            "group": None,
            "name": "benchmark_zeroshot",
            "tags": None,
            "mode": "online",
        }
    else:
        wandb_config = {
            "entity": None,
            "project": "img-spatial-alignment",
            "group": None,
            "name": "benchmark",
            "tags": None,
            "mode": "online",
        }

    wandb_logger = WandbLogger(config=config, **wandb_config)
    wandb_logger.set_step(0)

    model = load_model(config, device, pretrained_ckpt_path=config.model.get("checkpoint_path", None), train=False)
    model_name = config.model.get("name")
    model.eval()
    
    # Retrieval
    scores_img2spatial, scores_spatial2img = retrieval_score(model, query_dataloader, reference_dataloader, device, config)

    # Compute recall@K
    result = eval_retrieval(scores_img2spatial, scores_spatial2img)

    table = wandb.Table(columns=["metric", "value"])
    for metric, value in result.items():
        table.add_data(metric, value)
        print(f"{metric}: {value}")
    wandb_logger.update({f"{model_name}_retrieval_results": table})

    # Compute PCC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file")
    parser.add_argument("-out_dir", type=str, default=None, help="Output directory path")
    parser.add_argument("-zeroshot", type=bool, default=False, help="Run in zero-shot mode")
    args = parser.parse_args()

    main(args)