# Retrieval evaluation logic adapted from BLIP (https://github.com/salesforce/BLIP/blob/main/eval_retrieval_video.py)

import os
import argparse
from pathlib import Path
import pandas as pd
import pickle
import copy
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from ml_collections.config_dict import ConfigDict
import yaml

from model.utils import load_dataset, load_model, retrieval_score, eval_retrieval, WandbLogger
from model.datasets import multimodal_collate

def main(args):
    config_file = yaml.load(open(args.config_file, "r"), yaml.FullLoader)
    config = ConfigDict(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.outdir = config.get("outdir") if args.out_dir is None else args.out_dir
    if os.path.exists(config.outdir) == False:
        os.makedirs(config.outdir)
    # == Query dataset preparation ==
    query_dataset = load_dataset(config.query_dataset)
    # valid_indices = []
    # for i in range(len(query_dataset)):
    #     spatial = query_dataset[i]['spatial']
    #     if not torch.any(torch.isnan(spatial)):
    #         valid_indices.append(i)
    # with open("valid_indices_test.txt", "wb") as f:
    #     pickle.dump(valid_indices, f)
    with open("valid_indices_test.txt", "rb") as f:
        valid_indices = pickle.load(f)
    test_dataset = torch.utils.data.Subset(query_dataset, valid_indices)
    print(f"Total samples after filtering: {len(query_dataset)}")

    batch_size = config.get("test.batch_size", 1)
    num_samples = config.get("test.num_samples", len(test_dataset))
    sequential_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=sequential_sampler, num_workers=0, collate_fn=multimodal_collate)

    # Add wandb logging
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

    model = load_model(config, device, train=False)
    model_name = config.model.get("name")
    model.eval()
    reference_path = config.model.get("reference_path", None)
    assert reference_path is not None, f"Reference path for {model_name} is not set."
    
    # Retrieval
    scores_img2spatial, scores_spatial2img = retrieval_score(model, test_dataloader, device, config)
    # Compute recall@K
    result = eval_retrieval(scores_img2spatial, scores_spatial2img, test_dataloader.dataset)
    wandb_logger.update(
        {f"{model_name} image_recall_10": result["image_recall_10"],
         f"{model_name} image_recall_50": result["image_recall_50"],
         f"{model_name} image_recall_100": result["image_recall_100"],
         f"{model_name} spatial_recall_10": result["spatial_recall_10"],
         f"{model_name} spatial_recall_50": result["spatial_recall_50"],
         f"{model_name} spatial_recall_100": result["spatial_recall_100"]}
    )
    # Compute PCC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file")
    parser.add_argument("-out_dir", type=str, default=None, help="Output directory path")
    args = parser.parse_args()
    main(args)