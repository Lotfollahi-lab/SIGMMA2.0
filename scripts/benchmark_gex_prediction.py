# Retrieval evaluation logic adapted from BLIP (https://github.com/salesforce/BLIP/blob/main/eval_retrieval_video.py)

import os
import argparse
from pathlib import Path
import pandas as pd
import pickle
import copy
import timm

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from ml_collections.config_dict import ConfigDict
import yaml

from model.utils import load_dataset, load_model, linear_probing, WandbLogger
from model.model import UNI
from model.datasets import multimodal_collate

from scipy.stats import pearsonr

import wandb

def main(args):
    config_file = yaml.load(open(args.config_file, "r"), yaml.FullLoader)
    config = ConfigDict(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.outdir = config.get("outdir") if args.out_dir is None else args.out_dir
    if os.path.exists(config.outdir) == False:
        os.makedirs(config.outdir)

    g = torch.Generator()
    g.manual_seed(42)

    # == Train dataset preparation ==
    train_dataset = load_dataset(config.train_dataset)
    random_sampler = RandomSampler(train_dataset, replacement=False, num_samples=min(len(train_dataset), config.test.get("num_samples", len(train_dataset))), generator=g)
    train_dataloader = DataLoader(train_dataset, batch_size=config.test.batch_size, sampler=random_sampler, num_workers=config.test.num_workers, generator=g, collate_fn=multimodal_collate)

    # == Test dataset preparation ==
    test_dataset = load_dataset(config.test_dataset)
    sequential_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=config.test.batch_size, sampler=sequential_sampler, num_workers=config.test.num_workers, generator=g, collate_fn=multimodal_collate)

    # Add wandb logging
    wandb_config = {"entity": None,
                    "project": "img-spatial-alignment",
                    "group": None,
                    "name": "benchmark_gex_prediction",
                    "tags": None,
                    "mode": "online",
                }

    wandb_logger = WandbLogger(config=config, **wandb_config)
    wandb_logger.set_step(0)

    model_name = config.model.get("name")
    multimodal_model = load_model(config, device, pretrained_ckpt_path=config.model.get("checkpoint_path", None), train=False)
    multimodal_model.eval()

    unimodal_image_encoder = UNI(checkpoint_path=config.model.pretrained_image_encoder_ckpt_path)
    unimodal_image_encoder.to(device)
    unimodal_image_encoder.eval()
    
    # === Patch-level gene expression === 
    # Image encoder from multimodal learning model
    eval_multimodal = linear_probing(train_dataloader, test_dataloader, multimodal_model, device, predicted_label="topK_exp", log1p_exp_transform=True, use_pca=True, method='ridge')

    # Image encoder only
    eval_unimodal = linear_probing(train_dataloader, test_dataloader, unimodal_image_encoder, device, predicted_label="topK_exp", log1p_exp_transform=True, use_pca=True, method='ridge')
    table = wandb.Table(columns=["metric", "value"])
    for metric, value in eval_multimodal.items():
        if isinstance(value, list):
            continue
        table.add_data(f"multimodal_{metric}", value)
        print(f"{metric}: {value}")
    for metric, value in eval_unimodal.items():
        if isinstance(value, list):
            continue
        table.add_data(f"unimodal_{metric}", value)
        print(f"{metric}: {value}")
    wandb_logger.update({f"{model_name}_linear_probing_results": table})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file")
    parser.add_argument("-out_dir", type=str, default=None, help="Output directory path")
    parser.add_argument("-zeroshot", type=bool, default=False, help="Run in zero-shot mode")
    args = parser.parse_args()

    main(args)