# Retrieval evaluation logic adapted from BLIP (https://github.com/salesforce/BLIP/blob/main/eval_retrieval_video.py)

import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from ml_collections.config_dict import ConfigDict
import yaml

from model.utils import load_dataset, load_model, linear_probing, WandbLogger
from model.model import UNI, H_optimus_zero, ResNet50
from model.datasets import multimodal_collate

from huggingface_hub import login

from scipy.stats import pearsonr

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

    # == Train dataset preparation ==
    if config.dataset.split_type == "section_split":
        train_dataset = load_dataset(config.dataset.train)
        random_sampler = RandomSampler(train_dataset, replacement=False, num_samples=min(len(train_dataset), config.test.get("num_samples", len(train_dataset))), generator=g)
        train_dataloader = DataLoader(train_dataset, batch_size=config.test.batch_size, sampler=random_sampler, num_workers=config.test.num_workers, generator=g, collate_fn=multimodal_collate)

        # == Test dataset preparation ==
        test_dataset = load_dataset(config.dataset.test)
        sequential_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=config.test.batch_size, sampler=sequential_sampler, num_workers=config.test.num_workers, generator=g, collate_fn=multimodal_collate)

    elif config.dataset.split_type == "pooled_FOV_split":
        dataset = load_dataset(config.dataset)
        train_indices = np.load(config.dataset.train.indices, allow_pickle=True)
        test_indices = np.load(config.dataset.test.indices, allow_pickle=True)

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        random_sampler = RandomSampler(train_dataset, replacement=False, num_samples=min(len(train_dataset), config.test.get("num_samples", len(train_dataset))), generator=g)
        train_dataloader = DataLoader(train_dataset, batch_size=config.test.batch_size, sampler=random_sampler, num_workers=config.test.num_workers, generator=g, collate_fn=multimodal_collate)

        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        sequential_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=config.test.batch_size, sampler=sequential_sampler, num_workers=config.test.num_workers, generator=g, collate_fn=multimodal_collate)

    # Add wandb logging
    wandb_config = {"entity": None,
                    "project": f"img-spatial-alignment_{dataset_name}",
                    "group": None,
                    "name": "benchmark_gex_prediction",
                    "tags": None,
                    "mode": "online",
                }

    wandb_logger = WandbLogger(config=config, **wandb_config)
    wandb_logger.set_step(0)

    model_name = config.model.get("name")
    multimodal_model = load_model(config, device, pretrained_image_encoder=config.model.get("pretrained_image_encoder", None), pretrained_ckpt_path=config.model.get("checkpoint_path", None), train=False)
    multimodal_model.eval()

    unimodal_image_encoder_UNI = UNI(checkpoint_path="/nfs/team361/dj16/projects/img_spatial_alignment/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/")
    unimodal_image_encoder_UNI.to(device)
    unimodal_image_encoder_UNI.eval()

    unimodal_image_encoder_H_optimus_zero = H_optimus_zero()
    unimodal_image_encoder_H_optimus_zero.to(device)
    unimodal_image_encoder_H_optimus_zero.eval()

    unimodal_image_encoder_ResNet50 = ResNet50()
    unimodal_image_encoder_ResNet50.to(device)
    unimodal_image_encoder_ResNet50.eval()

    # === Patch-level gene expression ===
    # Image encoder only
    eval_unimodal_UNI = linear_probing(train_dataloader, test_dataloader, unimodal_image_encoder_UNI, device, predicted_label="topK_exp", norm='log1p', use_pca=True, method='ridge')
    eval_unimodal_H_optimus_zero = linear_probing(train_dataloader, test_dataloader, unimodal_image_encoder_H_optimus_zero, device, predicted_label="topK_exp", norm='log1p', use_pca=True, method='ridge')
    eval_unimodal_ResNet50 = linear_probing(train_dataloader, test_dataloader, unimodal_image_encoder_ResNet50, device, predicted_label="topK_exp", norm='log1p', use_pca=True, method='ridge')
    table = wandb.Table(columns=["metric", "value"])

    print("uni-modal image encoder: UNI")
    for metric, value in eval_unimodal_UNI.items():
        if isinstance(value, list):
            continue
        table.add_data(f"unimodal_UNI_{metric}", value)
        print(f"{metric}: {value}")

    print("uni-modal image encoder: H-optimus_zero")
    for metric, value in eval_unimodal_H_optimus_zero.items():
        if isinstance(value, list):
            continue
        table.add_data(f"unimodal_H_optimus_zero_{metric}", value)
        print(f"{metric}: {value}")

    print("uni-modal image encoder: ResNet50")
    for metric, value in eval_unimodal_ResNet50.items():
        if isinstance(value, list):
            continue
        table.add_data(f"unimodal_ResNet50_{metric}", value)
        print(f"{metric}: {value}")

    # Image encoder from multimodal learning model
    eval_multimodal = linear_probing(train_dataloader, test_dataloader, multimodal_model, device, predicted_label="topK_exp", norm='log1p', use_pca=True, method='ridge')

    print(f"multi-modal model image encoder: {model_name}")
    print("pretrained_image_encoder:", config.model.get("pretrained_image_encoder", "None"))
    for metric, value in eval_multimodal.items():
        if isinstance(value, list):
            continue
        table.add_data(f"multimodal_{metric}", value)
        print(f"{metric}: {value}")
    wandb_logger.update({f"{model_name}_linear_probing_results": table})
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file")
    parser.add_argument("-out_dir", type=str, default=None, help="Output directory path")
    parser.add_argument("-zeroshot", type=bool, default=False, help="Run in zero-shot mode")
    args = parser.parse_args()

    main(args)