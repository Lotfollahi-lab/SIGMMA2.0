import os
import argparse
import pickle
import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset

from ml_collections.config_dict import ConfigDict
import yaml

import timm

from model.utils import load_dataset, load_model, WandbLogger

from transformers import CLIPProcessor, CLIPModel
import loki

def main(trial):
    config_file = yaml.load(open(args.config_file, "r"), yaml.FullLoader)
    config = ConfigDict(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config.get("model.name")
    print(f"Training model: {model_name}")
    config.outdir = config.get("outdir") if args.out_dir is None else args.out_dir

    if os.path.exists(config.outdir) == False:
        os.makedirs(config.outdir)

    dataset1 = load_dataset(config.dataset1)
    dataset2 = load_dataset(config.dataset2)

    dataset = ConcatDataset([dataset1, dataset2])
    valid_indices = []
    for i in range(len(dataset)):
        spatial = dataset[i]['spatial']
        if not torch.any(torch.isnan(spatial)):
            valid_indices.append(i)
    with open("valid_indices_test.txt", "wb") as f:
        pickle.dump(valid_indices, f)
    with open("valid_indices_test.txt", "rb") as f:
        valid_indices = pickle.load(f)
    test_dataset = torch.utils.data.Subset(dataset, valid_indices)
    print(f"Total samples after filtering: {len(dataset)}")

    batch_size = config.get("train.batch_size")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

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

    model, _, _ = load_model(config, device)
    model.eval()

    # Retrieval


    # PCC, Recall@K calculation
    

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file")
    parser.add_argument("-out_dir", default=None, type=str, help="Output directory for model checkpoints and logs")
    args = parser.parse_args()
    main(args)