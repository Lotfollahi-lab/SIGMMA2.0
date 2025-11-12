# Retrieval evaluation logic adapted from BLIP (https://github.com/salesforce/BLIP/blob/main/eval_retrieval_video.py)

import copy
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
    model_name = config.model.get("name")
    print(f"Benchmarking model: {model_name}")
    g = torch.Generator()
    g.manual_seed(42)

    if config.dataset.split_type == "section_split":
        # == Query dataset preparation ==
        query_dataset = load_dataset(config.dataset.query_dataset, is_train=False, model_name=model_name)
        query_sampler = SequentialSampler(query_dataset)
        query_dataloader = DataLoader(query_dataset, batch_size=config.test.batch_size, sampler=query_sampler, num_workers=config.test.num_workers, collate_fn=multimodal_collate)

        # == Reference dataset preparation ==
        reference_dataset = copy.deepcopy(query_dataset)
        reference_sampler = SequentialSampler(reference_dataset)
        reference_dataloader = DataLoader(reference_dataset, batch_size=config.test.batch_size, sampler=reference_sampler, num_workers=config.test.num_workers, collate_fn=multimodal_collate)
        
        # Set dataset for load_model to determine spot_embedding
        dataset = query_dataset

    elif config.dataset.split_type == "pooled_FOV_split":
        dataset = load_dataset(config.dataset, is_train=False, model_name=model_name)
        query_indices = np.load(config.dataset.test.indices, allow_pickle=True)
        target_indices = [i for i in range(len(dataset)) if i not in query_indices]

        # == Query dataset preparation ==
        query_dataset = torch.utils.data.Subset(dataset, query_indices)

        query_sampler = SequentialSampler(query_dataset)
        query_dataloader = DataLoader(query_dataset, batch_size=config.test.batch_size, sampler=query_sampler, num_workers=config.test.num_workers, generator=g, collate_fn=multimodal_collate)

        # # == Reference dataset preparation ==
        reference_dataset = copy.deepcopy(query_dataset)
        reference_sampler = SequentialSampler(reference_dataset) 
        reference_dataloader = DataLoader(reference_dataset, batch_size=config.test.batch_size, sampler=reference_sampler, num_workers=config.test.num_workers, generator=g, collate_fn=multimodal_collate)

    model_name = config.model.get("name")
    # Add wandb logging
    if config.get("model.checkpoint_path", None) is None:
        wandb_config = {"entity": None,
                        "project": f"img-spatial-alignment_{dataset_name}",
                        "group": None,
                        "name": f"benchmark_zeroshot_retrieval_{model_name}",
                        "tags": None,
                        "mode": "online",
                        }
    else:
        wandb_config = {"entity": None,
                        "project": f"img-spatial-alignment_{dataset_name}",
                        "group": None,
                        "name": f"benchmark_retrieval_{model_name}",
                        "tags": None,
                        "mode": "online",
                        }

    wandb_logger = WandbLogger(config=config, **wandb_config)
    wandb_logger.set_step(0)

    # Pass dataset to load_model for BLEEP to determine spot_embedding dynamically
    model = load_model(config, device, pretrained_image_encoder=config.model.get("pretrained_image_encoder", None), pretrained_ckpt_path=config.model.get("checkpoint_path", None), train=False, dataset=dataset)
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