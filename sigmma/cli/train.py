"""`sigmma-train`: train SIGMMA (or a baseline) from a YAML config.

Runs under torchrun (NCCL process group is always initialised):
  torchrun --standalone --nproc_per_node 2 -m sigmma.cli.train -config_file CFG -out_dir DIR
"""
import os
import argparse
import torch
from torch.utils.data import random_split, DataLoader

import numpy as np
from pathlib import Path

from ml_collections.config_dict import ConfigDict
import yaml

from sigmma.utils import train, load_dataset, load_model, evaluate, fix_seed, seed_worker, WandbLogger
from sigmma.datasets import multimodal_collate

import torch.distributed as dist

def setup():
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True

def cleanup():
    dist.destroy_process_group()

def objective(args, rank=None, world_size=None):
    fix_seed(42)
    dataset_name = args.config_file.split('/')[-1].split('_')[0]
    config_file = yaml.load(open(args.config_file, "r"), yaml.FullLoader)
    config = ConfigDict(config_file)

    device = torch.device("cuda", rank)

    if config.get("model.pretrained_image_encoder") is None:
        config.model.pretrained_image_encoder = ""

    model_name = config.get("model.name")
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"dataset_name: {dataset_name}")
        print(f"Training model: {model_name}")
        print(f"image_encoder: {config.model.pretrained_image_encoder}")

    config.outdir = config.get("out_dir") if args.out_dir is None else args.out_dir

    os.makedirs(config.outdir, exist_ok=True)
    
    if config.dataset.split_type == "section_split":
        dataset = load_dataset(config.dataset.train, is_train=True, model_name=model_name)
        train_dataset = dataset
        eval_dataset = load_dataset(config.dataset.validation, is_train=False, model_name=model_name)
    elif config.dataset.split_type == "spatially_stratified_tile_split":
        dataset = load_dataset(config.dataset, is_train=True, model_name=model_name)

        train_indices_file = Path(config.dataset.train.indices)
        val_indices_file = Path(config.dataset.validation.indices)

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Loading pre-saved dataset split indices from {train_indices_file.parent}")

        train_indices = np.load(train_indices_file).tolist()
        val_indices = np.load(val_indices_file).tolist()
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        eval_dataset = torch.utils.data.Subset(dataset, val_indices)

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")

    g = torch.Generator()
    g.manual_seed(42)

    batch_size = config.get("train.batch_size")
    prefetch_factor = int(config.train.get("prefetch_factor", 4))
    # Train dataloader: full dataset, reshuffled each pass via sampler.set_epoch in the training loop
    if rank is not None:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset, generator=g)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, worker_init_fn=seed_worker, generator=g, collate_fn=multimodal_collate, num_workers=config.train.num_workers, drop_last=True, persistent_workers=config.train.num_workers > 0, pin_memory=True, prefetch_factor=prefetch_factor if config.train.num_workers > 0 else None)

    # Eval dataloader: fixed subset for reproducible evaluation
    num_eval_samples = config.train.get("max_num_eval_samples", len(eval_dataset))
    indices_eval = np.random.default_rng(0).choice(len(eval_dataset), size=min(num_eval_samples, len(eval_dataset)), replace=False)
    eval_subset = torch.utils.data.Subset(eval_dataset, indices_eval)
    if rank is not None:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_subset, shuffle=False, drop_last=False)
        eval_dataloader = DataLoader(eval_subset, batch_size=batch_size, sampler=eval_sampler, worker_init_fn=seed_worker, generator=g, collate_fn=multimodal_collate, num_workers=config.train.num_workers, drop_last=False, persistent_workers=config.train.num_workers > 0, pin_memory=True, prefetch_factor=prefetch_factor if config.train.num_workers > 0 else None)
    else:
        eval_dataloader = DataLoader(eval_subset, batch_size=batch_size, worker_init_fn=seed_worker, generator=g, collate_fn=multimodal_collate, num_workers=config.train.num_workers, shuffle=False, drop_last=False, persistent_workers=config.train.num_workers > 0, pin_memory=True, prefetch_factor=prefetch_factor if config.train.num_workers > 0 else None)

    # Add wandb logging
    wandb_config = {
        "entity": None,
        "project": f"img-spatial-alignment_{dataset_name}",
        "group": None,
        "name": f"{model_name}_trial_{dataset_name}_{config.model.pretrained_image_encoder}",
        "tags": None,
        "mode": "online",
    }

    # lr_candidates = [0.00001, 0.0001, 0.001]
    # lr_stepsize_candidates = [10, 50, 100]
    # wd_candidates = [0.00001, 0.0001, 0.001, 0.01]
    # temp_candidates = [0.01, 0.05, 0.1, 0.15]

    # config.train.learning_rate = trial.suggest_categorical("lr", lr_candidates)
    # config.train.lr_stepsize = trial.suggest_categorical("lr_stepsize", lr_stepsize_candidates)
    # config.train.weight_decay = trial.suggest_categorical("weight_decay", wd_candidates)
    # config.model.temperature = trial.suggest_categorical("temperature", temp_candidates)

    if int(os.environ.get("RANK", "0")) == 0: 
        wandb_logger = WandbLogger(config=dict(config.model)|dict(config.train), **wandb_config)
        wandb_logger.set_step(0)
    else:
        wandb_logger = None

    model, optimizer, lr_scheduler = load_model(config, pretrained_image_encoder=config.model.pretrained_image_encoder, rank=rank)

    train(
        config,
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        lr_scheduler,
        max_steps=config.get("train.max_steps"),
        device=device,
        wandb_logger=wandb_logger,
    )

    eval_loss, _, _, _, _, _ = evaluate(config, model, eval_dataloader, device)
    cleanup()
    return eval_loss


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file")
    parser.add_argument("-out_dir", default=None, type=str, help="Output directory for model checkpoints and logs")
    parser.add_argument("-stage", type=int, default=1, help="Stage of the training process (1 or 2)")
    args = parser.parse_args(argv)
    #study = optuna.create_study(direction="minimize")
    #study.optimize(objective, n_trials=10)
    #print("Best trial:", study.best_params)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    setup()
    objective(args, rank=local_rank, world_size=world_size)


if __name__ == "__main__":
    main()
