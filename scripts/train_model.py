import os
import argparse
import torch
from torch.utils.data import random_split, DataLoader

import numpy as np
from pathlib import Path

from ml_collections.config_dict import ConfigDict
import yaml

from sigmma.utils import train_one_epoch, load_dataset, load_model, evaluate, fix_seed, seed_worker, WandbLogger
from sigmma.datasets import multimodal_collate


def objective(args):
    fix_seed(42)
    dataset_name = args.config_file.split('/')[-1].split('_')[0]
    config_file = yaml.load(open(args.config_file, "r"), yaml.FullLoader)
    config = ConfigDict(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.get("model.pretrained_image_encoder") is None:
        config.model.pretrained_image_encoder = ""

    print(f"dataset_name: {dataset_name}")
    model_name = config.get("model.name")
    print(f"Training model: {model_name}")
    print(f"image_encoder: {config.model.pretrained_image_encoder}")
    config.outdir = config.get("out_dir") if args.out_dir is None else args.out_dir

    if os.path.exists(config.outdir) == False:
        os.makedirs(config.outdir)
    print(config.dataset.type_pretrained_cell_embedding)
    fov_size = config.get("dataset.fov_size", None)
    if config.dataset.split_type == "section_split":
        dataset = load_dataset(config.dataset.train, is_train=True, model_name=model_name)
        train_dataset = dataset
        eval_dataset = load_dataset(config.dataset.validation, is_train=False, model_name=model_name)
    elif config.dataset.split_type == "pooled_FOV_split":
        dataset = load_dataset(config.dataset, is_train=True, model_name=model_name)
        total = len(dataset)

        path_to_dataset = Path(config.dataset.manifest_path).parent
        # Split ratio: 8:1:1
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        train_len = int(total * train_ratio)
        val_len = int(total * val_ratio)

        train_indices = list(range(0, train_len))
        val_indices = list(range(train_len, train_len + val_len))
        test_indices = list(range(train_len + val_len, total))
        if fov_size != 256:
            indices_file = path_to_dataset/f"{dataset_name}_train_indices_{fov_size}.npy"
            if indices_file.exists():
                print(f"Loading {fov_size} pre-saved dataset split indices from {indices_file.parent}")
                train_indices = np.load(path_to_dataset/f"{dataset_name}_train_indices_{fov_size}.npy").tolist()
                val_indices = np.load(path_to_dataset/f"{dataset_name}_val_indices_{fov_size}.npy").tolist()
                test_indices = np.load(path_to_dataset/f"{dataset_name}_test_indices_{fov_size}.npy").tolist()
            else:
                np.save(path_to_dataset/f"{dataset_name}_train_indices_{fov_size}.npy", train_indices)
                np.save(path_to_dataset/f"{dataset_name}_val_indices_{fov_size}.npy", val_indices)
                np.save(path_to_dataset/f"{dataset_name}_test_indices_{fov_size}.npy", test_indices)
        else:
            indices_file = path_to_dataset/f"{dataset_name}_train_indices.npy"
            if indices_file.exists():
                if config.dataset.get("split_version") is not None:
                    print(f"Loading pre-saved dataset split indices version {config.dataset.split_version} from {indices_file.parent}")
                    train_indices = np.load(path_to_dataset/f"{dataset_name}_train_indices_split_{config.dataset.split_version}.npy")
                    val_indices = np.load(path_to_dataset/f"{dataset_name}_val_indices_split_{config.dataset.split_version}.npy")
                    test_indices = np.load(path_to_dataset/f"{dataset_name}_test_indices_split_{config.dataset.split_version}.npy")
                else:
                    print(f"Loading pre-saved dataset split indices from {indices_file.parent}")
                    train_indices = np.load(path_to_dataset/f"{dataset_name}_train_indices.npy").tolist()
                    val_indices = np.load(path_to_dataset/f"{dataset_name}_val_indices.npy").tolist()
                    test_indices = np.load(path_to_dataset/f"{dataset_name}_test_indices.npy").tolist()
            # else:
            #     np.save(path_to_dataset/f"{dataset_name}_train_indices.npy", train_indices)
            #     np.save(path_to_dataset/f"{dataset_name}_val_indices.npy", val_indices)
            #     np.save(path_to_dataset/f"{dataset_name}_test_indices.npy", test_indices) 
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        eval_dataset = torch.utils.data.Subset(dataset, val_indices)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    g = torch.Generator()
    g.manual_seed(42)

    batch_size = config.get("train.batch_size")
    random_sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=config.get("train.max_num_samples_per_epoch", len(train_dataset)), replacement=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=random_sampler, worker_init_fn=seed_worker, generator=g, collate_fn=multimodal_collate, num_workers=config.train.num_workers, drop_last=True)
    sequential_sampler = torch.utils.data.SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, sampler=sequential_sampler, worker_init_fn=seed_worker, generator=g, collate_fn=multimodal_collate, num_workers=config.train.num_workers)

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

    wandb_logger = WandbLogger(config=dict(config.model)|dict(config.train), **wandb_config)
    wandb_logger.set_step(0)

    # Pass dataset to load_model for BLEEP to determine spot_embedding dynamically
    model, optimizer, lr_scheduler = load_model(config, device, pretrained_image_encoder=config.model.pretrained_image_encoder, dataset=dataset)

    global_step, first_epoch, num_train_epochs = (
        0,
        0,
        config.get("train.epochs"),
    )

    for epoch_index in range(first_epoch, num_train_epochs):
        global_step = train_one_epoch(
            config,
            model,
            train_dataloader,
            eval_dataloader,
            optimizer,
            lr_scheduler,
            global_step,
            epoch_index,
            device=device,
            wandb_logger=wandb_logger,
        )

    eval_loss, _, _, _, _, _ = evaluate(config, model, eval_dataloader, device)
    return eval_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file")
    parser.add_argument("-out_dir", default=None, type=str, help="Output directory for model checkpoints and logs")
    parser.add_argument("-stage", type=int, default=1, help="Stage of the training process (1 or 2)")
    args = parser.parse_args()
    #study = optuna.create_study(direction="minimize")
    #study.optimize(objective, n_trials=10)
    #print("Best trial:", study.best_params)

    objective(args)
