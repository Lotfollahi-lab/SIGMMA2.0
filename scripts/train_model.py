import os
import argparse
import pickle
import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset

from ml_collections.config_dict import ConfigDict
import yaml

import timm

from model.utils import train_one_epoch, load_dataset, load_model, evaluate, WandbLogger

import optuna

def objective(trial):
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
    with open("valid_indices.txt", "wb") as f:
        pickle.dump(valid_indices, f)
    with open("valid_indices.txt", "rb") as f:
        valid_indices = pickle.load(f)
    dataset = torch.utils.data.Subset(dataset, valid_indices)
    print(f"Total samples after filtering: {len(dataset)}")

    total = len(dataset)
    train_len = int(0.8 * total)
    val_len = total - train_len

    train_dataset, eval_dataset = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = config.get("train.batch_size")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    # Add wandb logging
    wandb_config = {
        "entity": None,
        "project": "img-spatial-alignment",
        "group": None,
        "name": f"trail_{trial.number}",
        "tags": None,
        "mode": "online",
    }

    config.train.learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    config.train.weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
    config.model.temperature = trial.suggest_uniform("temperature", 0.05, 0.2)


    wandb_logger = WandbLogger(config=config, **wandb_config)
    wandb_logger.set_step(0)

    model, optimizer, lr_scheduler = load_model(config, device)

    global_step, first_epoch, num_train_epochs = (
        0,
        0,
        config.get("train.epochs"),
    )

    for current_epoch in range(first_epoch, num_train_epochs):
        global_step = train_one_epoch(
            config,
            model,
            train_dataloader,
            eval_dataloader,
            optimizer,
            lr_scheduler,
            global_step,
            device=device,
            wandb_logger=wandb_logger,
        )

    eval_loss, _, _ = evaluate(model, eval_dataloader, device)
    return eval_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_file")
    parser.add_argument("-out_dir", default=None, type=str, help="Output directory for model checkpoints and logs")
    args = parser.parse_args()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print("Best trial:", study.best_params)