import os
import argparse
import pickle
import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset

from ml_collections.config_dict import ConfigDict
import yaml

import timm

from model.utils import train_one_epoch, load_dataset, load_model, evaluate, fix_seed, seed_worker, WandbLogger
from model.datasets import multimodal_collate

import optuna

# def objective(trial):
def objective(args):
    fix_seed(42)
    config_file = yaml.load(open(args.config_file, "r"), yaml.FullLoader)
    config = ConfigDict(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config.get("model.name")
    print(f"Training model: {model_name}")
    config.outdir = config.get("out_dir") if args.out_dir is None else args.out_dir

    if os.path.exists(config.outdir) == False:
        os.makedirs(config.outdir)

    if config.dataset.split_type == "section_split":
        train_dataset = load_dataset(config.dataset.train)
        eval_dataset = load_dataset(config.dataset.validation)
    elif config.dataset.split_type == "pooled_FOV_split":
        dataset = load_dataset(config.dataset)
        total = len(dataset)
        train_len = int(0.8 * total)
        val_len = total - train_len

        train_dataset, eval_dataset = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(42),
        )

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
        "project": "img-spatial-alignment",
        "group": None,
        # "name": f"{model_name}_trail_{trial.number}",
        "name": f"{model_name}_trial_test",
        "tags": None,
        "mode": "online",
    }

    # config.train.learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    # config.train.weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
    # config.model.temperature = trial.suggest_uniform("temperature", 0.05, 0.2)

    wandb_logger = WandbLogger(config=dict(config.model)|dict(config.train), **wandb_config)
    wandb_logger.set_step(0)

    model, optimizer, lr_scheduler = load_model(config, device)

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

    eval_loss, _, _ = evaluate(config, model, eval_dataloader, device)
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
