import torch
from pathlib import Path
from model.datasets.dataset import SpatialPatchDataset
from model.model import base_contrastive


def load_dataset(config):
    """
    Load the training and evaluation datasets based on the configuration.
    """

    dataset = SpatialPatchDataset(
        image_file=Path(config.dataset.image_file),
        spatial_file=Path(config.dataset.spatial_file),
        indices_file=Path(config.dataset.indices_file),
        fov_size=config.dataset.fov_size,
    )

    return dataset


def load_model(config, image_encoder):
    """
    Load the model, optimizer, and learning rate scheduler based on the configuration.
    """
    model = base_contrastive(config, image_encoder)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.train.lr_step_size,
        gamma=config.train.lr_gamma,
    )

    return model, optimizer, lr_scheduler


def train_one_epoch(
    config,
    model,
    train_dataloader,
    eval_dataloader,
    optimizer,
    lr_scheduler,
    global_step,
    device,
):
    """
    One epoch of training for the model.
    """
    model =  model.to(device)
    model.train()

    for i, batch in enumerate(train_dataloader):
        images = batch["image"].to(device)
        spatial = batch["spatial"].to(device)

        # Forward pass
        with torch.cuda.amp.autocast():
            loss, image_embedding, spatial_embedding = model(images, spatial)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if (global_step + 1) % config.train.save_every == 0:
            save_checkpoint(
                model,
                config.output_dir,
                global_step + 1,
            )

        if (
            eval_dataloader is not None
            and (global_step + 1) % config.train.eval_every == 0
        ):
            eval_loss = evaluate(model, eval_dataloader, device)
            print(f"Eval Loss: {eval_loss}")

        global_step += 1

    return global_step


def save_checkpoint(
    model, optimizer, lr_scheduler, output_dir, global_step, logger=None
):
    save_path = Path(output_dir) / f"checkpoint-{global_step}.pth"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "global_step": global_step,
    }
    torch.save(checkpoint, save_path)
    if logger:
        logger.info(f"Checkpoint saved at {save_path}")
    else:
        print(f"Checkpoint saved at {save_path}")
    return save_path


def evaluate(model, eval_dataloader, device):
    """
    Evaluate the model on the validation dataset.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in eval_dataloader:
            images = batch["image"].to(device)
            spatial = batch["spatial"].to(device)

            loss, image_embedding, spatial_embedding = model(images, spatial)
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_dataloader)
    return avg_loss
