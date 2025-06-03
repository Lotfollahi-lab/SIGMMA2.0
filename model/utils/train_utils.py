import torch
from pathlib import Path


def train_one_epoch(config, logger, model, train_dataloader, eval_dataloader,optimizer, lr_scheduler, global_step, device):
    """
    One epoch of training for the model.
    """
    model.train()

    for i, batch in enumerate(train_dataloader):
        images = batch['image'].to(device) 
        spatial = batch['spatial'].to(device)

        # Forward pass
        loss = model(images, spatial)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if (global_step + 1) % config.experiment.save_every == 0:
            save_path = save_checkpoint(model, config.experiment.output_dir, global_step+1, logger=logger)
        
        if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0:
            eval_loss = evaluate(model, eval_dataloader, device)
            logger.info(f"Eval Loss: {eval_loss}")

        global_step += 1
        
    return global_step

def save_checkpoint(model, optimizer, lr_scheduler, output_dir, global_step, logger=None):
    save_path = Path(output_dir) / f"checkpoint-{global_step}.pth"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'global_step': global_step,
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
            images = batch['image'].to(device)
            spatial = batch['spatial'].to(device)

            loss = model(images, spatial)
            total_loss += loss.item()

    avg_loss = total_loss / len(eval_dataloader)
    return avg_loss