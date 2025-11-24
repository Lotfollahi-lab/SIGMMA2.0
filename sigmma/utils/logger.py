# Adapted from https://github.com/apple/ml-4m/blob/main/fourm/utils/logger.py
import wandb
import logging
import torch


class WandbLogger(object):
    def __init__(self, **kwargs):
        wandb.init(
            **kwargs,
            settings=wandb.Settings(_disable_stats=False),
        )

    @staticmethod
    def wandb_safe_log(*args, **kwargs):
        try:
            wandb.log(*args, **kwargs)
        except (wandb.CommError, BrokenPipeError):
            logging.error("wandb logging failed, skipping...")

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, metrics):
        log_dict = dict()
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            log_dict[k] = v

        self.wandb_safe_log(log_dict, step=self.step)

    def flush(self):
        pass

    def finish(self):
        try:
            wandb.finish()
        except (wandb.CommError, BrokenPipeError):
            logging.error("wandb failed to finish")
