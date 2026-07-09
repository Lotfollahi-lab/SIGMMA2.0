"""Shared training loop for H&E→ST regressor baselines.

Each baseline (ST-Net / HisToGene / DeepPT) wires up its own model + image
transform and reuses this trainer. The training target is the FOV-aggregated
log1p gene-count vector produced by ``FovGexDataset``.

Loss is MSE on log1p counts. Optionally adds an L2 reconstruction term for
DeepPT's autoencoder (controlled by ``recon_lambda``).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import FovGexDataset, collate_baseline


def _pearson_per_gene(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """pred/target: (N, G) → (G,) per-gene Pearson with NaN-safe handling."""
    G = pred.shape[1]
    out = np.full(G, np.nan, dtype=np.float64)
    for g in range(G):
        p = pred[:, g]
        t = target[:, g]
        sp = p.std()
        st = t.std()
        if sp < 1e-8 or st < 1e-8:
            continue
        out[g] = float(((p - p.mean()) * (t - t.mean())).mean() / (sp * st))
    return out


def train_baseline(
    *,
    model: nn.Module,
    image_transform: Callable,
    extra_forward_args: Optional[Callable] = None,
    train_dataset: FovGexDataset,
    val_dataset: Optional[FovGexDataset],
    out_dir: str,
    num_steps: int = 5000,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 8,
    log_every: int = 100,
    val_every: int = 500,
    recon_lambda: float = 0.0,
    device: Optional[torch.device] = None,
    wandb_run=None,
) -> dict:
    """Train ``model`` on ``train_dataset``; evaluate on ``val_dataset``.

    ``image_transform(image_uint8, device) -> Tensor`` produces model input
    (e.g. ImageNet-normalised 224×224 for ST-Net/DeepPT, flattened patches
    for HisToGene).

    ``extra_forward_args(batch_size, device) -> dict`` supplies any extra
    kwargs to ``model(image, **extra_forward_args(...))`` — used by HisToGene
    which needs ``centers``.

    ``recon_lambda > 0`` activates DeepPT's autoencoder reconstruction loss
    (requires the model's ``forward(x, return_recon=True)`` to return
    ``(pred, feats, recon)``).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_steps)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True,
        collate_fn=collate_baseline, persistent_workers=num_workers > 0,
    )

    history = {"step": [], "train_loss": [], "val_step": [], "val_loss": [],
               "val_pearson_median": []}
    step = 0
    best_val = math.inf
    best_state = None
    t0 = time.time()

    while step < num_steps:
        for images, targets, _metas in train_loader:
            if step >= num_steps:
                break
            images = image_transform(images, device=device)
            targets = targets.to(device, non_blocking=True)
            kwargs = extra_forward_args(images.shape[0], device=device) if extra_forward_args else {}

            optim.zero_grad()
            if recon_lambda > 0.0:
                pred, feats, recon = model(images, return_recon=True, **kwargs)
                pred_flat = pred.view(pred.shape[0], -1) if pred.ndim > 2 else pred
                loss_mse = F.mse_loss(pred_flat, targets)
                loss_recon = F.mse_loss(recon, feats.detach())
                loss = loss_mse + recon_lambda * loss_recon
            else:
                pred = model(images, **kwargs)
                pred_flat = pred.view(pred.shape[0], -1) if pred.ndim > 2 else pred
                loss = F.mse_loss(pred_flat, targets)

            loss.backward()
            optim.step()
            scheduler.step()

            if step % log_every == 0:
                history["step"].append(step)
                history["train_loss"].append(float(loss.detach().cpu()))
                print(f"[step {step:5d}/{num_steps}] loss={float(loss):.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
                if wandb_run is not None:
                    wandb_run.log({"train/loss": float(loss),
                                   "train/lr": scheduler.get_last_lr()[0],
                                   "step": step})

            if val_dataset is not None and len(val_dataset) > 0 and (step + 1) % val_every == 0:
                vmetrics = evaluate_fov_level(
                    model, val_dataset, image_transform,
                    extra_forward_args, device=device,
                    batch_size=batch_size, num_workers=num_workers,
                )
                history["val_step"].append(step + 1)
                history["val_loss"].append(vmetrics["mse"])
                history["val_pearson_median"].append(vmetrics["median_pearson"])
                print(f"  [val] mse={vmetrics['mse']:.4f} "
                      f"median_pearson={vmetrics['median_pearson']:.4f}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({"val/mse": vmetrics["mse"],
                                   "val/median_pearson": vmetrics["median_pearson"],
                                   "step": step + 1})
                if vmetrics["mse"] < best_val:
                    best_val = vmetrics["mse"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(best_state, out_path / "best.pt")
                model.train()
            step += 1

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, out_path / "best.pt")
    history["best_val_loss"] = best_val if math.isfinite(best_val) else None
    torch.save(best_state, out_path / "best.pt")
    with open(out_path / "history.json", "w") as f:
        json.dump({k: v for k, v in history.items() if k != "best_state_dict"},
                  f, indent=2)
    print(f"[done] best val mse={history['best_val_loss']} — saved to {out_path}",
          flush=True)
    return history


@torch.no_grad()
def evaluate_fov_level(
    model: nn.Module,
    dataset: FovGexDataset,
    image_transform: Callable,
    extra_forward_args: Optional[Callable] = None,
    *,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    """Run model on the full dataset; compute MSE + per-gene Pearson at FOV
    granularity. Returns ``{mse, median_pearson, per_gene_pearson, n}``.
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_baseline,
        persistent_workers=num_workers > 0,
    )
    preds, tgts = [], []
    for images, targets, _metas in loader:
        images = image_transform(images, device=device)
        kwargs = extra_forward_args(images.shape[0], device=device) if extra_forward_args else {}
        pred = model(images, **kwargs)
        if pred.ndim > 2:
            pred = pred.view(pred.shape[0], -1)
        preds.append(pred.cpu().numpy())
        tgts.append(targets.numpy())
    if not preds:
        return {"mse": float("nan"), "median_pearson": float("nan"),
                "per_gene_pearson": [], "n": 0}
    P = np.concatenate(preds, axis=0)
    T = np.concatenate(tgts, axis=0)
    mse = float(((P - T) ** 2).mean())
    per_gene = _pearson_per_gene(P, T)
    return {
        "mse": mse,
        "median_pearson": float(np.nanmedian(per_gene)),
        "per_gene_pearson": per_gene.tolist(),
        "n": int(P.shape[0]),
    }
