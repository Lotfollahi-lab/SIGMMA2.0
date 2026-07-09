"""Train one (model, decoder) pair on MoNuSAC, with the encoder fully frozen.

Same seed, same augmentation, same optimizer for every (model, decoder) pair so
component-4 comparisons are paired by image. Saves:
  - {out_dir}/best.pt              decoder weights at best val Dice
  - {out_dir}/test_artifacts.npz   per-test-tile prob maps + metadata
                                   (consumed by extract_artifacts + pipeline_eval)

Usage (single pair):
    python -m scripts.eva_critique.train_decoder \
        --sigmma_config configs/image_pathology_segmentation_per_scale_sigmma.yaml \
        --model sigmma --decoder linear \
        --data_root ./data --out_root ./outputs/eva_critique \
        --scale 0
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as tv2

from eva.vision import datasets as eva_datasets
from eva.vision.losses import DiceLoss as _EvaDiceLoss

from scripts.eva_critique.data import _AMBIGUOUS_IDX, _MEAN, _STD
from scripts.eva_critique.decoders import build_decoder
from scripts.eva_critique.models import load_sigmma, load_uni


NUM_CLASSES = 5  # MoNuSAC: 0=bg, 1..4=nuclei subtypes; ambiguous (5) ignored in loss.


# ============================================================
# Dataset wrapper
# ============================================================
class _MoNuSACTransformed(Dataset):
    """Wrap eva.MoNuSAC, apply joint train/eval transforms (returns tensors)."""

    def __init__(self, root: str, split: str, train: bool,
                 train_aug: str = "rrc_hflip"):
        self.ds = eva_datasets.MoNuSAC(root=root, split=split, download=False)
        self.train = train
        self.train_aug = train_aug
        self.tx_eval_img = tv2.Compose([
            tv2.Resize(size=224, antialias=True),
            tv2.CenterCrop(size=224),
            tv2.ToDtype(torch.float32, scale=True),
            tv2.Normalize(mean=list(_MEAN), std=list(_STD)),
        ])
        self.tx_eval_msk = tv2.Compose([
            tv2.Resize(size=224, antialias=False,
                       interpolation=tv2.InterpolationMode.NEAREST),
            tv2.CenterCrop(size=224),
        ])
        # `rrc_hflip`: legacy behaviour (RandomResizedCrop + HFlip). Used for
        # all SIGMMA/UNI runs prior to 2026-05-15.
        # `eva_resize`: deterministic Resize+CenterCrop+Normalize, matching
        # `eva.vision.data.transforms.common.ResizeAndCrop` exactly. This is
        # what EVA's offline pipeline applies (its decoder trains on cached
        # embeddings, so augmentation is impossible). Use for leaderboard-
        # comparable runs.
        self.tx_train = tv2.Compose([
            tv2.RandomResizedCrop(size=224, antialias=True),
            tv2.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img = self.ds.load_data(i)              # (3, H, W) uint8
        msk = self.ds.load_target(i)            # (H, W) int64
        if self.train:
            if self.train_aug == "eva_resize":
                # Match EVA offline: deterministic Resize+CenterCrop, no aug.
                img = self.tx_eval_img(img)
                msk = self.tx_eval_msk(msk)
            else:
                # `rrc_hflip` (default, legacy): RandomResizedCrop + HFlip.
                img, msk = self.tx_train(img, msk)
                img = tv2.ToDtype(torch.float32, scale=True)(img)
                img = tv2.Normalize(mean=list(_MEAN), std=list(_STD))(img)
        else:
            img = self.tx_eval_img(img)
            msk = self.tx_eval_msk(msk)
        msk = msk.to(torch.int64)
        return img, msk, self.ds.filename(i)


# ============================================================
# Trainable wrapper: frozen encoder + decoder
# ============================================================
class _SegHead(nn.Module):
    def __init__(self, encoder, decoder, pyramid_layers=None, needs_image=False):
        super().__init__()
        self.encoder = encoder        # frozen, eval mode
        self.decoder = decoder        # trainable
        self.pyramid_layers = pyramid_layers  # None or list of 1-indexed taps
        self.needs_image = needs_image  # decoder consumes raw RGB (EVA convdec_image)

    def forward(self, x):
        if self.pyramid_layers is not None:
            feats = self.encoder.forward_features_pyramid(x, self.pyramid_layers)
            return self.decoder(feats)
        feat = self.encoder.forward_features_grid(x)
        if self.needs_image:
            return self.decoder(feat, x)
        if getattr(self.decoder, "needs_image_size", False):
            return self.decoder(feat, image_size=x.shape[-2:])
        return self.decoder(feat)


def _dice_loss_softmax(logits, target, ignore_index, num_classes):
    """Multiclass Dice loss with one-hot target; ambiguous pixels excluded."""
    probs = logits.softmax(dim=1)  # (B, C, H, W)
    valid = target != ignore_index
    t = target.clone()
    t[~valid] = 0  # placeholder; masked out below
    onehot = F.one_hot(t, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    onehot = onehot * valid_f
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dim=dims)
    denom = probs.sum(dim=dims) + onehot.sum(dim=dims)
    loss = 1 - (2 * inter + 1e-6) / (denom + 1e-6)
    return loss.mean()


def _eval_dice(model, loader, device, ignore_index, num_classes) -> float:
    """EVA leaderboard-style Dice.

    Matches `eva.vision.metrics.MonaiDiceScore` as used in
    `MulticlassSegmentationMetrics` (include_background=False, ignore_empty=True,
    reduction='mean'): per-image-per-class Dice, NaN for empty-GT cells,
    then nanmean over (image, class).

    Background (class 0) is excluded from the headline mean — ambiguous pixels
    are mapped to background in both pred and target so they drop out.
    """
    from monai.metrics import DiceMetric

    metric = DiceMetric(
        include_background=False,
        reduction="mean",
        ignore_empty=True,
        num_classes=num_classes,
    )
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            amb = y == ignore_index
            pred = pred.masked_fill(amb, 0)
            tgt = y.masked_fill(amb, 0)
            metric(y_pred=pred.unsqueeze(1), y=tgt.unsqueeze(1))
    score = metric.aggregate().item()
    metric.reset()
    return score


# ============================================================
# Driver
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmma_config", required=True)
    ap.add_argument(
        "--model", required=True,
        choices=["uni", "sigmma", "sigmma_no_lora", "sigmma_multiscale",
                 "sigmma_mlp_lora",
                 "sigmma_scale_micro", "sigmma_scale_meso", "sigmma_scale_macro",
                 "bleep_uni_ft", "clip_uni_ft", "plip_uni_ft", "omiclip_uni_ft"],
    )
    ap.add_argument("--decoder", required=True,
                    choices=["linear", "mlp", "light_unet", "unet_skip",
                             "convdec_image", "convdec_ms"])
    ap.add_argument("--scale", type=int, default=0,
                    help="SIGMMA LoRA adapter index (ignored for --model uni).")
    ap.add_argument("--baseline_config_path", default=None,
                    help="Path to the baseline-UNI inference config; required "
                         "when --model is one of *_uni_ft.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument(
        "--weight_decay", type=float, default=0.0,
        help="AdamW weight_decay. Default 0.0 (legacy). Set 0.01 to match "
             "EVA upstream which falls back to PyTorch AdamW default.",
    )
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--train_aug", choices=["rrc_hflip", "eva_resize"], default="rrc_hflip",
        help="Train-time image transform. `rrc_hflip` = RandomResizedCrop + "
             "HFlip (legacy). `eva_resize` = deterministic Resize+CenterCrop, "
             "exact match to EVA offline pipeline (no augmentation).",
    )
    ap.add_argument("--unfreeze_last_n_blocks", type=int, default=0,
                    help="Unfreeze the last N transformer blocks of the encoder "
                         "(0 = all frozen, the default; >0 includes them in the "
                         "optimizer along with the decoder).")
    args = ap.parse_args()

    L.seed_everything(args.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "uni":
        encoder = load_uni(args.sigmma_config, device)
        out_dir = os.path.join(args.out_root, "uni", args.decoder)
    elif args.model == "sigmma":
        encoder = load_sigmma(args.sigmma_config, device, scale_idx=args.scale)
        # `encoder.name` is "sigmma_scale{N}" for per_scale=true configs
        # (multi-adapter routing) and "sigmma_shared_lora" for per_scale=false
        # (single shared adapter). Use the wrapper name so the on-disk layout
        # reflects which LoRA topology we actually loaded.
        out_dir = os.path.join(args.out_root, encoder.name, args.decoder)
    elif args.model == "sigmma_no_lora":
        encoder = load_sigmma(args.sigmma_config, device, scale_idx=0)
        out_dir = os.path.join(args.out_root, "sigmma_no_lora", args.decoder)
    elif args.model == "sigmma_multiscale":
        from scripts.eva_critique.models import load_sigmma_multiscale
        encoder = load_sigmma_multiscale(args.sigmma_config, device)
        out_dir = os.path.join(args.out_root, "sigmma_multiscale", args.decoder)
    elif args.model in ("sigmma_scale_micro", "sigmma_scale_meso", "sigmma_scale_macro"):
        from scripts.eva_critique.models import load_sigmma_scale_grid
        scale_idx = {"sigmma_scale_micro": 0,
                     "sigmma_scale_meso": 1,
                     "sigmma_scale_macro": 2}[args.model]
        encoder = load_sigmma_scale_grid(args.sigmma_config, device, scale_idx=scale_idx)
        out_dir = os.path.join(args.out_root, encoder.name, args.decoder)
    elif args.model == "sigmma_mlp_lora":
        # MLP-LoRA SIGMMA: same loader as `sigmma`, but the config points at
        # a checkpoint trained with LoRA on attention+MLP layers. We pick
        # scale 2 (macro) since that has been the best per-scale row.
        encoder = load_sigmma(args.sigmma_config, device, scale_idx=args.scale)
        out_dir = os.path.join(args.out_root, f"sigmma_mlp_lora_s{args.scale}", args.decoder)
    else:
        if args.baseline_config_path is None:
            raise SystemExit(
                f"--model={args.model} requires --baseline_config_path."
            )
        from scripts.eva_critique.models import load_baseline_uni  # local import
        encoder = load_baseline_uni(args.baseline_config_path, device)
        out_dir = os.path.join(args.out_root, args.model, args.decoder)
    os.makedirs(out_dir, exist_ok=True)

    pyramid = None
    decoder = build_decoder(args.decoder, encoder.out_features, NUM_CLASSES).to(device)
    if getattr(decoder, "needs_pyramid", False):
        pyramid = list(decoder.pyramid_layers)
    needs_image = getattr(decoder, "needs_image", False)

    model = _SegHead(encoder, decoder, pyramid_layers=pyramid,
                     needs_image=needs_image).to(device)
    # Encoder already frozen; double-check.
    for p in model.encoder.model.parameters():
        p.requires_grad = False
    # Optionally unfreeze the last N transformer blocks for joint encoder/
    # decoder training (experiment B: tests the frozen-encoder bottleneck).
    if args.unfreeze_last_n_blocks > 0:
        n = int(args.unfreeze_last_n_blocks)
        # encoder.model is the inner timm ViT; blocks is a ModuleList.
        for block in model.encoder.model.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        # Switch the unfrozen blocks to train mode so dropout etc. behave
        # correctly; the rest of the encoder stays in eval.
        for block in model.encoder.model.blocks[-n:]:
            block.train()
        print(f"[train_decoder] unfroze last {n} encoder block(s)")
    trainable = (
        [p for p in model.decoder.parameters() if p.requires_grad]
        + [p for p in model.encoder.parameters() if p.requires_grad]
    )
    n_train = sum(p.numel() for p in trainable)
    print(f"[train_decoder] total trainable params: {n_train:,}")

    optim = torch.optim.AdamW(trainable, lr=args.lr,
                              weight_decay=args.weight_decay,
                              betas=(0.9, 0.999))

    train_ds = _MoNuSACTransformed(os.path.join(args.data_root, "monusac"),
                                   "train", train=True,
                                   train_aug=args.train_aug)
    test_ds = _MoNuSACTransformed(os.path.join(args.data_root, "monusac"),
                                  "test", train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # EVA-faithful training loss: their MoNuSAC config uses
    #   eva.vision.losses.DiceLoss(softmax=True, batch=True, ignore_index=5)
    # which delegates to monai.losses.DiceLoss with EVA's wrapper that
    # zero-masks ignored pixels in BOTH logits and target (the masked logits
    # are then softmaxed). Matches `configs/.../monusac.yaml:65-69`.
    criterion = _EvaDiceLoss(softmax=True, batch=True,
                             ignore_index=_AMBIGUOUS_IDX).to(device)

    best_val = -1.0
    best_path = os.path.join(out_dir, "best.pt")
    step = 0
    log_every = 50
    eval_every = 200
    train_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        x, y, _ = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        step += 1
        if step % log_every == 0:
            print(f"[step {step}/{args.max_steps}] loss={loss.item():.4f}")
        if step % eval_every == 0 or step == args.max_steps:
            val = _eval_dice(model, test_loader, device,
                             ignore_index=_AMBIGUOUS_IDX, num_classes=NUM_CLASSES)
            print(f"[step {step}] val Dice (test)={val:.4f}")
            if val > best_val:
                best_val = val
                torch.save({"decoder_state": model.decoder.state_dict(),
                            "args": vars(args), "best_val": best_val,
                            "step": step}, best_path)
                print(f"  -> saved {best_path}")

    # Reload best, dump test artifacts.
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.decoder.load_state_dict(ckpt["decoder_state"])

    artifacts = {"image_id": [], "prob": [], "pred": [], "gt": [],
                 "entropy_map": []}
    model.eval()
    with torch.no_grad():
        for x, y, fn in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)                      # (B, C, 224, 224)
            probs = logits.softmax(dim=1)
            # foreground (nuclei vs bg) probability
            fg_prob = (1.0 - probs[:, 0]).cpu().numpy()    # (B, 224, 224)
            pred = probs.argmax(dim=1).cpu().numpy().astype(np.uint8)
            full_probs = probs.cpu().numpy()               # (B, C, 224, 224)
            ent = -(full_probs * np.log(np.clip(full_probs, 1e-12, 1.0))).sum(axis=1)
            for i, fpath in enumerate(fn):
                image_id = fpath.rsplit("/", 1)[-1].replace(".tif", "")
                artifacts["image_id"].append(image_id)
                artifacts["prob"].append(fg_prob[i].astype(np.float16))
                artifacts["pred"].append(pred[i])
                artifacts["gt"].append(y[i].numpy().astype(np.uint8))
                artifacts["entropy_map"].append(ent[i].astype(np.float16))

    np.savez_compressed(
        os.path.join(out_dir, "test_artifacts.npz"),
        image_id=np.array(artifacts["image_id"]),
        prob=np.stack(artifacts["prob"]),
        pred=np.stack(artifacts["pred"]),
        gt=np.stack(artifacts["gt"]),
        entropy=np.stack(artifacts["entropy_map"]),
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"best_val_dice": best_val, "model": args.model,
                   "decoder": args.decoder, "scale": args.scale,
                   "max_steps": args.max_steps, "lr": args.lr,
                   "seed": args.seed}, f, indent=2)
    print(f"[done] {out_dir}  best Dice={best_val:.4f}")


if __name__ == "__main__":
    main()
