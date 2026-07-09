"""Recompute per-class Dice on MoNuSAC test for one model.

Used to compare our `best_val_dice` (mean over 5 classes incl. bg) with EVA's
leaderboard reporting convention (mean over 4 nuclei subtypes only, bg excluded).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eva_critique.data import iter_monusac_test
from scripts.eva_critique.models import load_uni
from scripts.eva_critique.train_decoder import NUM_CLASSES
from scripts.eva_critique.data import _AMBIGUOUS_IDX
from scripts.eva_critique.decoders import build_decoder

DATA_ROOT = "/nfs/team361/dj16/projects/SIGMMA/data/monusac"
UNI_CKPT = "outputs/eva_critique/uni/convdec_image/best.pt"
CONFIG = "configs/in-house_train_sigmma.yaml"

CLASS_NAMES = ["Background", "Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]


class _ListDS(torch.utils.data.Dataset):
    def __init__(self, tiles):
        self.tiles = tiles
    def __len__(self): return len(self.tiles)
    def __getitem__(self, i):
        t = self.tiles[i]
        return t.image_norm, torch.from_numpy(t.sem_mask).long(), t.image_id


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_uni(CONFIG, device)
    decoder = build_decoder("convdec_image", encoder.out_features, NUM_CLASSES)
    state = torch.load(UNI_CKPT, map_location=device)
    decoder.load_state_dict(state["decoder_state"], strict=True)
    decoder.to(device).eval()

    tiles = list(iter_monusac_test(DATA_ROOT))
    loader = DataLoader(_ListDS(tiles), batch_size=8, shuffle=False)

    inter = torch.zeros(NUM_CLASSES, device=device)
    denom = torch.zeros(NUM_CLASSES, device=device)
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device); y = y.to(device)
            feat = encoder.forward_features_grid(x)
            logits = decoder(feat, x)
            pred = logits.argmax(dim=1)
            valid = y != _AMBIGUOUS_IDX
            for c in range(NUM_CLASSES):
                p = (pred == c) & valid
                g = (y == c) & valid
                inter[c] += (p & g).sum()
                denom[c] += p.sum() + g.sum()
    dice = (2 * inter / denom.clamp_min(1)).cpu().tolist()
    print(f"{'Class':<14} Dice")
    for n, d in zip(CLASS_NAMES, dice):
        print(f"  {n:<12} {d:.4f}")
    print(f"\n  mean over 5 (incl bg)  = {sum(dice)/5:.4f}  <- our reported best_val_dice")
    print(f"  mean over 4 (excl bg)  = {sum(dice[1:])/4:.4f}  <- EVA leaderboard convention")
    print(f"  mean over present cls* = computed below if any class has 0 GT")
    # Drop zero-GT classes
    present = [d for c, d in enumerate(dice) if denom[c].item() > 0 and c != 0]
    if present:
        print(f"  mean over fg present   = {sum(present)/len(present):.4f}")


if __name__ == "__main__":
    main()
