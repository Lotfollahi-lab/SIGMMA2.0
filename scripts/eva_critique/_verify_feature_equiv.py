"""Verify our feature-extraction path matches what EVA's features_only does.

Loads UNI once and calls two timm APIs on the SAME model:

  Path A (ours):  get_intermediate_layers(x, n=1, norm=True, reshape=True)
  Path B (EVA):   forward_intermediates(x, indices=1, output_fmt='NCHW',
                                         norm=True, intermediates_only=True)

`features_only=True + out_indices=1` is implemented internally as
`forward_intermediates(indices=N, ...)` — same code path. Comparing these two
on the same loaded model isolates whether the *normalization / reshape* choices
match.
"""
from __future__ import annotations
import argparse
import inspect
import torch
import timm
from pathlib import Path


def load_uni(ckpt_dir: str, device: torch.device):
    m = timm.create_model(
        "vit_large_patch16_224",
        img_size=224, patch_size=16,
        init_values=1e-5, num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(Path(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.to(device).eval()
    return m


def extract_a(model, x):
    """Our path."""
    sig = inspect.signature(model.get_intermediate_layers)
    kwargs = dict(reshape=True, norm=True)
    if "return_prefix_tokens" in sig.parameters:
        kwargs["return_prefix_tokens"] = False
    elif "return_class_token" in sig.parameters:
        kwargs["return_class_token"] = False
    feats = model.get_intermediate_layers(x, n=1, **kwargs)
    return feats[0]


def extract_b(model, x):
    """EVA features_only path — what FeatureGetterNet calls under the hood."""
    sig = inspect.signature(model.forward_intermediates)
    print(f"[B] forward_intermediates kwargs: {list(sig.parameters)}")
    kwargs = dict(indices=1, intermediates_only=True)
    if "output_fmt" in sig.parameters:
        kwargs["output_fmt"] = "NCHW"
    if "norm" in sig.parameters:
        kwargs["norm"] = True
    feats = model.forward_intermediates(x, **kwargs)
    if isinstance(feats, (list, tuple)):
        return feats[0] if len(feats) > 0 else feats
    return feats


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"[setup] device={device}")
    print(f"[setup] timm version: {timm.__version__}")

    model = load_uni(args.ckpt_dir, device)

    x = torch.randn(args.batch, 3, 224, 224, device=device)
    print(f"[input] x={tuple(x.shape)} mean={x.mean().item():.3f} std={x.std().item():.3f}")

    fa = extract_a(model, x)
    fb = extract_b(model, x)
    print(f"[A] shape={tuple(fa.shape)} dtype={fa.dtype}")
    print(f"[B] shape={tuple(fb.shape)} dtype={fb.dtype}")
    if fa.shape != fb.shape:
        print(f"!! SHAPE MISMATCH")
        return

    diff = (fa - fb).abs()
    print(f"[diff] max_abs={diff.max().item():.6e}")
    print(f"[diff] mean_abs={diff.mean().item():.6e}")
    print(f"[diff] L2_rel={(diff.norm() / fa.norm()).item():.6e}")
    cos = torch.nn.functional.cosine_similarity(fa.flatten(1), fb.flatten(1), dim=1)
    print(f"[cos] per-sample = {cos.cpu().numpy()}")

    if diff.max().item() < 1e-4:
        print("\n[verdict] EQUIVALENT")
    else:
        print("\n[verdict] DIVERGENT — sources differ")


if __name__ == "__main__":
    main()
