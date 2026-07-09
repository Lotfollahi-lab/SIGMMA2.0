"""Per-baseline inference adapters for the 2D GEX prediction task.

Each adapter exposes two methods:

  ``predict_fov(image_uint8) -> Tensor (G,)``
      Path A — one forward pass over the whole FOV.

  ``predict_bin_grid(image_uint8, target_grid, crop_px) -> Tensor (G, K, K)``
      Path B — extract K×K bin-centered crops, single batched forward,
      return predictions per bin.

``image_uint8`` is ``(3, fov_size, fov_size)`` uint8.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F


def _extract_bin_crops(image: torch.Tensor, target_grid: int,
                       crop_px: int) -> torch.Tensor:
    """``(3, H, W)`` → ``(K², 3, crop_px, crop_px)``.

    Bin centres are at ``(i + 0.5) * bin_width`` for i in [0, K). Crops are
    extracted with reflection padding when they cross the FOV boundary.
    """
    assert image.ndim == 3, f"image must be (3, H, W), got {image.shape}"
    C, H, W = image.shape
    bin_w = W / target_grid
    bin_h = H / target_grid
    half = crop_px // 2
    # Reflect-pad once to handle boundary bins.
    pad = half + 1
    padded = F.pad(image.unsqueeze(0).float(), (pad, pad, pad, pad), mode="reflect")[0]
    crops = []
    for iy in range(target_grid):
        cy = int(round((iy + 0.5) * bin_h)) + pad
        for ix in range(target_grid):
            cx = int(round((ix + 0.5) * bin_w)) + pad
            crop = padded[:, cy - half:cy - half + crop_px,
                          cx - half:cx - half + crop_px]
            crops.append(crop)
    return torch.stack(crops, dim=0).to(torch.uint8)


class InferenceAdapter(Protocol):
    crop_px: int

    def predict_fov(self, image_uint8: torch.Tensor) -> torch.Tensor: ...
    def predict_bin_grid(self, image_uint8: torch.Tensor, target_grid: int) -> torch.Tensor: ...


# ============================================================
# ST-Net adapter
# ============================================================
class STNetAdapter:
    crop_px = 256

    def __init__(self, model, device: torch.device):
        from .stnet.transforms import to_model_input
        self.model = model.to(device).eval()
        self.device = device
        self._to_input = to_model_input

    @torch.no_grad()
    def predict_fov(self, image_uint8: torch.Tensor) -> torch.Tensor:
        x = self._to_input(image_uint8.unsqueeze(0), target_size=224, device=self.device)
        return self.model(x)[0].cpu()

    @torch.no_grad()
    def predict_bin_grid(self, image_uint8: torch.Tensor, target_grid: int) -> torch.Tensor:
        crops = _extract_bin_crops(image_uint8, target_grid, self.crop_px)
        x = self._to_input(crops, target_size=224, device=self.device)
        preds = self.model(x)  # (K², G)
        G = preds.shape[1]
        return preds.cpu().reshape(target_grid, target_grid, G).permute(2, 0, 1)


# ============================================================
# DeepPT adapter
# ============================================================
class DeepPTAdapter:
    crop_px = 224

    def __init__(self, model, device: torch.device):
        from .deeppt.transforms import to_model_input
        self.model = model.to(device).eval()
        self.device = device
        self._to_input = to_model_input

    @torch.no_grad()
    def predict_fov(self, image_uint8: torch.Tensor) -> torch.Tensor:
        x = self._to_input(image_uint8.unsqueeze(0), target_size=224, device=self.device)
        return self.model(x)[0].cpu()

    @torch.no_grad()
    def predict_bin_grid(self, image_uint8: torch.Tensor, target_grid: int) -> torch.Tensor:
        crops = _extract_bin_crops(image_uint8, target_grid, self.crop_px)
        x = self._to_input(crops, target_size=224, device=self.device)
        preds = self.model(x)
        G = preds.shape[1]
        return preds.cpu().reshape(target_grid, target_grid, G).permute(2, 0, 1)


# ============================================================
# HisToGene adapter — multi-spot at eval time
# ============================================================
class HisToGeneAdapter:
    crop_px = 112

    def __init__(self, model, device: torch.device, n_pos: int = 64):
        from .histogene.transforms import make_center_default, to_model_input
        self.model = model.to(device).eval()
        self.device = device
        self.n_pos = n_pos
        self._to_input = to_model_input
        self._default_center = make_center_default

    @torch.no_grad()
    def predict_fov(self, image_uint8: torch.Tensor) -> torch.Tensor:
        patches = self._to_input(image_uint8.unsqueeze(0),
                                 patch_size=self.crop_px, device=self.device)
        # patches: (1, 1, 3*P*P)
        centers = self._default_center(1, n_pos_default=self.n_pos // 2,
                                       device=self.device)
        pred = self.model(patches, centers)  # (1, 1, G)
        return pred[0, 0].cpu()

    @torch.no_grad()
    def predict_bin_grid(self, image_uint8: torch.Tensor, target_grid: int) -> torch.Tensor:
        crops = _extract_bin_crops(image_uint8, target_grid, self.crop_px)  # (K², 3, P, P)
        K2 = crops.shape[0]
        patches = self._to_input(crops, patch_size=self.crop_px, device=self.device)
        patches = patches.reshape(1, K2, -1)
        # Coordinates: laid out as (y, x) on the K×K grid, mapped to [0, n_pos).
        ys = torch.arange(target_grid).repeat_interleave(target_grid)
        xs = torch.arange(target_grid).repeat(target_grid)
        scale = max(1, self.n_pos // max(1, target_grid))
        centers = torch.stack([xs * scale, ys * scale], dim=-1)
        centers = centers.unsqueeze(0).to(self.device)
        pred = self.model(patches, centers)  # (1, K², G)
        G = pred.shape[-1]
        return pred[0].cpu().reshape(target_grid, target_grid, G).permute(2, 0, 1)


# ============================================================
# Phoenix adapter — zero-shot generative flow-matching
# ============================================================
class PhoenixAdapter:
    """Zero-shot Phoenix (peng-lab/phoenix) at each FOV bin centre.

    Phoenix is a latent flow-matching model: ``vision_forward`` extracts DINOv2
    patch tokens, then an ODE integrates Gaussian noise → expression. It predicts
    a fixed 377-gene panel in log-normalised per-cell units. We sample
    ``n_samples`` draws per crop and average to estimate the conditional mean,
    then de-normalise (``clip(≥0) * std + mean``) and re-order to the SVG benchmark
    panel, filling **NaN** for genes Phoenix does not cover. Downstream
    ``summarize()`` produces per-gene metrics; the NaN (uncovered) genes drop out
    of the medians, leaving the Phoenix∩panel overlap for the fair comparison.

    Default ``crop_px=112`` (≈56 µm at 0.5 µm/px) gives each K×K bin a tighter
    cell-scale context than a full 224 px crop (which would overlap ~88 % between
    adjacent K=14 bins). Tunable.
    """

    def __init__(self, phoenix, panel, device: torch.device,
                 crop_px: int = 112, n_samples: int = 8,
                 atol: float = 1e-1, rtol: float = 1e-1,
                 batch_size: int = 64, t_0: float = 0.0, t_1: float = 1.0):
        import numpy as np

        from .phoenix.transforms import to_model_input

        self.model = phoenix.model            # already .to(device).eval()
        self.gene_list = phoenix.gene_list    # 377 Phoenix genes
        self.mean = phoenix.mean              # (377,)
        self.std = phoenix.std                # (377,)
        self.device = device
        self.crop_px = crop_px
        self.n_samples = n_samples
        self.atol = atol
        self.rtol = rtol
        self.batch_size = batch_size
        self.t_0 = t_0
        self.t_1 = t_1
        self._to_input = to_model_input
        # SVG-panel gene -> Phoenix-377 column index (-1 if Phoenix lacks it).
        phx_idx = {g: i for i, g in enumerate(self.gene_list)}
        self.col = np.asarray([phx_idx.get(g, -1) for g in panel], dtype=np.int64)
        self.G_panel = len(panel)

    @torch.no_grad()
    def _sample(self, crops_uint8: torch.Tensor):
        """``(M, 3, P, P)`` uint8 → ``(M, G_panel)`` de-normalised, panel-ordered
        (NaN where Phoenix lacks the gene)."""
        import numpy as np

        from phoenix.helpers.inference import run_flow

        M = crops_uint8.shape[0]
        n_genes = len(self.gene_list)
        covered = self.col >= 0
        cov_idx = self.col[covered]
        out = np.full((M, self.G_panel), np.nan, dtype=np.float32)
        use_amp = self.device.type == "cuda"
        for s in range(0, M, self.batch_size):
            chunk = crops_uint8[s:s + self.batch_size]
            x = self._to_input(chunk, target_size=224, device=self.device)
            m = x.shape[0]
            # fp16 autocast for the heavy DINOv2-giant encoder (inference only).
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                feats = self.model.vision_forward(x)      # (m, T, 1536)
            feats = feats.float()
            acc = torch.zeros(m, n_genes, 1, device=self.device)
            for _ in range(self.n_samples):
                noise = torch.randn(m, n_genes, 1, device=self.device)
                acc += run_flow(self.model, noise.float(), self.t_0, self.t_1,
                                feats, None, self.atol, self.rtol, device=self.device)
            gex = (acc / self.n_samples).squeeze(-1).float().cpu().numpy()  # (m, 377)
            gex = np.clip(gex, 0, None) * self.std + self.mean
            out[s:s + m][:, covered] = gex[:, cov_idx]
        return out

    @torch.no_grad()
    def predict_fov(self, image_uint8: torch.Tensor) -> torch.Tensor:
        preds = self._sample(image_uint8.unsqueeze(0))   # (1, G_panel)
        return torch.from_numpy(preds[0])

    @torch.no_grad()
    def predict_bin_grid(self, image_uint8: torch.Tensor, target_grid: int) -> torch.Tensor:
        crops = _extract_bin_crops(image_uint8, target_grid, self.crop_px)  # (K², 3, P, P)
        preds = self._sample(crops)                                         # (K², G_panel)
        grid = preds.reshape(target_grid, target_grid, self.G_panel).transpose(2, 0, 1)
        return torch.from_numpy(grid)                                       # (G_panel, K, K)


def build_adapter(baseline: str, model, device: torch.device, **kwargs) -> InferenceAdapter:
    if baseline == "stnet":
        return STNetAdapter(model, device)
    if baseline == "histogene":
        return HisToGeneAdapter(model, device, n_pos=kwargs.get("n_pos", 64))
    if baseline == "deeppt":
        return DeepPTAdapter(model, device)
    if baseline == "phoenix":
        return PhoenixAdapter(
            model, panel=kwargs["panel"], device=device,
            crop_px=kwargs.get("crop_px", 112), n_samples=kwargs.get("n_samples", 8),
            atol=kwargs.get("atol", 1e-1), rtol=kwargs.get("rtol", 1e-1),
            batch_size=kwargs.get("batch_size", 64),
        )
    raise ValueError(f"unknown baseline={baseline!r}")
