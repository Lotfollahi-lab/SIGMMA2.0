"""2D H&E → ST gene-expression probing head + train/eval helpers.

Frozen-encoder probing: a tiny 1×1-conv head maps cached patch-token features
``(D, K, K)`` to a ``(G, K, K)`` log-counts map. ``D`` depends on the input
representation:

  - ``concat``  : 3 × 1024 = 3072  (SIGMMA β: micro/meso/macro upsampled to K)
  - ``micro``   : 1024            (SIGMMA scale_0 only, native K = 56)
  - ``meso``    : 1024            (SIGMMA scale_1 only, upsampled to K)
  - ``macro``   : 1024            (SIGMMA scale_2 only, upsampled to K)
  - ``uni``     : 1024            (UNI baseline 14×14 single forward upsampled)

Training target is ``log1p(transcript_counts)`` (per-bin, per-gene). The loss
is selectable (``train_probing_head(..., loss=...)``): ``"mse"`` (default, plain
point regression) or a distributional NLL over the same log1p target —
``"lognormal"`` (heteroscedastic Gaussian, per-bin predicted σ) or
``"zilognormal"`` (zero-inflated lognormal: an extra ``P(y>0)`` classifier for
the empty-bin spike), both via :class:`LogNormalHead2D`. A coverage mask is
multiplied in elementwise so that bins known to fall outside any tissue are
downweighted (caller decides the mask).

Phase A (caching) is the responsibility of the driver script — this module is
pure: head + train + eval given cached tensors on disk.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sigmma.evaluation.gex_cache_io import SectionHandleCache, iter_section_bboxes


# ============================================================
# Probing head
# ============================================================
class ProbingHead2D(nn.Module):
    """1×1 → GroupNorm + ReLU → 1×1. No spatial conv: features arrive at K×K."""

    def __init__(self, in_channels: int, num_genes: int, hidden_dim: int = 256,
                 norm_groups: int = 8):
        super().__init__()
        # GroupNorm requires hidden_dim % norm_groups == 0.
        groups = norm_groups if hidden_dim % norm_groups == 0 else 1
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.norm = nn.GroupNorm(groups, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(hidden_dim, num_genes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.norm(self.proj(x))))

    def predict_log1p(self, x: torch.Tensor) -> torch.Tensor:
        """For the MSE head the raw output already lives in log1p-count space,
        so the point prediction *is* ``forward``. Defined for a uniform
        interface with :class:`LogNormalHead2D` (see :func:`predict_all`)."""
        return self.forward(x)


class LogNormalHead2D(nn.Module):
    """(Zero-inflated) lognormal head over **log1p-count** targets.

    Targets are already ``y = log1p(count)`` (see :class:`CachedFovDataset`), so
    instead of NB/ZINB on raw counts we model ``y`` directly as Gaussian — i.e.
    ``1 + count`` is lognormal. Unlike plain MSE (which fixes the variance), the
    trunk predicts a **per-bin** ``log σ`` so the likelihood is heteroscedastic;
    this is what makes the fit differ from MSE rather than collapsing back to it.

      - ``loss='lognormal'``   : trunk emits ``2G`` channels = ``[μ, log σ]``;
        ``y ~ N(μ, σ²)``. Point prediction (``predict_log1p``) is ``μ``.
      - ``loss='zilognormal'`` : trunk emits ``3G`` channels =
        ``[μ, log σ, p_logit]``; ``y = 0`` with prob ``1 − p`` (the zero/dropout
        spike), else ``y ~ N(μ, σ²)``. Expected value ``E[y] = p · μ``, so the
        zero classifier shields the regression head from the mass of empty bins.

    :meth:`predict_log1p` returns ``E[y]`` in the same log1p space the MSE head
    outputs, so every downstream metric stays apples-to-apples across losses.
    """

    def __init__(self, in_channels: int, num_genes: int, loss: str = "lognormal",
                 hidden_dim: int = 256, norm_groups: int = 8, eps: float = 1e-6):
        super().__init__()
        if loss not in {"lognormal", "zilognormal"}:
            raise ValueError(
                f"LogNormalHead2D expects loss in {{lognormal, zilognormal}}, got {loss!r}"
            )
        self.loss = loss
        self.num_genes = int(num_genes)
        self.eps = float(eps)
        n_out = num_genes * (3 if loss == "zilognormal" else 2)
        groups = norm_groups if hidden_dim % norm_groups == 0 else 1
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.norm = nn.GroupNorm(groups, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(hidden_dim, n_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """Return ``(mu, sigma, p_logit)`` each broadcastable to ``(B, G, K, K)``.

        ``p_logit`` is ``None`` for plain lognormal. ``sigma`` is strictly
        positive; ``mu`` is in log1p space (kept unconstrained — log1p targets
        are ≥ 0 but predicting slightly negative μ is harmless for the NLL)."""
        h = self.out(self.act(self.norm(self.proj(x))))
        G = self.num_genes
        mu = h[:, :G]
        # Clamp log σ before softplus so a cold-start head can't explode the NLL.
        sigma = F.softplus(h[:, G:2 * G].clamp(-8.0, 8.0)) + self.eps
        pi_logit = h[:, 2 * G:3 * G] if self.loss == "zilognormal" else None
        return mu, sigma, pi_logit

    def predict_log1p(self, x: torch.Tensor) -> torch.Tensor:
        mu, _sigma, pi_logit = self.forward(x)
        if pi_logit is None:
            return mu
        return torch.sigmoid(pi_logit) * mu


class LinearProbingHead2D(nn.Module):
    """Per-pixel linear regressor: ``BatchNorm2d(affine=False) → Conv2d(D, G, 1)``.

    Stricter than ``ProbingHead2D`` — no hidden layer, no non-linearity, no
    learnable normalization affine. Head capacity is one linear projection,
    so a positive Δ here is evidence that gene information is *linearly*
    decodable from the embedding, not just *contained* in it.

    The non-affine BatchNorm only re-scales each input channel to unit
    variance / zero mean (using running stats at eval), which lets us compare
    reprs with different native magnitudes (e.g. concat D=3072 vs single-scale
    D=1024) without conflating channel scale with representation quality.
    """

    def __init__(self, in_channels: int, num_genes: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.out = nn.Conv2d(in_channels, num_genes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(x))

    def predict_log1p(self, x: torch.Tensor) -> torch.Tensor:
        """Identity — the linear head's output is already in log1p-count space."""
        return self.forward(x)


class LinearSweepHead2D(nn.Module):
    """DINOv2-style multi-LR linear probing. Bundles ``len(lrs)`` copies of
    ``LinearProbingHead2D`` so they can be trained jointly (one forward, one
    backward, summed MSE), but each parameter group is driven by its own LR
    via the optimizer. After training, the best head by held-out val loss is
    selected — removing LR as a confounder when comparing reprs.
    """

    def __init__(self, in_channels: int, num_genes: int, lrs: list[float]):
        super().__init__()
        if not lrs:
            raise ValueError("lrs must be non-empty")
        self.lrs = [float(x) for x in lrs]
        self.heads = nn.ModuleDict({
            self.key_for_lr(lr): LinearProbingHead2D(in_channels, num_genes)
            for lr in self.lrs
        })

    @staticmethod
    def key_for_lr(lr: float) -> str:
        # nn.ModuleDict forbids '.' in keys; replace with 'p' (e.g. 0.001 → 0p001).
        return f"lr_{lr:.4g}".replace(".", "p").replace("-", "m")

    def forward(self, x: torch.Tensor) -> dict:
        return {k: h(x) for k, h in self.heads.items()}


# ============================================================
# Feature assembly
# ============================================================
def assemble_features(
    scale_tokens: list[torch.Tensor],
    input_repr: str,
    target_grid: int,
) -> torch.Tensor:
    """Take a list of per-scale patch tokens ``(B, D, h_s, w_s)`` and return
    the decoder input at the target grid.

    For ``concat``: bilinear-up each scale to ``(target_grid, target_grid)``
    and channel-concat. For a single scale: bilinear-up that one scale only.
    """
    def up(t):
        if t.shape[-1] == target_grid:
            return t
        return F.interpolate(t.float(), size=(target_grid, target_grid),
                             mode="bilinear", align_corners=False)

    if input_repr == "concat":
        return torch.cat([up(t) for t in scale_tokens], dim=1)
    if input_repr in {"micro", "scale0"}:
        return up(scale_tokens[0])
    if input_repr in {"meso", "scale1"}:
        return up(scale_tokens[1])
    if input_repr in {"macro", "scale2"}:
        return up(scale_tokens[2])
    if input_repr in {"uni", "lora_uni"}:
        # Single-scale list of length 1 (vanilla UNI or LoRA-adapted UNI patches).
        assert len(scale_tokens) == 1, (
            f"{input_repr} expects single-scale tokens, got {len(scale_tokens)}"
        )
        return up(scale_tokens[0])
    raise ValueError(f"unknown input_repr={input_repr!r}")


# ============================================================
# Cached-feature dataset
# ============================================================
def pool_predictions(pred, target_grid_coarse: int):
    """Sum-pool a prediction tensor across the trailing two spatial dims.

    Accepts ``torch.Tensor`` *or* ``np.ndarray`` and returns the same backend.
    Shape rank is preserved: ``(G, K_fine, K_fine)`` or ``(N, G, K_fine, K_fine)``
    → trailing dims downsampled to ``target_grid_coarse``.

    The pool is a sum, not a mean — for predictions that live in ``log1p(count)``
    space, the caller usually wants to do ``expm1 → pool → log1p`` to get a
    coarse-bin log-count that matches how targets are constructed at coarser
    binning (sum of fine counts, then log1p). This function is the inner
    "sum-pool over (factor × factor) windows" step.

    ``target_grid_coarse`` must divide ``K_fine``.
    """
    is_np = isinstance(pred, np.ndarray)
    if pred.ndim == 3:
        return pool_predictions(pred[None, ...] if is_np else pred.unsqueeze(0),
                                target_grid_coarse)[0]
    if pred.ndim != 4:
        raise ValueError(f"pred must be 3- or 4-D, got shape {tuple(pred.shape)}")
    N, G, K_fine, K_fine2 = pred.shape
    if K_fine != K_fine2:
        raise ValueError(f"non-square prediction: {pred.shape}")
    if K_fine == target_grid_coarse:
        return pred
    if K_fine % target_grid_coarse != 0:
        raise ValueError(
            f"target_grid_coarse={target_grid_coarse} must divide K_fine={K_fine}"
        )
    factor = K_fine // target_grid_coarse
    reshaped = pred.reshape(
        N, G, target_grid_coarse, factor, target_grid_coarse, factor
    )
    return reshaped.sum(axis=(3, 5)) if is_np else reshaped.sum(dim=(3, 5))


def _downsample_count_target(target_raw: torch.Tensor, mask_raw: torch.Tensor,
                              target_grid: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum-pool raw count tensor ``(G, K_native, K_native)`` to
    ``(G, target_grid, target_grid)``. Mask is mean-pooled.

    Counts must be summed (not averaged) so that the binned tensor remains a
    valid transcript count. ``log1p`` is applied *after* pooling by caller.
    Requires ``target_grid`` to divide ``K_native``.
    """
    K_native = target_raw.shape[-1]
    if K_native == target_grid:
        return target_raw, mask_raw
    if K_native % target_grid != 0:
        raise ValueError(
            f"target_grid={target_grid} must divide cache native grid={K_native}"
        )
    factor = K_native // target_grid
    G = target_raw.shape[0]
    target_pool = target_raw.reshape(G, target_grid, factor, target_grid, factor).sum(dim=(2, 4))
    mask_pool = mask_raw.reshape(target_grid, factor, target_grid, factor).mean(dim=(1, 3))
    return target_pool, mask_pool


class CachedFovDataset(Dataset):
    """Joins per-FOV target/mask + (optionally) UNI features + SIGMMA scales
    from three separate cache directories on a (sample, fov_id) key.

    Layout per FOV:
      - ``data_cache_dir/<sample>/fov_<id>.npz``   → ``{target, mask, ...}``
      - ``uni_cache_dir/<sample>/fov_<id>.npz``    → ``{features}``
        (loaded only when ``input_repr == 'uni'``)
      - ``sigmma_cache_dir/<sample>/fov_<id>.npz`` → ``{micro, meso, macro}``
        (loaded only when ``input_repr in {'concat','micro','meso','macro'}``)

    Returns ``(features fp32, log1p_target, mask)`` at ``target_grid``. If
    ``target_grid`` is coarser than the data cache's native binning, the
    count tensor is sum-pooled before ``log1p``. Features are bilinear-
    upsampled to ``target_grid`` by ``assemble_features``.

    ``in_memory=True`` eagerly decodes all FOVs at construction; later
    ``__getitem__`` serves from RAM (~1-2 MB / FOV uncompressed fp32). Cheap.
    """

    def __init__(
        self,
        fov_keys: list[tuple[str, int]],
        input_repr: str,
        target_grid: int,
        *,
        data_cache_dir: str | Path,
        uni_cache_dir: str | Path | None = None,
        sigmma_cache_dir: str | Path | None = None,
        sigmma_lora_cache_dir: str | Path | None = None,
        in_memory: bool = False,
    ):
        self.fov_keys = list(fov_keys)
        self.input_repr = input_repr
        self.target_grid = int(target_grid)
        self.data_cache_dir = Path(data_cache_dir)
        self.uni_cache_dir = Path(uni_cache_dir) if uni_cache_dir else None
        self.sigmma_cache_dir = Path(sigmma_cache_dir) if sigmma_cache_dir else None
        self.sigmma_lora_cache_dir = (
            Path(sigmma_lora_cache_dir) if sigmma_lora_cache_dir else None
        )
        self._needs_uni = input_repr == "uni"
        self._needs_sigmma = input_repr in {"concat", "micro", "meso", "macro"}
        self._needs_sigmma_lora = input_repr == "lora_uni"
        if self._needs_uni and self.uni_cache_dir is None:
            raise ValueError(f"input_repr={input_repr!r} requires uni_cache_dir")
        if self._needs_sigmma and self.sigmma_cache_dir is None:
            raise ValueError(f"input_repr={input_repr!r} requires sigmma_cache_dir")
        if self._needs_sigmma_lora and self.sigmma_lora_cache_dir is None:
            raise ValueError(f"input_repr={input_repr!r} requires sigmma_lora_cache_dir")
        if input_repr not in {"uni", "lora_uni", "concat", "micro", "meso", "macro"}:
            raise ValueError(f"unknown input_repr={input_repr!r}")
        # Fork-safe: leave the handle cache empty here (parent process). The
        # first _load() in each DataLoader worker opens that worker's own
        # per-section HDF5 handles. See sigmma.evaluation.gex_cache_io.
        self._handles = SectionHandleCache()
        self._cache = None
        if in_memory:
            self._cache = [self._load(*k) for k in self.fov_keys]
            # Decoded into RAM — drop the handles opened in the parent so they
            # are not inherited across the upcoming DataLoader fork.
            self._handles.close()

    def _load(self, sample: str, fov_id: int):
        r = self._handles.reader(self.data_cache_dir, sample)
        row = r.row(fov_id)
        target_raw = torch.from_numpy(r.f["target"][row].astype(np.float32))
        K_native = target_raw.shape[-1]
        mask_raw = (
            torch.from_numpy(r.f["mask"][row].astype(np.float32))
            if r.has("mask")
            else torch.ones(K_native, K_native, dtype=torch.float32)
        )
        target_pool, mask_pool = _downsample_count_target(
            target_raw, mask_raw, self.target_grid
        )
        target = torch.log1p(target_pool)
        mask = mask_pool

        uni_feats = None
        if self._needs_uni:
            ru = self._handles.reader(self.uni_cache_dir, sample)
            uni_feats = torch.from_numpy(ru.get("features", fov_id).astype(np.float32))

        lora_feats = None
        if self._needs_sigmma_lora:
            rl = self._handles.reader(self.sigmma_lora_cache_dir, sample)
            lora_feats = torch.from_numpy(rl.get("features", fov_id).astype(np.float32))

        scales = None
        if self._needs_sigmma:
            rs = self._handles.reader(self.sigmma_cache_dir, sample)
            scales = (
                torch.from_numpy(rs.get("micro", fov_id).astype(np.float32)),
                torch.from_numpy(rs.get("meso", fov_id).astype(np.float32)),
                torch.from_numpy(rs.get("macro", fov_id).astype(np.float32)),
            )
        return target, mask, uni_feats, lora_feats, scales

    def __len__(self) -> int:
        return len(self.fov_keys)

    def __getitem__(self, i: int):
        if self._cache is not None:
            target, mask, uni_feats, lora_feats, scales = self._cache[i]
        else:
            target, mask, uni_feats, lora_feats, scales = self._load(*self.fov_keys[i])
        if self.input_repr == "uni":
            feats = assemble_features([uni_feats.unsqueeze(0)], "uni", self.target_grid)[0]
        elif self.input_repr == "lora_uni":
            feats = assemble_features([lora_feats.unsqueeze(0)], "lora_uni", self.target_grid)[0]
        else:  # concat / micro / meso / macro — all from SIGMMA scales
            feats = assemble_features(
                [s.unsqueeze(0) for s in scales],
                self.input_repr,
                self.target_grid,
            )[0]
        return feats, target, mask


def make_dataset(
    fov_keys: list[tuple[str, int]],
    input_repr: str,
    target_grid: int,
    *,
    data_cache_dir: str | Path,
    uni_cache_dir: str | Path | None = None,
    sigmma_cache_dir: str | Path | None = None,
    sigmma_lora_cache_dir: str | Path | None = None,
    in_memory: bool = False,
) -> Dataset:
    """Convenience wrapper around :class:`CachedFovDataset`. Kept as a
    function for parity with prior API and to centralise validation."""
    return CachedFovDataset(
        fov_keys,
        input_repr,
        target_grid,
        data_cache_dir=data_cache_dir,
        uni_cache_dir=uni_cache_dir,
        sigmma_cache_dir=sigmma_cache_dir,
        sigmma_lora_cache_dir=sigmma_lora_cache_dir,
        in_memory=in_memory,
    )


# ============================================================
# Losses
# ============================================================
_LOG_2PI = math.log(2.0 * math.pi)


def _masked_channel_mean(per_elem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of a ``(B, G, K, K)`` per-element tensor over tissue bins × genes.

    Mirrors the MSE normalization used elsewhere: ``mask`` (``B, K, K``) is
    broadcast across the gene axis, the masked values are summed, and the sum is
    divided by ``mask.sum() × G`` so the scale matches the plain-MSE path."""
    m = mask.unsqueeze(1)
    G = per_elem.shape[1]
    return (per_elem * m).sum() / (m.sum() * G + 1e-8)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Tissue-masked MSE on log1p targets (the legacy / default objective)."""
    return _masked_channel_mean((pred - target) ** 2, mask)


def lognormal_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL of log1p target ``y`` under ``N(μ, σ²)``.

    ``= log σ + ½ ((y−μ)/σ)² + ½ log 2π`` averaged over tissue bins × genes.
    With a *predicted* per-bin σ this differs from MSE (a fixed σ would make μ's
    optimum identical to MSE)."""
    z = (y - mu) / sigma
    per_elem = torch.log(sigma) + 0.5 * z * z + 0.5 * _LOG_2PI
    return _masked_channel_mean(per_elem, mask)


def zilognormal_nll(mu: torch.Tensor, sigma: torch.Tensor, pi_logit: torch.Tensor,
                    y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero-inflated lognormal NLL on log1p target ``y``.

    A binary head ``p = σ(pi_logit)`` models ``P(y > 0)`` (BCE over all tissue
    bins); the Gaussian regression term is applied only where ``y > 0``. Joint
    objective ``BCE(p, 1{y>0}) + 1{y>0}·gaussian_nll`` — the zero classifier
    absorbs the empty-bin spike so the regression head isn't pulled toward 0."""
    is_pos = (y > 0).float()
    bce = F.binary_cross_entropy_with_logits(pi_logit, is_pos, reduction="none")
    z = (y - mu) / sigma
    reg = torch.log(sigma) + 0.5 * z * z + 0.5 * _LOG_2PI
    return _masked_channel_mean(bce + is_pos * reg, mask)


def _distributional_loss(loss: str, head_out, target: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """Dispatch a :class:`LogNormalHead2D` forward output to its NLL."""
    mu, sigma, pi_logit = head_out
    if loss == "lognormal":
        return lognormal_nll(mu, sigma, target, mask)
    if loss == "zilognormal":
        return zilognormal_nll(mu, sigma, pi_logit, target, mask)
    raise ValueError(f"unknown distributional loss={loss!r}")


# ============================================================
# Train loop
# ============================================================
def train_probing_head(
    head: nn.Module,
    train_keys: list[tuple[str, int]],
    target_grid: int,
    device: torch.device,
    *,
    input_repr: str = "uni",
    val_keys: Optional[list[tuple[str, int]]] = None,
    data_cache_dir: str | Path,
    uni_cache_dir: str | Path | None = None,
    sigmma_cache_dir: str | Path | None = None,
    sigmma_lora_cache_dir: str | Path | None = None,
    num_steps: int = 2000,
    batch_size: int = 64,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    num_workers: int = 4,
    log_every: int = 100,
    val_every: int = 500,
    in_memory: bool = False,
    loss: str = "mse",
) -> dict:
    """Train ``head`` on cached FOVs. Cached features must already match the
    head's expected ``in_channels``. Returns a dict of training/val curves.

    ``loss`` selects the objective: ``"mse"`` (legacy, point prediction in
    log1p space) or a distributional NLL — ``"lognormal"`` / ``"zilognormal"``
    — which requires ``head`` to be a :class:`LogNormalHead2D`. Targets stay in
    log1p space for every loss, so val-loss-based model selection is internally
    consistent within a run.
    """
    is_dist = loss in {"lognormal", "zilognormal"}
    head = head.to(device).train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    ds_kwargs = dict(
        data_cache_dir=data_cache_dir,
        uni_cache_dir=uni_cache_dir,
        sigmma_cache_dir=sigmma_cache_dir,
        sigmma_lora_cache_dir=sigmma_lora_cache_dir,
        in_memory=in_memory,
    )
    train_ds = make_dataset(train_keys, input_repr, target_grid, **ds_kwargs)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True, pin_memory=True)
    val_ds = (
        make_dataset(val_keys, input_repr, target_grid, **ds_kwargs)
        if val_keys else None
    )

    history = {"step": [], "train_loss": [], "val_step": [], "val_loss": []}
    step = 0
    best_val = math.inf
    best_state = None

    while step < num_steps:
        for feats, target, mask in train_dl:
            if step >= num_steps:
                break
            feats = feats.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if is_dist:
                loss_val = _distributional_loss(loss, head(feats), target, mask)
            else:
                loss_val = masked_mse_loss(head(feats), target, mask)
            opt.zero_grad()
            loss_val.backward()
            opt.step()

            if step % log_every == 0:
                history["step"].append(step)
                history["train_loss"].append(float(loss_val.detach().cpu()))
            if val_ds is not None and (step + 1) % val_every == 0:
                vloss = _eval_loss(head, val_ds, device, batch_size, num_workers, loss=loss)
                history["val_step"].append(step + 1)
                history["val_loss"].append(vloss)
                if vloss < best_val:
                    best_val = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                head.train()
            step += 1
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    history["best_val_loss"] = best_val if math.isfinite(best_val) else None
    history["best_state_dict"] = best_state
    return history


@torch.no_grad()
def _eval_loss(head, val_ds, device, batch_size, num_workers, loss: str = "mse"):
    head.eval()
    is_dist = loss in {"lognormal", "zilognormal"}
    dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    total = 0.0
    n = 0
    for feats, target, mask in dl:
        feats = feats.to(device); target = target.to(device); mask = mask.to(device)
        if is_dist:
            loss_val = _distributional_loss(loss, head(feats), target, mask)
        else:
            loss_val = masked_mse_loss(head(feats), target, mask)
        total += float(loss_val.cpu()) * feats.shape[0]
        n += feats.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def _eval_loss_sweep(head: LinearSweepHead2D, val_ds, device, batch_size, num_workers) -> dict:
    head.eval()
    dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    totals = {k: 0.0 for k in head.heads.keys()}
    n = 0
    for feats, target, mask in dl:
        feats = feats.to(device); target = target.to(device); mask = mask.to(device)
        preds = head(feats)
        denom = float(mask.unsqueeze(1).sum() * next(iter(preds.values())).shape[1] + 1e-8)
        for k, p in preds.items():
            sq = (p - target) ** 2
            loss = (sq * mask.unsqueeze(1)).sum() / denom
            totals[k] += float(loss.cpu()) * feats.shape[0]
        n += feats.shape[0]
    return {k: v / max(n, 1) for k, v in totals.items()}


def train_linear_sweep(
    head: LinearSweepHead2D,
    train_keys: list[tuple[str, int]],
    target_grid: int,
    device: torch.device,
    *,
    input_repr: str = "uni",
    val_keys: Optional[list[tuple[str, int]]] = None,
    data_cache_dir: str | Path,
    uni_cache_dir: str | Path | None = None,
    sigmma_cache_dir: str | Path | None = None,
    sigmma_lora_cache_dir: str | Path | None = None,
    num_steps: int = 2000,
    batch_size: int = 64,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    num_workers: int = 4,
    log_every: int = 100,
    val_every: int = 500,
    in_memory: bool = False,
) -> dict:
    """DINOv2-style joint training: K parallel ``LinearProbingHead2D`` heads,
    each with its own SGD LR via param-groups; shared cosine decay to 0;
    summed MSE backward; track best per-head val MSE; final selection picks
    the head with the lowest best-val MSE across the sweep.

    Returns a dict with: per-head LR list, per-step train losses, per-eval-step
    per-head val losses, best per-head val loss, the *single* best key and its
    state_dict (already a ``LinearProbingHead2D`` state).
    """
    head = head.to(device).train()
    param_groups = [
        {"params": list(h.parameters()), "lr": lr, "name": LinearSweepHead2D.key_for_lr(lr)}
        for lr, h in zip(head.lrs, head.heads.values())
    ]
    opt = torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps, eta_min=0.0)

    ds_kwargs = dict(
        data_cache_dir=data_cache_dir,
        uni_cache_dir=uni_cache_dir,
        sigmma_cache_dir=sigmma_cache_dir,
        sigmma_lora_cache_dir=sigmma_lora_cache_dir,
        in_memory=in_memory,
    )
    train_ds = make_dataset(train_keys, input_repr, target_grid, **ds_kwargs)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True, pin_memory=True)
    val_ds = (
        make_dataset(val_keys, input_repr, target_grid, **ds_kwargs)
        if val_keys else None
    )

    keys = list(head.heads.keys())
    history = {
        "lrs": list(head.lrs),
        "keys": keys,
        "step": [], "train_loss_sum": [],
        "val_step": [], "val_losses": [],
    }
    best_val_per_head = {k: math.inf for k in keys}
    best_state_per_head = {k: None for k in keys}
    step = 0

    while step < num_steps:
        for feats, target, mask in train_dl:
            if step >= num_steps:
                break
            feats = feats.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            preds = head(feats)
            denom = (mask.unsqueeze(1).sum() * next(iter(preds.values())).shape[1] + 1e-8)
            per_head_loss = []
            for k, p in preds.items():
                sq = (p - target) ** 2
                per_head_loss.append((sq * mask.unsqueeze(1)).sum() / denom)
            loss = torch.stack(per_head_loss).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            if step % log_every == 0:
                history["step"].append(step)
                history["train_loss_sum"].append(float(loss.detach().cpu()))
            if val_ds is not None and (step + 1) % val_every == 0:
                vls = _eval_loss_sweep(head, val_ds, device, batch_size, num_workers)
                history["val_step"].append(step + 1)
                history["val_losses"].append(vls)
                for k, v in vls.items():
                    if v < best_val_per_head[k]:
                        best_val_per_head[k] = v
                        # Save only the inner LinearProbingHead2D state, so eval
                        # can reload into a single LinearProbingHead2D.
                        best_state_per_head[k] = {
                            n: p.detach().cpu().clone()
                            for n, p in head.heads[k].state_dict().items()
                        }
                head.train()
            step += 1

    # Fallback if no val ever ran (e.g. val_paths=None or num_steps<val_every)
    for k in keys:
        if best_state_per_head[k] is None:
            best_state_per_head[k] = {
                n: p.detach().cpu().clone()
                for n, p in head.heads[k].state_dict().items()
            }

    best_key = min(best_val_per_head, key=best_val_per_head.get)
    history["best_val_per_head"] = best_val_per_head
    history["best_key"] = best_key
    history["best_val_loss"] = best_val_per_head[best_key]
    history["best_state_dict"] = best_state_per_head[best_key]
    return history


# ============================================================
# Eval
# ============================================================
@torch.no_grad()
def predict_all(
    head: nn.Module,
    keys: list[tuple[str, int]],
    target_grid: int,
    device: torch.device,
    *,
    input_repr: str = "uni",
    data_cache_dir: str | Path,
    uni_cache_dir: str | Path | None = None,
    sigmma_cache_dir: str | Path | None = None,
    sigmma_lora_cache_dir: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    in_memory: bool = False,
) -> dict:
    """Return predictions and aligned ground-truth targets for all FOVs.

    Returns ``{"pred": (N, G, K, K), "target": (N, G, K, K), "mask": (N, K, K)}``
    in numpy. ``target`` is ``log1p`` (matches training target).
    """
    head = head.to(device).eval()
    ds = make_dataset(
        keys, input_repr, target_grid,
        data_cache_dir=data_cache_dir,
        uni_cache_dir=uni_cache_dir,
        sigmma_cache_dir=sigmma_cache_dir,
        sigmma_lora_cache_dir=sigmma_lora_cache_dir,
        in_memory=in_memory,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    preds, targets, masks = [], [], []
    for feats, target, mask in dl:
        feats = feats.to(device, non_blocking=True)
        # predict_log1p returns the point/expected value in log1p-count space for
        # every head type (identity for MSE/linear heads, E[y] for LogNormalHead2D)
        # so all downstream metrics compare in one consistent space.
        out = head.predict_log1p(feats).cpu().numpy()
        preds.append(out)
        targets.append(target.numpy())
        masks.append(mask.numpy())

    # DataLoader is shuffle=False so output index n maps 1:1 to ds.fov_keys[n].
    # Slide-level metrics (SSIM, standardized MSE on stitched canvases) need
    # the per-FOV pixel coordinates that live in the data-cache npz files but
    # were not surfaced through CachedFovDataset.__getitem__. Read them now.
    n_fovs = len(ds.fov_keys)
    sample_ids = np.array([s for s, _ in ds.fov_keys], dtype=object)
    fov_ids = np.array([int(fid) for _, fid in ds.fov_keys], dtype=np.int64)
    x_starts = np.zeros(n_fovs, dtype=np.int64)
    y_starts = np.zeros(n_fovs, dtype=np.int64)
    fov_size = 0
    # Per-FOV pixel coords live in the data-cache .h5 (one file per section).
    # Read each section's bbox table once, then map fov_id -> (x_start, y_start).
    bbox_by_sample: dict[str, dict[int, tuple[int, int]]] = {}
    for i, (sample, fov_id) in enumerate(ds.fov_keys):
        bm = bbox_by_sample.get(sample)
        if bm is None:
            bm = {}
            for fid, x, y, fs in iter_section_bboxes(ds.data_cache_dir, sample):
                bm[int(fid)] = (int(x), int(y))
                if fov_size == 0:
                    fov_size = int(fs)
            bbox_by_sample[sample] = bm
        x, y = bm[int(fov_id)]
        x_starts[i] = x
        y_starts[i] = y
    return {
        "pred": np.concatenate(preds, axis=0) if preds else np.zeros((0,)),
        "target": np.concatenate(targets, axis=0) if targets else np.zeros((0,)),
        "mask": np.concatenate(masks, axis=0) if masks else np.zeros((0,)),
        "sample_ids": sample_ids,
        "fov_ids": fov_ids,
        "x_starts": x_starts,
        "y_starts": y_starts,
        "fov_size": int(fov_size) if fov_size else 0,
    }


def per_gene_pearson(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-gene Pearson r over masked bins, pooled across all FOVs.

    pred/target: (N, G, K, K). mask: (N, K, K). Returns (G,) array; NaN for
    genes with zero variance in either pred or target.
    """
    N, G, K, _ = pred.shape
    m = (mask > 0).reshape(N, 1, K, K)
    out = np.full(G, np.nan, dtype=np.float64)
    for g in range(G):
        p = pred[:, g][m[:, 0]]
        t = target[:, g][m[:, 0]]
        if p.size < 2:
            continue
        sp = p.std()
        st = t.std()
        if sp < 1e-8 or st < 1e-8:
            continue
        out[g] = float(((p - p.mean()) * (t - t.mean())).mean() / (sp * st))
    return out


def per_bin_r2(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Bin-wise R² across all masked bins and genes (pooled)."""
    m = (mask > 0)[:, None, :, :]
    p = pred[np.broadcast_to(m, pred.shape)]
    t = target[np.broadcast_to(m, target.shape)]
    if t.size < 2:
        return float("nan")
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    if ss_tot < 1e-8:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def per_fov_morans_i(maps: np.ndarray) -> np.ndarray:
    """Rook-adjacency Moran's I per (FOV, gene) on a K×K grid.

    maps: (N, G, K, K). Returns (N, G) Moran's I in [-1, 1].

    Moran's I is undefined for K=1 (no spatial neighbours). Returns all-NaN
    in that case so summarize() reports a NaN headline rather than crashing.
    """
    N, G, K, _ = maps.shape
    out = np.full((N, G), np.nan, dtype=np.float64)
    if K < 2:
        return out
    # Rook neighbour pairs (i, j) flattened indices.
    idx = np.arange(K * K).reshape(K, K)
    pairs = []
    pairs.extend(zip(idx[:, :-1].ravel().tolist(), idx[:, 1:].ravel().tolist()))   # horizontal
    pairs.extend(zip(idx[:-1, :].ravel().tolist(), idx[1:, :].ravel().tolist()))   # vertical
    pairs = np.asarray(pairs, dtype=np.int64)  # (E, 2)
    W = pairs.shape[0] * 2  # symmetric
    for n in range(N):
        for g in range(G):
            x = maps[n, g].ravel().astype(np.float64)
            xb = x - x.mean()
            denom = (xb ** 2).sum()
            if denom < 1e-12:
                continue
            num = float((xb[pairs[:, 0]] * xb[pairs[:, 1]]).sum() * 2)  # symmetric
            out[n, g] = (x.size / W) * (num / denom)
    return out


def per_bin_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-bin MSE pooled over genes; tissue bins only.

    pred/target: (N, G, K, K). mask: (N, K, K). Returns 1-D float64 array of
    length equal to the number of tissue bins across all FOVs.
    """
    sq = (pred - target) ** 2          # (N, G, K, K)
    mse_map = sq.mean(axis=1)          # (N, K, K) — mean over G genes
    return mse_map[mask > 0].astype(np.float64).ravel()


def per_gene_mse_raw(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-gene MSE on log1p counts, pooled across all FOVs (tissue bins only).

    pred/target: (N, G, K, K). mask: (N, K, K). Returns (G,) float64. NaN
    where no tissue bin exists for that gene. Companion summary to the
    existing per-bin pooled ``mse_per_bin`` (which averages over genes first).
    """
    N, G, K, _ = pred.shape
    m = (mask > 0).reshape(N, 1, K, K)
    out = np.full(G, np.nan, dtype=np.float64)
    for g in range(G):
        p = pred[:, g][m[:, 0]]
        t = target[:, g][m[:, 0]]
        if p.size < 1:
            continue
        out[g] = float(((p.astype(np.float64) - t.astype(np.float64)) ** 2).mean())
    return out


def _standardize_with_mask(
    x: np.ndarray, tissue: np.ndarray, eps: float = 1e-12,
) -> np.ndarray | None:
    """Min-max scale ``x`` to [0, 1] using only ``tissue`` bins, then zero out
    non-tissue. Returns None when there are <2 tissue bins or zero variance.

    Matches the iSCALE convention: per-array standardization so per-gene MSE
    and SSIM are scale-invariant. The zero-fill happens *after* standardization
    so a large no-tissue background doesn't dominate the min-max range.
    """
    tv = x[tissue]
    if tv.size < 2:
        return None
    lo = float(tv.min()); hi = float(tv.max())
    if (hi - lo) < eps:
        return None
    out = (x.astype(np.float64) - lo) / (hi - lo + eps)
    out = out * tissue.astype(np.float64)
    return out


def per_gene_mse_standardized(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
) -> np.ndarray:
    """Per-gene MSE on per-(FOV, gene) min-max standardized maps.

    pred/target: (N, G, K, K). mask: (N, K, K). For each (FOV, gene) pair,
    pred and target are independently scaled to [0, 1] using only tissue
    bins (iSCALE convention), then squared errors over tissue bins are
    pooled across FOVs and averaged per gene. Returns (G,) float64.
    """
    N, G, K, _ = pred.shape
    out = np.full(G, np.nan, dtype=np.float64)
    for g in range(G):
        ss = 0.0; n = 0
        for i in range(N):
            tissue = mask[i] > 0
            if tissue.sum() < 2:
                continue
            pn = _standardize_with_mask(pred[i, g], tissue)
            tn = _standardize_with_mask(target[i, g], tissue)
            if pn is None or tn is None:
                continue
            diff = pn[tissue] - tn[tissue]
            ss += float((diff * diff).sum())
            n += int(diff.size)
        if n > 0:
            out[g] = ss / n
    return out


def _stitch_per_slide(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
    sample_ids: np.ndarray, x_starts: np.ndarray, y_starts: np.ndarray,
    fov_size: int, target_grid: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Group per-FOV maps by slide and tile each group into a slide-level
    canvas at ``bin_px = fov_size / target_grid`` resolution.

    Returns ``{slide_id: (pred_canvas (G, Hb, Wb),
                          target_canvas (G, Hb, Wb),
                          mask_canvas (Hb, Wb))}``. Uncovered bins are NaN
    in pred/target and 0 in mask. Mirrors ``scripts/visualize_gex_2d_slide.py``'s
    ``stitch_to_slide`` but on the full (G,) gene axis and self-sized from
    FOV extents (no slide-level pixel dimensions required).
    """
    if pred.shape[0] == 0:
        return {}
    bin_px = float(fov_size) / float(target_grid)
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    G = pred.shape[1]
    sample_arr = np.asarray(sample_ids)
    for slide_id in np.unique(sample_arr):
        sel = np.where(sample_arr == slide_id)[0]
        xs = x_starts[sel]; ys = y_starts[sel]
        x_bins = np.round(xs.astype(np.float64) / bin_px).astype(np.int64)
        y_bins = np.round(ys.astype(np.float64) / bin_px).astype(np.int64)
        Hb = int(y_bins.max() + target_grid)
        Wb = int(x_bins.max() + target_grid)
        pcan = np.full((G, Hb, Wb), np.nan, dtype=np.float32)
        tcan = np.full((G, Hb, Wb), np.nan, dtype=np.float32)
        mcan = np.zeros((Hb, Wb), dtype=np.float32)
        for j, idx in enumerate(sel):
            y0 = int(y_bins[j]); x0 = int(x_bins[j])
            pcan[:, y0:y0 + target_grid, x0:x0 + target_grid] = pred[idx]
            tcan[:, y0:y0 + target_grid, x0:x0 + target_grid] = target[idx]
            mcan[y0:y0 + target_grid, x0:x0 + target_grid] = np.maximum(
                mcan[y0:y0 + target_grid, x0:x0 + target_grid], mask[idx]
            )
        out[str(slide_id)] = (pcan, tcan, mcan)
    return out


def per_slide_per_gene_ssim(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
    sample_ids: np.ndarray, x_starts: np.ndarray, y_starts: np.ndarray,
    fov_size: int, target_grid: int, win_size: int = 11,
) -> tuple[np.ndarray, list[str]]:
    """Stitch per-FOV maps to slide-level canvases (one per slide), then
    SSIM per (slide, gene) on min-max standardized maps.

    Returns ``((S, G) ssim_array, [slide_ids])``. NaN where a slide's gene
    map has zero variance or the canvas is too small for the window.
    """
    from skimage.metrics import structural_similarity

    if fov_size <= 0 or pred.shape[0] == 0:
        return np.zeros((0, pred.shape[1] if pred.ndim == 4 else 0)), []
    slides = _stitch_per_slide(
        pred, target, mask, sample_ids, x_starts, y_starts, fov_size, target_grid
    )
    slide_ids = sorted(slides.keys())
    G = pred.shape[1]
    out = np.full((len(slide_ids), G), np.nan, dtype=np.float64)
    ws_req = int(win_size)
    for s_idx, sid in enumerate(slide_ids):
        pcan, tcan, mcan = slides[sid]
        tissue = mcan > 0
        if tissue.sum() < 2:
            continue
        Hb, Wb = mcan.shape
        ws = ws_req
        # win_size must be odd and ≤ min(Hb, Wb)
        max_ws = min(Hb, Wb)
        if max_ws % 2 == 0:
            max_ws -= 1
        if ws > max_ws:
            ws = max_ws
        if ws < 3:
            continue
        for g in range(G):
            # Stitched canvases have NaN where no FOV covers — treat those
            # as out-of-tissue (mask already 0 there) and zero them so the
            # standardize step ignores them.
            pmap = np.where(np.isnan(pcan[g]), 0.0, pcan[g])
            tmap = np.where(np.isnan(tcan[g]), 0.0, tcan[g])
            pn = _standardize_with_mask(pmap, tissue)
            tn = _standardize_with_mask(tmap, tissue)
            if pn is None or tn is None:
                continue
            try:
                out[s_idx, g] = float(
                    structural_similarity(pn, tn, win_size=ws, data_range=1.0)
                )
            except ValueError:
                pass
    return out, slide_ids


def per_slide_per_gene_mse_standardized(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
    sample_ids: np.ndarray, x_starts: np.ndarray, y_starts: np.ndarray,
    fov_size: int, target_grid: int,
) -> tuple[np.ndarray, list[str]]:
    """Stitch per-FOV maps to slide-level canvases, then per-(slide, gene)
    MSE on min-max standardized maps (tissue bins only).

    Returns ``((S, G) mse_array, [slide_ids])``. NaN where standardization
    is undefined (zero variance) or no tissue.
    """
    if fov_size <= 0 or pred.shape[0] == 0:
        return np.zeros((0, pred.shape[1] if pred.ndim == 4 else 0)), []
    slides = _stitch_per_slide(
        pred, target, mask, sample_ids, x_starts, y_starts, fov_size, target_grid
    )
    slide_ids = sorted(slides.keys())
    G = pred.shape[1]
    out = np.full((len(slide_ids), G), np.nan, dtype=np.float64)
    for s_idx, sid in enumerate(slide_ids):
        pcan, tcan, mcan = slides[sid]
        tissue = mcan > 0
        if tissue.sum() < 2:
            continue
        for g in range(G):
            pmap = np.where(np.isnan(pcan[g]), 0.0, pcan[g])
            tmap = np.where(np.isnan(tcan[g]), 0.0, tcan[g])
            pn = _standardize_with_mask(pmap, tissue)
            tn = _standardize_with_mask(tmap, tissue)
            if pn is None or tn is None:
                continue
            diff = pn[tissue] - tn[tissue]
            out[s_idx, g] = float((diff * diff).mean())
    return out, slide_ids


def per_fov_per_gene_ssim(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
    win_size: int = 5,
) -> np.ndarray:
    """Per-(FOV, gene) SSIM on min-max standardized (K, K) maps.

    pred/target: (N, G, K, K). mask: (N, K, K). For each (FOV, gene) pair
    both maps are standardized to [0, 1] via :func:`_standardize_with_mask`
    (non-tissue zeroed out after scaling), then SSIM is computed with
    ``data_range=1.0`` and the requested window size.

    K=14 cuts off ~⌊win/2⌋ pixels from each edge of the SSIM kernel — at
    win_size=5 we keep a 10×10 valid region, at win_size=7 only 8×8.
    Defaults to 5 to retain meaningful coverage. Returns (N, G) — NaN where
    a map has zero variance or the window doesn't fit.
    """
    from skimage.metrics import structural_similarity

    N, G, K, _ = pred.shape
    out = np.full((N, G), np.nan, dtype=np.float64)
    ws = int(win_size)
    if ws < 3 or ws > K or ws % 2 == 0:
        return out
    for i in range(N):
        tissue = mask[i] > 0
        if tissue.sum() < 2:
            continue
        for g in range(G):
            pn = _standardize_with_mask(pred[i, g], tissue)
            tn = _standardize_with_mask(target[i, g], tissue)
            if pn is None or tn is None:
                continue
            try:
                out[i, g] = float(
                    structural_similarity(pn, tn, win_size=ws, data_range=1.0)
                )
            except ValueError:
                pass
    return out


def _compute_per_slide_scalars(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
    sample_ids: np.ndarray,
) -> dict[str, dict]:
    """For each unique slide (FOVs sharing a sample_id), compute the headline
    scalar metrics by restricting to that slide's FOVs.

    Returns ``{metric_name: {slide_id: value}}`` for:
    - ``median_per_gene_pearson``
    - ``per_bin_r2_pooled``
    - ``median_morans_i_diff``  (uses per-FOV Moran's I, then |pred−true|)
    - ``median_mse_per_gene_raw``
    - ``median_mse_per_gene_standardized``
    """
    arr = np.asarray(sample_ids)
    slide_ids = sorted(np.unique(arr).tolist(), key=str)
    out: dict[str, dict[str, float]] = {
        "median_per_gene_pearson": {},
        "per_bin_r2_pooled": {},
        "median_morans_i_diff": {},
        "median_mse_per_gene_raw": {},
        "median_mse_per_gene_standardized": {},
    }
    for sid in slide_ids:
        sel = np.where(arr == sid)[0]
        if sel.size == 0:
            continue
        p, t, m = pred[sel], target[sel], mask[sel]
        sid_str = str(sid)
        pearson_g = per_gene_pearson(p, t, m)
        r2 = per_bin_r2(p, t, m)
        mp = per_fov_morans_i(p); mt = per_fov_morans_i(t)
        moran_diff = np.abs(mp - mt)
        mse_raw_g = per_gene_mse_raw(p, t, m)
        mse_std_g = per_gene_mse_standardized(p, t, m)
        def med(x):
            if x.size == 0 or np.all(np.isnan(x)):
                return float("nan")
            return float(np.nanmedian(x))
        out["median_per_gene_pearson"][sid_str] = med(pearson_g)
        out["per_bin_r2_pooled"][sid_str] = (
            float("nan") if r2 != r2 else float(r2)
        )
        out["median_morans_i_diff"][sid_str] = med(moran_diff)
        out["median_mse_per_gene_raw"][sid_str] = med(mse_raw_g)
        out["median_mse_per_gene_standardized"][sid_str] = med(mse_std_g)
    return out


def summarize(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
    gene_panel: list[str],
    *,
    sample_ids: np.ndarray | None = None,
    x_starts: np.ndarray | None = None,
    y_starts: np.ndarray | None = None,
    fov_size: int = 0,
    target_grid: int | None = None,
    ssim_win_fov: int = 5,
    ssim_win_slide: int = 11,
) -> dict:
    """Pack the headline metrics into a JSON-serializable dict.

    The slide-level keys (SSIM + standardized MSE on stitched canvases) and
    per-FOV SSIM require the per-FOV pixel coordinates. They are skipped
    silently when ``sample_ids``/``x_starts``/``y_starts`` or ``fov_size``
    are missing (e.g. old caches without ``x_start``/``y_start`` metadata).
    """
    pearson = per_gene_pearson(pred, target, mask)
    r2 = per_bin_r2(pred, target, mask)
    moran_pred = per_fov_morans_i(pred)
    moran_true = per_fov_morans_i(target)
    moran_diff = np.abs(moran_pred - moran_true)
    mse_bin = per_bin_mse(pred, target, mask)
    mse_gene_raw = per_gene_mse_raw(pred, target, mask)
    mse_gene_std = per_gene_mse_standardized(pred, target, mask)

    K = pred.shape[-1]
    tg = int(target_grid) if target_grid else K
    have_coords = (
        sample_ids is not None and x_starts is not None and y_starts is not None
        and fov_size > 0
    )
    ssim_fov = per_fov_per_gene_ssim(pred, target, mask, win_size=ssim_win_fov)
    if have_coords:
        ssim_slide, slide_ids_ssim = per_slide_per_gene_ssim(
            pred, target, mask, sample_ids, x_starts, y_starts,
            fov_size=fov_size, target_grid=tg, win_size=ssim_win_slide,
        )
        mse_slide, slide_ids_mse = per_slide_per_gene_mse_standardized(
            pred, target, mask, sample_ids, x_starts, y_starts,
            fov_size=fov_size, target_grid=tg,
        )
    else:
        ssim_slide = np.zeros((0, pred.shape[1]))
        mse_slide = np.zeros((0, pred.shape[1]))
        slide_ids_ssim = []
        slide_ids_mse = []

    def _iqr(arr: np.ndarray) -> list[float]:
        if arr.size == 0 or np.all(np.isnan(arr)):
            return [float("nan"), float("nan")]
        return [float(np.nanpercentile(arr, 25)),
                float(np.nanpercentile(arr, 75))]

    def _median(arr: np.ndarray) -> float:
        if arr.size == 0 or np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmedian(arr))

    out = {
        "per_gene_pearson": {g: (None if np.isnan(p) else float(p))
                             for g, p in zip(gene_panel, pearson)},
        "median_per_gene_pearson": float(np.nanmedian(pearson)),
        "iqr_per_gene_pearson": [float(np.nanpercentile(pearson, 25)),
                                 float(np.nanpercentile(pearson, 75))],
        "per_bin_r2_pooled": r2,
        "median_morans_i_diff": float(np.nanmedian(moran_diff)),
        "mse_per_bin": mse_bin.tolist(),
        "mse_per_bin_summary": {
            "mean":   float(mse_bin.mean()),
            "median": float(np.median(mse_bin)),
            "p25":    float(np.percentile(mse_bin, 25)),
            "p75":    float(np.percentile(mse_bin, 75)),
            "p90":    float(np.percentile(mse_bin, 90)),
            "p95":    float(np.percentile(mse_bin, 95)),
            "n_bins": int(mse_bin.size),
        },

        # --- new per-gene MSE summaries ---
        "per_gene_mse_raw": {g: (None if np.isnan(v) else float(v))
                             for g, v in zip(gene_panel, mse_gene_raw)},
        "median_mse_per_gene_raw": _median(mse_gene_raw),
        "iqr_mse_per_gene_raw":    _iqr(mse_gene_raw),
        "per_gene_mse_standardized": {g: (None if np.isnan(v) else float(v))
                                      for g, v in zip(gene_panel, mse_gene_std)},
        "median_mse_per_gene_standardized": _median(mse_gene_std),
        "iqr_mse_per_gene_standardized":    _iqr(mse_gene_std),

        # --- new per-FOV SSIM ---
        "median_ssim_per_fov":   _median(ssim_fov),
        "iqr_ssim_per_fov":      _iqr(ssim_fov),
        "ssim_per_fov_win_size": int(ssim_win_fov),

        # --- new per-slide SSIM + standardized MSE ---
        "median_ssim_per_slide":   _median(ssim_slide),
        "iqr_ssim_per_slide":      _iqr(ssim_slide),
        "ssim_per_slide_win_size": int(ssim_win_slide),
        "ssim_per_slide_per_gene": {
            sid: {g: (None if np.isnan(v) else float(v))
                  for g, v in zip(gene_panel, ssim_slide[i])}
            for i, sid in enumerate(slide_ids_ssim)
        },
        "median_mse_per_slide_standardized": _median(mse_slide),
        "iqr_mse_per_slide_standardized":    _iqr(mse_slide),
        "mse_per_slide_per_gene_standardized": {
            sid: {g: (None if np.isnan(v) else float(v))
                  for g, v in zip(gene_panel, mse_slide[i])}
            for i, sid in enumerate(slide_ids_mse)
        },
        "n_slides_with_coords": int(len(slide_ids_ssim)),
    }

    # --- per-slide aggregation for headline metrics (error bars over slides) ---
    # Requires sample_ids; coords not strictly required for the pooled metrics
    # (Pearson, R², MoranΔ, MSE_raw, MSE_std) but the slide-level SSIM + MSE_std
    # come from the matrices computed above (slide_ids_ssim / slide_ids_mse).
    if sample_ids is not None and len(np.unique(np.asarray(sample_ids))) >= 1:
        per_slide_pooled = _compute_per_slide_scalars(pred, target, mask, sample_ids)

        # Add slide-level SSIM + MSE_std as their own per-slide-scalar entries
        # (median over genes per slide).
        per_slide_pooled["median_ssim_per_slide"] = {
            sid: float(np.nanmedian(ssim_slide[i])) if ssim_slide.size else float("nan")
            for i, sid in enumerate(slide_ids_ssim)
        }
        per_slide_pooled["median_mse_per_slide_standardized"] = {
            sid: float(np.nanmedian(mse_slide[i])) if mse_slide.size else float("nan")
            for i, sid in enumerate(slide_ids_mse)
        }

        def _summarize_across_slides(d: dict[str, float]) -> dict:
            vals = np.asarray(
                [v for v in d.values() if v == v],  # drop NaN
                dtype=np.float64,
            )
            if vals.size == 0:
                return {"mean": float("nan"), "std": float("nan"), "n": 0,
                        "values": {k: (None if v != v else float(v)) for k, v in d.items()}}
            return {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "n":    int(vals.size),
                "values": {k: (None if v != v else float(v)) for k, v in d.items()},
            }

        out["per_slide"] = {
            metric: _summarize_across_slides(per_slide_pooled[metric])
            for metric in per_slide_pooled
        }

    return out


# ============================================================
# Disk helpers (used by the driver too)
# ============================================================
def save_history(
    history: dict,
    out_dir: str,
    state_filename: str = "best.pt",
    history_filename: str = "history.json",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = history.pop("best_state_dict", None)
    if state is not None:
        torch.save(state, out / state_filename)
    with open(out / history_filename, "w") as f:
        json.dump(history, f, indent=2)
