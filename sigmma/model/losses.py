from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: image embeddings, shape (B, D)
        z_j: spatial embeddings, shape (B, D)
        """
        # Filter: Only valid (non-zero) pairs
        valid_mask = z_j.norm(dim=-1) > 0.0
        z_i = z_i[valid_mask]
        z_j = z_j[valid_mask]

        if len(z_i) == 0:
            print("No valid pairs found. Skipping loss computation.")
            return torch.tensor(0.0, device=z_i.device, requires_grad=True)

        z_i = torch.nn.functional.normalize(z_i, dim=-1)
        z_j = torch.nn.functional.normalize(z_j, dim=-1)

        # Compute cosine similarity
        logits = torch.mm(z_i, z_j.t()) / self.temperature

        # Create labels for positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Compute the loss
        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss


class symmetric_InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(symmetric_InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: image embeddings, shape (B, D)
        z_j: spatial embeddings, shape (B, D)
        """
        # Filter: Only valid (non-zero) pairs
        valid_mask = z_j.norm(dim=-1) > 0.0
        z_i = z_i[valid_mask]
        z_j = z_j[valid_mask]

        if len(z_i) == 0:
            print("No valid pairs found. Skipping loss computation.")
            return torch.tensor(0.0, device=z_i.device, requires_grad=True)

        z_i = torch.nn.functional.normalize(z_i, dim=-1)
        z_j = torch.nn.functional.normalize(z_j, dim=-1)

        # Compute cosine similarity
        logits_xy = torch.mm(z_i, z_j.t()) / self.temperature
        logits_yx = torch.mm(z_j, z_i.t()) / self.temperature

        # Create labels for positive pairs
        labels = torch.arange(z_i.size(0), device=z_i.device)

        # Compute the loss
        loss_xy = torch.nn.functional.cross_entropy(logits_xy, labels)
        loss_yx = torch.nn.functional.cross_entropy(logits_yx, labels)

        return (loss_xy + loss_yx) / 2


class GEXAwareSoftContrastiveLoss(nn.Module):
    """InfoNCE with GEX-similarity soft target on the off-diagonal.

    Mixes the standard identity target with a row-stochastic Q_gex built from
    raw-GEX cosine similarity (top-k softmax per row, diagonal masked). With
    alpha=0 reproduces standard symmetric InfoNCE exactly.
    """

    def __init__(
        self,
        tau: float = 0.07,
        tau_gex: float = 0.1,
        alpha: float = 0.2,
        mode: Literal["topk", "full_softmax"] = "topk",
        topk: int = 5,
        gex_preprocess: Literal["none", "log1p", "zscore", "log1p_zscore"] = "log1p_zscore",
        clamp_negative_gex_sim: bool = True,
        symmetric: bool = True,
        detach_gex_targets: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.tau = tau
        self.tau_gex = tau_gex
        self.alpha = alpha
        self.mode = mode
        self.topk = topk
        self.gex_preprocess = gex_preprocess
        self.clamp_negative_gex_sim = clamp_negative_gex_sim
        self.symmetric = symmetric
        self.detach_gex_targets = detach_gex_targets
        self.eps = eps

    def preprocess_gex(self, gex: torch.Tensor) -> torch.Tensor:
        x = gex
        if self.gex_preprocess in ("log1p", "log1p_zscore"):
            x = torch.log1p(x.clamp_min(0.0))
        if self.gex_preprocess in ("zscore", "log1p_zscore"):
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x = (x - mean) / (std + self.eps)
        return x

    def build_gex_soft_targets(self, gex_pre: torch.Tensor) -> torch.Tensor:
        """Return Q_gex (B, B), zero diagonal, rows sum to 1 (or 0 if B==1).

        Legacy single-panel path: assumes all rows in the batch share the
        same gene-axis (e.g. uniform shared HVG panel). Use
        ``build_gex_soft_targets_panel_aware`` for the multi-panel case where
        same-panel pairs are the only valid soft-target candidates.
        """
        B = gex_pre.size(0)
        device = gex_pre.device
        dtype = gex_pre.dtype
        if B <= 1:
            return torch.zeros(B, B, device=device, dtype=dtype)

        x_n = F.normalize(gex_pre, dim=-1)
        sim = x_n @ x_n.t()  # (B, B)
        if self.clamp_negative_gex_sim:
            sim = sim.clamp_min(0.0)
        eye = torch.eye(B, device=device, dtype=torch.bool)
        sim_masked = sim.masked_fill(eye, float("-inf"))

        if self.mode == "topk":
            k = min(self.topk, B - 1)
            top_vals, top_idx = sim_masked.topk(k, dim=-1)
            top_probs = F.softmax(top_vals / self.tau_gex, dim=-1)
            Q = torch.zeros(B, B, device=device, dtype=top_probs.dtype)
            Q.scatter_(-1, top_idx, top_probs)
        elif self.mode == "full_softmax":
            Q = F.softmax(sim_masked / self.tau_gex, dim=-1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        row_sum = Q.sum(dim=-1, keepdim=True)
        Q = Q / (row_sum + self.eps)
        if self.detach_gex_targets:
            Q = Q.detach()
        return Q

    def build_gex_soft_targets_panel_aware(
        self,
        gex_pre: torch.Tensor,                # (B, K_max)
        raw_gex_valid_mask: torch.Tensor,     # (B, K_max) bool
        panel_ids: torch.Tensor,              # (B,) int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Panel-aware Q_gex.

        Cross-panel positions are MASKED (not zero similarity): they are
        excluded from the GEX-soft target distribution and from its
        row-normalization. The InfoNCE component (computed by caller) still
        treats cross-panel pairs as standard negatives.

        Same-panel cosine similarity is computed on the SVG-valid positions
        per panel (all rows of one panel share the same K_panel = sum(mask)).

        Returns
        -------
        Q : (B, B) panel-block-diagonal soft targets (cross-panel = 0,
            same-panel off-diagonal rows sum to 1 per panel partner row).
        q_gex_mask : (B, B) bool — True where Q_gex is defined (off-diagonal
                     same-panel positions). Loss caller uses this to gate
                     per-row α.
        has_same_panel_partner : (B,) bool — True if row has ≥1 off-diagonal
                                 same-panel partner in this batch.
        """
        B = gex_pre.size(0)
        device = gex_pre.device
        dtype = gex_pre.dtype
        Q = torch.zeros(B, B, device=device, dtype=dtype)
        if B <= 1:
            zero_mask = torch.zeros(B, B, device=device, dtype=torch.bool)
            zero_partner = torch.zeros(B, device=device, dtype=torch.bool)
            return Q, zero_mask, zero_partner

        eye = torch.eye(B, device=device, dtype=torch.bool)
        panel_eq = panel_ids[:, None] == panel_ids[None, :]
        q_gex_mask = panel_eq & ~eye  # (B, B) bool

        # Loop over unique panels in batch (≤ 8 typically; cheap).
        for pid in torch.unique(panel_ids):
            idx = (panel_ids == pid).nonzero(as_tuple=True)[0]
            n_pid = int(idx.size(0))
            if n_pid < 2:
                continue
            # All rows of same panel share K_panel = valid_mask.sum() — take
            # the first row's mask as the canonical one.
            mask_row = raw_gex_valid_mask[idx[0]]
            K_panel = int(mask_row.sum().item())
            if K_panel == 0:
                continue
            valid_positions = mask_row.nonzero(as_tuple=True)[0]
            block_gex = gex_pre[idx][:, valid_positions]  # (n_pid, K_panel)
            x_n = F.normalize(block_gex, dim=-1)
            sim_block = x_n @ x_n.t()  # (n_pid, n_pid)
            if self.clamp_negative_gex_sim:
                sim_block = sim_block.clamp_min(0.0)
            eye_block = torch.eye(n_pid, device=device, dtype=torch.bool)
            sim_masked = sim_block.masked_fill(eye_block, float("-inf"))
            if self.mode == "topk":
                k_block = min(self.topk, n_pid - 1)
                top_vals, top_idx = sim_masked.topk(k_block, dim=-1)
                top_probs = F.softmax(top_vals / self.tau_gex, dim=-1)
                block_Q = torch.zeros(n_pid, n_pid, device=device,
                                      dtype=top_probs.dtype)
                block_Q.scatter_(-1, top_idx, top_probs)
            elif self.mode == "full_softmax":
                block_Q = F.softmax(sim_masked / self.tau_gex, dim=-1)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            row_sum_block = block_Q.sum(dim=-1, keepdim=True)
            block_Q = block_Q / (row_sum_block + self.eps)
            Q[idx[:, None], idx[None, :]] = block_Q.to(dtype)

        has_same_panel_partner = q_gex_mask.any(dim=-1)
        if self.detach_gex_targets:
            Q = Q.detach()
        return Q, q_gex_mask, has_same_panel_partner

    @staticmethod
    def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        return -(target * log_p).sum(dim=-1).mean()

    def forward(
        self,
        h_img: torch.Tensor,
        h_spatial: torch.Tensor,
        gex: torch.Tensor,
        raw_gex_valid_mask: Optional[torch.Tensor] = None,
        panel_ids: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        """h_img, h_spatial: (B, D); gex: (B, G_gex).

        ``gex`` is whatever the dataset feeds — either a top-K selected-gene
        subset (could be SVGs or HVGs depending on how the panel was built) or
        the FULL panel gene axis (``use_full_panel`` in the dataset). The
        per-panel cosine similarity here works for any gene count via
        ``raw_gex_valid_mask``. If the dataset already log-normalizes the FOV
        pseudobulk (``gex_lognorm``), set ``gex_preprocess='zscore'`` to avoid
        double-logging; otherwise ``gex`` is raw counts and ``log1p_zscore``
        applies the log here.

        Panel-aware mode (when `panel_ids` is given):
          - Q_gex defined only on same-panel off-diagonals (cross-panel
            positions masked / unobserved, NOT pushed to zero).
          - Per-row alpha_i = alpha · 1[row has same-panel partner]. Rows
            with no same-panel partners reduce to exact vanilla InfoNCE.
        Legacy mode (panel_ids is None): single-panel batch, original behavior.
        """
        valid_mask = h_spatial.norm(dim=-1) > 0.0
        h_img = h_img[valid_mask]
        h_spatial = h_spatial[valid_mask]
        gex = gex[valid_mask]
        if raw_gex_valid_mask is not None:
            raw_gex_valid_mask = raw_gex_valid_mask[valid_mask]
        if panel_ids is not None:
            panel_ids = panel_ids[valid_mask]

        if h_img.size(0) == 0:
            print("No valid pairs found. Skipping loss computation.")
            zero = torch.tensor(0.0, device=h_img.device, requires_grad=True)
            if return_dict:
                return {"loss": zero}
            return zero

        h_img_n = F.normalize(h_img, dim=-1)
        h_spa_n = F.normalize(h_spatial, dim=-1)
        logits_i2s = (h_img_n @ h_spa_n.t()) / self.tau
        B = logits_i2s.size(0)

        gex_pre = self.preprocess_gex(gex)

        eye = torch.eye(B, device=logits_i2s.device, dtype=gex_pre.dtype)
        if panel_ids is not None and raw_gex_valid_mask is not None:
            Q_gex, q_gex_mask, has_partner = self.build_gex_soft_targets_panel_aware(
                gex_pre, raw_gex_valid_mask, panel_ids,
            )
            # Per-row alpha: 0 if no same-panel partner (vanilla fallback).
            alpha_per_row = self.alpha * has_partner.to(gex_pre.dtype)
            Q = (1.0 - alpha_per_row[:, None]) * eye + alpha_per_row[:, None] * Q_gex
        else:
            Q_gex = self.build_gex_soft_targets(gex_pre)
            Q = (1.0 - self.alpha) * eye + self.alpha * Q_gex
        # Defensive row-normalization (rows where alpha_i=0 are exact one-hot,
        # rows where alpha_i=alpha are convex mix of two row-stochastic vectors).
        Q = Q / (Q.sum(dim=-1, keepdim=True) + self.eps)

        loss_i2s = self.soft_cross_entropy(logits_i2s, Q)
        if self.symmetric:
            logits_s2i = logits_i2s.t()
            # Same Q_gex applies: FOV-pair GEX similarity is symmetric, so the
            # mass row i puts on item j is the same whether anchor is image-i or
            # spatial-i. (Top-k selection per row keeps Q row-stochastic; rows of
            # Q_gex.T are NOT necessarily row-stochastic, so we re-use Q itself.)
            loss_s2i = self.soft_cross_entropy(logits_s2i, Q)
            loss = 0.5 * (loss_i2s + loss_s2i)
        else:
            loss_s2i = None
            loss = loss_i2s

        if not return_dict:
            return loss

        with torch.no_grad():
            off_diag = Q * (1.0 - eye)
            soft_pos_count = (off_diag > self.eps).float().sum(dim=-1)
            # gex_sim values where Q_gex put mass (off-diag positives)
            x_n = F.normalize(gex_pre, dim=-1)
            sim = x_n @ x_n.t()
            mass_indicator = (Q_gex > self.eps).float()
            gex_sim_pos_sum = (sim * mass_indicator).sum(dim=-1)
            gex_sim_pos_count = mass_indicator.sum(dim=-1).clamp_min(1.0)
            mean_gex_sim_soft_pos = (gex_sim_pos_sum / gex_sim_pos_count).mean()
            # alignment accuracies
            arange = torch.arange(B, device=logits_i2s.device)
            align_i2s = (logits_i2s.argmax(dim=-1) == arange).float().mean()
            align_s2i = (logits_i2s.t().argmax(dim=-1) == arange).float().mean()
            # logit summaries
            diag_logits = logits_i2s.diagonal()
            logits_off = logits_i2s.masked_fill(eye.bool(), float("-inf"))
            top1_neg_logits, top1_neg_idx = logits_off.max(dim=-1)
            mean_pos_logit = diag_logits.mean()
            mean_top1_neg_logit = top1_neg_logits.mean()
            # gex_sim at top-1 negative position per anchor
            gex_sim_top1_neg = sim.gather(-1, top1_neg_idx.unsqueeze(-1)).squeeze(-1).mean()

        return {
            "loss": loss,
            "loss_i2s": loss_i2s,
            "loss_s2i": loss_s2i,
            "mean_diag_target_mass": Q.diagonal().mean(),
            "mean_soft_positive_mass": off_diag.sum(dim=-1).mean(),
            "mean_num_soft_positives": soft_pos_count.mean(),
            "mean_gex_sim_soft_positives": mean_gex_sim_soft_pos,
            "alignment_acc_i2s": align_i2s,
            "alignment_acc_s2i": align_s2i,
            "mean_positive_logit": mean_pos_logit,
            "mean_top1_negative_logit": mean_top1_neg_logit,
            "mean_gex_sim_top1_negative": gex_sim_top1_neg,
        }


def bleep_cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
