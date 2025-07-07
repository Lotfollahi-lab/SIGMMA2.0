# Adapted code from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/dino_head.py
# Modified by dj16@sanger.ac.uk

# Original copyright notice:
# ----------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=10,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)

class CrossAttention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_heads: int = 4,
                 bias: bool = False):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads

        self.Q = nn.Linear(input_dim, output_dim, bias=bias)
        self.K = nn.Linear(input_dim, output_dim, bias=bias)
        self.V = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x1, x2):
        """
        x1: (B, D) - query
        x2: (B, D) - key/value
        """
        
        B, D = x1.shape

        Q = self.Q(x1).view(B, self.num_heads, -1).transpose(1, 2)
        K = self.K(x2).view(B, self.num_heads, -1).transpose(1, 2)
        V = self.V(x2).view(B, self.num_heads, -1).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous().view(B, -1)
        return output