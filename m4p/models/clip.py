#!/usr/bin/env python
# coding=utf-8


"""
https://arxiv.org/abs/2103.00020
"""

import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange

from m4p.models.vit import VIT, VitConfig

class TextConfing(object):
    def __init__(self) -> None:
        pass

class CLIP(nn.Module):
    def __init__(self, image_config : VitConfig, text_config : TextConfing) -> None:
        super().__init__()
        self.image_encoder = VIT(
            image_size = image_config.image_size,
            patch_size = image_config.patch_size,
            out_size = image_config.out_size,
            dim = image_config.dim,
            depth = image_config.depth,
            heads = image_config.heads,
            mlp_dim = image_config.mlp_dim,
            pool = image_config.pool,
            channels = image_config.channels,
            dim_head = image_config.dim_head,
            dropout = image_config.dropout,
            emb_dropout = image_config.emb_dropout
        )
        self.text_encoder = None

    def forward(self, text_x: torch.Tensor, img_x: torch.Tensor) -> torch.Tensor:
        pass