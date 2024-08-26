#!/usr/bin/env python
# coding=utf-8

"""
https://arxiv.org/abs/2305.05065
"""

import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange
from m4p.models.vit import VIT

class ImageConfig(object):
    def __init__(self) -> None:
        super().__init__()
        self.image_size = 224
        self.patch_size = 16
        self.out_size = 1024
        self.dim = 768
        self.depth = 12
        self.heads = 8
        self.mlp_dim = 2048
        self.pool = 'mean'
        self.channels = 3
        self.dim_head = 64
        self.dropout = 0.0
        self.emb_dropout = 0.0

class TextConfing(object):
    def __init__(self) -> None:
        pass

class RQVAE(nn.Module):
    def __init__(self, image_config : ImageConfig, text_config : TextConfing) -> None:
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