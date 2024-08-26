#!/usr/bin/env python
# coding=utf-8


"""
https://arxiv.org/abs/2103.00020
"""

import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange

class CLIP(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.0, emb_dropout = 0.0):
        super().__init__()
        self.image_encoder = None
        self.text_encoder = None

    def forward(self, text_x, img_x):
        pass