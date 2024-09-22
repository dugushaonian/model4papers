#!/usr/bin/env python
# coding=utf-8

"""
https://arxiv.org/abs/2305.05065
"""

import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange
from m4p.models.vit import VIT, VitConfig

class RQVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass