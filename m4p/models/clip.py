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
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = None
        self.text_encoder = None

    def forward(self, text_x: torch.Tensor, img_x: torch.Tensor) -> torch.Tensor:
        pass