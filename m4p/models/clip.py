#!/usr/bin/env python
# coding=utf-8


"""
https://arxiv.org/abs/2103.00020
"""

import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange

from m4p.models.vit import VIT, VitConfig, ViT

class TextConfing(object):
    def __init__(self) -> None:
        pass


class CLIP(nn.Module):
    def __init__(self, image_config : VitConfig, text_config : TextConfing) -> None:
        super().__init__()
        self.image_encoder = ViT(image_config)
        self.text_encoder = None

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text)

    def forward(self, text_x: torch.Tensor, img_x: torch.Tensor) -> torch.Tensor:
        # text_features = self.text_encoder(text_x)
        # image_features = self.image_encoder(img_x)

        # text_features = text_features / text_features.norm(dim = 1, keepdim = True)
        # image_features = image_features / image_features.norm(dim = 1, keepdim = True)

        # logit_scale = 
        pass
