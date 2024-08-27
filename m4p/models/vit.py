#!/usr/bin/env python
# coding=utf-8

"""
https://arxiv.org/abs/2010.11929
"""

import gin
import torch
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange

from m4p.models.transformer import Transformer
from m4p.utils.pair import pair

@gin.configurable
class VitConfig(object):
    def __init__(self,
        image_size: int = 224,
        patch_size: int = 16,
        out_size: int = 1024,
        dim: int = 768,
        depth: int = 12,
        heads: int = 8,
        mlp_dim: int = 2048,
        pool: str = 'mean',
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_size = out_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.pool = pool
        self.channels = channels
        self.dim_head = dim_head
        self.dropout = dropout
        self.emb_dropout = emb_dropout


class VIT(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        out_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = 'cls',
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0
    ) -> None:
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dim must be div by the patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be cls or mean'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, out_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class ViT(nn.Module):
    def __init__(self, vit_cfg: VitConfig) -> None:
        super().__init__()
        self.vit = VIT(
            image_size = vit_cfg.image_size,
            patch_size = vit_cfg.patch_size,
            out_size = vit_cfg.out_size,
            dim = vit_cfg.dim,
            depth = vit_cfg.depth,
            heads = vit_cfg.heads,
            mlp_dim = vit_cfg.mlp_dim,
            pool = vit_cfg.pool,
            channels = vit_cfg.channels,
            dim_head = vit_cfg.dim_head,
            dropout = vit_cfg.dropout,
            emb_dropout = vit_cfg.emb_dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)