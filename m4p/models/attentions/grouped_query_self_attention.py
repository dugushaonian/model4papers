#!/usr/bin/env python
# coding=utf-8

"""
Multi-headed grouped query self-attention (GQA) layer introduced in https://arxiv.org/pdf/2305.13245
"""
import torch
from torch import nn
from einops import rearrange

class GroupedQuerySelfAttention(nn.Module):
    """
    Attention
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass