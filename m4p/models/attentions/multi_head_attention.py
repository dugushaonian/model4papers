#!/usr/bin/env python
# coding=utf-8

import torch
from torch import nn
from einops import rearrange
from m4p.models.attentions.self_attention import SelfAttention

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention
    """
    def __init__(self, dim: int, dim_kq: int = 64, dim_v: int = 64, heads_n: int = 8) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(dim, dim_kq, dim_v) for _ in range(heads_n)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim = -1)


# if __name__ == "__main__":
#     self_attention = MultiHeadAttention(32, 4, 4, 8)
#     x = torch.randn(2, 32)
#     print(x)
#     print(self_attention(x))