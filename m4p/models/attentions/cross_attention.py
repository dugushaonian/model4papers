#!/usr/bin/env python
# coding=utf-8

import torch
from torch import nn
from einops import rearrange

class CrossAttention(nn.Module):
    """
    MultiHeadAttention
    """
    def __init__(self, dim: int, dim_kq: int = 64, dim_v: int = 64) -> None:
        super().__init__()
        self.dim_kq = dim_kq
        self.w_q = nn.Parameter(torch.rand(dim, dim_kq))
        self.w_k = nn.Parameter(torch.rand(dim, dim_kq))
        self.w_v = nn.Parameter(torch.rand(dim, dim_v))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        q = x1 @ self.w_q
        k = x2 @ self.w_k
        v = x2 @ self.w_v

        att_s = q @ k.T
        att_w = torch.softmax(
            att_s / self.dim_kq ** 0.5, dim = -1
        )
        out = att_w @ v
        return out


# if __name__ == "__main__":
#     attention = CrossAttention(4, 4, 4)
#     x1 = torch.randn(2, 4)
#     x2 = torch.randn(2, 4)
#     print(x1)
#     print(x2)
#     print(attention(x1, x2))