#!/usr/bin/env python
# coding=utf-8

"""
"""
import torch
from torch import nn
from einops import rearrange

class SelfAttention(nn.Module):
    """
    SelfAttention
    """
    def __init__(self, dim: int, dim_kq: int = 64, dim_v: int = 64) -> None:
        super().__init__()
        self.dim_kq = dim_kq
        self.w_q = nn.Parameter(torch.rand(dim, dim_kq))
        self.w_k = nn.Parameter(torch.rand(dim, dim_kq))
        self.w_v = nn.Parameter(torch.rand(dim, dim_v))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = x @ self.w_k
        q = x @ self.w_q
        v = x @ self.w_v

        att_s = q @ k.T
        att_w = torch.softmax(
            att_s / self.dim_kq ** 0.5, dim = -1
        )

        out = att_w @ v

        return out


# if __name__ == "__main__":
#     self_attention = SelfAttention(4,4,4)
#     x = torch.randn(2, 4)
#     print(x)
#     print(self_attention(x))