#!/usr/bin/env python
# coding=utf-8

"""
Multi-headed grouped query self-attention (GQA) layer introduced in https://arxiv.org/pdf/2305.13245
"""
import torch
from torch import nn
from einops import rearrange

class CausalSelfAttention(nn.Module):
    """
    Attention
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

        block_size = att_s.shape[0]
        mask = torch.triu(torch.ones(block_size, block_size), diagonal = 1)
        masked = att_s.masked_fill(mask.bool(), -torch.inf)

        att_w = torch.softmax(
            masked / self.dim_kq ** 0.5, dim = -1
        )

        out = att_w @ v
        return out


# if __name__ == "__main__":
#     attention = CausalSelfAttention(4,4,4)
#     x = torch.randn(2, 4)
#     print(x)
#     print(attention(x))