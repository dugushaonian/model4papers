#!/usr/bin/env python
# coding=utf-8

"""
https://arxiv.org/abs/2106.09685
"""

import torch
from torch import nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, dropout: float = 0.0, alpha: float = 1.0) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.zeros((in_dim, rank)))
        self.B = nn.Parameter(torch.zeros((rank, out_dim)))
        self.scaling = self.alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_shape = x.shape
        x_flat = x.view(-1, origin_shape[-1])
        lora_out = (self.dropout(x_flat) @ self.A @ self.B * self.scaling)
        lora_out = lora_out.view(*origin_shape[:-1], -1)
        out = x + lora_out
        return out


if __name__ == "__main__":
    import math
    lora_layer = LoRALayer(8, 8)
    nn.init.kaiming_uniform_(lora_layer.A, a=math.sqrt(5))
    nn.init.kaiming_uniform_(lora_layer.B, a=math.sqrt(5))
    x = torch.randn(2, 8)
    print(x)
    print(lora_layer(x))