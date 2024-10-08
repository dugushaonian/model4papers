#!/usr/bin/env python
# coding=utf-8

"""
https://arxiv.org/abs/2106.09685
"""

import torch
from torch import nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0) -> None:
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
        return lora_out


class LoRALinear(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, alpha: float = 1.0, bias = True, dropout: float = 0.0) -> None:
        super().__init__(in_dim, out_dim, bias)
        self.lora = LoRALayer(in_dim, out_dim, rank, alpha, dropout)

        # Freeze
        self.weight.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.lora(x)


# if __name__ == "__main__":
#     import math
#     lora_layer = LoRALayer(8, 8)
#     nn.init.kaiming_uniform_(lora_layer.lora.A, a=math.sqrt(5))
#     nn.init.kaiming_uniform_(lora_layer.lora.B, a=math.sqrt(5))
#     x = torch.randn(2, 8)
#     print(x)
#     print(lora_layer(x))