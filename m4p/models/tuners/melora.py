#!/usr/bin/env python
# coding=utf-8

"""
https://arxiv.org/pdf/2402.17263
"""

import torch
from torch import nn

class MELoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ranks: list = [4], alphas: list = [1.0], dropouts: list = [0.0]) -> None:
        super().__init__()
        self.l_num = len(ranks)
        self.f_in_dim = in_dim // self.l_num
        self.f_out_dim = out_dim // self.l_num
        self.ranks = ranks
        self.alphas = alphas
        while len(self.alphas) < len(self.ranks):
            self.alphas.append(1.0)

        while len(dropouts) < len(ranks):
            dropouts.append(0.0)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for dropout in dropouts])

        self.As = nn.ParameterList([])
        self.Bs = nn.ParameterList([])
        self.scalings = []
        for i, rank in enumerate(self.ranks):
            self.As.append(nn.Parameter(torch.zeros((self.f_in_dim, rank))))
            self.Bs.append(nn.Parameter(torch.zeros((rank, self.f_out_dim))))
            self.scalings.append(self.alphas[i] / rank)
            nn.init.kaiming_uniform_(self.As[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Bs[i], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = []
        for i, rank in enumerate(self.ranks):
            temp.append(
                self.dropouts[i](x[:, i * (self.f_in_dim) : (i + 1) * (self.f_in_dim)]) @ self.As[i] @ self.Bs[i] * self.scalings[i]
            )
        return torch.concat(temp, dim = -1)


class MELoRALinear(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, ranks: list = [4], alphas: list = [1.0], dropouts: list = [0.0], bias = False) -> None:
        super().__init__(in_dim, out_dim, bias)
        self.melora = MELoRALayer(in_dim, out_dim, ranks, alphas, dropouts)

        # Freeze
        self.weight.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.melora(x)


# if __name__ == "__main__":
#     import math
#     melora_layer = MELoRALinear(8, 8, [2, 3])
#     x = torch.randn(2, 8)
#     print(x)
#     print(melora_layer(x))