#!/usr/bin/env python
# coding=utf-8

import torch
from torch import nn

class FeedForward(nn.Module):
    """
    FeedForward
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feedforward(x)

