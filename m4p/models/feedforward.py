#!/usr/bin/env python
# coding=utf-8

from torch import nn

class FeedForward(nn.Module):
    """
    FeedForward
    """
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.feedforward(x)

