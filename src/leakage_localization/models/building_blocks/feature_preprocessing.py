from math import pi, log
from typing import Literal

import torch
from torch import nn

POSITION_EMBEDDING = Literal['learned', 'sinusoidal', 'none']

def _sinusoidal_position_embedding(seq_len: int, embedding_dim: int) -> torch.Tensor:
    assert embedding_dim % 2 == 0
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) * (-log(10000.0) / embedding_dim))
    pe = torch.zeros(seq_len, embedding_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class FourierEmbed(nn.Module):
    def __init__(
            self,
            *,
            in_dims: int,
            num_bands: int,
            sigma: float
    ):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dims, num_bands)*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = (2*pi*x) @ self.B
        out = torch.cat([proj.sin(), proj.cos()], dim=-1)
        return out

class Patchifier(nn.Module):
    def __init__(
            self,
            *,
            embedding_dim: int,
            in_channels: int,
            in_seq_len: int,
            patch_size: int,
            position_embedding: POSITION_EMBEDDING = 'none'
    ):
        super().__init__()

        assert in_seq_len % patch_size == 0
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.in_seq_len = in_seq_len
        self.patch_size = patch_size
        assert self.in_seq_len % self.patch_size == 0
        assert self.patch_size % 2 == 0
        self.position_embedding = position_embedding

        self.proj = nn.Conv1d(
            self.in_channels, self.embedding_dim,
            kernel_size=self.patch_size, #2*self.patch_size,
            stride=self.patch_size,
            bias=False,
            padding=self.patch_size//2,
            padding_mode='circular'
        )
        #self.norm = nn.LayerNorm(self.embedding_dim)
        if self.position_embedding == 'learned':
            self.pos_emb = nn.Parameter(torch.empty(self.in_seq_len//self.patch_size, self.embedding_dim))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        elif self.position_embedding == 'sinusoidal':
            self.register_buffer('pos_emb', _sinusoidal_position_embedding(self.in_seq_len//self.patch_size, self.embedding_dim))
        elif self.position_embedding == 'none':
            pass
        else:
            assert False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        assert seq_len == self.in_seq_len
        x = self.proj(x.transpose(1, 2)).transpose(1, 2)
        if self.position_embedding in ('learned', 'sinusoidal'):
            x = x + self.pos_emb.unsqueeze(0)
        x = self.norm(x)
        return x