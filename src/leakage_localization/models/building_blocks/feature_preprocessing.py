from math import pi

import torch
from torch import nn

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
            in_dims: int,
            in_seq_len: int,
            patch_size: int,
    ):
        super().__init__()

        assert in_seq_len % patch_size == 0
        self.in_dims = in_dims
        self.in_seq_len = in_seq_len
        self.patch_size = patch_size
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        assert seq_len == self.in_seq_len
        out = x.reshape(batch_size, self.in_seq_len//self.patch_size, self.patch_size*self.in_dims)
        return out

class Embed(nn.Module):
    def __init__(
            self,
            *,
            in_dims: int,
            in_seq_len: int,
            embedding_dim: int,
            learned_position_embedding: bool
    ):
        super().__init__()

        self.in_dims = in_dims
        self.in_seq_len = in_seq_len
        self.embedding_dim = embedding_dim
        self.learned_position_embedding = learned_position_embedding

        self.embedding = nn.Linear(self.in_dims, self.embedding_dim, bias=False)
        if self.learned_position_embedding:
            self.pos_emb = nn.Parameter(torch.empty(self.in_seq_len, self.embedding_dim))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, in_dims = x.shape
        assert seq_len == self.in_seq_len
        assert in_dims == self.in_dims
        out = self.embedding(x)
        if self.learned_position_embedding:
            pos_emb = self.pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
            out = out + pos_emb
        return out