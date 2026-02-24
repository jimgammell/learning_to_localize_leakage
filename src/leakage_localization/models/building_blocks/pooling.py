from typing import Optional

import torch
from torch import nn

from .attention import CrossAttention

class AveragePool(nn.Module):
    def __init__(
            self,
            *,
            output_tokens: int
    ):
        super().__init__()

        self.output_tokens = output_tokens
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1, keepdim=True).expand(-1, self.output_tokens, -1)
        return x

class TokenPool(nn.Module):
    def __init__(
            self,
            *,
            output_tokens: int
    ):
        super().__init__()

        self.output_tokens = output_tokens
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] >= self.output_tokens
        return x[:, -self.output_tokens:, :]

class AttentionPool(nn.Module):
    def __init__(
            self,
            *,
            output_tokens: int,
            embedding_dim: int,
            **kwargs
    ):
        super().__init__()
        self.attn_input_norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=True, bias=True)
        self.attn_context_norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=True, bias=True)
        self.cross_attn = CrossAttention(embedding_dim=embedding_dim, use_rope=True, **kwargs)
        self.pre_query = nn.Parameter(torch.empty((output_tokens, embedding_dim)))
        nn.init.trunc_normal_(self.pre_query, std=0.02)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, *_ = x.shape
        pre_query = self.pre_query.unsqueeze(0).expand(batch_size, -1, -1)
        return self.cross_attn(self.attn_input_norm(pre_query), context=self.attn_context_norm(x), context_mask=attn_mask)