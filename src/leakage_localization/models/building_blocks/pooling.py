from typing import Optional

import torch
from torch import nn

from .blocks import CrossAttentionBlock

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

class AttentionPool(CrossAttentionBlock):
    def __init__(
            self,
            *,
            output_tokens: int,
            **kwargs
    ):
        kwargs['use_rope'] = False
        super().__init__(**kwargs)

        self.pre_query = nn.Parameter(torch.empty((output_tokens, self.config.embedding_dim,)))
        nn.init.trunc_normal_(self.pre_query, std=0.02)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, *_ = x.shape
        pre_query = self.pre_query.unsqueeze(0).expand(batch_size, -1, -1)
        return super().forward(pre_query, x, context_mask=attn_mask)