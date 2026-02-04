from typing import Optional

import torch
from torch import nn

from ..building_blocks.blocks import SelfAttentionBlock, CrossAttentionBlock

class PerceiverBlock(nn.Module):
    def __init__(
            self,
            *,
            self_attn_blocks: int,
            embedding_dim: int,
            head_count: int,
            dropout_rate: float,
            use_bias: bool,
            use_rope: bool,
            expansion_factor: int,
    ):
        super().__init__()

        self.cross_attn_block = CrossAttentionBlock(
            embedding_dim=embedding_dim,
            head_count=head_count,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            use_rope=use_rope,
            expansion_factor=expansion_factor
        )
        self.self_attn_blocks = nn.ModuleList([
            SelfAttentionBlock(
                embedding_dim=embedding_dim,
                head_count=head_count,
                dropout_rate=dropout_rate,
                use_bias=use_bias,
                use_rope=use_rope,
                expansion_factor=expansion_factor
            ) for _ in range(self_attn_blocks)
        ])
    
    def forward(self, latent: torch.Tensor, input: torch.Tensor, input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.cross_attn_block(x=latent, context=input, context_mask=input_mask)
        for self_attn_block in self.self_attn_blocks:
            x = self_attn_block(x)
        return x

class PerceiverTrunk(nn.Module):
    def __init__(
            self,
            *,
            latent_dim: int,
            perceiver_blocks: int,
            self_attn_per_cross_attn_blocks: int,
            embedding_dim: int,
            head_count: int,
            dropout_rate: float,
            use_bias: bool,
            use_rope: bool,
            expansion_factor: int,
    ):
        super().__init__()

        self.latent = nn.Parameter(torch.empty((latent_dim, embedding_dim)))
        nn.init.trunc_normal_(self.latent, std=0.02)
        self.perceiver_blocks = nn.ModuleList([
            PerceiverBlock(
                self_attn_blocks=self_attn_per_cross_attn_blocks,
                embedding_dim=embedding_dim,
                head_count=head_count,
                dropout_rate=dropout_rate,
                use_bias=use_bias,
                use_rope=use_rope,
                expansion_factor=expansion_factor
            ) for _ in range(perceiver_blocks)
        ])
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, *_ = x.shape
        latent = self.latent.unsqueeze(0).expand(batch_size, -1, -1)
        for perceiver_block in self.perceiver_blocks:
            latent = perceiver_block(latent, x, input_mask=attn_mask)
        return latent