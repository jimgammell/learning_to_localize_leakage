from typing import Optional

import torch
from torch import nn

from ..building_blocks.blocks import SelfAttentionBlock

class TransformerTrunk(nn.Module):
    def __init__(
            self,
            *,
            transformer_blocks: int,
            embedding_dim: int,
            head_count: int,
            dropout_rate: float,
            use_bias: bool,
            use_rope: bool,
            expansion_factor: int
    ):
        super().__init__()

        self.transformer_blocks = nn.ModuleList([
            SelfAttentionBlock(
                embedding_dim=embedding_dim,
                head_count=head_count,
                dropout_rate=dropout_rate,
                use_bias=use_bias,
                use_rope=use_rope,
                expansion_factor=expansion_factor
            ) for _ in range(transformer_blocks)
        ])
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attn_mask=attn_mask)
        return x