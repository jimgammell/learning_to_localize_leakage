from typing import Optional, Literal

import torch
from torch import nn

from .attention import SelfAttention, CrossAttention
from .fnn import MLP, GatedFNN

FNN_STYLE = Literal[
    'mlp',
    'gated'
]

class SelfAttentionBlock(nn.Module):
    def __init__(
            self,
            *,
            embedding_dim: int,
            head_count: Optional[int],
            dropout_rate: float,
            use_bias: bool,
            use_rope: bool,
            expansion_factor: int,
            fnn_style: FNN_STYLE
    ):
        super().__init__()

        self.attn_norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=True, bias=use_bias)
        self.attn = SelfAttention(embedding_dim=embedding_dim, head_count=head_count, dropout_rate=dropout_rate, use_bias=use_bias, use_rope=use_rope)
        self.fnn_norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=True, bias=use_bias)
        if fnn_style == 'mlp':
            self.fnn = MLP(embedding_dim=embedding_dim, expansion_factor=expansion_factor, dropout_rate=dropout_rate, use_bias=use_bias)
        elif fnn_style == 'gated':
            self.fnn = GatedFNN(embedding_dim=embedding_dim, expansion_factor=expansion_factor, dropout_rate=dropout_rate, use_bias=use_bias)
        else:
            assert False
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask=attn_mask)
        x = x + self.fnn(self.fnn_norm(x))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            *,
            embedding_dim: int,
            head_count: Optional[int],
            dropout_rate: float,
            use_bias: bool,
            use_rope: bool,
            expansion_factor: int,
            fnn_style: FNN_STYLE
    ):
        super().__init__()

        self.attn_input_norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=True, bias=use_bias)
        self.attn_context_norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=True, bias=use_bias)
        self.attn = CrossAttention(embedding_dim=embedding_dim, head_count=head_count, dropout_rate=dropout_rate, use_bias=use_bias, use_rope=use_rope)
        self.fnn_norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=True, bias=use_bias)
        if fnn_style == 'mlp':
            self.fnn = MLP(embedding_dim=embedding_dim, expansion_factor=expansion_factor, dropout_rate=dropout_rate, use_bias=use_bias)
        elif fnn_style == 'gated':
            self.fnn = GatedFNN(embedding_dim=embedding_dim, expansion_factor=expansion_factor, dropout_rate=dropout_rate, use_bias=use_bias)
        else:
            assert False
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_input_norm(x), self.attn_context_norm(context), context_mask=context_mask)
        x = x + self.fnn(self.fnn_norm(x))
        return x