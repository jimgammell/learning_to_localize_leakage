# Adapted from nanoGPT implementation: https://github.com/karpathy/nanoGPT

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from rotary_embedding_torch import RotaryEmbedding

@dataclass
class AttentionConfig:
    embedding_dim: int
    head_count: Optional[int] = None
    dropout_rate: float
    use_bias: bool
    use_rope: bool
    
    def __post_init__(self):
        assert isinstance(self.embedding_dim, int) and self.embedding_dim > 0
        if self.head_count is None:
            assert self.embedding_dim % 64 == 0
            self.head_count = self.embedding_dim // 64
        assert isinstance(self.head_count, int) and (0 < self.head_count <= self.embedding_dim) and (self.embedding_dim % self.head_count == 0)
        assert isinstance(self.dropout_rate, float) and 0 <= self.dropout_rate < 1
        assert isinstance(self.use_bias, bool)
        assert isinstance(self.use_rope, bool)
        self.head_dim = self.embedding_dim // self.head_count
        if self.use_rope:
            self.rope_dim = self.head_dim // 2

class SelfAttention(nn.Module):
    def __init__(
            self,
            *,
            embedding_dim: int,
            head_count: Optional[int],
            dropout_rate: float,
            use_bias: bool,
            use_rope: bool,
    ):
        super().__init__()

        self.config = AttentionConfig(
            embedding_dim=embedding_dim,
            head_count=head_count,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            use_rope=use_rope
        )

        self.to_qkv = nn.Linear(self.config.embedding_dim, 3*self.config.embedding_dim, bias=self.config.use_bias)
        self.to_out = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias=self.config.use_bias)
        if self.config.use_rope:
            self.rope = RotaryEmbedding(dim=self.config.rope_dim)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, emb_dim = x.shape
        assert emb_dim == self.config.embedding_dim
        qkv = self.to_qkv(x)
        q, k, v = qkv.split(self.config.embedding_dim, dim=2)
        q = q.view(batch_size, seq_len, self.config.head_count, self.config.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.config.head_count, self.config.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.config.head_count, self.config.head_dim).transpose(1, 2)
        if self.config.use_rope:
            q = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)
        out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.config.dropout_rate if self.training else 0
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.embedding_dim)
        out = self.to_out(out)
        if self.config.dropout_rate > 0:
            out = nn.functional.dropout(out, p=self.config.dropout_rate, training=self.training)
        return out

class CrossAttention(nn.Module):
    def __init__(
            self,
            *,
            embedding_dim: int,
            head_count: Optional[int],
            dropout_rate: float,
            use_bias: bool,
            use_rope: bool
    ):
        super().__init__()

        self.config = AttentionConfig(
            embedding_dim=embedding_dim,
            head_count=head_count,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            use_rope=use_rope
        )

        self.to_q = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias=self.config.use_bias)
        self.to_kv = nn.Linear(self.config.embedding_dim, 2*self.config.embedding_dim, bias=self.config.use_bias)
        self.to_out = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias=self.config.use_bias)
        if self.config.use_rope:
            self.rope = RotaryEmbedding(dim=self.config.rope_dim)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, emb_dim = x.shape
        _, context_seq_len, _ = context.shape
        assert batch_size == context.shape[0]
        assert emb_dim == context.shape[2] == self.config.embedding_dim
        q = self.to_q(x)
        kv = self.to_kv(context)
        k, v = kv.split(self.config.embedding_dim, dim=2)
        q = q.view(batch_size, seq_len, self.config.head_count, self.config.head_dim).transpose(1, 2)
        k = k.view(batch_size, context_seq_len, self.config.head_count, self.config.head_dim).transpose(1, 2)
        v = v.view(batch_size, context_seq_len, self.config.head_count, self.config.head_dim).transpose(1, 2)
        if self.config.use_rope:
            q = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)
        out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=context_mask,
            dropout_p=self.config.dropout_rate if self.training else 0
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.embedding_dim)
        out = self.to_out(out)
        if self.config.dropout_rate > 0:
            out = nn.functional.dropout(out, p=self.config.dropout_rate, training=self.training)
        return out