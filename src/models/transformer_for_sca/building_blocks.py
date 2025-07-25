from typing import Optional
from math import sqrt, ceil

import numpy as np
import torch
from torch import nn
from rotary_embedding_torch import RotaryEmbedding

from .config import TransformerConfig

class NormLayer(nn.Module):
    def __init__(self, config: TransformerConfig, layer_num: Optional[int] = None):
        super().__init__()
        self.config = config
        if self.config.rescale_norm_outputs:
            assert layer_num is not None
            self.scale = 1./sqrt(layer_num)
        else:
            self.scale = 1.
        self.weight = nn.Parameter(torch.ones(self.config.embedding_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # I feel like scaling should be equivalent to just initializing self.weight to 1/self.scale instead of 1.
        #   Just going with this approach though in case I'm missing something.
        return self.scale*nn.functional.rms_norm(x, self.weight.shape, weight=self.weight, eps=self.config.norm_eps)

class AttentionLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.head_dim = self.config.embedding_dim // self.config.attn_head_count
        self.to_qkv = nn.Linear(self.config.embedding_dim, 3*self.config.embedding_dim, bias=False)
        self.to_out = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias=self.config.bias)
        self.out_dropout = nn.Dropout(self.config.dropout)
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        nn.init.xavier_uniform_(self.to_qkv.weight)
        nn.init.xavier_uniform_(self.to_out.weight)
        if self.config.bias:
            nn.init.constant_(self.to_out.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, token_count, embedding_dim = x.shape
        qkv = self.to_qkv(x).split(self.config.embedding_dim, dim=2)
        q, k, v = map(
            lambda x: x.view(batch_size, token_count, self.config.attn_head_count, self.head_dim).transpose(1, 2), qkv
        )
        q, k = map(lambda x: self.rotary_emb.rotate_queries_or_keys(x), (q, k))
        if mask is not None:
            mask = mask.reshape(batch_size, 1, 1, token_count).expand(-1, self.config.attn_head_count, token_count, -1)
        pre_out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.config.dropout if self.training else 0, is_causal=False, scale=1/sqrt(self.head_dim))
        pre_out = pre_out.transpose(1, 2).contiguous().view(batch_size, token_count, embedding_dim)
        out = self.to_out(pre_out)
        out = self.out_dropout(out)
        return out

class FeedForwardLayer(nn.Module):
    def __init__(self, config: TransformerConfig, output_dims: Optional[int] = None):
        super().__init__()
        self.config = config
        self.output_dims = output_dims
        self.w12 = nn.Linear(self.config.embedding_dim, 2*self.config.embedding_dim, bias=self.config.bias)
        self.w3 = nn.Linear(self.config.embedding_dim, self.output_dims or self.config.embedding_dim, bias=self.config.bias)
        self.dropout = nn.Dropout(self.config.dropout)
        nn.init.xavier_uniform_(self.w12.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        if self.config.bias:
            nn.init.constant_(self.w12.bias, 0)
            nn.init.constant_(self.w3.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).split(self.config.embedding_dim, dim=2)
        out = self.w3(nn.functional.silu(x1)*x2)
        out = self.dropout(out)
        return out

class AttentionPoolingLayer(nn.Module):
    def __init__(self, config: TransformerConfig, output_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.head_dim = self.config.embedding_dim // self.config.attn_head_count
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)
        self.output_token_count = (
            self.config.output_head_count if self.config.head_type == 'simple-shared'
            else 4*self.config.output_head_count+1 if self.config.head_type == 'ascadv1'
            else -1
        )
        self.pool_queries = nn.Parameter(torch.randn(1, self.output_token_count, self.config.embedding_dim))
        self.to_kv = nn.Linear(self.config.embedding_dim, 2*self.config.embedding_dim, bias=False)
        if self.config.shared_head:
            self.to_out = nn.Linear(self.config.embedding_dim, self.output_dim or self.config.embedding_dim, bias=True)
            nn.init.xavier_uniform_(self.to_out.weight)
            nn.init.constant_(self.to_out.bias, 0)
        else:
            self.to_out = nn.ModuleList([
                nn.Linear(self.config.embedding_dim, self.output_dim or self.config.output_head_classes, bias=True)
                for _ in range(self.config.output_head_count)
            ])
            for head in self.to_out:
                nn.init.xavier_uniform_(head.weight)
                nn.init.constant_(head.bias, 0)
        nn.init.xavier_uniform_(self.to_kv.weight)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        batch_size, token_count, embedding_dim = x.shape
        q = self.pool_queries.expand(batch_size, -1, -1)
        kv = self.to_kv(x).split(self.config.embedding_dim, dim=2)
        q, k, v = map(
            lambda x: x.view(batch_size, -1, self.config.attn_head_count, self.head_dim).transpose(1, 2), (q, *kv)
        )
        k = self.rotary_emb.rotate_queries_or_keys(k)
        if mask is not None:
            mask = mask.reshape(batch_size, 1, 1, token_count).expand(-1, self.config.attn_head_count, self.output_token_count, -1)
        pre_out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0, is_causal=False)
        pre_out = pre_out.transpose(1, 2).contiguous().view(batch_size, self.output_token_count, embedding_dim)
        if self.config.shared_head:
            logits = self.to_out(pre_out)
        else:
            logits = []
            for head, token in zip(self.to_out, pre_out.transpose(0, 1)):
                logits.append(head(token))
            logits = torch.stack(logits, dim=1)
        return logits

class Patchifier(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv1d(1, self.config.embedding_dim, kernel_size=self.config.patch_size, stride=self.config.patch_size, bias=self.config.bias)
        self.dropout = nn.Dropout(self.config.input_dropout)
    
    def forward(self, x):
        batch_size, _, dim = x.shape
        token_count = ceil(dim/self.config.patch_size)
        padding = self.config.patch_size*ceil(dim/self.config.patch_size) - dim
        if padding > 0:
            x = torch.cat([x, torch.zeros(batch_size, 1, padding, dtype=x.dtype, device=x.device)])
        mask = None
        if self.training:
            if self.config.input_noise_std > 0:
                x = x + self.config.input_noise_std*torch.randn_like(x)
            if self.config.input_jitter > 0:
                shifts = np.random.randint(-self.config.input_jitter, self.config.input_jitter+1, (batch_size,))
                for idx, shift in enumerate(shifts):
                    x[idx] = torch.roll(x[idx], shifts=shift, dims=-1)
        x = self.patch_embedding(x).transpose(1, 2)
        if self.training and self.config.dropword > 0:
            mask = torch.rand((batch_size, token_count), device=x.device) > self.config.dropword
            x = x * mask.reshape(batch_size, token_count, 1).to(x.dtype)
        x = self.dropout(x)
        return x, mask