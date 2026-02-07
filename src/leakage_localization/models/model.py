from typing import Literal, Optional, List, get_args
from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from .building_blocks.feature_preprocessing import Embed, FourierEmbed, Patchifier
from .building_blocks.pooling import AttentionPool, AveragePool, TokenPool
from .building_blocks.attention import SelfAttention, CrossAttention
from .trunks.transformer import TransformerTrunk
from .trunks.perceiver import PerceiverTrunk

TRUNK = Literal[
    'transformer',
    'perceiver'
]
POSITION_EMBEDDING = Literal[
    'learned',
    'rope'
]
POOLING = Literal[
    'average',
    'token',
    'attention'
]
HEAD = Literal[
    'tied',
    'untied'
]
FNN_STYLE = Literal[
    'mlp',
    'gated'
]

@dataclass
class ModelConfig:
    input_length: int
    output_dim: int
    output_count: int
    trunk: TRUNK
    position_embedding: POSITION_EMBEDDING
    pooling: POOLING
    head: HEAD
    fnn_style: FNN_STYLE
    patch_size: Optional[int]
    use_fourier_embed: bool
    fourier_embed_num_bands: Optional[int]
    fourier_embed_sigma: Optional[float]
    embedding_dim: int
    expansion_factor: int
    trunk_blocks: int
    head_count: Optional[int]
    register_tokens: int
    input_dropout_rate: float
    input_droppatch_rate: float
    hidden_dropout_rate: float
    use_bias: bool
    perceiver_latent_dim: Optional[int]
    perceiver_self_attn_per_cross_attn_blocks: Optional[int]

    def __post_init__(self):
        assert isinstance(self.input_length, int) and self.input_length > 0
        assert isinstance(self.output_dim, int) and self.output_dim > 0
        assert isinstance(self.output_count, int) and self.output_count > 0
        assert self.trunk in get_args(TRUNK)
        assert self.position_embedding in get_args(POSITION_EMBEDDING)
        assert self.pooling in get_args(POOLING)
        assert self.head in get_args(HEAD)
        assert self.fnn_style in get_args(FNN_STYLE)
        if self.pooling == 'average':
            assert self.head == 'untied'
        if self.patch_size is not None:
            assert isinstance(self.patch_size, int) and self.patch_size > 0 and self.input_length % self.patch_size == 0
            self.use_patchifier = True
        else:
            self.use_patchifier = False
        assert isinstance(self.use_fourier_embed, bool)
        if self.use_fourier_embed:
            assert isinstance(self.fourier_embed_num_bands, int) and self.fourier_embed_num_bands > 0
            assert isinstance(self.fourier_embed_sigma, float) and self.fourier_embed_sigma > 0
        else:
            assert self.fourier_embed_num_bands is None
            assert self.fourier_embed_sigma is None
        assert isinstance(self.embedding_dim, int) and self.embedding_dim > 0
        assert isinstance(self.expansion_factor, int) and self.expansion_factor > 0
        assert isinstance(self.trunk_blocks, int) and self.trunk_blocks > 0
        if self.head_count is not None:
            assert isinstance(self.head_count, int) and self.head_count > 0 and self.embedding_dim % self.head_count == 0
        assert isinstance(self.register_tokens, int) and self.register_tokens >= 0
        assert self.register_tokens == 0 # FIXME
        assert isinstance(self.input_dropout_rate, float) and 0 <= self.input_dropout_rate < 1
        assert isinstance(self.input_droppatch_rate, float) and 0 <= self.input_droppatch_rate < 1
        assert isinstance(self.hidden_dropout_rate, float) and 0 <= self.hidden_dropout_rate < 1
        assert isinstance(self.use_bias, bool)
        if self.trunk == 'perceiver':
            assert isinstance(self.perceiver_latent_dim, int) and self.perceiver_latent_dim > 0
            assert isinstance(self.perceiver_self_attn_per_cross_attn_blocks, int) and self.perceiver_self_attn_per_cross_attn_blocks > 0
        else:
            assert self.perceiver_latent_dim is None
            assert self.perceiver_self_attn_per_cross_attn_blocks is None

class Model(nn.Module):
    def __init__(
            self,
            *,
            input_length: int,
            output_dim: int,
            output_count: int,
            trunk: TRUNK,
            position_embedding: POSITION_EMBEDDING,
            pooling: POOLING,
            head: HEAD,
            fnn_style: FNN_STYLE,
            patch_size: Optional[int],
            use_fourier_embed: bool,
            fourier_embed_num_bands: Optional[int],
            fourier_embed_sigma: Optional[float],
            embedding_dim: int,
            expansion_factor: int,
            trunk_blocks: int,
            register_tokens: int,
            perceiver_latent_dim: Optional[int],
            perceiver_self_attn_per_cross_attn_blocks: Optional[int],
            head_count: Optional[int],
            input_dropout_rate: float,
            input_droppatch_rate: float,
            hidden_dropout_rate: float,
            use_bias: bool,
    ):
        super().__init__()

        self.config = ModelConfig(
            input_length=input_length,
            output_dim=output_dim,
            output_count=output_count,
            trunk=trunk,
            position_embedding=position_embedding,
            pooling=pooling,
            head=head,
            fnn_style=fnn_style,
            patch_size=patch_size,
            use_fourier_embed=use_fourier_embed,
            fourier_embed_num_bands=fourier_embed_num_bands,
            fourier_embed_sigma=fourier_embed_sigma,
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor,
            trunk_blocks=trunk_blocks,
            register_tokens=register_tokens,
            perceiver_latent_dim=perceiver_latent_dim,
            perceiver_self_attn_per_cross_attn_blocks=perceiver_self_attn_per_cross_attn_blocks,
            head_count=head_count,
            input_dropout_rate=input_dropout_rate,
            input_droppatch_rate=input_droppatch_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            use_bias=use_bias,
        )

        if self.config.use_fourier_embed:
            self.fourier_embedding = FourierEmbed(
                in_dims=1,
                num_bands=self.config.fourier_embed_num_bands,
                sigma=self.config.fourier_embed_sigma
            )
            in_dims = 2*self.config.fourier_embed_num_bands
        else:
            in_dims = 1
        if self.config.use_patchifier:
            self.patchifier = Patchifier(
                in_dims=in_dims,
                in_seq_len=self.config.input_length,
                patch_size=self.config.patch_size
            )
            in_dims = in_dims*self.config.patch_size
            in_seq_len = self.config.input_length // self.config.patch_size
        else:
            in_seq_len = self.config.input_length
        self.embedding = Embed(
            in_dims=in_dims,
            in_seq_len=in_seq_len,
            embedding_dim=self.config.embedding_dim,
            learned_position_embedding=self.config.position_embedding == 'learned'
        )

        if self.config.trunk == 'perceiver':
            self.trunk = PerceiverTrunk(
                latent_dim=self.config.perceiver_latent_dim,
                perceiver_blocks=self.config.trunk_blocks,
                self_attn_per_cross_attn_blocks=self.config.perceiver_self_attn_per_cross_attn_blocks,
                embedding_dim=self.config.embedding_dim,
                head_count=self.config.head_count,
                dropout_rate=self.config.hidden_dropout_rate,
                use_bias=self.config.use_bias,
                use_rope=self.config.position_embedding == 'rope',
                expansion_factor=self.config.expansion_factor,
                fnn_style=self.config.fnn_style
            )
        elif self.config.trunk == 'transformer':
            self.trunk = TransformerTrunk(
                transformer_blocks=self.config.trunk_blocks,
                embedding_dim=self.config.embedding_dim,
                head_count=self.config.head_count,
                dropout_rate=self.config.hidden_dropout_rate,
                use_bias=self.config.use_bias,
                use_rope=self.config.position_embedding == 'rope',
                expansion_factor=self.config.expansion_factor,
                fnn_style=self.config.fnn_style
            )
        else:
            assert False
        
        if self.config.pooling == 'attention':
            self.pool = AttentionPool(
                output_tokens=self.config.output_count,
                embedding_dim=self.config.embedding_dim,
                head_count=self.config.head_count,
                dropout_rate=self.config.hidden_dropout_rate,
                use_bias=self.config.use_bias,
                expansion_factor=self.config.expansion_factor,
                fnn_style=self.config.fnn_style
            )
        elif self.config.pooling == 'average':
            self.pool = AveragePool(
                output_tokens=self.config.output_count
            )
        elif self.config.pooling == 'token':
            self.cls_tokens = nn.Parameter(torch.empty(self.config.output_count, self.config.embedding_dim))
            nn.init.trunc_normal_(self.cls_tokens, std=0.02)
            self.pool = TokenPool(
                output_tokens=self.config.output_count
            )
        else:
            assert False
        
        if self.config.head == 'tied':
            self.heads = nn.Linear(self.config.embedding_dim, self.config.output_dim)
        elif self.config.head == 'untied':
            self.heads = nn.ModuleList([
                nn.Linear(self.config.embedding_dim, self.config.output_dim)
                for _ in range(self.config.output_count)
            ])
        else:
            assert False
        
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.trunc_normal_(mod.weight, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            if isinstance(mod, nn.LayerNorm):
                nn.init.ones_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        if self.config.trunk == 'perceiver':
            layer_count = self.config.trunk_blocks*(1 + self.config.perceiver_self_attn_per_cross_attn_blocks)
        elif self.config.trunk == 'transformer':
            layer_count = self.config.trunk_blocks
        else:
            assert False
        for mod in self.trunk.modules():
            if isinstance(mod, (SelfAttention, CrossAttention)):
                nn.init.trunc_normal_(mod.to_out.weight, std=0.02/sqrt(2*layer_count))
        for mod in self.heads.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, seq_len = x.shape
        device = x.device
        if attn_mask is not None:
            attn_mask = attn_mask.squeeze()
        x = x.permute(0, 2, 1)
        if self.config.input_dropout_rate > 0:
            x = nn.functional.dropout(x, p=self.config.input_dropout_rate, training=self.training)
        if self.config.use_fourier_embed:
            x = self.fourier_embedding(x)
        if self.config.use_patchifier:
            x = self.patchifier(x)
        x = self.embedding(x)
        _, seq_len, _ = x.shape
        if self.config.input_droppatch_rate > 0 and self.training:
            droppatch_mask = torch.rand((batch_size, seq_len), device=device) >= self.config.input_droppatch_rate
            all_dropped = ~droppatch_mask.any(dim=1)
            if all_dropped.any():
                droppatch_mask[all_dropped, 0] = True
            if attn_mask is not None:
                assert attn_mask.shape == droppatch_mask.shape
                attn_mask = attn_mask & droppatch_mask
            else:
                attn_mask = droppatch_mask
        if self.config.pooling == 'token':
            cls_tokens = self.cls_tokens.unsqueeze(0).expand(batch_size, -1, -1)
            x = torch.cat([x, cls_tokens], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.ones((batch_size, self.config.output_count), device=attn_mask.device, dtype=attn_mask.dtype)], dim=1)
                seq_len += self.config.output_count
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(batch_size, 1, 1, seq_len)
        x = self.trunk(x, attn_mask=attn_mask)
        x = self.pool(x)
        if self.config.head == 'tied':
            x = self.heads(x)
        elif self.config.head == 'untied':
            x = torch.cat([
                head(xp) for head, xp in zip(self.heads, x.unbind(dim=1))
            ], dim=1)
        else:
            assert False
        return x