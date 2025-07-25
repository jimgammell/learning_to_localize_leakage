from dataclasses import dataclass
from math import isfinite

from typing import Literal, Any

@dataclass
class TransformerConfig:
    patch_size: int = 1000
    layer_count: int = 12
    attn_head_count: int = 12
    embedding_dim: int = 768
    dropout: float = 0.1
    input_dropout: float = 0.1
    dropword: float = 0.1
    head_dropout: float = 0.1
    input_noise_std: float = 0.1
    input_jitter: int = 0
    bias: bool = False
    norm_eps: float = 1e-5
    rescale_norm_outputs: bool = True
    output_head_count: int = 16
    output_head_classes: int = 256
    shared_head: bool = True
    head_type: Literal['simple-shared', 'ascadv1'] = 'simple-shared'
    input_shape: Any = None
    output_classes: Any = None
    hidden_dropout: Any = None
    output_dropout: Any = None

    def __post_init__(self):
        assert isinstance(self.patch_size, int)
        assert isinstance(self.layer_count, int)
        assert isinstance(self.attn_head_count, int)
        assert isinstance(self.embedding_dim, int)
        assert isinstance(self.dropout, float)
        assert isinstance(self.input_dropout, float)
        assert isinstance(self.dropword, float)
        assert isinstance(self.bias, bool)
        assert isinstance(self.norm_eps, float)
        assert isinstance(self.rescale_norm_outputs, bool)
        assert isinstance(self.output_head_count, int)
        assert isinstance(self.output_head_classes, int)
        assert isinstance(self.shared_head, bool)
        assert self.head_type in ['simple-shared', 'ascadv1']
        assert 0 < self.patch_size
        assert 0 < self.layer_count
        assert 0 < self.attn_head_count
        assert 0 < self.embedding_dim
        assert 0 <= self.dropout < 1
        assert 0 <= self.input_dropout < 1
        assert 0 <= self.dropword < 1
        assert 0 < self.norm_eps and isfinite(self.norm_eps)
        assert 0 < self.output_head_count
        assert 0 < self.output_head_classes