from dataclasses import dataclass

import torch
from torch import nn

@dataclass
class FNNConfig:
    embedding_dim: int
    expansion_factor: int
    dropout_rate: float
    use_bias: bool

    def __post_init__(self):
        assert isinstance(self.embedding_dim, int) and self.embedding_dim > 0
        assert isinstance(self.expansion_factor, int) and self.expansion_factor > 0
        assert isinstance(self.dropout_rate, float) and 0 <= self.dropout_rate < 1
        assert isinstance(self.use_bias, bool)

class MLP(nn.Module):
    def __init__(
            self,
            *,
            embedding_dim: int,
            expansion_factor: int,
            dropout_rate: float,
            use_bias: bool,
    ):
        super().__init__()

        self.config = FNNConfig(
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_bias=use_bias
        )

        self.dense_in = nn.Linear(
            self.config.embedding_dim,
            self.config.expansion_factor*self.config.embedding_dim,
            bias=self.config.use_bias
        )
        self.dense_out = nn.Linear(
            self.config.expansion_factor*self.config.embedding_dim,
            self.config.embedding_dim,
            bias=self.config.use_bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_in(x)
        x = nn.functional.gelu(x)
        x = self.dense_out(x)
        if self.config.dropout_rate > 0:
            x = nn.functional.dropout(x, p=self.config.dropout_rate, training=self.training)
        return x

class GatedFNN(nn.Module):
    def __init__(
            self,
            *,
            embedding_dim: int,
            expansion_factor: int,
            dropout_rate: float,
            use_bias: bool,
    ):
        super().__init__()

        self.config = FNNConfig(
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_bias=use_bias
        )

        self.dense_val = nn.Linear(self.config.embedding_dim, self.config.expansion_factor*self.config.embedding_dim, bias=self.config.use_bias)
        self.dense_gate = nn.Linear(self.config.embedding_dim, self.config.expansion_factor*self.config.embedding_dim, bias=self.config.use_bias)
        self.dense_out = nn.Linear(self.config.expansion_factor*self.config.embedding_dim, self.config.embedding_dim, bias=self.config.use_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val = self.dense_val(x)
        gate = self.dense_gate(x)
        out = self.dense_out(val * nn.functional.silu(gate))
        if self.config.dropout_rate > 0:
            out = nn.functional.dropout(out, p=self.config.dropout_rate, training=self.training)
        return out