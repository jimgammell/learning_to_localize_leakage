

import torch
from torch import nn

class Head(nn.Module):
    def __init__(
            self,
            in_dims: int,
            out_dims: int,
            dense_dropout: float = 0.05
    ):
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.dense_dropout = dense_dropout

        self.layers = nn.Sequential(
            nn.Linear(self.in_dims, 256),
            nn.SiLU(), # called swish in Keras
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.Dropout(self.dense_dropout),
            nn.SiLU(),
            nn.Linear(256, self.out_dims)
        )

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight) # default Keras Dense init -- different from PyTorch default
                nn.init.zeros_(mod.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)