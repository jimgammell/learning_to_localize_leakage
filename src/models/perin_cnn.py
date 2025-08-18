from typing import Sequence, Optional
from collections import OrderedDict

import torch
from torch import nn

class PerinCNN(nn.Module):
    def __init__(self,
        input_shape: Sequence[int],
        output_classes: int = 256,
        noise_conditional: bool = False,
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        output_dropout: float = 0.0
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.noise_conditional = noise_conditional
        self.conv_stage = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv1d(2 if noise_conditional else 1, 4, kernel_size=41, stride=21, padding=20, padding_mode='circular')),
            ('norm_1', nn.BatchNorm1d(4)),
            ('act_1', nn.SELU()),
            ('pool_1', nn.AvgPool1d(2)),
            ('conv_2', nn.Conv1d(4, 32, kernel_size=9, stride=5, padding=4, padding_mode='circular')),
            ('norm_2', nn.BatchNorm1d(32)),
            ('act_2', nn.SELU()),
            ('pool_2', nn.AvgPool1d(2))
        ]))
        dim = self.conv_stage(torch.randn(1, input_shape[0]+1 if noise_conditional else input_shape[0], *input_shape[1:])).numel()
        self.dense_stage = nn.Sequential(OrderedDict([
            ('dense_1', nn.Linear(dim, 32)),
            ('act_1', nn.SELU()),
            ('dense_2', nn.Linear(32, 32)),
            ('act_2', nn.SELU()),
            ('dense_3', nn.Linear(32, output_classes))
        ]))
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.constant_(mod.bias, 0.01)
    
    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.noise_conditional:
            assert noise is not None
            x = torch.cat([x, noise], dim=1)
        else:
            assert noise is None
        x = self.conv_stage(x)
        x = self.dense_stage(x.flatten(start_dim=1))
        return x