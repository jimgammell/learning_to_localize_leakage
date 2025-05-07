from typing import Sequence

import torch
from torch import nn

class YapCNN(nn.Module): # based on https://github.com/trevor-yap/OccPoIs/blob/main/Architecure_and_Hyperparameter_SearchSpace_considered.pdf
    def __init__(self,
        input_shape: Sequence[int],
        output_classes: int = 256,
        noise_conditional: bool = False
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.noise_conditional = noise_conditional

        self.channels = input_shape[0]
        self.timesteps = input_shape[1]
        if self.noise_conditional:
            self.channels *= 2
        self.conv_stage = nn.Sequential(
            nn.Conv1d(self.channels, 128, kernel_size=25, stride=1, padding=12),
            nn.SELU(),
            nn.BatchNorm1d(128),
            nn.AvgPool1d(128)
        )
        self.conv_feature_count = self.conv_stage(torch.randn(1, self.channels, self.timesteps)).numel()
        self.fc_stage = nn.Sequential(
            nn.Linear(self.conv_feature_count, 20),
            nn.SELU(),
            nn.Linear(20, 15),
            nn.SELU(),
            nn.Linear(15, output_classes)
        )

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.constant_(mod.bias, 0)
            elif isinstance(mod, nn.Conv1d):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.constant_(mod.bias, 0)
            elif isinstance(mod, nn.BatchNorm1d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)
    
    def forward(self, *args):
        if self.noise_conditional:
            (x, noise) = args
            x = torch.cat([x, noise], dim=1)
        else:
            (x,) = args
        batch_size, channels, timesteps = x.shape
        assert channels == self.channels
        assert timesteps == self.timesteps
        x = self.conv_stage(x)
        x = x.view(batch_size, self.conv_feature_count)
        x = self.fc_stage(x)
        return x