# Adapted from https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py

from typing import Sequence

import torch
from torch import nn

from zaid_wouters_nets.keras_to_pytorch_utils import FlattenTranspose

class MLPBest(nn.Module):
    def __init__(self,
        input_shape: Sequence[int],
        output_classes: int = 256
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes

        self.channels = self.input_shape[0]
        self.timesteps = self.input_shape[1]
        self.mlp = nn.Sequential(
            FlattenTranspose(),
            nn.Linear(self.timesteps, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 256)
        )

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.constant_(mod.bias, 0)
    
    def forward(self, x):
        return self.mlp(x)

class CNNBest(nn.Module):
    def __init__(self,
        input_shape: Sequence[int],
        output_classes: int = 256
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_classes = output_classes

        self.channels = self.input_shape[0]
        self.timesteps = self.input_shape[1]
        self.conv_stage = nn.Sequential(
            nn.Conv1d(self.channels, 64, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(128, 256, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(256, 512, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(512, 512, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )
        self.flatten = FlattenTranspose() # important when using pretrained Keras weights
        self.conv_feature_count = self.conv_stage(torch.randn(1, self.channels, self.timesteps)).numel()
        self.fc_stage = nn.Sequential(
            nn.Linear(self.conv_feature_count, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.output_classes)
        )
    
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.constant_(mod.bias, 0)
            elif isinstance(mod, nn.Conv1d):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.constant_(mod.bias, 0)
    
    def forward(self, x):
        batch_size, channels, timesteps = x.shape
        assert channels == self.channels
        assert timesteps == self.timesteps
        x = self.conv_stage(x)
        x = self.flatten(x)
        x = self.fc_stage(x)
        return x