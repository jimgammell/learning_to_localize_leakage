# Adapted from https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py

from typing import Sequence
from collections import OrderedDict

import torch
from torch import nn

from common import *
from .zaid_wouters_nets.keras_to_pytorch_utils import FlattenTranspose, keras_to_torch_mod, unpack_keras_params

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
        self.mlp = nn.Sequential(OrderedDict([
            ('transpose', FlattenTranspose()),
            ('dense_1', nn.Linear(self.timesteps, 200)),
            ('relu_1', nn.ReLU()),
            ('dense_2', nn.Linear(200, 200)),
            ('relu_2', nn.ReLU()),
            ('dense_3', nn.Linear(200, 200)),
            ('relu_3', nn.ReLU()),
            ('dense_4', nn.Linear(200, 200)),
            ('relu_4', nn.ReLU()),
            ('dense_5', nn.Linear(200, 200)),
            ('relu_5', nn.ReLU()),
            ('dense_6', nn.Linear(200, 256))
        ]))

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.constant_(mod.bias, 0)
    
    def forward(self, x):
        return self.mlp(x)

    def load_pretrained_keras_params(self, weights_path):
        keras_params = unpack_keras_params(weights_path)
        keras_to_torch_mod(keras_params['dense_1'], self.mlp.dense_1)
        keras_to_torch_mod(keras_params['dense_2'], self.mlp.dense_2)
        keras_to_torch_mod(keras_params['dense_3'], self.mlp.dense_3)
        keras_to_torch_mod(keras_params['dense_4'], self.mlp.dense_4)
        keras_to_torch_mod(keras_params['dense_5'], self.mlp.dense_5)
        keras_to_torch_mod(keras_params['dense_6'], self.mlp.dense_6)

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
        self.conv_stage = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv1d(self.channels, 64, kernel_size=11, padding=5, stride=2 if self.timesteps == 1400 else 1)),
            ('relu_1', nn.ReLU()),
            ('pool_1', nn.AvgPool1d(2)),
            ('conv_2', nn.Conv1d(64, 128, kernel_size=11, padding=5)),
            ('relu_2', nn.ReLU()),
            ('pool_2', nn.AvgPool1d(2)),
            ('conv_3', nn.Conv1d(128, 256, kernel_size=11, padding=5)),
            ('relu_3', nn.ReLU()),
            ('pool_3', nn.AvgPool1d(2)),
            ('conv_4', nn.Conv1d(256, 512, kernel_size=11, padding=5)),
            ('relu_4', nn.ReLU()),
            ('pool_4', nn.AvgPool1d(2)),
            ('conv_5', nn.Conv1d(512, 512, kernel_size=11, padding=5)),
            ('relu_5', nn.ReLU()),
            ('pool_5', nn.AvgPool1d(2))
        ]))
        self.flatten = FlattenTranspose() # important when using pretrained Keras weights
        self.conv_feature_count = self.conv_stage(torch.randn(1, self.channels, self.timesteps)).numel()
        self.fc_stage = nn.Sequential(OrderedDict([
            ('dense_1', nn.Linear(self.conv_feature_count, 4096)),
            ('relu_1', nn.ReLU()),
            ('dense_2', nn.Linear(4096, 4096)),
            ('relu_2', nn.ReLU()),
            ('dense_3', nn.Linear(4096, self.output_classes))
        ]))
    
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
        print(self)
        print(x.shape)
        x = self.conv_stage(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.fc_stage(x)
        print(x.shape)
        return x
    
    def load_pretrained_keras_params(self, weights_path):
        keras_params = unpack_keras_params(weights_path)
        keras_to_torch_mod(keras_params['block1_conv1'], self.conv_stage.conv_1)
        keras_to_torch_mod(keras_params['block2_conv1'], self.conv_stage.conv_2)
        keras_to_torch_mod(keras_params['block3_conv1'], self.conv_stage.conv_3)
        keras_to_torch_mod(keras_params['block4_conv1'], self.conv_stage.conv_4)
        keras_to_torch_mod(keras_params['block5_conv1'], self.conv_stage.conv_5)
        keras_to_torch_mod(keras_params['fc1'], self.fc_stage.dense_1)
        keras_to_torch_mod(keras_params['fc2'], self.fc_stage.dense_2)
        keras_to_torch_mod(keras_params['predictions'], self.fc_stage.dense_3)