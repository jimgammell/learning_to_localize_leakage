from random import randint
from typing import List

import torch
from torch import nn

class Compose(nn.Module):
    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = transforms
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x

class Standardize(nn.Module):
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean.reshape(1, -1), dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale.reshape(1, -1), dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / (self.scale + 1e-8)
        return x

class Normalize(nn.Module):
    def __init__(self, min: torch.Tensor, max: torch.Tensor):
        super().__init__()
        self.register_buffer('min', min.reshape(1, -1))
        self.register_buffer('max', max.reshape(1, -1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.min) / (self.max - self.min + 1e-8)
        return x

class RandomRoll(nn.Module):
    def __init__(self, max_shift: int):
        super().__init__()
        assert isinstance(max_shift, int) and max_shift > 0
        self.max_shift = max_shift
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        roll_amount = randint(0, self.max_shift)
        if roll_amount > 0:
            x = x.roll(roll_amount, 1)
        return x

class AdditiveGaussianNoise(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        assert isinstance(scale, float) and scale > 0
        self.scale = scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.scale*torch.randn_like(x)
        x = x + noise
        return x