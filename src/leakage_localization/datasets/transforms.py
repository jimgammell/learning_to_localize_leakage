from typing import List
from random import randint

import numpy as np
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
    def __init__(self, shift_scale: float):
        super().__init__()
        assert isinstance(shift_scale, float) and shift_scale > 0
        self.shift_scale = shift_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift_sgn = 2*randint(0, 1) - 1
        shift_amt = int(abs(self.shift_scale*np.random.standard_normal()))
        if shift_amt != 0:
            x = nn.functional.pad(x, (shift_amt, shift_amt), mode='reflect')
            if shift_sgn > 0:
                x = x[..., :-2*shift_amt]
            else:
                x = x[..., 2*shift_amt:]
        return x

class RandomLPF(nn.Module):
    def __init__(self, smooth_scale: float):
        super().__init__()
        assert isinstance(smooth_scale, float) and smooth_scale > 0
        self.smooth_scale = smooth_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float16)
        smooth_radius = int(abs((self.smooth_scale*np.random.standard_normal())))
        if smooth_radius != 0:
            orig_shape = x.shape
            x = nn.functional.pad(x, (smooth_radius, smooth_radius), mode='reflect')
            x = nn.functional.avg_pool1d(x.view(1, 1, -1), kernel_size=2*smooth_radius + 1, stride=1).view(*orig_shape)
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