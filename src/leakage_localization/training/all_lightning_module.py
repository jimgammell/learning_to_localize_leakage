from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass

import torch
from torch import nn, optim

from .common import PHASE, LEAKAGE_MODEL, PREPROCESSING

@dataclass
class ALLLightningModule:
    model_constructor: Callable[[Dict[str, Any]], nn.Module]
    model_kwargs: Dict[str, Any]
    leakage_model: LEAKAGE_MODEL
    num_labels: int
    total_steps: int
    lr_warmup_steps: Optional[int]
    lr_const_steps: Optional[int]
    base_lr: float
    lr_decay_multiplier: float
    weight_decay: float
    label_smoothing: float
    mtd_kwargs: Dict[str, Any]
    additive_gaussian_noise: float
    mixup_alpha: float
    preprocessing: PREPROCESSING
    random_roll_scale: float
    random_lpf_scale: float