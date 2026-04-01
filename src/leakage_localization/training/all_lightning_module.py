from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass

import torch
from torch import nn, optim
import lightning

from .common import PHASE, LEAKAGE_MODEL, PREPROCESSING

@dataclass
class ALLModuleConfig:
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

class ALLModule(lightning.LightningModule):
    trace_mean: torch.Tensor
    trace_std: torch.Tensor
    trace_min: torch.Tensor
    trace_rng: torch.Tensor

    def __init__(
            self,
            *,
            model_constructor: Callable[[Dict[str, Any]], nn.Module],
            model_kwargs: Dict[str, Any],
            leakage_model: LEAKAGE_MODEL,
            num_labels: int,
            total_steps: int,
            lr_warmup_steps: Optional[int],
            lr_const_steps: Optional[int],
            base_lr: float,
            lr_decay_multiplier: float,
            weight_decay: float,
            label_smoothing: float,
            mtd_kwargs: Dict[str, Any],
            additive_gaussian_noise: float,
            mixup_alpha: float,
            preprocessing: PREPROCESSING,
            random_roll_scale: float,
            random_lpf_scale: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['trace_statistics'])