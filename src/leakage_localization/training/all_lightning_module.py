from typing import Callable, Dict, Any, Optional, get_args
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn, optim
import lightning

from leakage_localization.models.advll_submodules import SelectionMechanism
from leakage_localization.models.building_blocks.bits_and_bytes import BitLogitsToByteLogits, HwLogitsToByteLogits
from .cosine_decay_lr_scheduler import CosineDecayLRSched
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
    sm_lr_multiplier: float
    lr_decay_multiplier: float
    weight_decay: float
    label_smoothing: float
    sm_relax_temp: float
    gamma_bar: float
    mtd_kwargs: Dict[str, Any]
    additive_gaussian_noise: float
    mixup_alpha: float
    preprocessing: PREPROCESSING
    random_roll_scale: float
    random_lpf_scale: float

    def __post_init__(self):
        assert self.leakage_model in get_args(LEAKAGE_MODEL)
        self.num_classes = {'id': 256, 'hw': 9, 'bit': 8}[self.leakage_model]
        assert isinstance(self.num_labels, int) and self.num_labels > 0
        assert isinstance(self.num_classes, int) and self.num_classes > 0
        assert isinstance(self.total_steps, int) and self.total_steps > 0
        if self.lr_warmup_steps is not None:
            assert isinstance(self.lr_warmup_steps, int) and self.lr_warmup_steps >= 0
        if self.lr_const_steps is not None:
            assert isinstance(self.lr_const_steps, int) and self.lr_const_steps >= 0
        assert isinstance(self.base_lr, float) and self.base_lr > 0
        assert isinstance(self.sm_lr_multiplier, float) and self.sm_lr_multiplier > 0
        if self.lr_decay_multiplier is not None:
            assert isinstance(self.lr_decay_multiplier, float) and 0 <= self.lr_decay_multiplier <= 1
        assert isinstance(self.weight_decay, float) and self.weight_decay >= 0
        assert isinstance(self.label_smoothing, float) and 0 <= self.label_smoothing < 1
        assert isinstance(self.sm_relax_temp, float) and self.sm_relax_temp > 0
        assert isinstance(self.gamma_bar, float) and 0 < self.gamma_bar < 1
        assert isinstance(self.mtd_kwargs, dict) and all(isinstance(k, str) for k in self.mtd_kwargs)
        assert isinstance(self.additive_gaussian_noise, float) and self.additive_gaussian_noise >= 0
        assert isinstance(self.mixup_alpha, float) and self.mixup_alpha >= 0
        assert self.preprocessing in get_args(PREPROCESSING)
        assert isinstance(self.random_roll_scale, float) and self.random_roll_scale >= 0
        assert isinstance(self.random_lpf_scale, float) and self.random_lpf_scale >= 0

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
            trace_statistics: Dict[str, NDArray[np.floating]],
            additive_gaussian_noise: float,
            mixup_alpha: float,
            preprocessing: PREPROCESSING,
            random_roll_scale: float,
            random_lpf_scale: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['trace_statistics'])
        self.automatic_optimization = False
        self.config = ALLModuleConfig(**self.hparams)
        self.model: nn.Module = self.config.model_constructor(
            output_dim=self.config.num_classes,
            **self.config.model_kwargs
        )
        self.selection_mechanism = SelectionMechanism(
            in_features=self.config.model_kwargs['input_length'],
            gamma_bar=self.config.gamma_bar,
            relaxation_temp=self.config.sm_relax_temp
        )
        assert isinstance(self.model, nn.Module)
        if self.config.leakage_model == 'bit':
            self.logits_to_byte_logits = BitLogitsToByteLogits()
        elif self.config.leakage_model == 'hw':
            self.logits_to_byte_logits = HwLogitsToByteLogits()
        elif self.config.leakage_model == 'id':
            self.logits_to_byte_logits = nn.Identity()
        else:
            assert False
        self.register_buffer('trace_mean', torch.from_numpy(trace_statistics['mean']).float(), persistent=False)
        self.register_buffer('trace_std', torch.from_numpy(trace_statistics['var']).float().sqrt() + 1e-6, persistent=False)
        self.register_buffer('trace_min', torch.from_numpy(trace_statistics['min']).float(), persistent=False)
        self.register_buffer('trace_rng', torch.from_numpy(trace_statistics['max'] - trace_statistics['min']).float() + 1e-6, persistent=False)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        yes_wd_params, no_wd_params = [], []
        for param_name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or param_name.endswith('.bias'):
                no_wd_params.append(param)
            else:
                yes_wd_params.append(param)
        param_groups = [
            {'params': yes_wd_params, 'weight_decay': self.config.weight_decay},
            {'params': no_wd_params, 'weight_decay': 0}
        ]
        model_optimizer = optim.AdamW(
            param_groups,
            lr=self.config.base_lr
        )
        model_lr_scheduler = CosineDecayLRSched(
            model_optimizer,
            total_steps=self.config.total_steps,
            lr_warmup_steps=self.config.lr_warmup_steps,
            lr_const_steps=self.config.lr_const_steps,
            lr_decay_multiplier=self.config.lr_decay_multiplier
        )
        sm_optimizer = optim.AdamW(
            self.selection_mechanism.parameters(),
            lr=self.config.base_lr*self.config.sm_lr_multiplier,
            weight_decay=0.
        )
        sm_lr_scheduler = CosineDecayLRSched(
            sm_optimizer,
            total_steps=self.config.total_steps,
            lr_warmup_steps=self.config.lr_warmup_steps,
            lr_const_steps=self.config.lr_const_steps,
            lr_decay_multiplier=self.config.lr_decay_multiplier
        )
        return [
            {'optimizer': model_optimizer, 'lr_scheduler': {'scheduler': model_lr_scheduler, 'interval': 'step'}},
            {'optimizer': sm_optimizer, 'lr_scheduler': {'scheduler': sm_lr_scheduler, 'interval': 'step'}}
        ]