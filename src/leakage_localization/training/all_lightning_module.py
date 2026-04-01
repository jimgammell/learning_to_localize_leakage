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
from .common import PHASE, LEAKAGE_MODEL, PREPROCESSING, BATCH

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

    def prepare_batch(self, batch: BATCH, augment: bool = False) -> BATCH:
        trace, target, intermediate_variables = batch
        trace = trace.to(self.device)
        target = target.to(self.device)
        trace = trace.float()
        if self.config.preprocessing == 'standardize':
            trace = (trace - self.trace_mean) / self.trace_std
        elif self.config.preprocessing == 'normalize':
            trace = (trace - self.trace_min) / self.trace_rng
        else:
            assert False
        if augment and self.config.random_roll_scale > 0:
            shift_sgn = 1 if np.random.randint(2) else -1
            shift_amt = int(abs(self.config.random_roll_scale * np.random.standard_normal()))
            if shift_amt > 0:
                trace = nn.functional.pad(trace, (shift_amt, shift_amt), mode='reflect')
                if shift_sgn > 0:
                    trace = trace[..., :-2*shift_amt]
                else:
                    trace = trace[..., 2*shift_amt:]
        if augment and self.config.random_lpf_scale > 0:
            smooth_radius = int(abs(self.config.random_lpf_scale * np.random.standard_normal()))
            if smooth_radius > 0:
                trace = nn.functional.pad(trace, (smooth_radius, smooth_radius), mode='reflect')
                trace = nn.functional.avg_pool1d(trace, kernel_size=2*smooth_radius + 1, stride=1)
        if augment and self.config.additive_gaussian_noise > 0:
            trace = trace + self.config.additive_gaussian_noise*torch.randn_like(trace)
        trace = trace.to(self.dtype)
        return trace, target, intermediate_variables
    
    # equal to -mutual_information + constant
    def compute_loss(self, logits: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
        batch_size, output_count = _target.shape
        assert output_count == self.config.num_labels, (
            f'Target has {output_count} outputs but model expects {self.config.num_labels}. '
            f'Did you forget to pass target_byte/target_variable when loading the dataset?'
        )
        if self.config.leakage_model == 'bit':
            target = (_target.unsqueeze(-1) >> torch.arange(8, device=_target.device, dtype=torch.long)) & 1
            target = target.to(logits.dtype)
            if self.training and self.config.label_smoothing > 0:
                target = (1 - self.config.label_smoothing)*target + self.config.label_smoothing*0.5
            per_output_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction='none'
            ).mean(dim=-1)
        elif self.config.leakage_model == 'hw':
            target = ((_target.unsqueeze(-1) >> torch.arange(8, device=_target.device, dtype=torch.long)) & 1).sum(dim=-1)
            per_output_loss = nn.functional.cross_entropy(
                logits.reshape(batch_size*output_count, -1),
                target.reshape(batch_size*output_count),
                label_smoothing=self.config.label_smoothing if self.training else 0.,
                reduction='none'
            ).reshape(batch_size, output_count)
        elif self.config.leakage_model == 'id':
            target = _target
            per_output_loss = nn.functional.cross_entropy(
                logits.reshape(batch_size*output_count, -1),
                target.reshape(batch_size*output_count),
                label_smoothing=self.config.label_smoothing if self.training else 0.,
                reduction='none'
            ).reshape(batch_size, output_count)
        else:
            assert False
        return per_output_loss

    def _step(self, batch: BATCH, train_theta: bool = False, train_etat: bool = False) -> torch.Tensor:
        trace, target, intermediate_variables = self.prepare_batch(batch, augment=train_theta)
        batch_size, _, feature_count = trace.shape
        if train_theta:
            self.model.requires_grad_(True)
            theta_optimizer, _ = self.optimizers()
            theta_lr_scheduler, _ = self.lr_schedulers()
            theta_optimizer.zero_grad()
        else:
            self.model.requires_grad_(False)
        if train_etat:
            self.selection_mechanism.requires_grad_(True)
            _, etat_optimizer = self.optimizers()
            _, etat_lr_scheduler = self.lr_schedulers()
            etat_optimizer.zero_grad()
        else:
            self.selection_mechanism.requires_grad_(False)
        condition_mask = self.selection_mechanism.concrete_sample(batch_size)
        masked_trace = condition_mask*trace + (1 - condition_mask)*torch.randn_like(trace)
        logits = self.model(masked_trace)
        theta_loss = self.compute_loss(logits, target)
        etat_loss = -theta_loss
        if train_theta:
            assert not train_etat
            self.manual_backward(theta_loss, inputs=self.model.parameters())
        if train_etat:
            assert not train_theta
            self.manual_backward(etat_loss, inputs=self.selection_mechanism.parameters())
        if train_theta:
            theta_optimizer.step()
            theta_lr_scheduler.step()
        if train_etat:
            etat_optimizer.step()
            etat_lr_scheduler.step()
        