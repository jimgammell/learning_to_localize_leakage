from typing import Optional, Callable, Dict, Any, Literal, Tuple, get_args
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim
import lightning
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from .cosine_decay_lr_scheduler import CosineDecayLRSched
from ..evaluation.mttd import MinimumTracesToDisclosure
from ..evaluation.rank import Rank
from ..models.building_blocks.bits_and_bytes import ByteLogitsToBitLogits, ByteLogitsToHwLogits

LEAKAGE_MODEL = Literal[
    'bit',
    'id',
    'hw',
    'bit+id',
    'bit+hw',
    'id+hw',
    'bit+id+hw'
]
PHASE = Literal[
    'train',
    'val',
    'test'
]
PREPROCESSING = Literal[
    'standardize',
    'normalize'
]

@dataclass
class SupervisedModuleConfig:
    model_constructor: Callable[[Dict[str, Any]], nn.Module]
    model_kwargs: Dict[str, Any]
    leakage_model: LEAKAGE_MODEL
    num_labels: int
    total_steps: int
    lr_warmup_steps: Optional[int]
    lr_const_steps: Optional[int]
    base_lr: float
    lr_decay_multiplier: Optional[float]
    weight_decay: float
    label_smoothing: float
    mttd_kwargs: Dict[str, Any]
    additive_gaussian_noise: float
    preprocessing: PREPROCESSING

    def __post_init__(self):
        self.num_classes = 256
        assert self.leakage_model in get_args(LEAKAGE_MODEL)
        self.leakage_models = self.leakage_model.split('+')
        assert isinstance(self.num_labels, int) and self.num_labels > 0
        assert isinstance(self.num_classes, int) and self.num_classes > 0
        assert isinstance(self.total_steps, int) and self.total_steps > 0
        if self.lr_warmup_steps is not None:
            assert isinstance(self.lr_warmup_steps, int) and self.lr_warmup_steps >= 0
        if self.lr_const_steps is not None:
            assert isinstance(self.lr_const_steps, int) and self.lr_const_steps >= 0
        assert isinstance(self.base_lr, float) and self.base_lr > 0
        if self.lr_decay_multiplier is not None:
            assert isinstance(self.lr_decay_multiplier, float) and 0 <= self.lr_decay_multiplier <= 1
        assert isinstance(self.weight_decay, float) and self.weight_decay >= 0
        assert isinstance(self.label_smoothing, float) and 0 <= self.label_smoothing < 1
        assert isinstance(self.mttd_kwargs, dict) and all(isinstance(k, str) for k in self.mttd_kwargs)
        assert isinstance(self.additive_gaussian_noise, float) and self.additive_gaussian_noise >= 0
        assert self.preprocessing in get_args(PREPROCESSING)

class SupervisedModule(lightning.LightningModule):
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
            num_labels: int, # i.e. how many variables we are predicting
            total_steps: int,
            lr_warmup_steps: Optional[int],
            lr_const_steps: Optional[int],
            base_lr: float,
            lr_decay_multiplier: Optional[float],
            weight_decay: float,
            label_smoothing: float,
            mttd_kwargs: Dict[str, Any],
            trace_statistics: Dict[str, np.ndarray],
            additive_gaussian_noise: float,
            preprocessing: PREPROCESSING
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['trace_statistics'])
        self.config = SupervisedModuleConfig(**self.hparams)
        self.model: nn.Module = self.config.model_constructor(output_dim=self.config.num_classes, **self.config.model_kwargs)
        assert isinstance(self.model, nn.Module)
        if 'bit' in self.config.leakage_models:
            self.byte_logits_to_bit_logits = ByteLogitsToBitLogits()
        if 'hw' in self.config.leakage_models:
            self.byte_logits_to_hw_logits = ByteLogitsToHwLogits()
        self.register_buffer('trace_mean', torch.from_numpy(trace_statistics['mean']).float(), persistent=False)
        self.register_buffer('trace_std', torch.from_numpy(trace_statistics['var']).float().sqrt() + 1e-6, persistent=False)
        self.register_buffer('trace_min', torch.from_numpy(trace_statistics['min']).float(), persistent=False)
        self.register_buffer('trace_rng', torch.from_numpy(trace_statistics['max'] - trace_statistics['min']).float() + 1e-6, persistent=False)

        self.metrics = MetricCollection({
            **{
                f'{phase}/acc': MulticlassAccuracy(num_classes=256)
                for phase in get_args(PHASE)
            }, **{
                f'{phase}/acc/{idx}': MulticlassAccuracy(num_classes=256)
                for phase in get_args(PHASE) for idx in range(self.config.num_labels)
            }, **{
                f'{phase}/rank': Rank()
                for phase in get_args(PHASE)
            }, **{
                f'{phase}/rank/{idx}': Rank()
                for phase in get_args(PHASE) for idx in range(self.config.num_labels)
            }, **{
                f'{phase}/mttd': MinimumTracesToDisclosure(**self.config.mttd_kwargs)
                for phase in ['test']
            }
        })
    
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
        optimizer = optim.AdamW(
            param_groups,
            lr=self.config.base_lr
        )
        lr_scheduler = CosineDecayLRSched(
            optimizer,
            total_steps=self.config.total_steps,
            lr_warmup_steps=self.config.lr_warmup_steps,
            lr_const_steps=self.config.lr_const_steps,
            lr_decay_multiplier=self.config.lr_decay_multiplier
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'step'}}
    
    def prepare_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        trace, target, intermediate_variables = batch
        trace = trace.float()
        if self.config.preprocessing == 'standardize':
            trace = (trace - self.trace_mean) / self.trace_std
        elif self.config.preprocessing == 'normalize':
            trace = (trace - self.trace_min) / self.trace_rng
        else:
            assert False
        if self.training and self.config.additive_gaussian_noise > 0:
            trace = trace + self.config.additive_gaussian_noise*torch.randn_like(trace)
        trace = trace.to(self.dtype)
        trace = trace.unsqueeze(1)
        return trace, target, intermediate_variables

    def _step(self, batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]], phase: PHASE) -> torch.Tensor:
        trace, target, intermediate_variables = self.prepare_batch(batch)
        batch_size, output_count = target.shape
        logits: torch.Tensor = self.model(trace)
        granular_loss = nn.functional.cross_entropy(
            logits.reshape(batch_size*output_count, -1),
            target.reshape(batch_size*output_count),
            reduction='none'
        ).reshape(batch_size, output_count)
        per_output_loss = granular_loss.mean(dim=0)
        loss = per_output_loss.mean()
        
        losses = []
        if 'id' in self.config.leakage_models:
            id_loss = nn.functional.cross_entropy(
                logits.reshape(batch_size*output_count, -1),
                target.reshape(batch_size*output_count),
                label_smoothing=self.config.label_smoothing if self.training else 0.
            )
            losses.append(id_loss)
        if 'bit' in self.config.leakage_models:
            bit_logits = self.byte_logits_to_bit_logits(logits)
            target_bitstring = (target.unsqueeze(-1) >> torch.arange(8, device=target.device, dtype=torch.long)) & 1
            target_bitstring = target_bitstring.to(logits.dtype)
            if self.training and self.config.label_smoothing > 0:
                target_bitstring = (1 - self.config.label_smoothing)*target_bitstring + self.config.label_smoothing*0.5
            bit_loss = nn.functional.binary_cross_entropy_with_logits(bit_logits, target_bitstring)
            losses.append(bit_loss)
        if 'hw' in self.config.leakage_models:
            hw_logits = self.byte_logits_to_hw_logits(logits)
            target_bitstring = (target.unsqueeze(-1) >> torch.arange(8, device=target.device, dtype=torch.long)) & 1
            target_hw = target_bitstring.sum(dim=-1)
            hw_loss = nn.functional.cross_entropy(
                hw_logits.reshape(batch_size*output_count, -1),
                target_hw.reshape(batch_size*output_count),
                label_smoothing=self.config.label_smoothing if self.training else 0.
            )
            losses.append(hw_loss)
        assert 0 < len(losses) <= 3
        training_loss = sum(losses) / len(losses)

        self.metrics[f'{phase}/acc'].update(logits.reshape(batch_size*output_count, -1), target.reshape(batch_size*output_count))
        self.metrics[f'{phase}/rank'].update(logits, target)
        for idx in range(self.config.num_labels):
            self.metrics[f'{phase}/acc/{idx}'].update(logits[:, idx, :], target[:, idx])
            self.metrics[f'{phase}/rank/{idx}'].update(logits[:, idx, :], target[:, idx])
        if phase == 'test':
            self.metrics[f'{phase}/mttd'].update(logits, intermediate_variables)

        self.log(f'{phase}/loss', loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log(f'{phase}/acc', self.metrics[f'{phase}/acc'], on_epoch=True, on_step=False, prog_bar=False)
        self.log(f'{phase}/rank', self.metrics[f'{phase}/rank'], on_epoch=True, on_step=False, prog_bar=False)
        for idx in range(self.config.num_labels):
            self.log(f'{phase}/loss/{idx}', per_output_loss[idx], on_epoch=True, on_step=False)
            self.log(f'{phase}/acc/{idx}', self.metrics[f'{phase}/acc/{idx}'], on_epoch=True, on_step=False)
            self.log(f'{phase}/rank/{idx}', self.metrics[f'{phase}/rank/{idx}'], on_epoch=True, on_step=False)
        if phase == 'test':
            self.log(f'{phase}/mttd', self.metrics[f'{phase}/mttd'], on_epoch=True, on_step=False, prog_bar=True)

        return training_loss
    
    def _log_rank_stats(self, phase: str):
        per_byte_ranks = torch.tensor([
            self.metrics[f'{phase}/rank/{idx}'].compute().item()
            for idx in range(self.config.num_labels)
        ])
        self.log(f'{phase}/rank_min', per_byte_ranks.min(), prog_bar=True)
        self.log(f'{phase}/rank_med', per_byte_ranks.median(), prog_bar=True)
        self.log(f'{phase}/rank_max', per_byte_ranks.max(), prog_bar=True)

    def on_train_epoch_end(self):
        self._log_rank_stats('train')

    def on_validation_epoch_end(self):
        self._log_rank_stats('val')

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self._step(batch, phase='train')
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self._step(batch, phase='val')
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        return self._step(batch, phase='test')