from typing import Optional, Dict, Any, Tuple, Literal
from dataclasses import dataclass, field

import numpy as np
from einops import rearrange
from scipy.stats import spearmanr
import torch
from torch import nn, optim
import lightning

import utils.lr_schedulers as lr_schedulers
from utils.metrics import get_rank
from .modules import CondMutInfEstimator, SelectionMechanism

@dataclass
class _Hparams:
    timesteps_per_trace: int
    output_classes: int
    classifiers_name: str
    classifiers_kwargs: Dict[str, Any] = field(default_factory=dict)
    theta_lr_scheduler_name: Optional[str] = None
    theta_lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    etat_lr_scheduler_name: Optional[str] = None
    etat_lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    theta_lr: float = 1e-3
    etat_lr: float = 1e-3
    gamma_bar: float = 0.5
    relaxation_temp: float = 1.0
    gradient_estimator: Literal['gumbel', 'reinmax'] = 'gumbel'
    adversarial_mode: bool = True
    train_theta: bool = True
    train_etat: bool = True
    alternating_sgd: bool = False
    reference_leakage_assessment: Optional[np.ndarray] = None

class Module(lightning.LightningModule):
    hparams: _Hparams

    def __init__(self,
        timesteps_per_trace: int,
        output_classes: int,
        classifiers_name: str,
        classifiers_kwargs: Dict[str, Any] = {},
        theta_lr_scheduler_name: Optional[str] = None,
        theta_lr_scheduler_kwargs: Dict[str, Any] = {},
        etat_lr_scheduler_name: Optional[str] = None,
        etat_lr_scheduler_kwargs: Dict[str, Any] = {},
        theta_lr: float = 1e-3,
        theta_beta_1: float = 0.9,
        theta_weight_decay: float = 0.01,
        etat_lr: float = 1e-3,
        etat_beta_1: float = 0.9,
        gamma_bar: float = 0.5,
        norm_penalty_coeff: float = 0.0,
        relaxation_temp: float = 1.0,
        gradient_estimator: Literal['gumbel', 'reinmax'] = 'gumbel',
        penalty_style: Literal['budget', 'l1', 'l2', 'l1_plus_l2'] = 'budget',
        adversarial_mode: bool = True,
        omit_classifier_conditioning: bool = False,
        train_theta: bool = True,
        train_etat: bool = True,
        alternating_sgd: bool = False,
        reference_leakage_assessment: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.hparams.theta_lr_scheduler_name = self.hparams.theta_lr_scheduler_name or 'NoOpLRSched'
        self.hparams.etat_lr_scheduler_name = self.hparams.etat_lr_scheduler_name or 'NoOpLRSched'
        
        self.cmi_estimator = CondMutInfEstimator(
            self.hparams.timesteps_per_trace,
            self.hparams.output_classes,
            self.hparams.classifiers_name,
            classifiers_kwargs=self.hparams.classifiers_kwargs,
            omit_classifier_conditioning=self.hparams.omit_classifier_conditioning
        )
        self.selection_mechanism = SelectionMechanism(
            self.hparams.timesteps_per_trace,
            gamma_bar=self.hparams.gamma_bar,
            relaxation_temp=self.hparams.relaxation_temp,
            adversarial_mode=self.hparams.adversarial_mode,
            use_budget=True if self.hparams.penalty_style == 'budget' else False
        )
        self.train_etat_this_step = False
    
    def to_global_steps(self, steps: int) -> int: # Lightning considers it a 'step' every time we call optimizer.step. This computes training steps to Lightning steps one can pass to a trainer.
        out = 0
        if self.hparams.train_theta:
            out += steps
        if self.hparams.train_etat:
            out += steps
        return out
    
    def configure_optimizers(self):
        self.etat_optimizer = optim.AdamW(
            self.selection_mechanism.parameters(), lr=self.hparams.etat_lr, weight_decay=0.0, betas=(self.hparams.etat_beta_1, 0.999)
        )
        theta_yes_weight_decay, theta_no_weight_decay = [], []
        for name, param in self.cmi_estimator.named_parameters():
            if ('weight' in name) and not('norm' in name):
                theta_yes_weight_decay.append(param)
            else:
                theta_no_weight_decay.append(param)
        theta_param_groups = [{'params': theta_yes_weight_decay, 'weight_decay': self.hparams.theta_weight_decay}, {'params': theta_no_weight_decay, 'weight_decay': 0.0}]
        self.theta_optimizer = optim.AdamW(theta_param_groups, lr=self.hparams.theta_lr, betas=(self.hparams.theta_beta_1, 0.999))
        theta_lr_scheduler_constructor, etat_lr_scheduler_constructor = map(
            lambda x: (
                x if isinstance(x, (optim.lr_scheduler.LRScheduler))
                else getattr(lr_schedulers, x) if hasattr(lr_schedulers, x)
                else None
            ), (self.hparams.theta_lr_scheduler_name, self.hparams.etat_lr_scheduler_name)
        )
        assert (theta_lr_scheduler_constructor is not None) and (etat_lr_scheduler_constructor is not None)
        if self.trainer.max_epochs != -1:
            self.total_steps = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
        elif self.trainer.max_steps != -1:
            self.total_steps = self.trainer.max_steps
        else:
            assert False
        self.theta_lr_scheduler = theta_lr_scheduler_constructor(self.theta_optimizer, total_steps=self.total_steps, **self.hparams.theta_lr_scheduler_kwargs)
        self.etat_lr_scheduler = etat_lr_scheduler_constructor(self.etat_optimizer, total_steps=self.total_steps, **self.hparams.etat_lr_scheduler_kwargs)
        self.step_idx = 0
        rv = [
            {'optimizer': self.theta_optimizer, 'lr_scheduler': {'scheduler': self.theta_lr_scheduler, 'interval': 'step'}},
            {'optimizer': self.etat_optimizer, 'lr_scheduler': {'scheduler': self.etat_lr_scheduler, 'interval': 'step'}}
        ]
        return rv
    
    def step(self, batch: Tuple[torch.Tensor, torch.Tensor], train_theta: bool = True, train_etat: bool = True) -> Dict[str, Any]:
        trace, label = batch
        batch_size, *trace_dims = trace.shape
        assert (batch_size, 1, self.hparams.timesteps_per_trace) == tuple(trace.shape)
        if train_theta:
            self.cmi_estimator.requires_grad_(True)
            theta_optimizer, _ = self.optimizers()
            theta_lr_scheduler, _ = self.lr_schedulers()
            theta_optimizer.zero_grad()
        else:
            self.cmi_estimator.requires_grad_(False)
        if train_etat:
            self.selection_mechanism.requires_grad_(True)
            _, etat_optimizer = self.optimizers()
            _, etat_lr_scheduler = self.lr_schedulers()
            etat_optimizer.zero_grad()
        else:
            self.selection_mechanism.requires_grad_(False)
        rv = {}
        if self.hparams.gradient_estimator == 'gumbel':
            mask = self.selection_mechanism.concrete_sample(batch_size)
        elif self.hparams.gradient_estimator == 'reinmax':
            mask = self.selection_mechanism.reinmax_sample(batch_size)
        logits = self.cmi_estimator.get_logits(trace, mask)
        if len(logits.shape) > 2:
            batch_size, head_count, class_count = logits.shape
            assert (batch_size, head_count) == label.shape
            logits = rearrange(logits, 'b h c -> (b h) c')
            label = rearrange(label, 'b h -> (b h)')
        theta_loss = nn.functional.cross_entropy(logits, label)
        mutinf = self.cmi_estimator.get_mutinf_estimate_from_logits(logits, label)
        etat_loss = -mutinf.mean()
        if self.hparams.adversarial_mode:
            etat_loss = -1*etat_loss
        if self.hparams.penalty_style == 'mask_norm_penalty':
            etat_loss = etat_loss + self.hparams.norm_penalty_coeff*(mask.sum(dim=-1).mean() + mask.pow(2).sum(dim=-1).sqrt().mean())
        if self.hparams.penalty_style== 'gamma_norm_penalty':
            etat_loss = etat_loss + self.hparams.norm_penalty_coeff*self.selection_mechanism.get_gamma().sum(dim=-1).mean()
        if train_theta:
            self.manual_backward(theta_loss, inputs=list(self.cmi_estimator.parameters()), retain_graph=train_etat)
        if train_etat:
            self.manual_backward(etat_loss, inputs=list(self.selection_mechanism.parameters()))
        if train_theta:
            theta_optimizer.step()
            theta_lr_scheduler.step()
        if train_etat:
            etat_optimizer.step()
            etat_lr_scheduler.step()
        rv.update({
            'theta_loss': theta_loss.detach(),
            'theta_rank': get_rank(logits, label).mean(),
            'etat_loss': etat_loss.detach()
        })
        return rv
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        if self.hparams.alternating_sgd and self.hparams.train_theta and self.hparams.train_etat:
            train_etat = self.train_etat_this_step
            train_theta = not self.train_etat_this_step
            self.train_etat_this_step = not self.train_etat_this_step
        else:
            train_etat = self.hparams.train_etat
            train_theta = self.hparams.train_theta
        rv = self.step(batch, train_theta=train_theta, train_etat=train_etat)
        if self.hparams.reference_leakage_assessment is not None:
            log_gamma = self.selection_mechanism.get_log_gamma()
            correlation = spearmanr(
                self.hparams.reference_leakage_assessment.reshape(-1), log_gamma.reshape(-1).detach().cpu().numpy()
            ).statistic
            self.log('oracle_snr_corr', correlation, on_step=True, on_epoch=True, prog_bar=True)
        for key, val in rv.items():
            self.log(f'train_{key}', val, on_step=True, on_epoch=True, prog_bar=True if key in ('theta_loss', 'theta_rank') else False)
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        rv = self.step(batch, train_theta=False, train_etat=False)
        for key, val in rv.items():
            self.log(f'val_{key}', val, on_step=False, on_epoch=True, prog_bar=True if key in ('theta_loss', 'theta_rank') else False)