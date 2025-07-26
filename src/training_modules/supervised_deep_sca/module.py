from typing import *
from scipy.special import log_softmax
import numpy as np
import torch
from torch import nn, optim
import lightning as L

import models
import utils.lr_schedulers
from ..utils import *
from utils.metrics import get_rank
from utils.aes import AES_SBOX

class Module(L.LightningModule):
    def __init__(self,
        classifier_name: str,
        classifier_kwargs: dict = {},
        lr_scheduler_name: str = None,
        lr_scheduler_kwargs: dict = {},
        lr: float = 2e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        input_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        output_dropout: float = 0.0,
        grad_clip: float = 0.0,
        noise_scale: Optional[float] = None,
        timesteps_per_trace: Optional[int] = None,
        class_count: int = 256,
        compile: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        assert self.hparams.timesteps_per_trace is not None
        if self.hparams.lr_scheduler_name is None:
            self.hparams.lr_scheduler_name = 'NoOpLRSched'
        self.hparams.classifier_kwargs.update({
            'input_dropout': self.hparams.input_dropout,
            'hidden_dropout': self.hparams.hidden_dropout,
            'output_dropout': self.hparams.output_dropout
        })
        self.classifier = models.load(
            self.hparams.classifier_name, input_shape=(1, self.hparams.timesteps_per_trace),
            output_classes=self.hparams.class_count, **self.hparams.classifier_kwargs
        )
        if self.hparams.compile:
            self.classifier.compile()
    
    def configure_optimizers(self):
        yes_weight_decay, no_weight_decay = [], []
        for name, param in self.classifier.named_parameters():
            if ('weight' in name) and not('norm' in name):
                yes_weight_decay.append(param)
            else:
                no_weight_decay.append(param)
        param_groups = [{'params': yes_weight_decay, 'weight_decay': self.hparams.weight_decay}, {'params': no_weight_decay, 'weight_decay': 0.0}]
        self.optimizer = optim.AdamW(param_groups, lr=self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2), eps=self.hparams.eps)
        lr_scheduler_constructor = (
            self.hparams.lr_scheduler_name if isinstance(self.hparams.lr_scheduler_name, optim.lr_scheduler.LRScheduler)
            else getattr(utils.lr_schedulers, self.hparams.lr_scheduler_name) if hasattr(utils.lr_schedulers, self.hparams.lr_scheduler_name)
            else getattr(optim.lr_scheduler, self.hparams.lr_scheduler_name)
        )
        if self.trainer.max_epochs != -1:
            self.total_steps = self.trainer.max_epochs*len(self.trainer.datamodule.train_dataloader())
        elif self.trainer.max_steps != -1:
            self.total_steps = self.trainer.max_steps
        else:
            assert False
        self.lr_scheduler = lr_scheduler_constructor(self.optimizer, total_steps=self.total_steps, **self.hparams.lr_scheduler_kwargs)
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}
    
    def unpack_batch(self, batch):
        trace, label = batch
        assert trace.size(0) == label.size(0)
        if self.hparams.noise_scale is not None:
            trace += self.hparams.noise_scale*torch.randn_like(trace)
        return trace, label

    def step(self, batch, train: bool = False):
        if train:
            optimizer = self.optimizers()
            lr_scheduler = self.lr_schedulers()
            optimizer.zero_grad()
        trace, label = self.unpack_batch(batch)
        batch_size = trace.size(0)
        rv = {}
        logits = self.classifier(trace)
        logits = logits.reshape(-1, logits.size(-1))
        loss = nn.functional.cross_entropy(logits, label)
        rv.update({'loss': loss.detach(), 'rank': get_rank(logits, label).mean()})
        if train:
            self.manual_backward(loss)
            rv.update({'rms_grad': get_rms_grad(self.classifier)})
            if self.hparams.grad_clip > 0.:
                nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=self.hparams.grad_clip)
            optimizer.step()
            lr_scheduler.step()
        #assert all(torch.all(torch.isfinite(param)) for param in self.classifier.parameters())
        return rv
    
    def training_step(self, batch):
        rv = self.step(batch, train=True)
        for key, val in rv.items():
            self.log(f'train_{key}', val, on_step=False, on_epoch=True, prog_bar=True if key in ('loss', 'rank') else False)
    
    def validation_step(self, batch):
        rv = self.step(batch, train=False)
        for key, val in rv.items():
            self.log(f'val_{key}', val, on_step=False, on_epoch=True, prog_bar=True if key in ('loss', 'rank') else False)

r"""
    @torch.no_grad()
    def on_validation_epoch_end(self):
        attack_dataloader = self.trainer.datamodule.test_dataloader()
        attack_dataset = attack_dataloader.dataset
        hw_of_id = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)
        class_sizes = np.bincount(hw_of_id, minlength=9)
        log_class_sizes = np.log(class_sizes.astype(np.float32))
        self.classifier.eval()
        plaintexts = torch.from_numpy(attack_dataset.plaintexts).to(torch.long).to(self.device)
        hw_logits_list = []
        for traces, _ in attack_dataloader:
            traces = traces.to(self.device)
            hw_logits = self.classifier(traces)
            hw_logits_list.append(hw_logits)
        hw_logits = torch.cat(hw_logits_list, dim=0)
        id_logits = hw_logits - torch.as_tensor(log_class_sizes).to(self.device)
        id_logits = id_logits[:, torch.as_tensor(hw_of_id).to(self.device)]
        id_logp = nn.functional.log_softmax(id_logits, dim=1)
        keys = torch.arange(256, dtype=torch.long, device=self.device)
        sbox_out = torch.as_tensor(AES_SBOX).to(self.device).to(torch.long)[plaintexts.unsqueeze(1) ^ keys]
        per_trace_key_ll = id_logp.gather(1, sbox_out)
        true_key = attack_dataset.keys[0]
        assert (attack_dataset.keys == true_key).all()
        true_key = int(true_key)
        traces_to_disc = []
        mean_rank = []
        for _ in range(10):
            perm = torch.randperm(len(attack_dataset), device=self.device)
            ll_shuffled = per_trace_key_ll[perm]
            cum_ll = ll_shuffled.cumsum(dim=0)
            true_scores = cum_ll[:, true_key].unsqueeze(1)
            rank = (cum_ll >= true_scores).sum(dim=1)
            first_correct = (rank == 1).nonzero(as_tuple=False)
            ttd = first_correct[0, 0].item() + 1 if first_correct.numel() > 0 else len(attack_dataset)
            traces_to_disc.append(ttd)
            mean_rank.append(rank.float().mean())
        mttd = torch.tensor(traces_to_disc).float().mean().item()
        mean_rank = torch.tensor(mean_rank).mean().item()
        self.log('mttd', mttd, on_epoch=True, on_step=False, prog_bar=True)
        self.log('mean_rank', mean_rank, on_epoch=True, on_step=False, prog_bar=True)
"""