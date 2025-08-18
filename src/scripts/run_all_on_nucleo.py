#region Imports

import sys
sys.path.insert(0, r'/home/jgammell/Desktop/learning_to_localize_leakage/src')
import os
from random import randint
from typing import Optional, Sequence
import json

from matplotlib import pyplot as plt
import numpy as np
from scipy.special import log_softmax
from scipy.stats import spearmanr
import estraces
import torch
from torch.utils.data import TensorDataset, random_split, Dataset
import lightning

from common import *
from utils.aes import AES_SBOX, AES_INVERSE_SBOX
from datasets.data_module import DataModule
from training_modules.supervised_deep_sca import SupervisedModule
from trials.utils import get_training_curves
from utils.metrics.rank import get_rank
from utils.baseline_assessments.first_order_statistics import FirstOrderStatistics
from training_modules.adversarial_leakage_localization import ALLModule

#endregion
#region Global variable definitions

TRIAL_DIR = os.path.join(OUTPUT_DIR, 'nucleo_trials')
os.makedirs(TRIAL_DIR, exist_ok=True)
FIG_DIR = os.path.join(TRIAL_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
STEPS = 30000

#endregion
#region Dataset initialization

class NucleoDataset(Dataset):
    def __init__(self, dataset_path: str, desync_level: int = 10, indices: Optional[Sequence[int]] = None, transform=None, target_transform=None, hw_targets: bool = True, add_channel_dim: bool = True):
        self.desync_level = desync_level
        ets_dataset = estraces.read_ths_from_ets_file(dataset_path)
        self.traces = np.array(ets_dataset.samples)
        self.keys = np.array(ets_dataset.key)[:, 0]
        self.plaintexts = np.array(ets_dataset.plaintext)[:, 0]
        self.masks = np.array(ets_dataset.mask)
        self.targets = AES_SBOX[self.keys ^ self.plaintexts]
        if hw_targets:
            self.targets = np.unpackbits(self.targets.reshape(len(self.targets), 1), axis=1).sum(axis=1)
            self.class_count = 9
        else:
            self.class_count = 256
        if indices is not None:
            self.traces = self.traces[indices, :]
            self.targets = self.targets[indices]
            self.keys = self.keys[indices]
            self.plaintexts = self.plaintexts[indices]
            self.masks = self.masks[indices]
        self.transform = transform
        self.target_transform = target_transform
        self.timesteps_per_trace = self.traces.shape[-1]
        self.return_metadata = False
        self.add_channel_dim = add_channel_dim
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        trace = self.traces[idx]
        target = self.targets[idx]
        if self.desync_level > 0:
            shift = randint(0, self.desync_level)
            trace = np.roll(trace, shift, axis=-1)
        if self.add_channel_dim:
            trace = trace.reshape(1, -1)
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_metadata:
            metadata = dict()
            metadata['label'] = target
            metadata['subbytes'] = target
            metadata['subbytes__r0'] = target ^ self.masks[idx, 0]
            metadata['subbytes__r1'] = target ^ self.masks[idx, 1]
            metadata['r0__k__pt'] = self.masks[idx, 0] ^ self.keys[idx] ^ self.plaintexts[idx]
            metadata['r0'] = self.masks[idx, 0]
            metadata['r1'] = self.masks[idx, 1]
            return trace, target, metadata
        else:
            return trace, target
    
    def __len__(self) -> int:
        return len(self.traces)

dataset_path = r'/home/jgammell/Desktop/learning_to_localize_leakage/resources/nucleo/Nucleo_AES_masked_non_shuffled.ets'
indices = np.random.permutation(100000)
profiling_dataset = NucleoDataset(dataset_path, desync_level=5, indices=indices[:-20000])
attack_dataset = NucleoDataset(dataset_path, desync_level=0, indices=indices[-20000:])
snr_attack_dataset = NucleoDataset(dataset_path, desync_level=0, indices=indices[-20000:], hw_targets=False, add_channel_dim=False)
snr_profiling_dataset = NucleoDataset(dataset_path, desync_level=0, indices=indices[:-20000], hw_targets=False, add_channel_dim=False)
datamodule = DataModule(
    profiling_dataset, attack_dataset,
    data_mean=profiling_dataset.traces.mean(axis=0),
    data_var=profiling_dataset.traces.var(axis=0),
    train_batch_size=512
)
datamodule.val_dataset.dataset.desync_level = 0

#endregion
#region Computing ground truth signal to noise ratio

stats_calculator = FirstOrderStatistics(snr_attack_dataset, targets=['subbytes', 'r0', 'subbytes__r0', 'r1', 'subbytes__r1', 'r0__k__pt'])
snr_vals = stats_calculator.snr_vals
fig, axes = plt.subplots(1, len(snr_vals), figsize=(4*len(snr_vals), 4))
for ax, (var_name, var_snr) in zip(axes, snr_vals.items()):
    ax.plot(var_snr, color='blue', linewidth=0.5, markersize=1, marker='.')
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Estimated leakiness of $X_t$')
    ax.set_title(f'Target variable: {var_name}')
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'gt_snr.png'))
plt.close(fig)

prof_stats_calculator = FirstOrderStatistics(snr_profiling_dataset, targets=['subbytes', 'r0', 'subbytes__r0', 'r1', 'subbytes__r1', 'r0__k__pt'])
prof_snr_vals = prof_stats_calculator.snr_vals
prof_gt_snr = np.stack([prof_snr_vals['r0'], prof_snr_vals['subbytes__r0'], prof_snr_vals['r1'], prof_snr_vals['subbytes__r1']]).mean(axis=0)

gt_snr = np.stack([snr_vals['r0'], snr_vals['subbytes__r0'], snr_vals['r1'], snr_vals['subbytes__r1']]).mean(axis=0)

rand_agreements = np.stack([
    spearmanr(gt_snr, np.random.randn(*gt_snr.shape)).statistic for _ in range(5)
])
print(f'Random oracle agreement: {rand_agreements.mean()} +/- {rand_agreements.std()}')
print(f'Agreement between profiling + attack oracle: {spearmanr(prof_gt_snr, gt_snr).statistic}')

for trial_idx in range(50):
    trial_path = os.path.join(r'/home/jgammell/Desktop/learning_to_localize_leakage/outputs/nucleo_trials/all_pretrain', f'trial_idx={trial_idx}')
    checkpoint_path = os.path.join(trial_path, 'lightning_logs', 'version_1', 'checkpoints', 'epoch=119-step=30000.ckpt')
    module = ALLModule.load_from_checkpoint(checkpoint_path)
    assessment = module.selection_mechanism.get_log_gamma().detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(assessment.reshape(-1), color='blue', linestyle='-', linewidth=0.2, markersize=1)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Estimated leakiness of $X_t$')
    fig.tight_layout()
    fig.savefig(os.path.join(trial_path, 'assessment.png'))
    plt.close(fig)
assert False

#endregion
#region Supervised training

SUPERVISED_TRAINING_DIR = os.path.join(TRIAL_DIR, 'supervised')
os.makedirs(SUPERVISED_TRAINING_DIR, exist_ok=True)

learning_rates = [8e-4, 9e-4, 1e-3, 2e-3, 3e-3] #np.logspace(-4, -2, 10)
for lr in learning_rates:
    trial_dir = os.path.join(SUPERVISED_TRAINING_DIR, f'lr={lr}')
    if os.path.exists(trial_dir):
        print(f'Supervised trial exists for lr={lr}. Skipping.')
        continue
    print(f'Starting supervised trial with lr={lr}')
    supervised_module = SupervisedModule(
        classifier_name='perin-cnn',
        lr_scheduler_name='CosineDecayLRSched',
        lr_scheduler_kwargs=dict(),
        lr=lr,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        timesteps_per_trace=1400,
        class_count=9
    )
    trainer = lightning.Trainer(
        max_steps=STEPS,
        val_check_interval=1.,
        check_val_every_n_epoch=1,
        default_root_dir=trial_dir
    )
    trainer.fit(supervised_module, datamodule=datamodule)

#endregion
#region ALL training

ALL_PRETRAINING_DIR = os.path.join(TRIAL_DIR, 'all_pretrain')
os.makedirs(ALL_PRETRAINING_DIR, exist_ok=True)

trial_count = 250
for trial_idx in range(trial_count):
    trial_dir = os.path.join(ALL_PRETRAINING_DIR, f'trial_idx={trial_idx}')
    if os.path.exists(trial_dir):
        print(f'ALL pretraining trial exists for lr={lr}. Skipping.')
        continue
    print(f'Starting ALL pretraining trial with lr={lr}')
    hparams = dict(
        gamma_bar=float(np.random.uniform(0.05, 0.95)),
        theta_lr=10**np.random.uniform(-4, -2)
    )
    hparams['etat_lr'] = float(hparams['theta_lr']*10**np.random.uniform(0, 3))
    print(f'\tHparams: {hparams}')
    all_module = ALLModule(
        timesteps_per_trace=1400,
        output_classes=9,
        classifiers_name='perin-cnn',
        theta_weight_decay=0.0,
        train_theta=True,
        train_etat=False,
        reference_leakage_assessment=gt_snr,
        **hparams
    )
    trainer = lightning.Trainer(
        max_steps=STEPS//2,
        val_check_interval=1.,
        check_val_every_n_epoch=1,
        default_root_dir=trial_dir
    )
    profiling_dataset.desync_level = 5
    trainer.fit(all_module, datamodule=datamodule)
    all_module.hparams.train_etat = True
    trainer = lightning.Trainer(
        max_steps=STEPS,
        val_check_interval=1.,
        check_val_every_n_epoch=1,
        default_root_dir=trial_dir
    )
    profiling_dataset.desync_level = 0
    trainer.fit(all_module, datamodule=datamodule)
    with open(os.path.join(trial_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f)

#endregion