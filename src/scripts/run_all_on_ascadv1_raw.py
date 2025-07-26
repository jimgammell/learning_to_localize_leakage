#region Imports

import sys
sys.path.insert(0, r'/home/jgammell/Desktop/learning_to_localize_leakage/src')
import os
from random import randint
from typing import Optional, Sequence, Callable, Union
import json
import pickle
import argparse

from matplotlib import pyplot as plt
import numpy as np
from scipy.special import log_softmax
import h5py
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
import models

#endregion
#region Global variable definitions

TRIAL_DIR = os.path.join(OUTPUT_DIR, 'ascadv1f_raw_trials')
os.makedirs(TRIAL_DIR, exist_ok=True)
FIG_DIR = os.path.join(TRIAL_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
STEPS = 5000
parser = argparse.ArgumentParser()
parser.add_argument('--subdir-prefix', default=None, action='store')
clargs = parser.parse_args()
SUBDIR_PREFIX = '' if clargs.subdir_prefix is None else clargs.subdir_prefix

#endregion
#region Dataset initialization

print('Loading dataset into RAM...')

ascad_path = r'/home/jgammell/Desktop/learning_to_localize_leakage/resources/ascadv1-fixed/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5'
with h5py.File(ascad_path, 'r') as database:
    TRACES = np.array(database['traces'], dtype=np.float32)
    KEYS = np.array(database['metadata']['key'], dtype=np.uint8)
    PLAINTEXTS = np.array(database['metadata']['plaintext'], dtype=np.uint8)
    MASKS = np.array(database['metadata']['masks'], dtype=np.uint8)
    MASKS = np.concatenate([np.zeros((len(MASKS), 2), dtype=np.uint8), MASKS], axis=1)

print('\tDone.')

class ASCAD(Dataset):
    def __init__(
            self,
            dataset_path: str,
            phase: Literal['profile', 'attack'] = 'profile',
            transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            target_transform: Callable[[Union[int, torch.Tensor]], Union[int, torch.Tensor]] = None,
            add_channel_dim: bool = True,
            target_byte: int = 2
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset = {
            'traces': TRACES,
            'metadata': {'key': KEYS, 'plaintext': PLAINTEXTS, 'masks': MASKS}
        }
        self.transform = transform
        self.target_transform = target_transform
        self.add_channel_dim = add_channel_dim
        self.target_byte = target_byte
        if phase == 'profile':
            self.data_indices = np.arange(0, 50000, dtype=np.int64)
        elif phase == 'attack':
            self.data_indices = np.arange(50000, 60000, dtype=np.int64)
        else:
            assert False
        self.return_metadata = False
        self.timesteps_per_trace = 100000
        self.class_count = 256
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        idx = self.data_indices[idx]
        trace = np.array(self.dataset['traces'][idx, :], dtype=np.float32)
        key = self.dataset['metadata']['key'][idx, self.target_byte]
        plaintext = self.dataset['metadata']['plaintext'][idx, self.target_byte]
        r = self.dataset['metadata']['masks'][idx, self.target_byte]
        r_in = self.dataset['metadata']['masks'][idx, -2]
        r_out = self.dataset['metadata']['masks'][idx, -1]
        target = AES_SBOX[key ^ plaintext].astype(np.int64)
        if self.add_channel_dim:
            trace = trace.reshape(1, -1)
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_metadata:
            metadata = dict()
            metadata['label'] = target
            metadata['key'] = key
            metadata['plaintext'] = plaintext
            metadata['subbytes'] = AES_SBOX[key ^ plaintext]
            metadata['r'] = r
            metadata['r_in'] = r_in
            metadata['r_out'] = r_out
            metadata['subbytes__r'] = AES_SBOX[key ^ plaintext] ^ r
            metadata['subbytes__r_out'] = AES_SBOX[key ^ plaintext] ^ r_out
            metadata['key__plaintext__r_in'] = key ^ plaintext ^ r_in
            return trace, target, metadata
        else:
            return trace, target
    
    def __len__(self) -> int:
        return len(self.data_indices)

profiling_dataset = ASCAD(
    ascad_path,
    phase='profile',
    add_channel_dim=True
)
attack_dataset = ASCAD(
    ascad_path,
    phase='attack',
    add_channel_dim=True
)
snr_attack_dataset = ASCAD(
    ascad_path,
    phase='attack',
    add_channel_dim=False
)
mean_trace = TRACES.mean(axis=0)
var_trace = TRACES.var(axis=0)
datamodule = DataModule(profiling_dataset, attack_dataset, val_prop=0.1, data_mean=mean_trace, data_var=var_trace, train_batch_size=512, eval_batch_size=512, num_workers=1)

#endregion
#region  Computing ground truth signal to noise ratio

if not os.path.exists(os.path.join(TRIAL_DIR, 'snr.pickle')):
    stats_calculator = FirstOrderStatistics(
        snr_attack_dataset, chunk_size=1, bytes=2,
        targets=['subbytes', 'r', 'r_in', 'r_out', 'subbytes__r', 'subbytes__r_out', 'key__plaintext__r_in']
    )
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
    with open(os.path.join(TRIAL_DIR, 'snr.pickle'), 'wb') as f:
        pickle.dump(snr_vals, f)
with open(os.path.join(TRIAL_DIR, 'snr.pickle'), 'rb') as f:
    snr_vals = pickle.load(f)

#endregion
#region Training a supervised model on the dataset

SUPERVISED_TRAINING_DIR = os.path.join(TRIAL_DIR, SUBDIR_PREFIX + 'supervised')
os.makedirs(SUPERVISED_TRAINING_DIR, exist_ok=True)

# previously found to work well
transformer_kwargs = dict(
    patch_size=1000,
    layer_count=3,
    attn_head_count=12,
    embedding_dim=768,
    dropout=0.0,
    input_dropout=0.0,
    dropword=0.0,
    head_dropout=0.0,
    input_noise_std=0.0,
    input_jitter=0,
    bias=False,
    norm_eps=1.e-5,
    rescale_norm_outputs=False,
    output_head_count=1,
    output_head_classes=256,
    shared_head=True,
    head_type='simple-shared'
)
trial_idx = 0
while True:
    lr = float(10**np.random.uniform(-5, -2))
    final_prop = float(10**np.random.uniform(-2, 0))
    training_kwargs = dict(
        lr=lr,
        lr_scheduler_name='CosineDecayLRSched',
        lr_scheduler_kwargs=dict(warmup_prop=0., const_prop=0., final_prop=final_prop),
        beta_1=0.9,
        beta_2=0.99,
        eps=1.e-8,
        weight_decay=1.e-2,
        grad_clip=1.0,
        timesteps_per_trace=100000,
        class_count=256,
        compile=True
    )
    supervised_module = SupervisedModule(
        classifier_name='transformer',
        classifier_kwargs=transformer_kwargs,
        **training_kwargs
    )
    print(supervised_module.classifier)
    trainer = lightning.Trainer(
        max_steps=STEPS,
        val_check_interval=1.,
        check_val_every_n_epoch=1,
        default_root_dir=os.path.join(SUPERVISED_TRAINING_DIR, f'trial_{trial_idx}', 'lightning_logs')
    )
    trainer.fit(supervised_module, datamodule=datamodule)
    trial_idx += 1

#endregion
#region ALL training
r"""
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
        timesteps_per_trace=100000,
        output_classes=256,
        classifiers_name='transformer',
        classifiers_kwargs=transformer_kwargs,
        theta_weight_decay=1e-2,
        lr_scheduler_name='CosineDecayLRSched',
        lr_scheduler_kwargs=dict(warmup_prop=500./STEPS, const_prop=0., final_prop=0.1),
        beta_1=0.9,
        beta_2=0.99,
        eps=1.e-8,
        weight_decay=1.e-2,
        grad_clip=1.0,
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
"""
#endregion