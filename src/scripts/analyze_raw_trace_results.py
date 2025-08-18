import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle

from matplotlib import pyplot as plt
import numpy as np
import h5py
import torch
from torch import nn
from torch.utils.data import Dataset
from scipy.stats import spearmanr

from common import *
from utils.baseline_assessments import FirstOrderStatistics
from training_modules.supervised_deep_sca import SupervisedModule
from training_modules.adversarial_leakage_localization import ALLModule
from utils.baseline_assessments.neural_net_attribution import NeuralNetAttribution
from utils.aes import AES_SBOX
from datasets.data_module import DataModule

output_path = os.path.join(OUTPUT_DIR, r'raw_trace_visualization')

ascad_path = os.path.join(RESOURCE_DIR, r'ascadv1-fixed/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5')
with h5py.File(ascad_path, 'r') as database:
    TRACES = np.array(database['traces'], dtype=np.float32)
    KEYS = np.array(database['metadata']['key'], dtype=np.uint8)
    PLAINTEXTS = np.array(database['metadata']['plaintext'], dtype=np.uint8)
    MASKS = np.array(database['metadata']['masks'], dtype=np.uint8)
    MASKS = np.concatenate([np.zeros((len(MASKS), 2), dtype=np.uint8), MASKS], axis=1)

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
            'traces': torch.from_numpy(TRACES), #.share_memory_(),
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
        trace = self.dataset['traces'][idx, :].numpy()
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
    add_channel_dim=True,
    target_byte=2#np.arange(16)
)
attack_dataset = ASCAD(
    ascad_path,
    phase='attack',
    add_channel_dim=True
)
snr_attack_dataset = ASCAD(
    ascad_path,
    phase='attack',
    add_channel_dim=False,
    target_byte=2#np.arange(16)
)
mean_trace = TRACES.mean(axis=0)
var_trace = TRACES.var(axis=0)
datamodule = DataModule(
    profiling_dataset, attack_dataset, val_prop=0.1,
    data_mean=mean_trace, data_var=var_trace,
    train_batch_size=128, eval_batch_size=128,
    num_workers=1
)

with open(r'/home/jgammell/Desktop/learning_to_localize_leakage/gautschi_outputs/ascadv1f_raw_trials/snr.pickle', 'rb') as f:
    snr_vals = pickle.load(f)
snr = np.stack(list(snr_vals.values())).mean(axis=0)
with open(r'/home/jgammell/Desktop/learning_to_localize_leakage/gautschi_outputs/ascadv1f_raw_trials/profiling_snr.pickle', 'rb') as f:
    profiling_snr_vals = pickle.load(f)
profiling_snr = np.stack(list(profiling_snr_vals.values())).mean(axis=0)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for name, val in snr_vals.items():
    ax.plot(val, label=name, marker='.', markersize=1, linestyle='--', linewidth=0.2)
ax.set_xlabel(f'Time $t$')
ax.set_ylabel(f'Ground truth leakiness of $X_t$')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(output_path, 'gt_snr.png'))
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for name, val in profiling_snr_vals.items():
    ax.plot(val, label=name, marker='.', markersize=1, linestyle='--', linewidth=0.2)
ax.set_xlabel(f'Time $t$')
ax.set_ylabel(f'Ground truth leakiness of $X_t$')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(output_path, 'gt_prof_snr.png'))
plt.close(fig)

random_corrs = np.stack([spearmanr(snr, np.random.rand(*snr.shape)).statistic for _ in range(5)])
print(f'Random agreement: {random_corrs.mean()} +/- {random_corrs.std()}')
print(f'Agreement between profiling + attack SNR: {spearmanr(profiling_snr, snr).statistic}')

## Supervised interpretation results
for trial_idx in range(8):
    checkpoint_path = f'/home/jgammell/Desktop/learning_to_localize_leakage/gautschi_outputs/ascadv1f_raw_trials/supervised/trial_{trial_idx}/lightning_logs/lightning_logs/version_0/checkpoints/epoch=227-step=20000.ckpt'
    class ClassifierWrapper(nn.Module):
        def __init__(self, base_classifier: nn.Module, target_byte: int):
            super().__init__()
            self.base_classifier = base_classifier
            self.input_shape = self.base_classifier.input_shape
            self.target_byte = target_byte
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base_classifier(x)[:, self.target_byte, :]
    module = SupervisedModule.load_from_checkpoint(checkpoint_path, map_location='cuda')
    classifier = ClassifierWrapper(module.classifier, 2)
    attributor = NeuralNetAttribution(datamodule.profiling_dataloader(), classifier, seed=0, device='cuda')
    if not os.path.exists(os.path.join(output_path, f'gradvis_{trial_idx}.npy')):
        gradvis = attributor.compute_gradvis()
        np.save(os.path.join(output_path, f'gradvis_{trial_idx}.npy'), gradvis)
    gradvis = np.load(os.path.join(output_path, f'gradvis_{trial_idx}.npy'))
    print(f'GradVis oracle agreement (trial {trial_idx}): {spearmanr(gradvis, snr).statistic}')
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(gradvis, color='blue', marker='.', markersize=1, linewidth=0.2, linestyle='--')
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'Estimated leakiness of $X_t$')
    ax.set_title('GradVis')
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f'gradvis_{trial_idx}.png'))
    plt.close(fig)

    fig, axes = plt.subplots(1, len(snr_vals), figsize=(4*len(snr_vals), 4), sharey=True)
    for ax, (name, val) in zip(axes, snr_vals.items()):
        ax.plot(val, gradvis, marker='.', markersize=1, linestyle='none')
        ax.set_title(name)
        ax.set_xlabel('SNR')
        ax.set_ylabel('GradVis')
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f'gradvis_snr_{trial_idx}.png'))

## ALL results

TRIAL_COUNT = 11
fig, axes = plt.subplots(TRIAL_COUNT, 3, figsize=(3*4, TRIAL_COUNT*4))
for trial_idx in range(TRIAL_COUNT):
    checkpoint_path = f'/home/jgammell/Desktop/learning_to_localize_leakage/gautschi_outputs/ascadv1f_raw_trials/all_train_backup/trial_idx={trial_idx}/lightning_logs/version_0/checkpoints/epoch=113-step=20000.ckpt'
    os.makedirs(output_path, exist_ok=True)

    all_module = ALLModule.load_from_checkpoint(checkpoint_path, strict=False, map_location='cpu')
    log_gamma = all_module.selection_mechanism.get_log_gamma().detach().numpy().squeeze()
    print(f'Oracle agreement: {spearmanr(log_gamma, snr).statistic}')
    r"""ax = axes[trial_idx, 0]
    ax.plot(snr, color='blue', linestyle=':', marker='.', markersize=1, linewidth=0.2)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'SNR: Estimated leakiness of $X_t$')
    ax.set_yscale('log')
    ax = axes[trial_idx, 1]
    ax.plot(np.exp(log_gamma), color='blue', linestyle=':', marker='.', markersize=1, linewidth=0.2)
    ax.set_xlabel(r'Time $t$')
    ax.set_ylabel(r'ALL: Estimated leakiness of $X_t$')
    ax = axes[trial_idx, 2]
    ax.plot(snr, np.exp(log_gamma), color='blue', linestyle='none', marker='.', markersize=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'Ground truth leakiness of $X_t$')
    ax.set_ylabel(r'Estimated leakiness of $X_t$')"""

    fig, axes = plt.subplots(1, len(snr_vals), figsize=(4*len(snr_vals), 4), sharey=True)
    for ax, (name, val) in zip(axes, snr_vals.items()):
        ax.plot(val, np.exp(log_gamma), marker='.', markersize=1, linestyle='none')
        ax.set_title(name)
        ax.set_xlabel('SNR')
        ax.set_ylabel('ALL')
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f'all_snr_{trial_idx}.png'))
fig.tight_layout()
fig.savefig(os.path.join(output_path, f'qualitative_agreement.png'))
plt.close(fig)