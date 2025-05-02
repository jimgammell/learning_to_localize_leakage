from typing import Union, Dict

import numpy as np
from matplotlib import pyplot as plt

from common import *

def get_attack_point_label(attack_pt_name: str, dataset_name: str) -> str:
    if dataset_name in ['ascadv1-fixed', 'ascadv1-variable']:
        return {
            'label': r'$\operatorname{Sbox}(k_3 \oplus r_3)$',
            'subbytes': r'$\operatorname{Sbox}(k_3 \oplus r_3)$',
            'r_in': r'$r_{\mathrm{in}}$',
            'r': r'$r$',
            'r_out': r'$r_{\mathrm{out}}$',
            'plaintext__key__r_in': r'$k_3 \oplus w_3 \oplus r_{\mathrm{in}}$',
            'subbytes__r': r'$\operatorname{Sbox}(k_3 \oplus w_3) \oplus r_3$',
            'subbytes__r_out': r'$\operatorname{Sbox}(k_3 \oplus w_3) \oplus r_{\mathrm{out}}$',
            's_prev__subbytes__r_out': r'$S_{\mathrm{prev}} \oplus \operatorname{Sbox}(k_3 \oplus w_3) \oplus r_{\mathrm{out}}$',
            'security_load': r'$\operatorname{Sbox}(S_{\mathrm{prev}} \oplus r_{\mathrm{in}}) \oplus r_{\mathrm{out}}$'
        }[attack_pt_name]
    elif dataset_name == 'dpav4':
        return {
            'label': r'$\operatorname{Sbox}(k_0 \oplus w_0) \oplus m_0$'
        }[attack_pt_name]
    elif dataset_name == 'aes-hd':
        return {
            'label': r'$\operatorname{Sbox}(k^*_{11} \oplus c_{11}) \oplus c_7$'
        }[attack_pt_name]
    elif dataset_name == 'otp':
        return {
            'label': r'Dummy load?'
        }[attack_pt_name]
    elif dataset_name == 'otiait':
        return {
            'label': r'Ephemeral key nibble'
        }
    else:
        assert False

def plot_leakage_assessment(oracle_assessment: np.ndarray, leakage_assessment: np.ndarray, dest: str, dataset_name: str):
    timestep_count = leakage_assessment.shape[-1]
    leakage_assessment = leakage_assessment.reshape(-1, timestep_count)
    seed_count = leakage_assessment.shape[0]
    oracle_assessment.shape == (timestep_count,)
    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, PLOT_WIDTH))
    axes[0].set_xlabel('Timestep $t$')
    axes[0].set_ylabel('Estimated `leakiness\' of $X_t$')
    if seed_count > 1:
        mean, std = leakage_assessment.mean(axis=0), leakage_assessment.std(axis=0)
        axes[0].fill_between(np.arange(timestep_count), mean-std, mean+std, color='blue', alpha=0.25)
    axes[0].plot(np.arange(timestep_count), leakage_assessment.mean(axis=0), linestyle='none', marker='.', markersize=1, color='blue')
    axes[1].set_xlabel('Oracle SNR of $X_t$')
    axes[1].set_ylabel('Estimated `leakiness\' of $X_t$')
    sorted_indices = oracle_assessment.argsort()
    if seed_count > 1:
        axes[1].fill_between(oracle_assessment[sorted_indices], (mean-std)[sorted_indices], (mean+std)[sorted_indices], color='blue', alpha=0.25)
    axes[1].plot(oracle_assessment[sorted_indices], leakage_assessment.mean(axis=0)[sorted_indices], linestyle='none', marker='.', color='blue', **PLOT_KWARGS)
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)

def plot_oracle_assessment(oracle_assessment: Dict[str, np.ndarray], dest: str, dataset_name: str):
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_WIDTH))
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('Oracle `leakiness\' of $X_t$')
    for target, leakage_assessment in oracle_assessment.items():
        ax.plot(leakage_assessment, linestyle='-', linewidth=0.1, marker='.', markersize=1, label=get_attack_point_label(target, dataset_name), **PLOT_KWARGS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, **SAVEFIG_KWARGS)
    plt.close(fig)