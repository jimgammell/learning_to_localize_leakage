import argparse
from typing import Literal, get_args
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

from experiments.initialization import *
from leakage_localization.datasets.common import PARTITION

DATASET = Literal['ascadv1-fixed']

def _annotate_spearman(ax, x, y):
    rho = spearmanr(x, y).correlation
    ax.text(0.95, 0.95, f'{rho:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=6)

def visualize_ascadv1_snr(base_dir: Path, partition: PARTITION = 'attack'):
    snr_dir = base_dir / 'snr'
    snr_vals = dict()
    for file in snr_dir.iterdir():
        var_name, fpartition, extension = file.name.split('.')
        if not fpartition == partition:
            continue
        if not extension == 'npy':
            continue
        var_snr = np.load(file)
        snr_vals[var_name] = var_snr

    model_dir = base_dir / 'best_models'
    for subdir in model_dir.iterdir():
        gradvis_path = subdir / 'gradvis.npy'
        gradvis = np.load(gradvis_path)

        fig, axes = plt.subplots(5, 16, figsize=(2*16, 2*5), sharex=True, sharey=True)
        for ax in axes.flatten():
            ax.set_xscale('log')
            ax.set_yscale('log')
        full_axes = axes[0, :]
        subbytes_axes = axes[1, :]
        rin_axes = axes[2, :]
        r_axes = axes[3, :]
        rout_axes = axes[4, :]
        plot_kwargs = dict(
            color='blue',
            marker='.',
            linestyle='none',
            markersize=2,
            alpha=0.25,
            rasterized=True
        )
        for byte_idx in range(16):
            if byte_idx >= 2:
                full_x = 0.25*(
                    snr_vals['subbytes'][byte_idx, :]
                    + 0.5*(snr_vals['r_in'][0, :] + snr_vals['p__xor__k__xor__r_in'][byte_idx, :])
                    + 0.5*(snr_vals['r'][byte_idx, :] + snr_vals['subbytes__xor__r'][byte_idx, :])
                    + 0.5*(snr_vals['r_out'][0, :] + snr_vals['subbytes__xor__r_out'][byte_idx, :])
                )
                full_axes[byte_idx].plot(full_x, gradvis[byte_idx, :], **plot_kwargs)
                _annotate_spearman(full_axes[byte_idx], full_x, gradvis[byte_idx, :])
            else:
                full_axes[byte_idx].plot(snr_vals['subbytes'][byte_idx, :], gradvis[byte_idx, :], **plot_kwargs)
                _annotate_spearman(full_axes[byte_idx], snr_vals['subbytes'][byte_idx, :], gradvis[byte_idx, :])
            subbytes_x = snr_vals['subbytes'][byte_idx, :]
            subbytes_axes[byte_idx].plot(subbytes_x, gradvis[byte_idx, :], **plot_kwargs)
            _annotate_spearman(subbytes_axes[byte_idx], subbytes_x, gradvis[byte_idx, :])
            rin_x = 0.5*(snr_vals['r_in'][0, :] + snr_vals['p__xor__k__xor__r_in'][byte_idx, :])
            rin_axes[byte_idx].plot(rin_x, gradvis[byte_idx, :], **plot_kwargs)
            _annotate_spearman(rin_axes[byte_idx], rin_x, gradvis[byte_idx, :])
            r_x = 0.5*(snr_vals['r'][byte_idx, :] + snr_vals['subbytes__xor__r'][byte_idx, :])
            r_axes[byte_idx].plot(r_x, gradvis[byte_idx, :], **plot_kwargs)
            _annotate_spearman(r_axes[byte_idx], r_x, gradvis[byte_idx, :])
            rout_x = 0.5*(snr_vals['r_out'][0, :] + snr_vals['subbytes__xor__r_out'][byte_idx, :])
            rout_axes[byte_idx].plot(rout_x, gradvis[byte_idx, :], **plot_kwargs)
            _annotate_spearman(rout_axes[byte_idx], rout_x, gradvis[byte_idx, :])
        fig.tight_layout()
        fig.savefig(subdir / 'gradvis_vs_snr.pdf')
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', choices=get_args(DATASET), required=True)
    parser.add_argument('--partition', action='store', choices=get_args(PARTITION), default='attack')
    args = parser.parse_args()

    dataset: DATASET = args.dataset
    partition: PARTITION = args.partition
    if dataset == 'ascadv1-fixed':
        output_dir = OUTPUTS_ROOT / 'ascadv1_fixed'
        visualize_ascadv1_snr(output_dir, partition)

if __name__ == '__main__':
    main()