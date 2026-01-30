from pathlib import Path
from typing import get_args
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt

from experiments.initialization import *
from .entrypoint import main
from leakage_localization.datasets import DATASET, PARTITION

def visualize_snr_for_ascadv1(
        *,
        dataset_id: DATASET,
        partition_id: PARTITION,
        dest: Path,
        overwrite: bool
):
    assert dataset_id in ['ascadv1-fixed', 'ascadv1-variable']
    from leakage_localization.datasets.ascadv1 import repr_target, TARGET_BYTE, TARGET_VARIABLE
    for file in dest.iterdir():
        if not file.name.endswith('.npy'):
            continue
        target_var, partition, _ = file.name.split('.')
        if partition != partition_id:
            continue
        snr_vals = np.load(file)
        byte_count, timestep_count = snr_vals.shape
        fig, axes = plt.subplots(4, 4, figsize=(2*WIDTH, 2*WIDTH), sharex=True, sharey=True)
        for byte_val, ax in zip(range(byte_count), axes.flatten()):
            ax.plot(snr_vals[byte_val, :], color='blue', marker='.', markersize=2, linestyle=':', rasterized=True)
            ax.set_xlabel(r'Timestep $t$')
            ax.set_ylabel(r'SNR of $X_t$')
            ax.set_title(rf'Byte: {byte_val}')
            ax.set_yscale('log')
        fig.suptitle(f'Target: {repr_target(target_var)}')
        fig.tight_layout()
        fig.savefig(dest / f'{target_var}.{partition_id}.pdf', dpi=DPI)
        plt.close(fig)

def visualize_snr_for_dataset(
        *,
        dataset_id: DATASET,
        partition_id: PARTITION,
        dest: Path,
        overwrite: bool
):
    if dataset_id in ['ascadv1-fixed', 'ascadv1-variable']:
        visualize_snr_for_ascadv1(dataset_id=dataset_id, partition_id=partition_id, dest=dest, overwrite=overwrite)

if __name__ == '__main__':
    main(visualize_snr_for_dataset)