from typing import get_args, Callable
from pathlib import Path
import argparse
import logging

import numpy as np
from matplotlib import pyplot as plt
from leakage_localization.datasets import DATASET, PARTITION
from leakage_localization.parametric.snr import compute_snr

from init_things import *
from utils.load_dataset import load_numpy_dataset

def run_compute_snr(
        dataset_id: DATASET,
        partition_id: PARTITION,
        dest: Path,
        chunk_size: int = 1024,
        use_progress_bar: bool = True,
        dtype: np.dtype = np.float32
):
    assert dest.exists()
    dataset = load_numpy_dataset(dataset_id, partition_id)
    snr_vals = compute_snr(dataset, chunk_size=chunk_size, use_progress_bar=use_progress_bar, dtype=dtype)
    for int_var_key, int_var_snr in snr_vals.items():
        dest_file = dest / f'{int_var_key}.{partition_id}.npy'
        if dest_file.exists():
            logging.warning(f'File {dest_file} already exists, so the new SNR will not be saved.')
            continue
        np.save(dest_file, int_var_snr)

def run_visualize_snr(
        dest: Path,
        dataset_id: DATASET,
        partition_id: PARTITION
):
    assert dest.exists()
    snr_vals = dict()
    for file in dest.iterdir():
        if not file.name.endswith('.npy'):
            continue
        var_name, _partition_id, _ = file.name.split('.')
        if not _partition_id == partition_id:
            continue
        snr_val = np.load(file)
        snr_vals[var_name] = snr_val
    var_count = len(snr_vals)
    byte_count = max(snr_val.shape[0] for snr_val in snr_vals.values())
    fig, axes = plt.subplots(var_count, byte_count, sharex=True, sharey='row', figsize=(2*byte_count, 2*var_count))
    for row_idx, var_name in enumerate(snr_vals):
        snr_val = snr_vals[var_name]
        for col_idx in range(byte_count):
            ax = axes[row_idx, col_idx]
            if col_idx < snr_val.shape[0]:
                ax.plot(snr_val[col_idx], color='blue', marker='.', markersize=2, linestyle=':', rasterized=True)
                ax.set_yscale('log')
            else:
                ax.axis('off')
    for row_idx in range(var_count):
        axes[row_idx, 0].set_ylabel(f'SNR of {list(snr_vals.keys())[row_idx]}')
    for col_idx in range(byte_count):
        axes[0, col_idx].set_title(f'Byte {col_idx}')
        axes[-1, col_idx].set_xlabel('Time')
    fig.suptitle(f'SNR of dataset {dataset_id}[{partition_id}]')
    fig.tight_layout()
    fig.savefig(dest / f'{dataset_id}.{partition_id}.pdf', dpi=DPI)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', required=True, choices=get_args(DATASET),
        help='Dataset for which to compute the SNR.'
    )
    parser.add_argument(
        '--partition', required=True, choices=get_args(PARTITION),
        help='Partition of the dataset for which to compute the SNR.'
    )
    parser.add_argument(
        '--dest', type=Path, default=None,
        help=f'Directory in which to save results (defaults to `{OUTPUTS_ROOT/"<DATASET>"}`)'
    )
    parser.add_argument(
        '--compute', default=False, action='store_true',
        help='Compute the SNR on the specified dataset.'
    )
    parser.add_argument(
        '--plot', default=False, action='store_true',
        help='Plot the SNR for the specified dataset.'
    )
    append_directory_clargs(parser)
    args = parser.parse_args()

    dataset_id: DATASET = args.dataset
    partition_id: PARTITION = args.partition
    dest: Optional[Path] = args.dest
    compute: bool = args.compute
    plot: bool = args.plot

    assert dataset_id in get_args(DATASET)
    assert partition_id in get_args(PARTITION)
    if dest is None:
        dest = OUTPUTS_ROOT/f'{dataset_id}'.replace('-', '_') / 'snr'
    else:
        assert isinstance(dest, Path)
    dest.mkdir(exist_ok=True, parents=True)
    assert isinstance(compute, bool)
    assert isinstance(plot, bool)

    if compute:
        logging.info(f'Computing SNR of dataset {dataset_id}[{partition_id}]')
        run_compute_snr(dataset_id, partition_id, dest)
    if plot:
        logging.info(f'Visualizing SNR of dataset {dataset_id}[{partition_id}]')
        run_visualize_snr(dest, dataset_id, partition_id)

if __name__ == '__main__':
    main()