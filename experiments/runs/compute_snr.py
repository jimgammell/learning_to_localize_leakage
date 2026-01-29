import argparse
from typing import get_args, Optional
from pathlib import Path

import numpy as np

from experiments.initialization import *
from leakage_localization.datasets import DATASET, PARTITION
from leakage_localization.parametric_attribution import compute_snr

def compute_snr_for_dataset(
        *,
        dataset_id: DATASET,
        partition_id: PARTITION,
        dest: Path
):
    if dataset_id == 'ascadv1-fixed':
        from leakage_localization.datasets.ascadv1 import ASCADv1_NumpyDataset, TARGET_VARIABLE
        dataset = ASCADv1_NumpyDataset(
            root=ASCADV1_FIXED_ROOT,
            partition=partition_id,
            variable_key=False,
            cropped_traces=False,
            binary_trace_file=True
        )
    elif dataset_id == 'ascadv1-variable':
        from leakage_localization.datasets.ascadv1 import ASCADv1_NumpyDataset, TARGET_VARIABLE
        dataset = ASCADv1_NumpyDataset(
            root=ASCADV1_FIXED_ROOT,
            partition=partition_id,
            variable_key=True,
            cropped_traces=False,
            binary_trace_file=True
        )
    else:
        assert False
    snr_vals = compute_snr(dataset, chunk_size=4096, use_progress_bar=True, dtype=np.float32)
    for int_var_key, int_var_snr in snr_vals.items():
        dest_file = dest / f'{int_var_key}.{partition_id}.npy'
        np.save(dest_file, int_var_snr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=get_args(DATASET))
    parser.add_argument('--partition', required=True, choices=get_args(PARTITION))
    parser.add_argument('--dest', type=Path, default=None)
    append_directory_clargs(parser)
    args = parser.parse_args()

    dataset_id: DATASET = args.dataset
    partition_id: PARTITION = args.partition
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = OUTPUTS_ROOT / f'{dataset_id}_snr'
    dest.mkdir(exist_ok=True, parents=True)

    assert dataset_id in get_args(DATASET)
    assert dest.exists()

    compute_snr_for_dataset(dataset_id=dataset_id, partition_id=partition_id, dest=dest)

if __name__ == '__main__':
    main()