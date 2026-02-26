from pathlib import Path

import numpy as np

from experiments.initialization import *
from .entrypoint import main
from leakage_localization.datasets import DATASET, PARTITION
from leakage_localization.parametric_attribution import compute_snr

def compute_snr_for_dataset(
        *,
        dataset_id: DATASET,
        partition_id: PARTITION,
        dest: Path,
        overwrite: bool
):
    if dataset_id == 'ascadv1-fixed':
        from leakage_localization.datasets.ascadv1 import ASCADv1_NumpyDataset
        dataset = ASCADv1_NumpyDataset(
            root=ASCADV1_FIXED_ROOT,
            partition=partition_id,
            variable_key=False,
            cropped_traces=False,
            binary_trace_file=True
        )
    elif dataset_id == 'ascadv1-variable':
        from leakage_localization.datasets.ascadv1 import ASCADv1_NumpyDataset
        dataset = ASCADv1_NumpyDataset(
            root=ASCADV1_VARIABLE_ROOT,
            partition=partition_id,
            variable_key=True,
            cropped_traces=False,
            binary_trace_file=True
        )
    elif dataset_id == 'ascadv2':
        from leakage_localization.datasets.ascadv2 import ASCADv2_NumpyDataset
        dataset = ASCADv2_NumpyDataset(
            root=ASCADV2_ROOT,
            partition=partition_id
        )
    elif dataset_id == 'ches-ctf-2018':
        from leakage_localization.datasets.ches_ctf_2018 import CHESCTF2018_NumpyDataset
        dataset = CHESCTF2018_NumpyDataset(
            root=CHES_CTF_2018_ROOT,
            partition=partition_id
        )
    elif dataset_id == 'dpav4d2':
        from leakage_localization.datasets.dpav4_2 import DPAv4d2_NumpyDataset
        dataset = DPAv4d2_NumpyDataset(
            root=DPAV4d2_ROOT,
            partition=partition_id
        )
    else:
        assert False
    snr_vals = compute_snr(dataset, chunk_size=4096, use_progress_bar=True, dtype=np.float32)
    for int_var_key, int_var_snr in snr_vals.items():
        dest_file = dest / f'{int_var_key}.{partition_id}.npy'
        np.save(dest_file, int_var_snr)

if __name__ == '__main__':
    main(compute_snr_for_dataset)