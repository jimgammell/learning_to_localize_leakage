from typing import List, Tuple

from torch.utils.data import DataLoader

from init_things import *
from leakage_localization.datasets import DATASET, PARTITION, Base_NumpyDataset, Base_TorchDataset

def load_numpy_dataset(dataset_id: DATASET, partition_id: PARTITION) -> Base_NumpyDataset:
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
    return dataset

def load_torch_dataset(dataset_id: DATASET, partition_id: PARTITION) -> Base_TorchDataset:
    if dataset_id == 'ascadv1-fixed':
        from leakage_localization.datasets.ascadv1 import ASCADv1_TorchDataset
        dataset = ASCADv1_TorchDataset(
            root=ASCADV1_FIXED_ROOT,
            partition=partition_id,
            variable_key=False,
            cropped_traces=False,
            binary_trace_file=True
        )
    elif dataset_id == 'ascadv1-variable':
        from leakage_localization.datasets.ascadv1 import ASCADv1_TorchDataset
        dataset = ASCADv1_TorchDataset(
            root=ASCADV1_VARIABLE_ROOT,
            partition=partition_id,
            variable_key=True,
            cropped_traces=False,
            binary_trace_file=True
        )
    elif dataset_id == 'ascadv2':
        from leakage_localization.datasets.ascadv2 import ASCADv2_TorchDataset
        dataset = ASCADv2_TorchDataset(
            root=ASCADV2_ROOT,
            partition=partition_id
        )
    elif dataset_id == 'ches-ctf-2018':
        from leakage_localization.datasets.ches_ctf_2018 import CHESCTF2018_TorchDataset
        dataset = CHESCTF2018_TorchDataset(
            root=CHES_CTF_2018_ROOT,
            partition=partition_id
        )
    elif dataset_id == 'dpav4d2':
        from leakage_localization.datasets.dpav4_2 import DPAv4d2_TorchDataset
        dataset = DPAv4d2_TorchDataset(
            root=DPAV4d2_ROOT,
            partition=partition_id
        )
    else:
        assert False
    return dataset

def construct_loaders(
        train_sets: List[Base_TorchDataset], eval_sets: List[Base_TorchDataset], batch_size: int = 256, num_workers: int = 4
) -> Tuple[DataLoader, ...]:
    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True
    )
    loaders = []
    for train_set in train_sets:
        loader = DataLoader(train_set, shuffle=True, **common_kwargs)
        loaders.append(loader)
    for eval_set in eval_sets:
        loader = DataLoader(eval_set, shuffle=False, **common_kwargs)
        loaders.append(loader)
    return loaders