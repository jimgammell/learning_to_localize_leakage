from typing import Literal, get_args
from pathlib import Path

from .base_dataset import Base_NumpyDataset

DATASET = Literal[
    'ascadv1-fixed',
    'ascadv1-variable'
]

def load_dataset(
        dataset_id: DATASET,
        **kwargs
) -> Base_NumpyDataset:
    assert dataset_id in get_args(DATASET)
    if dataset_id == 'ascadv1-fixed':
        from .ascadv1 import ASCADv1_NumpyDataset
        kwargs.update({
            'variable_key': False,
            'cropped_traces': False,
        })
        dataset = ASCADv1_NumpyDataset(**kwargs)
    elif dataset_id == 'ascadv1-variable':
        from .ascadv1 import ASCADv1_NumpyDataset
        kwargs.update({
            'variable_key': True,
            'cropped_traces': False
        })
        dataset = ASCADv1_NumpyDataset(**kwargs)
    else:
        assert False
    return dataset