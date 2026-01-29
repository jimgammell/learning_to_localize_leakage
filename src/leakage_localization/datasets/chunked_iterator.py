from math import ceil
from typing import Iterator, Tuple, Dict

import numpy as np
from numpy.typing import NDArray

from leakage_localization.datasets.base_dataset import Base_NumpyDataset

def chunked_iterator(
        dataset: Base_NumpyDataset,
        chunk_size: int
) -> Iterator[Tuple[NDArray[np.floating], NDArray[np.integer], Dict[str, NDArray[np.integer]]]]:
    chunk_count = ceil(len(dataset)/chunk_size)
    start_idx = 0
    for _ in range(chunk_count):
        end_idx = min(start_idx + chunk_size, len(dataset))
        traces, targets, intermediate_values = dataset[start_idx:end_idx]
        for idx in range(end_idx - start_idx):
            trace = traces[idx, :]
            target = targets[idx, ...]
            intermediate_vals = {k: v[idx, ...] for k, v in intermediate_values.items()}
            yield trace, target, intermediate_vals
        start_idx = end_idx