from typing import Tuple, Optional
from math import ceil

from tqdm import tqdm
import numpy as np

from .base_dataset import Base_NumpyDataset

def compute_trace_mean_and_variance(
        dataset: Base_NumpyDataset,
        chunk_size: Optional[int] = None,
        use_progress_bar: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    row_count = dataset.trace_count
    col_count = dataset.timestep_count
    if chunk_size is None:
        chunk_size = row_count
    if use_progress_bar:
        progress_bar = tqdm(total=2*row_count)

    mean = np.zeros((col_count,), dtype=np.float32)
    start_idx = 0
    for _ in range(ceil(row_count/chunk_size)):
        end_idx = min(start_idx + chunk_size, row_count)
        added_count = end_idx - start_idx
        chunk, _, _ = dataset[start_idx:end_idx]
        mean = (start_idx/end_idx)*mean + (added_count/end_idx)*chunk.mean(axis=0)
        if use_progress_bar:
            progress_bar.update(end_idx - start_idx)
        start_idx = end_idx
    assert start_idx == row_count

    var = np.zeros((col_count,), dtype=np.float32)
    start_idx = 0
    for _ in range(ceil(row_count/chunk_size)):
        end_idx = min(start_idx + chunk_size, row_count)
        added_count = end_idx - start_idx
        chunk, _, _ = dataset[start_idx:end_idx]
        var = (start_idx/end_idx)*var + (added_count/end_idx)*((chunk - mean)**2).mean(axis=0)
        if use_progress_bar:
            progress_bar.update(end_idx - start_idx)
        start_idx = end_idx
    assert start_idx == row_count

    return mean, var