from typing import Tuple, Optional
from math import ceil

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from .base_dataset import Base_NumpyDataset

def compute_trace_statistics(
        dataset: Base_NumpyDataset,
        chunk_size: Optional[int] = None,
        use_progress_bar: bool = False,
        dtype: np.dtype = np.float32
) -> Tuple[NDArray[np.floating], ...]:
    row_count = dataset.trace_count
    col_count = dataset.timestep_count
    if chunk_size is None:
        chunk_size = row_count
    if use_progress_bar:
        progress_bar = tqdm(total=2*row_count)

    trace_min = np.full((col_count,), np.inf, dtype=dtype)
    trace_max = np.full((col_count,), -np.inf, dtype=dtype)
    trace_mean = np.zeros((col_count,), dtype=dtype)
    start_idx = 0
    for _ in range(ceil(row_count/chunk_size)):
        end_idx = min(start_idx + chunk_size, row_count)
        added_count = end_idx - start_idx
        chunk, _, _ = dataset[start_idx:end_idx]
        trace_mean = (start_idx/end_idx)*trace_mean + (added_count/end_idx)*chunk.mean(axis=0)
        trace_min = np.concatenate([trace_min[np.newaxis, :], chunk], axis=0).min(axis=0)
        trace_max = np.concatenate([trace_max[np.newaxis, :], chunk], axis=0).max(axis=0)
        if use_progress_bar:
            progress_bar.update(end_idx - start_idx)
        start_idx = end_idx
    assert start_idx == row_count

    trace_var = np.zeros((col_count,), dtype=dtype)
    start_idx = 0
    for _ in range(ceil(row_count/chunk_size)):
        end_idx = min(start_idx + chunk_size, row_count)
        added_count = end_idx - start_idx
        chunk, _, _ = dataset[start_idx:end_idx]
        trace_var = (start_idx/end_idx)*trace_var + (added_count/end_idx)*((chunk - trace_mean)**2).mean(axis=0)
        if use_progress_bar:
            progress_bar.update(end_idx - start_idx)
        start_idx = end_idx
    assert start_idx == row_count

    return trace_mean, trace_var, trace_min, trace_max