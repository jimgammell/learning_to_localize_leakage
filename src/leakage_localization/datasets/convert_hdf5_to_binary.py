from pathlib import Path
from typing import Optional
from math import ceil

from tqdm import tqdm
import h5py
import numpy as np

def convert_hdf5_to_binary(
        dataset: h5py.Dataset,
        dest: Path,
        indices: Optional[np.ndarray] = None,
        chunk_size: int = 4096,
        use_progress_bar: bool = False
    ):
    if indices is None:
        indices = np.arange(len(dataset))
    row_count = len(indices)
    _, col_count = dataset.shape
    dest_memmap = np.memmap(dest, mode='w+', dtype=np.int8, shape=(row_count, col_count), order='C')
    start_idx = 0
    if use_progress_bar:
        progress_bar = tqdm(total=row_count)
    for _ in range(ceil(row_count/chunk_size)):
        end_idx = min(start_idx + chunk_size, row_count)
        data_indices = indices[start_idx:end_idx]
        chunk = dataset[data_indices, :]
        dest_memmap[start_idx:end_idx] = chunk
        if use_progress_bar:
            progress_bar.update(end_idx - start_idx)
        start_idx = end_idx
    assert end_idx == start_idx
    dest_memmap.flush()