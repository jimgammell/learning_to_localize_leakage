from pathlib import Path
from typing import Optional, Sequence
from math import ceil

from tqdm import tqdm
import h5py
import numpy as np
import trsfile

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

def convert_trs_to_binary(
        data_paths: Sequence[Path],
        dest: Path,
        sample_offsets: Optional[Sequence[int]] = None,
        use_progress_bar: bool = False
):
    data_files = [
        trsfile.open(data_path, 'r')
        for data_path in data_paths
    ]
    if sample_offsets is None:
        sample_offsets = len(data_files)*[0]
    row_count = sum(len(data_file) for data_file in data_files)
    col_count = len(data_files[0][0].samples)
    dest_memmap = np.memmap(dest, mode='w+', dtype=np.int8, shape=(row_count, col_count), order='C')
    if use_progress_bar:
        progress_bar = tqdm(total=row_count)
    memmap_idx = 0
    for data_file, sample_offset in zip(data_files, sample_offsets):
        for trace in data_file:
            assert 0 <= memmap_idx < row_count
            trace = np.array(trace.samples)
            if sample_offset is not None:
                trace = np.roll(trace, -sample_offset)
            dest_memmap[memmap_idx, :] = trace
            memmap_idx += 1
            if use_progress_bar:
                progress_bar.update(1)
    assert memmap_idx == row_count
    dest_memmap.flush()
    for data_file in data_files:
        data_file.close()