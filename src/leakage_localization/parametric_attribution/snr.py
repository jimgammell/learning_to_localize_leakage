from typing import Optional, Dict

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray

from leakage_localization.datasets.base_dataset import Base_NumpyDataset
from leakage_localization.datasets.chunked_iterator import chunked_iterator

def compute_snr(
        dataset: Base_NumpyDataset,
        chunk_size: Optional[int] = None,
        use_progress_bar: bool = False,
        dtype: np.dtype = np.float32
) -> Dict[str, NDArray[np.floating]]:
    row_count = dataset.trace_count
    col_count = dataset.timestep_count
    if use_progress_bar:
        progress_bar = tqdm(total=2*row_count)
    
    if chunk_size is not None:
        assert isinstance(chunk_size, int) and 0 < chunk_size <= len(dataset)
        dataset_iter = chunked_iterator(dataset, chunk_size)
    else:
        dataset_iter = iter(dataset)
    per_target_means = dict()
    per_target_counts = dict()
    for trace, _, intermediate_values in dataset_iter:
        for int_val_key, int_val in intermediate_values.items():
            byte_count, = int_val.shape
            if not int_val_key in per_target_means:
                per_target_means[int_val_key] = np.zeros((byte_count, 256, col_count), dtype=dtype)
                assert not int_val_key in per_target_counts
                per_target_counts[int_val_key] = np.zeros((byte_count, 256), dtype=np.int64)
            for byte_val in range(byte_count):
                current_mean = per_target_means[int_val_key][byte_val, int_val[byte_val], :]
                current_count = per_target_counts[int_val_key][byte_val, int_val[byte_val]]
                per_target_means[int_val_key][byte_val, int_val[byte_val], :] = (current_count/(current_count+1))*current_mean + (1/(current_count+1))*trace
                per_target_counts[int_val_key][byte_val, int_val[byte_val]] += 1
        if use_progress_bar:
            progress_bar.update(1)

    if chunk_size is not None:
        assert isinstance(chunk_size, int) and 0 < chunk_size <= len(dataset)
        dataset_iter = chunked_iterator(dataset, chunk_size)
    else:
        dataset_iter = iter(dataset)
    noise_variance = dict()
    for count, (trace, _, intermediate_values) in enumerate(dataset_iter):
        for int_val_key, int_val in intermediate_values.items():
            byte_count, = int_val.shape
            if not int_val_key in noise_variance:
                noise_variance[int_val_key] = np.zeros((byte_count, col_count), dtype=dtype)
            for byte_val in range(byte_count):
                mean = per_target_means[int_val_key][byte_val, int_val[byte_val], :]
                current_var = noise_variance[int_val_key][byte_val, :]
                noise_variance[int_val_key][byte_val, :] = (count/(count+1))*current_var + (1/(count+1))*(trace - mean)**2
        if use_progress_bar:
            progress_bar.update(1)
    
    snr_vals = dict()
    for int_val_key, int_val_means in per_target_means.items():
        byte_count = int_val_means.shape[0]
        if not int_val_key in snr_vals:
            snr_vals[int_val_key] = np.full((byte_count, col_count), np.nan, dtype=dtype)
        for byte_val in range(byte_count):
            signal_var = np.var(per_target_means[int_val_key][byte_val, :, :], axis=0)
            noise_var = noise_variance[int_val_key][byte_val, :]
            snr = signal_var / noise_var
            snr_vals[int_val_key][byte_val, :] = snr
    assert all(np.isfinite(snr_val).all() for snr_val in snr_vals.values())
    if use_progress_bar:
        progress_bar.close()
    return snr_vals