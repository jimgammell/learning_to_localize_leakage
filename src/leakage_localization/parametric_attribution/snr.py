from typing import Dict

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import numba

from leakage_localization.datasets.base_dataset import Base_NumpyDataset


@numba.njit(parallel=True)
def _accumulate_pass1(
    traces: np.ndarray,
    int_vals: np.ndarray,
    per_target_sums: np.ndarray,
    per_target_counts: np.ndarray
) -> None:
    """
    Accumulate per-target sums and counts.

    traces: (batch_size, col_count) - float32
    int_vals: (batch_size, byte_count) - uint8
    per_target_sums: (byte_count, 256, col_count) - float32
    per_target_counts: (byte_count, 256) - int64
    """
    batch_size = traces.shape[0]
    byte_count = int_vals.shape[1]

    # Parallelize over bytes (no race conditions since each byte writes to separate slice)
    for byte_idx in numba.prange(byte_count):
        for row_idx in range(batch_size):
            target_val = int_vals[row_idx, byte_idx]
            per_target_sums[byte_idx, target_val, :] += traces[row_idx, :]
            per_target_counts[byte_idx, target_val] += 1


@numba.njit(parallel=True)
def _accumulate_pass2(
    traces: np.ndarray,
    int_vals: np.ndarray,
    per_target_means: np.ndarray,
    noise_sum_sq: np.ndarray
) -> None:
    """
    Accumulate squared deviations from per-target means.

    traces: (batch_size, col_count) - float32
    int_vals: (batch_size, byte_count) - uint8
    per_target_means: (byte_count, 256, col_count) - float32
    noise_sum_sq: (byte_count, col_count) - float32
    """
    batch_size = traces.shape[0]
    byte_count = int_vals.shape[1]

    # Parallelize over bytes
    for byte_idx in numba.prange(byte_count):
        for row_idx in range(batch_size):
            target_val = int_vals[row_idx, byte_idx]
            mean = per_target_means[byte_idx, target_val, :]
            diff = traces[row_idx, :] - mean
            noise_sum_sq[byte_idx, :] += diff * diff


def compute_snr(
        dataset: Base_NumpyDataset,
        chunk_size: int = 4096,
        use_progress_bar: bool = False,
        dtype: np.dtype = np.float32
) -> Dict[str, NDArray[np.floating]]:
    """
    Compute Signal-to-Noise Ratio (SNR) for side-channel analysis.

    Uses Numba JIT compilation for fast iteration over samples.
    """
    row_count = dataset.trace_count
    col_count = dataset.timestep_count

    if use_progress_bar:
        progress_bar = tqdm(total=2 * row_count, desc="Computing SNR")

    # Pass 1: Accumulate per-target sums and counts
    per_target_sums = dict()
    per_target_counts = dict()

    for chunk_start in range(0, row_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, row_count)
        traces, _, intermediate_values = dataset[chunk_start:chunk_end]
        batch_size = traces.shape[0]
        traces_float = np.ascontiguousarray(traces.astype(dtype))

        for int_val_key, int_val in intermediate_values.items():
            int_val_u8 = np.ascontiguousarray(int_val.astype(np.uint8))
            byte_count = int_val_u8.shape[1]

            if int_val_key not in per_target_sums:
                per_target_sums[int_val_key] = np.zeros((byte_count, 256, col_count), dtype=dtype)
                per_target_counts[int_val_key] = np.zeros((byte_count, 256), dtype=np.int64)

            _accumulate_pass1(
                traces_float,
                int_val_u8,
                per_target_sums[int_val_key],
                per_target_counts[int_val_key]
            )

        if use_progress_bar:
            progress_bar.update(batch_size)

    # Compute per-target means
    per_target_means = dict()
    for int_val_key in per_target_sums:
        counts = per_target_counts[int_val_key]
        counts_safe = np.where(counts > 0, counts, 1)
        per_target_means[int_val_key] = per_target_sums[int_val_key] / counts_safe[..., np.newaxis]

    del per_target_sums

    # Pass 2: Accumulate squared deviations for noise variance
    noise_sum_sq = dict()

    for chunk_start in range(0, row_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, row_count)
        traces, _, intermediate_values = dataset[chunk_start:chunk_end]
        batch_size = traces.shape[0]
        traces_float = np.ascontiguousarray(traces.astype(dtype))

        for int_val_key, int_val in intermediate_values.items():
            int_val_u8 = np.ascontiguousarray(int_val.astype(np.uint8))
            byte_count = int_val_u8.shape[1]

            if int_val_key not in noise_sum_sq:
                noise_sum_sq[int_val_key] = np.zeros((byte_count, col_count), dtype=dtype)

            _accumulate_pass2(
                traces_float,
                int_val_u8,
                per_target_means[int_val_key],
                noise_sum_sq[int_val_key]
            )

        if use_progress_bar:
            progress_bar.update(batch_size)

    if use_progress_bar:
        progress_bar.close()

    # Compute noise variance and SNR
    snr_vals = dict()
    for int_val_key, means in per_target_means.items():
        signal_var = np.var(means, axis=1)
        noise_var = noise_sum_sq[int_val_key] / row_count
        snr_vals[int_val_key] = signal_var / noise_var

    assert all(np.isfinite(snr_val).all() for snr_val in snr_vals.values())
    return snr_vals