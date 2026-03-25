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
    Processes each intermediate variable independently to limit peak memory usage.
    """
    row_count = dataset.trace_count
    col_count = dataset.timestep_count

    # Probe the first chunk to discover intermediate variable keys and their byte counts
    probe_traces, _, probe_intermediate_values = dataset[0:min(chunk_size, row_count)]
    int_val_keys = list(probe_intermediate_values.keys())
    int_val_byte_counts = {key: val.shape[1] for key, val in probe_intermediate_values.items()}
    del probe_traces, probe_intermediate_values

    total_work = 2 * row_count * len(int_val_keys)
    if use_progress_bar:
        progress_bar = tqdm(total=total_work, desc="Computing SNR")

    snr_vals = dict()

    # Process each intermediate variable independently to avoid holding all accumulators in memory
    for int_val_key in int_val_keys:
        byte_count = int_val_byte_counts[int_val_key]

        # Pass 1: Accumulate per-target sums and counts
        per_target_sums = np.zeros((byte_count, 256, col_count), dtype=dtype)
        per_target_counts = np.zeros((byte_count, 256), dtype=np.int64)

        for chunk_start in range(0, row_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, row_count)
            traces, _, intermediate_values = dataset[chunk_start:chunk_end]
            batch_size = traces.shape[0]
            traces_float = np.ascontiguousarray(traces.astype(dtype))
            int_val_u8 = np.ascontiguousarray(intermediate_values[int_val_key].astype(np.uint8))
            del traces, intermediate_values

            _accumulate_pass1(traces_float, int_val_u8, per_target_sums, per_target_counts)
            del traces_float, int_val_u8

            if use_progress_bar:
                progress_bar.update(batch_size)

        # Compute per-target means
        counts_safe = np.where(per_target_counts > 0, per_target_counts, 1)
        per_target_means = per_target_sums / counts_safe[..., np.newaxis]
        del per_target_sums

        # Pass 2: Accumulate squared deviations for noise variance
        noise_sum_sq = np.zeros((byte_count, col_count), dtype=dtype)

        for chunk_start in range(0, row_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, row_count)
            traces, _, intermediate_values = dataset[chunk_start:chunk_end]
            batch_size = traces.shape[0]
            traces_float = np.ascontiguousarray(traces.astype(dtype))
            int_val_u8 = np.ascontiguousarray(intermediate_values[int_val_key].astype(np.uint8))
            del traces, intermediate_values

            _accumulate_pass2(traces_float, int_val_u8, per_target_means, noise_sum_sq)
            del traces_float, int_val_u8

            if use_progress_bar:
                progress_bar.update(batch_size)

        # Compute SNR for this intermediate variable
        # Weighted variance of per-target means, using counts as weights,
        # so that unobserved target values (zero-filled) don't bias the result.
        weights = per_target_counts[..., np.newaxis].astype(dtype)  # (byte_count, 256, 1)
        total_weights = weights.sum(axis=1, keepdims=True)  # (byte_count, 1, 1)
        weighted_mean = (weights * per_target_means).sum(axis=1, keepdims=True) / total_weights  # (byte_count, 1, col_count)
        signal_var = (weights * (per_target_means - weighted_mean) ** 2).sum(axis=1) / total_weights.squeeze(1)  # (byte_count, col_count)
        noise_var = noise_sum_sq / row_count
        snr_vals[int_val_key] = signal_var / noise_var
        del per_target_means, per_target_counts, noise_sum_sq

    if use_progress_bar:
        progress_bar.close()

    assert all(np.isfinite(snr_val).all() for snr_val in snr_vals.values())
    return snr_vals