import numpy as np
from numpy.typing import NDArray
from numba import jit, prange

@jit(nopython=True, parallel=True)
def fit_means(traces: NDArray[np.float32], targets: NDArray[np.integer], unique_targets: NDArray[np.integer]) -> NDArray[np.float32]:
    trace_count, feature_count = traces.shape
    means = np.full((len(unique_targets), feature_count), np.nan, dtype=np.float32)
    for target_idx in prange(len(unique_targets)):
        target = unique_targets[target_idx]
        traces_target = traces[targets == target, :]
        means[target_idx, :] = np.sum(traces_target, axis=0) / len(traces_target)
    return means

@jit(nopython=True)
def fit_covs(traces: NDArray[np.float32], targets: NDArray[np.float32], means: NDArray[np.float32], unique_targets: NDArray[np.integer]) -> NDArray[np.float32]:
    trace_count, feature_count = traces.shape
    covs = np.full((len(unique_targets), feature_count, feature_count), np.nan, dtype=np.float32)
    for target_idx in range(len(unique_targets)):
        target = unique_targets[target_idx]
        mean = means[target_idx]
        traces_target = traces[targets == target, :]
        trace_count, _ = traces_target.shape
        diff = traces_target - mean
        cov = diff.T @ diff / (trace_count - 1)
        cov = 0.5*(cov + cov.T)
        D, U = np.linalg.eigh(cov)
        D[D < 0] = 0
        cov = U @ np.diag(D) @ U.T
        covs[target_idx, :, :] = cov
    return covs

@jit(nopython=True)
def choldecomp_covs(covs: NDArray[np.float32]) -> NDArray[np.float32]:
    target_count, feature_count, _ = covs.shape
    decomps = np.full((target_count, feature_count, feature_count), np.nan, dtype=np.float32)
    for cov_idx in range(len(covs)):
        cov = covs[cov_idx, :, :]
        L = np.linalg.cholesky(cov + 1.e-2*np.eye(feature_count, dtype=np.float32))
        decomps[cov_idx, :, :] = L
    return decomps

@jit(nopython=True)
def compute_log_gaussian_density(x: NDArray[np.float32], mu: NDArray[np.float32], L: NDArray[np.float32]) -> np.float32:
    y = np.linalg.solve(L, x - mu)
    logdet = 2*np.sum(np.log(np.diag(L)))
    return -0.5*np.dot(y, y) - 0.5*logdet

@jit(nopython=True)
def compute_log_p_y(targets: NDArray[np.integer], unique_targets: NDArray[np.integer]) -> NDArray[np.float32]:
    p_y = np.zeros((len(unique_targets)), dtype=np.float32)
    for target_idx in range(len(targets)):
        target = targets[target_idx]
        for i in range(len(unique_targets)):
            if unique_targets[i] == target:
                p_y[i] += 1
                break
    log_p_y = np.log(p_y) - np.log(p_y.sum())
    return log_p_y

@jit(nopython=True, parallel=True)
def compute_log_p_x_mid_y(traces: NDArray[np.float32], means: NDArray[np.float32], Ls: NDArray[np.float32], unique_targets: NDArray[np.integer]) -> NDArray[np.float32]:
    trace_count, feature_count = traces.shape
    log_p_x_mid_y = np.full((trace_count, len(unique_targets)), np.nan, dtype=np.float32)
    for trace_idx in prange(trace_count):
        trace = traces[trace_idx, :]
        for target_idx in range(len(unique_targets)):
            log_p_x_mid_y[trace_idx, target_idx] = compute_log_gaussian_density(trace, means[target_idx], Ls[target_idx])
    return log_p_x_mid_y