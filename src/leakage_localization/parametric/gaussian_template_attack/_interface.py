from typing import Tuple, Optional
from collections import defaultdict
from math import ceil

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from leakage_localization.datasets import Base_NumpyDataset
from leakage_localization.evaluation.mtd import accumulate_ranks
from ._compiled_methods import *

class GaussianTemplateAttack:
    def __init__(
            self,
            points_of_interest: NDArray[np.integer],
            target_key: str,
            target_idx: int
    ):
        self.points_of_interest = points_of_interest
        self.target_key = target_key
        self.target_idx = target_idx
        self.trace_mean = None
        self.trace_std = None
        self.unique_targets = None
        self.log_p_y = None
        self.means = None
        self.Ls = None
    
    def extract_dataset(self, dataset: Base_NumpyDataset, chunk_size: int = 4096) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
        datapoint_count = len(dataset)
        feature_count = len(self.points_of_interest)
        traces = np.full((datapoint_count, feature_count), np.nan, dtype=np.float32)
        targets = np.full((datapoint_count,), -1, dtype=np.int64)
        collected_metadata = defaultdict(list)
        start_idx = 0
        for _ in range(ceil(len(dataset)/chunk_size)):
            end_idx = min(start_idx + chunk_size, len(dataset))
            trace, _, metadata = dataset[start_idx:end_idx]
            if len(trace.shape) > 2:
                assert trace.shape[1] == 1
                trace = trace.squeeze(axis=1)
            trace = trace[:, self.points_of_interest]
            target = metadata[self.target_key][:, self.target_idx]
            traces[start_idx:end_idx, :] = trace
            targets[start_idx:end_idx] = target
            for k, v in metadata.items():
                assert v.ndim == 2
                if v.shape[1] > 1:
                    v = v[:, self.target_idx]
                else:
                    v = v[:, 0]
                collected_metadata[k].append(v)
            start_idx = end_idx
        assert start_idx == len(dataset)
        collected_metadata = {k: np.concatenate(v, axis=0) for k, v in collected_metadata.items()}
        return traces, targets, collected_metadata
    
    def has_profiled(self) -> bool:
        state_vars = (self.trace_mean, self.trace_std, self.unique_targets, self.log_p_y, self.means, self.Ls)
        if all(x is not None for x in state_vars):
            return True
        elif all(x is None for x in state_vars):
            return False
        else:
            assert False
    
    def profile(self, profiling_dataset: Base_NumpyDataset):
        assert not self.has_profiled()
        traces, targets, _ = self.extract_dataset(profiling_dataset)
        self.trace_mean = traces.mean(axis=0)
        self.trace_std = traces.std(axis=0) + 1.e-6
        traces = (traces - self.trace_mean)/self.trace_std
        self.unique_targets = np.unique(targets)
        self.unique_targets.sort()
        assert np.isfinite(self.unique_targets).all()
        self.log_p_y = compute_log_p_y(targets, self.unique_targets)
        assert np.isfinite(self.log_p_y).all()
        self.means = fit_means(traces, targets, self.unique_targets)
        assert np.isfinite(self.means).all()
        covs = fit_covs(traces, targets, self.means, self.unique_targets)
        assert np.isfinite(covs).all()
        self.Ls = choldecomp_covs(covs)
        assert np.isfinite(self.Ls).all()
    
    def get_logits(self, traces: Optional[NDArray[np.floating]] = None, dataset: Optional[Base_NumpyDataset] = None) -> NDArray[np.float32]:
        assert self.has_profiled()
        if traces is None:
            assert dataset is not None
            traces, _, _ = self.extract_dataset(dataset)
            traces = (traces - self.trace_mean)/self.trace_std
        else:
            assert dataset is None
        log_p_x_mid_y = compute_log_p_x_mid_y(traces, self.means, self.Ls, self.unique_targets)
        logits = log_p_x_mid_y + self.log_p_y
        logits = logits - logsumexp(logits, axis=-1, keepdims=True) # convert these to log-probs. Probably not necessary.
        return logits
    
    def attack(self, dataset: Base_NumpyDataset, attack_count: int = 1000, traces_per_attack: Optional[int] = None,) -> NDArray[np.floating]:
        assert self.has_profiled()
        traces, _, int_vars = self.extract_dataset(dataset)
        traces = (traces - self.trace_mean)/self.trace_std
        logits = self.get_logits(traces=traces)
        rank_over_time = accumulate_ranks(
            logits, int_vars, dataset.target_preds_to_key_preds, attack_count=attack_count, traces_per_attack=traces_per_attack
        )
        return rank_over_time