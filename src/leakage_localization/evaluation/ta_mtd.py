from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from leakage_localization.datasets.base_dataset import Base_NumpyDataset
from leakage_localization.parametric.gaussian_template_attack._interface import GaussianTemplateAttack
from leakage_localization.evaluation.mtd import compute_mtd

def _run_template_attack(
        points_of_interest: NDArray[np.integer],
        profiling_set: Base_NumpyDataset,
        attack_set: Base_NumpyDataset,
        target_key: str,
        target_idx: int,
        max_traces: Optional[int] = None,
) -> Tuple[float, NDArray[np.floating], NDArray[np.floating]]:
    template_attack = GaussianTemplateAttack(
        points_of_interest,
        target_key,
        target_idx
    )
    template_attack.profile(profiling_set)
    rank_over_time = template_attack.attack(attack_set, max_traces=max_traces)   # (attack_count, trace_count, 1)
    # per_attack_mtd: shape (attack_count,) — MTD for each individual simulated attack.
    # accumulate_ranks seeds each attack by index, so the same attack_idx uses the
    # same trace ordering across all bytes; stacking and taking max(axis=1) gives the
    # correct full-key MTD.
    per_attack_mtd = compute_mtd(rank_over_time, reduction='none')[:, 0]  # (attack_count,)
    mtd = float(per_attack_mtd.mean())
    rank_over_time = rank_over_time.mean(axis=(0, 2))     # (trace_count,)
    return mtd, rank_over_time, per_attack_mtd

def _select_pois(
        leakiness_estimates: NDArray[np.floating],
        bin_count: int,
        pois_per_bin: int
) -> NDArray[np.integer]:
    feature_count, = leakiness_estimates.shape
    bin_width = feature_count//bin_count
    pois = np.full((bin_count, pois_per_bin), -1, dtype=int)
    start_idx = 0
    for bin_idx in range(bin_count - 1):
        end_idx = start_idx + bin_width
        bin_leakiness_estimates = leakiness_estimates[start_idx:end_idx]
        bin_pois = np.argsort(bin_leakiness_estimates)[-pois_per_bin:]
        pois[bin_idx, :] = bin_pois + start_idx
        start_idx = end_idx
    bin_leakiness_estimates = leakiness_estimates[start_idx:] # we let the last bin be longer if bin_width doesn't perfectly divide feature_count
    bin_pois = np.argsort(bin_leakiness_estimates)[-pois_per_bin:]
    pois[-1, :] = bin_pois + start_idx
    assert (pois > -1).all()
    pois = pois.reshape(-1)
    pois.sort()
    return pois

def compute_ta_mtd(
        leakiness_estimates: NDArray[np.floating],
        profiling_set: Base_NumpyDataset,
        attack_set: Base_NumpyDataset,
        bin_count: int = 25,
        pois_per_bin: int = 4,
        progress_bar: bool = False,
        max_traces: Optional[int] = None,
) -> Tuple[NDArray[np.floating], ...]:
    byte_count, feature_count = leakiness_estimates.shape
    assert len(profiling_set.config.target_variable) == 1
    target_key = profiling_set.config.target_variable[0]
    ta_mtd = np.full((byte_count,), np.nan, dtype=np.float32)
    trace_count = len(attack_set) if max_traces is None else min(len(attack_set), max_traces)
    rank_over_time = np.full((byte_count, trace_count), np.nan, dtype=np.float32)
    per_attack_mtds = []
    byte_iter = range(byte_count)
    if progress_bar:
        byte_iter = tqdm(byte_iter, desc='TA-MTD bytes')
    for byte_idx in byte_iter:
        pois = _select_pois(leakiness_estimates[byte_idx, :], bin_count, pois_per_bin)
        byte_mtd, byte_rank_over_time, byte_per_attack_mtd = _run_template_attack(pois, profiling_set, attack_set, target_key, byte_idx, max_traces=max_traces)
        ta_mtd[byte_idx] = byte_mtd
        rank_over_time[byte_idx, :] = byte_rank_over_time
        per_attack_mtds.append(byte_per_attack_mtd)
    assert np.isfinite(ta_mtd).all()
    assert np.isfinite(rank_over_time).all()
    # Full-key MTD: for each simulated attack, the key is fully disclosed when the
    # last byte hits rank 1.  Take max over bytes per attack, then mean over attacks.
    per_attack_mtds = np.stack(per_attack_mtds, axis=1)  # (attack_count, byte_count)
    full_key_mtd = float(per_attack_mtds.max(axis=1).mean())
    return ta_mtd, rank_over_time, full_key_mtd