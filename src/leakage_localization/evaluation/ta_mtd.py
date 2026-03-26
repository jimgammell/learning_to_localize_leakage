import numpy as np
from numpy.typing import NDArray

from leakage_localization.datasets import Base_NumpyDataset
from leakage_localization.parametric import GaussianTemplateAttack
from leakage_localization.evaluation import compute_mtd

def _run_template_attack(
        points_of_interest: NDArray[np.integer],
        profiling_set: Base_NumpyDataset,
        attack_set: Base_NumpyDataset,
        target_key: str,
        target_idx: int
) -> float:
    template_attack = GaussianTemplateAttack(
        points_of_interest,
        target_key,
        target_idx
    )
    template_attack.profile(profiling_set)
    rank_over_time = template_attack.attack(attack_set)
    mtd = compute_mtd(rank_over_time, reduction='mean')
    return mtd

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
        pois_per_bin: int = 4
) -> NDArray[np.floating]:
    byte_count, feature_count = leakiness_estimates.shape
    assert len(profiling_set.config.target_variable) == 1
    target_key = profiling_set.config.target_variable[0]
    ta_mtd = np.full((byte_count,), np.nan, dtype=np.float32)
    for byte_idx in range(byte_count):
        pois = _select_pois(leakiness_estimates[byte_idx, :], bin_count, pois_per_bin)
        byte_mtd = _run_template_attack(pois, profiling_set, attack_set, target_key, byte_idx)
        ta_mtd[byte_idx] = byte_mtd
    assert np.isfinite(ta_mtd).all()
    return ta_mtd