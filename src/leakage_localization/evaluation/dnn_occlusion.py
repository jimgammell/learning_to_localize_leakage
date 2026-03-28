from pathlib import Path
from typing import Literal, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakage_localization.datasets import Base_NumpyDataset
from leakage_localization.training import SupervisedModule
from leakage_localization.evaluation.mtd import accumulate_ranks, compute_mtd

OCCLUSION_ORDER = Literal[
    'forward',
    'reverse'
]

@torch.inference_mode()
def _get_logits_and_int_vars(
        module: SupervisedModule,
        attack_loader: DataLoader,
        mask: torch.Tensor,
        byte_count: int,
        class_count: int
) -> Tuple[NDArray[np.floating], Dict[str, np.integer]]:
    assert torch.cuda.is_available()
    trace_count = len(attack_loader.dataset)
    collected_logits = np.full((trace_count, byte_count, class_count), np.nan, dtype=np.float32)
    collected_int_vars = defaultdict(lambda: np.full((trace_count, byte_count), -1, dtype=int))
    start_idx = 0
    for batch in attack_loader:
        trace, _, int_vars = module.prepare_batch(batch)
        batch_size = len(trace)
        end_idx = start_idx + batch_size
        assert end_idx <= trace_count
        trace = trace*mask.unsqueeze(0)
        logits = module.model(trace)
        collected_logits[start_idx:end_idx, :, :] = logits.cpu().numpy()
        for k, v in int_vars.items():
            collected_int_vars[k][start_idx:end_idx, :] = v.cpu().numpy()
        start_idx = end_idx
    assert start_idx == trace_count
    assert np.isfinite(collected_logits).all()
    assert all((x >= 0).all() for x in collected_int_vars.values())
    return collected_logits, collected_int_vars

@torch.inference_mode()
def compute_dnn_occlusion_mtd(
        leakiness_estimates: NDArray[np.floating],
        profiling_set: Base_NumpyDataset,
        attack_loader: DataLoader,
        ckpt_path: Path,
        order: OCCLUSION_ORDER,
        bin_count: int = 100,
        byte_idx: Optional[int] = None,
        attack_count: int = 100,
        progress_bar: bool = False,
) -> NDArray[np.floating]:
    module = SupervisedModule.load_from_checkpoint(ckpt_path, trace_statistics=profiling_set.get_trace_statistics(), weights_only=False)
    module.cuda()
    module.eval()
    if byte_idx is not None:
        leakiness_estimates = leakiness_estimates[byte_idx, :]
    else:
        leakiness_estimates = leakiness_estimates.mean(axis=0)
    features_to_include = leakiness_estimates.argsort()
    bin_width = len(features_to_include) // bin_count
    drop_count = len(features_to_include) % bin_count
    current_mask = torch.zeros(1, len(leakiness_estimates), dtype=torch.float, device='cuda')
    if order == 'forward':
        features_to_include = features_to_include[::-1].copy()
        if drop_count > 0:
            features_to_include = features_to_include[:-drop_count]
    elif order == 'reverse':
        features_to_include = features_to_include[drop_count:]
    else:
        assert False
    features_to_include = features_to_include.reshape(bin_count, bin_width)
    dnno_mtd = np.full((bin_count,), np.nan, dtype=np.float32)
    bin_iter = enumerate(features_to_include)
    if progress_bar:
        bin_iter = tqdm(bin_iter, total=bin_count, desc=f'DNN-occl ({order})')
    for bin_idx, bin_features_to_include in bin_iter:
        current_mask[:, bin_features_to_include] = 1.
        logits, int_vars = _get_logits_and_int_vars(module, attack_loader, current_mask, profiling_set.byte_count, profiling_set.config.num_classes)
        if byte_idx is not None:
            logits = logits[:, byte_idx, :]
            int_vars = {k: v[:, byte_idx] for k, v in int_vars.items()}
        ranks = accumulate_ranks(logits, int_vars, profiling_set.target_preds_to_key_preds, attack_count=attack_count)
        mtd = compute_mtd(ranks, reduction='mean')
        dnno_mtd[bin_idx] = mtd
    return dnno_mtd