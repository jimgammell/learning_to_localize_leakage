from typing import Dict, Callable, List, Optional, Literal, get_args

from numba import jit, prange
import numpy as np
from numpy.typing import NDArray
import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from .rank import _get_rank

REDUCTION = Literal['max', 'mean']

@jit(nopython=True, parallel=True)
def _accumulate_ranks(
    log_key_probs: NDArray[np.floating],
    keys: NDArray[np.uint8],
    indices: NDArray[np.integer]
) -> NDArray[np.integer]:
    attack_count, trace_count = indices.shape
    _, byte_count, class_count = log_key_probs.shape
    full_rank_over_time = np.full((attack_count, trace_count, byte_count), np.nan, dtype=np.float64)
    for attack_idx in prange(attack_count):
        rank_over_time = np.full((trace_count, byte_count), np.nan, dtype=np.float64)
        predictions = np.full((trace_count, byte_count, class_count), np.nan, dtype=np.float64)
        for res_idx, trace_idx in enumerate(indices[attack_idx, :]):
            for byte_idx in range(byte_count):
                if res_idx == 0:
                    predictions[res_idx, byte_idx, :] = log_key_probs[trace_idx, byte_idx, :]
                else:
                    predictions[res_idx, byte_idx, :] = predictions[res_idx - 1, byte_idx, :] + log_key_probs[trace_idx, byte_idx, :]
                ranks = _get_rank(predictions[res_idx, byte_idx, np.newaxis, :], keys[trace_idx, byte_idx, np.newaxis])
                rank_over_time[res_idx, byte_idx] = ranks.astype(np.float64)[0]
        full_rank_over_time[attack_idx, :, :] = rank_over_time
    return full_rank_over_time

def accumulate_ranks(
        logits: torch.Tensor,
        int_vars: Dict[str, torch.Tensor],
        target_preds_to_key_preds: Callable[[NDArray[np.floating], Dict[str, NDArray[np.uint8]]], NDArray[np.floating]],
        attack_count: int = 1000,
        traces_per_attack: Optional[int] = None
) -> NDArray[np.floating]:
    total_trace_count, byte_count, class_count = logits.shape
    assert all(total_trace_count == x.shape[0] for x in int_vars.values())
    log_target_probs = torch.log_softmax(logits.double(), dim=-1).cpu().numpy()
    int_vars = {k: v.cpu().numpy() for k, v in int_vars.items()}
    log_key_probs = target_preds_to_key_preds(log_target_probs, int_vars)
    if traces_per_attack is None:
        traces_per_attack = logits.shape[0]
    elif traces_per_attack > logits.shape[0]:
        traces_per_attack = logits.shape[0]
    indices = np.stack([
        np.random.default_rng(seed=idx).choice(total_trace_count, size=traces_per_attack, replace=False)
        for idx in range(attack_count)
    ])
    rank_over_time = _accumulate_ranks(log_key_probs, int_vars['key'], indices)
    assert np.isfinite(rank_over_time).all()
    return rank_over_time

class MinimumTracesToDisclosure(Metric):
    def __init__(
            self,
            target_preds_to_key_preds: Callable[[NDArray[np.floating], Dict[str, NDArray[np.uint8]]], NDArray[np.floating]],
            int_var_keys: List[str],
            attack_count: int = 1000,
            traces_per_attack: Optional[int] = None,
            reduction: REDUCTION = 'min',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.target_preds_to_key_preds = target_preds_to_key_preds
        self.int_var_keys = int_var_keys
        self.attack_count = attack_count
        self.traces_per_attack = traces_per_attack
        self.reduction = reduction

        self.add_state('preds', default=[], dist_reduce_fx='cat')
        for key in int_var_keys:
            self.add_state(f'{key}', default=[], dist_reduce_fx='cat')
    
    @torch.inference_mode()
    def update(self, prediction: torch.Tensor, intermediate_variables: Dict[str, torch.Tensor]):
        self.preds.append(prediction)
        for int_var_key in self.int_var_keys:
            assert int_var_key in intermediate_variables
            state = getattr(self, f'{int_var_key}', None)
            assert state is not None
            state.append(intermediate_variables[int_var_key])

    def compute(self) -> torch.Tensor:
        preds = dim_zero_cat(self.preds)
        intermediate_variables = dict()
        for int_var_key in self.int_var_keys:
            state = getattr(self, f'{int_var_key}', None)
            assert state is not None
            intermediate_variables[int_var_key] = dim_zero_cat(state)
        rank_over_time = accumulate_ranks(
            preds, intermediate_variables, self.target_preds_to_key_preds,
            attack_count=self.attack_count, traces_per_attack=self.traces_per_attack
        )
        assert np.isfinite(rank_over_time).all()
        assert (rank_over_time >= 1).all()
        if self.reduction == 'max':
            incorrect = rank_over_time > 1
            first_correct = incorrect.shape[1] - np.argmax(incorrect[:, ::-1, :], axis=1) + 1
            first_correct[~incorrect.any(axis=1)] = 1
            per_byte_mtd = (first_correct).astype(np.float32)
            mtd = per_byte_mtd.max(axis=1).mean()
        elif self.reduction == 'mean':
            mtd = rank_over_time.mean() # not really MTD, but more granular and preferable for hyperparameter tuning when not all key bytes learn
        else:
            assert False
        return torch.tensor(mtd, dtype=torch.float32)