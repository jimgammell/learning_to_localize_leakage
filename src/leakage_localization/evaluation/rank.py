from typing import Union, List

import numpy as np
from numpy.typing import NDArray
from numba import jit
import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

@jit(nopython=True)
def _get_rank(preds: NDArray[np.floating], targets: NDArray[np.integer]) -> NDArray[np.integer]:
    batch_size, class_count = preds.shape
    ranks = np.full((batch_size,), 0, dtype=np.int64)
    for batch_idx in range(batch_size):
        target = targets[batch_idx]
        correct_pred = preds[batch_idx, target]
        rank = (preds[batch_idx, :] >= correct_pred).sum()
        ranks[batch_idx] = rank
    return ranks

def get_rank(
        preds: Union[torch.Tensor, NDArray[np.floating]],
        targets: Union[torch.Tensor, NDArray[np.integer]]
) -> NDArray[np.int64]:
    if isinstance(preds, torch.Tensor):
        preds = preds.float().detach().cpu().numpy()
    else:
        assert isinstance(preds, np.ndarray)
        preds = preds.astype(np.float32)
    if isinstance(targets, torch.Tensor):
        targets = targets.long().detach().cpu().numpy()
    else:
        assert isinstance(targets, np.ndarray)
        targets = targets.astype(np.int64)
    *batch_dims, class_count = preds.shape
    assert batch_dims == list(targets.shape)
    batch_size = np.prod(batch_dims, dtype=np.int64)
    preds = preds.reshape(batch_size, class_count)
    targets = targets.reshape(batch_size)
    ranks = _get_rank(preds, targets)
    assert (ranks[:] >= 1).all()
    ranks = ranks.reshape(*batch_dims)
    return ranks

def get_rank_torch(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    *batch_dims, class_count = preds.shape
    assert batch_dims == list(targets.shape)
    preds = preds.reshape(-1, class_count)
    targets = targets.reshape(-1)
    correct_logit = preds.gather(1, targets[:, None])
    ranks = (preds >= correct_logit).sum(dim=1).to(torch.float32)
    ranks = ranks.reshape(*batch_dims)
    return ranks

class Rank(Metric):
    rank_sum: torch.Tensor
    rank_count: torch.Tensor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('rank_sum', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('rank_count', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        ranks = get_rank_torch(preds, targets)
        self.rank_sum += ranks.sum()
        self.rank_count += targets.numel()
    
    def compute(self) -> torch.Tensor:
        mean_rank = self.rank_sum / self.rank_count
        return mean_rank