from typing import Callable, List, Tuple, Optional
from itertools import combinations
from math import ceil

from tqdm import tqdm
import numpy as np
import torch

class SecondOrderOcclusion:
    def __init__(self, model: Callable, perturbations_per_eval: int = 1, window_size: int = 1):
        self.model = model
        self.perturbations_per_eval = perturbations_per_eval
        self.window_size = window_size
    
    def get_ablation_indices(self, dim: int) -> List[Tuple[int, ...]]:
        if self.window_size == 1:
            first_order = list((i,) for i in range(dim))
            second_order = [((i,), (j,)) for i, j in combinations(range(dim), 2)]
        elif self.window_size > 1:
            first_order = [tuple(range(i, i+self.window_size)) for i in range(dim-self.window_size+1)]
            second_order = list(combinations(first_order, 2))
        else:
            assert False
        return first_order + second_order
    
    def estimate_runtime(self, x, y, iter_count=100):
        assert self.window_size == 1 # should be same big-O complexity regardless of window size
        batch_size, _, dim = x.shape
        unperturbed_logits = self.model(x)[torch.arange(batch_size), y]
        ablation_indices = self.get_ablation_indices(dim)
        ablation_index_batches = [
            ablation_indices[idx*self.perturbations_per_eval:min((idx+1)*self.perturbations_per_eval, len(ablation_indices))]
            for idx in (range(ceil(len(ablation_indices)/self.perturbations_per_eval)))
        ]
        interaction_map = np.zeros((dim, dim), dtype=np.float32)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for idx, ablation_index_batch in enumerate(ablation_index_batches):
            if idx == iter_count:
                start_event.record()
            if idx == 2*iter_count:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_min = 1e-3*start_event.elapsed_time(end_event)/60
                total_time_estimate = elapsed_time_min*len(ablation_index_batches)/iter_count
                return total_time_estimate
            xx = []
            for ablation_index in ablation_index_batch:
                ablated_x = x.clone()
                ablated_x[..., ablation_index] = 0
                xx.append(ablated_x)
            xx = torch.cat(xx, dim=0)
            yy = torch.cat([y.clone() for _ in ablation_index_batch], dim=0)
            perturbed_logits = self.model(xx)[torch.arange(len(yy)), yy]
            for i, abl_indices in enumerate(ablation_index_batch):
                if isinstance(abl_indices, tuple):
                    (abl1, abl2) = abl_indices
                else:
                    abl1 = abl2 = abl_indices
                interaction_map[abl1, abl2] = interaction_map[abl2, abl1] = (
                    unperturbed_logits - perturbed_logits[i*batch_size:(i+1)*batch_size]
                ).abs().mean().cpu().mean()
        assert False
    
    def attribute(self, x, y):
        batch_size, _, dim = x.shape
        unperturbed_logits = self.model(x)[torch.arange(batch_size), y].cpu().numpy()
        ablation_indices = self.get_ablation_indices(dim)
        ablation_idx_batches = [
            ablation_indices[idx*self.perturbations_per_eval:min((idx+1)*self.perturbations_per_eval, len(ablation_indices))]
            for idx in (range(ceil(len(ablation_indices)/self.perturbations_per_eval)))
        ]
        sum_map, count_map = map(lambda _: np.zeros((dim, dim), dtype=np.float32), range(2))
        for ablation_idx_batch in tqdm(ablation_idx_batches):
            ablated_x = x.unsqueeze(0).repeat(len(ablation_idx_batch), 1, 1, 1)
            for idx, ablation_idx in enumerate(ablation_idx_batch):
                if isinstance(ablation_idx[0], int):
                    indices = list(ablation_idx)
                elif isinstance(ablation_idx[0], tuple):
                    window1, window2 = ablation_idx
                    indices = list(window1) + list(window2)
                else:
                    assert False
                ablated_x[idx, ..., indices] = 0 # Note: we are standardizing the inputs so each feature has mean 0 and std. dev. 1. While some prior work
                                            #  emphasizes the distinction between e.g. occlude w/ zero vs. occlude w/ mean, these are the same here.
            ablated_x = ablated_x.reshape(-1, *x.shape[1:])
            yy = y.unsqueeze(0).repeat(len(ablation_idx_batch), 1).reshape(-1) #torch.cat([y.clone() for _ in ablation_idx_batch], dim=0)
            perturbed_logits = self.model(ablated_x)[torch.arange(len(yy)), yy].cpu().numpy()
            for idx, ablation_idx in enumerate(ablation_idx_batch):
                # Vectorized version by ChatGPT
                diff = np.abs((unperturbed_logits - perturbed_logits[idx*batch_size:(idx+1)*batch_size])).mean()
                
                if isinstance(ablation_idx[0], int):
                    indices = np.array(ablation_idx, dtype=np.int64)
                    I, J = np.meshgrid(indices, indices, indexing='ij')
                    np.add.at(sum_map, (I.ravel(), J.ravel()), diff)
                    np.add.at(count_map, (I.ravel(), J.ravel()), 1)
                elif isinstance(ablation_idx[0], tuple):
                    window1, window2 = ablation_idx
                    w1 = np.array(window1, dtype=np.int64)
                    w2 = np.array(window2, dtype=np.int64)
                    I, J = np.meshgrid(w1, w2, indexing='ij')
                    np.add.at(sum_map, (I.ravel(), J.ravel()), diff)
                    np.add.at(sum_map, (J.ravel(), I.ravel()), diff)
                    np.add.at(count_map, (I.ravel(), J.ravel()), 1)
                    np.add.at(count_map, (J.ravel(), I.ravel()), 1)
                else:
                    raise ValueError("Unexpected ablation index format")
                r"""diff = (unperturbed_logits - perturbed_logits[idx*batch_size:(idx+1)*batch_size]).abs().mean()
                if isinstance(ablation_idx[0], int):
                    for i0 in ablation_idx:
                        for i1 in ablation_idx:
                            sum_map[i0, i1] += diff
                            count_map[i0, i1] += 1
                elif isinstance(ablation_idx[0], tuple):
                    window1, window2 = ablation_idx
                    for i0 in window1:
                        for i1 in window2:
                            sum_map[i0, i1] += diff
                            sum_map[i1, i0] += diff
                            count_map[i0, i1] += 1
                            count_map[i1, i0] += 1
                else:
                    assert False"""
        return (sum_map / count_map).mean(axis=1)