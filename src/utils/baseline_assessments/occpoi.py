# Implementation of the OccPOI algorithm (Yap 2025) from https://eprint.iacr.org/2023/1055.pdf

from typing import Union, Optional, Sequence, Literal
import os
import time
from tqdm import tqdm
from copy import copy
from random import shuffle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from utils.aes_multi_trace_eval import AESMultiTraceEvaluator
from utils.metrics import get_rank

class OccludedModel(nn.Module):
    def __init__(self, model: nn.Module, points_to_occlude: Sequence[int]):
        super().__init__()
        self.model = model
        self.points_to_occlude = points_to_occlude
        self.input_shape = self.model.input_shape
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        occluded_x = x.clone()
        occluded_x[..., self.points_to_occlude] = 0
        logits = self.model(occluded_x)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

class OccPOI:
    def __init__(self,
        attack_dataloader, model: Union[nn.Module, str], seed: Optional[int] = None, device: Optional[str] = None,
        dataset_name: Literal['dpav4', 'aes-hd', 'ascadv1-fixed', 'ascadv1-variable'] = None
    ):
        # Setting these values to approximately the 'traces to disclosure' shown on pg. 35 of my paper.
        #  Note that having fewer attack traces actually seems to make the method perform better.
        assert dataset_name is not None
        if dataset_name == 'dpav4':
            attack_traces = 100
        elif dataset_name == 'ascadv1-fixed':
            attack_traces = 1000
        elif dataset_name == 'ascadv1-variable':
            attack_traces = 10000
        elif dataset_name == 'aes-hd':
            attack_traces = 10000
        elif dataset_name == 'otiait':
            attack_traces = 100
        elif dataset_name == 'otp':
            attack_traces = 100
        else:
            assert False
        attack_dataset = Subset(attack_dataloader.dataset, np.arange(attack_traces))
        #attack_dataset.dataset.traces = torch.from_numpy(attack_dataset.dataset.traces).to(device)
        self.attack_dataloader = DataLoader(attack_dataset, batch_size=attack_traces)
        self.base_model = model
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        self.seed = seed
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model = self.base_model.to(self.device)
        self.dataset_name = dataset_name
        self.trace_shape = (1, self.base_model.input_shape[-1])
        self.model = OccludedModel(self.base_model, [])
        base_guessing_entropy = self.compute_guessing_entropy([])
        self.lbda = base_guessing_entropy + 1 if dataset_name not in ['otp', 'toy_gaussian'] else base_guessing_entropy + 0.1 # generalizes lambda in paper to settings where we don't get down to zero guessing entropy
    
    # Using the test set as part of the algorithm is problematic. But baselines should significantly outperform this regardless, so I'm leaving as-is.
    #   Probably some better options would be: 1) cut test set in half, use half here and half for evaluation so that we can still accumulate predictions
    #   for a fixed key. Still not 100% kosher because it leaks the fixed evaluation key value into training. 2) just use a validation partition of the
    #   training set. We can't accumulate predictions in this case, but I feel like it should be fine.
    def compute_guessing_entropy(self, points_to_occlude: Sequence[int]):
        self.model.points_to_occlude = points_to_occlude
        if self.dataset_name in ['ascadv1-fixed', 'ascadv1-variable', 'dpav4', 'aes-hd']:
            multi_trace_evaluator = AESMultiTraceEvaluator(
                dataloader=self.attack_dataloader, model=self.model, seed=self.seed, device=self.device, dataset_name=self.dataset_name 
            )
            rank_over_time = multi_trace_evaluator()
            guessing_entropy = rank_over_time[-1]
            return guessing_entropy
        elif self.dataset_name in ['otiait', 'otp', 'toy_gaussian']:
            trace, target = next(iter(self.attack_dataloader))
            trace, target = trace.to(self.device), target.to(self.device)
            logits = self.model(trace)
            mean_rank = get_rank(logits, target).mean()
            return mean_rank
        else:
            assert False
    
    def run_kgo_procedure(self, starting_queue: Optional[Sequence[int]] = None):
        # Implementation of Algorithm 1 from the OccPOI paper.
        # Note that the paper's algorithm is inconsistent with their code. Paper sets has_converged=False if we identify a new leaky point, and code
        #   sets has_converged=False if we identify a new *nonleaky* point. Since successive iterations only look at the 'leaky' points identified in
        #   the last iteration, the paper version intuitively + empirically doesn't converge. I'm going with the code version.
        queue = copy(list(starting_queue)) if starting_queue is not None else list(range(self.trace_shape[-1]))
        has_converged = False
        points_to_occlude = list(set(range(self.trace_shape[-1])) - set(queue))
        iteration = 0
        while not has_converged:
            has_converged = True
            shuffle(queue)
            important_index = []
            print(f'OccPOI iteration {iteration}...')
            for spt in (progress_bar := tqdm(queue)):
                points_to_occlude.append(spt)
                guessing_entropy = self.compute_guessing_entropy(points_to_occlude)
                progress_bar.set_description(f'Guessing entropy: {guessing_entropy}')
                if guessing_entropy >= self.lbda:
                    important_index.append(spt)
                    points_to_occlude.pop()
                else:
                    has_converged = False
            print(f'Iteration complete. Current \'important point\' list:')
            print(set(important_index))
            queue = important_index
            points_to_occlude = list(set(range(self.trace_shape[-1])) - set(queue))
            iteration += 1
        # Best-effort implementation of '1-Key Guessing Occlusion' method proposed on page 11. I don't think they have this anywhere in their code.
        occpois = queue
        ranked_occpois = []
        base_ge = self.compute_guessing_entropy([])
        for x in occpois:
            occluded_ge = self.compute_guessing_entropy(list((set(range(self.trace_shape[-1])) - set(occpois)).union(set([x]))))
            ranked_occpois.append(occluded_ge - base_ge)
        ranked_occpois = np.array(ranked_occpois, dtype=np.float32) + 1 # adding 1 to avoid division by zero below -- doesn't change order
        ranked_occpois /= ranked_occpois.sum() # for aesthetic reasons
        leakage_assessment = np.zeros(self.trace_shape[-1], dtype=np.float32)
        leakage_assessment[..., occpois] = ranked_occpois # for consistency with the other baselines
        return queue, leakage_assessment
    
    # Best-effort implementation of 'Extending KGO by applying it multiple times' technique proposed on page 18. I don't see this implemented in their code.
    def run_extended_kgo_procedure(self):
        # This algorithm takes an absurd amount of time to run. I'm just going to cut it off at 10x the runtime of my algorithm and note this in paper.
        if self.dataset_name == 'ascadv1-fixed':
            max_time_min = 64.2
        elif self.dataset_name == 'ascadv1-variable':
            max_time_min = 90
        elif self.dataset_name == 'dpav4':
            max_time_min = 31
        elif self.dataset_name == 'aes-hd':
            max_time_min = 46
        elif self.dataset_name == 'otiait':
            max_time_min = 30
        elif self.dataset_name == 'otp':
            max_time_min = 21
        else:
            assert False
        start_time = time.time()
        print(f'Running OccPOI. Lambda value: {self.lbda}.')
        all_timesteps = set(range(self.trace_shape[-1]))
        occpois, leakage_assessment = self.run_kgo_procedure()
        occpois = set(occpois)
        print(f'Pre-start: occpois={occpois}')
        while (len(all_timesteps - occpois) > 0) and (self.compute_guessing_entropy(list(occpois)) < self.lbda) and (time.time()-start_time < 60*max_time_min):
            queue = list(all_timesteps - occpois)
            new_occpois, new_leakage_assessment = self.run_kgo_procedure(queue)
            new_occpois = set(new_occpois)
            leakage_assessment += new_leakage_assessment # should have disjoint support
            if len(new_occpois) == 0:
                break
            occpois = occpois.union(new_occpois)
            print(f'New sub-trial finished. Current occpois: {occpois}')
        return leakage_assessment
    
    def __call__(self, extended=False):
        if extended:
            leakage_assessment = self.run_extended_kgo_procedure()
        else:
            _, leakage_assessment = self.run_kgo_procedure()
        return leakage_assessment