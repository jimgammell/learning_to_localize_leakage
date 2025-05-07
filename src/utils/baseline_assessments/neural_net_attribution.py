from typing import *
import os
from collections import defaultdict
from copy import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from captum.attr import InputXGradient, FeatureAblation, Saliency, LRP, Occlusion

from utils.chunk_iterator import chunk_iterator
from models.zaid_wouters_nets import pretrained_models
from .second_order_occlusion import SecondOrderOcclusion

class ReshapeOutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        logits = self.model(x)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

class NeuralNetAttribution:
    def __init__(self, dataloader, model: nn.Module, seed: Optional[int] = None, device: Optional[str] = None):
        self.dataloader = DataLoader(dataloader.dataset, batch_size=len(dataloader.dataset), num_workers=dataloader.num_workers)
        self.base_model = model
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model.eval()
        self.base_model.requires_grad_(False)
        self.base_model = self.base_model.to(self.device)
        self.trace_shape = self.base_model.input_shape
        self.model = ReshapeOutput(self.base_model)
    
    def accumulate_attributions(self, attr_fn: Callable, timing=False):
        if not timing:
            attribution_map = torch.zeros(*self.trace_shape)
        count = 0
        for trace, target in self.dataloader:
            batch_size = trace.size(0)
            trace, target = trace.to(self.device), target.to(self.device)
            if not timing:
                batch_attribution_map = attr_fn(trace, target)
                attribution_map = (count/(count+batch_size))*attribution_map + (batch_size/(count+batch_size))*batch_attribution_map
            else:
                prop = attr_fn(trace, target)
            count += batch_size
        return attribution_map.numpy().reshape(-1) if not timing else prop
    
    def measure_occl2o_runtime(self):
        pass
    
    def time_things(self):
        results = defaultdict(list)
        for name, fn in zip(
            ['occl2o'], #, 'gradvis', 'saliency', 'lrp', 'occlusion', 'inputxgrad'],
            [None]# , self.compute_gradvis, self.compute_saliency, self.compute_lrp, self.compute_occlusion, self.compute_inputxgrad]
        ):
            for seed in range(5):
                if name == 'occl2o':
                    x, y = next(iter(self.dataloader))
                    x, y = x.to(self.device), y.to(self.device)
                    occludor = SecondOrderOcclusion(self.model, perturbations_per_eval=10)
                    results[name].append(occludor.estimate_runtime(x, y))
                else:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    _ = fn()
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed_time_min = 1e-3*start_event.elapsed_time(end_event)/60
                    results[name].append(elapsed_time_min)
        return {key: np.stack(val) for key, val in results.items()}
    
    def compute_gradvis(self):
        def attr_fn(trace, target):
            trace.requires_grad = True
            logits = self.model(trace)
            loss = nn.functional.cross_entropy(logits, target)
            self.model.zero_grad()
            loss.backward()
            trace_grad = trace.grad.detach().abs().mean(dim=0).cpu()
            return trace_grad
        return self.accumulate_attributions(attr_fn)
    
    def compute_saliency(self):
        saliency = Saliency(self.model)
        def attr_fn(trace, target):
            trace.requires_grad = True
            return saliency.attribute(trace, target=target.to(torch.long)).detach().abs().mean(axis=0).cpu()
        return self.accumulate_attributions(attr_fn)
    
    def compute_lrp(self):
        lrp = LRP(self.model)
        def attr_fn(trace, target):
            trace.requires_grad = True
            return lrp.attribute(trace, target=target.to(torch.long)).detach().abs().mean(axis=0).cpu()
        return self.accumulate_attributions(attr_fn)
    
    @torch.no_grad()
    def compute_occlusion(self):
        ablator = FeatureAblation(self.model)
        def attr_fn(trace, target):
            return ablator.attribute(trace, target=target.to(torch.long), perturbations_per_eval=10).abs().mean(axis=0).cpu()
        return self.accumulate_attributions(attr_fn)
    
    @torch.no_grad()
    def compute_n_occlusion(self, n):
        occludor = Occlusion(self.model)
        def attr_fn(trace, target):
            return occludor.attribute(trace, sliding_window_shapes=(1, n), strides=(1,), target=target.to(torch.long), perturbations_per_eval=8).abs().mean(axis=0).cpu()
        return self.accumulate_attributions(attr_fn)
    
    @torch.no_grad()
    def compute_second_order_occlusion(self, timing=False, window_size=1):
        occludor = SecondOrderOcclusion(self.model, perturbations_per_eval=8, window_size=window_size)
        def attr_fn(trace, target):
            return occludor.attribute(trace, target.to(torch.long))
        return self.accumulate_attributions(attr_fn, timing=timing)
    
    def compute_inputxgrad(self):
        input_x_grad = InputXGradient(self.model)
        def attr_fn(trace, target):
            trace.requires_grad = True
            return input_x_grad.attribute(trace, target=target.to(torch.long)).detach().abs().mean(axis=0).cpu()
        return self.accumulate_attributions(attr_fn)