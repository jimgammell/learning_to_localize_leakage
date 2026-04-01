from typing import Tuple, Dict, Callable, Literal, Optional
from functools import partial
import logging

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import captum

from leakage_localization.training.supervised_lightning_module import SupervisedModule

ATTRIBUTION_METHOD = Literal[
    'gradvis',
    'input-x-gradient',
    'shapley',
    'n-occlusion'
]

class Attributor:
    def __init__(
            self,
            module: SupervisedModule
    ):
        self.module = module
        self.module.eval()
        #self.module.requires_grad_(False)
    
    def _get_loss(
            self,
            trace: torch.Tensor,
            target: torch.Tensor,
            head_idx: Optional[int] = None
    ) -> torch.Tensor:
        logits = self.module.model(trace)
        loss = self.module.compute_loss(logits, target)
        if head_idx is not None:
            loss = loss[:, head_idx]
        return loss
    
    def compute_gradvis(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
            smoothing_std: float = 0.0,
            smoothing_count: int = 1
    ) -> torch.Tensor:
        _trace, target, intermediate_values = self.module.prepare_batch(batch)
        batch_size, *_, feature_count = _trace.shape
        *_, head_count = target.shape
        grads = torch.zeros((batch_size, head_count, feature_count), dtype=_trace.dtype, device=_trace.device)
        for _ in range(smoothing_count):
            for head_idx in range(head_count):
                trace = _trace.clone().detach().requires_grad_(True)
                if smoothing_std > 0:
                    trace = trace + smoothing_std*torch.randn_like(trace)
                loss = self._get_loss(trace, target, head_idx=head_idx)
                grad = torch.autograd.grad(outputs=loss.sum(), inputs=trace)[0]
                grads[:, head_idx, :] += grad.detach().view(batch_size, feature_count)
        attribution = grads.abs() / smoothing_count
        return attribution

    def compute_input_x_gradient(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        _trace, target, _ = self.module.prepare_batch(batch)
        batch_size, *_, feature_count = _trace.shape
        *_, head_count = target.shape
        attrs = torch.zeros((batch_size, head_count, feature_count), dtype=_trace.dtype, device=_trace.device)
        for head_idx in range(head_count):
            trace = _trace.clone().detach().requires_grad_(True)
            loss = self._get_loss(trace, target, head_idx=head_idx)
            grad = torch.autograd.grad(outputs=loss.sum(), inputs=trace)[0]
            attrs[:, head_idx, :] = (grad * trace).detach().view(batch_size, feature_count).abs()
        return attrs
    
    # too expensive to run on uncropped datasets
    @torch.inference_mode()
    def compute_n_occlusion(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
            window_size: int = 1,
            perturbations_per_eval: int = 256
    ):
        trace, target, intermediate_values = self.module.prepare_batch(batch)
        batch_size, *_, feature_count = trace.shape
        if feature_count > 10_000:
            logging.warning(f'This method doesn\'t scale well to long traces. It\'s recommended to use gradient-based methods instead.')
        *_, head_count = target.shape
        attrs = torch.zeros((batch_size, head_count, feature_count), dtype=trace.dtype, device=trace.device)
        for head_idx in range(head_count):
            occludor = captum.attr.Occlusion(
                partial(self._get_loss, target=target, head_idx=head_idx)
            )
            attr = occludor.attribute(
                trace,
                sliding_window_shapes=(1, window_size),
                strides=(1,),
                perturbations_per_eval=perturbations_per_eval,
            )
            attrs[:, head_idx, :] = attr.view(batch_size, feature_count).abs()
        return attrs
    
    # too expensive to run on uncropped datasets
    @torch.no_grad()
    def compute_shapley(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
            sample_count: int = 1000
    ) -> torch.Tensor:
        trace, target, intermediate_values = self.module.prepare_batch(batch)
        batch_size, *_, feature_count = trace.shape
        if feature_count > 10_000:
            logging.warning(f'This method doesn\'t scale well to long traces. It\'s recommended to use gradient-based methods instead.')
        *_, head_count = target.shape
        attrs = torch.empty((batch_size, head_count, feature_count), dtype=trace.dtype, device=trace.device)
        for head_idx in range(head_count):
            svs = captum.attr.ShapleyValueSampling(
                partial(self._get_loss, target=target, head_idx=head_idx)
            )
            attr = svs.attribute(trace, n_samples=sample_count)
            attrs[:, head_idx, :] = attr.view(batch_size, feature_count)
        attrs = attrs.abs()
        return attrs
    
    def aggregate_attributions(
            self, 
            attr_fn: Callable[[torch.Tensor], torch.Tensor],
            dataloader: DataLoader,
            show_progress_bar: bool = False
    ) -> torch.Tensor:
        if show_progress_bar:
            progress_bar = tqdm(total=len(dataloader.dataset))
        attrs = None
        old_count = 0
        for batch in dataloader:
            attr = attr_fn(batch)
            if attrs is None:
                *_, head_count, feature_count = attr.shape
                attrs = torch.zeros((head_count, feature_count), dtype=self.module.dtype, device=self.module.device)
            new_count = old_count + len(attr)
            attrs = (old_count/new_count)*attrs + (len(attr)/new_count)*attr.mean(dim=0)
            old_count = new_count
            if show_progress_bar:
                progress_bar.update(len(attr))
        return attrs
    
    def __call__(
            self,
            attr_method: ATTRIBUTION_METHOD,
            dataloader: DataLoader,
            show_progress_bar: bool = False,
            **attr_kwargs
    ) -> torch.Tensor:
        if attr_method == 'gradvis':
            attr_fn = partial(self.compute_gradvis, **attr_kwargs)
        elif attr_method == 'input-x-gradient':
            attr_fn = partial(self.compute_input_x_gradient, **attr_kwargs)
        elif attr_method == 'shapley':
            attr_fn = partial(self.compute_shapley, **attr_kwargs)
        elif attr_method == 'n-occlusion':
            attr_fn = partial(self.compute_n_occlusion, **attr_kwargs)
        else:
            assert False
        attr = self.aggregate_attributions(attr_fn, dataloader, show_progress_bar=show_progress_bar)
        return attr