from typing import Tuple, Dict, Callable, Literal, Optional
from functools import partial

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import captum

from leakage_localization.training.supervised_lightning_module import SupervisedModule

ATTRIBUTION_METHOD = Literal[
    'gradvis',
    'shapley'
]

class Attributor:
    def __init__(
            self,
            module: SupervisedModule
    ):
        self.module = module
        self.module.eval()
        self.module.requires_grad_(False)
    
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
            trace = _trace.clone().detach().requires_grad_(True)
            trace = trace + smoothing_std*torch.randn_like(trace)
            loss = self._get_loss(trace, target)
            batch_size, head_count = loss.shape
            *_, feature_count = trace.shape
            for head_idx in range(head_count):
                grad = torch.autograd.grad(
                    outputs=loss[:, head_idx].sum(),
                    inputs=trace,
                    retain_graph=(head_idx < head_count - 1)
                )[0]
                grads[:, head_idx, :] += grad.detach().view(batch_size, feature_count)
        attribution = grads.abs() / smoothing_count
        return attribution
    
    @torch.no_grad()
    def compute_shapley(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
            sample_count: int = 1000
    ) -> torch.Tensor:
        trace, target, intermediate_values = self.module.prepare_batch(batch)
        batch_size, *_, feature_count = trace.shape
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
        eg_batch = next(iter(dataloader))
        eg_trace, eg_target, eg_intermediate_values = self.module.prepare_batch(eg_batch)
        batch_size, *_, feature_count = eg_trace.shape
        *_, head_count = eg_target.shape
        attrs = torch.zeros((head_count, feature_count), dtype=self.module.dtype, device=self.module.device)
        old_count = 0
        for batch in dataloader:
            attr = attr_fn(batch)
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
        elif attr_method == 'shapley':
            attr_fn = partial(self.compute_shapley, **attr_kwargs)
        else:
            assert False
        attr = self.aggregate_attributions(attr_fn, dataloader, show_progress_bar=show_progress_bar)
        return attr