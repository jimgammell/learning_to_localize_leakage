"""RISE (Randomized Input Sampling for Explanation) attribution.

Zero-order method: masks random subsets of input features, runs a forward pass,
and accumulates weighted masks to estimate per-feature attribution.

For each feature i and head h:
    attribution[h, i] approx E[loss_h(trace * m) * m_i] / E[m_i]

where m is a Bernoulli mask with P(m_i=1) = (1 - mask_prob).

Unlike gradient-based methods this requires no backprop, but needs many passes
to converge. Use run_passes(n) to accumulate more samples incrementally and
inspect convergence via the attribution property at any point.
"""
from typing import Optional

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from leakage_localization.training.supervised_lightning_module import SupervisedModule


class RISEAttributor:
    """Stateful RISE attributor that accumulates estimates across multiple passes.

    Args:
        module: trained SupervisedModule (moved to its own device before calling)
        dataloader: dataloader over the dataset to attribute
        mask_prob: probability of zeroing each feature (default 0.5)
    """

    def __init__(
        self,
        module: SupervisedModule,
        dataloader: DataLoader,
        mask_prob: float = 0.5,
    ):
        self.module = module
        self.module.eval()
        self.dataloader = dataloader
        self.mask_prob = mask_prob

        self._attr_sum: Optional[torch.Tensor] = None  # (head_count, feature_count)
        self._mask_sum: Optional[torch.Tensor] = None  # (feature_count,)
        self._n_passes: int = 0

    @torch.no_grad()
    def run_passes(self, n: int = 1, show_progress: bool = False) -> None:
        """Accumulate n more complete passes over the dataloader."""
        for _ in range(n):
            it = self.dataloader
            if show_progress:
                it = tqdm(it, desc=f'RISE pass {self._n_passes + 1}', leave=False)
            for batch in it:
                trace, target, _ = self.module.prepare_batch(batch)
                batch_size, *mid_dims, feature_count = trace.shape
                *_, head_count = target.shape

                if self._attr_sum is None:
                    self._attr_sum = torch.zeros(head_count, feature_count, dtype=torch.float32)
                    self._mask_sum = torch.zeros(feature_count, dtype=torch.float32)

                # mask: (batch_size, feature_count); 1 = keep, 0 = zero out
                mask = torch.bernoulli(
                    torch.full(
                        (batch_size, feature_count),
                        1.0 - self.mask_prob,
                        device=trace.device,
                    )
                )
                mask_expanded = mask.view(batch_size, *([1] * len(mid_dims)), feature_count)
                masked_trace = trace * mask_expanded

                logits = self.module.model(masked_trace)
                loss = self.module.compute_loss(logits, target)  # (batch_size, head_count)

                # loss.T: (head_count, batch_size) @ mask: (batch_size, feature_count)
                # -> (head_count, feature_count) weighted sum of masks by loss
                self._attr_sum += loss.T.float().cpu() @ mask.float().cpu()
                self._mask_sum += mask.sum(dim=0).float().cpu()

            self._n_passes += 1

    @property
    def attribution(self) -> torch.Tensor:
        """Current attribution estimate, shape (head_count, feature_count)."""
        if self._attr_sum is None:
            raise RuntimeError('No data accumulated yet -- call run_passes() first.')
        denom = self._mask_sum.clamp(min=1.0)
        return self._attr_sum / denom.unsqueeze(0)

    @property
    def n_passes(self) -> int:
        """Number of complete dataloader passes accumulated so far."""
        return self._n_passes
