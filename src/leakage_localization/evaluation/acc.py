import torch
from torchmetrics import Metric

class FullKeyAccuracy(Metric):
    correct_sum: torch.Tensor
    count: torch.Tensor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('correct_sum', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        batch_size, byte_count = targets.shape
        assert preds.shape == (batch_size, byte_count, 256)
        preds = preds.argmax(dim=-1)
        correct_bytes = preds == targets
        correct_keys = correct_bytes.all(dim=-1)
        self.correct_sum += correct_keys.float().sum()
        self.count += batch_size
    
    def compute(self) -> torch.Tensor:
        mean_acc = self.correct_sum / self.count
        return mean_acc