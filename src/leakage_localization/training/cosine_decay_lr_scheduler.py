from typing import Optional

import numpy as np
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class CosineDecayLRSched(LambdaLR):
    def __init__(
            self,
            optimizer: Optimizer,
            total_steps: int,
            lr_warmup_steps: Optional[int] = None,
            lr_const_steps: Optional[int] = None,
            lr_decay_multiplier: Optional[float] = None
    ):
        assert isinstance(total_steps, int) and total_steps > 0
        if lr_warmup_steps is None:
            lr_warmup_steps = 0
        assert isinstance(lr_warmup_steps, int) and lr_warmup_steps >= 0
        if lr_const_steps is None:
            lr_const_steps = 0
        assert isinstance(lr_const_steps, int) and lr_const_steps >= 0
        lr_decay_steps = total_steps - lr_warmup_steps - lr_const_steps
        assert lr_decay_steps >= 0
        if lr_decay_multiplier is None:
            lr_decay_multiplier = 1.
        assert isinstance(lr_decay_multiplier, float) and 0 <= lr_decay_multiplier <= 1
        
        self.total_steps = total_steps
        self.schedule = np.concatenate([
            np.linspace(0, 1, lr_warmup_steps),
            np.ones(lr_const_steps),
            (1 - lr_decay_multiplier)*(0.5*np.cos(np.linspace(0, np.pi, lr_decay_steps)) + 0.5) + lr_decay_multiplier
        ])
        super().__init__(optimizer, self.lr_lambda)
    
    def lr_lambda(self, current_step: int) -> float:
        assert isinstance(current_step, int) and current_step >= 0
        if current_step >= self.total_steps:
            return 0.
        else:
            return self.schedule[current_step]