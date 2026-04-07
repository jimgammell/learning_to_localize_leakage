from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
import pandas

def parse_metrics(path: Path) -> Tuple[Dict[str, NDArray[np.number]], ...]:
    metrics = pandas.read_csv(path)
    assert 'train/loss' in metrics.columns
    assert 'val/loss' in metrics.columns
    train_mask = ~metrics['train/loss'].isna()
    val_mask = ~metrics['val/loss'].isna()
    train_rv = {
        'step': metrics['step'][train_mask]
    }
    val_rv = {
        'step': metrics['step'][val_mask]
    }
    for key in ['acc', 'loss', 'rank']:
        train_rv[key] = metrics[f'train/{key}'][train_mask]
        val_rv[key] = metrics[f'val/{key}'][val_mask]
        for byte_idx in range(16):
            train_rv[f'{key}/{byte_idx}'] = metrics[f'train/{key}/{byte_idx}'][train_mask]
            val_rv[f'{key}/{byte_idx}'] = metrics[f'val/{key}/{byte_idx}'][val_mask]
    train_rv = {k: np.array(v) for k, v in train_rv.items()}
    val_rv = {k: np.array(v) for k, v in val_rv.items()}
    assert all(np.isfinite(v).all() for v in train_rv.values())
    assert all(np.isfinite(v).all() for v in val_rv.values())
    return train_rv, val_rv