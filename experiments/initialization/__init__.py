import random
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
import torch

from . import directories

SEED: int = 0

def set_seed(seed: Optional[int] = None):
    global SEED
    if seed is not None:
        assert isinstance(seed, int) and 0 <= seed < 0xFFFFFFFF
        SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def init():
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'Times New Roman',
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amssymb}'
    })
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    set_seed()

__all__ = ['set_seed', 'init', *directories.__all__]