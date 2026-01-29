import random
from typing import Optional, Dict, Any

import numpy as np
from matplotlib import pyplot as plt
import torch

from . import directories
from . import mpl_constants

SEED: int = 0

def set_seed(seed: Optional[int] = None):
    global SEED
    if seed is not None:
        assert isinstance(seed, int) and 0 <= seed < 0xFFFFFFFF
        SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def init(clargs: Optional[Dict[str, Any]] = None):
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'Times New Roman',
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amssymb}'
    })
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    set_seed()
    directory_config = directories.load_directory_config()
    directories.init_directories(clargs, directory_config)

init()
from .directories import *
from .mpl_constants import *