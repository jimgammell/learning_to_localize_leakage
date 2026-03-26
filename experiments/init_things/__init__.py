import random
from typing import Optional, Dict, Any
import shutil
import logging
from binascii import crc32

import numpy as np
from matplotlib import pyplot as plt
import torch

from .directories import *
from .mpl_constants import *

SEED: int = 0

def set_seed(seed: Optional[int] = None):
    global SEED
    if seed is not None:
        assert isinstance(seed, int) and 0 <= seed < 0xFFFFFFFF
        SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def str_to_seed(s: str) -> int:
    return (SEED + crc32(s.encode())) & 0xFFFFFFFF

def init(clargs: Optional[Dict[str, Any]] = None):
    logging.basicConfig(level=logging.INFO)
    latex_available = shutil.which('latex') is not None
    if not latex_available:
        logging.warning('Latex installation not found. Falling back to non-Latex plotting, which might look ugly.')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'text.usetex': latex_available,
        **(
            {'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amssymb}'}
            if latex_available else {}
        )
    })
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    set_seed()
    directory_config = directories.load_directory_config()
    directories.init_directories(clargs, directory_config)

init()
from .directories import * # since global variables change after the init function is run