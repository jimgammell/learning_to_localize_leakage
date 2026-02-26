from typing import Literal, get_args
from pathlib import Path

from .common import PARTITION
from .base_dataset import Base_NumpyDataset

DATASET = Literal[
    'ascadv1-fixed',
    'ascadv1-variable',
    'ascadv2',
    'ches-ctf-2018',
    'dpav4d2'
]