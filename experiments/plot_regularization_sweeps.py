from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray

from experiments.initialization import *

def load_sweeps(base_dir: Path, prefix: str) -> Dict[str, NDArray[np.floating]]:
    pass

def plot_sweeps():
    fig, axes = plt.subplots(3, 7, figsize=(WIDTH, 3*WIDTH/7), sharey='row')
    