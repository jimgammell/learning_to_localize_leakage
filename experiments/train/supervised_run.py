from pathlib import Path

from .entrypoint import main
from experiments.initialization import *
from leakage_localization.datasets import DATASET, PARTITION

def train_model():
    pass

if __name__ == '__main__':
    main(train_model)