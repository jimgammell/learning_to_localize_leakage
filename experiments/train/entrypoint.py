import argparse
from pathlib import Path
from typing import Callable, get_args

from experiments.initialization import *
from leakage_localization.datasets import DATASET

def main(
        run_fn: Callable
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=get_args(DATASET))
    parser.add_argument('--dest', required=True, type=Path)
    append_directory_clargs(parser)
    args = parser.parse_args()

    dataset_id: DATASET = args.dataset
    dest: Path = args.dest
    dest.mkdir(exist_ok=True, parents=True)

    assert dataset_id in get_args(DATASET)
    assert dest.exists()

    run_fn(
        dataset_id=dataset_id,
        dest=dest
    )