import argparse
from typing import Optional, get_args
import logging
import time
from pathlib import Path

from tqdm import tqdm
import numpy as np

from experiments.initialization import *
from leakage_localization.datasets import DATASET, load_dataset

def time_dataset_iteration_speed(
        *,
        dataset_id: DATASET,
        root: Path,
        truncate_length: Optional[int]
):
    dataset = load_dataset(dataset_id, partition='profile', root=root, binary_trace_file=True)
    if truncate_length is None:
        truncate_length = len(dataset)
    else:
        assert truncate_length <= len(dataset)
    
    print('Testing sequential read speed...')
    t0 = time.perf_counter_ns()
    for idx in tqdm(range(truncate_length)):
        _ = dataset[idx]
    t1 = time.perf_counter_ns()
    elapsed_time_sec = (t1 - t0) / 1e9
    print(f'\tTime per iteration: {elapsed_time_sec / truncate_length} sec.')
    print(f'\tTime to traverse dataset: {elapsed_time_sec * len(dataset) / truncate_length} sec.')

    print('Testing random read speed...')
    t0 = time.perf_counter_ns()
    for idx in tqdm(np.random.choice(len(dataset), truncate_length, replace=False)):
        _ = dataset[idx]
    t1 = time.perf_counter_ns()
    elapsed_time_sec = (t1 - t0) / 1e9
    print(f'\tTime per iteration: {elapsed_time_sec / truncate_length} sec.')
    print(f'\tTime to traverse dataset: {elapsed_time_sec * len(dataset) / truncate_length} sec.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=get_args(DATASET))
    parser.add_argument('--truncate-length', default=None, type=int)
    args = parser.parse_args()

    dataset_id: DATASET = args.dataset
    if dataset_id == 'ascadv1-fixed':
        root = ASCADV1_FIXED_ROOT
    elif dataset_id == 'ascadv1-variable':
        root = ASCADV1_VARIABLE_ROOT
    else:
        assert False
    truncate_length: Optional[int] = args.truncate_length
    assert dataset_id in get_args(DATASET)
    if truncate_length is not None:
        assert isinstance(truncate_length, int) and truncate_length > 0

    time_dataset_iteration_speed(dataset_id=dataset_id, truncate_length=truncate_length, root=root)

if __name__ == '__main__':
    main()