import argparse
from typing import get_args, Callable
from pathlib import Path

from experiments.initialization import *
from leakage_localization.datasets import DATASET, PARTITION

def main(
        run_fn: Callable[[DATASET, PARTITION, Path], None]
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=get_args(DATASET))
    parser.add_argument('--partition', required=True, choices=get_args(PARTITION))
    parser.add_argument('--dest', type=Path, default=None)
    parser.add_argument('--overwrite', default=False, action='store_true')
    append_directory_clargs(parser)
    args = parser.parse_args()

    dataset_id: DATASET = args.dataset
    partition_id: PARTITION = args.partition
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = OUTPUTS_ROOT / f'{dataset_id}'.replace('-', '_') / 'snr'
    dest.mkdir(exist_ok=True, parents=True)
    overwrite: bool = args.overwrite

    assert dataset_id in get_args(DATASET)
    assert partition_id in get_args(PARTITION)
    assert dest.exists()
    assert isinstance(overwrite, bool)

    run_fn(
        dataset_id=dataset_id,
        partition_id=partition_id,
        dest=dest,
        overwrite=overwrite
    )