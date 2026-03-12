import argparse
from typing import Literal, get_args

from experiments.initialization import *

DATASET = Literal['ascadv1-fixed']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', choices=get_args(DATASET), required=True)
    args = parser.parse_args()

    dataset: DATASET = args.dataset
    if dataset == 'ascadv1-fixed':
        output_dir = OUTPUTS_ROOT / 'ascadv1_fixed'
        snr_dir = output_dir / 'snr'

if __name__ == '__main__':
    main()