from typing import Optional, Literal, List, get_args
from pathlib import Path
import argparse
import logging

import numpy as np
from numpy.typing import NDArray
from leakage_localization.datasets import DATASET, PARTITION
from leakage_localization.evaluation import OracleAgreement, compute_ta_mtd, compute_dnn_occlusion_mtd
from leakage_localization.evaluation.dnn_occlusion import OCCLUSION_ORDER

from init_things import *
from utils.load_data import load_numpy_dataset, load_torch_dataset, construct_loaders

METRIC = Literal[
    'oracle-agreement',
    'fwd-dnno-occl',
    'rev-dnno-occl',
    'ta-mtd'
]

def run_compute_oracle_agreement(leakiness_estimates: NDArray[np.floating], dataset_id: DATASET) -> NDArray[np.floating]:
    snr_dir = get_output_dir(dataset_id) / 'snr'
    assert snr_dir.exists()
    get_oracle_agreement = OracleAgreement(snr_dir, dataset_id)
    oracle_agreement = get_oracle_agreement(leakiness_estimates)
    return oracle_agreement

def _run_compute_dnn_occl(leakiness_estimates: NDArray[np.floating], dataset_id: DATASET, order: OCCLUSION_ORDER) -> NDArray[np.floating]:
    profiling_set = load_numpy_dataset(dataset_id, 'profile')
    attack_set = load_torch_dataset(dataset_id, 'attack')
    attack_loader, = construct_loaders([], [attack_set])
    ckpt_path = r'/home/jgammell/leakage-localization-publishable/outputs/ascadv1_fixed/reg_sweep/gaussian_noise_0./seed_0/best_val_mtd.ckpt' # FIXME
    dnno_mtd = compute_dnn_occlusion_mtd(leakiness_estimates, profiling_set, attack_loader, ckpt_path, order, progress_bar=True)
    return dnno_mtd

def run_compute_fwd_dnn_occl(leakiness_estimates: NDArray[np.floating], dataset_id: DATASET) -> NDArray[np.floating]:
    return _run_compute_dnn_occl(leakiness_estimates, dataset_id, 'forward')

def run_compute_rev_dnn_occl(leakiness_estimates: NDArray[np.floating], dataset_id: DATASET) -> NDArray[np.floating]:
    return _run_compute_dnn_occl(leakiness_estimates, dataset_id, 'reverse')

def run_compute_ta_mtd(leakiness_estimates: NDArray[np.floating], dataset_id: DATASET) -> NDArray[np.floating]:
    profiling_set = load_numpy_dataset(dataset_id, 'profile')
    attack_set = load_numpy_dataset(dataset_id, 'attack')
    ta_mtd = compute_ta_mtd(leakiness_estimates, profiling_set, attack_set, progress_bar=True)
    return ta_mtd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', required=True, choices=get_args(DATASET),
        help='Dataset for which to evaluate localization performance.'
    )
    parser.add_argument(
        '--path-to-eval', required=True, type=Path,
        help='Path to the leakiness estimates to be evaluated.'
    )
    parser.add_argument(
        '--dest', type=Path, default=None,
        help='Directory in which to save outputs. If unspecified, will default to the parent directory of PATH_TO_EVAL.'
    )
    parser.add_argument(
        '--metrics', default=[], nargs='*', choices=get_args(METRIC),
        help='Metric(s) to compute for the leakiness estimates at PATH_TO_EVAL.'
    )
    parser.add_argument(
        '--overwrite', default=False, action='store_true',
        help='If this argument is passed, already-cached leakiness estimates will be overwritten. Else, we will skip computation of these.'
    )
    args = parser.parse_args()

    dataset_id: DATASET = args.dataset
    assert dataset_id in get_args(DATASET)
    path_to_eval: Path = args.path_to_eval
    assert isinstance(path_to_eval, Path) and path_to_eval.exists() and path_to_eval.name.endswith('.npy')
    dest: Optional[Path] = args.dest
    if dest is None:
        dest = path_to_eval.parent
    metric_ids: List[METRIC] = args.metrics
    assert isinstance(metric_ids, list) and all(x in get_args(METRIC) for x in metric_ids)
    metric_ids = set(metric_ids)
    overwrite: bool = args.overwrite
    assert isinstance(overwrite, bool)

    for metric_id in metric_ids:
        dest_path = dest / f'{dash_to_uscr(metric_id)}.{path_to_eval.stem}.npy'
        should_compute = True
        if dest_path.exists():
            if overwrite:
                logging.info(f'File `{dest_path}` already exists. Recomputing and overwriting it.')
            else:
                logging.info(f'File `{dest_path}` already exists. Skipping computation.')
                should_compute = False
        if should_compute:
            leakiness_estimates = np.load(path_to_eval)
            if metric_id == 'oracle-agreement':
                metric = run_compute_oracle_agreement(leakiness_estimates, dataset_id)
            elif metric_id == 'fwd-dnno-occl':
                metric = run_compute_fwd_dnn_occl(leakiness_estimates, dataset_id)
            elif metric_id == 'rev-dnno-occl':
                metric = run_compute_rev_dnn_occl(leakiness_estimates, dataset_id)
            elif metric_id == 'ta-mtd':
                metric = run_compute_ta_mtd(leakiness_estimates, dataset_id)
            else:
                assert False
            np.save(dest_path, metric)
            logging.info(f'Stored metric {metric_id} for file `{path_to_eval}` at `{dest_path}`.')
        metric = np.load(dest_path)
        logging.info(f'Metric {metric_id} for file `{path_to_eval}`: {metric} (mean={metric.mean()}, std={metric.std()})')

if __name__ == '__main__':
    main()