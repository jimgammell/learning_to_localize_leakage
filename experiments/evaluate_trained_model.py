from typing import Optional, Literal, Dict, get_args
from pathlib import Path
import argparse
import logging

import numpy as np
from numpy.typing import NDArray
from lightning import Trainer
from leakage_localization.datasets import DATASET, PARTITION
from leakage_localization.evaluation import OracleAgreement, compute_ta_mtd
from leakage_localization.evaluation.dnn_occlusion import compute_dnn_occlusion_mtd, OCCLUSION_ORDER
from leakage_localization.evaluation.mtd import accumulate_ranks, compute_mtd
from leakage_localization.evaluation.mtd import accumulate_ranks, compute_mtd

from init_things import *
from experiments.utils.load_things import load_numpy_dataset, load_torch_dataset, construct_loaders, load_trained_model
from utils.training_config import SupervisedTrainingConfig

def run_compute_oracle_agreement(leakiness_estimates: NDArray[np.floating], dataset_id: DATASET) -> NDArray[np.floating]:
    snr_dir = get_output_dir(dataset_id) / 'snr'
    assert snr_dir.exists()
    get_oracle_agreement = OracleAgreement(snr_dir, dataset_id)
    oracle_agreement = get_oracle_agreement(leakiness_estimates)
    return oracle_agreement

def _run_compute_dnn_occl(
        leakiness_estimates: NDArray[np.floating],
        dataset_id: DATASET,
        strong_attacker_path: Path,
        order: OCCLUSION_ORDER,
) -> NDArray[np.floating]:
    profiling_set = load_numpy_dataset(dataset_id, 'profile')
    attack_set = load_torch_dataset(dataset_id, 'attack')
    attack_loader, = construct_loaders([], [attack_set])
    dnno_mtd = compute_dnn_occlusion_mtd(leakiness_estimates, profiling_set, attack_loader, strong_attacker_path, order, progress_bar=True)
    return dnno_mtd

def run_compute_fwd_dnn_occl(leakiness_estimates: NDArray[np.floating], dataset_id: DATASET, strong_attacker_path: Path) -> NDArray[np.floating]:
    return _run_compute_dnn_occl(leakiness_estimates, dataset_id, strong_attacker_path, 'forward')

def run_compute_rev_dnn_occl(leakiness_estimates: NDArray[np.floating], dataset_id: DATASET, strong_attacker_path: Path) -> NDArray[np.floating]:
    return _run_compute_dnn_occl(leakiness_estimates, dataset_id, strong_attacker_path, 'reverse')

def run_compute_ta_mtd(leakiness_estimates: NDArray[np.floating], dataset_id: DATASET) -> NDArray[np.floating]:
    profiling_set = load_numpy_dataset(dataset_id, 'profile')
    attack_set = load_numpy_dataset(dataset_id, 'attack')
    ta_mtd = compute_ta_mtd(leakiness_estimates, profiling_set, attack_set, progress_bar=True)
    return ta_mtd

def run_attack_performance_evalutaion(ckpt_path: Path, dataset_id: DATASET) -> Dict[str, NDArray[np.floating]]:
    profiling_set = load_numpy_dataset(dataset_id, 'profile')
    attack_set = load_torch_dataset(dataset_id, 'attack')
    attack_loader, = construct_loaders([], [attack_set])
    module = load_trained_model(ckpt_path, profiling_set)
    module.to('cuda')
    trainer = Trainer(
        accelerator='gpu',
        precision='bf16-mixed',
        default_root_dir=None,
        logger=False
    )
    results = trainer.test(module, dataloaders=attack_loader)
    metrics = {k: np.array(v) for k, v in results[0].items()}

    # Use trainer.predict() to get byte logits and intermediate variables for
    # computing per-byte rank trajectories, saved as [num_bytes, num_traces].
    outputs = trainer.predict(module, dataloaders=attack_loader)
    all_byte_logits = torch.cat([o[0] for o in outputs])
    all_int_vars = {k: torch.cat([o[1][k] for o in outputs]) for k in outputs[0][1]}
    mtd_kwargs = module.config.mtd_kwargs
    rank_over_time = accumulate_ranks(
        all_byte_logits, all_int_vars,
        mtd_kwargs['target_preds_to_key_preds'],
        attack_count=mtd_kwargs['attack_count'],
        traces_per_attack=mtd_kwargs.get('traces_per_attack')
    )
    # rank_over_time: [num_attacks, num_traces, num_bytes]
    metrics['rank_over_time'] = rank_over_time.mean(axis=0).T.astype(np.float32)  # [num_bytes, num_traces]
    metrics['per_byte_mtd'] = compute_mtd(rank_over_time, reduction='none').mean(axis=0)  # [num_bytes]
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path-to-eval', required=False, default=None, type=Path,
        help='Path to the leakiness estimates to be evaluated. Not required when only computing attack-performance.'
    )
    parser.add_argument(
        '--model-ckpt-path', default=None, type=Path,
        help='Path to the model whose attack performance will be evaluated. Required if we are evaluating attack performance.'
    )
    parser.add_argument(
        '--strong-attacker-ckpt-path', default=None, type=Path,
        help='Path to the model to use for the DNN occlusion tests.'
    )
    parser.add_argument(
        '--dataset', default=None, choices=get_args(DATASET),
        help='Dataset for which to evaluate localization performance. By default, we will try to load a config.yaml file in the parent directory of PATH_TO_EVAL and use the dataset specified there.'
    )
    parser.add_argument(
        '--dest', type=Path, default=None,
        help='Directory in which to save outputs. If unspecified, will default to the parent directory of PATH_TO_EVAL.'
    )
    parser.add_argument(
        '--metrics', default=[], nargs='*', choices=['attack-performance', *get_args(LOC_METRIC)],
        help='Metric(s) to compute for the leakiness estimates at PATH_TO_EVAL.'
    )
    parser.add_argument(
        '--overwrite', default=False, action='store_true',
        help='If this argument is passed, already-cached leakiness estimates will be overwritten. Else, we will skip computation of these.'
    )
    args = parser.parse_args()

    path_to_eval: Optional[Path] = args.path_to_eval
    if path_to_eval is not None:
        assert path_to_eval.exists() and path_to_eval.name.endswith('.npy')
    metric_ids = set(args.metrics)
    loc_metrics = metric_ids - {'attack-performance'}
    assert loc_metrics <= set(get_args(LOC_METRIC))
    assert path_to_eval is not None or loc_metrics == set(), \
        '--path-to-eval is required when computing localization metrics'
    dataset_id: Optional[DATASET] = args.dataset
    if dataset_id is None:
        config_dir = path_to_eval.parent if path_to_eval is not None else (
            args.model_ckpt_path.parent if args.model_ckpt_path is not None else None
        )
        assert config_dir is not None, 'Cannot auto-detect dataset: provide --dataset or a --path-to-eval / --model-ckpt-path whose parent contains config.yaml'
        config_path = config_dir / 'config.yaml'
        assert config_path.exists(), f'Failed to find config file in {config_path}. Please either ensure a config file exists, or pass dataset id explicitly.'
        with open(config_path, 'r') as f:
            config_kw = safe_load_yaml(f)
        config = SupervisedTrainingConfig(**config_kw)
        dataset_id = config.data.id
    dest: Optional[Path] = args.dest
    if dest is None:
        if path_to_eval is not None:
            dest = path_to_eval.parent
        elif args.model_ckpt_path is not None:
            dest = args.model_ckpt_path.parent
        else:
            assert False, 'Cannot determine output directory: provide --dest or --path-to-eval / --model-ckpt-path'
    overwrite: bool = args.overwrite
    assert isinstance(overwrite, bool)
    if 'attack-performance' in metric_ids:
        model_ckpt_path: Optional[Path] = args.model_ckpt_path
        assert model_ckpt_path is not None
    else:
        model_ckpt_path = None
    if 'fwd-dnno-occl' in metric_ids or 'rev-dnno-occl' in metric_ids:
        strong_attacker_ckpt_path: Optional[Path] = args.strong_attacker_ckpt_path
        assert strong_attacker_ckpt_path is not None
    else:
        strong_attacker_ckpt_path = None

    for metric_id in metric_ids:
        if metric_id == 'attack-performance':
            dest_path = dest / 'attack_metrics.npz'
        else:
            dest_path = dest / (f'{dash_to_uscr(metric_id)}.{path_to_eval.stem}' + ('.npz' if metric_id == 'ta-mtd' else '.npy'))
        should_compute = True
        if dest_path.exists():
            if overwrite:
                logging.info(f'File `{dest_path}` already exists. Recomputing and overwriting it.')
            else:
                logging.info(f'File `{dest_path}` already exists. Skipping computation.')
                should_compute = False
        if should_compute:
            if metric_id == 'attack-performance':
                assert model_ckpt_path is not None
                attack_metrics = run_attack_performance_evalutaion(model_ckpt_path, dataset_id)
                np.savez(dest_path, **attack_metrics)
            else:
                leakiness_estimates = np.load(path_to_eval)
                if metric_id == 'white-box-agreement':
                    metric = run_compute_oracle_agreement(leakiness_estimates, dataset_id)
                    np.save(dest_path, metric)
                elif metric_id == 'fwd-dnno-occl':
                    assert strong_attacker_ckpt_path is not None
                    metric = run_compute_fwd_dnn_occl(leakiness_estimates, dataset_id, strong_attacker_ckpt_path)
                    np.save(dest_path, metric)
                elif metric_id == 'rev-dnno-occl':
                    assert strong_attacker_ckpt_path is not None
                    metric = run_compute_rev_dnn_occl(leakiness_estimates, dataset_id, strong_attacker_ckpt_path)
                    np.save(dest_path, metric)
                elif metric_id == 'ta-mtd':
                    mtd, rank_over_time = run_compute_ta_mtd(leakiness_estimates, dataset_id)
                    metric = {'mtd': mtd, 'rank_over_time': rank_over_time}
                    np.savez(dest_path, **metric)
                else:
                    assert False
            logging.info(f'Stored metric {metric_id} for file `{path_to_eval}` at `{dest_path}`.')
        metric = np.load(dest_path)
        logging.info(f'Metric {metric_id} for file `{path_to_eval}`: {metric if isinstance(metric, np.ndarray) else [(k, v.shape) for k, v in metric.items()]})')

if __name__ == '__main__':
    main()