"""Generate random and oracle (profiling-SNR) baseline attribution heatmaps.

Both baselines are saved as .npy files alongside a copy of the dataset config,
so they can be passed directly to evaluate_trained_model.py with no extra flags:

  python experiments/evaluate_trained_model.py \
      --path-to-eval ./outputs/ascadv1_fixed/baselines/random.npy \
      --metrics white-box-agreement ta-mtd

  python experiments/evaluate_trained_model.py \
      --path-to-eval ./outputs/ascadv1_fixed/baselines/oracle.npy \
      --metrics white-box-agreement ta-mtd
"""
import argparse
import shutil
import logging

import numpy as np

from init_things import *
from leakage_localization.evaluation import OracleAgreement
from utils.training_config import SupervisedTrainingConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file', required=True, type=str,
        help='Config file name (without .yaml), same as used for training.'
    )
    parser.add_argument('--config-root', type=Path, default=CONFIG_ROOT)
    parser.add_argument(
        '--dest', type=Path, default=None,
        help='Output directory. Defaults to outputs/<dataset_id>/baselines/.'
    )
    parser.add_argument(
        '--overwrite', action='store_true', default=False,
        help='Overwrite existing files.'
    )
    append_directory_clargs(parser)
    args = parser.parse_args()

    config_src: Path = args.config_root / f'{args.config_file}.yaml'
    assert config_src.exists(), f'Config not found: {config_src}'
    with open(config_src, 'r') as f:
        config_kw = safe_load_yaml(f)
    config = SupervisedTrainingConfig(**config_kw)
    dataset_id = config.data.id

    dest: Path = args.dest or (get_output_dir(dataset_id) / 'baselines')
    dest.mkdir(exist_ok=True, parents=True)
    snr_dir = get_output_dir(dataset_id) / 'snr'

    # Random baseline
    random_path = dest / 'random.npy'
    if not random_path.exists() or args.overwrite:
        rng = np.random.default_rng(seed=0)
        random_heatmap = rng.random((oracle.byte_count, oracle.feature_count)).astype(np.float32)
        np.save(random_path, random_heatmap)
        logging.info(f'Saved random baseline: {random_path}')
    else:
        logging.info(f'Skipping random baseline (already exists): {random_path}')

    # Oracle baseline: profiling-set SNR (independent data from the attack set)
    if snr_dir.exists():
        oracle = OracleAgreement(snr_dir, dataset_id)
        oracle_path = dest / 'oracle.npy'
        if not oracle_path.exists() or args.overwrite:
            oracle_heatmap = oracle.get_oracle_leakiness('profile').astype(np.float32)
            np.save(oracle_path, oracle_heatmap)
            logging.info(f'Saved oracle baseline: {oracle_path}')
    else:
        logging.info(f'Skipping oracle baseline (already exists): {oracle_path}')

    # Copy config.yaml so evaluate_trained_model.py can auto-detect dataset + target_byte
    config_dest = dest / 'config.yaml'
    if not config_dest.exists() or args.overwrite:
        shutil.copy(config_src, config_dest)
        logging.info(f'Copied config to {config_dest}')

if __name__ == '__main__':
    main()