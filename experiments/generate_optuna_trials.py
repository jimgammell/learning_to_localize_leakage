"""Pre-generate QMC hyperparameter configurations for a parallel Slurm hyperparameter search.

Run this once *before* launching the Slurm array job.  It enqueues ``--n-trials``
WAITING trials in the Optuna study using a Sobol/Halton quasi-random sequence, so
that parallel workers can call ``study.optimize(n_trials=1)`` without racing on
``_find_sample_id``.

Usage:
    python experiments/generate_optuna_trials.py \\
        --config-file ascadv1_fixed \\
        --optuna-study-path ./outputs/ascadv1_fixed/htune_highdropout/study.log \\
        --n-trials 128 \\
        --seed 0

Re-running is safe: already-enqueued (WAITING) and in-progress (RUNNING) trials are
counted and skipped so no duplicates are created.
"""

from pathlib import Path
from typing import Optional
import argparse
import logging

import yaml

from init_things import *
from utils.training_config import SupervisedTrainingConfig
from leakage_localization.training.hyperparameter_tuning import (
    SamplerType,
    get_study,
    generate_qmc_trials,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config-file', required=True, type=str,
                        help='Config file name without .yaml extension')
    parser.add_argument('--config-root', type=Path, default=CONFIG_ROOT)
    parser.add_argument('--optuna-study-path', required=True, type=Path,
                        help='Path to the Optuna journal file (study.log)')
    parser.add_argument('--n-trials', required=True, type=int,
                        help='Total number of trials to enqueue')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for the QMC engine (default: global SEED)')
    parser.add_argument('--qmc-type', default='sobol', choices=['sobol', 'halton'])
    append_directory_clargs(parser)
    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    config_root: Path = args.config_root
    config_path = config_root / f'{args.config_file}.yaml'
    assert config_path.exists(), f'Config not found: {config_path}'

    with open(config_path, 'r') as f:
        config_kw = safe_load_yaml(f)
    config_kw.pop('commit_hash', None)
    config = SupervisedTrainingConfig(**config_kw)

    study_path: Path = args.optuna_study_path
    study_path.parent.mkdir(exist_ok=True, parents=True)

    study = get_study(
        study_path,
        study_direction={'min': 'minimize', 'max': 'maximize'}[config.training.early_stop_mode],
        sampler_type='qmc',
        seed=args.seed,
    )

    n_before = sum(1 for t in study.trials if t.state.name in ('WAITING', 'RUNNING'))
    generate_qmc_trials(
        study=study,
        search_space=config.search_space,
        n_trials=args.n_trials,
        seed=args.seed,
        qmc_type=args.qmc_type,
    )
    n_after = sum(1 for t in study.trials if t.state.name in ('WAITING', 'RUNNING'))
    logging.info(f'Enqueued {n_after - n_before} new trials ({n_after} total WAITING/RUNNING).')


if __name__ == '__main__':
    main()