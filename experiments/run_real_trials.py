import os
import argparse

import yaml

from leakage_localization.common import *
from leakage_localization.trials.real_trials import Trial
from init_things import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', required=True, action='store', choices=[
            'ascadv1-fixed', 'ascadv1-variable', 'dpav4', 'aes-hd', 'otiait', 'otp'
        ],
        help='Which dataset to run experiments on.'
    )
    parser.add_argument(
        '--trial-name', action='store', default=None,
        help=f'Anything saved during this trial will be saved in `{os.path.join(OUTPUT_DIR), "<TRIAL_NAME>"}`. Defaults to dataset identifier.'
    )
    parser.add_argument(
        '--trial-config', action='store', default=None,
        help=f'Which config file to use. Defaults to the file for this dataset in `{CONFIG_DIR}`.'
    )
    parser.add_argument(
        '--seed-count', type=int, default=1, action='store',
        help='Number of random seeds to repeat the trial for.'
    )
    parser.add_argument(
        '--run-particular-seeds', default=[], nargs='*', type=int, help='Run only particular seeds instead of every seed -- e.g. for splitting workload across machines.'
    )
    parser.add_argument(
        '--run-parametric-trials', default=False, action='store_true', help='Run the parametric localization algorithms on this dataset.'
    )
    parser.add_argument(
        '--run-supervised-trials', default=False, action='store_true', help='Train a neural net to attack the dataset, then interpret with feature attribution methods.'
    )
    parser.add_argument(
        '--run-advll-trials', default=False, action='store_true', help='Run ALL on this dataset.'
    )
    parser.add_argument(
        '--full-experiments', default=False, action='store_true',
        help='We have a lot of time-consuming ablations and baselines. By default I\'ll disable these because I don\'t think most people will use them, but you can pass this argument to enable them.'
    )
    args = parser.parse_args()

    config_name = args.trial_config or args.dataset
    config_path = os.path.join(CONFIG_DIR, f'{config_name}.yaml')
    assert os.path.exists(config_path), f'Invalid config path specified: {config_path}'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if 'dataset' in config:
        assert config['dataset'] == args.dataset
    trial_name = args.trial_name or args.dataset
    trial_path = os.path.join(OUTPUT_DIR, trial_name)
    trial = Trial(
        dataset_name=args.dataset,
        trial_config=config,
        seed_count=args.seed_count,
        logging_dir=trial_path,
        run_particular_seeds=args.run_particular_seeds
    )
    trial(
        run_parametric_trials=args.run_parametric_trials,
        run_supervised_trials=args.run_supervised_trials,
        run_all_trials=args.run_advll_trials,
        full_experiments=args.full_experiments
    )

if __name__ == '__main__':
    main()