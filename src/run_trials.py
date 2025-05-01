import os
from copy import copy
import argparse

import yaml

from common import *
from datasets import download, AVAILABLE_DATASETS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trial-name', default=None, action='store', help=f'Anything saved during this trial will be stored in `{os.path.join(OUTPUT_DIR, "<TRIAL_NAME>")}`.'
    )
    parser.add_argument(
        '--seed-count', default=1, action='store', type=int, help='Trials will be repeated for this number of seeds.'
    )
    parser.add_argument(
        '--config-file', default=None, action='store', help=f'Configuration file specifying the settings of this trial. Expected to exist at `{os.path.join(CONFIG_DIR, "<CONFIG_FILE>.yaml")}`.',
        choices=[x.split('.')[0] for x in os.listdir(CONFIG_DIR) if x not in ['global_variables.yaml']]
    )
    subparsers = parser.add_subparsers(dest='action', required=True)
    download_parser = subparsers.add_parser('download')
    download_parser.add_argument('--dataset', default=AVAILABLE_DATASETS, choices=AVAILABLE_DATASETS, nargs='*', help='Download the specified dataset(s).')
    toy_gaussian_parser = subparsers.add_parser('run-toy-gaussian-trials')
    synthetic_parser = subparsers.add_parser('run-synthetic-trials')
    real_parser = subparsers.add_parser('run-real-trials')
    real_sub_assessments = ['compute_random', 'compute_1o_parametric_stats']
    real_sub_hints = [
        'Compute and save the random leakage assessment baseline',
        'Compute and save the first-order parametric statistics-based leakage assessments'
    ]
    for real_sub_assessment, real_sub_hint in zip(real_sub_assessments, real_sub_hints):
        real_parser.add_argument(f'--{real_sub_assessment.replace("_", "-")}', action='store_true', default=False, help=real_sub_hint)
    clargs = parser.parse_args()
    if clargs.action == 'download':
        print('Downloading things.')
        for dataset_name in clargs.dataset:
            print(f'\tDownloading dataset: {dataset_name}')
            download(dataset_name)
    else:
        assert clargs.config_file is not None, 'Must specify a configuration file via the --config-file argument.'
        config_filename = f'{clargs.config_file}.yaml'
        config_filepath = os.path.join(CONFIG_DIR, config_filename)
        with open(config_filepath, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        trial_name = clargs.trial_name or clargs.config_file
        trial_dir = os.path.join(OUTPUT_DIR, trial_name)
        seed_count = clargs.seed_count
        print(f'Starting trial of type `{clargs.action}`.')
        print(f'\tConfig path: `{config_filepath}`')
        print(f'\tOutput directory: `{trial_dir}`')
        print(f'\tSeed count: {seed_count}')
        if clargs.action == 'run-real-trials':
            from trials.real_trials import Trial
            trial = Trial(
                dataset_name=config['dataset'],
                trial_config=config,
                seed_count=seed_count,
                logging_dir=trial_dir
            )
            sub_assessment_flags = {key: getattr(clargs, key) for key in real_sub_assessments}
            if all(val == False for val in sub_assessment_flags.values()):
                sub_assessment_flags = {key: True for key in sub_assessment_flags.keys()}
            trial(**sub_assessment_flags)
        elif clargs.action == 'run-synthetic-trials':
            from trials.synthetic_data_experiments import Trial
            trial = Trial(
                logging_dir=trial_dir,
                seed_count=seed_count
            )
            trial()
        elif clargs.action == 'run-toy-gaussian-trials':
            from trials.toy_gaussian_experiments import Trial
            trial = Trial(
                logging_dir=trial_dir,
                seed_count=seed_count
            )
            trial()

if __name__ == '__main__':
    main()