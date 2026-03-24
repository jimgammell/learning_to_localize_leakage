import os
import argparse

from leakage_localization.common import *
from leakage_localization.trials.toy_gaussian_experiments import Trial
from init_things import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trial-name', default='toy_gaussian', action='store', help=f'Anything saved during this trial will be stored in `{os.path.join(OUTPUT_DIR, "<TRIAL_NAME>")}`.'
    )
    parser.add_argument(
        '--seed-count', default=1, action='store', type=int, help='Number of random seeds to repeat trial for.'
    )
    args = parser.parse_args()

    trial = Trial(
        logging_dir=os.path.join(OUTPUT_DIR, args.trial_name),
        seed_count=args.seed_count
    )
    trial()

if __name__ == '__main__':
    main()