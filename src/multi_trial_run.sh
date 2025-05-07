#!/bin/bash

python src/run_trials.py --seed-count=5 --config-file=ascadv1_variable run-real-trials
python src/run_trials.py --seed-count=5 --config-file=ascadv1_fixed run-real-trials
python src/run_trials.py --seed-count=5 --config-file=dpav4 run-real-trials
python src/run_trials.py --seed-count=5 --config-file=otiait run-real-trials
python src/run_trials.py --seed-count=5 --config-file=otp run-real-trials
python src/run_trials.py --seed-count=5 --config-file=aes_hd run-real-trials