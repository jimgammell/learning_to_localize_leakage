#!/bin/bash

#SBATCH --job-name=dpav4_2-htune
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=12:00:00
#SBATCH --output=./outputs/dpav4_2/htune/slurm_%A_%a.out
#SBATCH --error=./outputs/dpav4_2/htune/slurm_%A_%a.out
#SBATCH --array=0-999%8

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest ./outputs/dpav4_2/htune \
    --config-file dpav4_2 \
    --optuna-study-path ./outputs/dpav4_2/htune/optuna_study.log