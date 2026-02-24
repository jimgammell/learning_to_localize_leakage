#!/bin/bash

#SBATCH --job-name=ascadv2-htune
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=4:00:00
#SBATCH --output=./outputs/ascadv2/htune/slurm_%A_%a.out
#SBATCH --error=./outputs/ascadv2/htune/slurm_%A_%a.out
#SBATCH --array=0-999%4

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest ./outputs/ascadv2/htune \
    --config-file ascadv2 \
    --optuna-study-path ./outputs/ascadv2/htune/optuna_study.log