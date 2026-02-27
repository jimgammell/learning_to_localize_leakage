#!/bin/bash

#SBATCH --job-name=ascadv1f-htune
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=4:00:00
#SBATCH --output=./outputs/ascadv1_fixed/htune/slurm_%A_%a.out
#SBATCH --error=./outputs/ascadv1_fixed/htune/slurm_%A_%a.out
#SBATCH --array=0-999%4

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest ./outputs/ascadv1_fixed/htune \
    --config-file ascadv1_fixed \
    --optuna-study-path ./outputs/ascadv1_fixed/htune/optuna_study.log