#!/bin/bash

#SBATCH --job-name=optuna-ascadv1f-arch-search
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=4:00:00
#SBATCH --output=./outputs/ascadv1_fixed/optuna_arch_search/slurm_%a.out
#SBATCH --error=./outputs/ascadv1_fixed/optuna_arch_search/slurm_%a.out
#SBATCH --array=0-999%8

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest ./outputs/ascadv1_fixed/optuna_arch_search \
    --config-file ascadv1_fixed_transformer \
    --optuna-study-path ./outputs/ascadv1_fixed/optuna_arch_search/optuna_study.log