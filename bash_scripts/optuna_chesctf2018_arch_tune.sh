#!/bin/bash

#SBATCH --job-name=optuna-chesctf2018-arch-search
#SBATCH --partition=cocosys
#SBATCH --account=cocosys
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --time=12:00:00
#SBATCH --output=./outputs/ches_ctf_2018/optuna_arch_search/slurm_%a.out
#SBATCH --error=./outputs/ches_ctf_2018/optuna_arch_search/slurm_%a.out
#SBATCH --array=0-999%12

source ~/.bashrc
micromamba activate leakage-localization
python -m experiments.train.supervised \
    --dest ./outputs/ches_ctf_2018/optuna_arch_search \
    --config-file ches_ctf_2018 \
    --optuna-study-path ./outputs/ches_ctf_2018/optuna_arch_search/optuna_study.log